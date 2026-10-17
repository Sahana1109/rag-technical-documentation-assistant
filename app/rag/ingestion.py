"""
Document Ingestion Pipeline with Section-Aware Chunking,
ChromaDB Persistence, Deduplication, and SSRF-Safe URL Fetching.
"""
import hashlib
import ipaddress
import logging
import os
import re
import socket
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import chromadb
from chromadb.utils import embedding_functions
from bs4 import BeautifulSoup
import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)

# Cached Chroma Client and Collection
_chroma_client: Optional[chromadb.PersistentClient] = None
_chroma_collection: Optional[chromadb.Collection] = None


def get_embedding_function():
    """Returns the SentenceTransformer embedding function for ChromaDB."""
    settings = get_settings()
    try:
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=settings.embedding_model
        )
    except Exception as e:
        logger.warning(f"Could not load SentenceTransformerEmbeddingFunction: {e}. Using Chroma default.")
        return embedding_functions.DefaultEmbeddingFunction()


def get_chroma_client() -> chromadb.PersistentClient:
    """Returns a persistent ChromaDB client."""
    global _chroma_client
    if _chroma_client is None:
        settings = get_settings()
        os.makedirs(settings.chroma_persist_directory, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=settings.chroma_persist_directory)
    return _chroma_client


def get_chroma_collection() -> chromadb.Collection:
    """Returns or creates the ChromaDB collection with proper embeddings."""
    global _chroma_collection
    if _chroma_collection is None:
        settings = get_settings()
        client = get_chroma_client()
        embed_fn = get_embedding_function()
        _chroma_collection = client.get_or_create_collection(
            name=settings.chroma_collection_name,
            embedding_function=embed_fn,
            metadata={"description": "FastAPI Technical Documentation Corpus"}
        )
    return _chroma_collection


# ------------------------------------------------------------------------------
# SSRF Protection & URL Fetching
# ------------------------------------------------------------------------------

def is_safe_ip(ip_str: str) -> bool:
    """
    Checks if an IP address string is safe from SSRF.
    Rejects loopback, private, link-local, multicast, and reserved addresses.
    """
    try:
        ip = ipaddress.ip_address(ip_str)
        if (
            ip.is_loopback
            or ip.is_private
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            return False
        # Specific check for AWS/GCP metadata address
        if str(ip) == "169.254.169.254":
            return False
        return True
    except ValueError:
        return False


def validate_url_safe(url: str) -> Tuple[bool, str]:
    """
    Validates that a URL uses http/https and does not resolve to disallowed IPs.
    Returns (is_safe, error_reason).
    """
    parsed = urlparse(url)
    if parsed.scheme.lower() not in ("http", "https"):
        return False, f"Unsupported URL scheme '{parsed.scheme}'. Only http and https are allowed."

    hostname = parsed.hostname
    if not hostname:
        return False, "URL does not contain a valid hostname."

    # Prevent common local hostnames
    if hostname.lower() in ("localhost", "127.0.0.1", "::1", "0.0.0.0"):
        return False, f"Access to localhost/loopback address '{hostname}' is forbidden."

    # DNS resolution check
    try:
        addr_info = socket.getaddrinfo(hostname, None)
        for entry in addr_info:
            ip = entry[4][0]
            if not is_safe_ip(ip):
                return False, f"Hostname '{hostname}' resolves to private or restricted IP '{ip}'."
    except socket.gaierror as e:
        return False, f"DNS resolution failed for host '{hostname}': {e}"
    except Exception as e:
        return False, f"Error validating URL host: {e}"

    return True, ""


def fetch_url_content(url: str, timeout_seconds: int = 10) -> str:
    """
    Safely fetches content from an HTTP/HTTPS URL with SSRF protection,
    timeout handling, and redirect security verification.
    """
    is_safe, reason = validate_url_safe(url)
    if not is_safe:
        raise ValueError(f"SSRF Protection Error: {reason}")

    current_url = url
    max_redirects = 5

    with httpx.Client(timeout=timeout_seconds, follow_redirects=False) as client:
        for _ in range(max_redirects):
            is_safe, reason = validate_url_safe(current_url)
            if not is_safe:
                raise ValueError(f"SSRF Protection Error on redirect: {reason}")

            response = client.get(current_url, headers={"User-Agent": "FastAPI-Doc-Assistant/1.0"})
            if response.is_redirect:
                location = response.headers.get("location")
                if not location:
                    raise ValueError("Redirect response missing Location header.")
                from urllib.parse import urljoin
                current_url = urljoin(current_url, location)
                continue

            response.raise_for_status()
            return response.text

    raise ValueError("Exceeded maximum redirect limit.")


# ------------------------------------------------------------------------------
# Document Parsing and Section-Aware Chunking
# ------------------------------------------------------------------------------

def extract_metadata_and_text(content: str, filename: str, source_url: Optional[str] = None) -> Tuple[str, Optional[str], str]:
    """
    Extracts the document title, source URL, and body content from Markdown or HTML.
    """
    title = None
    url = source_url

    if filename.endswith(".html") or "<html" in content.lower():
        soup = BeautifulSoup(content, "html.parser")
        # Extract title from title or h1 tag
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
        elif soup.h1:
            title = soup.h1.get_text().strip()

        # Remove irrelevant elements
        for tag in soup(["script", "style", "nav", "footer", "aside"]):
            tag.decompose()
        body_text = soup.get_text(separator="\n\n")
        resolved_title = title or os.path.splitext(filename)[0].replace("_", " ").title()
        return resolved_title, url, body_text

    # Markdown processing
    lines = content.splitlines()
    body_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("# ") and not title:
            title = stripped[2:].strip()
        elif stripped.lower().startswith("source:") and not url:
            url = stripped[7:].strip()
        else:
            body_lines.append(line)

    resolved_title = title or os.path.splitext(filename)[0].replace("_", " ").title()
    return resolved_title, url, "\n".join(body_lines)


def chunk_markdown_content(
    content: str,
    title: str,
    source_filename: str,
    source_url: Optional[str] = None,
    chunk_size: int = 800,
    chunk_overlap: int = 120
) -> List[Dict]:
    """
    Splits text or markdown content into coherent chunks, preserving section
    headings and code blocks where possible.
    """
    # Split content by markdown section headings (e.g. ## or ###)
    section_pattern = re.compile(r"(^#{1,3}\s+.+$)", re.MULTILINE)
    parts = section_pattern.split(content)

    sections: List[Tuple[str, str]] = []
    current_heading = "Overview"

    for part in parts:
        part_str = part.strip()
        if not part_str:
            continue
        if part_str.startswith("#"):
            current_heading = part_str.lstrip("#").strip()
        else:
            sections.append((current_heading, part_str))

    chunks: List[Dict] = []
    chunk_index = 0

    for section_heading, text in sections:
        # If the section text fits in a chunk, keep it as is
        if len(text) <= chunk_size:
            chunks.append({
                "chunk_index": chunk_index,
                "document_title": title,
                "section": section_heading,
                "source_filename": source_filename,
                "source_url": source_url,
                "content": text
            })
            chunk_index += 1
        else:
            # Split large section into overlapping windows on paragraph or sentence boundaries
            paragraphs = text.split("\n\n")
            current_chunk = ""
            for p in paragraphs:
                p_clean = p.strip()
                if not p_clean:
                    continue
                if len(current_chunk) + len(p_clean) + 2 <= chunk_size:
                    current_chunk = f"{current_chunk}\n\n{p_clean}".strip()
                else:
                    if current_chunk:
                        chunks.append({
                            "chunk_index": chunk_index,
                            "document_title": title,
                            "section": section_heading,
                            "source_filename": source_filename,
                            "source_url": source_url,
                            "content": current_chunk
                        })
                        chunk_index += 1
                    # If paragraph itself is bigger than chunk_size, split by sentences/lines
                    if len(p_clean) > chunk_size:
                        start = 0
                        while start < len(p_clean):
                            end = start + chunk_size
                            chunk_text = p_clean[start:end]
                            chunks.append({
                                "chunk_index": chunk_index,
                                "document_title": title,
                                "section": section_heading,
                                "source_filename": source_filename,
                                "source_url": source_url,
                                "content": chunk_text
                            })
                            chunk_index += 1
                            start += chunk_size - chunk_overlap
                        current_chunk = ""
                    else:
                        current_chunk = p_clean

            if current_chunk:
                chunks.append({
                    "chunk_index": chunk_index,
                    "document_title": title,
                    "section": section_heading,
                    "source_filename": source_filename,
                    "source_url": source_url,
                    "content": current_chunk
                })
                chunk_index += 1

    return chunks


# ------------------------------------------------------------------------------
# Ingestion and Vector Store Persistence
# ------------------------------------------------------------------------------

def ingest_document(
    content: str,
    filename: str,
    source_url: Optional[str] = None,
    source_type: str = "file"
) -> Tuple[int, str]:
    """
    Ingests a single document: parses, chunks, generates embeddings,
    and saves to ChromaDB with deduplication.
    Returns (num_chunks, source_id).
    """
    settings = get_settings()
    collection = get_chroma_collection()

    source_id = source_url if source_url else filename
    title, resolved_url, body = extract_metadata_and_text(content, filename, source_url)

    chunks = chunk_markdown_content(
        content=body,
        title=title,
        source_filename=filename,
        source_url=resolved_url,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap
    )

    if not chunks:
        return 0, source_id

    # Deduplication: Remove any existing chunks for this source_id before re-inserting
    try:
        existing = collection.get(where={"source_id": source_id})
        if existing and existing.get("ids"):
            collection.delete(ids=existing["ids"])
            logger.info(f"Deleted {len(existing['ids'])} existing chunks for source '{source_id}' to prevent duplicates.")
    except Exception as e:
        logger.debug(f"No existing chunks to clean for '{source_id}': {e}")

    # Prepare batches for ChromaDB
    ids = []
    documents = []
    metadatas = []

    for chunk in chunks:
        chunk_id = f"{source_id}#chunk_{chunk['chunk_index']}"
        ids.append(chunk_id)
        documents.append(chunk["content"])
        metadatas.append({
            "source_id": source_id,
            "source_filename": chunk["source_filename"],
            "source_url": chunk["source_url"] or "",
            "document_title": chunk["document_title"],
            "section": chunk["section"],
            "chunk_index": chunk["chunk_index"],
            "source_type": source_type
        })

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )

    logger.info(f"Ingested '{source_id}' successfully: {len(chunks)} chunks stored.")
    return len(chunks), source_id


def ingest_directory(directory_path: str) -> Dict:
    """
    Ingests all supported files from a local directory into ChromaDB.
    """
    settings = get_settings()
    allowed_exts = tuple(settings.allowed_extensions_list)

    if not os.path.exists(directory_path):
        return {"documents_ingested": 0, "chunks_created": 0, "sources": [], "errors": [f"Directory not found: {directory_path}"]}

    total_chunks = 0
    sources = []
    errors = []

    for root, _, files in os.walk(directory_path):
        for f in sorted(files):
            if f.endswith(allowed_exts):
                file_path = os.path.join(root, f)
                try:
                    with open(file_path, "r", encoding="utf-8") as file:
                        content = file.read()
                    chunks_count, source_id = ingest_document(
                        content=content,
                        filename=f,
                        source_type="file"
                    )
                    total_chunks += chunks_count
                    sources.append(source_id)
                except Exception as e:
                    errors.append(f"Failed to ingest {f}: {str(e)}")

    return {
        "documents_ingested": len(sources),
        "chunks_created": total_chunks,
        "sources": sources,
        "errors": errors
    }


def list_indexed_documents() -> List[Dict]:
    """
    Lists unique indexed documents and metadata from ChromaDB.
    """
    collection = get_chroma_collection()
    results = collection.get()

    if not results or not results.get("metadatas"):
        return []

    docs_map: Dict[str, Dict] = {}
    for meta in results["metadatas"]:
        src_id = meta.get("source_id", "unknown")
        if src_id not in docs_map:
            docs_map[src_id] = {
                "source_id": src_id,
                "document_title": meta.get("document_title", "Untitled"),
                "source_filename": meta.get("source_filename", "unknown"),
                "source_url": meta.get("source_url") or None,
                "chunk_count": 0,
                "source_type": meta.get("source_type", "file")
            }
        docs_map[src_id]["chunk_count"] += 1

    return list(docs_map.values())
