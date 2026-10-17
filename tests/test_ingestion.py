"""
Test Suite for Ingestion Pipeline: Chunking, Deduplication, and SSRF Security
"""
import pytest

from app.rag.ingestion import (
    chunk_markdown_content,
    extract_metadata_and_text,
    is_safe_ip,
    validate_url_safe,
    ingest_document,
    get_chroma_collection,
    list_indexed_documents,
)


def test_chunk_markdown_preserves_sections():
    """Test that Markdown chunking splits along headers and retains section names."""
    markdown_text = """## Section 1: Overview
FastAPI is built on Starlette and Pydantic. It provides automatic OpenAPI docs.

## Section 2: Path Parameters
Path parameters capture variables from the URL path.
"""
    chunks = chunk_markdown_content(
        content=markdown_text,
        title="FastAPI Guide",
        source_filename="guide.md",
        chunk_size=500,
        chunk_overlap=50
    )

    assert len(chunks) == 2
    assert chunks[0]["section"] == "Section 1: Overview"
    assert "FastAPI is built" in chunks[0]["content"]
    assert chunks[1]["section"] == "Section 2: Path Parameters"
    assert "Path parameters capture" in chunks[1]["content"]
    assert chunks[0]["chunk_index"] == 0
    assert chunks[1]["chunk_index"] == 1


def test_extract_metadata_and_text_markdown():
    """Test title and source URL extraction from Markdown headers."""
    doc = """# Advanced Routing in FastAPI
Source: https://fastapi.tiangolo.com/tutorial/routing/

Content body goes here.
"""
    title, url, body = extract_metadata_and_text(doc, "routing.md")
    assert title == "Advanced Routing in FastAPI"
    assert url == "https://fastapi.tiangolo.com/tutorial/routing/"
    assert "Content body goes here." in body


def test_extract_metadata_and_text_html():
    """Test HTML parsing removes scripts and extracts title."""
    html_doc = """
    <html>
      <head><title>FastAPI Dependencies</title></head>
      <body>
        <script>alert('malicious')</script>
        <nav><a href="/">Home</a></nav>
        <h1>Dependency Injection Overview</h1>
        <p>FastAPI uses Depends() to inject reusable dependencies.</p>
      </body>
    </html>
    """
    title, url, body = extract_metadata_and_text(html_doc, "dependencies.html")
    assert title == "FastAPI Dependencies"
    assert "alert('malicious')" not in body
    assert "Home" not in body  # nav removed
    assert "Depends()" in body


def test_ssrf_ip_filtering():
    """Test comprehensive IP security verification for SSRF protection."""
    # Disallowed IPs
    assert is_safe_ip("127.0.0.1") is False       # Loopback
    assert is_safe_ip("127.0.1.5") is False       # Loopback range
    assert is_safe_ip("::1") is False             # IPv6 Loopback
    assert is_safe_ip("10.0.0.1") is False        # RFC 1918 Class A
    assert is_safe_ip("172.16.0.1") is False      # RFC 1918 Class B
    assert is_safe_ip("172.31.255.255") is False  # RFC 1918 Class B
    assert is_safe_ip("192.168.1.100") is False   # RFC 1918 Class C
    assert is_safe_ip("169.254.169.254") is False # Cloud metadata link-local
    assert is_safe_ip("0.0.0.0") is False         # Unspecified

    # Allowed Public IPs
    assert is_safe_ip("8.8.8.8") is True          # Google DNS
    assert is_safe_ip("1.1.1.1") is True          # Cloudflare DNS
    assert is_safe_ip("93.184.216.34") is True    # example.com


def test_validate_url_safe_schemes():
    """Test URL scheme validation."""
    is_safe, reason = validate_url_safe("file:///etc/passwd")
    assert is_safe is False
    assert "unsupported url scheme" in reason.lower()

    is_safe, reason = validate_url_safe("gopher://localhost:70")
    assert is_safe is False

    is_safe, _ = validate_url_safe("http://127.0.0.1:8000")
    assert is_safe is False


def test_deduplication_and_reingest():
    """Test that re-ingesting a document updates existing chunks without producing duplicates."""
    collection = get_chroma_collection()
    initial_count = collection.count()

    doc_content_v1 = """# Version One
## Feature A
Feature A documentation content.
"""
    doc_filename = "unique_test_dedup.md"

    # Ingest Version 1
    chunks_1, src_id_1 = ingest_document(doc_content_v1, doc_filename)
    assert chunks_1 >= 1
    count_after_v1 = collection.count()
    assert count_after_v1 == initial_count + chunks_1

    # Ingest Version 2 (updated content with same filename)
    doc_content_v2 = """# Version Two
## Feature A
Updated Feature A content with more details.
## Feature B
New Feature B content.
"""
    chunks_2, src_id_2 = ingest_document(doc_content_v2, doc_filename)
    count_after_v2 = collection.count()

    # The collection count should reflect the replacement of v1 with v2, NOT v1 + v2
    assert count_after_v2 == initial_count + chunks_2

    # Clean up test doc
    existing = collection.get(where={"source_id": doc_filename})
    if existing and existing["ids"]:
        collection.delete(ids=existing["ids"])
