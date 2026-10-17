"""
CLI Script to Ingest Technical Documentation into ChromaDB
Usage:
    python scripts/ingest_docs.py
    python scripts/ingest_docs.py --dir ./data/docs
    python scripts/ingest_docs.py --url https://fastapi.tiangolo.com/tutorial/
    python scripts/ingest_docs.py --reset
"""
import argparse
import logging
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import get_settings
from app.rag.ingestion import (
    ingest_directory,
    get_chroma_collection,
    get_chroma_client,
    fetch_url_content,
    ingest_document,
    list_indexed_documents,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("ingest_docs")


def main():
    parser = argparse.ArgumentParser(description="Ingest technical documentation into ChromaDB vector store.")
    parser.add_argument(
        "--dir",
        type=str,
        default="./data/docs",
        help="Path to documentation directory (default: ./data/docs)"
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="Optional external URL to fetch and ingest"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset and recreate the ChromaDB collection before ingestion"
    )

    args = parser.parse_args()
    settings = get_settings()

    logger.info("=" * 60)
    logger.info("RAG Documentation Assistant - Document Ingestion Tool")
    logger.info("=" * 60)
    logger.info(f"Persistence Directory: {settings.chroma_persist_directory}")
    logger.info(f"Collection Name:       {settings.chroma_collection_name}")
    logger.info(f"Embedding Model:       {settings.embedding_model}")
    logger.info(f"Chunk Size / Overlap:  {settings.chunk_size} / {settings.chunk_overlap}")

    if args.reset:
        logger.warning(f"Resetting collection '{settings.chroma_collection_name}'...")
        client = get_chroma_client()
        try:
            client.delete_collection(name=settings.chroma_collection_name)
            logger.info("Existing collection deleted.")
        except Exception as e:
            logger.info(f"No existing collection to delete: {e}")

    # Process directory
    if os.path.exists(args.dir):
        logger.info(f"Ingesting documents from directory: '{args.dir}'...")
        stats = ingest_directory(args.dir)
        logger.info(f"Ingested {stats['documents_ingested']} documents with {stats['chunks_created']} total chunks.")
        if stats["errors"]:
            for err in stats["errors"]:
                logger.error(f"  - Error: {err}")
    else:
        logger.warning(f"Directory '{args.dir}' does not exist.")

    # Process URL if provided
    if args.url:
        logger.info(f"Ingesting URL: '{args.url}'...")
        try:
            content = fetch_url_content(args.url, timeout_seconds=settings.request_timeout_seconds)
            from urllib.parse import urlparse
            path_filename = os.path.basename(urlparse(args.url).path) or "webpage.html"
            chunks_count, source_id = ingest_document(
                content=content,
                filename=path_filename,
                source_url=args.url,
                source_type="url"
            )
            logger.info(f"Ingested URL '{args.url}' ({chunks_count} chunks).")
        except Exception as e:
            logger.error(f"Failed to ingest URL: {e}")

    # Summary
    indexed = list_indexed_documents()
    total_chunks = sum(d["chunk_count"] for d in indexed)
    logger.info("=" * 60)
    logger.info(f"Ingestion Complete. Total Documents: {len(indexed)} | Total Chunks: {total_chunks}")
    for doc in indexed:
        logger.info(f"  • {doc['document_title']} ({doc['source_filename']}) - {doc['chunk_count']} chunks")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
