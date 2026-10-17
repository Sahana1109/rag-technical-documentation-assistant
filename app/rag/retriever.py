"""
ChromaDB Vector Retrieval Module
Retrieves top-k relevant documentation chunks based on dense semantic embeddings.
"""
import logging
from typing import Any, Dict, List, Optional

from app.config import get_settings
from app.rag.ingestion import get_chroma_collection

logger = logging.getLogger(__name__)


def retrieve_documents(query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Queries ChromaDB vector collection for top-k semantically relevant chunks.

    Args:
        query: The search query string.
        top_k: Number of chunks to retrieve (defaults to config setting).

    Returns:
        List of dictionaries with chunk content, metadata, and similarity score.
    """
    settings = get_settings()
    k = top_k if top_k is not None else settings.retrieval_top_k

    try:
        collection = get_chroma_collection()
        total_count = collection.count()

        if total_count == 0:
            logger.warning("ChromaDB collection is empty. No documents to retrieve.")
            return []

        # Cap k to available document count
        actual_k = min(k, total_count)

        results = collection.query(
            query_texts=[query],
            n_results=actual_k,
            include=["documents", "metadatas", "distances"]
        )

        retrieved: List[Dict[str, Any]] = []

        if not results or not results.get("documents") or not results["documents"][0]:
            return []

        docs = results["documents"][0]
        metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
        distances = results["distances"][0] if results.get("distances") else [0.0] * len(docs)

        for doc, meta, dist in zip(docs, metas, distances):
            # Calculate a normalized similarity score (0.0 to 1.0)
            score = max(0.0, 1.0 - (dist / 2.0)) if dist is not None else 1.0
            retrieved.append({
                "content": doc,
                "source_id": meta.get("source_id", "unknown"),
                "source_filename": meta.get("source_filename", "unknown"),
                "source_url": meta.get("source_url") or None,
                "document_title": meta.get("document_title", "FastAPI Documentation"),
                "section": meta.get("section", "General"),
                "chunk_index": meta.get("chunk_index", 0),
                "score": round(score, 4)
            })

        logger.info(f"Retrieved {len(retrieved)} chunks for query: '{query[:50]}...'")
        return retrieved

    except Exception as e:
        logger.error(f"Error during document retrieval: {e}", exc_info=True)
        return []
