"""
Query Service Layer
Encapsulates running the Self-Corrective LangGraph and transforming results into API response schemas.
"""
import logging
from typing import Dict, Any

from app.config import get_settings
from app.rag.graph import rag_graph
from app.rag.state import create_initial_state
from app.schemas import QueryResponse, Citation, SourceMetadata

logger = logging.getLogger(__name__)


def run_rag_pipeline(question: str) -> QueryResponse:
    """
    Executes the self-corrective LangGraph workflow for a user question.
    """
    settings = get_settings()
    logger.info(f"Executing RAG pipeline for query: '{question}'")

    # Initialize GraphState
    initial_state = create_initial_state(
        question=question,
        maximum_retries=settings.max_retries
    )

    # Run LangGraph StateGraph
    final_state: Dict[str, Any] = rag_graph.invoke(initial_state)

    # Convert citations
    raw_citations = final_state.get("citations", [])
    citations = [
        Citation(
            document_title=c.get("document_title", "Document"),
            section=c.get("section", "General"),
            source_filename=c.get("source_filename", "unknown"),
            source_url=c.get("source_url") or None,
            chunk_index=c.get("chunk_index", 0)
        )
        for c in raw_citations
    ]

    # Convert relevant sources metadata
    raw_relevant_docs = final_state.get("relevant_documents", [])
    relevant_sources = [
        SourceMetadata(
            source_filename=doc.get("source_filename", "unknown"),
            document_title=doc.get("document_title", "FastAPI Documentation"),
            section=doc.get("section", "General"),
            source_url=doc.get("source_url") or None,
            score=doc.get("score"),
            snippet=doc.get("content", "")[:300] + ("..." if len(doc.get("content", "")) > 300 else "")
        )
        for doc in raw_relevant_docs
    ]

    return QueryResponse(
        question=final_state.get("original_question", question),
        answer=final_state.get("answer", "No answer could be generated."),
        citations=citations,
        relevant_sources=relevant_sources,
        query_type=final_state.get("query_type", "conceptual"),
        retry_count=final_state.get("retry_count", 0),
        grounding_status=final_state.get("grounding_status", "not_checked"),
        sufficient_context=final_state.get("sufficient_context", True)
    )
