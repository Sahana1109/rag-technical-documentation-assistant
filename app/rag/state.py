"""
LangGraph State Schema
Defines the typed dictionary that flows through the Self-Corrective RAG StateGraph.
"""
from typing import TypedDict, List, Dict, Any, Optional


class GraphState(TypedDict, total=False):
    """
    State dictionary tracked across all nodes in the LangGraph workflow.
    """
    # Query details
    original_question: str
    rewritten_query: str
    query_type: str  # conceptual, how-to, troubleshooting, api_reference

    # Retrieval and grading
    retrieved_documents: List[Dict[str, Any]]
    graded_documents: List[Dict[str, Any]]
    relevant_documents: List[Dict[str, Any]]

    # Generation & output
    answer: str
    citations: List[Dict[str, Any]]

    # Retry and loop prevention controls
    retry_count: int
    maximum_retries: int
    sufficient_context: bool

    # Grounding & safety
    grounding_status: str  # grounded, ungrounded, insufficient_evidence, not_checked
    errors: List[str]


def create_initial_state(question: str, maximum_retries: int = 2) -> GraphState:
    """Helper to initialize a clean GraphState dictionary."""
    return GraphState(
        original_question=question,
        rewritten_query=question,
        query_type="conceptual",
        retrieved_documents=[],
        graded_documents=[],
        relevant_documents=[],
        answer="",
        citations=[],
        retry_count=0,
        maximum_retries=maximum_retries,
        sufficient_context=True,
        grounding_status="not_checked",
        errors=[]
    )
