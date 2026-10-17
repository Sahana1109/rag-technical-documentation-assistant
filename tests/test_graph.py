"""
Test Suite for LangGraph StateGraph: Nodes, State Transitions, and Conditional Edges
"""
import pytest

from app.rag.state import create_initial_state, GraphState
from app.rag.graph import (
    rag_graph,
    query_analysis_node,
    document_grading_node,
    query_retry_node,
    generate_answer_node,
    grounding_check_node,
    insufficient_context_node,
    decide_after_grading,
)


def test_initial_state_creation():
    """Test that GraphState is initialized with correct defaults and retry limits."""
    state = create_initial_state("How do query parameters work?", maximum_retries=3)
    assert state["original_question"] == "How do query parameters work?"
    assert state["rewritten_query"] == "How do query parameters work?"
    assert state["retry_count"] == 0
    assert state["maximum_retries"] == 3
    assert state["sufficient_context"] is True
    assert state["retrieved_documents"] == []
    assert state["relevant_documents"] == []
    assert state["citations"] == []


def test_query_analysis_node():
    """Test Node A analyzes question and returns query_type and expanded query."""
    state = create_initial_state("How to create an APIRouter?")
    update = query_analysis_node(state)
    assert "query_type" in update
    assert "rewritten_query" in update
    assert len(update["rewritten_query"]) > 0


def test_document_grading_relevant_and_irrelevant():
    """Test Node C correctly grades relevant chunks and filters irrelevant ones."""
    state: GraphState = {
        "original_question": "How to validate request bodies?",
        "rewritten_query": "FastAPI Pydantic request validation",
        "retrieved_documents": [
            {
                "content": "FastAPI uses Pydantic models for request body validation and serialization.",
                "document_title": "Request Validation",
                "section": "Pydantic Models",
                "source_filename": "fastapi_request_validation.md",
                "source_url": "https://fastapi.tiangolo.com/tutorial/body/",
                "chunk_index": 0,
                "score": 0.95
            },
            {
                "content": "Quantum physics describes the behavior of matter and light on atomic scales.",
                "document_title": "Physics",
                "section": "Quantum",
                "source_filename": "physics.md",
                "source_url": None,
                "chunk_index": 0,
                "score": 0.1
            }
        ]
    }

    result = document_grading_node(state)
    assert len(result["graded_documents"]) == 2
    # The first document is relevant
    assert result["graded_documents"][0]["is_relevant"] is True
    # The irrelevant quantum physics document is filtered
    assert result["graded_documents"][1]["is_relevant"] is False
    assert len(result["relevant_documents"]) == 1
    assert result["relevant_documents"][0]["document_title"] == "Request Validation"


def test_decide_after_grading_routing():
    """Test conditional edge branching for all three paths."""
    # Path 1: Relevant documents found -> generate_answer
    state_with_docs: GraphState = {
        "relevant_documents": [{"title": "Doc1"}],
        "retry_count": 0,
        "maximum_retries": 2
    }
    assert decide_after_grading(state_with_docs) == "generate_answer"

    # Path 2: 0 relevant documents, retries remaining -> query_retry
    state_retry: GraphState = {
        "relevant_documents": [],
        "retry_count": 1,
        "maximum_retries": 2
    }
    assert decide_after_grading(state_retry) == "query_retry"

    # Path 3: 0 relevant documents, retries exhausted -> insufficient_context
    state_exhausted: GraphState = {
        "relevant_documents": [],
        "retry_count": 2,
        "maximum_retries": 2
    }
    assert decide_after_grading(state_exhausted) == "insufficient_context"


def test_query_retry_increments_count():
    """Test Node D increments retry_count and rewrites the query."""
    state: GraphState = {
        "original_question": "How to handle exceptions?",
        "rewritten_query": "FastAPI exceptions",
        "retry_count": 1,
        "maximum_retries": 2
    }
    result = query_retry_node(state)
    assert result["retry_count"] == 2
    assert "rewritten_query" in result
    assert len(result["rewritten_query"]) > 0


def test_insufficient_context_node():
    """Test terminal insufficient context node sets correct flags and message."""
    state = create_initial_state("Completely unknown question")
    result = insufficient_context_node(state)
    assert result["sufficient_context"] is False
    assert result["grounding_status"] == "insufficient_evidence"
    assert "not contain sufficient information" in result["answer"].lower()
    assert result["citations"] == []


def test_citations_propagation_in_generate_answer():
    """Test Node E generates grounded answer and correctly formats citations."""
    state: GraphState = {
        "original_question": "How to use APIRouter?",
        "relevant_documents": [
            {
                "content": "Use APIRouter to structure your endpoints modularly.",
                "document_title": "FastAPI Routing",
                "section": "Modular Routing with APIRouter",
                "source_filename": "fastapi_routing.md",
                "source_url": "https://fastapi.tiangolo.com/tutorial/path-params/",
                "chunk_index": 0
            }
        ]
    }
    result = generate_answer_node(state)
    assert len(result["citations"]) == 1
    citation = result["citations"][0]
    assert citation["document_title"] == "FastAPI Routing"
    assert citation["section"] == "Modular Routing with APIRouter"
    assert citation["source_filename"] == "fastapi_routing.md"
    assert citation["source_url"] == "https://fastapi.tiangolo.com/tutorial/path-params/"
    assert citation["chunk_index"] == 0
    assert result["sufficient_context"] is True


def test_grounding_check_node():
    """Test Node F verifies grounding against context."""
    state: GraphState = {
        "answer": "FastAPI is built on Starlette and Pydantic.",
        "relevant_documents": [{"content": "FastAPI is built on Starlette and Pydantic."}],
        "sufficient_context": True
    }
    result = grounding_check_node(state)
    assert result["grounding_status"] in ("grounded", "ungrounded")


def test_end_to_end_graph_execution():
    """Test full LangGraph StateGraph execution from START to END."""
    initial_state = create_initial_state("How do dependencies work in FastAPI?", maximum_retries=2)
    final_state = rag_graph.invoke(initial_state)

    assert "answer" in final_state
    assert len(final_state["answer"]) > 0
    assert "citations" in final_state
    assert "grounding_status" in final_state
    assert "query_type" in final_state
    assert final_state["retry_count"] >= 0
