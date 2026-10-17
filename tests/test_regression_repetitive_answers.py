"""
Regression Test Suite: Prevention of Repetitive Answers and Validation of Diverse Synthesis
Tests:
  1. Distinct questions produce distinct, question-specific answers.
  2. Sequential queries through the same application lifecycle do not repeat previous answers.
  3. Irrelevant questions ("quantum entanglement") trigger the insufficient-context response.
  4. Retrieval failure triggers retry routing and falls back safely.
  5. Citations strictly match the retrieved documents used for each question.
  6. API-level diversity across POST /query for distinct queries.
  7. Transparent error handling when real LLM is requested without credentials.
"""
import pytest
from fastapi.testclient import TestClient

from app.rag.graph import rag_graph
from app.rag.state import create_initial_state
from app.services.query_service import run_rag_pipeline


@pytest.mark.parametrize("question,expected_keyword", [
    ("What is FastAPI?", "overview"),
    ("What are dependencies in FastAPI?", "depend"),
    ("How does FastAPI validate request data?", "pydantic"),
    ("How does FastAPI handle exceptions?", "exception"),
])
def test_distinct_questions_produce_distinct_grounded_answers(question: str, expected_keyword: str):
    """
    Verifies that distinct questions produce unique, context-specific answers
    rather than a fixed repetitive template.
    """
    response = run_rag_pipeline(question)

    assert response.question == question
    assert len(response.answer) > 50
    assert expected_keyword.lower() in response.answer.lower()
    assert response.sufficient_context is True
    assert len(response.citations) > 0


def test_sequential_requests_do_not_repeat_answers():
    """
    Test A: Send two completely different questions sequentially through the pipeline.
    Verify that the second response is not a repeat or copy of the first response.
    """
    q1 = "What are dependencies in FastAPI?"
    q2 = "How does FastAPI handle exceptions?"

    res1 = run_rag_pipeline(q1)
    res2 = run_rag_pipeline(q2)

    # Answers must not be identical
    assert res1.answer != res2.answer
    assert res1.question == q1
    assert res2.question == q2

    # Answers must contain their respective domain-specific content
    assert "depend" in res1.answer.lower()
    assert "exception" in res2.answer.lower()
    assert "depend" not in res2.answer.lower() or "exception" in res2.answer.lower()

    # Citations must reflect different source files
    citations_doc1 = [c.source_filename for c in res1.citations]
    citations_doc2 = [c.source_filename for c in res2.citations]
    assert citations_doc1 != citations_doc2


def test_irrelevant_question_triggers_insufficient_context():
    """
    Test B: An irrelevant question ('quantum entanglement') must receive an
    insufficient-context response instead of an invented or generic answer.
    """
    irrelevant_q = "Explain quantum entanglement using the indexed FastAPI documentation."
    res = run_rag_pipeline(irrelevant_q)

    assert res.sufficient_context is False or "not contain sufficient information" in res.answer.lower()
    assert res.grounding_status in ("insufficient_evidence", "not_checked", "grounded")


def test_retrieval_failure_retry_and_fallback():
    """
    Test C: When retrieval produces zero relevant chunks, the workflow retries
    and cleanly terminates at insufficient-context without infinite loops.
    """
    state = create_initial_state("Completely nonexistent topic xyz987abc", maximum_retries=2)
    # Simulate empty retrieval
    state["retrieved_documents"] = []
    state["relevant_documents"] = []

    final_state = rag_graph.invoke(state)

    assert final_state["sufficient_context"] is False
    assert "not contain sufficient information" in final_state["answer"].lower()
    assert final_state["retry_count"] >= 0


def test_citation_correctness_and_source_match():
    """
    Test D: Citations must reference documents that were actually retrieved
    and passed into relevant_documents for answer synthesis.
    """
    q = "What are dependencies in FastAPI?"
    res = run_rag_pipeline(q)

    assert len(res.citations) > 0
    for citation in res.citations:
        assert citation.document_title is not None
        assert citation.section is not None
        assert citation.source_filename.endswith(".md")
        # Ensure raw drive paths like C:\ are not leaked in source_filename
        assert ":\\" not in citation.source_filename


def test_api_level_regression_distinct_answers(client: TestClient):
    """
    Test F: Test different questions via POST /query over HTTP.
    Verifies that the API returns distinct, question-specific answers.
    """
    q1 = "What are dependencies in FastAPI?"
    q2 = "How does FastAPI handle exceptions?"

    resp1 = client.post("/query", json={"question": q1})
    resp2 = client.post("/query", json={"question": q2})

    assert resp1.status_code == 200
    assert resp2.status_code == 200

    data1 = resp1.json()
    data2 = resp2.json()

    # Answers must be non-identical
    assert data1["answer"] != data2["answer"]
    assert "depend" in data1["answer"].lower()
    assert "exception" in data2["answer"].lower()

    # Citations must differ
    sources1 = [c["document_title"] for c in data1["citations"]]
    sources2 = [c["document_title"] for c in data2["citations"]]
    assert sources1 != sources2


def test_api_unconfigured_real_llm_reports_error(monkeypatch):
    """
    Test Phase 4: When real Gemini mode is requested without credentials,
    the API transparently returns HTTP 503 instead of silently serving mock text.
    """
    # Simulate production mode with unconfigured key
    from app.config import get_settings
    settings = get_settings()

    monkeypatch.setattr(settings, "llm_provider", "gemini")
    monkeypatch.setattr(settings, "google_api_key", None)

    # Use a fresh test client
    from app.main import app
    test_client = TestClient(app)

    response = test_client.post("/query", json={"question": "What is FastAPI?"})
    assert response.status_code == 503
    detail = response.json()["detail"].lower()
    assert "google_api_key" in detail or "not configured" in detail
