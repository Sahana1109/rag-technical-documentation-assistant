"""
Test Suite for FastAPI Endpoints: /query, /ingest, /documents, /feedback, /health
"""
import io
import pytest
from fastapi.testclient import TestClient


def test_root_endpoint(client: TestClient):
    """Test GET / returns API metadata and status."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "/docs" in data["docs_url"]
    assert "version" in data


def test_health_endpoint(client: TestClient):
    """Test GET /health returns application and dependency diagnostics."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("healthy", "degraded")
    assert "vector_db_status" in data
    assert "total_indexed_chunks" in data
    assert data["llm_provider"] in ("mock", "gemini", "openai")


def test_query_validation_empty(client: TestClient):
    """Test POST /query rejects empty question with HTTP 422 or 400."""
    response = client.post("/query", json={"question": ""})
    assert response.status_code in (400, 422)


def test_query_validation_whitespace(client: TestClient):
    """Test POST /query rejects whitespace-only string."""
    response = client.post("/query", json={"question": "   \n\t  "})
    assert response.status_code in (400, 422)


def test_query_validation_too_short(client: TestClient):
    """Test POST /query rejects question with fewer than 3 characters."""
    response = client.post("/query", json={"question": "hi"})
    assert response.status_code == 422


def test_query_validation_too_long(client: TestClient):
    """Test POST /query rejects question longer than 1000 characters."""
    response = client.post("/query", json={"question": "a" * 1001})
    assert response.status_code == 422


def test_query_success_with_mock(client: TestClient):
    """Test POST /query successfully returns answer, citations, and metadata."""
    response = client.post("/query", json={"question": "How do dependencies work in FastAPI?"})
    assert response.status_code == 200
    data = response.json()
    assert "question" in data
    assert "answer" in data
    assert len(data["answer"]) > 0
    assert "citations" in data
    assert "relevant_sources" in data
    assert "query_type" in data
    assert "retry_count" in data
    assert "grounding_status" in data
    assert "sufficient_context" in data


def test_get_documents_catalog(client: TestClient):
    """Test GET /documents lists indexed documents without leaking internal filepaths."""
    response = client.get("/documents")
    assert response.status_code == 200
    data = response.json()
    assert "total_documents" in data
    assert "total_chunks" in data
    assert "documents" in data
    assert isinstance(data["documents"], list)

    for doc in data["documents"]:
        assert "source_id" in doc
        assert "document_title" in doc
        assert "source_filename" in doc
        assert "chunk_count" in doc
        # Ensure raw Windows absolute path C:\ is not leaked as filename
        assert ":\\" not in doc["source_filename"]


def test_feedback_submission(client: TestClient):
    """Test POST /feedback persists thumbs_up feedback and returns feedback_id."""
    response = client.post("/feedback", json={
        "rating": "thumbs_up",
        "question": "How do dependencies work?",
        "comment": "Very clear and helpful explanation."
    })
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "feedback_id" in data
    assert data["feedback_id"] > 0


def test_feedback_invalid_rating(client: TestClient):
    """Test POST /feedback rejects invalid rating strings."""
    response = client.post("/feedback", json={
        "rating": "neutral",
        "comment": "Average answer"
    })
    assert response.status_code == 422


def test_ingest_no_payload(client: TestClient):
    """Test POST /ingest returns 400 when neither file nor URL is provided."""
    response = client.post("/ingest")
    assert response.status_code == 400


def test_ingest_invalid_file_extension(client: TestClient):
    """Test POST /ingest rejects executable or non-text file extensions."""
    file_content = b"Binary or executable script content"
    files = {"files": ("malicious.exe", io.BytesIO(file_content), "application/octet-stream")}
    response = client.post("/ingest", files=files)
    assert response.status_code == 400
    assert "rejected" in response.json()["detail"].lower()


def test_ingest_oversized_file(client: TestClient):
    """Test POST /ingest rejects files exceeding the maximum size limit."""
    # 6 MB payload (exceeds 5 MB limit)
    large_payload = b"A" * (6 * 1024 * 1024)
    files = {"files": ("large_doc.md", io.BytesIO(large_payload), "text/markdown")}
    response = client.post("/ingest", files=files)
    assert response.status_code == 400
    assert "exceeds" in response.json()["detail"].lower()


def test_ingest_valid_markdown_file(client: TestClient, sample_markdown_doc: str):
    """Test POST /ingest successfully parses and indexes uploaded markdown."""
    files = {"files": ("test_doc.md", io.BytesIO(sample_markdown_doc.encode("utf-8")), "text/markdown")}
    response = client.post("/ingest", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("success", "partial_success")
    assert data["documents_ingested"] >= 1
    assert data["chunks_created"] >= 1


def test_ingest_invalid_url_scheme(client: TestClient):
    """Test POST /ingest rejects non-HTTP schemes (e.g. ftp, file)."""
    response = client.post("/ingest", data={"url": "ftp://files.example.com/doc.txt"})
    assert response.status_code == 400
    assert "unsupported url scheme" in response.json()["detail"].lower()


def test_ingest_ssrf_loopback(client: TestClient):
    """Test POST /ingest blocks loopback localhost / 127.0.0.1 requests."""
    response = client.post("/ingest", data={"url": "http://127.0.0.1:8000/internal"})
    assert response.status_code == 400
    assert "disallowed" in response.json()["detail"].lower() or "forbidden" in response.json()["detail"].lower()


def test_ingest_ssrf_private_network(client: TestClient):
    """Test POST /ingest blocks RFC 1918 private network IPs (192.168.x.x)."""
    response = client.post("/ingest", data={"url": "http://192.168.1.1/admin"})
    assert response.status_code == 400
    assert "private or restricted" in response.json()["detail"].lower()


def test_ingest_ssrf_cloud_metadata(client: TestClient):
    """Test POST /ingest blocks cloud metadata IP 169.254.169.254."""
    response = client.post("/ingest", data={"url": "http://169.254.169.254/latest/meta-data"})
    assert response.status_code == 400
    assert "private or restricted" in response.json()["detail"].lower()
