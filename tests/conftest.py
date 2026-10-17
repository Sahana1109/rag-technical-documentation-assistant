"""
Pytest Fixtures and Global Test Configuration
"""
import os
import sys
import tempfile
import pytest

# Ensure root directory is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import app
from app.rag.llm import DeterministicMockLLM


@pytest.fixture(scope="session", autouse=True)
def configure_test_environment():
    """Sets test environment variables before tests run."""
    os.environ["LLM_PROVIDER"] = "mock"
    os.environ["MAX_RETRIES"] = "2"
    os.environ["REQUEST_TIMEOUT_SECONDS"] = "5"
    # Clear settings cache so new env applies
    get_settings.cache_clear()


@pytest.fixture
def client() -> TestClient:
    """Provides a FastAPI TestClient."""
    return TestClient(app)


@pytest.fixture
def mock_llm() -> DeterministicMockLLM:
    """Provides an isolated DeterministicMockLLM instance."""
    return DeterministicMockLLM()


@pytest.fixture
def sample_markdown_doc() -> str:
    """Provides a valid Markdown document string for ingestion testing."""
    return """# FastAPI Dependency Injection System
Source: https://fastapi.tiangolo.com/tutorial/dependencies/

## Understanding Dependencies
FastAPI has a very powerful and intuitive Dependency Injection system.
It allows declaring reusable components using Depends().

## Sub-dependencies
Dependencies can have sub-dependencies forming a dependency tree.
"""
