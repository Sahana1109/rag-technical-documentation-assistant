"""
Application Configuration Module
Loads environment variables and provides a centralized Settings object.
"""
import os
from functools import lru_cache
from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# Pre-load .env file if it exists
load_dotenv()


class Settings(BaseSettings):
    """Application settings with environment variable fallbacks."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Application Information
    app_name: str = "RAG Technical Documentation Assistant"
    app_version: str = "1.0.0"

    # LLM Configuration
    # Options: "gemini", "openai", "mock"
    llm_provider: str = os.getenv("LLM_PROVIDER", "gemini").lower()
    llm_model: str = os.getenv("LLM_MODEL", "gemini-1.5-flash")
    google_api_key: Optional[str] = os.getenv("GOOGLE_API_KEY", None)
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY", None)

    # Embedding Configuration
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # ChromaDB Vector Store Configuration
    chroma_persist_directory: str = os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/chroma")
    chroma_collection_name: str = os.getenv("CHROMA_COLLECTION_NAME", "fastapi_docs")

    # Ingestion & Chunking Parameters
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "120"))
    retrieval_top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "4"))
    max_retries: int = int(os.getenv("MAX_RETRIES", "2"))

    # Security & File Ingestion Limits
    max_file_size_mb: int = int(os.getenv("MAX_FILE_SIZE_MB", "5"))
    request_timeout_seconds: int = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "10"))
    allowed_extensions: str = os.getenv("ALLOWED_EXTENSIONS", ".md,.txt,.html")

    # Feedback Storage
    feedback_db_path: str = os.getenv("FEEDBACK_DB_PATH", "./data/feedback.db")

    # Server & CORS
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))
    cors_allowed_origins: str = os.getenv(
        "CORS_ALLOWED_ORIGINS",
        "http://localhost:3000,http://localhost:8501,http://127.0.0.1:8501"
    )

    @property
    def allowed_extensions_list(self) -> List[str]:
        return [ext.strip().lower() for ext in self.allowed_extensions.split(",") if ext.strip()]

    @property
    def cors_origins_list(self) -> List[str]:
        return [origin.strip() for origin in self.cors_allowed_origins.split(",") if origin.strip()]

    @property
    def is_llm_configured(self) -> bool:
        if self.llm_provider == "gemini":
            return bool(self.google_api_key and self.google_api_key != "your_google_api_key_here")
        if self.llm_provider == "openai":
            return bool(self.openai_api_key and self.openai_api_key != "your_openai_api_key_here")
        if self.llm_provider == "mock":
            return True
        return False


@lru_cache()
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
