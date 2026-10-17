"""
FastAPI Application Entry Point
Exposes REST API endpoints for RAG querying, document ingestion, document cataloging,
user feedback, and health diagnostics.
"""
import logging
import os
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    status
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.schemas import (
    QueryRequest,
    QueryResponse,
    IngestResponse,
    DocumentListResponse,
    DocumentInfo,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
)
from app.services.query_service import run_rag_pipeline
from app.services.feedback_service import init_feedback_db, record_feedback
from app.rag.ingestion import (
    ingest_document,
    ingest_directory,
    list_indexed_documents,
    fetch_url_content,
    validate_url_safe,
    get_chroma_collection,
)

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    Initializes database and auto-indexes default documentation if empty.
    """
    settings = get_settings()
    logger.info(f"Starting {settings.app_name} v{settings.app_version}...")

    # Initialize feedback SQLite database
    init_feedback_db()

    # Pre-index default corpus from data/docs if collection is empty
    try:
        collection = get_chroma_collection()
        if collection.count() == 0:
            docs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "docs")
            if os.path.exists(docs_dir):
                logger.info(f"Initial collection is empty. Auto-indexing default docs from: {docs_dir}")
                result = ingest_directory(docs_dir)
                logger.info(f"Auto-indexed {result['documents_ingested']} documents with {result['chunks_created']} chunks.")
    except Exception as e:
        logger.warning(f"Note on startup indexing: {e}")

    yield

    logger.info("Shutting down application...")


settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Self-Corrective RAG Documentation Assistant using LangGraph StateGraph, "
        "FastAPI, ChromaDB, and Sentence Transformers."
    ),
    lifespan=lifespan
)

# Configure CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------------------
# API Endpoints
# ------------------------------------------------------------------------------

@app.get("/", tags=["General"])
def read_root():
    """Returns basic API service metadata and links to interactive documentation."""
    return {
        "app_name": settings.app_name,
        "version": settings.app_version,
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "health_url": "/health",
        "status": "online"
    }


@app.get("/health", response_model=HealthResponse, tags=["Diagnostics"])
def health_check():
    """
    Evaluates system health and dependency connectivity.
    Reports ChromaDB vector count, embedding model, and LLM provider configuration status.
    """
    current_settings = get_settings()
    vdb_status = "unreachable"
    indexed_chunks = 0

    try:
        collection = get_chroma_collection()
        indexed_chunks = collection.count()
        vdb_status = "connected"
    except Exception as e:
        logger.error(f"Vector database health check failed: {e}")
        vdb_status = f"error: {str(e)}"

    is_healthy = (vdb_status == "connected") and (current_settings.is_llm_configured or current_settings.llm_provider == "mock")

    return HealthResponse(
        status="healthy" if is_healthy else "degraded",
        app_name=current_settings.app_name,
        version=current_settings.app_version,
        vector_db_status=vdb_status,
        total_indexed_chunks=indexed_chunks,
        llm_provider=current_settings.llm_provider,
        llm_configured=current_settings.is_llm_configured,
        embedding_model=current_settings.embedding_model
    )


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
def query_documentation(request: QueryRequest):
    """
    Executes the self-corrective LangGraph RAG workflow on a user question.
    - Query analysis and classification
    - Dense retrieval from ChromaDB
    - Relevance grading of retrieved chunks
    - Conditional query rewriting on irrelevant retrieval
    - Grounded answer generation with citations
    - Grounding verification
    """
    current_settings = get_settings()

    if not request.question or not request.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty or contain only whitespace."
        )

    # Check for unconfigured real LLM provider
    if current_settings.llm_provider != "mock" and not current_settings.is_llm_configured:
        key_var = "GOOGLE_API_KEY" if current_settings.llm_provider == "gemini" else "OPENAI_API_KEY"
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"LLM provider '{current_settings.llm_provider}' is selected, but {key_var} is not configured. "
                f"Please add a valid {key_var} to your .env file or set LLM_PROVIDER=mock for offline testing."
            )
        )

    try:
        response = run_rag_pipeline(request.question)
        return response
    except ValueError as ve:
        logger.error(f"Configuration or validation error during RAG query: {ve}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(ve)
        )
    except RuntimeError as re_err:
        logger.error(f"Upstream LLM runtime failure during RAG query: {re_err}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Upstream LLM failure: {str(re_err)}"
        )
    except Exception as e:
        logger.error(f"Query processing failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while executing the RAG query: {str(e)}"
        )


@app.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_documents(
    files: Optional[List[UploadFile]] = File(None),
    url: Optional[str] = Form(None)
):
    """
    Ingests technical documentation from uploaded files (.md, .txt, .html)
    or an external HTTP/HTTPS URL with SSRF protection.
    """
    if not files and not url:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Must provide either files to upload or a URL to ingest."
        )

    total_chunks = 0
    sources_ingested: List[str] = []
    errors: List[str] = []
    max_bytes = settings.max_file_size_mb * 1024 * 1024
    allowed_exts = tuple(settings.allowed_extensions_list)

    # 1. Process File Uploads
    if files:
        for file in files:
            filename = file.filename or "unknown.txt"
            # Validate extension
            if not any(filename.lower().endswith(ext) for ext in allowed_exts):
                errors.append(f"File '{filename}' rejected: extension must be one of {settings.allowed_extensions_list}")
                continue

            try:
                content_bytes = await file.read()
                if len(content_bytes) > max_bytes:
                    errors.append(f"File '{filename}' rejected: size exceeds maximum limit of {settings.max_file_size_mb} MB.")
                    continue

                content_str = content_bytes.decode("utf-8", errors="replace")
                chunks_count, source_id = ingest_document(
                    content=content_str,
                    filename=filename,
                    source_type="file"
                )
                total_chunks += chunks_count
                sources_ingested.append(source_id)
            except Exception as e:
                errors.append(f"Failed to process file '{filename}': {str(e)}")

    # 2. Process URL Ingestion
    if url:
        clean_url = url.strip()
        is_safe, error_reason = validate_url_safe(clean_url)
        if not is_safe:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid or disallowed URL: {error_reason}"
            )

        try:
            fetched_content = fetch_url_content(clean_url, timeout_seconds=settings.request_timeout_seconds)
            from urllib.parse import urlparse
            path_filename = os.path.basename(urlparse(clean_url).path) or "webpage.html"
            chunks_count, source_id = ingest_document(
                content=fetched_content,
                filename=path_filename,
                source_url=clean_url,
                source_type="url"
            )
            total_chunks += chunks_count
            sources_ingested.append(source_id)
        except HTTPException:
            raise
        except Exception as e:
            errors.append(f"Failed to fetch or index URL '{clean_url}': {str(e)}")

    if not sources_ingested and errors:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="; ".join(errors)
        )

    return IngestResponse(
        status="success" if not errors else "partial_success",
        documents_ingested=len(sources_ingested),
        chunks_created=total_chunks,
        sources=sources_ingested,
        errors=errors
    )


@app.get("/documents", response_model=DocumentListResponse, tags=["Catalog"])
def get_indexed_documents():
    """
    Lists indexed documents and catalog metadata stored in the vector database.
    Does not expose sensitive system paths.
    """
    try:
        raw_docs = list_indexed_documents()
        doc_models = [
            DocumentInfo(
                source_id=d["source_id"],
                document_title=d["document_title"],
                source_filename=d["source_filename"],
                source_url=d["source_url"],
                chunk_count=d["chunk_count"],
                source_type=d["source_type"]
            )
            for d in raw_docs
        ]
        total_chunks = sum(d.chunk_count for d in doc_models)
        return DocumentListResponse(
            total_documents=len(doc_models),
            total_chunks=total_chunks,
            documents=doc_models
        )
    except Exception as e:
        logger.error(f"Failed to list documents: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve document catalog: {str(e)}"
        )


@app.post("/feedback", response_model=FeedbackResponse, tags=["Feedback"])
def submit_feedback(feedback: FeedbackRequest):
    """
    Submits user feedback (thumbs up / thumbs down with optional comment)
    and persists it to the SQLite database.
    """
    try:
        feedback_id = record_feedback(
            rating=feedback.rating,
            query_id=feedback.query_id,
            question=feedback.question,
            comment=feedback.comment
        )
        return FeedbackResponse(
            feedback_id=feedback_id,
            status="success",
            message="Feedback submitted successfully."
        )
    except Exception as e:
        logger.error(f"Failed to save feedback: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to record feedback: {str(e)}"
        )
