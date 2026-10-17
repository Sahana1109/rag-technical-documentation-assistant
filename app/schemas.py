"""
Pydantic Schemas for Request and Response Models
"""
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    """Incoming user query request."""
    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="The technical question to query against the documentation corpus.",
        examples=["How do dependencies work in FastAPI?"]
    )

    @field_validator("question")
    @classmethod
    def validate_not_blank(cls, v: str) -> str:
        trimmed = v.strip()
        if not trimmed:
            raise ValueError("Question cannot be empty or contain only whitespace.")
        return trimmed


class Citation(BaseModel):
    """Citation pointing to a specific document section and chunk."""
    document_title: str
    section: str
    source_filename: str
    source_url: Optional[str] = None
    chunk_index: int


class SourceMetadata(BaseModel):
    """Detailed metadata for a retrieved and relevant documentation chunk."""
    source_filename: str
    document_title: str
    section: str
    source_url: Optional[str] = None
    score: Optional[float] = None
    snippet: str


class QueryResponse(BaseModel):
    """Response returned by the query endpoint."""
    question: str
    answer: str
    citations: List[Citation]
    relevant_sources: List[SourceMetadata]
    query_type: str
    retry_count: int
    grounding_status: str
    sufficient_context: bool


class IngestResponse(BaseModel):
    """Result of document ingestion."""
    status: str
    documents_ingested: int
    chunks_created: int
    sources: List[str]
    errors: List[str] = []


class DocumentInfo(BaseModel):
    """Metadata summary of an indexed document in the vector store."""
    source_id: str
    document_title: str
    source_filename: str
    source_url: Optional[str] = None
    chunk_count: int
    source_type: str


class DocumentListResponse(BaseModel):
    """Response returned by the documents listing endpoint."""
    total_documents: int
    total_chunks: int
    documents: List[DocumentInfo]


class FeedbackRequest(BaseModel):
    """User feedback on a query answer."""
    query_id: Optional[str] = Field(None, description="Optional identifier of the query")
    question: Optional[str] = Field(None, description="The question asked")
    rating: str = Field(
        ...,
        pattern="^(thumbs_up|thumbs_down)$",
        description="Rating: 'thumbs_up' or 'thumbs_down'"
    )
    comment: Optional[str] = Field(
        None,
        max_length=1000,
        description="Optional textual comment"
    )


class FeedbackResponse(BaseModel):
    """Response confirming feedback persistence."""
    feedback_id: int
    status: str
    message: str


class HealthResponse(BaseModel):
    """System health and dependency diagnostic information."""
    status: str
    app_name: str
    version: str
    vector_db_status: str
    total_indexed_chunks: int
    llm_provider: str
    llm_configured: bool
    embedding_model: str
