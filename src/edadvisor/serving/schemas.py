from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=100)
    question: str = Field(min_length=1, max_length=500)
    doc_filter: str | None = Field(
        default=None,
        description="Optional source filter: 'pdf', 'html', 'docx'",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "student-abc-123",
                "question": "What are the English language requirements for Oxford MSc Computer Science?",
            }
        }


class SourceRef(BaseModel):
    source_n: int
    source: str
    excerpt: str
    page: int | None = None
    doc_type: str
    section: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceRef]
    confidence: float
    escalated: bool
    session_id: str
    prompt_version: str


class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded"]
    index_loaded: bool
    uptime_seconds: float
