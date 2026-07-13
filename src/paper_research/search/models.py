from datetime import date

from pydantic import BaseModel, Field


class PaperCandidate(BaseModel):
    source: str
    source_id: str
    title: str
    abstract: str | None = None
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    publication_date: date | None = None
    venue: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    source_url: str | None = None
    pdf_url: str | None = None
    citation_count: int = 0
    is_open_access: bool = False
    relevance_score: float = 0.0
    matched_queries: list[str] = Field(default_factory=list)


class SearchRequest(BaseModel):
    query: str = Field(min_length=2)
    limit: int = Field(default=20, ge=1, le=100)
    year_from: int | None = Field(default=None, ge=1900, le=2100)
    year_to: int | None = Field(default=None, ge=1900, le=2100)
    open_access_only: bool = False


class SearchResponse(BaseModel):
    original_query: str
    rewritten_queries: list[str]
    candidates: list[PaperCandidate]
    source_errors: dict[str, str] = Field(default_factory=dict)
    telemetry: dict[str, object] = Field(default_factory=dict)
