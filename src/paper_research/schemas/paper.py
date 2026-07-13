import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from paper_research.models.paper import PaperStatus


class PaperCreate(BaseModel):
    title: str = Field(min_length=1, max_length=1000)
    authors: list[str] = Field(default_factory=list)
    abstract: str | None = None
    keywords: list[str] = Field(default_factory=list)
    year: int | None = Field(default=None, ge=1600, le=2100)
    venue: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    source_type: str = "upload"
    source_url: str | None = None
    language: str | None = None


class PaperRead(PaperCreate):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    parse_status: PaperStatus
    index_status: str
    created_at: datetime
    updated_at: datetime


class PaperUploadResponse(BaseModel):
    paper: PaperRead
    duplicate: bool
    artifacts: dict[str, str]
