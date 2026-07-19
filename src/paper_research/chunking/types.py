import uuid

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    paper_id: str
    block_ids: list[str]
    section_path: list[str] = Field(default_factory=list)
    block_type: str
    page_start: int
    page_end: int
    block_page_map: dict[str, int] | None = None
    chunk_text: str
    parent_context: str | None = None
    previous_context: str | None = None
    next_context: str | None = None
    token_count: int
