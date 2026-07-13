from typing import Literal

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float


class PaperMetadata(BaseModel):
    title: str | None = None
    authors: list[str] = Field(default_factory=list)
    page_count: int
    pdf_metadata: dict[str, str] = Field(default_factory=dict)


class PaperBlock(BaseModel):
    block_id: str
    block_type: Literal["title", "heading", "paragraph", "table", "formula", "reference"]
    section_path: list[str] = Field(default_factory=list)
    page_start: int
    page_end: int
    block_index: int
    text: str
    bbox: BoundingBox
    parent_block_id: str | None = None
    previous_block_id: str | None = None
    next_block_id: str | None = None
    source_page: int | None = None
    is_ocr: bool = False
    ocr_confidence: float | None = None


class ParseWarning(BaseModel):
    code: str
    message: str
    page: int | None = None


class ParsedPaper(BaseModel):
    parser: str
    parser_name: str | None = None
    is_ocr: bool = False
    ocr_confidence: float | None = None
    metadata: PaperMetadata
    blocks: list[PaperBlock]
    warnings: list[ParseWarning] = Field(default_factory=list)
