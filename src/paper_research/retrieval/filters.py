from pydantic import BaseModel, Field

from paper_research.chunking.types import Chunk


class RetrievalFilter(BaseModel):
    paper_ids: list[str] | None = None
    sections: list[str] | None = None
    block_types: list[str] | None = None
    page_from: int | None = Field(default=None, ge=1)
    page_to: int | None = Field(default=None, ge=1)

    def matches(self, chunk: Chunk) -> bool:
        if self.paper_ids and chunk.paper_id not in self.paper_ids:
            return False
        if self.sections and not any(
            wanted.lower() in " > ".join(chunk.section_path).lower()
            for wanted in self.sections
        ):
            return False
        if self.block_types and chunk.block_type not in self.block_types:
            return False
        if self.page_from and chunk.page_end < self.page_from:
            return False
        return not (self.page_to and chunk.page_start > self.page_to)
