from pydantic import BaseModel

from paper_research.retrieval.fusion import FusedResult


class ContextItem(BaseModel):
    chunk_id: str
    paper_id: str
    section_path: list[str]
    page_start: int
    page_end: int
    evidence: str
    score: float


class ContextBuilder:
    def __init__(self, include_neighbors: bool = True, max_characters: int = 12000) -> None:
        self.include_neighbors = include_neighbors
        self.max_characters = max_characters

    def build(self, results: list[FusedResult]) -> list[ContextItem]:
        items: list[ContextItem] = []
        used = 0
        for result in results:
            parts = [result.chunk.chunk_text]
            if self.include_neighbors and result.chunk.previous_context:
                parts.insert(0, result.chunk.previous_context)
            if self.include_neighbors and result.chunk.next_context:
                parts.append(result.chunk.next_context)
            evidence = "\n\n".join(parts)
            remaining = self.max_characters - used
            if remaining <= 0:
                break
            evidence = evidence[:remaining]
            used += len(evidence)
            items.append(
                ContextItem(
                    chunk_id=result.chunk.chunk_id,
                    paper_id=result.chunk.paper_id,
                    section_path=result.chunk.section_path,
                    page_start=result.chunk.page_start,
                    page_end=result.chunk.page_end,
                    evidence=evidence,
                    score=result.score,
                )
            )
        return items
