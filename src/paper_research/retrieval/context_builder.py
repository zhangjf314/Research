from pydantic import BaseModel, Field

from paper_research.retrieval.fusion import FusedResult


class ContextItem(BaseModel):
    chunk_id: str
    paper_id: str
    block_ids: list[str] = Field(default_factory=list)
    section_path: list[str]
    page_start: int
    page_end: int
    evidence: str
    score: float


class ContextBuildTrace(BaseModel):
    input_candidate_count: int = 0
    output_chunk_ids: list[str] = Field(default_factory=list)
    duplicate_chunk_ids: list[str] = Field(default_factory=list)
    token_budget: int | None = None
    estimated_tokens: int = 0
    truncated_chunk_id: str | None = None
    truncation_strategy: str = "rank_order_prefix"


class ContextBuilder:
    def __init__(
        self,
        include_neighbors: bool = True,
        max_characters: int = 12000,
        max_tokens: int | None = None,
    ) -> None:
        self.include_neighbors = include_neighbors
        self.max_characters = max_characters
        self.max_tokens = max_tokens
        self.last_trace = ContextBuildTrace(token_budget=max_tokens)

    def build(self, results: list[FusedResult]) -> list[ContextItem]:
        items: list[ContextItem] = []
        used = 0
        used_tokens = 0
        seen: set[str] = set()
        duplicates: list[str] = []
        truncated_chunk_id = None
        for result in results:
            if result.chunk.chunk_id in seen:
                duplicates.append(result.chunk.chunk_id)
                continue
            seen.add(result.chunk.chunk_id)
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
            estimated_tokens = max(1, (len(evidence) + 3) // 4)
            if self.max_tokens is not None:
                remaining_tokens = self.max_tokens - used_tokens
                if remaining_tokens <= 0:
                    break
                if estimated_tokens > remaining_tokens:
                    evidence = evidence[: remaining_tokens * 4]
                    estimated_tokens = remaining_tokens
                    truncated_chunk_id = result.chunk.chunk_id
            used += len(evidence)
            used_tokens += estimated_tokens
            items.append(
                ContextItem(
                    chunk_id=result.chunk.chunk_id,
                    paper_id=result.chunk.paper_id,
                    block_ids=result.chunk.block_ids,
                    section_path=result.chunk.section_path,
                    page_start=result.chunk.page_start,
                    page_end=result.chunk.page_end,
                    evidence=evidence,
                    score=result.score,
                )
            )
            if truncated_chunk_id:
                break
        self.last_trace = ContextBuildTrace(
            input_candidate_count=len(results),
            output_chunk_ids=[item.chunk_id for item in items],
            duplicate_chunk_ids=duplicates,
            token_budget=self.max_tokens,
            estimated_tokens=used_tokens,
            truncated_chunk_id=truncated_chunk_id,
        )
        return items
