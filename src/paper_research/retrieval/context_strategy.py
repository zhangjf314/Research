"""Auditable context selection and bounded structural expansion."""

from collections import Counter, defaultdict
from dataclasses import dataclass

from pydantic import BaseModel, Field, model_validator

from paper_research.chunking.types import Chunk
from paper_research.retrieval.context_builder import ContextItem
from paper_research.retrieval.fusion import FusedResult


class ContextStrategy(BaseModel):
    retrieval_k: int = Field(default=20, ge=1)
    context_k: int = Field(default=10, ge=1)
    neighbor_window: int = Field(default=0, ge=0)
    page_expansion: bool = False
    max_blocks_per_page: int | None = Field(default=None, ge=1)
    max_blocks_per_section: int | None = Field(default=None, ge=1)
    max_context_characters: int = Field(default=12000, ge=1)
    max_context_tokens: int = Field(default=12000, ge=1)
    dense_weight: float = Field(default=0.5, ge=0)
    lexical_weight: float = Field(default=0.5, ge=0)

    @model_validator(mode="after")
    def validate_weights(self) -> "ContextStrategy":
        if self.dense_weight + self.lexical_weight <= 0:
            raise ValueError("at least one hybrid weight must be positive")
        return self


class ContextCandidateTrace(BaseModel):
    chunk_id: str
    paper_id: str
    original_rank: int | None
    original_score: float | None
    expansion_reason: str
    expansion_source_chunk_id: str | None = None
    deduplicated: bool = False
    excluded_reason: str | None = None
    final_context_rank: int | None = None
    estimated_tokens: int = 0
    token_truncated: bool = False


class StrategicContextTrace(BaseModel):
    strategy: ContextStrategy
    input_candidate_count: int
    candidate_trace: list[ContextCandidateTrace] = Field(default_factory=list)
    duplicate_chunk_ids: list[str] = Field(default_factory=list)
    output_chunk_ids: list[str] = Field(default_factory=list)
    estimated_tokens: int = 0
    truncated_chunk_ids: list[str] = Field(default_factory=list)
    page_counts: dict[str, int] = Field(default_factory=dict)
    section_counts: dict[str, int] = Field(default_factory=dict)


@dataclass(frozen=True)
class StrategicContextResult:
    context: list[ContextItem]
    trace: StrategicContextTrace


def _block_number(chunk: Chunk) -> int:
    for block_id in chunk.block_ids:
        digits = "".join(character for character in block_id if character.isdigit())
        if digits:
            return int(digits)
    return 10**12


def _section_key(chunk: Chunk) -> str:
    return " / ".join(chunk.section_path) or "<root>"


def _page_keys(chunk: Chunk) -> list[str]:
    return [f"{chunk.paper_id}:{page}" for page in range(chunk.page_start, chunk.page_end + 1)]


class StrategicContextBuilder:
    """Select retrieved chunks, expand structurally, deduplicate, cap, and truncate."""

    def __init__(self, chunks: list[Chunk], strategy: ContextStrategy) -> None:
        self.strategy = strategy
        grouped: dict[str, list[Chunk]] = defaultdict(list)
        for chunk in chunks:
            grouped[chunk.paper_id].append(chunk)
        self.by_paper = {
            paper_id: sorted(
                items,
                key=lambda item: (item.page_start, _block_number(item), item.chunk_id),
            )
            for paper_id, items in grouped.items()
        }
        self.position = {
            chunk.chunk_id: index
            for items in self.by_paper.values()
            for index, chunk in enumerate(items)
        }

    def build(self, fused: list[FusedResult]) -> StrategicContextResult:
        ranked = fused[: self.strategy.retrieval_k]
        initial = ranked[: self.strategy.context_k]
        original = {
            item.chunk.chunk_id: (rank, item.score)
            for rank, item in enumerate(ranked, start=1)
        }
        proposed: list[tuple[Chunk, str, str | None]] = []
        for item in initial:
            proposed.append((item.chunk, "retrieved", None))
            self._add_neighbors(proposed, item)
            self._add_same_page(proposed, item)
        return self._select(proposed, original, len(ranked))

    def _add_neighbors(
        self,
        proposed: list[tuple[Chunk, str, str | None]],
        item: FusedResult,
    ) -> None:
        source = item.chunk
        paper_chunks = self.by_paper[source.paper_id]
        position = self.position[source.chunk_id]
        for distance in range(1, self.strategy.neighbor_window + 1):
            for neighbor_position in (position - distance, position + distance):
                if 0 <= neighbor_position < len(paper_chunks):
                    proposed.append(
                        (
                            paper_chunks[neighbor_position],
                            f"neighbor_window_{distance}",
                            source.chunk_id,
                        )
                    )

    def _add_same_page(
        self,
        proposed: list[tuple[Chunk, str, str | None]],
        item: FusedResult,
    ) -> None:
        if not self.strategy.page_expansion:
            return
        source = item.chunk
        source_pages = set(range(source.page_start, source.page_end + 1))
        for candidate in self.by_paper[source.paper_id]:
            candidate_pages = set(range(candidate.page_start, candidate.page_end + 1))
            if source_pages & candidate_pages:
                proposed.append((candidate, "same_page", source.chunk_id))

    def _select(
        self,
        proposed: list[tuple[Chunk, str, str | None]],
        original: dict[str, tuple[int, float]],
        input_count: int,
    ) -> StrategicContextResult:
        seen: set[str] = set()
        duplicates: list[str] = []
        traces: list[ContextCandidateTrace] = []
        accepted: list[tuple[Chunk, float]] = []
        page_counts: Counter[str] = Counter()
        section_counts: Counter[str] = Counter()
        used_tokens = 0
        used_characters = 0
        truncated: list[str] = []
        for chunk, reason, source_id in proposed:
            rank, score = original.get(chunk.chunk_id, (None, None))
            candidate_trace = ContextCandidateTrace(
                chunk_id=chunk.chunk_id,
                paper_id=chunk.paper_id,
                original_rank=rank,
                original_score=score,
                expansion_reason=reason,
                expansion_source_chunk_id=source_id,
                estimated_tokens=chunk.token_count,
            )
            exclusion = self._exclusion(chunk, seen, page_counts, section_counts)
            if exclusion:
                candidate_trace.excluded_reason = exclusion
                candidate_trace.deduplicated = exclusion == "duplicate"
                if exclusion == "duplicate":
                    duplicates.append(chunk.chunk_id)
                traces.append(candidate_trace)
                continue
            seen.add(chunk.chunk_id)
            remaining_tokens = self.strategy.max_context_tokens - used_tokens
            remaining_characters = self.strategy.max_context_characters - used_characters
            if remaining_tokens <= 0 or remaining_characters <= 0:
                candidate_trace.excluded_reason = "token_budget"
                candidate_trace.token_truncated = True
                truncated.append(chunk.chunk_id)
                traces.append(candidate_trace)
                continue
            evidence_characters = min(len(chunk.chunk_text), remaining_characters)
            estimated_tokens = max(1, (evidence_characters + 3) // 4)
            tokens = min(estimated_tokens, remaining_tokens)
            evidence_characters = min(evidence_characters, tokens * 4)
            candidate_trace.estimated_tokens = tokens
            candidate_trace.token_truncated = evidence_characters < len(chunk.chunk_text)
            if candidate_trace.token_truncated:
                truncated.append(chunk.chunk_id)
            candidate_trace.final_context_rank = len(accepted) + 1
            traces.append(candidate_trace)
            accepted.append((chunk, score or 0.0))
            used_tokens += tokens
            used_characters += evidence_characters
            page_counts.update(_page_keys(chunk))
            section_counts[f"{chunk.paper_id}:{_section_key(chunk)}"] += 1
            if candidate_trace.token_truncated:
                break
        return self._result(
            accepted,
            traces,
            duplicates,
            truncated,
            page_counts,
            section_counts,
            used_tokens,
            input_count,
        )

    def _exclusion(
        self,
        chunk: Chunk,
        seen: set[str],
        page_counts: Counter[str],
        section_counts: Counter[str],
    ) -> str | None:
        if chunk.chunk_id in seen:
            return "duplicate"
        if self.strategy.max_blocks_per_page is not None and any(
            page_counts[key] >= self.strategy.max_blocks_per_page for key in _page_keys(chunk)
        ):
            return "page_cap"
        section_key = f"{chunk.paper_id}:{_section_key(chunk)}"
        if (
            self.strategy.max_blocks_per_section is not None
            and section_counts[section_key] >= self.strategy.max_blocks_per_section
        ):
            return "section_cap"
        return None

    def _result(
        self,
        accepted: list[tuple[Chunk, float]],
        traces: list[ContextCandidateTrace],
        duplicates: list[str],
        truncated: list[str],
        page_counts: Counter[str],
        section_counts: Counter[str],
        used_tokens: int,
        input_count: int,
    ) -> StrategicContextResult:
        trace_by_id = {
            item.chunk_id: item for item in traces if item.final_context_rank is not None
        }
        context = [
            ContextItem(
                chunk_id=chunk.chunk_id,
                paper_id=chunk.paper_id,
                block_ids=chunk.block_ids,
                section_path=chunk.section_path,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                evidence=chunk.chunk_text[: trace_by_id[chunk.chunk_id].estimated_tokens * 4],
                score=score,
            )
            for chunk, score in accepted
        ]
        return StrategicContextResult(
            context=context,
            trace=StrategicContextTrace(
                strategy=self.strategy,
                input_candidate_count=input_count,
                candidate_trace=traces,
                duplicate_chunk_ids=duplicates,
                output_chunk_ids=[item.chunk_id for item in context],
                estimated_tokens=used_tokens,
                truncated_chunk_ids=truncated,
                page_counts=dict(page_counts),
                section_counts=dict(section_counts),
            ),
        )
