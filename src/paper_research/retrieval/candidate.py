"""V2 candidate provenance, independent of legacy fusion/reranker objects."""

from pydantic import BaseModel, Field, SerializeAsAny

from paper_research.chunking.types import Chunk
from paper_research.retrieval.fusion import FusedResult


class RetrievalCandidate(BaseModel):
    chunk: SerializeAsAny[Chunk]
    raw_text: str
    contextual_text: str
    dense_score: float | None = None
    sparse_score: float | None = None
    fusion_score: float
    rerank_score: float | None = None
    dense_rank: int | None = None
    sparse_rank: int | None = None
    retrieval_sources: list[str] = Field(default_factory=list)
    ranking_scores: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, object] = Field(default_factory=dict)

    @classmethod
    def from_fused(
        cls, item: FusedResult, dense_scores: dict[str, float], sparse_scores: dict[str, float]
    ) -> "RetrievalCandidate":
        cid = item.chunk.chunk_id
        return cls(
            chunk=item.chunk,
            raw_text=item.chunk.chunk_text,
            contextual_text=getattr(item.chunk, "contextual_text", item.chunk.chunk_text),
            dense_score=dense_scores.get(cid),
            sparse_score=sparse_scores.get(cid),
            fusion_score=item.score,
            dense_rank=item.dense_rank,
            sparse_rank=item.sparse_rank,
            retrieval_sources=[
                name
                for name, scores in (("dense", dense_scores), ("sparse", sparse_scores))
                if cid in scores
            ],
            metadata=getattr(item.chunk, "contextual_metadata", {}),
        )

    def as_fused(self) -> FusedResult:
        return FusedResult(
            self.chunk,
            self.rerank_score if self.rerank_score is not None else self.fusion_score,
            self.dense_rank,
            self.sparse_rank,
        )
