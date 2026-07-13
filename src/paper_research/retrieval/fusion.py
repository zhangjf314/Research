from dataclasses import dataclass

from paper_research.chunking.types import Chunk
from paper_research.retrieval.dense import RetrievalResult


@dataclass(frozen=True)
class FusedResult:
    chunk: Chunk
    score: float
    dense_rank: int | None = None
    sparse_rank: int | None = None


def reciprocal_rank_fusion(
    dense: list[RetrievalResult], sparse: list[RetrievalResult], k: int = 60
) -> list[FusedResult]:
    combined: dict[str, dict[str, object]] = {}
    for source, results in (("dense", dense), ("sparse", sparse)):
        for rank, result in enumerate(results, start=1):
            entry = combined.setdefault(
                result.chunk.chunk_id,
                {"chunk": result.chunk, "score": 0.0, "dense_rank": None, "sparse_rank": None},
            )
            entry["score"] = float(entry["score"]) + 1 / (k + rank)
            entry[f"{source}_rank"] = rank
    fused = [
        FusedResult(
            chunk=entry["chunk"],  # type: ignore[arg-type]
            score=float(entry["score"]),
            dense_rank=entry["dense_rank"],  # type: ignore[arg-type]
            sparse_rank=entry["sparse_rank"],  # type: ignore[arg-type]
        )
        for entry in combined.values()
    ]
    return sorted(fused, key=lambda item: item.score, reverse=True)
