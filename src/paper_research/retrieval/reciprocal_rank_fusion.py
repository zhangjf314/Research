"""Deterministic reciprocal-rank fusion."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

HYBRID_RRF_VERSION = "hybrid-rrf-v1-candidate"
FROZEN_RRF_GRID = {
    "rrf_k": [20, 40, 60],
    "dense_weight": [1.0],
    "bm25_weight": [0.5, 1.0],
    "numeric_weight": [0.5, 1.0, 1.5],
}


class RankedLike(Protocol):
    doc_id: str
    rank: int


@dataclass(frozen=True)
class FusedResult:
    doc_id: str
    score: float
    rank: int
    sources: tuple[str, ...]


def reciprocal_rank_fusion(
    ranked_lists: dict[str, Iterable[RankedLike]],
    *,
    rrf_k: int = 60,
    weights: dict[str, float] | None = None,
    top_k: int = 12,
) -> tuple[FusedResult, ...]:
    weights = weights or {}
    scores: dict[str, float] = {}
    sources: dict[str, set[str]] = {}
    for name, results in ranked_lists.items():
        weight = weights.get(name, 1.0)
        for item in results:
            scores[item.doc_id] = scores.get(item.doc_id, 0.0) + weight / (rrf_k + item.rank)
            sources.setdefault(item.doc_id, set()).add(name)
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
    return tuple(
        FusedResult(
            doc_id=doc_id,
            score=score,
            rank=rank,
            sources=tuple(sorted(sources[doc_id])),
        )
        for rank, (doc_id, score) in enumerate(ordered, 1)
    )
