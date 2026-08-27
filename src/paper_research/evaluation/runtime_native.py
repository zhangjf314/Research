"""Runtime-native paired finalization helpers for RQ17-R1.

The helper deliberately owns no retriever and accepts a completed P0 common
trunk.  That makes an accidental second retrieval mechanically impossible in
the C4 arm.
"""

from __future__ import annotations

from dataclasses import dataclass

from paper_research.retrieval.context_builder import ContextBuilder, ContextItem
from paper_research.retrieval.fusion import FusedResult
from paper_research.retrieval.hybrid import CommonTrunkOutput, HybridRetriever
from paper_research.retrieval.reranker import Reranker, RerankerProviderError


@dataclass(frozen=True)
class C4Finalization:
    """The post-common-trunk C4 output and explicit provider reliability data."""

    ranked: list[FusedResult]
    context: list[ContextItem]
    reranker_scores: list[float]
    latency_ms: float
    api_request_count: int
    failure_reason: str | None
    fallback: bool


class C4RuntimeNativeFinalizer:
    """Rerank exactly the supplied P0 candidates using the original question."""

    def __init__(self, reranker: Reranker, context_builder: ContextBuilder) -> None:
        self.reranker = reranker
        self.context_builder = context_builder

    def finalize(self, trunk: CommonTrunkOutput, *, top_k: int) -> C4Finalization:
        candidates = trunk.candidates
        try:
            outcome = self.reranker.rerank_with_trace(
                trunk.original_query, candidates, len(candidates)
            )
            ranked = outcome.results
            failure_reason = outcome.failure_reason
            fallback = outcome.fallback_occurred
            latency_ms = outcome.latency_ms
            request_count = outcome.api_request_count
        except RerankerProviderError as exc:
            # Registered C4-R1 terminal fallback: preserve P0 order only.
            ranked = candidates
            failure_reason = str(exc)
            fallback = True
            latency_ms = 0.0
            request_count = exc.api_request_count
        context_candidates = HybridRetriever._context_candidates(
            trunk.original_query,
            ranked,
            top_k=top_k,
            retrieval_scope=trunk.retrieval_scope,
        )
        context = self.context_builder.build(context_candidates)
        return C4Finalization(
            ranked=ranked,
            context=context,
            reranker_scores=[item.score for item in ranked],
            latency_ms=latency_ms,
            api_request_count=request_count,
            failure_reason=failure_reason,
            fallback=fallback,
        )
