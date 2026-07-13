import time
from concurrent.futures import ThreadPoolExecutor

from paper_research.retrieval.context_builder import ContextBuilder, ContextItem
from paper_research.retrieval.dense import DenseRetriever, RetrievalResult
from paper_research.retrieval.filters import RetrievalFilter
from paper_research.retrieval.fusion import FusedResult, reciprocal_rank_fusion
from paper_research.retrieval.reranker import Reranker, RerankerProviderError, RerankOutcome
from paper_research.retrieval.sparse import BM25Retriever
from paper_research.retrieval.trace import (
    JsonlTraceRepository,
    RerankTraceCandidate,
    RetrievalTrace,
    TraceResult,
)


class HybridRetrievalResult:
    def __init__(self, context: list[ContextItem], trace: RetrievalTrace) -> None:
        self.context = context
        self.trace = trace


class HybridRetriever:
    def __init__(
        self,
        dense: DenseRetriever,
        sparse: BM25Retriever,
        reranker: Reranker,
        context_builder: ContextBuilder,
        trace_repository: JsonlTraceRepository | None = None,
        provider_metadata: dict[str, object] | None = None,
        rerank_input_k: int = 30,
        rerank_output_k: int = 30,
    ) -> None:
        self.dense = dense
        self.sparse = sparse
        self.reranker = reranker
        self.context_builder = context_builder
        self.trace_repository = trace_repository
        self.provider_metadata = provider_metadata or {}
        self.rerank_input_k = rerank_input_k
        self.rerank_output_k = rerank_output_k

    def retrieve(
        self,
        query: str,
        retrieval_filter: RetrievalFilter | None = None,
        *,
        recall_k: int = 20,
        top_k: int = 5,
        retrieval_scope: str = "unspecified",
    ) -> HybridRetrievalResult:
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=2) as executor:
            dense_future = executor.submit(
                self.dense.retrieve,
                query,
                retrieval_filter=retrieval_filter,
                top_k=recall_k,
            )
            sparse_future = executor.submit(
                self.sparse.retrieve,
                query,
                retrieval_filter=retrieval_filter,
                top_k=recall_k,
            )
            dense_results = dense_future.result()
            sparse_results = sparse_future.result()
        retrieval_finished = time.perf_counter()
        fused = reciprocal_rank_fusion(dense_results, sparse_results)
        candidate_limit = (
            recall_k if self.reranker.provider_name == "disabled" else self.rerank_input_k
        )
        rerank_candidates = fused[:candidate_limit]
        try:
            if self.reranker.provider_name == "disabled":
                outcome = RerankOutcome(
                    results=rerank_candidates,
                    provider="disabled",
                    model="none",
                    input_count=len(rerank_candidates),
                    output_count=len(rerank_candidates),
                    latency_ms=0.0,
                )
            else:
                rerank_top_n = min(self.rerank_output_k, len(rerank_candidates))
                outcome = self.reranker.rerank_with_trace(
                    query, rerank_candidates, rerank_top_n
                )
        except RerankerProviderError as exc:
            rerank_finished = time.perf_counter()
            trace = self._trace(
                query=query,
                retrieval_filter=retrieval_filter,
                dense_results=dense_results,
                sparse_results=sparse_results,
                fused=fused,
                rerank_candidates=rerank_candidates,
                outcome=None,
                final_context=[],
                started=started,
                retrieval_finished=retrieval_finished,
                rerank_finished=rerank_finished,
                context_finished=rerank_finished,
                retrieval_scope=retrieval_scope,
                failure_reason=str(exc),
                api_request_count=exc.api_request_count,
            )
            if self.trace_repository:
                self.trace_repository.append(trace)
            raise
        rerank_finished = time.perf_counter()
        context = self.context_builder.build(outcome.results[:top_k])
        context_finished = time.perf_counter()
        trace = self._trace(
            query=query,
            retrieval_filter=retrieval_filter,
            dense_results=self._plain_trace(dense_results),
            sparse_results=sparse_results,
            fused=fused,
            rerank_candidates=rerank_candidates,
            outcome=outcome,
            final_context=context,
            started=started,
            retrieval_finished=retrieval_finished,
            rerank_finished=rerank_finished,
            context_finished=context_finished,
            retrieval_scope=retrieval_scope,
        )
        if self.trace_repository:
            self.trace_repository.append(trace)
        return HybridRetrievalResult(context, trace)

    def _trace(
        self,
        *,
        query: str,
        retrieval_filter: RetrievalFilter | None,
        dense_results: list[RetrievalResult] | list[TraceResult],
        sparse_results: list[RetrievalResult],
        fused: list[FusedResult],
        rerank_candidates: list[FusedResult],
        outcome: RerankOutcome | None,
        final_context: list[ContextItem],
        started: float,
        retrieval_finished: float,
        rerank_finished: float,
        context_finished: float,
        retrieval_scope: str,
        failure_reason: str | None = None,
        api_request_count: int = 0,
    ) -> RetrievalTrace:
        dense_trace = (
            dense_results
            if not dense_results or isinstance(dense_results[0], TraceResult)
            else self._plain_trace(dense_results)
        )
        reranked = outcome.results if outcome else []
        return RetrievalTrace(
            query=query,
            filters=retrieval_filter.model_dump(exclude_none=True) if retrieval_filter else {},
            dense_results=dense_trace,  # type: ignore[arg-type]
            sparse_results=self._plain_trace(sparse_results),
            fusion_results=self._fused_trace(fused),
            rerank_results=self._fused_trace(reranked),
            rerank_candidates=self._rerank_trace(rerank_candidates, reranked),
            final_context=final_context,
            latency_ms=round((context_finished - started) * 1000, 3),
            retrieval_latency_ms=round((retrieval_finished - started) * 1000, 3),
            rerank_latency_ms=(
                outcome.latency_ms
                if outcome
                else round((rerank_finished - retrieval_finished) * 1000, 3)
            ),
            context_build_latency_ms=round((context_finished - rerank_finished) * 1000, 3),
            pre_rerank_candidate_count=len(rerank_candidates),
            rerank_output_count=len(reranked),
            rerank_fallback_occurred=outcome.fallback_occurred if outcome else False,
            rerank_failure_reason=(outcome.failure_reason if outcome else failure_reason),
            rerank_api_request_count=(outcome.api_request_count if outcome else api_request_count),
            retrieval_scope=retrieval_scope,
            **self.provider_metadata,
        )

    @staticmethod
    def _rerank_trace(
        candidates: list[FusedResult], reranked: list[FusedResult]
    ) -> list[RerankTraceCandidate]:
        post = {
            item.chunk.chunk_id: (rank, item.score)
            for rank, item in enumerate(reranked, start=1)
        }
        return [
            RerankTraceCandidate(
                chunk_id=item.chunk.chunk_id,
                paper_id=item.chunk.paper_id,
                pre_rerank_rank=rank,
                pre_rerank_score=item.score,
                post_rerank_rank=post.get(item.chunk.chunk_id, (None, None))[0],
                rerank_score=post.get(item.chunk.chunk_id, (None, None))[1],
                dense_rank=item.dense_rank,
                sparse_rank=item.sparse_rank,
            )
            for rank, item in enumerate(candidates, start=1)
        ]

    @staticmethod
    def _plain_trace(results: list[RetrievalResult]) -> list[TraceResult]:
        return [
            TraceResult(
                chunk_id=item.chunk.chunk_id,
                paper_id=item.chunk.paper_id,
                score=item.score,
                rank=rank,
            )
            for rank, item in enumerate(results, start=1)
        ]

    @staticmethod
    def _fused_trace(results: list[FusedResult]) -> list[TraceResult]:
        return [
            TraceResult(
                chunk_id=item.chunk.chunk_id,
                paper_id=item.chunk.paper_id,
                score=item.score,
                rank=rank,
                dense_rank=item.dense_rank,
                sparse_rank=item.sparse_rank,
            )
            for rank, item in enumerate(results, start=1)
        ]
