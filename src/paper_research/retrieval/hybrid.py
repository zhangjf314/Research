import time
from concurrent.futures import ThreadPoolExecutor

from paper_research.retrieval.context_builder import ContextBuilder, ContextItem
from paper_research.retrieval.dense import DenseRetriever, RetrievalResult
from paper_research.retrieval.filters import RetrievalFilter
from paper_research.retrieval.fusion import FusedResult, reciprocal_rank_fusion
from paper_research.retrieval.reranker import Reranker
from paper_research.retrieval.sparse import BM25Retriever
from paper_research.retrieval.trace import JsonlTraceRepository, RetrievalTrace, TraceResult


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
    ) -> None:
        self.dense = dense
        self.sparse = sparse
        self.reranker = reranker
        self.context_builder = context_builder
        self.trace_repository = trace_repository
        self.provider_metadata = provider_metadata or {}

    def retrieve(
        self,
        query: str,
        retrieval_filter: RetrievalFilter | None = None,
        *,
        recall_k: int = 20,
        top_k: int = 5,
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
        rerank_started = time.perf_counter()
        reranked = self.reranker.rerank(query, fused, top_k)
        rerank_finished = time.perf_counter()
        context = self.context_builder.build(reranked)
        context_finished = time.perf_counter()
        trace = RetrievalTrace(
            query=query,
            filters=retrieval_filter.model_dump(exclude_none=True) if retrieval_filter else {},
            dense_results=self._plain_trace(dense_results),
            sparse_results=self._plain_trace(sparse_results),
            fusion_results=self._fused_trace(fused),
            rerank_results=self._fused_trace(reranked),
            final_context=context,
            latency_ms=round((context_finished - started) * 1000, 3),
            retrieval_latency_ms=round((retrieval_finished - started) * 1000, 3),
            rerank_latency_ms=round((rerank_finished - rerank_started) * 1000, 3),
            context_build_latency_ms=round((context_finished - rerank_finished) * 1000, 3),
            **self.provider_metadata,
        )
        if self.trace_repository:
            self.trace_repository.append(trace)
        return HybridRetrievalResult(context, trace)

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
