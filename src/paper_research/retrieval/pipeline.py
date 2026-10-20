"""Opt-in Retrieval v2; HybridRetriever remains the frozen legacy path."""

import re
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

from paper_research.chunking.types import Chunk
from paper_research.config import Settings
from paper_research.indexing.identity import assert_runtime_compatible
from paper_research.indexing.registry import IndexRegistry
from paper_research.retrieval.candidate import RetrievalCandidate
from paper_research.retrieval.context_builder import ContextBuilder
from paper_research.retrieval.contextual import CONTEXTUAL_VERSION, contextualize_corpus
from paper_research.retrieval.dense import DenseRetriever
from paper_research.retrieval.filters import RetrievalFilter
from paper_research.retrieval.fusion import reciprocal_rank_fusion
from paper_research.retrieval.hybrid import HybridRetrievalResult, HybridRetriever
from paper_research.retrieval.reranker import Reranker, RerankerProviderError, RerankOutcome
from paper_research.retrieval.sparse import BM25Retriever
from paper_research.retrieval.trace import JsonlTraceRepository


class RankStage(Protocol):
    """Ordered optional stages: late interaction can precede the model adapter.

    TODO: add a measured late-interaction adapter when a suitable index exists.
    Stages must preserve IDs, raw text and provenance, and return the whole pool.
    """

    name: str

    def rank(
        self,
        query: str,
        candidates: list[RetrievalCandidate],
    ) -> list[RetrievalCandidate]: ...


def selective_decision(
    query: str,
    candidates: list[RetrievalCandidate],
    *,
    margin_threshold: float = 0.02,
    disagreement_threshold: float = 0.5,
) -> tuple[bool, str]:
    if len(candidates) < 2:
        return False, "insufficient_candidates"
    # Explicit multiple conditions/comparisons, not a gold- or identity-based rule.
    if re.search(r"\b(and|versus|vs|compare|comparison|whereas)\b|以及|比较|并且", query.lower()):
        return True, "multi_condition_query"
    dense = {c.chunk.chunk_id for c in candidates if c.dense_rank and c.dense_rank <= 5}
    sparse = {c.chunk.chunk_id for c in candidates if c.sparse_rank and c.sparse_rank <= 5}
    if dense and sparse:
        disagreement = 1 - len(dense & sparse) / len(dense | sparse)
        if disagreement > disagreement_threshold:
            return True, "dense_sparse_disagreement"
    # Relative RRF margin: absolute cosine/BM25 scales must not be mixed here.
    first, second = sorted((c.fusion_score for c in candidates), reverse=True)[:2]
    margin = (first - second) / max(abs(first), 1e-12)
    if margin <= margin_threshold:
        return True, "weak_fusion_margin"
    return False, "well_separated_candidates"


class RetrievalPipeline(HybridRetriever):
    def __init__(
        self,
        dense: DenseRetriever,
        sparse: BM25Retriever,
        reranker: Reranker,
        context_builder: ContextBuilder,
        trace_repository: JsonlTraceRepository | None = None,
        provider_metadata: dict[str, object] | None = None,
        *,
        candidate_top_k: int = 30,
        final_top_k: int = 5,
        rerank_mode: str = "off",
        rank_stages: tuple[RankStage, ...] = (),
        margin_threshold: float = 0.02,
        disagreement_threshold: float = 0.5,
        contextual_required: bool = False,
        index_manifest: dict | None = None,
        selective_policy: Callable[[str, list[RetrievalCandidate]], tuple[bool, str]] | None = None,
    ) -> None:
        if index_manifest is not None:
            assert_runtime_compatible(
                index_manifest, dense.embedding, collection=dense.vector_store.collection
            )
        if not 1 <= final_top_k <= candidate_top_k <= 100:
            raise ValueError("require 1 <= final_top_k <= candidate_top_k <= 100")
        if rerank_mode not in {"off", "always", "selective"}:
            raise ValueError("rerank_mode must be off, always or selective")
        if rerank_mode != "off" and reranker.provider_name == "disabled":
            raise ValueError("active v2 reranking requires a model/ablation reranker")
        if not 0 <= margin_threshold <= 1 or not 0 <= disagreement_threshold <= 1:
            raise ValueError("selective thresholds must be in [0, 1]")
        if len({stage.name for stage in rank_stages}) != len(rank_stages):
            raise ValueError("ranking stage names must be unique")
        super().__init__(
            dense,
            sparse,
            reranker,
            context_builder,
            trace_repository,
            provider_metadata,
            candidate_top_k,
            candidate_top_k,
        )
        self.candidate_top_k = candidate_top_k
        self.final_top_k = final_top_k
        self.rerank_mode = rerank_mode
        self.rank_stages = rank_stages
        self.margin_threshold = margin_threshold
        self.disagreement_threshold = disagreement_threshold
        self.contextual_required = contextual_required
        self.selective_policy = selective_policy

    def retrieve(
        self,
        query: str,
        retrieval_filter: RetrievalFilter | None = None,
        *,
        recall_k: int | None = None,
        top_k: int | None = None,
        retrieval_scope: str = "unspecified",
    ) -> HybridRetrievalResult:
        started = time.perf_counter()
        final_k = self.final_top_k if top_k is None else top_k
        pool_k = max(self.candidate_top_k, recall_k or 0, final_k)
        if not 1 <= final_k <= pool_k <= 100:
            raise ValueError("retrieval depth must be between 1 and 100")
        routed, signals = self._route_query(
            query,
            retrieval_scope=retrieval_scope,
            experiment_design_query=self._is_experiment_design_query(query),
        )

        def timed_retrieve(retriever):
            begin = time.perf_counter()
            results = retriever.retrieve(routed, retrieval_filter=retrieval_filter, top_k=pool_k)
            return results, (time.perf_counter() - begin) * 1000

        with ThreadPoolExecutor(max_workers=2) as executor:
            df = executor.submit(timed_retrieve, self.dense)
            sf = executor.submit(timed_retrieve, self.sparse)
            dense, dense_ms = df.result()
            sparse, sparse_ms = sf.result()
        if self.contextual_required and any(
            not getattr(item.chunk, "contextual_text", "") for item in dense
        ):
            raise ValueError("v2 contextual collection contains raw payloads; rebuild a new index")
        begin = time.perf_counter()
        fused = reciprocal_rank_fusion(dense, sparse)
        dense_scores = {r.chunk.chunk_id: r.score for r in dense}
        sparse_scores = {r.chunk.chunk_id: r.score for r in sparse}
        candidates = [
            RetrievalCandidate.from_fused(r, dense_scores, sparse_scores) for r in fused[:pool_k]
        ]
        fusion_ms = (time.perf_counter() - begin) * 1000
        retrieval_finished = time.perf_counter()
        original_candidates = [c.as_fused() for c in candidates]
        triggered, reason = False, "mode_off"
        if self.rerank_mode == "always":
            triggered, reason = bool(candidates), "mode_always" if candidates else "empty_pool"
        elif self.rerank_mode == "selective":
            if self.selective_policy is not None:
                triggered, reason = self.selective_policy(query, candidates)
            else:
                triggered, reason = selective_decision(
                    query,
                    candidates,
                    margin_threshold=self.margin_threshold,
                    disagreement_threshold=self.disagreement_threshold,
                )
        timings = {"dense": dense_ms, "sparse": sparse_ms, "fusion": fusion_ms}
        outcome = None
        failure = None
        try:
            for stage in self.rank_stages:
                begin = time.perf_counter()
                before = {c.chunk.chunk_id: c.model_dump() for c in candidates}
                ranked = stage.rank(query, candidates)
                ids = [c.chunk.chunk_id for c in ranked]
                if len(ids) != len(before) or set(ids) != set(before):
                    raise ValueError("rank stage must return every candidate exactly once")
                for candidate in ranked:
                    source = before[candidate.chunk.chunk_id]
                    if (
                        candidate.chunk.model_dump() != source["chunk"]
                        or candidate.raw_text != source["raw_text"]
                        or candidate.contextual_text != source["contextual_text"]
                        or candidate.metadata != source["metadata"]
                        or any(
                            getattr(candidate, name) != source[name]
                            for name in (
                                "dense_score",
                                "sparse_score",
                                "fusion_score",
                                "dense_rank",
                                "sparse_rank",
                            )
                        )
                    ):
                        raise ValueError("rank stage changed candidate provenance")
                candidates = ranked
                timings[stage.name] = (time.perf_counter() - begin) * 1000
            if triggered:
                outcome = self.reranker.rerank_with_trace(
                    query, [c.as_fused() for c in candidates], len(candidates)
                )
                scores = {r.chunk.chunk_id: r.score for r in outcome.results}
                if len(scores) != len(candidates) or set(scores) != {
                    c.chunk.chunk_id for c in candidates
                }:
                    raise RerankerProviderError("reranker changed the candidate identity set")
                by_id = {c.chunk.chunk_id: c for c in candidates}
                candidates = [
                    by_id[r.chunk.chunk_id].model_copy(
                        update={
                            "rerank_score": r.score,
                            "ranking_scores": {
                                **by_id[r.chunk.chunk_id].ranking_scores,
                                "reranker": r.score,
                            },
                            "retrieval_sources": [
                                *by_id[r.chunk.chunk_id].retrieval_sources,
                                "reranked",
                            ],
                        }
                    )
                    for r in outcome.results
                ]
        except (RerankerProviderError, ValueError) as exc:
            failure = exc
        rerank_finished = time.perf_counter()
        ranked_fused = [c.as_fused() for c in candidates]
        if outcome is None and failure is None:
            outcome = RerankOutcome(
                ranked_fused, "disabled", "none", len(candidates), len(candidates), 0.0
            )
        context = (
            []
            if failure
            else self.context_builder.build(
                self._context_candidates(
                    query, ranked_fused, top_k=final_k, retrieval_scope=retrieval_scope
                )
            )
        )
        finished = time.perf_counter()
        trace = self._trace(
            query=query,
            routed_query=routed,
            query_routing_signals=signals,
            retrieval_filter=retrieval_filter,
            dense_results=dense,
            sparse_results=sparse,
            fused=fused,
            rerank_candidates=original_candidates,
            outcome=outcome,
            final_context=context,
            started=started,
            retrieval_finished=retrieval_finished,
            rerank_finished=rerank_finished,
            context_finished=finished,
            retrieval_scope=retrieval_scope,
            failure_reason=str(failure) if failure else None,
            api_request_count=getattr(failure, "api_request_count", 0),
        )
        timings.update(rerank=trace.rerank_latency_ms, total=trace.latency_ms)
        trace.retrieval_mode = f"v2_{self.rerank_mode}"
        trace.contextual_retrieval = self.contextual_required
        trace.contextual_sparse = self.sparse.contextual
        trace.contextual_version = CONTEXTUAL_VERSION if self.contextual_required else None
        trace.rerank_triggered, trace.rerank_reason = triggered, reason
        trace.candidate_top_k, trace.final_top_k = pool_k, final_k
        trace.dense_candidate_count, trace.sparse_candidate_count = len(dense), len(sparse)
        trace.fusion_candidate_count = len(fused)
        trace.stage_latency_ms = {name: round(ms, 3) for name, ms in timings.items()}
        trace.candidate_scores = [
            {
                "chunk_id": c.chunk.chunk_id,
                "paper_id": c.chunk.paper_id,
                "dense_score": c.dense_score,
                "sparse_score": c.sparse_score,
                "fusion_score": c.fusion_score,
                "rerank_score": c.rerank_score,
                "ranking_scores": c.ranking_scores,
                "retrieval_sources": c.retrieval_sources,
            }
            for c in candidates
        ]
        trace.rerank_provider = outcome.provider if outcome else self.reranker.provider_name
        trace.rerank_model = outcome.model if outcome else self.reranker.model_name
        if not triggered:
            for item in trace.rerank_candidates:
                item.rerank_score = None
        if self.trace_repository:
            self.trace_repository.append(trace)
        if failure:
            raise failure
        return HybridRetrievalResult(context, trace)


def build_v2_retriever(
    settings: Settings,
    dense: DenseRetriever,
    chunks: list[Chunk],
    reranker: Reranker,
) -> RetrievalPipeline:
    if settings.retrieval_v2_contextual_enabled:
        chunks = contextualize_corpus(chunks, settings.parsed_papers_dir)
    return RetrievalPipeline(
        dense,
        BM25Retriever(chunks, contextual=settings.retrieval_v2_contextual_sparse),
        reranker,
        ContextBuilder(
            include_neighbors=False,
            max_characters=settings.qa_context_token_budget * 4,
            max_tokens=settings.qa_context_token_budget,
        ),
        JsonlTraceRepository(settings.retrieval_trace_path),
        provider_metadata={
            **settings.provider_metadata,
            "collection": dense.vector_store.collection,
        },
        candidate_top_k=settings.retrieval_v2_candidate_top_k,
        final_top_k=settings.retrieval_v2_final_top_k,
        rerank_mode=settings.retrieval_v2_rerank_mode,
        margin_threshold=settings.retrieval_v2_margin_threshold,
        disagreement_threshold=settings.retrieval_v2_disagreement_threshold,
        contextual_required=settings.retrieval_v2_contextual_enabled,
    )


def retrieval_collection(settings: Settings, registry: IndexRegistry) -> str:
    baseline = registry.resolve(settings.active_collection)
    if not (settings.retrieval_v2_enabled and settings.retrieval_v2_contextual_enabled):
        return baseline
    contextual = registry.resolve(settings.retrieval_v2_collection or "")
    if contextual == baseline:
        raise ValueError("contextual collection resolves to the baseline index")
    return contextual
