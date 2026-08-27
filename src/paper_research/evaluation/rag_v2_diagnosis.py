"""RAG v2 retrieval diagnosis: per-claim traces, failure taxonomy, metrics.

Offline, deterministic, label-driven. No provider calls.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

DIAGNOSIS_VERSION = "rag-v2-diagnosis-v1"
METRIC_NOT_COMPUTABLE = "METRIC_NOT_COMPUTABLE"

FAILURE_TAXONOMY = [
    "PARSING_FAILURE",
    "CHUNK_ALIGNMENT_FAILURE",
    "DENSE_RECALL_FAILURE",
    "SPARSE_RECALL_FAILURE",
    "CANDIDATE_POOL_FAILURE",
    "FUSION_RANKING_FAILURE",
    "RERANK_FAILURE",
    "CONTEXT_SELECTION_FAILURE",
    "CONTEXT_BUDGET_EVICTION",
    "CONTEXT_REDUNDANCY",
    "QUERY_UNDERSPECIFIED",
    "MULTI_HOP_QUERY_FAILURE",
    "GENERATION_FAILURE",
    "CITATION_SELECTION_FAILURE",
    "GOLD_ANNOTATION_MISMATCH",
    "UNKNOWN",
]


def recall_at(ranked: Sequence[str], gold: Iterable[str], k: int) -> float:
    gold_set = set(gold)
    if not gold_set:
        return 1.0
    top = list(ranked[:k])
    return len(set(top) & gold_set) / len(gold_set)


def any_hit_at(ranked: Sequence[str], gold: Iterable[str], k: int) -> bool:
    gold_set = set(gold)
    if not gold_set:
        return True
    return bool(set(ranked[:k]) & gold_set)


def reciprocal_rank(ranked: Sequence[str], gold: Iterable[str]) -> float:
    gold_set = set(gold)
    for rank, doc_id in enumerate(ranked, 1):
        if doc_id in gold_set:
            return 1.0 / rank
    return 0.0


def ndcg_at(ranked: Sequence[str], gold: Iterable[str], k: int) -> float:
    gold_set = set(gold)
    if not gold_set:
        return 1.0
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc_id in enumerate(ranked[:k], 1)
        if doc_id in gold_set
    )
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(gold_set), k) + 1))
    return dcg / ideal if ideal else 0.0


def precision_at(ranked: Sequence[str], gold: Iterable[str], k: int) -> float:
    top = list(ranked[:k])
    if not top:
        return 0.0
    gold_set = set(gold)
    return sum(doc_id in gold_set for doc_id in top) / len(top)


@dataclass
class StageResult:
    """One retrieval stage's ranked doc ids (deduped, in rank order)."""

    name: str
    ranked: list[str]
    scores: dict[str, float] = field(default_factory=dict)
    # True when this stage's ranks were reconstructed from gold relations
    # (missing runtime candidates) rather than measured from a real run.
    fallback: bool = False

    def rank_of(self, doc_id: str) -> int:
        try:
            return self.ranked.index(doc_id) + 1
        except ValueError:
            return 0


@dataclass
class ContextPackingResult:
    selected: list[str]
    excluded: list[tuple[str, str]]  # (doc_id, reason)
    token_count: int
    budget: int
    truncated_boundary_chunk: bool


def pack_context(
    ranked: Sequence[str],
    token_counts: dict[str, int],
    budget: int,
    top_k: int,
    chars_per_token: int = 4,
    text_lengths: dict[str, int] | None = None,
) -> ContextPackingResult:
    """Emulate production ContextBuilder: rank-order greedy prefix packing.

    Mirrors src/paper_research/retrieval/context_builder.py behavior:
    dedupe by id, stop entirely at the first chunk that overflows the budget.
    """

    text_lengths = text_lengths or {}
    selected: list[str] = []
    excluded: list[tuple[str, str]] = []
    seen: set[str] = set()
    used = 0
    truncated = False
    for doc_id in ranked:
        if len(selected) >= top_k:
            excluded.append((doc_id, "TOP_K_EXCEEDED"))
            continue
        if doc_id in seen:
            excluded.append((doc_id, "DUPLICATE"))
            continue
        seen.add(doc_id)
        tokens = token_counts.get(
            doc_id,
            max(1, (text_lengths.get(doc_id, 0) + chars_per_token - 1) // chars_per_token),
        )
        if used + tokens > budget:
            excluded.append((doc_id, "CONTEXT_BUDGET_EXCEEDED"))
            truncated = True
            # Production builder breaks the loop on first overflow.
            for rest in ranked[len(selected) + len(excluded) - 1 :]:
                if rest not in seen:
                    excluded.append((rest, "CONTEXT_BUDGET_EXCEEDED"))
            break
        selected.append(doc_id)
        used += tokens
    return ContextPackingResult(
        selected=selected,
        excluded=excluded,
        token_count=used,
        budget=budget,
        truncated_boundary_chunk=truncated,
    )


@dataclass
class ClaimTrace:
    benchmark_sample_id: str
    claim_text: str
    gold: set[str]
    dense: StageResult
    sparse: StageResult
    fusion: StageResult
    rerank: StageResult | None
    context: ContextPackingResult
    answer_correct: bool | None = None
    category: str = ""

    def to_record(self) -> dict[str, Any]:
        gold = self.gold
        union = set(self.dense.ranked) | set(self.sparse.ranked)
        return {
            "benchmark_sample_id": self.benchmark_sample_id,
            "query": self.claim_text,
            "category": self.category,
            "gold_doc_ids": sorted(gold),
            "dense_candidates": [
                {"doc_id": d, "rank": i + 1} for i, d in enumerate(self.dense.ranked)
            ],
            "dense_fallback": self.dense.fallback,
            "sparse_candidates": [
                {"doc_id": d, "rank": i + 1} for i, d in enumerate(self.sparse.ranked)
            ],
            "fusion_candidates": [
                {"doc_id": d, "rank": i + 1} for i, d in enumerate(self.fusion.ranked)
            ],
            "rerank_candidates": (
                [
                    {"doc_id": d, "rank": i + 1}
                    for i, d in enumerate(self.rerank.ranked)
                ]
                if self.rerank
                else None
            ),
            "selected_context_ids": self.context.selected,
            "excluded_context_ids": [
                {"doc_id": d, "reason": r} for d, r in self.context.excluded
            ],
            "context_token_count": self.context.token_count,
            "context_budget": self.context.budget,
            "context_boundary_truncated": self.context.truncated_boundary_chunk,
            "gold_in_dense_pool": bool(set(self.dense.ranked) & gold),
            "gold_in_sparse_pool": bool(set(self.sparse.ranked) & gold),
            "gold_in_union_pool": bool(union & gold),
            "gold_after_fusion": bool(set(self.fusion.ranked) & gold),
            "gold_after_rerank": (
                bool(set(self.rerank.ranked) & gold) if self.rerank else None
            ),
            "gold_in_final_context": bool(set(self.context.selected) & gold),
            "answer_correct": self.answer_correct,
            "failure_codes": classify_failure(self),
        }


def classify_failure(trace: ClaimTrace) -> list[str]:
    """Assign failure taxonomy codes for one claim (may be empty on success)."""

    gold = trace.gold
    if not gold:
        return ["GOLD_ANNOTATION_MISMATCH"]
    codes: list[str] = []
    in_dense = bool(set(trace.dense.ranked) & gold)
    in_sparse = bool(set(trace.sparse.ranked) & gold)
    union_hit = in_dense or in_sparse
    if not union_hit:
        if not in_dense:
            codes.append("DENSE_RECALL_FAILURE")
        if not in_sparse:
            codes.append("SPARSE_RECALL_FAILURE")
        codes.append("CANDIDATE_POOL_FAILURE")
        return codes
    final_retrieval = trace.rerank.ranked if trace.rerank else trace.fusion.ranked
    in_final = bool(set(final_retrieval[: len(final_retrieval)]) & gold)
    if not in_final:
        # Ranked outside the returned pool after fusion/rerank.
        codes.append("FUSION_RANKING_FAILURE")
        return codes
    if trace.rerank and not bool(set(trace.rerank.ranked) & gold):
        codes.append("RERANK_FAILURE")
        return codes
    in_context = bool(set(trace.context.selected) & gold)
    if not in_context:
        evicted = [reason for _, reason in trace.context.excluded]
        if "CONTEXT_BUDGET_EXCEEDED" in evicted:
            codes.append("CONTEXT_BUDGET_EVICTION")
        codes.append("CONTEXT_SELECTION_FAILURE")
        return codes
    if trace.answer_correct is False:
        codes.append("GENERATION_FAILURE")
    return codes


FROZEN_METRIC_SEMANTICS = {
    "claim_hit_rate@k": "claims (denominator = claims) with >=1 gold doc in top-k",
    "gold_block_recall@k": "mean |top-k ∩ gold| / |gold| (denominator = gold blocks/claim)",
    "MRR": "mean over claims of 1/rank(first gold doc)",
    "NDCG@k": "mean over claims of binary-gain NDCG over top-k (denominator = ideal DCG)",
    "context_precision": "mean |context ∩ gold| / |context| (denominator = context chunks)",
    "context_recall": "fraction of claims with >=1 gold doc in the final LLM context",
    "candidate_pool_recall": "claims with >=1 gold doc in the dense-sparse candidate pool",
}


def validate_metric_names(metrics: dict[str, Any]) -> list[str]:
    """Return violations of the frozen metric naming contract (RQ8 §1).

    A bare '<stage>_Recall@k' key is ambiguous (hit-rate vs block fraction)
    and must never appear in v2 reports.
    """

    violations = []
    for key in metrics:
        tail = key.split("_", 1)[-1] if "_" in key else key
        if tail.startswith("Recall@") or (
            tail.startswith("Recall") and "@" in tail
        ):
            violations.append(key)
    return violations


def summarize_traces(traces: Sequence[ClaimTrace]) -> dict[str, Any]:
    """Aggregate retrieval metrics across claims.

    Metrics beyond the available pool depth are reported as
    METRIC_NOT_COMPUTABLE rather than extrapolated.
    """

    records = [t.to_record() for t in traces]
    total = max(len(traces), 1)
    metrics: dict[str, Any] = {}
    stage_ranked = {
        "dense": [t.dense.ranked for t in traces],
        "sparse": [t.sparse.ranked for t in traces],
        "fusion": [t.fusion.ranked for t in traces],
        "context": [t.context.selected for t in traces],
    }
    for stage, ranked_lists in stage_ranked.items():
        golds = [t.gold for t in traces]
        # Computability is per stage: a metric at depth k is only truthful if
        # that stage's pool actually reaches depth k.
        stage_depth = max((len(r) for r in ranked_lists), default=0)
        for k in [1, 3, 5, 10, 20, 50]:
            # Metric names are frozen (RQ8 §1): bare "Recall@k" is banned.
            # claim_hit_rate@k: fraction of claims with any gold in top-k.
            # gold_block_recall@k: mean fraction of gold blocks recalled in top-k.
            hit_key = f"{stage}_claim_hit_rate@{k}"
            block_key = f"{stage}_gold_block_recall@{k}"
            if k > stage_depth:
                metrics[hit_key] = METRIC_NOT_COMPUTABLE
                metrics[block_key] = METRIC_NOT_COMPUTABLE
            else:
                metrics[hit_key] = sum(
                    any_hit_at(r, g, k)
                    for r, g in zip(ranked_lists, golds, strict=True)
                ) / total
                metrics[block_key] = sum(
                    recall_at(r, g, k)
                    for r, g in zip(ranked_lists, golds, strict=True)
                ) / total
        metrics[f"{stage}_MRR"] = sum(
            reciprocal_rank(ranked, gold)
            for ranked, gold in zip(ranked_lists, golds, strict=True)
        ) / total
        for k in [5, 10]:
            key = f"{stage}_NDCG@{k}"
            if k > stage_depth:
                metrics[key] = METRIC_NOT_COMPUTABLE
            else:
                metrics[key] = sum(
                    ndcg_at(r, g, k) for r, g in zip(ranked_lists, golds, strict=True)
                ) / total
    metrics["context_precision"] = sum(
        precision_at(t.context.selected, t.gold, len(t.context.selected)) for t in traces
    ) / total
    metrics["candidate_pool_recall"] = sum(
        bool((set(t.dense.ranked) | set(t.sparse.ranked)) & t.gold) for t in traces
    ) / total
    metrics["context_recall"] = sum(
        bool(set(t.context.selected) & t.gold) for t in traces
    ) / total
    metrics["mean_candidates_per_query"] = sum(
        len(t.fusion.ranked) for t in traces
    ) / total
    metrics["mean_context_tokens_per_query"] = sum(
        t.context.token_count for t in traces
    ) / total
    metrics["mean_context_chunks_per_query"] = sum(
        len(t.context.selected) for t in traces
    ) / total

    failure_distribution: dict[str, int] = {code: 0 for code in FAILURE_TAXONOMY}
    for record in records:
        for code in record["failure_codes"]:
            failure_distribution[code] = failure_distribution.get(code, 0) + 1
    # Bucket summary for the report.
    buckets = {
        "candidate_recall": sum(
            1
            for r in records
            if "CANDIDATE_POOL_FAILURE" in r["failure_codes"]
        ),
        "fusion_ranking": sum(
            1 for r in records if "FUSION_RANKING_FAILURE" in r["failure_codes"]
        ),
        "rerank": sum(1 for r in records if "RERANK_FAILURE" in r["failure_codes"]),
        "context_selection": sum(
            1
            for r in records
            if "CONTEXT_SELECTION_FAILURE" in r["failure_codes"]
            or "CONTEXT_BUDGET_EVICTION" in r["failure_codes"]
        ),
        "generation": sum(
            1 for r in records if "GENERATION_FAILURE" in r["failure_codes"]
        ),
        "annotation": sum(
            1 for r in records if "GOLD_ANNOTATION_MISMATCH" in r["failure_codes"]
        ),
    }
    return {
        "diagnosis_version": DIAGNOSIS_VERSION,
        "sample_count": len(traces),
        "metrics": metrics,
        "failure_distribution": failure_distribution,
        "failure_buckets": buckets,
    }
