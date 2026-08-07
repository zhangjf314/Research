from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Protocol

from pydantic import BaseModel, Field

from paper_research.chunking.types import Chunk
from paper_research.evaluation.rag_benchmark import read_jsonl
from paper_research.evaluation.rag_official_baseline import (
    DEV_DATASET_HASH,
    aggregate_retrieval_rows,
    retrieval_bad_cases,
    retrieval_row,
)
from paper_research.indexing.embedding import EmbeddingProvider
from paper_research.retrieval.dense import RetrievalResult
from paper_research.retrieval.fusion import FusedResult

OPT_ROOT = Path("data/evaluation/rag-optimization")
OPT_DOCS = Path("docs/rag-optimization")
RAG_ROOT = Path("data/evaluation/rag-benchmark")
BOOTSTRAP_SEED = 20260807


class QueryEmbeddingProvider(Protocol):
    def embed_query(self, text: str) -> list[float]: ...


class CachedQueryEmbeddingProvider(EmbeddingProvider):
    """In-process query embedding cache for DEV-only ablations."""

    def __init__(self, wrapped: EmbeddingProvider) -> None:
        self.wrapped = wrapped
        self.dimensions = wrapped.dimensions
        self.provider_name = wrapped.provider_name
        self.model_name = wrapped.model_name
        self.revision = wrapped.revision
        self.query_cache: dict[str, list[float]] = {}

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.wrapped.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        if text not in self.query_cache:
            self.query_cache[text] = self.wrapped.embed_query(text)
        return self.query_cache[text]


class RetrievalExperimentConfig(BaseModel):
    experiment_id: str
    mode: str
    dense_weight: float = 1.0
    sparse_weight: float = 1.0
    rrf_k: int = 60
    recall_k: int = 20
    final_k: int = 20
    reranker: str = "none"
    reranker_candidate_k: int | None = None
    reranker_output_k: int | None = None
    frozen_baseline_equivalent: bool = False


class RetrievalExperimentPlan(BaseModel):
    schema_version: str = "stage2a-hybrid-experiment-plan-v1"
    dataset_version: str = "rag-gold-v1"
    dataset_hash: str = DEV_DATASET_HASH
    split: str = "dev"
    test_questions_allowed: bool = False
    bootstrap_seed: int = BOOTSTRAP_SEED
    bootstrap_resamples: int = 1000
    decision_thresholds: dict[str, float] = Field(
        default_factory=lambda: {
            "recall_at_10_absolute_gain": 0.05,
            "evidence_coverage_at_10_absolute_gain": 0.05,
            "p95_latency_max_baseline_multiplier": 2.5,
        }
    )
    experiments: list[RetrievalExperimentConfig]


@dataclass(frozen=True)
class RankedCandidate:
    chunk: Chunk
    score: float
    dense_rank: int | None = None
    sparse_rank: int | None = None


def load_dev_gold() -> list[dict[str, Any]]:
    rows = read_jsonl(RAG_ROOT / "gold-dev-v1.jsonl")
    non_dev = [row["question_id"] for row in rows if row.get("split") != "dev"]
    if non_dev:
        raise ValueError(f"TEST_PROTOCOL_VIOLATION: non-dev rows in dev file: {non_dev[:5]}")
    return rows


def load_baseline_retrieval_dev_rows() -> list[dict[str, Any]]:
    rows = read_jsonl(RAG_ROOT / "retrieval-baseline-items-v1.jsonl")
    dev = [row for row in rows if row.get("split") == "dev"]
    if len(dev) != 98:
        raise ValueError(f"expected 98 DEV retrieval rows, found {len(dev)}")
    return dev


def load_baseline_generation_rows() -> list[dict[str, Any]]:
    return read_jsonl(RAG_ROOT / "generation-baseline-items-v1.jsonl")


def context_from_candidates(candidates: list[RankedCandidate]) -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": candidate.chunk.chunk_id,
            "paper_id": candidate.chunk.paper_id,
            "block_ids": list(candidate.chunk.block_ids),
            "section_path": list(candidate.chunk.section_path),
            "page_start": candidate.chunk.page_start,
            "page_end": candidate.chunk.page_end,
            "evidence": candidate.chunk.chunk_text,
            "score": candidate.score,
            "dense_rank": candidate.dense_rank,
            "sparse_rank": candidate.sparse_rank,
        }
        for candidate in candidates
    ]


def as_ranked_from_retrieval(results: list[RetrievalResult], source: str) -> list[RankedCandidate]:
    return [
        RankedCandidate(
            chunk=item.chunk,
            score=item.score,
            dense_rank=rank if source == "dense" else None,
            sparse_rank=rank if source == "sparse" else None,
        )
        for rank, item in enumerate(results, start=1)
    ]


def as_ranked_from_fused(results: list[FusedResult]) -> list[RankedCandidate]:
    return [
        RankedCandidate(
            chunk=item.chunk,
            score=item.score,
            dense_rank=item.dense_rank,
            sparse_rank=item.sparse_rank,
        )
        for item in results
    ]


def evaluate_ranked_candidates(
    gold: dict[str, Any],
    candidates: list[RankedCandidate],
    *,
    latency_ms: float,
    paper_id_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    return retrieval_row(
        gold,
        context_from_candidates(candidates),
        latency_ms=latency_ms,
        paper_id_map=paper_id_map,
    )


def retrieval_headroom(rows: list[dict[str, Any]]) -> dict[str, Any]:
    answerable = [row for row in rows if row.get("answerable")]
    gold_in_top5 = sum(1 for row in answerable if row["metrics"].get("recall_at_5", 0) > 0)
    gold_in_top10 = sum(1 for row in answerable if row["metrics"].get("recall_at_10", 0) > 0)
    gold_in_top20 = sum(1 for row in answerable if row["metrics"].get("recall_at_20", 0) > 0)
    full10 = sum(
        1 for row in answerable if row["metrics"].get("evidence_coverage_at_10", 0) >= 1
    )
    full20 = sum(
        1 for row in answerable if row["metrics"].get("evidence_coverage_at_20", 0) >= 1
    )
    partial10 = sum(
        1
        for row in answerable
        if 0 < row["metrics"].get("evidence_coverage_at_10", 0) < 1
    )
    partial20 = sum(
        1
        for row in answerable
        if 0 < row["metrics"].get("evidence_coverage_at_20", 0) < 1
    )
    same_paper_wrong_block = sum(
        1
        for row in answerable
        if row["metrics"].get("paper_recall_at_10", 0) > 0
        and row["metrics"].get("evidence_coverage_at_10", 0) == 0
    )
    count = len(answerable) or 1
    return {
        "answerable_count": len(answerable),
        "gold_in_top5": gold_in_top5,
        "gold_in_top10": gold_in_top10,
        "gold_in_top20": gold_in_top20,
        "top20_but_not_top10": gold_in_top20 - gold_in_top10,
        "top20_but_not_top5": gold_in_top20 - gold_in_top5,
        "correct_paper_top10_no_correct_evidence_top10": same_paper_wrong_block,
        "same_paper_wrong_block_rate": round(same_paper_wrong_block / count, 6),
        "top20_to_top10_rerank_headroom": round((gold_in_top20 - gold_in_top10) / count, 6),
        "top10_to_top5_rerank_headroom": round((gold_in_top10 - gold_in_top5) / count, 6),
        "partial_gold_evidence_in_top10": partial10,
        "full_gold_evidence_in_top10": full10,
        "partial_gold_evidence_in_top20": partial20,
        "full_gold_evidence_in_top20": full20,
        "evidence_full_coverage_rate_at_10": round(full10 / count, 6),
        "evidence_full_coverage_rate_at_20": round(full20 / count, 6),
        "required_claim_evidence_coverage_at_5": _required_claim_coverage(rows, 5),
        "required_claim_evidence_coverage_at_10": _required_claim_coverage(rows, 10),
        "required_claim_evidence_coverage_at_20": _required_claim_coverage(rows, 20),
    }


def _required_claim_coverage(rows: list[dict[str, Any]], k: int) -> float:
    total = 0
    covered = 0
    gold_by_id = {row["question_id"]: row for row in load_dev_gold()}
    for row in rows:
        if not row.get("answerable"):
            continue
        gold = gold_by_id.get(row["question_id"])
        if gold is None:
            continue
        retrieved_blocks = {
            block
            for result in row.get("ranked_results", [])[:k]
            for block in result.get("block_ids", [])
        }
        for claim in gold.get("required_claims", []):
            claim_blocks = set(claim.get("gold_block_ids") or gold.get("gold_block_ids") or [])
            if not claim_blocks:
                continue
            total += 1
            if retrieved_blocks & claim_blocks:
                covered += 1
    return round(covered / total, 6) if total else 0.0


def grouped_headroom(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(field))].append(row)
    return {key: retrieval_headroom(value) for key, value in sorted(groups.items())}


def complementarity(
    dense_rows: list[dict[str, Any]],
    sparse_rows: list[dict[str, Any]],
    hybrid_rows: list[dict[str, Any]],
    *,
    metric: str = "recall_at_10",
) -> dict[str, Any]:
    dense = {row["question_id"]: row for row in dense_rows if row.get("answerable")}
    sparse = {row["question_id"]: row for row in sparse_rows if row.get("answerable")}
    hybrid = {row["question_id"]: row for row in hybrid_rows if row.get("answerable")}
    items = []
    counts: Counter[str] = Counter()
    for qid, dense_row in sorted(dense.items()):
        dense_hit = (dense_row["metrics"].get(metric) or 0) > 0
        sparse_hit = (sparse[qid]["metrics"].get(metric) or 0) > 0
        hybrid_hit = (hybrid[qid]["metrics"].get(metric) or 0) > 0
        if dense_hit and sparse_hit:
            bucket = "both_success"
        elif dense_hit:
            bucket = "dense_only_success"
        elif sparse_hit:
            bucket = "sparse_only_success"
        else:
            bucket = "both_failure"
        counts[bucket] += 1
        if hybrid_hit and not dense_hit:
            counts["hybrid_recovers_dense_failure"] += 1
        if hybrid_hit and not sparse_hit:
            counts["hybrid_recovers_sparse_failure"] += 1
        items.append(
            {
                "question_id": qid,
                "category": dense_row.get("category"),
                "difficulty": dense_row.get("difficulty"),
                "dense_hit": dense_hit,
                "sparse_hit": sparse_hit,
                "hybrid_hit": hybrid_hit,
                "bucket": bucket,
            }
        )
    return {"summary": dict(sorted(counts.items())), "items": items}


def paired_comparison(
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    *,
    metric: str = "recall_at_10",
) -> dict[str, Any]:
    baseline = {row["question_id"]: row for row in baseline_rows if row.get("answerable")}
    candidate = {row["question_id"]: row for row in candidate_rows if row.get("answerable")}
    counts = Counter()
    items = []
    for qid, base_row in sorted(baseline.items()):
        before = float(base_row["metrics"].get(metric) or 0)
        after = float(candidate[qid]["metrics"].get(metric) or 0)
        label = "tie"
        if after > before:
            label = "win"
        elif after < before:
            label = "loss"
        counts[label] += 1
        items.append({"question_id": qid, "before": before, "after": after, "outcome": label})
    return {
        "metric": metric,
        "win_count": counts["win"],
        "tie_count": counts["tie"],
        "loss_count": counts["loss"],
        "items": items,
    }


def bootstrap_delta_ci(
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    *,
    metric: str,
    resamples: int = 1000,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    baseline = {row["question_id"]: row for row in baseline_rows if row.get("answerable")}
    candidate = {row["question_id"]: row for row in candidate_rows if row.get("answerable")}
    qids = sorted(baseline)
    deltas = [
        float(candidate[qid]["metrics"].get(metric) or 0)
        - float(baseline[qid]["metrics"].get(metric) or 0)
        for qid in qids
    ]
    rng = random.Random(seed)
    samples = []
    for _ in range(resamples):
        draw = [deltas[rng.randrange(len(deltas))] for _ in deltas]
        samples.append(mean(draw))
    samples.sort()
    return {
        "metric": metric,
        "mean_delta": round(mean(deltas), 6) if deltas else 0.0,
        "ci95_low": round(samples[int(0.025 * (len(samples) - 1))], 6) if samples else 0.0,
        "ci95_high": round(samples[int(0.975 * (len(samples) - 1))], 6) if samples else 0.0,
        "resamples": resamples,
        "seed": seed,
    }


def bad_case_delta(
    baseline_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    before = retrieval_bad_cases(baseline_rows)["distribution"]
    after = retrieval_bad_cases(candidate_rows)["distribution"]
    keys = sorted(set(before) | set(after))
    return {
        "before": before,
        "after": after,
        "delta": {key: int(after.get(key, 0)) - int(before.get(key, 0)) for key in keys},
    }


def generation_metric_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    answerable = [row for row in rows if row.get("gold", {}).get("answerable")]
    citation_precision_num = 0
    citation_precision_den = 0
    citation_recall_num = 0
    citation_recall_den = 0
    status_counts: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    for row in answerable:
        gold = row["gold"]
        gold_blocks = set(gold.get("gold_block_ids", []))
        gold_papers = set(gold.get("gold_paper_ids", []))
        gold_pages = {int(page) for page in gold.get("gold_pages", [])}
        claims = row.get("answer", {}).get("claims", []) or []
        citations = [citation for claim in claims for citation in claim.get("citations", [])]
        correct = [
            citation
            for citation in citations
            if citation.get("block_id") in gold_blocks
            and citation.get("paper_id") in gold_papers
            and _safe_int(citation.get("page")) in gold_pages
        ]
        citation_precision_num += len(correct)
        citation_precision_den += len(citations)
        cited_gold_blocks = {citation.get("block_id") for citation in citations} & gold_blocks
        citation_recall_num += len(cited_gold_blocks)
        citation_recall_den += len(gold_blocks)
        if not citations:
            bucket = "no_citation"
        elif len(correct) == 0:
            bucket = "wrong_citation"
        elif len(correct) == len(citations):
            bucket = "all_citations_correct"
        else:
            bucket = "partially_correct_citation"
        status_counts[bucket] += 1
        if len(examples) < 10 or bucket not in {item["bucket"] for item in examples}:
            examples.append(
                {
                    "question_id": row["question_id"],
                    "bucket": bucket,
                    "gold_paper_ids": sorted(gold_papers),
                    "gold_block_ids": sorted(gold_blocks),
                    "gold_pages": sorted(gold_pages),
                    "generated_citations": citations[:5],
                    "correct_citation_count": len(correct),
                    "citation_count": len(citations),
                    "reported_metrics": row.get("generation_metrics", {}),
                }
            )
    precision = (
        citation_precision_num / citation_precision_den if citation_precision_den else 0.0
    )
    recall = citation_recall_num / citation_recall_den if citation_recall_den else 0.0
    return {
        "status": "METRICS_VALID",
        "answerable_questions": len(answerable),
        "citation_precision": {
            "numerator": citation_precision_num,
            "denominator": citation_precision_den,
            "value": round(precision, 6),
        },
        "citation_recall": {
            "numerator": citation_recall_num,
            "denominator": citation_recall_den,
            "value": round(recall, 6),
        },
        "representation": {
            "gold": "paper_id + page + block_id from gold_paper_ids/gold_pages/gold_block_ids",
            "generated": "claim.citations[].paper_id/page/block_id",
            "paper_id_normalization": "exact string comparison",
            "block_id_normalization": "exact string comparison",
            "page_normalization": "integer comparison",
        },
        "sampled_examples": examples[:12],
        "sample_bucket_counts": dict(sorted(status_counts.items())),
        "explanation": (
            "Precision can be 0 while recall is positive because precision requires an exact "
            "paper/page/block triple for every generated citation, while recall only counts "
            "whether any generated citation mentions a gold block ID. In the frozen results, "
            "some answers cite gold block IDs but fail exact page or paper matching."
        ),
    }


def _safe_int(value: Any) -> int | None:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def summarize_context_selection_from_generation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    for row in rows:
        if not row.get("gold", {}).get("answerable"):
            continue
        metrics = row.get("generation_metrics", {})
        retrieval = metrics.get("gold_evidence_in_context")
        success = (metrics.get("required_claim_coverage") or 0) >= 1
        if not retrieval:
            counts["A_insufficient_retrieved_or_final_context"] += 1
        elif not success:
            counts["C_sufficient_final_context_generation_omission"] += 1
        else:
            counts["success"] += 1
    total = sum(counts.values()) or 1
    return {
        "counts": dict(sorted(counts.items())),
        "proportions": {key: round(value / total, 6) for key, value in sorted(counts.items())},
        "context_selection_hypothesis_supported": counts.get(
            "B_sufficient_retrieved_but_insufficient_final_context", 0
        )
        > counts.get("C_sufficient_final_context_generation_omission", 0),
        "note": (
            "Stage 1C generation artifacts only expose whether gold evidence reached final "
            "context, not a separate retrieved-before-context-selection pool; therefore B is "
            "not directly observable from saved artifacts."
        ),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def metrics_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return aggregate_retrieval_rows(rows)


def category_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("category"))].append(row)
    return {key: aggregate_retrieval_rows(value) for key, value in sorted(groups.items())}


def difficulty_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("difficulty"))].append(row)
    return {key: aggregate_retrieval_rows(value) for key, value in sorted(groups.items())}


def is_success_candidate(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    p95_multiplier: float = 2.5,
) -> bool:
    recall_gain = (candidate.get("recall_at_10") or 0) - (baseline.get("recall_at_10") or 0)
    evidence_gain = (candidate.get("evidence_coverage_at_10") or 0) - (
        baseline.get("evidence_coverage_at_10") or 0
    )
    mrr_drop = (baseline.get("mrr_at_10") or 0) - (candidate.get("mrr_at_10") or 0)
    base_p95 = (baseline.get("latency_ms") or {}).get("p95") or 0
    candidate_p95 = (candidate.get("latency_ms") or {}).get("p95") or 0
    latency_ok = not base_p95 or candidate_p95 <= base_p95 * p95_multiplier
    return (recall_gain >= 0.05 or evidence_gain >= 0.05) and mrr_drop < 0.03 and latency_ok
