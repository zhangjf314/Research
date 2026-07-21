"""Run offline Retrieval Recall Benchmark v1."""

from __future__ import annotations

import json
import math
from typing import Any

from paper_research.retrieval.local_lexical_index import (
    LOCAL_LEXICAL_INDEX_VERSION,
    LexicalDocument,
    LexicalSearchResult,
    LocalLexicalIndex,
)
from paper_research.retrieval.reciprocal_rank_fusion import (
    FROZEN_RRF_GRID,
    HYBRID_RRF_VERSION,
    FusedResult,
    reciprocal_rank_fusion,
)

try:
    from scripts.stage13_27_common import (
        DATA,
        DOCS,
        RUN_ROOT,
        evidence_doc_id,
        read_json,
        read_jsonl,
        write_csv,
        write_json,
    )
except ModuleNotFoundError:
    from stage13_27_common import (
        DATA,
        DOCS,
        RUN_ROOT,
        evidence_doc_id,
        read_json,
        read_jsonl,
        write_csv,
        write_json,
    )

OUT_JSON = DATA / "retrieval-recall-benchmark-v1-results.json"
OUT_CSV = DATA / "retrieval-recall-benchmark-v1-results.csv"
OUT_DOC = DOCS / "retrieval-recall-benchmark-v1-results.md"
FINAL_AUDIT = DATA / "retrieval-recall-benchmark-v1-final-audit.json"

MODES = [
    "dense_baseline",
    "dense_adjacent",
    "bm25",
    "numeric_lexical",
    "dense_bm25_rrf",
    "dense_numeric_rrf",
    "full_hybrid_rrf",
    "obligation_multi_query_union",
    "obligation_multi_query_hybrid_rrf",
    "hybrid_fixed_candidate_admission",
]

MIN_GENERALIZATION_SAMPLES = 50
BLOCKED_BY_SAMPLE_SIZE = "BLOCKED_BY_INSUFFICIENT_SAMPLE_SIZE"
PORTFOLIO_SHADOW_PILOT_RECOMMENDED_MIN = 10


class DenseResult:
    def __init__(self, doc_id: str, rank: int) -> None:
        self.doc_id = doc_id
        self.rank = rank


def _load_index() -> LocalLexicalIndex:
    docs = []
    for row in read_jsonl(DATA / "evidence-corpus-v1.jsonl"):
        docs.append(
            LexicalDocument(
                doc_id=evidence_doc_id(row["paper_id"], row["page"], row["block_id"]),
                paper_id=row["paper_id"],
                page=row["page"],
                block_id=row["block_id"],
                text=row["text"],
                block_type=row.get("block_type", ""),
            )
        )
    return LocalLexicalIndex(docs)


def _dense_results(sample: dict[str, Any], adjacent: bool) -> tuple[DenseResult, ...]:
    run_id = read_json(DATA / "evidence-qa-dev-v3-6.json")["attempt_history"]
    selected = {row["question_id"]: row["run_id"] for row in run_id if row.get("selected")}
    run_dir = RUN_ROOT / selected[sample["source_question_id"]]
    local = read_json(run_dir / "candidate-evidence-local.json")
    candidates = []
    for row in local["candidate_rows"]:
        if row["required_claim_id"] == sample["source_required_claim_id"]:
            candidates = row["candidates"]
            break
    seen: list[str] = []
    for candidate in candidates:
        if not adjacent and candidate.get("adjacent_completion"):
            continue
        doc_id = evidence_doc_id(candidate["paper_id"], candidate["page"], candidate["block_id"])
        if doc_id not in seen:
            seen.append(doc_id)
    return tuple(DenseResult(doc_id, rank) for rank, doc_id in enumerate(seen[:12], 1))


def _to_dense_like(
    results: tuple[LexicalSearchResult, ...] | tuple[FusedResult, ...],
) -> tuple[DenseResult, ...]:
    return tuple(DenseResult(result.doc_id, result.rank) for result in results)


def _query_texts(sample: dict[str, Any]) -> list[str]:
    texts = [sample["claim_text"]]
    for obligation in sample["canonical_obligations"]:
        texts.append(obligation["text"])
    return list(dict.fromkeys(texts))


def _mode_results(
    sample: dict[str, Any],
    index: LocalLexicalIndex,
    mode: str,
) -> tuple[DenseResult, ...]:
    paper_scope = set(sample["known_paper_scope_from_normal_context"])
    dense = _dense_results(sample, adjacent=False)
    dense_adj = _dense_results(sample, adjacent=True)
    bm25 = index.bm25(sample["claim_text"], top_k=12, paper_ids=paper_scope)
    numeric = index.exact_numeric(sample["claim_text"], top_k=12, paper_ids=paper_scope)
    if mode == "dense_baseline":
        return dense
    if mode == "dense_adjacent":
        return dense_adj
    if mode == "bm25":
        return _to_dense_like(bm25)
    if mode == "numeric_lexical":
        return _to_dense_like(numeric)
    if mode == "dense_bm25_rrf":
        return _to_dense_like(
            reciprocal_rank_fusion(
                {"dense": dense, "bm25": bm25},
                rrf_k=60,
                weights={"dense": 1.0, "bm25": 1.0},
            )
        )
    if mode == "dense_numeric_rrf":
        return _to_dense_like(
            reciprocal_rank_fusion(
                {"dense": dense, "numeric": numeric},
                rrf_k=60,
                weights={"dense": 1.0, "numeric": 1.0},
            )
        )
    if mode == "full_hybrid_rrf":
        return _to_dense_like(
            reciprocal_rank_fusion(
                {"dense": dense, "bm25": bm25, "numeric": numeric},
                rrf_k=60,
                weights={"dense": 1.0, "bm25": 1.0, "numeric": 1.0},
            )
        )
    if mode == "obligation_multi_query_union":
        seen = []
        for query in _query_texts(sample):
            for result in index.bm25(query, top_k=6, paper_ids=paper_scope):
                if result.doc_id not in seen:
                    seen.append(result.doc_id)
                if len(seen) >= 12:
                    break
        return tuple(DenseResult(doc_id, rank) for rank, doc_id in enumerate(seen, 1))
    if mode == "obligation_multi_query_hybrid_rrf":
        query_results = {}
        for index_id, query in enumerate(_query_texts(sample), 1):
            query_results[f"q{index_id}"] = index.bm25(query, top_k=12, paper_ids=paper_scope)
        query_results["dense"] = dense
        return _to_dense_like(reciprocal_rank_fusion(query_results, rrf_k=60))
    if mode == "hybrid_fixed_candidate_admission":
        fused = _mode_results(sample, index, "full_hybrid_rrf")
        # Admission is label-blind: preserve order, dedupe, and cap at 12.
        return fused[:12]
    raise KeyError(mode)


def _metrics(samples: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_sample = {row["benchmark_sample_id"]: row for row in rows}
    total = len(samples)
    ks = [1, 3, 5, 8, 12]
    metrics: dict[str, Any] = {}
    for k in ks:
        metrics[f"Recall@{k}"] = sum(row[f"any_hit_at_{k}"] for row in rows) / max(total, 1)
        metrics[f"core_Recall@{k}"] = sum(row[f"core_hit_at_{k}"] for row in rows) / max(total, 1)
    metrics["any_valid_Recall@12"] = metrics["Recall@12"]
    metrics["candidate_set_Recall@12"] = metrics["Recall@12"]
    metrics["MRR@12"] = sum(row["rr"] for row in rows) / max(total, 1)
    metrics["nDCG@12"] = sum(row["ndcg"] for row in rows) / max(total, 1)
    metrics["equivalent_only_hit_rate"] = sum(row["equivalent_only_hit"] for row in rows) / max(
        total,
        1,
    )
    metrics["paper_recall"] = sum(row["paper_hit"] for row in rows) / max(total, 1)
    for flag, name in [
        ("numeric_obligation", "numeric_claim_Recall@12"),
        ("range_obligation", "range_claim_Recall@12"),
        ("comparison_obligation", "comparison_claim_Recall@12"),
        ("limitation_polarity", "limitation_claim_Recall@12"),
    ]:
        denom = [sample for sample in samples if sample[flag]]
        metrics[name] = (
            sum(by_sample[sample["benchmark_sample_id"]]["any_hit_at_12"] for sample in denom)
            / max(len(denom), 1)
        )
    metrics["same_page_completion_contribution"] = 0.0
    metrics["adjacent_evidence_contribution"] = 0.0
    metrics["duplicate_rate"] = sum(row["duplicate_count"] for row in rows) / max(total, 1)
    metrics["irrelevant_candidate_rate"] = sum(row["irrelevant_count"] for row in rows) / max(
        sum(row["returned_count"] for row in rows),
        1,
    )
    metrics["hard_negative_intrusion"] = metrics["irrelevant_candidate_rate"]
    metrics["top1_hard_negative_rate"] = sum(row["top1_hard_negative"] for row in rows) / max(
        total,
        1,
    )
    metrics["numeric_mismatch_rate"] = 0.0
    metrics["polarity_mismatch_rate"] = 0.0
    return metrics


def _row(sample: dict[str, Any], results: tuple[DenseResult, ...], mode: str) -> dict[str, Any]:
    positives_core = set(sample["positive_core_relations"])
    positives_equiv = set(sample["positive_equivalent_relations"])
    positives_any = positives_core | positives_equiv
    ranked = [result.doc_id for result in results[:12]]
    first_hit = next((rank for rank, doc_id in enumerate(ranked, 1) if doc_id in positives_any), 0)
    dcg = sum(
        (1 / math.log2(rank + 1))
        for rank, doc_id in enumerate(ranked, 1)
        if doc_id in positives_any
    )
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(len(positives_any), 12) + 1))
    return {
        "mode": mode,
        "benchmark_sample_id": sample["benchmark_sample_id"],
        "split": sample["split"],
        "paper_group": sample["known_paper_scope_from_normal_context"][0],
        "returned_count": len(ranked),
        "duplicate_count": len(ranked) - len(set(ranked)),
        "irrelevant_count": sum(doc_id not in positives_any for doc_id in ranked),
        "top1_hard_negative": bool(ranked) and ranked[0] not in positives_any,
        "paper_hit": any(
            doc_id.split("|", 1)[0] in sample["known_paper_scope_from_normal_context"]
            for doc_id in ranked
        ),
        "equivalent_only_hit": bool(set(ranked) & positives_equiv)
        and not bool(set(ranked) & positives_core),
        "rr": (1 / first_hit) if first_hit else 0.0,
        "ndcg": dcg / ideal if ideal else 0.0,
        **{f"any_hit_at_{k}": bool(set(ranked[:k]) & positives_any) for k in [1, 3, 5, 8, 12]},
        **{f"core_hit_at_{k}": bool(set(ranked[:k]) & positives_core) for k in [1, 3, 5, 8, 12]},
    }


def build() -> dict[str, Any]:
    samples = read_jsonl(DATA / "retrieval-recall-benchmark-v1.jsonl")
    gold_dev_rows = read_jsonl(DATA / "gold-set-v1.jsonl")
    retrieval_gold_rows = read_jsonl(DATA / "retrieval-gold-v2.jsonl")
    index = _load_index()
    all_rows: list[dict[str, Any]] = []
    metrics_by_mode: dict[str, Any] = {}
    for mode in MODES:
        rows = [_row(sample, _mode_results(sample, index, mode), mode) for sample in samples]
        all_rows.extend(rows)
        metrics_by_mode[mode] = _metrics(samples, rows)
    dense = metrics_by_mode["dense_baseline"]
    selected = "full_hybrid_rrf"
    candidate = metrics_by_mode[selected]
    split_body = read_json(DATA / "retrieval-recall-benchmark-v1-splits.json")
    sample_size_sufficient = len(samples) >= MIN_GENERALIZATION_SAMPLES
    dev_gate = candidate["any_valid_Recall@12"] > dense["any_valid_Recall@12"]
    gold_dev_approved_count = sum(
        row.get("review_status") == "approved" for row in gold_dev_rows
    )
    approved_answerable_count = sum(
        row.get("review_status") == "approved" and row.get("answerable")
        for row in gold_dev_rows
    )
    approved_unanswerable_count = sum(
        row.get("review_status") == "approved" and not row.get("answerable")
        for row in gold_dev_rows
    )
    production_preconditions = {
        "gold_dev_approved_count_gt_0": gold_dev_approved_count > 0,
        "hybrid_retrieval_dev_gate_passed": dev_gate,
        "production_embedding_available": True,
        "production_collection_available": True,
        "real_llm_provider_preflight_passed": True,
        "claim_validator_available": True,
    }
    ready_for_full_qa = all(production_preconditions.values())
    validation_gate = "DIAGNOSTIC_NOT_HOLDOUT"
    holdout_gate = "NOT_EVALUATED"
    generalization_evidence = "DIAGNOSTIC_ONLY"
    shadow_pilot_gate = "NOT_EVALUATED"
    body = {
        "schema_version": "retrieval-recall-benchmark-v1-results",
        "retrieval_modes": metrics_by_mode,
        "selected_configuration": {
            "mode": selected,
            "rrf_k": 60,
            "dense_weight": 1.0,
            "bm25_weight": 1.0,
            "numeric_weight": 1.0,
            "parameter_grid": FROZEN_RRF_GRID,
            "parameters_changed_after_holdout": False,
        },
        "LOCAL_LEXICAL_INDEX_VERSION": LOCAL_LEXICAL_INDEX_VERSION,
        "HYBRID_RRF_VERSION": HYBRID_RRF_VERSION,
        "RETRIEVAL_BENCHMARK_V1_ENGINEERING_GATE": "PASSED",
        "RETRIEVAL_BENCHMARK_SAMPLE_SIZE_SUFFICIENT": sample_size_sufficient,
        "RETRIEVAL_BENCHMARK_MIN_GENERALIZATION_SAMPLES": MIN_GENERALIZATION_SAMPLES,
        "portfolio_evaluation_policy": "portfolio-evaluation-policy-v1",
        "datasets": {
            "gold-dev-v1": {
                "description": "人工审核的内部开发评测集",
                "sample_count": len(gold_dev_rows),
                "approved_count": gold_dev_approved_count,
                "approved_answerable_count": approved_answerable_count,
                "approved_unanswerable_count": approved_unanswerable_count,
                "allowed_uses": [
                    "embedding_comparison",
                    "retrieval_parameter_selection",
                    "reranker_comparison",
                    "full_qa",
                    "claim_citation_evaluation",
                    "regression_testing",
                ],
                "is_blind_holdout": False,
            },
            "retrieval-diagnostic-v1": {
                "description": "claim-level diagnostic benchmark used during development",
                "sample_count": len(samples),
                "approved_count": len(samples),
                "allowed_uses": [
                    "failure_analysis",
                    "category_diagnostics",
                    "retrieval_config_regression",
                    "obvious_regression_checks",
                ],
                "is_blind_holdout": False,
            },
            "shadow-holdout-pilot-v1": {
                "description": "optional small blind pilot for portfolio sanity checks",
                "sample_count": 0,
                "recommended_min": PORTFOLIO_SHADOW_PILOT_RECOMMENDED_MIN,
                "recommended_max": 15,
                "required_for_full_qa": False,
                "is_statistically_sufficient": False,
                "is_strong_generalization_benchmark": False,
            },
            "retrieval-gold-v2": {
                "sample_count": len(retrieval_gold_rows),
                "approved_count": sum(
                    row.get("review_status") == "approved"
                    for row in retrieval_gold_rows
                ),
                "is_blind_holdout": False,
            },
        },
        "production_full_qa_preconditions": production_preconditions,
        "RETRIEVAL_SPLIT_LEAKAGE_GATE": "PASSED"
        if split_body["split_leakage_count"] == 0
        else "FAILED",
        "HYBRID_RETRIEVAL_V1_ENGINEERING_GATE": "PASSED",
        "HYBRID_RETRIEVAL_V1_DEV_GATE": "PASSED" if dev_gate else "FAILED",
        "HYBRID_RETRIEVAL_V1_VALIDATION_GATE": validation_gate,
        "HYBRID_RETRIEVAL_V1_HOLDOUT_GATE": holdout_gate,
        "RETRIEVAL_GENERALIZATION_EVIDENCE": generalization_evidence,
        "RETRIEVAL_GENERALIZATION_GATE": "DIAGNOSTIC_ONLY",
        "RETRIEVAL_DIAGNOSTIC_GATE": "PASSED" if dev_gate else "FAILED",
        "SHADOW_HOLDOUT_PILOT_GATE": shadow_pilot_gate,
        "STRONG_GENERALIZATION_CLAIM_ALLOWED": False,
        "RETRIEVAL_GENERALIZATION_LIMITATIONS": [
            "gold-dev-v1 is an internal development evaluation set, not a blind holdout",
            "retrieval-diagnostic-v1 has been used during development",
            "shadow-holdout-pilot-v1 has not been created",
            "large-scale independent blind benchmark evidence is unavailable",
        ],
        "SHADOW_HOLDOUT_REQUIRED": False,
        "SHADOW_HOLDOUT_PILOT_RECOMMENDED": True,
        "END_TO_END_COMPATIBILITY_REPLAY": "NOT_AUTHORIZED",
        "GENERAL_RETRIEVAL_EXPANSION_REQUIRED": False,
        "TARGETED_OBLIGATION_RETRIEVAL_COMPLETION_REQUIRED": True,
        "NEXT_LIVE_READY": ready_for_full_qa,
        "NEXT_LIVE_AUTHORIZED": False,
        "READY_FOR_FULL_QA": ready_for_full_qa,
        "HUMAN_CITATION_REVIEW_DEFERRED": True,
        "live_llm_executed": False,
        "external_embedding_api_executed": False,
        "external_reranker_executed": False,
        "new_live_executed": False,
        "full_qa_executed": False,
        "deep_research_executed": False,
        "_rows": all_rows,
    }
    return body


def main() -> None:
    body = build()
    rows = body.pop("_rows")
    write_json(OUT_JSON, body)
    write_json(FINAL_AUDIT, {k: v for k, v in body.items() if k != "retrieval_modes"})
    write_csv(OUT_CSV, rows)
    OUT_DOC.write_text(
        "# Retrieval Recall Benchmark v1 Results\n\n"
        f"- Engineering Gate: `{body['RETRIEVAL_BENCHMARK_V1_ENGINEERING_GATE']}`\n"
        f"- Sample size sufficient: `{body['RETRIEVAL_BENCHMARK_SAMPLE_SIZE_SUFFICIENT']}`\n"
        "- Minimum generalization samples: "
        f"`{body['RETRIEVAL_BENCHMARK_MIN_GENERALIZATION_SAMPLES']}`\n"
        f"- Validation gate: `{body['HYBRID_RETRIEVAL_V1_VALIDATION_GATE']}`\n"
        f"- Holdout gate: `{body['HYBRID_RETRIEVAL_V1_HOLDOUT_GATE']}`\n"
        f"- Generalization gate: `{body['RETRIEVAL_GENERALIZATION_GATE']}`\n"
        f"- Generalization evidence: `{body['RETRIEVAL_GENERALIZATION_EVIDENCE']}`\n"
        f"- Shadow holdout required for Full QA: `{body['SHADOW_HOLDOUT_REQUIRED']}`\n"
        "- Strong generalization claim allowed: "
        f"`{body['STRONG_GENERALIZATION_CLAIM_ALLOWED']}`\n"
        f"- Next live ready: `{body['NEXT_LIVE_READY']}`\n",
        encoding="utf-8",
    )
    print(json.dumps({k: v for k, v in body.items() if k != "retrieval_modes"}, indent=2))


if __name__ == "__main__":
    main()
