"""Generate Stage 13.28 retrieval failure and generalization audits.

This script is offline-only.  It uses the frozen Stage 13.27 benchmark labels for
scoring and never calls an LLM, embedding provider, reranker, or external service.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from paper_research.retrieval.local_lexical_index import LOCAL_LEXICAL_INDEX_VERSION

try:
    from scripts.run_retrieval_recall_benchmark_v1 import (
        _load_index,
        _mode_results,
    )
    from scripts.stage13_27_common import DATA, DOCS, evidence_doc_id, read_json, read_jsonl
except ModuleNotFoundError:
    from run_retrieval_recall_benchmark_v1 import _load_index, _mode_results
    from stage13_27_common import DATA, DOCS, evidence_doc_id, read_json, read_jsonl


FAILURE_JSONL = DATA / "retrieval-failures-v1.jsonl"
GENERALIZATION_JSON = DATA / "retrieval-generalization-v1.json"
GENERALIZATION_CSV = DATA / "retrieval-generalization-v1.csv"
SHADOW_HOLDOUT_JSON = DATA / "shadow-holdout-requirements-v1.json"
FAILURE_DOC = DOCS / "retrieval-failure-analysis-v1.md"
GENERALIZATION_DOC = DOCS / "retrieval-generalization-audit-v1.md"
SHADOW_HOLDOUT_DOC = DOCS / "shadow-holdout-requirements-v1.md"


def _doc_lookup() -> dict[str, dict[str, Any]]:
    lookup = {}
    for row in read_jsonl(DATA / "evidence-corpus-v1.jsonl"):
        lookup[evidence_doc_id(row["paper_id"], row["page"], row["block_id"])] = row
    return lookup


def _rank(results: list[str], positives: set[str]) -> int | None:
    for idx, doc_id in enumerate(results, 1):
        if doc_id in positives:
            return idx
    return None


def _category(sample: dict[str, Any], dense_rank: int | None, sparse_rank: int | None) -> str:
    if not sample["positive_core_relations"] and sample["positive_equivalent_relations"]:
        return "GOLD_LABEL_ISSUE"
    if sample["numeric_obligation"] or sample["range_obligation"]:
        return "TABLE_OR_FORMULA"
    if sample["comparison_obligation"]:
        return "MULTI_HOP"
    if dense_rank is None and sparse_rank is not None:
        return "DENSE_MISS"
    if sparse_rank is None and dense_rank is not None:
        return "SPARSE_MISS"
    if dense_rank is not None and sparse_rank is not None:
        return "FUSION_DEGRADATION"
    return "UNKNOWN"


def _top_rows(doc_ids: list[str], docs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for rank, doc_id in enumerate(doc_ids[:10], 1):
        doc = docs.get(doc_id, {})
        rows.append(
            {
                "rank": rank,
                "relation": doc_id,
                "paper_id": doc.get("paper_id", doc_id.split("|")[0]),
                "page": doc.get("page"),
                "block_id": doc.get("block_id", doc_id.split("|")[-1]),
                "block_type": doc.get("block_type"),
                "text_preview": str(doc.get("text", ""))[:240],
            }
        )
    return rows


def build() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    samples = read_jsonl(DATA / "retrieval-recall-benchmark-v1.jsonl")
    results = read_json(DATA / "retrieval-recall-benchmark-v1-results.json")
    splits = read_json(DATA / "retrieval-recall-benchmark-v1-splits.json")
    docs = _doc_lookup()
    index = _load_index()
    failures: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    selected_mode = results["selected_configuration"]["mode"]
    for sample in samples:
        dense_docs = [r.doc_id for r in _mode_results(sample, index, "dense_baseline")[:12]]
        sparse_docs = [r.doc_id for r in _mode_results(sample, index, "bm25")[:12]]
        hybrid_docs = [r.doc_id for r in _mode_results(sample, index, selected_mode)[:12]]
        core = set(sample["positive_core_relations"])
        equiv = set(sample["positive_equivalent_relations"])
        any_positive = core | equiv
        dense_rank = _rank(dense_docs, any_positive)
        sparse_rank = _rank(sparse_docs, any_positive)
        hybrid_rank = _rank(hybrid_docs, any_positive)
        core_rank = _rank(hybrid_docs, core)
        failure_category = _category(sample, dense_rank, sparse_rank)
        core_hit_at_10 = bool(set(hybrid_docs[:10]) & core)
        any_hit_at_10 = bool(set(hybrid_docs[:10]) & any_positive)
        failed = not core_hit_at_10
        summary_rows.append(
            {
                "question_id": sample["source_question_id"],
                "required_claim_id": sample["source_required_claim_id"],
                "benchmark_sample_id": sample["benchmark_sample_id"],
                "split": sample["split"],
                "claim_type": sample["claim_type"],
                "core_hit_at_10": core_hit_at_10,
                "any_hit_at_10": any_hit_at_10,
                "dense_rank": dense_rank or "",
                "sparse_rank": sparse_rank or "",
                "hybrid_rank": hybrid_rank or "",
                "failure_category": "" if not failed else failure_category,
            }
        )
        if failed:
            gold_relations = list(core or any_positive)
            failures.append(
                {
                    "question_id": sample["source_question_id"],
                    "required_claim_id": sample["source_required_claim_id"],
                    "benchmark_sample_id": sample["benchmark_sample_id"],
                    "category": sample["claim_type"],
                    "scope": "paper",
                    "gold_paper": sample["known_paper_scope_from_normal_context"],
                    "gold_relations": gold_relations,
                    "gold_pages": sorted(
                        {int(relation.split("|")[1]) for relation in gold_relations}
                    ),
                    "gold_blocks": [relation.split("|")[2] for relation in gold_relations],
                    "top10": _top_rows(hybrid_docs, docs),
                    "dense_rank": dense_rank,
                    "sparse_rank": sparse_rank,
                    "hybrid_rank": hybrid_rank,
                    "core_rank": core_rank,
                    "metadata_filter_excluded": False,
                    "chunk_missing": False,
                    "query_formulation_issue": dense_rank is None and sparse_rank is None,
                    "index_error": False,
                    "gold_label_issue": not bool(core) and bool(equiv),
                    "synonym_issue": dense_rank is None and sparse_rank is not None,
                    "multi_hop": sample["comparison_obligation"],
                    "table_or_formula": sample["numeric_obligation"] or sample["range_obligation"],
                    "ocr_content": False,
                    "failure_category": failure_category,
                }
            )
    category_counts = Counter(row["failure_category"] for row in failures)
    limitations = list(results.get("RETRIEVAL_GENERALIZATION_LIMITATIONS", []))
    body = {
        "schema_version": "retrieval-generalization-v1",
        "dataset_version": "retrieval-recall-benchmark-v1",
        "sample_count": len(samples),
        "approved_count": len(samples),
        "split_counts": splits["split_counts"],
        "split_strategy": splits["split_strategy"],
        "split_leakage_count": splits["split_leakage_count"],
        "paraphrase_leakage_count": splits["paraphrase_leakage_count"],
        "relation_leakage_count": splits["relation_leakage_count"],
        "holdout_contaminated": True,
        "holdout_contamination_reason": (
            "The fixed 27-claim benchmark and its failures have been repeatedly inspected "
            "during Stage 13.27/13.28; it is diagnostic, not a strict blind holdout."
        ),
        "embedding_provider": "offline frozen candidate evidence; no external embedding API",
        "collection": "papers_jina_eval34_v2__20260713152149",
        "dense_parameters": {"source": "frozen Stage 13.21/13.26 candidate evidence order"},
        "sparse_parameters": {"index": LOCAL_LEXICAL_INDEX_VERSION, "top_k": 12},
        "fusion_parameters": results["selected_configuration"],
        "reranker_enabled": False,
        "top_k": 12,
        "thresholds": {
            "portfolio_full_qa_shadow_holdout_required": False,
            "shadow_holdout_pilot_recommended_min": 10,
            "shadow_holdout_pilot_recommended_max": 15,
            "strong_generalization_requires_future_strict_blind_holdout": True,
        },
        "gates": {
            key: results[key]
            for key in [
                "RETRIEVAL_BENCHMARK_V1_ENGINEERING_GATE",
                "RETRIEVAL_BENCHMARK_SAMPLE_SIZE_SUFFICIENT",
                "RETRIEVAL_BENCHMARK_MIN_GENERALIZATION_SAMPLES",
                "RETRIEVAL_SPLIT_LEAKAGE_GATE",
                "HYBRID_RETRIEVAL_V1_DEV_GATE",
                "RETRIEVAL_DIAGNOSTIC_GATE",
                "HYBRID_RETRIEVAL_V1_VALIDATION_GATE",
                "HYBRID_RETRIEVAL_V1_HOLDOUT_GATE",
                "SHADOW_HOLDOUT_PILOT_GATE",
                "RETRIEVAL_GENERALIZATION_EVIDENCE",
                "RETRIEVAL_GENERALIZATION_GATE",
                "STRONG_GENERALIZATION_CLAIM_ALLOWED",
                "NEXT_LIVE_READY",
                "READY_FOR_FULL_QA",
            ]
        },
        "datasets": results["datasets"],
        "production_full_qa_preconditions": results["production_full_qa_preconditions"],
        "retrieval_modes": results["retrieval_modes"],
        "failure_count": len(failures),
        "failure_category_counts": dict(category_counts),
        "limitations": limitations,
        "shadow_holdout_required": bool(results.get("SHADOW_HOLDOUT_REQUIRED", False)),
        "shadow_holdout_pilot_recommended": bool(
            results.get("SHADOW_HOLDOUT_PILOT_RECOMMENDED", True)
        ),
        "portfolio_allowed_claims": [
            "基于 50 条人工审核的内部评测数据完成检索和问答评测",
            "27 条 claim-level diagnostic 集用于失败分析和回归检查",
        ],
        "forbidden_claims": [
            "strict generalization benchmark",
            "large-scale independent blind benchmark",
            "production-grade generalization proven",
            "statistically sufficient holdout",
        ],
        "configuration_frozen_for_holdout": False,
        "configuration_hash": None,
        "READY_FOR_FULL_QA": bool(results["READY_FOR_FULL_QA"]),
        "NEXT_LIVE_READY": bool(results["NEXT_LIVE_READY"]),
        "live_llm_executed": False,
        "external_embedding_api_executed": False,
        "external_reranker_executed": False,
        "full_qa_executed": False,
        "deep_research_executed": False,
    }
    return failures, body | {"_summary_rows": summary_rows}


def _write_csv(path, rows: list[dict[str, Any]]) -> None:
    import csv

    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _shadow_holdout_requirements(body: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "shadow-holdout-requirements-v1",
        "status": "OPTIONAL_RECOMMENDED",
        "reason": "portfolio_sanity_check_not_full_qa_blocker",
        "recommended_min_samples": 10,
        "recommended_max_samples": 15,
        "current_samples": 0,
        "additional_samples_needed_for_recommended_min": 10,
        "required_for_full_qa": False,
        "required_for_portfolio_v1_0": False,
        "statistically_sufficient": False,
        "strong_generalization_claim_allowed": False,
        "must_be_blind_until_configuration_frozen": True,
        "paper_requirements": {
            "minimum_new_papers": 5,
            "maximum_samples_per_paper": 3,
            "minimum_unanswerable_questions": 2,
            "mark_revealed_after_first_formal_run": True,
        },
        "allowed_sources": [
            "new manually reviewed claim-level evidence labels",
            (
                "new question-level labels only if converted to claim/evidence "
                "relations before scoring"
            ),
        ],
        "prohibited_sources": [
            "LLM auto-approved labels",
            "labels inferred from retrieval hits",
            "samples previously inspected during retrieval tuning",
        ],
        "required_fields": [
            "question_id",
            "required_claim_id",
            "claim_text",
            "target_papers",
            "positive_core_relations",
            "positive_equivalent_relations",
            "split",
            "review_status",
            "reviewer",
            "reviewed_at",
            "source_hashes",
        ],
        "release_effect": {
            "READY_FOR_FULL_QA": body["READY_FOR_FULL_QA"],
            "NEXT_LIVE_READY": body["NEXT_LIVE_READY"],
            "v1_0_portfolio_allowed": False,
            "generalization_evidence_if_not_run": "DIAGNOSTIC_ONLY",
            "generalization_evidence_if_passed": "LIMITED_BUT_ACCEPTABLE_FOR_PORTFOLIO",
        },
    }


def main() -> None:
    failures, body = build()
    summary_rows = body.pop("_summary_rows")
    FAILURE_JSONL.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in failures) + "\n",
        encoding="utf-8",
    )
    GENERALIZATION_JSON.write_text(
        json.dumps(body, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    shadow = _shadow_holdout_requirements(body)
    SHADOW_HOLDOUT_JSON.write_text(
        json.dumps(shadow, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_csv(GENERALIZATION_CSV, summary_rows)
    FAILURE_DOC.write_text(
        "# Retrieval Failure Analysis v1\n\n"
        f"- Samples: `{body['sample_count']}`\n"
        f"- Failure records: `{body['failure_count']}`\n"
        f"- Failure categories: `{body['failure_category_counts']}`\n"
        "- Gold labels are used only for offline scoring and failure attribution.\n"
        "- No LLM, external embedding API, external reranker, Full QA, or Deep Research was run.\n",
        encoding="utf-8",
    )
    GENERALIZATION_DOC.write_text(
        "# Retrieval Generalization Audit v1\n\n"
        f"- Engineering gate: `{body['gates']['RETRIEVAL_BENCHMARK_V1_ENGINEERING_GATE']}`\n"
        "- Dataset policy: `portfolio-evaluation-policy-v1`\n"
        "- gold-dev-v1: "
        f"`{body['datasets']['gold-dev-v1']['sample_count']}` samples, "
        f"`{body['datasets']['gold-dev-v1']['approved_count']}` approved; "
        "internal development evaluation set, not blind holdout\n"
        "- retrieval-diagnostic-v1: "
        f"`{body['datasets']['retrieval-diagnostic-v1']['sample_count']}` "
        "claim-level samples; diagnostic, not blind\n"
        "- Sample size sufficient: "
        f"`{body['gates']['RETRIEVAL_BENCHMARK_SAMPLE_SIZE_SUFFICIENT']}`\n"
        f"- Dev retrieval gate: `{body['gates']['HYBRID_RETRIEVAL_V1_DEV_GATE']}`\n"
        f"- Diagnostic gate: `{body['gates']['RETRIEVAL_DIAGNOSTIC_GATE']}`\n"
        f"- Validation gate: `{body['gates']['HYBRID_RETRIEVAL_V1_VALIDATION_GATE']}`\n"
        f"- Holdout gate: `{body['gates']['HYBRID_RETRIEVAL_V1_HOLDOUT_GATE']}`\n"
        f"- Shadow pilot gate: `{body['gates']['SHADOW_HOLDOUT_PILOT_GATE']}`\n"
        f"- Generalization gate: `{body['gates']['RETRIEVAL_GENERALIZATION_GATE']}`\n"
        f"- Generalization evidence: `{body['gates']['RETRIEVAL_GENERALIZATION_EVIDENCE']}`\n"
        "- Strong generalization claim allowed: "
        f"`{body['gates']['STRONG_GENERALIZATION_CLAIM_ALLOWED']}`\n"
        f"- Holdout contaminated: `{body['holdout_contaminated']}`\n"
        f"- Shadow pilot required for Full QA: `{body['shadow_holdout_required']}`\n"
        f"- NEXT_LIVE_READY: `{body['NEXT_LIVE_READY']}`\n"
        f"- READY_FOR_FULL_QA: `{body['READY_FOR_FULL_QA']}`\n\n"
        "## Limitations\n\n"
        + "\n".join(f"- {reason}" for reason in body["limitations"])
        + "\n\n"
        "## Portfolio-safe claims\n\n"
        + "\n".join(f"- {claim}" for claim in body["portfolio_allowed_claims"])
        + "\n\n"
        "## Forbidden claims\n\n"
        + "\n".join(f"- {claim}" for claim in body["forbidden_claims"])
        + "\n\n"
        "Full QA is no longer blocked solely by the absence of a 50-sample strict blind "
        "shadow holdout. The project may proceed to Production Full QA under the portfolio "
        "policy while explicitly disclosing that large-scale independent blind "
        "generalization evidence is not available.\n",
        encoding="utf-8",
    )
    SHADOW_HOLDOUT_DOC.write_text(
        "# Shadow Holdout Requirements v1\n\n"
        f"- Status: `{shadow['status']}`\n"
        f"- Recommended samples: `{shadow['recommended_min_samples']}`"
        f"-`{shadow['recommended_max_samples']}`\n"
        f"- Current samples: `{shadow['current_samples']}`\n"
        "- Additional samples needed for recommended minimum: "
        f"`{shadow['additional_samples_needed_for_recommended_min']}`\n"
        f"- Required for Full QA: `{shadow['required_for_full_qa']}`\n"
        f"- Statistically sufficient: `{shadow['statistically_sufficient']}`\n"
        "- Strong generalization claim allowed: "
        f"`{shadow['strong_generalization_claim_allowed']}`\n"
        "- The shadow holdout must remain blind until the retrieval configuration is frozen.\n"
        "- LLM auto-approval, retrieval-hit-derived labels, and previously inspected "
        "samples are prohibited.\n",
        encoding="utf-8",
    )
    print(json.dumps({k: v for k, v in body.items() if k != "retrieval_modes"}, indent=2))


if __name__ == "__main__":
    main()
