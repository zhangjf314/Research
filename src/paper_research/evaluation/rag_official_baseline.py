from __future__ import annotations

# ruff: noqa: E501
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from paper_research.evaluation.rag_benchmark import (
    evaluate_retrieval_question,
    first_relevant_rank,
    mean,
    write_json,
)

SYSTEM_UNDER_TEST_COMMIT = "f97746e84b98d6b4e07984a3abbdab206f156839"
BASELINE_CONFIG_HASH = "8817891ed73fc0c0fa2f3a7fc90baf3591b3ab0006934bbb058cbff87e657c94"
FULL_DATASET_HASH = "b5a2a25e540affc1acaed3e339067d0c541afd4014dd29f211ad33f85f033935"
DEV_DATASET_HASH = "f61fc199c559250d32811f755db1400114b131b5da7b94b7abab6be0340c722a"
TEST_DATASET_HASH = "e991feb4d1d60a852926d736ed4e0a97f72a437b67def5dfb9afe4cde4e0eaf8"
DATASET_VERSION = "rag-gold-v1"

BAD_CASE_TYPES = {
    "RETRIEVAL_MISS",
    "RANKING_ERROR",
    "PARTIAL_EVIDENCE_RETRIEVAL",
    "WRONG_PAPER",
    "QUERY_UNDERSTANDING",
    "CHUNKING_OR_PARSE_ERROR",
    "METADATA_ERROR",
    "UNANSWERABLE_FALSE_POSITIVE",
    "OTHER",
}

GENERATION_BAD_CASE_TYPES = {
    "RETRIEVAL_ROOTED",
    "GENERATION_OMISSION",
    "UNSUPPORTED_CLAIM",
    "CITATION_MISMATCH",
    "ANSWER_INCOMPLETE",
    "OVERGENERALIZATION",
    "ABSTENTION_FAILURE",
    "PROVIDER_FAILURE",
    "CONTEXT_OVERLOAD",
    "OTHER",
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[max(0, math.ceil(len(ordered) * fraction) - 1)], 3)


def split_records(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {
        "full": records,
        "dev": [row for row in records if row.get("split") == "dev"],
        "test": [row for row in records if row.get("split") == "test"],
    }


def safe_context_item(
    item: dict[str, Any], rank: int, paper_id_map: dict[str, str] | None = None
) -> dict[str, Any]:
    paper_id = str(item.get("paper_id"))
    if paper_id_map:
        paper_id = paper_id_map.get(paper_id, paper_id)
    return {
        "rank": rank,
        "chunk_id": item.get("chunk_id"),
        "paper_id": paper_id,
        "block_ids": item.get("block_ids") or [item.get("chunk_id")],
        "page_start": item.get("page_start"),
        "page_end": item.get("page_end"),
        "score": item.get("score"),
        "section_path": item.get("section_path") or [],
    }


def retrieval_row(
    gold: dict[str, Any],
    context: list[dict[str, Any]],
    *,
    latency_ms: float,
    failure: str | None = None,
    paper_id_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    ranked = [
        safe_context_item(item, rank, paper_id_map=paper_id_map)
        for rank, item in enumerate(context, start=1)
    ]
    metrics = evaluate_retrieval_question(gold, ranked) if not failure else {}
    found_at = {}
    gold_blocks = set(gold.get("gold_block_ids") or [])
    for k in (5, 10, 20):
        retrieved = {
            block_id
            for item in ranked[:k]
            for block_id in (item.get("block_ids") or [item.get("chunk_id")])
        }
        found_at[str(k)] = sorted(gold_blocks & retrieved)
    first_rank = first_relevant_rank(
        [
            bool(set(item.get("block_ids") or [item.get("chunk_id")]) & gold_blocks)
            for item in ranked
        ]
    )
    return {
        "question_id": gold["question_id"],
        "split": gold.get("split"),
        "category": gold.get("category"),
        "difficulty": gold.get("difficulty"),
        "answerable": gold.get("answerable"),
        "gold_paper_ids": gold.get("gold_paper_ids", []),
        "gold_block_ids": gold.get("gold_block_ids", []),
        "retrieved_paper_ids": [item.get("paper_id") for item in ranked],
        "retrieved_block_ids": [
            block_id for item in ranked for block_id in (item.get("block_ids") or [])
        ],
        "ranked_results": ranked,
        "gold_first_rank": first_rank,
        "gold_blocks_found_at_5": found_at["5"],
        "gold_blocks_found_at_10": found_at["10"],
        "gold_blocks_found_at_20": found_at["20"],
        "paper_recall": metrics.get("paper_recall_at_10"),
        "evidence_recall": metrics.get("recall_at_10"),
        "metrics": metrics,
        "retrieval_failure": failure,
        "latency_ms": latency_ms,
    }


def aggregate_retrieval_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    answerable = [row for row in rows if row.get("answerable") and not row.get("retrieval_failure")]
    unanswerable = [row for row in rows if not row.get("answerable") and not row.get("retrieval_failure")]
    failed = [row for row in rows if row.get("retrieval_failure")]

    def avg(metric: str, source: list[dict[str, Any]] = answerable) -> float | None:
        values = [row["metrics"].get(metric) for row in source if row.get("metrics", {}).get(metric) is not None]
        return round(mean([float(value) for value in values]), 6) if values else None

    return {
        "question_count": len(rows),
        "completed": len(rows) - len(failed),
        "failed": len(failed),
        "answerable_count": len(answerable),
        "unanswerable_count": len(unanswerable),
        "recall_at_5": avg("recall_at_5"),
        "recall_at_10": avg("recall_at_10"),
        "recall_at_20": avg("recall_at_20"),
        "mrr_at_10": avg("mrr_at_10"),
        "ndcg_at_10": avg("ndcg_at_10"),
        "paper_recall_at_5": avg("paper_recall_at_5"),
        "paper_recall_at_10": avg("paper_recall_at_10"),
        "evidence_coverage_at_5": avg("evidence_coverage_at_5"),
        "evidence_coverage_at_10": avg("evidence_coverage_at_10"),
        "evidence_coverage_at_20": avg("evidence_coverage_at_20"),
        "unanswerable_irrelevant_retrieval_rate": avg("irrelevant_retrieval_rate", unanswerable),
        "latency_ms": {
            "mean": round(mean([row["latency_ms"] for row in rows]), 3) if rows else None,
            "p50": percentile([row["latency_ms"] for row in rows], 0.5),
            "p95": percentile([row["latency_ms"] for row in rows], 0.95),
        },
    }


def grouped_retrieval(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(field))].append(row)
    return {key: aggregate_retrieval_rows(value) for key, value in sorted(groups.items())}


def classify_retrieval_bad_case(row: dict[str, Any]) -> str | None:
    if row.get("retrieval_failure"):
        return "OTHER"
    if not row.get("answerable"):
        return "UNANSWERABLE_FALSE_POSITIVE" if row.get("ranked_results") else None
    metrics = row.get("metrics", {})
    if metrics.get("recall_at_20") == 0:
        return "RETRIEVAL_MISS"
    if metrics.get("evidence_coverage_at_20", 0) < 1 and metrics.get("recall_at_20", 0) > 0:
        return "PARTIAL_EVIDENCE_RETRIEVAL"
    if metrics.get("recall_at_10", 0) == 0 or metrics.get("mrr_at_10", 0) == 0:
        return "RANKING_ERROR"
    if metrics.get("paper_recall_at_10", 0) == 0:
        return "WRONG_PAPER"
    return None


def retrieval_bad_cases(rows: list[dict[str, Any]]) -> dict[str, Any]:
    items = []
    counts: Counter[str] = Counter()
    for row in rows:
        case = classify_retrieval_bad_case(row)
        if not case:
            continue
        counts[case] += 1
        items.append(
            {
                "question_id": row["question_id"],
                "split": row.get("split"),
                "category": row.get("category"),
                "difficulty": row.get("difficulty"),
                "primary_bad_case": case,
                "gold_first_rank": row.get("gold_first_rank"),
                "diagnosis": _retrieval_diagnosis(case),
            }
        )
    return {"distribution": dict(sorted(counts.items())), "items": items}


def _retrieval_diagnosis(case: str) -> str:
    return {
        "RETRIEVAL_MISS": "Gold evidence is absent from Top20.",
        "RANKING_ERROR": "Gold evidence is retrieved late but not in the evaluated cutoff.",
        "PARTIAL_EVIDENCE_RETRIEVAL": "Only part of a multi-block Gold evidence set is retrieved.",
        "WRONG_PAPER": "Top results do not cover the Gold paper set.",
        "UNANSWERABLE_FALSE_POSITIVE": "Unanswerable query retrieved plausible non-Gold context.",
        "OTHER": "Technical retrieval failure or unclassified retrieval issue.",
    }.get(case, "Unclassified retrieval issue.")


def claim_text(claim: Any) -> str:
    if isinstance(claim, dict):
        return str(claim.get("text", ""))
    return str(claim)


def _claim_overlap(expected: str, actual: str) -> float:
    expected_terms = {token for token in expected.lower().split() if token}
    actual_terms = {token for token in actual.lower().split() if token}
    return len(expected_terms & actual_terms) / max(1, len(expected_terms))


def evaluate_generation_answer(answer: dict[str, Any], gold: dict[str, Any], retrieval: dict[str, Any] | None = None) -> dict[str, Any]:
    if not gold.get("answerable"):
        citations = [citation for claim in answer.get("claims", []) for citation in claim.get("citations", [])]
        return {
            "required_claim_coverage": None,
            "supported_claim_ratio": None,
            "citation_precision": 1.0 if not citations else 0.0,
            "citation_recall": 1.0 if not citations else 0.0,
            "answer_completeness": 1.0 if not answer.get("answerable") else 0.0,
            "abstention_accuracy": 1.0 if not answer.get("answerable") else 0.0,
            "unsupported_claim_count": 0 if not citations else len(answer.get("claims", [])),
            "gold_evidence_in_context": False,
        }
    required = [claim_text(claim) for claim in gold.get("required_claims", [])]
    generated_claims = answer.get("claims", [])
    coverage = [
        max((_claim_overlap(req, str(claim.get("text", ""))) for claim in generated_claims), default=0.0)
        >= 0.35
        for req in required
    ]
    citations = [citation for claim in generated_claims for citation in claim.get("citations", [])]
    gold_blocks = set(gold.get("gold_block_ids", []))
    gold_papers = set(gold.get("gold_paper_ids", []))
    gold_pages = set(gold.get("gold_pages", []))
    precise = [
        citation.get("block_id") in gold_blocks
        and citation.get("paper_id") in gold_papers
        and citation.get("page") in gold_pages
        for citation in citations
    ]
    cited_gold_blocks = {citation.get("block_id") for citation in citations} & gold_blocks
    unsupported = sum(
        not any(
            citation.get("block_id") in gold_blocks
            and citation.get("paper_id") in gold_papers
            and citation.get("page") in gold_pages
            for citation in claim.get("citations", [])
        )
        for claim in generated_claims
    )
    retrieval_blocks = set()
    if retrieval:
        retrieval_blocks = {
            block_id
            for item in retrieval.get("ranked_results", [])[:5]
            for block_id in item.get("block_ids", [])
        }
    return {
        "required_claim_coverage": sum(coverage) / len(coverage) if coverage else 0.0,
        "supported_claim_ratio": 1 - unsupported / len(generated_claims) if generated_claims else 0.0,
        "citation_precision": sum(precise) / len(precise) if precise else 0.0,
        "citation_recall": len(cited_gold_blocks) / len(gold_blocks) if gold_blocks else 0.0,
        "answer_completeness": sum(coverage) / len(coverage) if coverage else 0.0,
        "abstention_accuracy": None,
        "unsupported_claim_count": unsupported,
        "gold_evidence_in_context": bool(retrieval_blocks & gold_blocks),
    }


def aggregate_generation_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [row for row in rows if row.get("status") == "COMPLETED"]
    failed = [row for row in rows if row.get("status") == "FAILED"]
    answerable = [row for row in completed if row["gold"]["answerable"]]
    unanswerable = [row for row in completed if not row["gold"]["answerable"]]

    def avg(name: str, source: list[dict[str, Any]]) -> float | None:
        values = [row["generation_metrics"].get(name) for row in source if row["generation_metrics"].get(name) is not None]
        return round(mean([float(value) for value in values]), 6) if values else None

    latencies = [
        row.get("answer", {}).get("latency", {}).get("total_latency_ms")
        for row in completed
        if row.get("answer", {}).get("latency")
    ]
    usage = [row.get("answer", {}).get("model_usage", {}) for row in completed]
    costs = [usage_item.get("estimated_cost_usd") for usage_item in usage]
    return {
        "question_count": len(rows),
        "completed": len(completed),
        "failed": len(failed),
        "required_claim_coverage": avg("required_claim_coverage", answerable),
        "supported_claim_ratio": avg("supported_claim_ratio", answerable),
        "citation_precision": avg("citation_precision", answerable),
        "citation_recall": avg("citation_recall", answerable),
        "answer_completeness": avg("answer_completeness", answerable),
        "abstention_accuracy": avg("abstention_accuracy", unanswerable),
        "unsupported_claim_count": sum(row["generation_metrics"].get("unsupported_claim_count") or 0 for row in completed),
        "input_tokens": sum(int(item.get("input_tokens") or 0) for item in usage),
        "output_tokens": sum(int(item.get("output_tokens") or 0) for item in usage),
        "total_tokens": sum(int(item.get("total_tokens") or 0) for item in usage),
        "cost": round(sum(float(value) for value in costs if value is not None), 8) if any(value is not None for value in costs) else None,
        "provider_request_count": sum(int(row.get("api_request_count") or 0) for row in rows),
        "provider_failure_count": len(failed),
        "latency_ms": {
            "mean": round(mean([float(value) for value in latencies]), 3) if latencies else None,
            "p50": percentile([float(value) for value in latencies], 0.5),
            "p95": percentile([float(value) for value in latencies], 0.95),
        },
    }


def grouped_generation(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(field))].append(row)
    return {key: aggregate_generation_rows(value) for key, value in sorted(groups.items())}


def classify_generation_bad_case(row: dict[str, Any]) -> str | None:
    if row.get("status") == "FAILED":
        return "PROVIDER_FAILURE"
    metrics = row.get("generation_metrics", {})
    if not row["gold"].get("answerable") and metrics.get("abstention_accuracy") != 1.0:
        return "ABSTENTION_FAILURE"
    if row["gold"].get("answerable") and not metrics.get("gold_evidence_in_context"):
        return "RETRIEVAL_ROOTED"
    if row["gold"].get("answerable") and (metrics.get("required_claim_coverage") or 0) < 1:
        return "GENERATION_OMISSION"
    if row["gold"].get("answerable") and (metrics.get("citation_precision") or 0) < 1:
        return "CITATION_MISMATCH"
    if row["gold"].get("answerable") and (metrics.get("unsupported_claim_count") or 0) > 0:
        return "UNSUPPORTED_CLAIM"
    return None


def generation_bad_cases(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    items = []
    for row in rows:
        case = classify_generation_bad_case(row)
        if not case:
            continue
        counts[case] += 1
        items.append(
            {
                "question_id": row["question_id"],
                "split": row.get("split"),
                "category": row.get("category"),
                "difficulty": row.get("difficulty"),
                "primary_bad_case": case,
                "diagnosis": _generation_diagnosis(case),
            }
        )
    return {"distribution": dict(sorted(counts.items())), "items": items}


def _generation_diagnosis(case: str) -> str:
    return {
        "RETRIEVAL_ROOTED": "Gold evidence did not enter the actual generation context.",
        "GENERATION_OMISSION": "Gold evidence was available but required claims were omitted or weakly covered.",
        "CITATION_MISMATCH": "Generated citations did not exactly match Gold paper/page/block evidence.",
        "UNSUPPORTED_CLAIM": "At least one generated claim lacked exact Gold citation support.",
        "ABSTENTION_FAILURE": "Unanswerable item was not refused cleanly.",
        "PROVIDER_FAILURE": "Provider or API call failed for this item.",
    }.get(case, "Unclassified generation issue.")


def failure_attribution(retrieval_cases: dict[str, Any], generation_cases: dict[str, Any]) -> dict[str, Any]:
    counts = Counter()
    counts.update(retrieval_cases.get("distribution", {}))
    counts.update(generation_cases.get("distribution", {}))
    total = sum(counts.values())
    return {
        "total_failed_or_degraded_cases": total,
        "distribution": dict(sorted(counts.items())),
        "proportions": {
            key: round(value / total, 6) if total else 0 for key, value in sorted(counts.items())
        },
        "largest_bottleneck": counts.most_common(1)[0][0] if counts else None,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    tmp.replace(target)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json_artifact(path: Path, payload: dict[str, Any]) -> None:
    write_json(path, payload)
