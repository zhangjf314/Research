from __future__ import annotations

# ruff: noqa: E501
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

BAD_CASE_TYPES = {
    "RETRIEVAL_MISS",
    "RANKING_ERROR",
    "QUERY_UNDERSTANDING",
    "CHUNKING_ERROR",
    "CONTEXT_SELECTION",
    "GENERATION_OMISSION",
    "UNSUPPORTED_CLAIM",
    "CITATION_MISMATCH",
    "ABSTENTION_FAILURE",
    "METADATA_OR_PARSE_ERROR",
    "OTHER",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def canonical_json_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_hash(records: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: item.get("question_id", "")):
        digest.update(
            json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        )
        digest.update(b"\n")
    return digest.hexdigest()


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def first_relevant_rank(relevance: list[bool], k: int | None = None) -> int | None:
    window = relevance[:k] if k else relevance
    return next((index for index, value in enumerate(window, 1) if value), None)


def recall_at(retrieved: set[str], gold: set[str]) -> float:
    if not gold:
        return 0.0
    return len(retrieved & gold) / len(gold)


def ndcg_at(relevance: list[bool], ideal_count: int, k: int) -> float:
    dcg = sum((1.0 if value else 0.0) / math.log2(index + 2) for index, value in enumerate(relevance[:k]))
    ideal = sum(1.0 / math.log2(index + 2) for index in range(min(ideal_count, k)))
    return dcg / ideal if ideal else 0.0


def normalize_retrieved_blocks(row: dict[str, Any]) -> set[str]:
    if "block_id" in row and row["block_id"]:
        return {str(row["block_id"])}
    return {str(item) for item in row.get("block_ids", []) if item}


def evaluate_retrieval_question(gold: dict[str, Any], ranked_results: list[dict[str, Any]]) -> dict[str, Any]:
    gold_blocks = {str(item) for item in gold.get("gold_block_ids", [])}
    gold_papers = {str(item) for item in gold.get("gold_paper_ids", [])}
    answerable = bool(gold.get("answerable", True)) and bool(gold_blocks)
    result_blocks_by_rank = [normalize_retrieved_blocks(row) for row in ranked_results]
    result_papers_by_rank = [{str(row.get("paper_id", ""))} for row in ranked_results]
    block_relevance = [bool(blocks & gold_blocks) for blocks in result_blocks_by_rank]
    if not answerable:
        return {
            "question_id": gold["question_id"],
            "answerable": False,
            "returned_count": len(ranked_results),
            "irrelevant_retrieval": bool(ranked_results),
            "irrelevant_retrieval_rate": 1.0 if ranked_results else 0.0,
        }

    metrics: dict[str, Any] = {
        "question_id": gold["question_id"],
        "answerable": True,
        "returned_count": len(ranked_results),
    }
    for k in (5, 10, 20):
        retrieved_blocks = set().union(*result_blocks_by_rank[:k]) if result_blocks_by_rank[:k] else set()
        retrieved_papers = set().union(*result_papers_by_rank[:k]) if result_papers_by_rank[:k] else set()
        metrics[f"recall_at_{k}"] = recall_at(retrieved_blocks, gold_blocks)
        metrics[f"paper_recall_at_{k}"] = recall_at(retrieved_papers, gold_papers)
        metrics[f"evidence_coverage_at_{k}"] = recall_at(retrieved_blocks, gold_blocks)
    first_block_rank = first_relevant_rank(block_relevance, 10)
    metrics["mrr_at_10"] = 1 / first_block_rank if first_block_rank else 0.0
    metrics["ndcg_at_10"] = ndcg_at(block_relevance, len(gold_blocks), 10)
    return metrics


def aggregate_retrieval(per_question: list[dict[str, Any]]) -> dict[str, Any]:
    answerable_rows = [row for row in per_question if row.get("answerable")]
    unanswerable_rows = [row for row in per_question if not row.get("answerable")]
    metric_names = [
        "recall_at_5",
        "recall_at_10",
        "recall_at_20",
        "mrr_at_10",
        "ndcg_at_10",
        "paper_recall_at_5",
        "paper_recall_at_10",
        "evidence_coverage_at_5",
        "evidence_coverage_at_10",
        "evidence_coverage_at_20",
    ]
    aggregate = {
        "question_count": len(per_question),
        "answerable_count": len(answerable_rows),
        "unanswerable_count": len(unanswerable_rows),
    }
    for name in metric_names:
        aggregate[name] = round(mean([float(row.get(name, 0.0)) for row in answerable_rows]), 6)
    aggregate["irrelevant_retrieval_rate"] = round(
        mean([float(row.get("irrelevant_retrieval_rate", 0.0)) for row in unanswerable_rows]),
        6,
    )
    return aggregate


def stratify(rows: list[dict[str, Any]], gold_by_id: dict[str, dict[str, Any]], field: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = str(gold_by_id[row["question_id"]].get(field, "unknown"))
        grouped[key].append(row)
    return {key: aggregate_retrieval(value) for key, value in sorted(grouped.items())}


def audit_gold(records: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts = Counter(str(row.get("review_status", "missing")) for row in records)
    approved = [row for row in records if row.get("review_status") == "approved"]
    answerable = [row for row in approved if row.get("answerable") is True]
    unanswerable = [row for row in approved if row.get("answerable") is False]
    required_claim_counts = [len(row.get("required_claims", [])) for row in answerable]
    gold_paper_counts = [len(row.get("gold_paper_ids", [])) for row in answerable]
    gold_page_counts = [len(row.get("gold_pages", [])) for row in answerable]
    gold_block_counts = [len(row.get("gold_block_ids", [])) for row in answerable]
    target = 150
    category_counts = Counter(str(row.get("category", "unknown")) for row in approved)
    target_categories = [
        "single-hop factual",
        "multi-evidence synthesis",
        "cross-paper comparison",
        "methods / experiments",
        "limitations / research gaps",
        "unanswerable",
    ]
    return {
        "total": len(records),
        "approved": len(approved),
        "review_status_distribution": dict(sorted(status_counts.items())),
        "answerable": len(answerable),
        "unanswerable": len(unanswerable),
        "category_distribution": dict(sorted(category_counts.items())),
        "difficulty_distribution": dict(
            sorted(Counter(str(row.get("difficulty", "unknown")) for row in approved).items())
        ),
        "required_claim_coverage": {
            "questions_with_required_claims": sum(1 for value in required_claim_counts if value > 0),
            "total_required_claims": sum(required_claim_counts),
            "mean_required_claims_per_answerable": round(mean([float(v) for v in required_claim_counts]), 6),
        },
        "gold_paper_coverage": {
            "questions_with_gold_paper": sum(1 for value in gold_paper_counts if value > 0),
            "unique_gold_papers": len({paper for row in answerable for paper in row.get("gold_paper_ids", [])}),
        },
        "gold_page_coverage": {
            "questions_with_gold_pages": sum(1 for value in gold_page_counts if value > 0),
            "total_gold_page_refs": sum(gold_page_counts),
        },
        "gold_evidence_coverage": {
            "answerable_questions_with_gold_blocks": sum(1 for value in gold_block_counts if value > 0),
            "total_gold_block_refs": sum(gold_block_counts),
            "complete_for_answerable": all(value > 0 for value in gold_block_counts),
        },
        "gap_plan": {
            "recommended_target_gold_count": target,
            "current_approved_count": len(approved),
            "questions_to_add": max(0, target - len(approved)),
            "target_question_types": target_categories,
            "manual_review_required": True,
            "do_not_bulk_generate_with_llm": True,
        },
    }


def evaluate_generation_item(item: dict[str, Any]) -> dict[str, Any]:
    metrics = item.get("metrics", {})
    answer = item.get("answer", {})
    gold = item.get("gold", {})
    answerable = bool(gold.get("answerable", True))
    retrieval_succeeded = bool(item.get("gold_block_present", False)) or not answerable
    generation_succeeded = item.get("status") == "COMPLETED"
    if not retrieval_succeeded:
        failure_stage = "retrieval failed"
    elif not generation_succeeded:
        failure_stage = "retrieval succeeded but generation failed"
    else:
        failure_stage = "completed"
    return {
        "question_id": item["question_id"],
        "answerable": answerable,
        "status": item.get("status"),
        "retrieval_succeeded": retrieval_succeeded,
        "generation_succeeded": generation_succeeded,
        "failure_stage": failure_stage,
        "required_claim_coverage": float(metrics.get("required_claim_coverage", 0.0)),
        "supported_claim_ratio": float(metrics.get("claim_citation_binding_rate", 0.0)),
        "citation_precision": float(metrics.get("citation_precision", 0.0)),
        "citation_recall": float(metrics.get("citation_recall", 0.0)),
        "answer_completeness": float(metrics.get("required_claim_coverage", 0.0)),
        "abstention_correct": (not answerable and not bool(answer.get("answerable", True))),
        "latency_ms": float(item.get("wall_ms") or answer.get("latency", {}).get("total_latency_ms") or 0.0),
        "input_tokens": int(answer.get("model_usage", {}).get("input_tokens", 0)),
        "output_tokens": int(answer.get("model_usage", {}).get("output_tokens", 0)),
        "total_tokens": int(answer.get("model_usage", {}).get("total_tokens", 0)),
        "cost": float(answer.get("model_usage", {}).get("estimated_cost_usd", 0.0)),
    }


def aggregate_generation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    answerable_rows = [row for row in rows if row["answerable"]]
    unanswerable_rows = [row for row in rows if not row["answerable"]]
    return {
        "question_count": len(rows),
        "answerable_count": len(answerable_rows),
        "unanswerable_count": len(unanswerable_rows),
        "retrieval_failed_count": sum(1 for row in rows if row["failure_stage"] == "retrieval failed"),
        "generation_failed_after_retrieval_count": sum(
            1 for row in rows if row["failure_stage"] == "retrieval succeeded but generation failed"
        ),
        "required_claim_coverage": round(mean([row["required_claim_coverage"] for row in answerable_rows]), 6),
        "supported_claim_ratio": round(mean([row["supported_claim_ratio"] for row in answerable_rows]), 6),
        "citation_precision": round(mean([row["citation_precision"] for row in answerable_rows]), 6),
        "citation_recall": round(mean([row["citation_recall"] for row in answerable_rows]), 6),
        "answer_completeness": round(mean([row["answer_completeness"] for row in answerable_rows]), 6),
        "abstention_accuracy": round(mean([float(row["abstention_correct"]) for row in unanswerable_rows]), 6),
        "latency_ms_mean": round(mean([row["latency_ms"] for row in rows]), 6),
        "input_tokens": sum(row["input_tokens"] for row in rows),
        "output_tokens": sum(row["output_tokens"] for row in rows),
        "total_tokens": sum(row["total_tokens"] for row in rows),
        "cost": round(sum(row["cost"] for row in rows), 8),
    }


def classify_bad_case(item: dict[str, Any]) -> tuple[str, str]:
    metrics = item.get("metrics", {})
    answer = item.get("answer", {})
    gold = item.get("gold", {})
    if gold.get("answerable") is False and answer.get("answerable") is not False:
        return "generation", "ABSTENTION_FAILURE"
    if item.get("status") != "COMPLETED":
        return "generation", "OTHER"
    if not item.get("gold_block_present", False) and gold.get("answerable") is not False:
        return "retrieval", "RETRIEVAL_MISS"
    if float(metrics.get("required_claim_coverage", 0.0)) < 1.0:
        return "generation", "GENERATION_OMISSION"
    if float(metrics.get("citation_precision", 1.0)) < 1.0:
        return "generation", "CITATION_MISMATCH"
    return "none", "OTHER"
