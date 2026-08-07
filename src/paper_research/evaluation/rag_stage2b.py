from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from pydantic import BaseModel, Field, model_validator

from paper_research.evaluation.rag_official_baseline import aggregate_retrieval_rows
from paper_research.evaluation.rag_stage2a import (
    load_baseline_generation_rows,
    load_dev_gold,
    summarize_context_selection_from_generation,
)

STAGE2B_CACHE = Path(".runtime/rag-stage2b-query-rewrites")
SYSTEM_PROMPT = (
    "You transform user questions into retrieval queries for an existing paper corpus. "
    "Do not answer the question. Do not generate citations. Do not add facts, numbers, "
    "entities, models, datasets, metrics, or claims that are not present in the question. "
    "Preserve method names, paper names, model names, dataset names, metrics, numbers, "
    "acronyms, comparison targets, and negation."
)
SINGLE_REWRITE_PROMPT_VERSION = "stage2b-single-rewrite-v1"
DECOMPOSITION_PROMPT_VERSION = "stage2b-decomposition-v1"


class SingleRewrite(BaseModel):
    rewritten_query: str = Field(min_length=1)


class DecomposedQuery(BaseModel):
    query: str = Field(min_length=1)
    purpose: str = Field(min_length=1)


class Decomposition(BaseModel):
    queries: list[DecomposedQuery] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_query_limit(self) -> Decomposition:
        if len(self.queries) > 3:
            raise ValueError("decomposition may generate at most 3 queries")
        return self


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def single_rewrite_user_prompt(question: str) -> str:
    return (
        "Return JSON only with shape {\"rewritten_query\":\"...\"}.\n"
        "Create one standalone, specific, retrieval-oriented query for this question.\n"
        "The output must be a search query, not an answer.\n"
        f"Question:\n{question}"
    )


def decomposition_user_prompt(question: str) -> str:
    return (
        "Return JSON only with shape {\"queries\":[{\"query\":\"...\",\"purpose\":\"...\"}]}.\n"
        "Generate 0 to 3 focused retrieval queries. Use [] if the original question is already "
        "sufficient. Queries must stay grounded in the question text and must not answer it.\n"
        f"Question:\n{question}"
    )


def assert_no_gold_leakage(prompt: str, gold: dict[str, Any]) -> None:
    forbidden_values: list[str] = []
    for key in ("gold_answer", "gold_block_ids", "gold_pages", "required_claims"):
        value = gold.get(key)
        if isinstance(value, str):
            forbidden_values.append(value)
        elif isinstance(value, list):
            forbidden_values.extend(str(item) for item in value if not isinstance(item, dict))
            forbidden_values.extend(
                str(item.get("text", "")) for item in value if isinstance(item, dict)
            )
    source_question = str(gold.get("question") or gold.get("original_question") or "")
    for value in forbidden_values:
        if not value:
            continue
        if len(value) < 4 and value not in {"yes", "no"}:
            continue
        if value in source_question:
            continue
        if value in prompt:
            raise ValueError(f"rewrite prompt leaks gold field value: {value[:40]}")


def drift_labels(original: str, queries: list[str]) -> list[str]:
    combined = " ".join(queries).lower()
    if not queries:
        return ["NONE"]
    labels: list[str] = []
    original_lower = original.lower()
    numbers = re.findall(r"\b\d+(?:\.\d+)?\b", original)
    if any(number not in combined for number in numbers):
        labels.append("NUMERIC_CONSTRAINT_DROPPED")
    if any(term in original_lower and term not in combined for term in ("not", "no ", "without")):
        labels.append("NEGATION_DROPPED")
    comparison_terms = [" and ", " versus ", " vs ", " differ", "compare"]
    if any(term in original_lower for term in comparison_terms):
        candidates = _entity_like_terms(original)
        missing = [term for term in candidates if term.lower() not in combined]
        if missing:
            labels.append("COMPARISON_TARGET_DROPPED")
    entity_terms = _entity_like_terms(original)
    present_entities = sum(term.lower() in combined for term in entity_terms)
    if entity_terms and present_entities < max(1, len(entity_terms) // 2):
        labels.append("ENTITY_DROPPED")
    if len(combined.split()) < max(3, len(original_lower.split()) // 4):
        labels.append("OVER_BROADENED")
    return sorted(set(labels)) or ["NONE"]


def _entity_like_terms(text: str) -> list[str]:
    terms = re.findall(r"\b[A-Z][A-Za-z0-9.-]*(?:[-/][A-ZA-Za-z0-9.-]+)?\b", text)
    acronyms = re.findall(r"\b[A-Z]{2,}(?:-[A-Z0-9]+)?\b", text)
    return sorted(set(terms + acronyms))


def cache_key(question_id: str, prompt_version: str, prompt_digest: str, model: str) -> str:
    raw = f"{question_id}|{prompt_version}|{prompt_digest}|{model}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def read_cache(kind: str, key: str) -> dict[str, Any] | None:
    path = STAGE2B_CACHE / kind / f"{key}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_cache(kind: str, key: str, payload: dict[str, Any]) -> None:
    path = STAGE2B_CACHE / kind / f"{key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def queries_for_config(
    config_id: str,
    original_query: str,
    rewrite_item: dict[str, Any] | None = None,
) -> tuple[list[str], bool]:
    """Return retrieval queries and whether provider failure fallback occurred."""
    item = rewrite_item or {}
    if config_id == "Q0_CURRENT_HYBRID":
        return [original_query], False
    if config_id == "Q1_SINGLE_REWRITE_REPLACE":
        if item.get("single_status") != "success" or not item.get("single_rewrite"):
            return [], True
        return [str(item["single_rewrite"])], False
    if config_id == "Q2_ORIGINAL_PLUS_SINGLE_REWRITE":
        if item.get("single_status") != "success" or not item.get("single_rewrite"):
            return [original_query], True
        return deduplicate_queries([original_query, str(item["single_rewrite"])]), False
    if config_id == "Q3_ORIGINAL_PLUS_DECOMPOSITION":
        generated = [str(query) for query in item.get("decomposition_queries", [])]
        if item.get("decomposition_status") != "success":
            return [original_query], True
        return deduplicate_queries([original_query, *generated])[:4], False
    raise ValueError(f"unknown Stage 2B config: {config_id}")


def deduplicate_queries(queries: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped = []
    for query in queries:
        normalized = " ".join(query.split()).casefold()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(query)
    return deduped


def fusion_key(item: dict[str, Any]) -> str:
    paper_id = str(item.get("paper_id") or "")
    block_ids = [str(block_id) for block_id in item.get("block_ids", [])]
    if not block_ids:
        block_ids = [str(item.get("chunk_id") or "")]
    return f"{paper_id}|{'/'.join(block_ids)}"


def fuse_ranked_contexts(
    ranked_lists: list[list[dict[str, Any]]],
    *,
    rrf_k: int = 60,
    top_k: int = 20,
) -> list[dict[str, Any]]:
    """Deterministic RRF over candidate sets returned by multiple rewritten queries."""
    scores: dict[str, float] = {}
    best_item: dict[str, dict[str, Any]] = {}
    first_seen: dict[str, tuple[int, int]] = {}
    for query_index, ranked in enumerate(ranked_lists):
        for rank, item in enumerate(ranked, start=1):
            key = fusion_key(item)
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
            if key not in best_item or float(item.get("score") or 0.0) > float(
                best_item[key].get("score") or 0.0
            ):
                best_item[key] = dict(item)
            first_seen.setdefault(key, (query_index, rank))
    ordered = sorted(
        scores,
        key=lambda key: (-scores[key], first_seen[key][0], first_seen[key][1], key),
    )
    fused = []
    for rank, key in enumerate(ordered[:top_k], start=1):
        item = dict(best_item[key])
        item["score"] = round(scores[key], 12)
        item["rank"] = rank
        item["multi_query_fusion_key"] = key
        fused.append(item)
    return fused


def add_extended_row_metrics(row: dict[str, Any], gold: dict[str, Any]) -> dict[str, Any]:
    metrics = dict(row.get("metrics") or {})
    if row.get("answerable") and not row.get("retrieval_failure"):
        for k in (5, 10, 20):
            retrieved = {
                block
                for item in row.get("ranked_results", [])[:k]
                for block in item.get("block_ids", [])
            }
            claim_total = 0
            claim_covered = 0
            for claim in gold.get("required_claims", []):
                claim_blocks = set(claim.get("gold_block_ids") or gold.get("gold_block_ids") or [])
                if not claim_blocks:
                    continue
                claim_total += 1
                if retrieved & claim_blocks:
                    claim_covered += 1
            evidence_coverage = metrics.get(f"evidence_coverage_at_{k}") or 0.0
            metrics[f"required_claim_evidence_coverage_at_{k}"] = (
                round(claim_covered / claim_total, 6) if claim_total else 0.0
            )
            metrics[f"full_evidence_coverage_at_{k}"] = 1.0 if evidence_coverage >= 1 else 0.0
    row["metrics"] = metrics
    return row


def query_signature(items: list[dict[str, Any]]) -> dict[str, float | int]:
    counts = [1 + len(item.get("decomposition_queries", [])) for item in items]
    generated_counts = [count - 1 for count in counts]
    return {
        "average_generated_queries": round(mean(generated_counts), 6) if counts else 0.0,
        "queries_p50": _percentile(counts, 0.5),
        "queries_p95": _percentile(counts, 0.95),
    }


def _percentile(values: list[int | float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return round(ordered[index], 6)


def aggregate_rewrite_usage(items: list[dict[str, Any]]) -> dict[str, Any]:
    success = [item for item in items if item.get("single_status") == "success"]
    decomp_success = [item for item in items if item.get("decomposition_status") == "success"]
    latencies = [
        float(value)
        for item in items
        for value in (item.get("single_latency_ms"), item.get("decomposition_latency_ms"))
        if value is not None
    ]
    historical_provider_requests = sum(
        int(item.get("single_cache_historical_provider_requests") or 0)
        + int(item.get("decomposition_cache_historical_provider_requests") or 0)
        for item in items
    )
    current_provider_requests = sum(
        int(item.get("single_provider_requests") or 0)
        + int(item.get("decomposition_provider_requests") or 0)
        for item in items
    )
    usage_sources = sorted({str(item.get("usage_source") or "unknown") for item in items})
    estimated_cost = round(
        sum(float(item.get("estimated_cost_usd") or 0.0) for item in items), 8
    )
    cost_status = (
        "unavailable_cache_schema_gap"
        if "cache_text_estimated_after_interrupted_provider_run" in usage_sources
        and historical_provider_requests
        else "provider_reported"
    )
    return {
        "logical_questions": len(items),
        "single_rewrite_requests": sum(
            int(item.get("single_provider_requests") or 0) for item in items
        ),
        "decomposition_requests": sum(
            int(item.get("decomposition_provider_requests") or 0) for item in items
        ),
        "provider_requests": current_provider_requests,
        "historical_provider_requests_from_cache": historical_provider_requests,
        "effective_provider_requests_for_artifact": (
            current_provider_requests + historical_provider_requests
        ),
        "provider_failures": sum(
            1
            for item in items
            if item.get("single_status") == "failed"
            or item.get("decomposition_status") == "failed"
        ),
        "rewrite_success_rate": round(len(success) / len(items), 6) if items else 0.0,
        "decomposition_success_rate": round(len(decomp_success) / len(items), 6) if items else 0.0,
        "input_tokens": sum(int(item.get("input_tokens") or 0) for item in items),
        "output_tokens": sum(int(item.get("output_tokens") or 0) for item in items),
        "total_tokens": sum(int(item.get("total_tokens") or 0) for item in items),
        "estimated_cost": estimated_cost,
        "cost_accounting_status": cost_status,
        "usage_sources": usage_sources,
        "rewrite_latency_p50": _percentile(latencies, 0.5),
        "rewrite_latency_p95": _percentile(latencies, 0.95),
        **query_signature(items),
    }


def full_evidence_coverage(rows: list[dict[str, Any]], k: int) -> float:
    answerable = [row for row in rows if row.get("answerable") and not row.get("retrieval_failure")]
    if not answerable:
        return 0.0
    return round(
        sum(1 for row in answerable if (row["metrics"].get(f"evidence_coverage_at_{k}") or 0) >= 1)
        / len(answerable),
        6,
    )


def required_claim_coverage(rows: list[dict[str, Any]], k: int) -> float:
    gold = {row["question_id"]: row for row in load_dev_gold()}
    total = 0
    covered = 0
    for row in rows:
        if not row.get("answerable") or row.get("retrieval_failure"):
            continue
        retrieved = {
            block
            for item in row.get("ranked_results", [])[:k]
            for block in item.get("block_ids", [])
        }
        for claim in gold[row["question_id"]].get("required_claims", []):
            blocks = set(
                claim.get("gold_block_ids")
                or gold[row["question_id"]].get("gold_block_ids")
                or []
            )
            if not blocks:
                continue
            total += 1
            if retrieved & blocks:
                covered += 1
    return round(covered / total, 6) if total else 0.0


def extended_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = aggregate_retrieval_rows(rows)
    metrics["required_claim_evidence_coverage_at_5"] = required_claim_coverage(rows, 5)
    metrics["required_claim_evidence_coverage_at_10"] = required_claim_coverage(rows, 10)
    metrics["required_claim_evidence_coverage_at_20"] = required_claim_coverage(rows, 20)
    metrics["full_evidence_coverage_at_10"] = full_evidence_coverage(rows, 10)
    metrics["full_evidence_coverage_at_20"] = full_evidence_coverage(rows, 20)
    return metrics


def rewrite_specific_metrics(
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    rewrite_items: list[dict[str, Any]],
) -> dict[str, Any]:
    base_cases = {row["question_id"]: _bad_case(row) for row in baseline_rows}
    cand = {row["question_id"]: row for row in candidate_rows}
    miss_ids = [qid for qid, case in base_cases.items() if case == "RETRIEVAL_MISS"]
    partial_ids = [qid for qid, case in base_cases.items() if case == "PARTIAL_EVIDENCE_RETRIEVAL"]
    recovered = [
        qid
        for qid in miss_ids
        if (cand[qid]["metrics"].get("evidence_coverage_at_20") or 0) > 0
    ]
    partial_counts = Counter()
    for qid in partial_ids:
        before = next(row for row in baseline_rows if row["question_id"] == qid)
        before_cov = before["metrics"].get("evidence_coverage_at_20") or 0
        after_cov = cand[qid]["metrics"].get("evidence_coverage_at_20") or 0
        if after_cov >= 1:
            partial_counts["partial_to_full"] += 1
        elif after_cov > before_cov:
            partial_counts["improved_still_partial"] += 1
        elif after_cov == before_cov:
            partial_counts["unchanged"] += 1
        else:
            partial_counts["worse"] += 1
    regressions = [
        row["question_id"]
        for row in candidate_rows
        if row.get("answerable")
        and (row["metrics"].get("evidence_coverage_at_20") or 0)
        < (
            next(base for base in baseline_rows if base["question_id"] == row["question_id"])[
                "metrics"
            ].get("evidence_coverage_at_20")
            or 0
        )
    ]
    drift_counts = Counter(
        label for item in rewrite_items for label in item.get("drift_labels", [])
    )
    return {
        "miss_recovery": {
            "baseline_miss_count": len(miss_ids),
            "recovered": len(recovered),
            "still_missing": len(miss_ids) - len(recovered),
            "miss_recovery_rate": round(len(recovered) / len(miss_ids), 6) if miss_ids else 0.0,
        },
        "partial_evidence_completion": dict(sorted(partial_counts.items())),
        "new_miss_rate": round(len(regressions) / max(1, len(candidate_rows)), 6),
        "regression_rate": round(len(regressions) / max(1, len(candidate_rows)), 6),
        "rewrite_drift": dict(sorted(drift_counts.items())),
    }


def _bad_case(row: dict[str, Any]) -> str | None:
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


def selection_gate(q0: dict[str, Any], candidate: dict[str, Any], specific: dict[str, Any]) -> bool:
    gains = [
        (candidate.get("recall_at_10") or 0) - (q0.get("recall_at_10") or 0),
        (candidate.get("evidence_coverage_at_10") or 0)
        - (q0.get("evidence_coverage_at_10") or 0),
        (candidate.get("required_claim_evidence_coverage_at_10") or 0)
        - (q0.get("required_claim_evidence_coverage_at_10") or 0),
    ]
    return (
        max(gains) >= 0.05
        and specific.get("new_miss_rate", 1) <= 0.05
        and (candidate.get("paper_recall_at_10") or 0) >= (q0.get("paper_recall_at_10") or 0) - 0.03
    )


def generation_utilization_diagnosis() -> dict[str, Any]:
    return summarize_context_selection_from_generation(load_baseline_generation_rows())


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
