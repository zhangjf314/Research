# ruff: noqa: E501
"""Shared, deterministic Stage 13.2 Dev QA protocol helpers."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/evaluation"
DOCS = ROOT / "docs"
RUN_ROOT = DATA / "evidence-qa-dev-v1" / "runs"
MANIFEST = DATA / "evidence-qa-dev-manifest-v1.json"
MANIFEST_DOC = DOCS / "evidence-qa-dev-manifest-v1.md"
SUMMARY_JSON = DATA / "evidence-qa-dev-v1.json"
SUMMARY_CSV = DATA / "evidence-qa-dev-v1.csv"
SUMMARY_DOC = DOCS / "evidence-qa-dev-v1.md"
AUDIT_JSONL = DATA / "evidence-qa-dev-citation-audit-v1.jsonl"
AUDIT_DOC = DOCS / "evidence-qa-dev-citation-audit-v1.md"
FINAL_AUDIT = DATA / "evidence-qa-dev-final-audit-v1.json"

# Frozen before any Stage 13.2 live result existed. The set covers every Phase B
# gain, all requested difficulty bands, eight categories, one unanswerable item,
# one multi-paper item, baseline-hit controls, and persistent retrieval misses.
DEV_IDS = ["q001", "q002", "q004", "q005", "q007", "q008", "q013", "q015", "q019", "q050"]
GAIN_IDS = {"q002", "q007", "q013", "q050"}
VARIANT_B = "retrieval_only"
VARIANT_C = "evidence_centric"
BLOCKED_C_REASON = "blocked_by_known_selector_defect"
CLAIM_MATCH_THRESHOLD = 0.35


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    body = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def phase_b_rows() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    payload = json.loads((DATA / "evidence-retrieval-phase-b-ablation-v1.json").read_text(encoding="utf-8"))
    variants = {item["name"]: item for item in payload["variants"]}
    baseline = {row["question_id"]: row for row in variants["stage13_routed_baseline"]["queries"]}
    candidate = {
        row["question_id"]: row
        for row in variants["phase_b_adjacent_same_page_completion"]["queries"]
    }
    return baseline, candidate


def build_manifest() -> dict[str, Any]:
    gold = {row["question_id"]: row for row in read_jsonl(DATA / "gold-set-v1.jsonl")}
    protocol = {row["question_id"]: row for row in read_jsonl(DATA / "retrieval-gold-v2.jsonl")}
    baseline, candidate = phase_b_rows()
    rows = []
    for question_id in DEV_IDS:
        item = gold[question_id]
        retrieval = protocol[question_id]
        reasons = []
        if question_id in GAIN_IDS:
            reasons.append("phase_b_exact_hit_gain")
        if baseline[question_id]["metrics"].get("exact_gold_block_available") is True:
            reasons.append("baseline_already_hit_control")
        if (
            item["answerable"]
            and candidate[question_id]["metrics"].get("exact_gold_block_available") is False
        ):
            reasons.append("persistent_retrieval_miss")
        if not item["answerable"]:
            reasons.append("unanswerable_refusal")
        if retrieval["retrieval_scope"] == "multi_paper":
            reasons.append("multi_paper_coverage")
        rows.append(
            {
                "question_id": question_id,
                "category": item["category"],
                "difficulty": item["difficulty"],
                "answerable": item["answerable"],
                "paper_scope": retrieval["retrieval_scope"],
                "multi_paper": retrieval["retrieval_scope"] == "multi_paper",
                "baseline_retrieval_status": baseline[question_id]["metrics"].get(
                    "exact_gold_block_available"
                ),
                "phase_b_retrieval_status": candidate[question_id]["metrics"].get(
                    "exact_gold_block_available"
                ),
                "inclusion_reason": reasons,
            }
        )
    body = {
        "schema_version": "evidence-qa-dev-manifest-v1",
        "selection_status": "frozen_before_live_stage13_2_results",
        "selection_protocol": "deterministic_stratified_stage13_dev10_v1",
        "question_count": 10,
        "question_ids": DEV_IDS,
        "questions": rows,
        "artifacts": {
            "gold_set": "gold-set-v1",
            "gold_set_sha256": sha256(DATA / "gold-set-v1.jsonl"),
            "retrieval_gold": "retrieval-gold-v2",
            "retrieval_gold_sha256": sha256(DATA / "retrieval-gold-v2.jsonl"),
            "production_corpus": "production-corpus-v1",
            "production_corpus_sha256": sha256(DATA / "production-corpus-v1.json"),
            "evidence_corpus_signature": json.loads(
                (DATA / "stage13-baseline-freeze-v1.json").read_text(encoding="utf-8")
            )["corpus_signature"],
            "phase_b_ablation_sha256": sha256(
                DATA / "evidence-retrieval-phase-b-ablation-v1.json"
            ),
            "claim_evidence_pilot_sha256": sha256(
                DATA / "claim-evidence-gold-pilot-v1.jsonl"
            ),
        },
        "variants": {
            "historical_stage11c": {
                "source": "historical_stage11c",
                "prompt_version": "qa-production-v1",
                "live_requests": 0,
            },
            VARIANT_B: {
                "retrieval": "phase_b_adjacent_same_page_completion",
                "prompt_version": "qa-production-v1",
                "context_version": "phase-b-adjacent-same-page-v1",
            },
            VARIANT_C: {
                "status": BLOCKED_C_REASON,
                "reason": "Phase B did not repair the known per-claim minimal selector compression defect.",
                "live_requests": 0,
            },
        },
        "budgets": {
            "per_question_iterations": 1,
            "per_question_primary_requests": 1,
            "per_question_citation_retries": 1,
            "per_question_total_attempts": 2,
            "per_question_tokens": 20000,
            "per_question_elapsed_seconds": 180,
            "dev_request_attempts": 40,
            "dev_tokens": 300000,
            "dev_elapsed_seconds": 1800,
            "monetary_cost_usd": "0",
        },
    }
    body["manifest_hash"] = canonical_hash(body)
    return body


def write_manifest() -> dict[str, Any]:
    body = build_manifest()
    if MANIFEST.exists():
        existing = json.loads(MANIFEST.read_text(encoding="utf-8"))
        if existing != body:
            raise RuntimeError("frozen Dev manifest differs from deterministic protocol")
    else:
        MANIFEST.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Evidence-Centric Dev Manifest v1",
        "",
        f"- Manifest hash: `{body['manifest_hash']}`",
        "- Frozen before Stage 13.2 live results: **yes**",
        "- Result-dependent resampling: **no**",
        "- Historical baseline source: `historical_stage11c`",
        f"- Evidence-centric variant C: `{BLOCKED_C_REASON}`",
        "",
        "| ID | Category | Difficulty | Scope | Answerable | Baseline exact | Phase B exact | Inclusion |",
        "|---|---|---|---|---:|---:|---:|---|",
    ]
    for row in body["questions"]:
        lines.append(
            f"| {row['question_id']} | {row['category']} | {row['difficulty']} | "
            f"{row['paper_scope']} | {row['answerable']} | {row['baseline_retrieval_status']} | "
            f"{row['phase_b_retrieval_status']} | {', '.join(row['inclusion_reason'])} |"
        )
    MANIFEST_DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return body


def terms(text: str) -> set[str]:
    return {
        token.lower().strip(".,:;!?()[]{}'\"")
        for token in text.split()
        if token.strip(".,:;!?()[]{}'\"")
    }


def overlap(expected: str, actual: str) -> float:
    expected_terms = terms(expected)
    return len(expected_terms & terms(actual)) / max(1, len(expected_terms))


def evaluate_answer(answer: dict[str, Any], gold: dict[str, Any], allowed: set[tuple[str, int, str]]) -> dict[str, Any]:
    citations = [citation for claim in answer.get("claims", []) for citation in claim.get("citations", [])]
    triples = {(item["paper_id"], int(item["page"]), item["block_id"]) for item in citations}
    gold_blocks = set(gold["gold_block_ids"])
    gold_pages = set(gold["gold_pages"])
    gold_papers = set(gold["gold_paper_ids"])
    exact = [
        item["paper_id"] in gold_papers
        and item["block_id"] in gold_blocks
        and int(item["page"]) in gold_pages
        for item in citations
    ]
    page = [item["paper_id"] in gold_papers and int(item["page"]) in gold_pages for item in citations]
    required = []
    for claim in gold["required_claims"]:
        scores = [
            overlap(claim, generated.get("claim_text") or generated.get("text") or "")
            for generated in answer.get("claims", [])
        ]
        required.append({"required_claim": claim, "best": max(scores, default=0.0)})
    unsupported = sum(
        not any(
            citation["paper_id"] in gold_papers
            and citation["block_id"] in gold_blocks
            and int(citation["page"]) in gold_pages
            for citation in claim.get("citations", [])
        )
        for claim in answer.get("claims", [])
    )
    generated_claims = len(answer.get("claims", []))
    cited_gold = {item["block_id"] for item in citations} & gold_blocks
    valid = [triple in allowed for triple in triples]
    coverage = (
        sum(item["best"] >= CLAIM_MATCH_THRESHOLD for item in required) / len(required)
        if required
        else None
    )
    return {
        "answerable_correct": answer.get("answerable") == gold["answerable"],
        "required_claim_scores": required,
        "required_claim_coverage": coverage,
        "omitted_required_claims": sum(item["best"] < CLAIM_MATCH_THRESHOLD for item in required),
        "unsupported_before_generation": 0,
        "unsupported_after_generation": unsupported,
        "extra_claims": max(0, generated_claims - len(required)),
        "exact_citation_precision": sum(exact) / len(exact) if exact else (1.0 if not gold["answerable"] else 0.0),
        "citation_recall": len(cited_gold) / len(gold_blocks) if gold_blocks else (1.0 if not citations else 0.0),
        "page_citation_precision": sum(page) / len(page) if page else (1.0 if not gold["answerable"] else 0.0),
        "claim_citation_binding_accuracy": (
            sum(bool(claim.get("citations")) for claim in answer.get("claims", [])) / generated_claims
            if generated_claims
            else (1.0 if not answer.get("answerable") else 0.0)
        ),
        "allocated_to_correct_claim_rate": None,
        "invalid_citation_rate": 1 - (sum(valid) / len(valid)) if valid else 0.0,
        "automated_semantic_support_signal": None,
        "human_citation_support": None,
    }


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[max(0, math.ceil(len(ordered) * fraction) - 1)], 3)


def summarize_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [row for row in rows if row["status"] == "completed"]
    answerable = [row for row in completed if row["gold"]["answerable"]]
    unanswerable = [row for row in completed if not row["gold"]["answerable"]]

    def avg(name: str, source: list[dict[str, Any]]) -> float | None:
        values = [row["metrics"].get(name) for row in source if row["metrics"].get(name) is not None]
        return round(mean(values), 6) if values else None

    claims = sum(len(row.get("answer", {}).get("claims", [])) for row in completed)
    elapsed = [float(row["elapsed_seconds"]) * 1000 for row in completed]
    return {
        "attempted": len(rows),
        "completed": len(completed),
        "provider_failures": sum(row["status"] == "provider_failed" for row in rows),
        "execution_failures": sum(row["status"] != "completed" for row in rows),
        "json_schema_success": round(len(completed) / len(rows), 6) if rows else 0.0,
        "answerable_accuracy": avg("answerable_correct", answerable),
        "refusal_accuracy": avg("answerable_correct", unanswerable),
        "required_claim_coverage": avg("required_claim_coverage", answerable),
        "omitted_required_claims": sum(row["metrics"].get("omitted_required_claims", 0) for row in completed),
        "unsupported_before_generation": sum(row["metrics"].get("unsupported_before_generation", 0) for row in completed),
        "unsupported_after_generation": sum(row["metrics"].get("unsupported_after_generation", 0) for row in completed),
        "unsupported_claim_rate": round(
            sum(row["metrics"].get("unsupported_after_generation", 0) for row in completed) / max(1, claims), 6
        ),
        "extra_claims": sum(row["metrics"].get("extra_claims", 0) for row in completed),
        "exact_citation_precision": avg("exact_citation_precision", answerable),
        "citation_recall": avg("citation_recall", answerable),
        "page_citation_precision": avg("page_citation_precision", answerable),
        "claim_citation_binding_accuracy": avg("claim_citation_binding_accuracy", completed),
        "allocated_to_correct_claim_rate": avg("allocated_to_correct_claim_rate", completed),
        "invalid_citation_rate": avg("invalid_citation_rate", completed),
        "citation_retry_rate": round(sum(row.get("citation_retry_count", 0) for row in rows) / max(1, len(rows)), 6),
        "input_tokens": sum(row.get("usage", {}).get("input_tokens", 0) for row in completed),
        "output_tokens": sum(row.get("usage", {}).get("output_tokens", 0) for row in completed),
        "total_tokens": sum(row.get("usage", {}).get("total_tokens", 0) for row in completed),
        "request_attempts": sum(row.get("request_attempt_count", 0) for row in rows),
        "provider_completed_requests": sum(
            int(row.get("provider_completed_request_count") or 0) for row in rows
        ),
        "provider_completion_unknown": sum(
            row.get("provider_completed_request_count") is None for row in rows
        ),
        "active_reserved_tokens": sum(row.get("active_reserved_tokens", 0) for row in rows),
        "elapsed": {
            "mean_ms": round(mean(elapsed), 3) if elapsed else None,
            "p50_ms": percentile(elapsed, 0.5),
            "p95_ms": percentile(elapsed, 0.95),
        },
        "monetary_cost_usd": "0",
    }


def slice_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for field in ("category", "difficulty"):
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[row[field]].append(row)
        output[field] = {key: summarize_metrics(value) for key, value in sorted(groups.items())}
    return output
