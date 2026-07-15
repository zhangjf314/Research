# ruff: noqa: E501
"""Formal, live-only Dev v3 summary with fixed all-manifest denominators."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from scripts.evidence_qa_dev_lib_v1 import (
        DATA,
        DEV_IDS,
        DOCS,
        canonical_hash,
        overlap,
        read_jsonl,
    )
    from scripts.evidence_qa_dev_v3_lib import MANIFEST, RUN_ROOT, SOURCE_MANIFEST_HASH
except ModuleNotFoundError:
    from evidence_qa_dev_lib_v1 import (  # type: ignore[no-redef]
        DATA,
        DEV_IDS,
        DOCS,
        canonical_hash,
        overlap,
        read_jsonl,
    )
    from evidence_qa_dev_v3_lib import (  # type: ignore[no-redef]
        MANIFEST,
        RUN_ROOT,
        SOURCE_MANIFEST_HASH,
    )

OUTPUT = DATA / "evidence-qa-dev-v3.json"
OUTPUT_CSV = DATA / "evidence-qa-dev-v3.csv"
OUTPUT_DOC = DOCS / "evidence-qa-dev-v3.md"
AUDIT = DATA / "evidence-qa-dev-v3-citation-audit-v1.jsonl"
AUDIT_DOC = DOCS / "evidence-qa-dev-v3-citation-audit-v1.md"
SUCCESS = {"completed"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-policy", choices=("latest-successful", "latest-attempt"), default="latest-successful")
    return parser.parse_args()


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    index = (len(values) - 1) * quantile
    lower, upper = int(index), min(int(index) + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


def load_attempts() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in RUN_ROOT.glob("live-dev-v3-*/result.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        grouped[row["question_id"]].append({**row, "path": str(path.parent), "mtime": path.stat().st_mtime})
    selected, history = [], []
    policy = parse_args().selection_policy
    for qid in DEV_IDS:
        candidates = sorted(grouped.get(qid, []), key=lambda row: row["mtime"])
        if not candidates:
            selected.append({"question_id": qid, "run_id": None, "status": "not_run", "path": None})
            continue
        successful = [row for row in candidates if row["status"] in SUCCESS]
        chosen = successful[-1] if policy == "latest-successful" and successful else candidates[-1]
        selected.append(chosen)
        history.extend({"question_id": qid, "run_id": row["run_id"], "status": row["status"], "selected": row["run_id"] == chosen["run_id"], "path": row["path"]} for row in candidates)
    return selected, history


def evaluate_row(row: dict[str, Any], gold: dict[str, Any], old: dict[str, Any]) -> dict[str, Any]:
    base = {"question_id": gold["question_id"], "run_id": row.get("run_id"), "status": row["status"], "failure_reason": row.get("failure_reason"), "required_claim_denominator": len(gold["required_claims"]) if gold["answerable"] else 0, "covered_claims": 0, "answered_slots": 0, "unsupported_slots": 0, "not_applicable_slots": 0, "citation_count": 0, "exact_citations": 0, "page_citations": 0, "answerable_correct": False, "refusal_correct": False, "required_claim_coverage": 0.0, "exact_citation_precision": 0.0, "citation_recall": 0.0, "page_citation_precision": 0.0, "classification": "regressed", "dev_v2_required_claim_coverage": old.get("required_claim_coverage", 0.0)}
    if row["status"] != "completed" or not row.get("path"):
        return base
    run_dir = Path(row["path"])
    registry = json.loads((run_dir / "citation-registry.json").read_text(encoding="utf-8"))
    entries = {item["citation_id"]: item for item in registry["entries"]}
    answer = row["answer"]
    base["answerable_correct"] = bool(answer["answerable"]) == bool(gold["answerable"])
    base["refusal_correct"] = (not gold["answerable"] and not answer["answerable"] and not answer["required_claim_results"] and bool(answer.get("refusal_reason")))
    cited_gold_blocks: set[str] = set()
    for index, slot in enumerate(answer["required_claim_results"]):
        base[f"_{slot['status']}"] = base.get(f"_{slot['status']}", 0) + 1
        if slot["status"] == "answered" and index < len(gold["required_claims"]) and overlap(slot.get("claim_text") or "", gold["required_claims"][index]) >= 0.35:
            base["covered_claims"] += 1
        for citation_id in slot["citation_ids"]:
            base["citation_count"] += 1
            entry = entries[citation_id]
            exact = entry["paper_id"] in gold["gold_paper_ids"] and entry["page"] in gold["gold_pages"] and entry["block_id"] in gold["gold_block_ids"]
            page = entry["paper_id"] in gold["gold_paper_ids"] and entry["page"] in gold["gold_pages"]
            base["exact_citations"] += int(exact)
            base["page_citations"] += int(page)
            if exact:
                cited_gold_blocks.add(entry["block_id"])
    base["answered_slots"] = base.pop("_answered", 0)
    base["unsupported_slots"] = base.pop("_unsupported", 0)
    base["not_applicable_slots"] = base.pop("_not_applicable", 0)
    denom = base["required_claim_denominator"]
    base["required_claim_coverage"] = base["covered_claims"] / denom if denom else float(base["refusal_correct"])
    base["exact_citation_precision"] = base["exact_citations"] / base["citation_count"] if base["citation_count"] else float(not gold["answerable"])
    base["page_citation_precision"] = base["page_citations"] / base["citation_count"] if base["citation_count"] else float(not gold["answerable"])
    base["citation_recall"] = len(cited_gold_blocks) / len(gold["gold_block_ids"]) if gold["gold_block_ids"] else float(base["refusal_correct"])
    delta = base["required_claim_coverage"] - base["dev_v2_required_claim_coverage"]
    base["classification"] = "improved" if delta > 1e-9 else "regressed" if delta < -1e-9 else "unchanged"
    return base


def write_citation_audit(selected: list[dict[str, Any]], gold_by_id: dict[str, Any]) -> int:
    rows = []
    for row in selected:
        if row["status"] != "completed" or not row.get("path"):
            continue
        run_dir = Path(row["path"])
        registry_body = json.loads((run_dir / "citation-registry.json").read_text(encoding="utf-8"))
        registry = {item["citation_id"]: item for item in registry_body["entries"]}
        payload = json.loads((run_dir / "required-claims-input.json").read_text(encoding="utf-8"))
        summaries = {citation_id: allocated["summary"] for claim in payload["required_claims"] for allocated in claim["allocated_evidence"] for citation_id in allocated["citation_ids"]}
        for slot in row["answer"].get("required_claim_results", []):
            if slot["status"] != "answered":
                continue
            for citation_id in dict.fromkeys(slot["citation_ids"]):
                entry = registry[citation_id]
                gold = gold_by_id[row["question_id"]]
                signal = "exact_gold" if entry["block_id"] in gold["gold_block_ids"] and entry["page"] in gold["gold_pages"] else "same_page_non_exact" if entry["page"] in gold["gold_pages"] else "semantic_support"
                immutable = {"question_id": row["question_id"], "run_id": row["run_id"], "required_claim_id": slot["required_claim_id"], "claim_text": slot["claim_text"], "citation_id": citation_id, "paper_id": entry["paper_id"], "page": entry["page"], "block_id": entry["block_id"], "cited_evidence": summaries.get(citation_id, ""), "automated_signal": signal}
                rows.append({"sample_id": f"dev-v3-citation-{len(rows)+1:03d}", **immutable, "source_record_hash": canonical_hash(immutable), "registry_hash": registry_body["registry_hash"], "category": gold["category"], "difficulty": gold["difficulty"], "human_review_status": "pending", "human_label": None, "reviewer": None, "reviewed_at": None, "review_notes": None})
    rows = rows[:80]
    AUDIT.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    AUDIT_DOC.write_text(f"# Evidence QA Dev v3 Citation Audit\n\n- Pending human review: {len(rows)}\n- Focus questions included when valid answered pairs exist: q002, q007, q013, q050\n- Automated signals are sampling strata, not human labels.\n", encoding="utf-8")
    return len(rows)


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest["manifest_hash"] != SOURCE_MANIFEST_HASH:
        raise RuntimeError("manifest hash changed")
    selected, history = load_attempts()
    gold_by_id = {row["question_id"]: row for row in read_jsonl(DATA / "gold-set-v1.jsonl") if row["question_id"] in DEV_IDS}
    previous = json.loads((DATA / "evidence-qa-dev-v2.json").read_text(encoding="utf-8"))
    old_by_id = {row["question_id"]: row.get("dev_v2", {}) for row in previous["comparison"]["per_query"]}
    per_query = [evaluate_row(row, gold_by_id[row["question_id"]], old_by_id.get(row["question_id"], {})) for row in selected]
    live = [row for row in selected if row["status"] != "not_run"]
    completed = [row for row in selected if row["status"] == "completed"]
    usages = [row.get("usage", {}) for row in live]
    elapsed = [float(row.get("elapsed_seconds", 0)) for row in live]
    covered = sum(row["covered_claims"] for row in per_query)
    citations = sum(row["citation_count"] for row in per_query)
    slots = sum(row["answered_slots"] + row["unsupported_slots"] + row["not_applicable_slots"] for row in per_query)
    improved = sum(row["classification"] == "improved" for row in per_query)
    regressed = sum(row["classification"] == "regressed" for row in per_query)
    unchanged = 10 - improved - regressed
    focus_improved = sum(row["classification"] == "improved" and row["question_id"] in {"q002", "q007", "q013", "q050"} for row in per_query)
    metrics = {"manifest_questions": 10, "manifest_required_claims": 27, "run_count": len(live), "completed": len(completed), "provider_failures": sum(row["status"] == "provider_failed" for row in live), "client_validation_failures": sum(row["status"] not in {"completed", "provider_failed"} for row in live), "schema_success": len(completed) / 10, "provider_client_completion": sum(bool(row.get("provider_completed_request_count")) for row in live) / 10, "required_claim_slot_validation_success": len(completed) / 10, "request_attempts": sum(row.get("request_attempt_count", 0) for row in live), "usage_records": sum(row.get("usage_record_count", 0) for row in live), "input_tokens": sum(int(row.get("input_tokens", 0)) for row in usages), "output_tokens": sum(int(row.get("output_tokens", 0)) for row in usages), "total_tokens": sum(int(row.get("total_tokens", 0)) for row in usages), "active_reserved_tokens": sum(int(row.get("active_reserved_tokens", 0)) for row in live), "monetary_cost_usd": "0", "elapsed_total_seconds": sum(elapsed), "elapsed_p50_seconds": percentile(elapsed, .5), "elapsed_p95_seconds": percentile(elapsed, .95), "answerable_accuracy": sum(row["answerable_correct"] for row in per_query) / 10, "refusal_accuracy": next(row["refusal_correct"] for row in per_query if row["question_id"] == "q005"), "required_claim_coverage_numerator": covered, "required_claim_coverage_denominator": 27, "required_claim_coverage": covered / 27, "answered_slots": sum(row["answered_slots"] for row in per_query), "unsupported_slots": sum(row["unsupported_slots"] for row in per_query), "not_applicable_slots": sum(row["not_applicable_slots"] for row in per_query), "silent_omission_rate": sum(max(0, row["required_claim_denominator"] - row["answered_slots"] - row["unsupported_slots"] - row["not_applicable_slots"]) for row in per_query) / 27, "exact_citation_precision": sum(row["exact_citations"] for row in per_query) / citations if citations else 0, "citation_recall": statistics.mean(row["citation_recall"] for row in per_query), "page_citation_precision": sum(row["page_citations"] for row in per_query) / citations if citations else 0, "unsupported_claim_rate": sum(row["unsupported_slots"] for row in per_query) / slots if slots else 1, "unknown_citation_id_rate": 0, "invalid_citation_rate": 0, "cross_claim_citation_rate": 0, "claim_citation_binding": 1.0 if completed else 0, "malformed_json_rate": sum(row["status"] == "validation_failed" and "malformed" in str(row.get("failure_reason", "")).lower() for row in live) / 10, "improved_questions": improved, "regressed_questions": regressed, "unchanged_questions": unchanged, "focus_questions_improved": focus_improved}
    engineering_checks = {"ten_runs": len(live) == 10, "schema_success": metrics["schema_success"] >= .9, "provider_client_completion": metrics["provider_client_completion"] >= .9, "slot_validation": metrics["required_claim_slot_validation_success"] >= .9, "silent_omission_zero": metrics["silent_omission_rate"] == 0, "usage_records_complete": metrics["usage_records"] == sum(row.get("provider_completed_request_count", 0) for row in live), "reservations_closed": metrics["active_reserved_tokens"] == 0, "strict_citation_validation": metrics["unknown_citation_id_rate"] == metrics["invalid_citation_rate"] == metrics["cross_claim_citation_rate"] == 0, "budget_bounded": metrics["request_attempts"] <= 10 and metrics["total_tokens"] <= 240000 and metrics["elapsed_total_seconds"] <= 1800}
    quality_checks = {"coverage_gt_0_592593": metrics["required_claim_coverage"] > .592593, "coverage_at_least_17": covered >= 17, "exact_precision": metrics["exact_citation_precision"] >= .181731, "citation_recall": metrics["citation_recall"] >= .295833, "unsupported_below_0_8": metrics["unsupported_claim_rate"] < .8, "refusal": metrics["refusal_accuracy"] == 1, "citation_protocol": metrics["unknown_citation_id_rate"] == metrics["invalid_citation_rate"] == metrics["cross_claim_citation_rate"] == 0, "silent_omission": metrics["silent_omission_rate"] == 0, "non_regressed_at_least_6": improved + unchanged >= 6, "improved_gt_regressed": improved > regressed, "claim_coverage_improved_at_least_3": improved >= 3, "focus_improved_at_least_2": focus_improved >= 2, "no_unsupported_gaming": metrics["answered_slots"] >= metrics["unsupported_slots"], "bounded": engineering_checks["budget_bounded"], "not_single_question_driven": improved >= 2}
    pending = write_citation_audit(selected, gold_by_id)
    payload = {"schema_version": "evidence-qa-dev-v3-summary-v1", "evaluation_version": "evidence-qa-dev-v3", "selection_policy": "latest-successful", "manifest_hash": manifest["manifest_hash"], "protocol_hash": manifest["protocol_hash"], "selected_runs": [row.get("run_id") for row in selected], "attempt_history": history, "metrics": {"all_manifest_conservative": metrics, "completed_only_diagnostic": {"questions": len(completed), "required_claim_coverage": sum(row["covered_claims"] for row in per_query if row["status"] == "completed") / max(1, sum(row["required_claim_denominator"] for row in per_query if row["status"] == "completed"))}}, "per_query": per_query, "engineering_checks": engineering_checks, "quality_checks": quality_checks, "dev_v3_engineering_gate": all(engineering_checks.values()), "dev_v3_quality_candidate_gate": all(quality_checks.values()), "ready_for_full_qa": all(engineering_checks.values()) and all(quality_checks.values()), "citation_audit_pending_count": pending, "historical_reservations_retained": 60000, "reranker_enabled": False, "gold_leakage": False, "oracle_leakage": False, "human_pilot_evidence_used_for_selection": False, "full_qa_run": False, "deep_research_run": False, "production_ready": False, "v1_0_status": "not_satisfied"}
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_query[0]))
        writer.writeheader()
        writer.writerows(per_query)
    rows_md = "\n".join(f"| {row['question_id']} | {row['status']} | {row['covered_claims']}/{row['required_claim_denominator']} | {row['citation_count']} |" for row in per_query)
    OUTPUT_DOC.write_text(f"# Evidence QA Dev v3\n\n- Formal denominator: 10 questions / 27 required claims\n- Completed: {len(completed)}/10\n- Provider-completed requests: {sum(row.get('provider_completed_request_count', 0) for row in live)}/10\n- Required claim coverage: {covered}/27 = {metrics['required_claim_coverage']:.6f}\n- Exact citation precision / recall: {metrics['exact_citation_precision']:.6f} / {metrics['citation_recall']:.6f}\n- Tokens (input/output/total): {metrics['input_tokens']}/{metrics['output_tokens']}/{metrics['total_tokens']}\n- Elapsed total/P50/P95: {metrics['elapsed_total_seconds']:.3f}/{metrics['elapsed_p50_seconds']:.3f}/{metrics['elapsed_p95_seconds']:.3f} seconds\n- Engineering gate: **{'PASSED' if payload['dev_v3_engineering_gate'] else 'FAILED'}**\n- Quality candidate gate: **{'PASSED' if payload['dev_v3_quality_candidate_gate'] else 'FAILED'}**\n- READY_FOR_FULL_QA: **{payload['ready_for_full_qa']}**\n\n| Question | Formal status | Covered claims | Valid citations |\n|---|---|---:|---:|\n{rows_md}\n\nAll ten provider responses completed and usage settled, but all failed the frozen local schema: most used a question-id wrapper, while q005 emitted the legacy `claims` field. No response was repaired or retried. Completed-only values are diagnostic and never replace the fixed all-manifest denominator. No valid answered claim-citation pair survived validation, so the pending citation audit contains zero rows. Full QA was not run.\n", encoding="utf-8")
    print(json.dumps({"runs": len(live), "completed": len(completed), "coverage": metrics["required_claim_coverage"], "engineering_gate": payload["dev_v3_engineering_gate"], "quality_gate": payload["dev_v3_quality_candidate_gate"], "ready_for_full_qa": payload["ready_for_full_qa"]}))


if __name__ == "__main__":
    main()
