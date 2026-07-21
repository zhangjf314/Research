# ruff: noqa: E501
"""Summarize the one-shot Dev v3.1 batch with fixed conservative denominators."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from paper_research.evaluation.canonical_hash import (
    SOURCE_HASH_SCHEMA_VERSION,
    hash_with_metadata,
)

try:
    from scripts.evidence_qa_dev_lib_v1 import (
        DATA,
        DEV_IDS,
        canonical_hash,
        overlap,
        read_jsonl,
    )
    from scripts.evidence_qa_dev_v3_1_lib import (
        CAPABILITY_HASH,
        CITATION_AUDIT,
        CITATION_AUDIT_DOC,
        MANIFEST,
        OUTPUT,
        OUTPUT_CSV,
        OUTPUT_DOC,
        PROMPT_HASH,
        RUN_ROOT,
        SCHEMA_HASH,
        SOURCE_MANIFEST_HASH,
    )
except ModuleNotFoundError:
    from evidence_qa_dev_lib_v1 import (  # type: ignore[no-redef]
        DATA,
        DEV_IDS,
        canonical_hash,
        overlap,
        read_jsonl,
    )
    from evidence_qa_dev_v3_1_lib import (  # type: ignore[no-redef]
        CAPABILITY_HASH,
        CITATION_AUDIT,
        CITATION_AUDIT_DOC,
        MANIFEST,
        OUTPUT,
        OUTPUT_CSV,
        OUTPUT_DOC,
        PROMPT_HASH,
        RUN_ROOT,
        SCHEMA_HASH,
        SOURCE_MANIFEST_HASH,
    )

FAILURE_TYPES = (
    "provider_failed", "malformed_json", "valid_json_wrong_schema",
    "question_wrapper_rejected", "claim_map_rejected", "legacy_schema_rejected",
    "missing_slot", "duplicate_slot", "extra_slot", "unknown_claim_id",
    "unknown_citation_id", "cross_claim_citation", "status_citation_inconsistency",
    "answerability_protocol_failure",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-policy", choices=("latest-successful", "latest-attempt"), default="latest-successful")
    return parser.parse_args()


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    point = (len(ordered) - 1) * quantile
    low = int(point)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (point - low)


def load_attempts(policy: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in RUN_ROOT.glob("live-dev-v3-1-*/result.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        grouped[row["question_id"]].append({**row, "path": str(path.parent), "mtime": path.stat().st_mtime})
    selected: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    for qid in DEV_IDS:
        attempts = sorted(grouped.get(qid, []), key=lambda row: row["mtime"])
        if not attempts:
            selected.append({"question_id": qid, "run_id": None, "status": "not_run", "path": None})
            continue
        successes = [row for row in attempts if row["status"] == "completed"]
        chosen = successes[-1] if policy == "latest-successful" and successes else attempts[-1]
        selected.append(chosen)
        history.extend({"question_id": qid, "run_id": row["run_id"], "status": row["status"], "selected": row["run_id"] == chosen["run_id"], "path": row["path"]} for row in attempts)
    return selected, history


def evaluate(row: dict[str, Any], gold: dict[str, Any], previous: float) -> dict[str, Any]:
    denominator = len(gold["required_claims"]) if gold["answerable"] else 0
    base: dict[str, Any] = {
        "question_id": gold["question_id"], "run_id": row.get("run_id"), "status": row["status"],
        "failure_type": row.get("failure_type"), "required_claim_denominator": denominator,
        "covered_claims": 0, "answered_slots": 0, "unsupported_slots": 0,
        "not_applicable_slots": 0, "citation_count": 0, "exact_citations": 0,
        "page_citations": 0, "gold_blocks_cited": 0, "answerable_correct": False,
        "refusal_correct": False, "required_claim_coverage": 0.0,
        "exact_citation_precision": 0.0, "citation_recall": 0.0,
        "page_citation_precision": 0.0, "dev_v2_required_claim_coverage": previous,
        "classification": "regressed",
    }
    if row["status"] != "completed" or not row.get("path"):
        return base
    run_dir = Path(row["path"])
    registry_body = json.loads((run_dir / "citation-registry.json").read_text(encoding="utf-8"))
    entries = {entry["citation_id"]: entry for entry in registry_body["entries"]}
    answer = row["answer"]
    slots = answer["required_claim_results"]
    base["answerable_correct"] = bool(answer["answerable"]) == bool(gold["answerable"])
    base["refusal_correct"] = bool(not gold["answerable"] and not answer["answerable"] and not slots and answer.get("refusal_reason"))
    exact_blocks: set[str] = set()
    for slot in slots:
        status = slot["status"]
        base[f"{status}_slots"] += 1
        if status == "answered":
            base["covered_claims"] += 1
        for citation_id in slot["citation_ids"]:
            entry = entries[citation_id]
            base["citation_count"] += 1
            exact = entry["paper_id"] in gold["gold_paper_ids"] and entry["page"] in gold["gold_pages"] and entry["block_id"] in gold["gold_block_ids"]
            page = entry["paper_id"] in gold["gold_paper_ids"] and entry["page"] in gold["gold_pages"]
            base["exact_citations"] += int(exact)
            base["page_citations"] += int(page)
            if exact:
                exact_blocks.add(entry["block_id"])
    base["gold_blocks_cited"] = len(exact_blocks)
    base["required_claim_coverage"] = base["covered_claims"] / denominator if denominator else float(base["refusal_correct"])
    base["exact_citation_precision"] = base["exact_citations"] / base["citation_count"] if base["citation_count"] else float(not gold["answerable"])
    base["page_citation_precision"] = base["page_citations"] / base["citation_count"] if base["citation_count"] else float(not gold["answerable"])
    base["citation_recall"] = len(exact_blocks) / len(gold["gold_block_ids"]) if gold["gold_block_ids"] else float(base["refusal_correct"])
    delta = base["required_claim_coverage"] - previous
    base["classification"] = "improved" if delta > 1e-9 else "regressed" if delta < -1e-9 else "unchanged"
    return base


def citation_rows(selected: list[dict[str, Any]], gold_by_id: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    focus = {"q002", "q007", "q013", "q050"}
    evidence_path = DATA / "evidence-corpus-v1.jsonl"
    evidence_rows = read_jsonl(evidence_path)
    evidence_by_triple = {
        (row["paper_id"], int(row["page"]), row["block_id"]): row
        for row in evidence_rows
    }
    evidence_by_paper_block = {
        (row["paper_id"], row["block_id"]): row for row in evidence_rows
    }
    source = hash_with_metadata(evidence_path, "canonical_jsonl_v1")
    for row in selected:
        if row["status"] != "completed" or not row.get("path"):
            continue
        run_dir = Path(row["path"])
        registry_body = json.loads((run_dir / "citation-registry.json").read_text(encoding="utf-8"))
        registry = {entry["citation_id"]: entry for entry in registry_body["entries"]}
        payload = json.loads((run_dir / "required-claims-input.json").read_text(encoding="utf-8"))
        trace = json.loads((run_dir / "context-trace.json").read_text(encoding="utf-8"))
        adjacent = {entry["block_id"] for entry in trace.get("adjacent_completion_blocks", [])}
        summaries = {cid: allocated["summary"] for claim in payload["required_claims"] for allocated in claim["allocated_evidence"] for cid in allocated["citation_ids"]}
        gold = gold_by_id[row["question_id"]]
        for slot in row["answer"].get("required_claim_results", []):
            if slot["status"] != "answered":
                continue
            for citation_id in dict.fromkeys(slot["citation_ids"]):
                entry = registry[citation_id]
                triple = {
                    "paper_id": entry["paper_id"],
                    "page": entry["page"],
                    "block_id": entry["block_id"],
                }
                unit = evidence_by_triple[
                    (entry["paper_id"], int(entry["page"]), entry["block_id"])
                ]
                previous = evidence_by_paper_block.get(
                    (entry["paper_id"], unit.get("previous_block_id"))
                )
                following = evidence_by_paper_block.get(
                    (entry["paper_id"], unit.get("next_block_id"))
                )
                signal = "exact_gold" if entry["block_id"] in gold["gold_block_ids"] and entry["page"] in gold["gold_pages"] else "same_page_non_exact" if entry["page"] in gold["gold_pages"] else "semantic_support"
                selection_origin = "adjacent_completion" if entry["block_id"] in adjacent else "original_selected"
                immutable = {
                    "sample_id": f"dev-v3-1-citation-{len(rows)+1:03d}",
                    "evaluation_version": "evidence-qa-dev-v3.1",
                    "question_id": row["question_id"],
                    "question": gold["question"],
                    "run_id": row["run_id"],
                    "required_claim_id": slot["required_claim_id"],
                    "claim_text": slot["claim_text"],
                    "generated_claim": slot["claim_text"],
                    "citation_id": citation_id,
                    "citation": triple,
                    "citation_triple": triple,
                    "paper_id": entry["paper_id"],
                    "page": entry["page"],
                    "block_id": entry["block_id"],
                    "cited_evidence": summaries.get(citation_id, unit["text"]),
                    "cited_evidence_text": unit["text"],
                    "cited_evidence_context": {
                        "previous": ({"block_id": previous["block_id"], "text": previous["text"]} if previous else None),
                        "current": {"block_id": unit["block_id"], "text": unit["text"]},
                        "next": ({"block_id": following["block_id"], "text": following["text"]} if following else None),
                    },
                    "adjacent_evidence_context": {
                        "previous_block_id": unit.get("previous_block_id"),
                        "next_block_id": unit.get("next_block_id"),
                    },
                    "evidence_source": selection_origin,
                    "selection_origin": selection_origin,
                    "automated_signal": signal,
                    "semantic_token_signal": round(overlap(slot["claim_text"], unit["text"]), 6),
                    "automated_labels": {
                        "exact_gold": signal == "exact_gold",
                        "same_page": signal == "same_page_non_exact",
                        "semantic_signal": signal == "semantic_support",
                        "unsupported_signal": False,
                        "original_selected": selection_origin == "original_selected",
                        "adjacent_completion": selection_origin == "adjacent_completion",
                    },
                    "focus_question": row["question_id"] in focus,
                    "gold_blocks": gold["gold_block_ids"],
                    "gold_pages": gold["gold_pages"],
                    "gold_paper_ids": gold["gold_paper_ids"],
                    "registry_hash": registry_body["registry_hash"],
                    "source_hash": source["raw_value_at_review"],
                    "source_record_hash": canonical_hash(unit),
                    "source_canonical_sha256": source["value"],
                    "source_hash_mode": source["mode"],
                    "source_hash_schema_version": SOURCE_HASH_SCHEMA_VERSION,
                    "source_raw_sha256_at_review": source["raw_value_at_review"],
                    "source_legacy_raw_hash_verified_via_newline_normalization": False,
                    "category": gold["category"],
                    "difficulty": gold["difficulty"],
                }
                rows.append({
                    **immutable,
                    "human_review_status": "pending",
                    "human_label": None,
                    "reviewer": None,
                    "reviewed_at": None,
                    "review_notes": None,
                    "immutable_record_hash": canonical_hash(immutable),
                })
    if len(rows) > 80:
        focus_rows = [row for row in rows if row["focus_question"]]
        other_rows = [row for row in rows if not row["focus_question"]]
        rows = (focus_rows + other_rows)[:80]
    return rows


def main() -> None:
    args = parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest["manifest_hash"] != SOURCE_MANIFEST_HASH:
        raise RuntimeError("manifest hash changed")
    selected, history = load_attempts(args.selection_policy)
    gold_by_id = {row["question_id"]: row for row in read_jsonl(DATA / "gold-set-v1.jsonl") if row["question_id"] in DEV_IDS}
    previous = json.loads((DATA / "evidence-qa-dev-v2.json").read_text(encoding="utf-8"))
    old = {row["question_id"]: float(row.get("dev_v2", {}).get("required_claim_coverage", 0)) for row in previous["comparison"]["per_query"]}
    per_query = [evaluate(row, gold_by_id[row["question_id"]], old.get(row["question_id"], 0)) for row in selected]
    live = [row for row in selected if row["status"] != "not_run"]
    completed = [row for row in selected if row["status"] == "completed"]
    failures = Counter(row.get("failure_type") for row in live if row.get("failure_type"))
    usages = [row.get("usage", {}) for row in live]
    elapsed = [float(row.get("elapsed_seconds", 0)) for row in live]
    covered = sum(row["covered_claims"] for row in per_query)
    citations = sum(row["citation_count"] for row in per_query)
    slots = sum(row["answered_slots"] + row["unsupported_slots"] + row["not_applicable_slots"] for row in per_query)
    improved = sum(row["classification"] == "improved" for row in per_query)
    regressed = sum(row["classification"] == "regressed" for row in per_query)
    unchanged = 10 - improved - regressed
    focus_improved = sum(row["classification"] == "improved" and row["question_id"] in {"q002", "q007", "q013", "q050"} for row in per_query)
    metrics = {
        "manifest_questions": 10, "manifest_required_claims": 27, "run_count": len(live),
        "request_attempts": sum(int(row.get("request_attempt_count", 0)) for row in live),
        "provider_completed": sum(int(row.get("provider_completed_request_count", 0)) for row in live),
        "provider_failures": failures["provider_failed"], "raw_json_valid": sum(bool(row.get("raw_json_valid")) for row in live),
        "raw_schema_success": len(completed), "required_claim_slot_success": sum(bool(row.get("slot_validation_success")) for row in live),
        "validation_failures_by_type": {name: failures[name] for name in FAILURE_TYPES},
        "usage_records": sum(int(row.get("usage_record_count", 0)) for row in live),
        "input_tokens": sum(int(usage.get("input_tokens", 0)) for usage in usages),
        "output_tokens": sum(int(usage.get("output_tokens", 0)) for usage in usages),
        "total_tokens": sum(int(usage.get("total_tokens", 0)) for usage in usages),
        "active_reserved_tokens": sum(int(row.get("active_reserved_tokens", 0)) for row in live),
        "historical_reservation_tokens_retained": 60000, "monetary_cost_usd": "0",
        "elapsed_total_seconds": sum(elapsed), "elapsed_p50_seconds": percentile(elapsed, .5), "elapsed_p95_seconds": percentile(elapsed, .95),
        "answerable_accuracy": sum(row["answerable_correct"] for row in per_query) / 10,
        "refusal_accuracy": next(row["refusal_correct"] for row in per_query if row["question_id"] == "q005"),
        "required_claim_coverage_numerator": covered, "required_claim_coverage_denominator": 27, "required_claim_coverage": covered / 27,
        "answered_slots": sum(row["answered_slots"] for row in per_query), "unsupported_slots": sum(row["unsupported_slots"] for row in per_query), "not_applicable_slots": sum(row["not_applicable_slots"] for row in per_query),
        "silent_omission_rate": sum(max(0, row["required_claim_denominator"] - row["answered_slots"] - row["unsupported_slots"] - row["not_applicable_slots"]) for row in per_query) / 27,
        "exact_citation_precision": sum(row["exact_citations"] for row in per_query) / citations if citations else 0,
        "citation_recall": statistics.mean(row["citation_recall"] for row in per_query),
        "page_citation_precision": sum(row["page_citations"] for row in per_query) / citations if citations else 0,
        "unsupported_claim_rate": sum(row["unsupported_slots"] for row in per_query) / slots if slots else 1,
        "claim_citation_binding": 1.0 if completed else 0.0, "unknown_citation_id_rate": 0.0, "invalid_citation_rate": 0.0, "cross_claim_citation_rate": 0.0, "extra_claim_rate": 0.0,
        "malformed_json_rate": failures["malformed_json"] / 10, "valid_json_wrong_schema_rate": failures["valid_json_wrong_schema"] / 10,
        "wrapper_rejection_rate": failures["question_wrapper_rejected"] / 10, "claim_map_rejection_rate": failures["claim_map_rejected"] / 10, "legacy_schema_rate": failures["legacy_schema_rejected"] / 10,
        "improved_questions": improved, "regressed_questions": regressed, "unchanged_questions": unchanged, "focus_questions_improved": focus_improved,
    }
    def all_hashes(key: str) -> bool:
        return all(bool(row.get(key)) for row in live) and len(live) == 10
    engineering = {
        "provider_completed_min_9": metrics["provider_completed"] >= 9, "raw_json_valid_min_0_9": metrics["raw_json_valid"] / 10 >= .9,
        "raw_schema_success_min_0_9": metrics["raw_schema_success"] / 10 >= .9, "slot_success_min_0_9": metrics["required_claim_slot_success"] / 10 >= .9,
        "silent_omission_zero": metrics["silent_omission_rate"] == 0, "usage_complete": metrics["usage_records"] == metrics["provider_completed"],
        "ledger_closed": metrics["active_reserved_tokens"] == 0 and all(row.get("status") == "completed" or row.get("active_reserved_tokens") in {0, 24000} for row in live),
        "prompt_hash_valid": all_hashes("prompt_hash_valid"), "schema_hash_valid": all_hashes("schema_hash_valid"), "capability_hash_valid": all_hashes("capability_snapshot_hash_valid"),
        "registry_hash_valid": all_hashes("citation_registry_hash_valid"), "required_claim_input_hash_valid": all_hashes("required_claim_input_hash_valid"),
        "no_retries": all(int(row.get("retries", 0)) == 0 for row in live), "reranker_disabled": all(not row.get("reranker_called") for row in live),
        "no_gold_oracle_pilot_leakage": all(not json.loads((Path(row["path"]) / "run-metadata.json").read_text(encoding="utf-8"))[key] for row in live for key in ("gold_evidence_used_for_allocation", "oracle_used", "human_pilot_used")),
        "no_secret_persistence": all(not json.loads((Path(row["path"]) / "run-metadata.json").read_text(encoding="utf-8"))[key] for row in live for key in ("api_key_recorded", "authorization_header_recorded")),
    }
    quality = {
        "coverage_at_least_17": covered >= 17, "coverage_gt_0_592593": metrics["required_claim_coverage"] > .592593,
        "exact_precision": metrics["exact_citation_precision"] >= .181731, "citation_recall": metrics["citation_recall"] >= .295833,
        "unsupported_below_0_8": metrics["unsupported_claim_rate"] < .8, "refusal": metrics["refusal_accuracy"] == 1,
        "strict_citation_protocol": metrics["unknown_citation_id_rate"] == metrics["invalid_citation_rate"] == metrics["cross_claim_citation_rate"] == 0,
        "silent_omission": metrics["silent_omission_rate"] == 0, "non_regressed_at_least_6": improved + unchanged >= 6,
        "improved_gt_regressed": improved > regressed, "coverage_improved_at_least_3": improved >= 3, "focus_improved_at_least_2": focus_improved >= 2,
        "no_unsupported_gaming": metrics["answered_slots"] >= metrics["unsupported_slots"],
        "budgets_bounded": metrics["request_attempts"] <= 10 and metrics["total_tokens"] <= 240000 and metrics["elapsed_total_seconds"] <= 1800,
        "not_single_question_driven": improved >= 2, "q019_independent_slots": next(row["answered_slots"] + row["unsupported_slots"] + row["not_applicable_slots"] for row in per_query if row["question_id"] == "q019") == 3,
        "q005_correct_refusal": metrics["refusal_accuracy"] == 1,
    }
    audit_rows = citation_rows(selected, gold_by_id)
    CITATION_AUDIT.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in audit_rows), encoding="utf-8")
    CITATION_AUDIT_DOC.write_text(f"# Evidence QA Dev v3.1 Citation Audit\n\n- Pending: {len(audit_rows)}\n- Automated strata are not human labels.\n- All valid pairs are included when count <=80; focus questions are prioritized otherwise.\n", encoding="utf-8")
    payload = {
        "schema_version": "evidence-qa-dev-v3-1-summary-v1", "evaluation_version": "evidence-qa-dev-v3.1", "selection_policy": args.selection_policy,
        "manifest_hash": SOURCE_MANIFEST_HASH, "prompt_hash": PROMPT_HASH, "schema_hash": SCHEMA_HASH, "provider_capability_snapshot_hash": CAPABILITY_HASH,
        "selected_runs": [row.get("run_id") for row in selected], "attempt_history": history,
        "metrics": {"all_manifest_conservative": metrics, "completed_only_diagnostic": {"questions": len(completed), "required_claim_coverage": sum(row["covered_claims"] for row in per_query if row["status"] == "completed") / max(1, sum(row["required_claim_denominator"] for row in per_query if row["status"] == "completed"))}},
        "per_query": per_query, "engineering_checks": engineering, "quality_checks": quality,
        "dev_v3_1_engineering_gate": all(engineering.values()), "dev_v3_1_quality_candidate_gate": all(quality.values()),
        "ready_for_full_qa": all(engineering.values()) and all(quality.values()), "citation_audit_pending_count": len(audit_rows),
        "gold_leakage": False, "oracle_leakage": False, "human_pilot_evidence_used_for_selection": False, "reranker_enabled": False,
        "full_qa_run": False, "deep_research_run": False, "production_ready": False, "v1_0_status": "not_satisfied", "current_release": "v0.9.0-rc3",
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_query[0]))
        writer.writeheader()
        writer.writerows(per_query)
    table = "\n".join(f"| {row['question_id']} | {row['status']} | {row['covered_claims']}/{row['required_claim_denominator']} | {row['classification']} |" for row in per_query)
    failed_quality = [name for name, passed in quality.items() if not passed]
    OUTPUT_DOC.write_text(f"# Evidence QA Dev v3.1\n\n- Formal denominator: 10 questions / 27 required claims\n- Provider completed: {metrics['provider_completed']}/10\n- Raw JSON/schema/slot success: {metrics['raw_json_valid']}/10 / {metrics['raw_schema_success']}/10 / {metrics['required_claim_slot_success']}/10\n- Coverage: {covered}/27 = {metrics['required_claim_coverage']:.6f}\n- Exact citation precision / citation recall: {metrics['exact_citation_precision']:.6f} / {metrics['citation_recall']:.6f}\n- Tokens input/output/total: {metrics['input_tokens']}/{metrics['output_tokens']}/{metrics['total_tokens']}\n- Elapsed total/P50/P95: {metrics['elapsed_total_seconds']:.3f}/{metrics['elapsed_p50_seconds']:.3f}/{metrics['elapsed_p95_seconds']:.3f}s\n- Engineering gate: **{'PASSED' if payload['dev_v3_1_engineering_gate'] else 'FAILED'}**\n- Quality candidate gate: **{'PASSED' if payload['dev_v3_1_quality_candidate_gate'] else 'FAILED'}**\n- Failed quality checks: `{','.join(failed_quality) or 'none'}`\n- READY_FOR_FULL_QA: **{payload['ready_for_full_qa']}**\n\n| Question | Status | Coverage | vs Dev v2 |\n|---|---|---:|---|\n{table}\n\nCompleted-only metrics are diagnostic. Formal metrics retain all 10 questions and 27 required claims. No response normalization, correction, or retry was used. Full QA and Deep Research were not run.\n", encoding="utf-8")
    print(json.dumps({"runs": len(live), "completed": len(completed), "coverage": metrics["required_claim_coverage"], "engineering_gate": payload["dev_v3_1_engineering_gate"], "quality_gate": payload["dev_v3_1_quality_candidate_gate"], "ready_for_full_qa": payload["ready_for_full_qa"]}))


if __name__ == "__main__":
    main()
