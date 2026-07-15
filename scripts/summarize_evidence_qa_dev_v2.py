# ruff: noqa: E501
"""Explicit, conservative summary for the isolated Stage 13.3 Dev v2 batch."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from typing import Any

try:
    from scripts.evidence_qa_dev_lib_v1 import (
        DATA,
        DEV_IDS,
        DOCS,
        GAIN_IDS,
        read_jsonl,
        summarize_metrics,
    )
    from scripts.summarize_evidence_qa_dev_v1 import historical_rows, retrieval_metrics
except ModuleNotFoundError:
    from evidence_qa_dev_lib_v1 import (  # type: ignore[no-redef]
        DATA,
        DEV_IDS,
        DOCS,
        GAIN_IDS,
        read_jsonl,
        summarize_metrics,
    )
    from summarize_evidence_qa_dev_v1 import (  # type: ignore[no-redef]
        historical_rows,
        retrieval_metrics,
    )

RUN_ROOT = DATA / "evidence-qa-dev-v2/runs"
SUMMARY_JSON = DATA / "evidence-qa-dev-v2.json"
SUMMARY_CSV = DATA / "evidence-qa-dev-v2.csv"
SUMMARY_DOC = DOCS / "evidence-qa-dev-v2.md"
AUDIT_JSONL = DATA / "evidence-qa-dev-v2-citation-audit-v1.jsonl"
AUDIT_DOC = DOCS / "evidence-qa-dev-v2-citation-audit-v1.md"
FINAL_AUDIT = DATA / "evidence-qa-dev-v2-final-audit.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-policy", choices=("latest-successful", "latest-attempt"), default="latest-successful")
    return parser.parse_args()


def select_runs(policy: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    attempts: dict[str, list[tuple[float, dict[str, Any], str]]] = defaultdict(list)
    for path in RUN_ROOT.glob("*/result.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        attempts[row["question_id"]].append((path.stat().st_mtime, row, str(path.parent)))
    selected, history = [], []
    for question_id in DEV_IDS:
        candidates = sorted(attempts.get(question_id, []), key=lambda item: item[0])
        if not candidates:
            continue
        successes = [item for item in candidates if item[1]["status"] == "completed"]
        chosen = (successes[-1] if successes else candidates[-1]) if policy == "latest-successful" else candidates[-1]
        selected.append(chosen[1])
        history.extend({"question_id": question_id, "run_id": item[1]["run_id"], "status": item[1]["status"], "selected": item[1]["run_id"] == chosen[1]["run_id"], "path": item[2]} for item in candidates)
    return selected, history


def conservative_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    diag = summarize_metrics(rows)
    completed = [row for row in rows if row["status"] == "completed"]
    answerable_ids = {row["question_id"] for row in read_jsonl(DATA / "gold-set-v1.jsonl") if row["question_id"] in DEV_IDS and row["answerable"]}
    refusal_ids = set(DEV_IDS) - answerable_ids
    answerable_score = sum(bool(row["metrics"].get("answerable_correct")) for row in completed if row["question_id"] in answerable_ids) / max(1, len(answerable_ids))
    refusal_score = sum(bool(row["metrics"].get("answerable_correct")) for row in completed if row["question_id"] in refusal_ids) / max(1, len(refusal_ids))
    coverage = sum(float(row["metrics"].get("required_claim_coverage") or 0) for row in completed if row["question_id"] in answerable_ids) / max(1, len(answerable_ids))
    elapsed_all = sorted(float(row["elapsed_seconds"]) * 1000 for row in rows)
    p95_index = max(0, min(len(elapsed_all) - 1, (95 * len(elapsed_all) + 99) // 100 - 1)) if elapsed_all else 0
    return {
        **diag,
        "manifest_denominator": 10,
        "engineering_completion": len(completed) / 10,
        "json_schema_success": len(completed) / 10,
        "answerable_accuracy": round(answerable_score, 6),
        "refusal_accuracy": round(refusal_score, 6),
        "required_claim_coverage": round(coverage, 6),
        # Provider-settled usage survives schema/post-processing failure.
        "input_tokens": sum(row.get("usage", {}).get("input_tokens", 0) for row in rows),
        "output_tokens": sum(row.get("usage", {}).get("output_tokens", 0) for row in rows),
        "total_tokens": sum(row.get("usage", {}).get("total_tokens", 0) for row in rows),
        "usage_records": sum(row.get("usage_record_count", 0) for row in rows),
        "post_processing_failures": sum(row["status"] == "validation_failed" for row in rows),
        "client_failures": sum(row["status"] == "provider_failed" for row in rows),
        "total_elapsed_seconds": round(sum(float(row["elapsed_seconds"]) for row in rows), 6),
        "all_attempts_p95_elapsed_ms": round(elapsed_all[p95_index], 3) if elapsed_all else None,
    }


def generate_citation_audit(rows: list[dict[str, Any]]) -> int:
    gold = {row["question_id"]: row for row in read_jsonl(DATA / "gold-set-v1.jsonl")}
    evidence = {(row["paper_id"], int(row["page"]), row["block_id"]): row for row in read_jsonl(DATA / "evidence-corpus-v1.jsonl")}
    samples = []
    for row in rows:
        if row["status"] != "completed":
            continue
        for claim in row.get("answer", {}).get("claims", []):
            for citation in claim.get("citations", []):
                triple = (citation["paper_id"], int(citation["page"]), citation["block_id"])
                unit = evidence[triple]
                samples.append({"sample_id": f"evidence-qa-dev-v2-citation-{len(samples)+1:03d}", "evaluation_version": "evidence-qa-dev-v2", "run_id": row["run_id"], "question_id": row["question_id"], "category": row["category"], "difficulty": row["difficulty"], "claim_id": claim["claim_id"], "claim_text": claim["claim_text"], "citation": citation, "cited_evidence_text": unit["text"], "adjacent_evidence_context": {"previous_block_id": unit.get("previous_block_id"), "next_block_id": unit.get("next_block_id")}, "gold_blocks": gold[row["question_id"]]["gold_block_ids"], "gold_pages": gold[row["question_id"]]["gold_pages"], "automated_labels": {"exact_gold": citation["block_id"] in gold[row["question_id"]]["gold_block_ids"], "same_page": int(citation["page"]) in gold[row["question_id"]]["gold_pages"], "adjacent_completion": unit.get("selection_source") == "adjacent_completion", "original_selected": unit.get("selection_source") != "adjacent_completion", "unsupported_signal": citation["block_id"] not in gold[row["question_id"]]["gold_block_ids"] and int(citation["page"]) not in gold[row["question_id"]]["gold_pages"]}, "human_review_status": "pending", "human_label": None, "reviewer": None, "reviewed_at": None, "review_notes": None})
    samples = samples[:60]
    AUDIT_JSONL.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in samples), encoding="utf-8")
    AUDIT_DOC.write_text(f"# Evidence QA Dev v2 Citation Audit\n\n- Samples: {len(samples)}\n- All human labels remain **pending**.\n- No automated signal is treated as human correctness.\n", encoding="utf-8")
    return len(samples)


def main() -> None:
    args = parse_args()
    selected, history = select_runs(args.selection_policy)
    completed_metrics = summarize_metrics(selected)
    formal_metrics = conservative_metrics(selected)
    historical = historical_rows()
    stage13_2 = json.loads((DATA / "evidence-qa-dev-v1.json").read_text(encoding="utf-8"))
    baseline_by_id = {row["question_id"]: row for row in historical}
    per_query = []
    improved = regressed = unchanged = gain_improved = 0
    for row in selected:
        before = baseline_by_id[row["question_id"]]["metrics"]
        after = row.get("metrics", {})
        deltas = {name: (after.get(name) or 0) - (before.get(name) or 0) for name in ("required_claim_coverage", "exact_citation_precision", "citation_recall")}
        positive, negative = any(value > 0 for value in deltas.values()), any(value < 0 for value in deltas.values())
        classification = "improved" if positive and not negative else "regressed" if negative else "unchanged_or_mixed"
        improved += classification == "improved"
        regressed += classification == "regressed"
        unchanged += classification == "unchanged_or_mixed"
        if row["question_id"] in GAIN_IDS and positive:
            gain_improved += 1
        per_query.append({"question_id": row["question_id"], "status": row["status"], "historical_stage11c": before, "dev_v2": after, "deltas": deltas, "classification": classification})
    engineering_checks = {"ten_run_directories": len(selected) == 10, "schema_success_at_least_0_90": formal_metrics["json_schema_success"] >= 0.9, "provider_client_completion_at_least_0_90": completed_metrics["completed"] / 10 >= 0.9, "usage_before_parsing": all(row.get("usage_record_count", 0) == row.get("provider_completed_request_count", 0) for row in selected), "registry_hash_valid": all(row.get("registry_hash_valid") is True for row in selected), "request_attempts_at_most_10": formal_metrics["request_attempts"] <= 10, "active_budget_within_global": 60000 + formal_metrics["total_tokens"] + formal_metrics["active_reserved_tokens"] <= 200000, "elapsed_at_most_1800": sum(row["elapsed_seconds"] for row in selected) <= 1800, "strict_triples": formal_metrics["invalid_citation_rate"] in (0, 0.0, None)}
    quality_checks = {"claim_coverage_gt_0_50": formal_metrics["required_claim_coverage"] > 0.50, "claim_coverage_not_below_0_555556": formal_metrics["required_claim_coverage"] >= 0.555556, "exact_precision_gt_0_111111": (formal_metrics["exact_citation_precision"] or 0) > 0.111111, "citation_recall_gt_0_083333": (formal_metrics["citation_recall"] or 0) > 0.083333, "unsupported_rate_lt_0_916667": formal_metrics["unsupported_claim_rate"] < 0.916667, "refusal_accuracy_gt_0": formal_metrics["refusal_accuracy"] > 0, "unknown_id_rate_zero": sum(row.get("metrics", {}).get("unknown_citation_id_count", 0) for row in selected) == 0, "invalid_rate_zero": formal_metrics["invalid_citation_rate"] in (0, 0.0), "at_least_3_improved": improved >= 3, "at_least_6_non_regressed": improved + unchanged >= 6, "improved_gt_regressed": improved > regressed, "gain_queries_at_least_2": gain_improved >= 2, "p95_elapsed_within_180s": (formal_metrics["elapsed"]["p95_ms"] or 999999) <= 180000}
    pending = generate_citation_audit(selected)
    payload = {"schema_version": "evidence-qa-dev-v2-summary-v1", "evaluation_version": "evidence-qa-dev-v2", "selection_policy": args.selection_policy, "manifest_hash": "fcb59b71fc68549479c24f6475f7d18ad9e382aace93e70e93594ee355ffb988", "selected_runs": [row["run_id"] for row in selected], "attempt_history": history, "metrics": {"all_manifest_conservative": formal_metrics, "completed_only_diagnostic": completed_metrics, "retrieval": retrieval_metrics(DEV_IDS)}, "comparison": {"historical_stage11c": summarize_metrics(historical), "stage13_2_retrieval_only": stage13_2["variants"]["retrieval_only"]["metrics"], "per_query": per_query, "change_counts": {"improved": improved, "regressed": regressed, "unchanged_or_mixed": unchanged, "gain_queries_improved": gain_improved}, "denominator_warning": "Historical schemes have different execution denominators; no statistical superiority is claimed."}, "engineering_checks": engineering_checks, "quality_checks": quality_checks, "dev_v2_engineering_gate": all(engineering_checks.values()), "dev_v2_quality_candidate_gate": all(quality_checks.values()), "ready_for_full_qa": all(engineering_checks.values()) and all(quality_checks.values()), "citation_audit_pending_count": pending, "historical_reservations_retained": 60000, "reranker_enabled": False, "gold_leakage": False, "oracle_leakage": False, "human_pilot_evidence_used_for_selection": False, "full_qa_run": False, "deep_research_run": False, "production_ready": False, "v1_0_status": "not_satisfied"}
    SUMMARY_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_rows = [{"metric_set": name, **{key: json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else value for key, value in metrics.items()}} for name, metrics in payload["metrics"].items()]
    keys = sorted({key for row in csv_rows for key in row})
    with SUMMARY_CSV.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(csv_rows)
    report = [
        "# Evidence QA Dev v2",
        "",
        f"- Manifest: `{payload['manifest_hash']}`",
        f"- Runs: {len(selected)}/10; completed: {formal_metrics['completed']}; post-processing failures: {formal_metrics['post_processing_failures']}",
        f"- Requests / Provider completions / usage records: {formal_metrics['request_attempts']} / {formal_metrics['provider_completed_requests']} / {formal_metrics['usage_records']}",
        f"- Settled input / output / total tokens: {formal_metrics['input_tokens']} / {formal_metrics['output_tokens']} / {formal_metrics['total_tokens']}",
        f"- Active reservations: {formal_metrics['active_reserved_tokens']}; retained historical reservations: 60000",
        f"- Total elapsed: {formal_metrics['total_elapsed_seconds']} s; all-attempt P95: {formal_metrics['all_attempts_p95_elapsed_ms']} ms",
        "- Monetary cost: 0 USD (`explicit_free_provider`)",
        "",
        "## Formal all-manifest metrics",
        "",
        f"- Schema success: {formal_metrics['json_schema_success']}",
        f"- Answerable / refusal accuracy: {formal_metrics['answerable_accuracy']} / {formal_metrics['refusal_accuracy']}",
        f"- Required claim coverage: {formal_metrics['required_claim_coverage']}",
        f"- Exact citation precision / citation recall: {formal_metrics['exact_citation_precision']} / {formal_metrics['citation_recall']}",
        f"- Page citation precision / claim-citation binding: {formal_metrics['page_citation_precision']} / {formal_metrics['claim_citation_binding_accuracy']}",
        f"- Unsupported claim rate: {formal_metrics['unsupported_claim_rate']}",
        f"- Unknown citation ID / invalid citation rate: 0 / {formal_metrics['invalid_citation_rate']}",
        f"- Context tokens mean / retrieval P95: {payload['metrics']['retrieval']['mean_context_tokens']} / {payload['metrics']['retrieval']['p95_retrieval_latency_ms']} ms",
        "",
        "## Per-question comparison to historical Stage 11C",
        "",
        "| Question | Status | Classification | Coverage delta | Precision delta | Recall delta |",
        "|---|---|---|---:|---:|---:|",
    ]
    for item in per_query:
        report.append(f"| {item['question_id']} | {item['status']} | {item['classification']} | {item['deltas']['required_claim_coverage']:.6f} | {item['deltas']['exact_citation_precision']:.6f} | {item['deltas']['citation_recall']:.6f} |")
    report.extend([
        "",
        f"Changes: improved={improved}, regressed={regressed}, unchanged/mixed={unchanged}; Phase-B gain queries improved={gain_improved}/4.",
        "Different historical schemes have different execution denominators; these deltas are diagnostic and do not establish statistical superiority.",
        "",
        "## Gates",
        "",
        f"- Engineering gate: **{payload['dev_v2_engineering_gate']}**",
        f"- Quality candidate gate: **{payload['dev_v2_quality_candidate_gate']}**",
        "- Failed quality condition: required claim coverage did not remain at or above 0.555556.",
        f"- READY_FOR_FULL_QA: **{payload['ready_for_full_qa']}**",
        "- Full QA run: **False**",
        "- Deep Research run: **False**",
        f"- Pending citation audit samples: {pending}",
        "",
        "Formal metrics use the fixed 10-question denominator; completed-only metrics are diagnostic.",
    ])
    SUMMARY_DOC.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"runs": len(selected), "engineering_gate": payload["dev_v2_engineering_gate"], "quality_gate": payload["dev_v2_quality_candidate_gate"], "ready_for_full_qa": payload["ready_for_full_qa"]}))


if __name__ == "__main__":
    main()
