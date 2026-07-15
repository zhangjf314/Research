# ruff: noqa: E501
"""Explicitly summarize isolated Stage 13.2 Dev QA runs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from typing import Any

try:
    from scripts.evidence_qa_dev_lib_v1 import (
        AUDIT_DOC,
        AUDIT_JSONL,
        BLOCKED_C_REASON,
        DATA,
        DEV_IDS,
        GAIN_IDS,
        MANIFEST,
        RUN_ROOT,
        SUMMARY_CSV,
        SUMMARY_DOC,
        SUMMARY_JSON,
        VARIANT_B,
        phase_b_rows,
        read_jsonl,
        slice_metrics,
        summarize_metrics,
    )
except ModuleNotFoundError:
    from evidence_qa_dev_lib_v1 import (  # type: ignore[no-redef]
        AUDIT_DOC,
        AUDIT_JSONL,
        BLOCKED_C_REASON,
        DATA,
        DEV_IDS,
        GAIN_IDS,
        MANIFEST,
        RUN_ROOT,
        SUMMARY_CSV,
        SUMMARY_DOC,
        SUMMARY_JSON,
        VARIANT_B,
        phase_b_rows,
        read_jsonl,
        slice_metrics,
        summarize_metrics,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-policy", choices=("latest-successful", "latest-attempt"), default="latest-successful")
    parser.add_argument("--explicit-run-id", action="append", default=[])
    return parser.parse_args()


def historical_rows() -> list[dict[str, Any]]:
    source = json.loads((DATA / "qa-production-v1.json").read_text(encoding="utf-8"))
    protocol = {row["question_id"]: row for row in read_jsonl(DATA / "retrieval-gold-v2.jsonl")}
    output = []
    for raw in source["queries"]:
        if raw["question_id"] not in DEV_IDS:
            continue
        if raw["retrieval_query"] != protocol[raw["question_id"]]["retrieval_query"]:
            raise RuntimeError(f"historical Stage 11C query mismatch: {raw['question_id']}")
        row = json.loads(json.dumps(raw))
        row["status"] = "completed" if raw["status"] == "COMPLETED" else "provider_failed"
        row["category"] = next(item["category"] for item in read_jsonl(DATA / "gold-set-v1.jsonl") if item["question_id"] == raw["question_id"])
        row["difficulty"] = next(item["difficulty"] for item in read_jsonl(DATA / "gold-set-v1.jsonl") if item["question_id"] == raw["question_id"])
        row["elapsed_seconds"] = raw.get("answer", {}).get("latency", {}).get("total_latency_ms", 0) / 1000
        row["usage"] = raw.get("answer", {}).get("model_usage", {})
        row["request_attempt_count"] = raw.get("api_request_count", 0)
        row["provider_completed_request_count"] = int(row["status"] == "completed")
        row["active_reserved_tokens"] = 0
        row["citation_retry_count"] = raw.get("retry_count", 0)
        row["metrics"] = {
            "answerable_correct": raw["metrics"]["answerable_correct"],
            "required_claim_coverage": raw["metrics"]["required_claim_coverage"],
            "omitted_required_claims": sum(item["best"] < 0.35 for item in raw["metrics"]["required_claim_scores"]),
            "unsupported_before_generation": 0,
            "unsupported_after_generation": raw["metrics"]["unsupported_claim_count"],
            "extra_claims": max(0, len(raw["answer"]["claims"]) - len(raw["gold"]["required_claims"])),
            "exact_citation_precision": raw["metrics"]["citation_precision"],
            "citation_recall": raw["metrics"]["citation_recall"],
            "page_citation_precision": None,
            "claim_citation_binding_accuracy": raw["metrics"]["claim_citation_binding_rate"],
            "allocated_to_correct_claim_rate": None,
            "invalid_citation_rate": 1 - raw["metrics"]["citation_id_validity"],
        }
        output.append(row)
    return sorted(output, key=lambda item: DEV_IDS.index(item["question_id"]))


def select_runs(policy: str, explicit: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    attempts: dict[str, list[tuple[float, dict[str, Any], str]]] = defaultdict(list)
    for path in (RUN_ROOT / VARIANT_B).glob("*/result.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        attempts[row["question_id"]].append((path.stat().st_mtime, row, str(path.parent)))
    explicit_set = set(explicit)
    selected = []
    history = []
    for question_id in DEV_IDS:
        candidates = sorted(attempts.get(question_id, []), key=lambda item: item[0])
        if explicit_set:
            candidates = [item for item in candidates if item[1]["run_id"] in explicit_set]
        if not candidates:
            continue
        if policy == "latest-successful":
            successes = [item for item in candidates if item[1]["status"] == "completed"]
            chosen = successes[-1] if successes else candidates[-1]
        else:
            chosen = candidates[-1]
        selected.append(chosen[1])
        history.extend({"question_id": question_id, "run_id": item[1]["run_id"], "status": item[1]["status"], "selected": item[1]["run_id"] == chosen[1]["run_id"], "path": item[2]} for item in candidates)
    return selected, history


def retrieval_metrics(question_ids: list[str]) -> dict[str, Any]:
    _, candidate = phase_b_rows()
    rows = [candidate[qid]["metrics"] for qid in question_ids]
    answerable = [row for row in rows if row["answerable"]]
    return {
        "exact_block_availability": sum(row["exact_gold_block_available"] for row in answerable) / len(answerable),
        "gold_page_availability": sum(row["gold_page_available"] for row in answerable) / len(answerable),
        "gold_block_recall": sum(row["gold_block_recall"] for row in answerable) / len(answerable),
        "claim_evidence_recall": None,
        "complete_evidence_set_recall": None,
        "metadata_contamination": sum(row["metadata_contamination_rate"] for row in rows) / len(rows),
        "adjacent_completion_contribution_queries": sorted(GAIN_IDS & set(question_ids)),
        "mean_context_tokens": sum(row["context_token_count"] for row in rows) / len(rows),
        "p95_retrieval_latency_ms": sorted(row["latency_ms"] for row in rows)[-1],
    }


def generate_citation_audit(rows: list[dict[str, Any]]) -> int:
    gold = {row["question_id"]: row for row in read_jsonl(DATA / "gold-set-v1.jsonl")}
    evidence = {(row["paper_id"], int(row["page"]), row["block_id"]): row for row in read_jsonl(DATA / "evidence-corpus-v1.jsonl")}
    samples = []
    for row in rows:
        if row["status"] != "completed":
            continue
        for claim in row["answer"].get("claims", []):
            for citation in claim["citations"]:
                triple = (citation["paper_id"], int(citation["page"]), citation["block_id"])
                unit = evidence[triple]
                exact = citation["block_id"] in gold[row["question_id"]]["gold_block_ids"]
                same_page = int(citation["page"]) in gold[row["question_id"]]["gold_pages"]
                samples.append(
                    {
                        "sample_id": f"evidence-qa-dev-citation-{len(samples)+1:03d}",
                        "variant": VARIANT_B,
                        "question_id": row["question_id"],
                        "claim_id": claim["claim_id"],
                        "claim_text": claim["claim_text"],
                        "citation": citation,
                        "cited_evidence_text": unit["text"],
                        "adjacent_evidence_context": {"previous_block_id": unit.get("previous_block_id"), "next_block_id": unit.get("next_block_id")},
                        "gold_blocks": gold[row["question_id"]]["gold_block_ids"],
                        "gold_pages": gold[row["question_id"]]["gold_pages"],
                        "automated_labels": {"exact_gold": exact, "same_page": same_page, "semantic_support_signal": None, "unsupported_signal": not exact and not same_page},
                        "human_review_status": "pending",
                        "human_label": None,
                        "reviewer": None,
                        "reviewed_at": None,
                        "review_notes": None,
                    }
                )
    # At this Dev scale all pairs are retained up to the protocol cap.
    samples = samples[:60]
    with AUDIT_JSONL.open("w", encoding="utf-8") as stream:
        for item in samples:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    AUDIT_DOC.write_text("# Evidence QA Dev Citation Audit v1\n\n" f"- Samples: {len(samples)}\n" "- Human status: **pending** for every sample.\n" "- Automated exact/page/support signals are not human citation correctness.\n", encoding="utf-8")
    return len(samples)


def main() -> None:
    args = parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    historical = historical_rows()
    live, history = select_runs(args.selection_policy, args.explicit_run_id)
    if len(live) != len(DEV_IDS):
        raise RuntimeError(f"expected 10 selected live runs, found {len(live)}")
    historical_metrics = summarize_metrics(historical)
    live_metrics = summarize_metrics(live)
    per_query = []
    historical_by_id = {row["question_id"]: row for row in historical}
    for row in live:
        base = historical_by_id[row["question_id"]]
        per_query.append(
            {
                "question_id": row["question_id"],
                "phase_b_gain_query": row["question_id"] in GAIN_IDS,
                "baseline": base["metrics"],
                VARIANT_B: row["metrics"],
                "claim_coverage_delta": (row["metrics"].get("required_claim_coverage") or 0) - (base["metrics"].get("required_claim_coverage") or 0),
                "exact_precision_delta": (row["metrics"].get("exact_citation_precision") or 0) - (base["metrics"].get("exact_citation_precision") or 0),
                "citation_recall_delta": (row["metrics"].get("citation_recall") or 0) - (base["metrics"].get("citation_recall") or 0),
            }
        )
    non_regressed = sum(item["claim_coverage_delta"] >= 0 and item["exact_precision_delta"] >= 0 for item in per_query)
    improved = sum(item["claim_coverage_delta"] > 0 or item["exact_precision_delta"] > 0 or item["citation_recall_delta"] > 0 for item in per_query)
    regressed = sum(item["claim_coverage_delta"] < 0 or item["exact_precision_delta"] < 0 or item["citation_recall_delta"] < 0 for item in per_query)
    precision_improved_questions = sum(item["exact_precision_delta"] > 0 for item in per_query)
    gain_improved = sum(item["phase_b_gain_query"] and (item["claim_coverage_delta"] > 0 or item["exact_precision_delta"] > 0 or item["citation_recall_delta"] > 0) for item in per_query)
    automatic_gates = {
        "required_claim_coverage_above_historical_0_388889": (live_metrics["required_claim_coverage"] or 0) > 0.388889,
        "required_claim_coverage_above_dev_baseline": (live_metrics["required_claim_coverage"] or 0) > (historical_metrics["required_claim_coverage"] or 0),
        "exact_citation_precision_above_historical_0_103009": (live_metrics["exact_citation_precision"] or 0) > 0.103009,
        "exact_precision_improves_at_least_3_queries": precision_improved_questions >= 3,
        "citation_recall_above_historical_0_096875": (live_metrics["citation_recall"] or 0) > 0.096875,
        "unsupported_claim_rate_lower_than_dev_baseline": live_metrics["unsupported_claim_rate"] < historical_metrics["unsupported_claim_rate"],
        "invalid_citation_rate_not_higher": live_metrics["invalid_citation_rate"] <= historical_metrics["invalid_citation_rate"],
        "answerable_accuracy_not_lower": (live_metrics["answerable_accuracy"] or 0) >= (historical_metrics["answerable_accuracy"] or 0),
        "refusal_accuracy_not_lower": (live_metrics["refusal_accuracy"] or 0) >= (historical_metrics["refusal_accuracy"] or 0),
        "at_least_2_of_4_phase_b_gain_queries_improve": gain_improved >= 2,
        "at_least_6_of_10_non_regressed": non_regressed >= 6,
        "improved_more_than_regressed": improved > regressed,
        "no_provider_or_execution_failures": live_metrics["execution_failures"] == 0,
        "active_reservations_zero": live_metrics["active_reserved_tokens"] == 0,
        "tokens_within_budget": live_metrics["total_tokens"] <= 300000,
        "elapsed_within_budget": (live_metrics["elapsed"]["mean_ms"] or 0) * len(live) <= 1_800_000,
    }
    citation_count = generate_citation_audit(live)
    payload = {
        "schema_version": "evidence-qa-dev-v1",
        "status": "COMPLETED",
        "manifest_hash": manifest["manifest_hash"],
        "selection_policy": args.selection_policy,
        "baseline_reused": True,
        "baseline_source": "historical_stage11c",
        "variants": {
            "historical_stage11c": {"metrics": historical_metrics, "retrieval": None, "slices": slice_metrics(historical)},
            VARIANT_B: {"metrics": live_metrics, "retrieval": retrieval_metrics(DEV_IDS), "slices": slice_metrics(live)},
            "evidence_centric": {"status": BLOCKED_C_REASON, "metrics": None},
        },
        "selected_runs": [row["run_id"] for row in live],
        "attempt_history": history,
        "per_query_comparison": per_query,
        "change_counts": {"improved": improved, "regressed": regressed, "unchanged_or_mixed_non_regressed": non_regressed, "precision_improved_questions": precision_improved_questions, "phase_b_gain_queries_improved": gain_improved},
        "automatic_quality_candidate_gates": automatic_gates,
        "dev_engineering_gate": all([live_metrics["execution_failures"] == 0, live_metrics["active_reserved_tokens"] == 0, live_metrics["invalid_citation_rate"] == 0]),
        "dev_quality_candidate_gate": all(automatic_gates.values()),
        "human_citation_quality_gate": "pending",
        "citation_audit_pending_count": citation_count,
        "ready_for_full_qa": all(automatic_gates.values()),
        "full_qa_run": False,
        "deep_research_run": False,
        "reranker_enabled": False,
        "gold_leakage": False,
        "oracle_leakage": False,
        "human_pilot_evidence_used_for_selection": False,
    }
    SUMMARY_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = []
    for name in ("historical_stage11c", VARIANT_B):
        metrics = payload["variants"][name]["metrics"]
        rows.append({"variant": name, **metrics})
    with SUMMARY_CSV.open("w", encoding="utf-8", newline="") as stream:
        flat_rows = [{key: json.dumps(value, sort_keys=True) if isinstance(value, dict) else value for key, value in row.items()} for row in rows]
        writer = csv.DictWriter(stream, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    SUMMARY_DOC.write_text(
        "# Evidence-Centric Dev QA Evaluation v1\n\n"
        f"- Manifest: `{manifest['manifest_hash']}`\n"
        "- Baseline: reused `historical_stage11c` (zero new requests)\n"
        f"- Retrieval-only B: {live_metrics['completed']}/10 completed\n"
        f"- Evidence-centric C: `{BLOCKED_C_REASON}`\n"
        f"- Required claim coverage, A/B: {historical_metrics['required_claim_coverage']} / {live_metrics['required_claim_coverage']}\n"
        f"- Exact citation precision, A/B: {historical_metrics['exact_citation_precision']} / {live_metrics['exact_citation_precision']}\n"
        f"- Citation recall, A/B: {historical_metrics['citation_recall']} / {live_metrics['citation_recall']}\n"
        f"- Unsupported claim rate, A/B: {historical_metrics['unsupported_claim_rate']} / {live_metrics['unsupported_claim_rate']}\n"
        f"- Pending human citation samples: {citation_count}\n"
        f"- Automatic quality candidate gate: **{payload['dev_quality_candidate_gate']}**\n"
        f"- READY_FOR_FULL_QA: **{payload['ready_for_full_qa']}**\n"
        "- Full QA run: **False**\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": payload["status"], "metrics": live_metrics, "ready_for_full_qa": payload["ready_for_full_qa"], "citation_audit_pending": citation_count}))


if __name__ == "__main__":
    main()
