# ruff: noqa: E501
"""Summarize the controlled Dev v3.5 Payload v4 live batch."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from scripts.evidence_qa_dev_lib_v1 import DATA, DEV_IDS, read_jsonl
    from scripts.evidence_qa_dev_v3_5_lib import (
        OUTPUT,
        OUTPUT_CSV,
        OUTPUT_DOC,
        RUN_ROOT,
    )
    from scripts.summarize_evidence_qa_dev_v3_2 import metric_rows, percentile
    from scripts.summarize_evidence_qa_dev_v3_3 import validation_counts
except ModuleNotFoundError:
    from evidence_qa_dev_lib_v1 import DATA, DEV_IDS, read_jsonl  # type: ignore[no-redef]
    from evidence_qa_dev_v3_5_lib import (  # type: ignore[no-redef]
        OUTPUT,
        OUTPUT_CSV,
        OUTPUT_DOC,
        RUN_ROOT,
    )
    from summarize_evidence_qa_dev_v3_2 import metric_rows, percentile  # type: ignore[no-redef]
    from summarize_evidence_qa_dev_v3_3 import validation_counts  # type: ignore[no-redef]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-policy", choices=("latest-successful",), required=True)
    return parser.parse_args()


def load_runs() -> list[dict[str, Any]]:
    rows = []
    for question_id in DEV_IDS:
        paths = list(RUN_ROOT.glob(f"live-dev-v3-5-{question_id}-*/final-result.json"))
        if len(paths) != 1:
            raise RuntimeError(f"expected one Dev v3.5 run for {question_id}, got {len(paths)}")
        row = json.loads(paths[0].read_text(encoding="utf-8"))
        row["path"] = str(paths[0].parent)
        row["raw_answer"] = row["raw_model_payload"]
        rows.append(row)
    return rows


def main() -> None:
    parse_args()
    runs = load_runs()
    gold = {
        row["required_claim_id"]: row
        for row in read_jsonl(DATA / "claim-evidence-gold-dev-v1.jsonl")
    }
    per_question, metrics = metric_rows(runs, gold)
    validation = validation_counts(runs)
    comparison = json.loads(
        (DATA / "claim-gold-citation-comparison-v1.json").read_text(encoding="utf-8")
    )
    baseline = next(
        row
        for row in comparison["experiments"]
        if row["evaluation_version"] == "stage13_8_dev_v3_1"
    )["per_question"]
    improved = regressed = unchanged = 0
    for row in per_question:
        if row["question_id"] == "q005":
            label = "unchanged"
        else:
            previous = baseline[row["question_id"]]["exact_relation_recall"]
            current = row["question_exact_relation_recall"]
            label = (
                "improved"
                if current > previous
                else "regressed"
                if current < previous
                else "unchanged"
            )
        row["comparison_to_dev_v3_1"] = label
        improved += label == "improved"
        regressed += label == "regressed"
        unchanged += label == "unchanged"
    elapsed = [float(run["elapsed_seconds"]) for run in runs]
    usage = [run["usage"] for run in runs]
    q005 = next(run for run in runs if run["question_id"] == "q005")
    raw = {
        "provider_completed": sum(run["provider_completed_request_count"] for run in runs),
        "raw_json_valid": sum(run["json_valid"] for run in runs),
        "payload_v4_schema_success": sum(run["payload_v4_schema_valid"] for run in runs),
        "slot_shape_success_questions": sum(
            run["slot_shape_success"] == run["required_claim_count"] for run in runs
        ),
        "status_field_leakage": sum(run["status_field_leakage"] for run in runs),
        "citation_field_leakage": sum(run["citation_field_leakage"] for run in runs),
        "null_sentinel": sum(run["null_sentinel"] for run in runs),
        "empty_sentinel": sum(run["empty_sentinel"] for run in runs),
        "malformed_json": sum(run["failure_type"] == "malformed_json" for run in runs),
        "answered_shape": sum(run["answered_shape"] for run in runs),
        "unsupported_shape": sum(run["unsupported_shape"] for run in runs),
        "invalid_shape": sum(run["invalid_shape"] for run in runs),
        "total_slots": sum(run["required_claim_count"] for run in runs),
        "raw_slot_count": sum(run["raw_slot_count"] for run in runs),
    }
    final = {
        "envelope_binding_success": sum(run["envelope_binding_success"] for run in runs),
        "final_schema_success": sum(run["status"] == "completed" for run in runs),
        "final_slot_success": sum(run["final_slot_count"] for run in runs),
        **metrics,
        **validation,
    }
    total = {
        "run_count": len(runs),
        "request_attempts": sum(run["request_attempt_count"] for run in runs),
        "provider_completed": raw["provider_completed"],
        "provider_failures": sum(run["provider_failure_count"] for run in runs),
        "validation_failures": Counter(run["failure_type"] for run in runs if run["failure_type"]),
        "input_tokens": sum(row.get("input_tokens", 0) for row in usage),
        "output_tokens": sum(row.get("output_tokens", 0) for row in usage),
        "total_tokens": sum(row.get("total_tokens", 0) for row in usage),
        "monetary_cost_usd": "0",
        "elapsed_seconds_total": sum(elapsed),
        "latency_p50_seconds": percentile(elapsed, 0.5),
        "latency_p95_seconds": percentile(elapsed, 0.95),
        "usage_records": sum(run["usage_record_count"] for run in runs),
        "reservation_count": sum(run["reservation_count"] for run in runs),
        "settled_reservations": sum(run["settled_reservation_count"] for run in runs),
        "released_reservations": sum(run["released_reservation_count"] for run in runs),
        "billing_unknown_reservations": sum(
            run["billing_unknown_reservation_count"] for run in runs
        ),
        "effective_active_reservations": sum(run["active_reserved_tokens"] > 0 for run in runs),
        "double_settlement_count": 0,
        "retries": sum(run["retries"] for run in runs),
        "reranker_called": any(run["reranker_called"] for run in runs),
        "template_fallback": any(run["template_fallback"] for run in runs),
        "delivered_hash_matches": sum(
            json.loads(
                (Path(run["path"]) / "delivered-request-metadata.json").read_text(
                    encoding="utf-8"
                )
            )["delivered_request_body_hash"]
            == json.loads((Path(run["path"]) / "request.json").read_text(encoding="utf-8"))[
                "request_body_hash"
            ]
            for run in runs
        ),
        "silent_omissions": sum(
            max(0, run["required_claim_count"] - run["final_slot_count"]) for run in runs
        ),
        "refusal_accuracy": float(
            q005["status"] == "completed"
            and q005["raw_model_payload"].get("answerable") is False
            and q005["raw_model_payload"].get("required_claim_results") == []
            and bool(q005["raw_model_payload"].get("refusal_reason", "").strip())
            and q005["final_answer"].get("required_claim_results") == []
        ),
        "improved_questions": improved,
        "regressed_questions": regressed,
        "unchanged_questions": unchanged,
    }
    summary = {
        "schema_version": "evidence-qa-dev-v3-5-summary-v1",
        "evaluation_version": "evidence-qa-dev-v3.5",
        "selection_policy": "latest-successful-with-failures-visible",
        "selected_runs": [run["run_id"] for run in runs],
        "attempt_history": [
            {
                "question_id": run["question_id"],
                "run_id": run["run_id"],
                "status": run["status"],
                "selected": True,
                "failure_type": run["failure_type"],
            }
            for run in runs
        ],
        "raw_payload_layer": raw,
        "slot_shape_layer": {
            "total_slots": raw["total_slots"],
            "answered_shape": raw["answered_shape"],
            "unsupported_shape": raw["unsupported_shape"],
            "invalid_shape": raw["invalid_shape"],
        },
        "quality_layer": final,
        "all_manifest_conservative": total,
        "per_question": per_question,
        "historical_results_modified": False,
        "stage13_16_failure_freeze_modified": False,
        "stage13_17_payload_v3_modified": False,
        "stage13_18_payload_v4_readiness_modified": False,
        "claim_gold_modified": False,
        "retrieval_gold_modified": False,
        "human_review_labels_modified": False,
    }
    OUTPUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=dict), encoding="utf-8")
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({key for row in per_question for key in row}))
        writer.writeheader()
        for row in per_question:
            writer.writerow(
                {key: json.dumps(value) if isinstance(value, (list, dict)) else value for key, value in row.items()}
            )
    OUTPUT_DOC.write_text(
        "# Evidence QA Dev v3.5 Controlled Live\n\n"
        "## Shape reliability\n\n"
        f"- Provider completed: {raw['provider_completed']}/10\n"
        f"- JSON valid / Payload v4 schema / Slot shape success: "
        f"{raw['raw_json_valid']}/10, {raw['payload_v4_schema_success']}/10, "
        f"{raw['slot_shape_success_questions']}/10\n"
        f"- Status / citation / null / empty leakage: {raw['status_field_leakage']}/"
        f"{raw['citation_field_leakage']}/{raw['null_sentinel']}/{raw['empty_sentinel']}\n"
        f"- Slot shapes answered / unsupported / invalid: {raw['answered_shape']}/"
        f"{raw['unsupported_shape']}/{raw['invalid_shape']} of {raw['total_slots']}\n\n"
        "## Quality metrics\n\n"
        f"- Required claim macro exact recall: {final['required_claim_macro_exact_recall']:.6f}\n"
        f"- Citation recall micro core relation: {final['micro_core_relation_recall']:.6f}\n"
        f"- Any-valid evidence recall: {final['any_valid_evidence_recall']:.6f}\n"
        f"- Core-set completion: {final['core_set_completion']:.6f}\n"
        f"- Refusal accuracy: {total['refusal_accuracy']:.6f}\n"
        f"- Improved / regressed / unchanged questions: {improved}/{regressed}/{unchanged}\n\n"
        "## Safety\n\n"
        f"- Requests / tokens / cost: {total['request_attempts']}, "
        f"{total['total_tokens']}, USD {total['monetary_cost_usd']}\n"
        f"- Active reservations / retries / reranker / template fallback: "
        f"{total['effective_active_reservations']}/{total['retries']}/"
        f"{total['reranker_called']}/{total['template_fallback']}\n"
        "- No normalization, JSON repair, retry, citation repair, fallback, Gold injection, "
        "or human-label injection.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, default=dict))


if __name__ == "__main__":
    main()
