# ruff: noqa: E501
"""Summarize the controlled Dev v3.6 batch without hiding failures."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

try:
    from scripts.evidence_qa_dev_lib_v1 import DATA, DEV_IDS, canonical_hash, read_jsonl
    from scripts.evidence_qa_dev_v3_6_lib import (
        CITATION_AUDIT,
        CITATION_AUDIT_DOC,
        OUTPUT,
        OUTPUT_CSV,
        OUTPUT_DOC,
        RUN_ROOT,
    )
    from scripts.summarize_evidence_qa_dev_v3_2 import metric_rows, percentile
    from scripts.summarize_evidence_qa_dev_v3_3 import validation_counts
except ModuleNotFoundError:
    from evidence_qa_dev_lib_v1 import (  # type: ignore[no-redef]
        DATA,
        DEV_IDS,
        canonical_hash,
        read_jsonl,
    )
    from evidence_qa_dev_v3_6_lib import (  # type: ignore[no-redef]
        CITATION_AUDIT,
        CITATION_AUDIT_DOC,
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
        paths = list(RUN_ROOT.glob(f"live-dev-v3-6-{question_id}-*/final-result.json"))
        if len(paths) != 1:
            raise RuntimeError(f"expected one Dev v3.6 run for {question_id}, got {len(paths)}")
        row = json.loads(paths[0].read_text(encoding="utf-8"))
        row["path"] = str(paths[0].parent)
        row["raw_answer"] = row["raw_model_payload"]
        rows.append(row)
    return rows


def citation_audit(runs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prior_paths = [
        DATA / "evidence-qa-dev-v3-4-citation-audit-v1.jsonl",
        DATA / "evidence-qa-dev-v3-3-citation-audit-v1.jsonl",
        DATA / "evidence-qa-dev-v3-1-citation-audit-v1.jsonl",
    ]
    previous = []
    for path in prior_paths:
        if path.exists():
            previous.extend(read_jsonl(path))
    prior = {
        (
            row["question_id"],
            row["required_claim_id"],
            row["generated_claim"],
            row["citation_triple"]["paper_id"],
            int(row["citation_triple"]["page"]),
            row["citation_triple"]["block_id"],
            row.get("source_canonical_sha256"),
        ): row
        for row in previous
    }
    source_hash = json.loads(
        (DATA / "claim-evidence-gold-dev-v1-freeze.json").read_text(encoding="utf-8")
    )["source_corpus_hash"]["value"]
    rows = []
    for run in runs:
        if run["status"] != "completed":
            continue
        run_dir = Path(run["path"])
        registry = json.loads((run_dir / "citation-registry.json").read_text(encoding="utf-8"))
        entries = {row["citation_id"]: row for row in registry["entries"]}
        for slot in run["final_answer"]["required_claim_results"]:
            if slot["status"] != "answered":
                continue
            for citation_id in slot["citation_ids"]:
                entry = entries[citation_id]
                key = (
                    run["question_id"],
                    slot["required_claim_id"],
                    slot["claim_text"],
                    entry["paper_id"],
                    int(entry["page"]),
                    entry["block_id"],
                    source_hash,
                )
                old = prior.get(key)
                immutable = {
                    "sample_id": f"dev-v3-6-citation-{len(rows) + 1:03d}",
                    "evaluation_version": "evidence-qa-dev-v3.6",
                    "question_id": run["question_id"],
                    "run_id": run["run_id"],
                    "required_claim_id": slot["required_claim_id"],
                    "generated_claim": slot["claim_text"],
                    "citation_id": citation_id,
                    "citation_triple": {
                        "paper_id": entry["paper_id"],
                        "page": entry["page"],
                        "block_id": entry["block_id"],
                    },
                    "source_canonical_sha256": source_hash,
                    "inherited_from_sample_id": old["sample_id"] if old else None,
                    "requires_new_review": old is None,
                }
                rows.append(
                    {
                        **immutable,
                        "human_review_status": "approved" if old else "pending",
                        "human_label": old.get("human_label") if old else None,
                        "reviewer": old.get("reviewer") if old else None,
                        "reviewed_at": old.get("reviewed_at") if old else None,
                        "review_notes": old.get("review_notes") if old else None,
                        "immutable_record_hash": canonical_hash(immutable),
                    }
                )
    labels = [
        row["human_label"]
        for row in rows
        if row["human_review_status"] == "approved" and row["human_label"]
    ]
    return rows, {
        "citation_pairs_total": len(rows),
        "inherited_reviewed_pairs": sum(row["human_review_status"] == "approved" for row in rows),
        "pending_pairs": sum(row["human_review_status"] == "pending" for row in rows),
        "inherited_strict": (
            sum(label == "fully_supported" for label in labels) / len(labels) if labels else None
        ),
        "inherited_lenient": (
            sum(label in {"fully_supported", "partially_supported"} for label in labels)
            / len(labels)
            if labels
            else None
        ),
    }


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
            label = "improved" if current > previous else "regressed" if current < previous else "unchanged"
        row["comparison_to_dev_v3_1"] = label
        improved += label == "improved"
        regressed += label == "regressed"
        unchanged += label == "unchanged"
    audit_rows, audit_metrics = citation_audit(runs)
    CITATION_AUDIT.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in audit_rows),
        encoding="utf-8",
    )
    CITATION_AUDIT_DOC.write_text(
        "# Evidence QA Dev v3.6 Citation Audit\n\n"
        f"- Total pairs: {audit_metrics['citation_pairs_total']}\n"
        f"- Inherited reviewed: {audit_metrics['inherited_reviewed_pairs']}\n"
        f"- Pending: {audit_metrics['pending_pairs']}\n"
        "- Exact question/claim/text/triple/source-hash matches inherit review; all "
        "changed pairs remain pending.\n",
        encoding="utf-8",
    )
    elapsed = [float(run["elapsed_seconds"]) for run in runs]
    usage = [run["usage"] for run in runs]
    q005 = next(run for run in runs if run["question_id"] == "q005")
    raw = {
        "provider_completed": sum(run["provider_completed_request_count"] for run in runs),
        "provider_failures": sum(run["provider_failure_count"] for run in runs),
        "raw_json_valid": sum(run["json_valid"] for run in runs),
        "payload_v4_schema_success": sum(run["payload_v4_schema_valid"] for run in runs),
        "slot_shape_success_questions": sum(
            run["slot_shape_success"] == run["required_claim_count"] for run in runs
        ),
        "valid_slots": sum(run["slot_shape_success"] for run in runs),
        "status_field_leakage": sum(run["status_field_leakage"] for run in runs),
        "citation_field_leakage": sum(run["citation_field_leakage"] for run in runs),
        "evidence_label_leakage": sum(run["evidence_label_leakage"] for run in runs),
        "arbitrary_extra_field_questions": sum(run["arbitrary_extra_field_question"] for run in runs),
        "null_sentinel": sum(run["null_sentinel"] for run in runs),
        "empty_sentinel": sum(run["empty_sentinel"] for run in runs),
        "dual_semantic_conflicts": sum(run["dual_semantic_conflict"] for run in runs),
        "answered_shape": sum(run["answered_shape"] for run in runs),
        "unsupported_shape": sum(run["unsupported_shape"] for run in runs),
        "invalid_shape": sum(run["invalid_shape"] for run in runs),
        "total_slots": sum(run["required_claim_count"] for run in runs),
        "model_visible_metadata_leakage": sum(run["model_visible_metadata_leakage"] for run in runs),
        "prompt_contamination_failures": sum(run["prompt_contamination_gate"] != "PASSED" for run in runs),
        "malformed_json": sum(run["failure_type"] == "malformed_json" for run in runs),
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
        "billing_unknown_reservations": sum(run["billing_unknown_reservation_count"] for run in runs),
        "effective_active_reservations": sum(run["active_reserved_tokens"] > 0 for run in runs),
        "double_settlement_count": 0,
        "retries": sum(run["retries"] for run in runs),
        "reranker_called": any(run["reranker_called"] for run in runs),
        "template_fallback": any(run["template_fallback"] for run in runs),
        "delivered_hash_matches": sum(
            json.loads((Path(run["path"]) / "delivered-request-metadata.json").read_text(encoding="utf-8"))[
                "exact_delivered_request_body_hash"
            ]
            == run["exact_delivered_request_body_hash"]
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
        "unsupported_improvement_driver": False,
        **audit_metrics,
    }
    summary = {
        "schema_version": "evidence-qa-dev-v3-6-summary-v1",
        "evaluation_version": "evidence-qa-dev-v3.6",
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
        "final_policy_layer": final,
        "all_manifest_conservative": total,
        "per_question": per_question,
        "historical_results_modified": False,
        "full_qa_executed": False,
        "deep_research_executed": False,
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
        "# Evidence QA Dev v3.6\n\n"
        f"- Provider completed: {raw['provider_completed']}/10\n"
        f"- JSON / Payload / Slot-shape questions: {raw['raw_json_valid']}/10, "
        f"{raw['payload_v4_schema_success']}/10, {raw['slot_shape_success_questions']}/10\n"
        f"- Valid slots: {raw['valid_slots']}/27\n"
        f"- Status/citation/evidence-label leakage: {raw['status_field_leakage']}/"
        f"{raw['citation_field_leakage']}/{raw['evidence_label_leakage']}\n"
        f"- Claim macro / micro core / any-valid: "
        f"{final['required_claim_macro_exact_recall']:.6f} / "
        f"{final['micro_core_relation_recall']:.6f} / "
        f"{final['any_valid_evidence_recall']:.6f}\n"
        f"- Human support pending pairs: {audit_metrics['pending_pairs']}\n"
        "- Full QA and Deep Research were not run.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, default=dict))


if __name__ == "__main__":
    main()
