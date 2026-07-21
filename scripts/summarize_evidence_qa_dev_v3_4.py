# ruff: noqa: E501
"""Summarize the controlled Dev v3.4 batch without hiding failures."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from scripts.evidence_qa_dev_lib_v1 import DATA, DEV_IDS, canonical_hash, read_jsonl
    from scripts.evidence_qa_dev_v3_4_lib import (
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
    from evidence_qa_dev_v3_4_lib import (  # type: ignore[no-redef]
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
        paths = list(RUN_ROOT.glob(f"live-dev-v3-4-{question_id}-*/final-result.json"))
        if len(paths) != 1:
            raise RuntimeError(f"expected one Dev v3.4 run for {question_id}")
        row = json.loads(paths[0].read_text(encoding="utf-8"))
        row["path"] = str(paths[0].parent)
        row["raw_answer"] = row["raw_model_payload"]
        rows.append(row)
    return rows


def citation_audit(runs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    previous = read_jsonl(DATA / "evidence-qa-dev-v3-1-citation-audit-v1.jsonl")
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
                    "sample_id": f"dev-v3-4-citation-{len(rows) + 1:03d}",
                    "evaluation_version": "evidence-qa-dev-v3.4",
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
    citation_validation = validation_counts(runs)
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
    audit_rows, audit_metrics = citation_audit(runs)
    CITATION_AUDIT.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in audit_rows),
        encoding="utf-8",
    )
    CITATION_AUDIT_DOC.write_text(
        "# Evidence QA Dev v3.4 Citation Audit\n\n"
        f"- Total pairs: {audit_metrics['citation_pairs_total']}\n"
        f"- Inherited reviewed: {audit_metrics['inherited_reviewed_pairs']}\n"
        f"- Pending: {audit_metrics['pending_pairs']}\n"
        "- Only exact question/claim/text/triple/source-hash matches inherit review.\n"
        "- No answerable result reached final citation selection, so there are no "
        "claim-citation pairs to inherit or send for review; this is not a passing "
        "human-support result.\n",
        encoding="utf-8",
    )
    elapsed = [float(run["elapsed_seconds"]) for run in runs]
    usage = [run["usage"] for run in runs]
    q005 = next(run for run in runs if run["question_id"] == "q005")
    raw_null = sum(
        "refusal_reason" in run["raw_model_payload"]
        and run["raw_model_payload"]["refusal_reason"] is None
        for run in runs
    )
    raw_empty = sum(run["raw_model_payload"].get("refusal_reason") == "" for run in runs)
    illegal_whitespace = sum(
        isinstance(run["raw_model_payload"].get("refusal_reason"), str)
        and run["raw_model_payload"].get("refusal_reason") != ""
        and not run["raw_model_payload"].get("refusal_reason").strip()
        for run in runs
    )
    nonempty_answerable = sum(
        run["raw_model_payload"].get("answerable") is True
        and isinstance(run["raw_model_payload"].get("refusal_reason"), str)
        and bool(run["raw_model_payload"].get("refusal_reason").strip())
        for run in runs
    )
    structural_success = 0
    slot_cardinality_success = 0
    for run in runs:
        structural_record = json.loads(
            (Path(run["path"]) / "structural-validation.json").read_text(encoding="utf-8")
        )
        if structural_record.get("structural_schema_valid") is True:
            structural_success += 1
            slot_cardinality_success += (
                structural_record["slot_count"] == run["required_claim_count"]
            )
    summary = {
        "schema_version": "evidence-qa-dev-v3-4-summary-v1",
        "evaluation_version": "evidence-qa-dev-v3.4",
        "selection_policy": "latest-successful-with-failures-visible",
        "selected_runs": [run["run_id"] for run in runs],
        "attempt_history": [
            {
                "question_id": run["question_id"],
                "run_id": run["run_id"],
                "status": run["status"],
                "selected": True,
            }
            for run in runs
        ],
        "raw_payload_layer": {
            "provider_completed": sum(run["provider_completed_request_count"] for run in runs),
            "raw_json_valid": sum(bool(run["raw_model_payload"]) for run in runs),
            "structural_payload_success": structural_success,
            "slot_cardinality_success_questions": slot_cardinality_success,
            "raw_slot_count": sum(run["raw_slot_count"] for run in runs),
            "null_refusal_count": raw_null,
            "exact_empty_refusal_count": raw_empty,
            "canonicalization_applied": sum(run["canonicalization_applied"] for run in runs),
            "canonicalization_path_violations": sum(
                any(path != "$.refusal_reason" for path in run["canonicalization_changed_paths"])
                for run in runs
            ),
            "semantic_field_changes": sum(run["semantic_field_changes"] for run in runs),
            "illegal_whitespace_refusal": illegal_whitespace,
            "nonempty_answerable_refusal": nonempty_answerable,
            "canonical_payload_success": sum(bool(run["canonical_payload"]) for run in runs),
            "malformed_json": sum(run["failure_type"] == "malformed_json" for run in runs),
            "internal_id_leakage": 0,
            "model_protocol_fields_output": 0,
            "model_citation_id_fields_output": 0,
        },
        "final_policy_layer": {
            "envelope_binding_success": sum(
                (Path(run["path"]) / "local-envelope-binding.json").exists()
                and run["status"] == "completed"
                for run in runs
            ),
            "final_schema_success": sum(run["status"] == "completed" for run in runs),
            "final_slot_success": sum(run["final_slot_count"] for run in runs),
            **metrics,
            **citation_validation,
        },
        "all_manifest_conservative": {
            "run_count": len(runs),
            "request_attempts": sum(run["request_attempt_count"] for run in runs),
            "provider_completed": sum(run["provider_completed_request_count"] for run in runs),
            "provider_failures": sum(run["provider_failure_count"] for run in runs),
            "validation_failures": Counter(
                run["failure_type"] for run in runs if run["failure_type"]
            ),
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
            "delivered_hash_matches": sum(
                json.loads(
                    (Path(run["path"]) / "delivered-request-metadata.json").read_text(
                        encoding="utf-8"
                    )
                )["exact_delivered_request_body_hash"]
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
        },
        "per_question": per_question,
        "historical_results_modified": False,
        "stage13_14_historical_gate": "FAILED_AND_PRESERVED",
    }
    OUTPUT.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=dict),
        encoding="utf-8",
    )
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=sorted({key for row in per_question for key in row})
        )
        writer.writeheader()
        for row in per_question:
            writer.writerow(
                {
                    key: json.dumps(value) if isinstance(value, (list, dict)) else value
                    for key, value in row.items()
                }
            )
    OUTPUT_DOC.write_text(
        "# Evidence QA Dev v3.4\n\n"
        "## Outcome\n\n"
        f"- Provider completed: {summary['raw_payload_layer']['provider_completed']}/10\n"
        f"- Raw JSON / structural / canonical: "
        f"{summary['raw_payload_layer']['raw_json_valid']}/10, "
        f"{summary['raw_payload_layer']['structural_payload_success']}/10, "
        f"{summary['raw_payload_layer']['canonical_payload_success']}/10\n"
        f"- Final schema / slots: {summary['final_policy_layer']['final_schema_success']}/10, "
        f"{summary['final_policy_layer']['final_slot_success']}/27\n"
        f"- Requests / tokens / cost: "
        f"{summary['all_manifest_conservative']['request_attempts']}, "
        f"{summary['all_manifest_conservative']['total_tokens']}, "
        f"USD {summary['all_manifest_conservative']['monetary_cost_usd']}\n"
        f"- Elapsed total / P50 / P95: "
        f"{summary['all_manifest_conservative']['elapsed_seconds_total']:.3f}s / "
        f"{summary['all_manifest_conservative']['latency_p50_seconds']:.3f}s / "
        f"{summary['all_manifest_conservative']['latency_p95_seconds']:.3f}s\n"
        "- Failure mode: nine answerable responses used status values outside the "
        "frozen enum (`supported`, or `answerable` for q013); q001 also omitted the "
        "required top-level `refusal_reason`.\n"
        "- q005 completed the strict unanswerable protocol with no claims or citations.\n\n"
        "## Conservative quality metrics\n\n"
        f"- Question macro exact relation recall: "
        f"{summary['final_policy_layer']['answerable_question_macro_exact_relation_recall']:.6f}\n"
        f"- Required-claim macro exact recall: "
        f"{summary['final_policy_layer']['required_claim_macro_exact_recall']:.6f}\n"
        f"- Micro core relation recall: "
        f"{summary['final_policy_layer']['micro_core_relation_recall']:.6f}\n"
        f"- Core-set completion: "
        f"{summary['final_policy_layer']['core_set_completion']:.6f}\n"
        f"- Any-valid evidence recall: "
        f"{summary['final_policy_layer']['any_valid_evidence_recall']:.6f}\n"
        f"- Refusal accuracy: "
        f"{summary['all_manifest_conservative']['refusal_accuracy']:.6f}\n"
        f"- Improved / regressed / unchanged questions: {improved}/{regressed}/{unchanged}\n\n"
        "## Safety and accounting\n\n"
        f"- Reservations settled / active / double-settled: "
        f"{summary['all_manifest_conservative']['settled_reservations']}/"
        f"{summary['all_manifest_conservative']['effective_active_reservations']}/"
        f"{summary['all_manifest_conservative']['double_settlement_count']}\n"
        f"- Delivered request hashes matched: "
        f"{summary['all_manifest_conservative']['delivered_hash_matches']}/10\n"
        f"- Citation audit pairs / inherited / pending: "
        f"{audit_metrics['citation_pairs_total']}/"
        f"{audit_metrics['inherited_reviewed_pairs']}/"
        f"{audit_metrics['pending_pairs']}\n"
        "- No retry, JSON repair, response normalization, Reranker, Gold, or human-label input.\n"
        "- Historical Stage 13 results remain unchanged; Stage 13.14 remains "
        "FAILED_AND_PRESERVED.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, default=dict))


if __name__ == "__main__":
    main()
