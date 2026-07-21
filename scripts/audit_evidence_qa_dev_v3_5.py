"""Apply frozen Dev v3.5 engineering and automated quality gates."""

from __future__ import annotations

import json

try:
    from scripts.evidence_qa_dev_v3_5_lib import (
        FAILURE_FREEZE,
        FAILURE_FREEZE_DOC,
        FINAL_AUDIT,
        OUTPUT,
    )
except ModuleNotFoundError:
    from evidence_qa_dev_v3_5_lib import (  # type: ignore[no-redef]
        FAILURE_FREEZE,
        FAILURE_FREEZE_DOC,
        FINAL_AUDIT,
        OUTPUT,
    )


def main() -> None:
    summary = json.loads(OUTPUT.read_text(encoding="utf-8"))
    raw = summary["raw_payload_layer"]
    slots = summary["slot_shape_layer"]
    quality = summary["quality_layer"]
    total = summary["all_manifest_conservative"]
    engineering_checks = {
        "provider_completed_ge_9": raw["provider_completed"] >= 9,
        "json_valid_ge_9": raw["raw_json_valid"] >= 9,
        "payload_v4_schema_ge_9": raw["payload_v4_schema_success"] >= 9,
        "slot_shape_success_ge_9": raw["slot_shape_success_questions"] >= 9,
        "status_field_leakage_zero": raw["status_field_leakage"] == 0,
        "citation_field_leakage_zero": raw["citation_field_leakage"] == 0,
        "null_sentinel_zero": raw["null_sentinel"] == 0,
        "empty_sentinel_zero": raw["empty_sentinel"] == 0,
        "invalid_shape_zero": slots["invalid_shape"] == 0,
        "q005_refusal": total["refusal_accuracy"] == 1.0,
        "delivered_hashes": total["delivered_hash_matches"] == total["run_count"],
        "usage_complete": total["usage_records"] == total["provider_completed"],
        "ledger_closed": total["effective_active_reservations"] == 0,
        "no_retries": total["retries"] == 0,
        "reranker_disabled": total["reranker_called"] is False,
        "template_fallback_false": total["template_fallback"] is False,
    }
    engineering = all(engineering_checks.values())
    quality_checks = {
        "required_claim_coverage_gt_0_592593": quality[
            "required_claim_macro_exact_recall"
        ]
        > 0.592593,
        "citation_recall_not_below_0_295000": quality["micro_core_relation_recall"]
        >= 0.295000,
        "any_valid_evidence_recall_not_below_0_295833": quality[
            "any_valid_evidence_recall"
        ]
        >= 0.295833,
        "core_set_completion_not_below_0_148148": quality["core_set_completion"]
        >= 0.148148,
        "unknown_zero": quality["unknown_citation_id"] == 0,
        "invalid_zero": quality["invalid_citation_id"] == 0,
        "cross_claim_zero": quality["cross_claim_citation"] == 0,
        "silent_omission_zero": total["silent_omissions"] == 0,
        "refusal_accuracy": total["refusal_accuracy"] == 1.0,
        "not_single_question_only": total["improved_questions"] >= 3,
    }
    quality_gate = engineering and all(quality_checks.values())
    audit = {
        "schema_version": "evidence-qa-dev-v3-5-final-audit-v1",
        "dev_v3_5_engineering_checks": engineering_checks,
        "dev_v3_5_quality_candidate_checks": quality_checks,
        "provider_health": "PASSED",
        "safe_to_start_batch": True,
        "dev_v3_5_engineering_gate": "PASSED" if engineering else "FAILED",
        "dev_v3_5_quality_candidate_gate": "PASSED" if quality_gate else "FAILED",
        "ready_for_full_qa": quality_gate,
        "next_live_authorized": False,
        "human_citation_audit_next": engineering and quality_gate,
        "failure_freeze": not engineering,
        "full_qa_executed": False,
        "deep_research_executed": False,
        "production_ready": False,
        "v1_0": False,
        "current_release": "v0.9.0-rc3",
    }
    FINAL_AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    if not engineering:
        freeze = {
            "schema_version": "dev-v3-5-failure-freeze-v1",
            "evaluation_version": "evidence-qa-dev-v3.5",
            "reason": "DEV_V3_5_FAILURE_FREEZE",
            "failed_engineering_checks": [
                key for key, value in engineering_checks.items() if not value
            ],
            "run_count": total["run_count"],
            "attempt_history": summary["attempt_history"],
            "raw_payload_layer": raw,
            "slot_shape_layer": slots,
            "no_protocol_repair_authorized": True,
            "next_live_authorized": False,
        }
        FAILURE_FREEZE.write_text(
            json.dumps(freeze, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        FAILURE_FREEZE_DOC.write_text(
            "# Dev v3.5 Failure Freeze\n\n"
            "- Reason: `DEV_V3_5_FAILURE_FREEZE`\n"
            f"- Failed checks: {', '.join(freeze['failed_engineering_checks'])}\n"
            "- No normalization, repair, retry, fallback, or protocol redesign was applied.\n",
            encoding="utf-8",
        )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
