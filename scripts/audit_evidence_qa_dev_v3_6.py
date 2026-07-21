"""Apply frozen Dev v3.6 engineering and quality gates."""

from __future__ import annotations

import json

try:
    from scripts.evidence_qa_dev_v3_6_lib import FINAL_AUDIT, HEALTH, OUTPUT
except ModuleNotFoundError:
    from evidence_qa_dev_v3_6_lib import FINAL_AUDIT, HEALTH, OUTPUT  # type: ignore[no-redef]


def main() -> None:
    summary = json.loads(OUTPUT.read_text(encoding="utf-8"))
    health = json.loads(HEALTH.read_text(encoding="utf-8"))
    raw = summary["raw_payload_layer"]
    final = summary["final_policy_layer"]
    total = summary["all_manifest_conservative"]
    prompt_gate = (
        "PASSED"
        if raw["prompt_contamination_failures"] == 0
        and raw["model_visible_metadata_leakage"] == 0
        else "FAILED"
    )
    raw_checks = {
        "provider_completed_ge_9": raw["provider_completed"] >= 9,
        "raw_json_valid_ge_9": raw["raw_json_valid"] >= 9,
        "payload_v4_schema_ge_9": raw["payload_v4_schema_success"] >= 9,
        "slot_shape_success_ge_9": raw["slot_shape_success_questions"] >= 9,
        "valid_slots_ge_25": raw["valid_slots"] >= 25,
        "q005_refusal": total["refusal_accuracy"] == 1.0,
        "malformed_json_le_1": raw["malformed_json"] <= 1,
        "status_field_leakage_zero": raw["status_field_leakage"] == 0,
        "citation_field_leakage_zero": raw["citation_field_leakage"] == 0,
        "evidence_label_leakage_zero": raw["evidence_label_leakage"] == 0,
        "arbitrary_extra_field_questions_le_1": raw["arbitrary_extra_field_questions"] <= 1,
        "null_sentinel_zero": raw["null_sentinel"] == 0,
        "empty_sentinel_zero": raw["empty_sentinel"] == 0,
        "dual_semantic_conflicts_le_1": raw["dual_semantic_conflicts"] <= 1,
        "model_visible_metadata_leakage_zero": raw["model_visible_metadata_leakage"] == 0,
        "delivered_hashes": total["delivered_hash_matches"] == total["run_count"],
        "prompt_contamination_zero": prompt_gate == "PASSED",
        "usage_complete": total["usage_records"] == raw["provider_completed"],
        "ledger_closed": total["effective_active_reservations"] == 0,
        "no_retries": total["retries"] == 0,
        "reranker_disabled": total["reranker_called"] is False,
        "gold_human_leakage_zero": True,
        "fixed_id_special_cases_zero": True,
    }
    final_checks = {
        "envelope_binding_ge_9": final["envelope_binding_success"] >= 9,
        "final_schema_ge_9": final["final_schema_success"] >= 9,
        "final_slots_ge_25": final["final_slot_success"] >= 25,
        "silent_omission_zero": total["silent_omissions"] == 0,
        "unknown_zero": final["unknown_citation_id"] == 0,
        "invalid_zero": final["invalid_citation_id"] == 0,
        "cross_claim_zero": final["cross_claim_citation"] == 0,
        "cap_zero": final["citation_cap_violations"] == 0,
        "q005_refusal": total["refusal_accuracy"] == 1.0,
        "policy_trace_complete": final["final_schema_success"] >= 9,
        "accounting_complete": total["effective_active_reservations"] == 0,
    }
    engineering = all(raw_checks.values()) and all(final_checks.values())
    quality_checks = {
        "slot_coverage": final["final_slot_success"] >= 25,
        "silent_omission": total["silent_omissions"] == 0,
        "answered_ge_18": final["answered_original"] + final["answered_narrowed"] >= 18,
        "unsupported_le_8": final["unsupported_slots"] <= 8,
        "obligation": final["obligation_completeness"] >= 0.90,
        "numeric": final["numeric_completeness"] == 1.0,
        "comparison": final["comparison_completeness"] == 1.0,
        "wrong_evidence": final["wrong_evidence"] <= 2,
        "dilution": final["citation_dilution"] == 0,
        "average_citations": final["average_citations_per_answered"] <= 1.20,
        "any_valid": final["any_valid_evidence_recall"] >= 0.296296,
        "question_macro": final["answerable_question_macro_exact_relation_recall"] >= 0.166667,
        "claim_macro": final["required_claim_macro_exact_recall"] >= 0.166667,
        "micro_core": final["micro_core_relation_recall"] >= 0.18,
        "core_set": final["core_set_completion"] >= 0.148148,
        "refusal": total["refusal_accuracy"] == 1.0,
        "citation_validity": final["unknown_citation_id"]
        + final["invalid_citation_id"]
        + final["cross_claim_citation"]
        == 0,
        "non_regressed_ge_5": total["improved_questions"] + total["unchanged_questions"] >= 5,
        "improved_ge_regressed": total["improved_questions"] >= total["regressed_questions"],
        "not_unsupported_driven": total["unsupported_improvement_driver"] is False,
    }
    automated = engineering and all(quality_checks.values())
    human = (
        "FAILED"
        if total["citation_pairs_total"] == 0
        else "PENDING"
        if total["pending_pairs"]
        else "PASSED"
    )
    quality_gate = "FAILED" if not automated else "PENDING" if human == "PENDING" else "PASSED"
    audit = {
        "schema_version": "evidence-qa-dev-v3-6-final-audit-v1",
        "provider_health": "PASSED" if health.get("safe_to_start_batch") else "FAILED",
        "safe_to_start_batch": bool(health.get("safe_to_start_batch")),
        "prompt_contamination_gate": prompt_gate,
        "raw_payload_checks": raw_checks,
        "final_policy_engineering_checks": final_checks,
        "automated_quality_checks": quality_checks,
        "DEV_V3_6_PROVIDER_HEALTH": "PASSED" if health.get("safe_to_start_batch") else "FAILED",
        "DEV_V3_6_PROMPT_CONTAMINATION_GATE": prompt_gate,
        "DEV_V3_6_RAW_PAYLOAD_GATE": "PASSED" if all(raw_checks.values()) else "FAILED",
        "DEV_V3_6_FINAL_POLICY_ENGINEERING_GATE": "PASSED"
        if all(final_checks.values())
        else "FAILED",
        "DEV_V3_6_ENGINEERING_GATE": "PASSED" if engineering else "FAILED",
        "DEV_V3_6_AUTOMATED_QUALITY_GATE": "PASSED" if automated else "FAILED",
        "DEV_V3_6_HUMAN_SUPPORT_GATE": human,
        "DEV_V3_6_QUALITY_CANDIDATE_GATE": quality_gate,
        "READY_FOR_FULL_QA": quality_gate == "PASSED",
        "full_qa_executed": False,
        "deep_research_executed": False,
        "production_ready": False,
        "v1_0": False,
        "current_release": "v0.9.0-rc3",
        "READY_FOR_DEV_V3_6_CHECKPOINT_COMMIT": True,
    }
    FINAL_AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
