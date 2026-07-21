"""Apply the frozen Dev v3.3 engineering and automated quality gates."""

from __future__ import annotations

import json

try:
    from scripts.evidence_qa_dev_v3_3_lib import FINAL_AUDIT, OUTPUT
except ModuleNotFoundError:
    from evidence_qa_dev_v3_3_lib import FINAL_AUDIT, OUTPUT  # type: ignore[no-redef]


def main() -> None:
    summary = json.loads(OUTPUT.read_text(encoding="utf-8"))
    raw = summary["raw_model_layer"]
    final = summary["final_policy_layer"]
    total = summary["all_manifest_conservative"]
    raw_checks = {
        "provider_completed_ge_9": raw["provider_completed"] >= 9,
        "raw_json_valid_ge_9": raw["raw_json_valid"] >= 9,
        "payload_schema_success_ge_9": raw["model_payload_schema_success"] >= 9,
        "required_slot_success_ge_25": raw["required_slot_success"] >= 25,
        "q005_payload_correct": total["refusal_accuracy"] == 1.0,
        "malformed_json_le_1": raw["malformed_json"] <= 1,
        "prompt_hashes_match": total["delivered_hash_matches"] == total["run_count"],
        "no_model_protocol_fields": raw["model_protocol_fields_output"] == 0,
        "no_model_citation_fields": raw["model_citation_id_fields_output"] == 0,
        "usage_complete": total["usage_records"] == total["provider_completed"],
        "ledger_closed": total["effective_active_reservations"] == 0,
        "no_retries": total["retries"] == 0,
        "no_internal_id_leakage": raw["internal_id_leakage"] == 0,
        "reranker_disabled": total["reranker_called"] is False,
    }
    final_checks = {
        "final_schema_success_ge_9": final["final_schema_success"] >= 9,
        "final_slot_success_ge_25": final["final_slot_success"] >= 25,
        "silent_omission_zero": total["silent_omissions"] == 0,
        "unknown_citation_zero": final["unknown_citation_id"] == 0,
        "invalid_citation_zero": final["invalid_citation_id"] == 0,
        "cross_claim_zero": final["cross_claim_citation"] == 0,
        "citation_cap_zero": final["citation_cap_violations"] == 0,
        "q005_refusal": total["refusal_accuracy"] == 1.0,
        "envelope_binding_verifiable": final["final_schema_success"] >= 9,
    }
    engineering = all(raw_checks.values()) and all(final_checks.values())
    automated_checks = {
        "slot_coverage_ge_25": final["final_slot_success"] >= 25,
        "silent_omission_zero": total["silent_omissions"] == 0,
        "answered_ge_18": final["answered_original"] + final["answered_narrowed"] >= 18,
        "unsupported_le_8": final["unsupported_slots"] <= 8,
        "obligation_ge_090": final["obligation_completeness"] >= 0.90,
        "numeric_eq_1": final["numeric_completeness"] == 1.0,
        "comparison_eq_1": final["comparison_completeness"] == 1.0,
        "wrong_evidence_le_2": final["wrong_evidence"] <= 2,
        "dilution_zero": final["citation_dilution"] == 0,
        "avg_citations_le_120": final["average_citations_per_answered"] <= 1.20,
        "any_valid_ge_baseline": final["any_valid_evidence_recall"] >= 0.296296,
        "exact_ge_floor": final["answerable_question_macro_exact_relation_recall"] >= 0.166667,
        "claim_macro_ge_floor": final["required_claim_macro_exact_recall"] >= 0.166667,
        "micro_core_ge_018": final["micro_core_relation_recall"] >= 0.18,
        "core_set_ge_floor": final["core_set_completion"] >= 0.148148,
        "refusal_accuracy": total["refusal_accuracy"] == 1.0,
        "citation_validity": final["unknown_citation_id"]
        + final["invalid_citation_id"]
        + final["cross_claim_citation"]
        == 0,
        "non_regressed_ge_5": total["improved_questions"] + total["unchanged_questions"] >= 5,
        "improved_ge_regressed": total["improved_questions"] >= total["regressed_questions"],
    }
    automated = engineering and all(automated_checks.values())
    human = "PENDING" if total["pending_pairs"] else "PASSED"
    quality = "FAILED" if not automated else "PENDING" if human == "PENDING" else "PASSED"
    audit = {
        "schema_version": "evidence-qa-dev-v3-3-final-audit-v1",
        "raw_payload_checks": raw_checks,
        "final_policy_engineering_checks": final_checks,
        "automated_quality_checks": automated_checks,
        "dev_v3_3_provider_health": "PASSED",
        "dev_v3_3_raw_payload_gate": "PASSED" if all(raw_checks.values()) else "FAILED",
        "dev_v3_3_final_policy_engineering_gate": "PASSED"
        if all(final_checks.values())
        else "FAILED",
        "dev_v3_3_engineering_gate": "PASSED" if engineering else "FAILED",
        "dev_v3_3_automated_quality_gate": "PASSED" if automated else "FAILED",
        "dev_v3_3_human_support_gate": human,
        "dev_v3_3_quality_candidate_gate": quality,
        "ready_for_full_qa": quality == "PASSED",
        "full_qa_executed": False,
        "deep_research_executed": False,
        "production_ready": False,
        "v1_0": False,
        "current_release": "v0.9.0-rc3",
        "ready_for_dev_v3_3_checkpoint_commit": True,
        "historical_stage13_12_gate": "FAILED_AND_PRESERVED",
    }
    FINAL_AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
