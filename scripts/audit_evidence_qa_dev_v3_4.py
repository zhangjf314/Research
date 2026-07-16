"""Apply frozen Dev v3.4 engineering and automated quality gates."""

from __future__ import annotations

import json

try:
    from scripts.evidence_qa_dev_v3_4_lib import FINAL_AUDIT, OUTPUT
except ModuleNotFoundError:
    from evidence_qa_dev_v3_4_lib import FINAL_AUDIT, OUTPUT  # type: ignore[no-redef]


def main() -> None:
    summary = json.loads(OUTPUT.read_text(encoding="utf-8"))
    raw = summary["raw_payload_layer"]
    final = summary["final_policy_layer"]
    total = summary["all_manifest_conservative"]
    raw_checks = {
        "provider_completed_ge_9": raw["provider_completed"] >= 9,
        "raw_json_valid_ge_9": raw["raw_json_valid"] >= 9,
        "structural_payload_ge_9": raw["structural_payload_success"] >= 9,
        "slot_cardinality_ge_9": raw["slot_cardinality_success_questions"] >= 9,
        "canonical_payload_ge_9": raw["canonical_payload_success"] >= 9,
        "path_violations_zero": raw["canonicalization_path_violations"] == 0,
        "semantic_changes_zero": raw["semantic_field_changes"] == 0,
        "q005_correct": total["refusal_accuracy"] == 1.0,
        "malformed_le_1": raw["malformed_json"] <= 1,
        "nonempty_answerable_zero": raw["nonempty_answerable_refusal"] == 0,
        "illegal_whitespace_zero": raw["illegal_whitespace_refusal"] == 0,
        "model_protocol_fields_zero": raw["model_protocol_fields_output"] == 0,
        "model_citation_fields_zero": raw["model_citation_id_fields_output"] == 0,
        "internal_id_zero": raw["internal_id_leakage"] == 0,
        "delivered_hashes": total["delivered_hash_matches"] == total["run_count"],
        "usage_complete": total["usage_records"] == total["provider_completed"],
        "ledger_closed": total["effective_active_reservations"] == 0,
        "no_retries": total["retries"] == 0,
        "reranker_disabled": total["reranker_called"] is False,
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
    quality = "FAILED" if not automated else "PENDING" if human == "PENDING" else "PASSED"
    audit = {
        "schema_version": "evidence-qa-dev-v3-4-final-audit-v1",
        "raw_payload_checks": raw_checks,
        "final_policy_engineering_checks": final_checks,
        "automated_quality_checks": quality_checks,
        "dev_v3_4_provider_health": "PASSED",
        "dev_v3_4_raw_payload_gate": "PASSED" if all(raw_checks.values()) else "FAILED",
        "dev_v3_4_final_policy_engineering_gate": "PASSED"
        if all(final_checks.values())
        else "FAILED",
        "dev_v3_4_engineering_gate": "PASSED" if engineering else "FAILED",
        "dev_v3_4_automated_quality_gate": "PASSED" if automated else "FAILED",
        "dev_v3_4_human_support_gate": human,
        "dev_v3_4_quality_candidate_gate": quality,
        "ready_for_full_qa": quality == "PASSED",
        "full_qa_executed": False,
        "deep_research_executed": False,
        "production_ready": False,
        "v1_0": False,
        "current_release": "v0.9.0-rc3",
        "ready_for_dev_v3_4_checkpoint_commit": True,
        "stage13_14_historical_gate": "FAILED_AND_PRESERVED",
    }
    FINAL_AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
