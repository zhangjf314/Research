"""Apply frozen Stage 13.12 engineering and automated quality gates."""

from __future__ import annotations

import json

try:
    from scripts.evidence_qa_dev_v3_2_lib import FINAL_AUDIT, OUTPUT
except ModuleNotFoundError:
    from evidence_qa_dev_v3_2_lib import FINAL_AUDIT, OUTPUT  # type: ignore[no-redef]


def main() -> None:
    summary = json.loads(OUTPUT.read_text(encoding="utf-8"))
    raw = summary["raw_model_layer"]
    final = summary["final_policy_layer"]
    conservative = summary["all_manifest_conservative"]
    engineering_checks = {
        "provider_completed_ge_9": raw["provider_completed"] >= 9,
        "raw_json_valid_ge_9": raw["raw_json_valid"] >= 9,
        "raw_schema_success_ge_9": raw["raw_schema_success"] >= 9,
        "raw_slot_success_ge_90pct": raw["raw_slot_success"] >= 25,
        "final_schema_success_ge_9": final["final_schema_success"] >= 9,
        "final_slot_success_ge_90pct": final["final_slot_success"] >= 25,
        "usage_complete": conservative["usage_records"] == 10,
        "ledger_closed": conservative["active_reserved_tokens"] == 0,
        "no_retries": conservative["retries"] == 0,
        "reranker_disabled": conservative["reranker_called"] is False,
        "q005_refusal": conservative["refusal_accuracy"] == 1.0,
    }
    automated_checks = {
        "slot_coverage_ge_25": final["final_slot_success"] >= 25,
        "silent_omission_zero": conservative["silent_omissions"] == 0,
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
        "refusal_accuracy": conservative["refusal_accuracy"] == 1.0,
    }
    engineering = all(engineering_checks.values())
    automated = engineering and all(automated_checks.values())
    pending = conservative["new_pending_pairs"] > 0
    human_gate = "PENDING" if pending else "PASSED"
    quality = "PENDING" if automated and pending else "PASSED" if automated else "FAILED"
    audit = {
        "schema_version": "evidence-qa-dev-v3-2-final-audit-v1",
        "engineering_checks": engineering_checks,
        "automated_quality_checks": automated_checks,
        "dev_v3_2_engineering_gate": "PASSED" if engineering else "FAILED",
        "dev_v3_2_automated_quality_gate": "PASSED" if automated else "FAILED",
        "dev_v3_2_human_support_gate": human_gate,
        "dev_v3_2_quality_candidate_gate": quality,
        "ready_for_full_qa": quality == "PASSED",
        "full_qa_executed": False,
        "deep_research_executed": False,
        "production_ready": False,
        "v1_0": False,
        "ready_for_checkpoint_commit": True,
        "historical_gate_modified": False,
    }
    FINAL_AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
