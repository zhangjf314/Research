# ruff: noqa: E501
"""Fail-closed final offline audit for Stage 13.2 Dev QA."""

from __future__ import annotations

import json

try:
    from scripts.evidence_qa_dev_lib_v1 import (
        AUDIT_JSONL,
        DEV_IDS,
        FINAL_AUDIT,
        MANIFEST,
        SUMMARY_JSON,
        canonical_hash,
        read_jsonl,
    )
except ModuleNotFoundError:
    from evidence_qa_dev_lib_v1 import (  # type: ignore[no-redef]
        AUDIT_JSONL,
        DEV_IDS,
        FINAL_AUDIT,
        MANIFEST,
        SUMMARY_JSON,
        canonical_hash,
        read_jsonl,
    )


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest_body = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    summary = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
    audit_rows = read_jsonl(AUDIT_JSONL)
    checks = {
        "manifest_fixed_10": manifest["question_count"] == 10 and manifest["question_ids"] == DEV_IDS,
        "manifest_hash_valid": canonical_hash(manifest_body) == manifest["manifest_hash"],
        "manifest_matches_summary": summary["manifest_hash"] == manifest["manifest_hash"],
        "historical_baseline_reused": summary["baseline_reused"] is True,
        "variant_c_blocked_known_defect": summary["variants"]["evidence_centric"]["status"] == "blocked_by_known_selector_defect",
        "run_count_10": len(summary["selected_runs"]) == 10,
        "request_ledger_complete": summary["variants"]["retrieval_only"]["metrics"]["request_attempts"] == 10,
        "usage_valid": summary["variants"]["retrieval_only"]["metrics"]["total_tokens"] > 0,
        "active_reservations_zero": summary["variants"]["retrieval_only"]["metrics"]["active_reserved_tokens"] == 0,
        "strict_citation_triples": summary["variants"]["retrieval_only"]["metrics"]["invalid_citation_rate"] == 0,
        "reranker_disabled": summary["reranker_enabled"] is False,
        "no_gold_leakage": summary["gold_leakage"] is False,
        "no_oracle_leakage": summary["oracle_leakage"] is False,
        "no_pilot_selection_injection": summary["human_pilot_evidence_used_for_selection"] is False,
        "human_audit_pending": bool(audit_rows) and all(row["human_review_status"] == "pending" and row["human_label"] is None for row in audit_rows),
        "full_qa_not_run": summary["full_qa_run"] is False,
        "deep_research_not_run": summary["deep_research_run"] is False,
    }
    payload = {
        "schema_version": "evidence-qa-dev-final-audit-v1",
        "checks": checks,
        "dev_engineering_gate": all(checks.values()),
        "dev_quality_candidate_gate": summary["dev_quality_candidate_gate"],
        "ready_for_full_qa": summary["ready_for_full_qa"] and all(checks.values()),
        "human_citation_quality_gate": "pending",
        "full_qa_run": False,
        "production_ready": False,
        "v1_0_status": "not_satisfied",
    }
    FINAL_AUDIT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload))
    if not payload["dev_engineering_gate"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
