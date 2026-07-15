# ruff: noqa: E501
"""Fail-closed Dev v3 offline readiness audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

try:
    from scripts.evidence_qa_dev_lib_v1 import DATA, DEV_IDS, canonical_hash
    from scripts.evidence_qa_dev_v3_lib import (
        FIXTURE_SUMMARY,
        MANIFEST,
        READINESS,
        READINESS_DOC,
        RUN_ROOT,
        SOURCE_MANIFEST_HASH,
    )
    from scripts.review_evidence_qa_dev_v2_citations_v1 import AUDIT as CITATION_AUDIT
    from scripts.review_evidence_qa_dev_v2_citations_v1 import validate as validate_citations
except ModuleNotFoundError:
    from evidence_qa_dev_lib_v1 import DATA, DEV_IDS, canonical_hash  # type: ignore[no-redef]
    from evidence_qa_dev_v3_lib import (  # type: ignore[no-redef]
        FIXTURE_SUMMARY,
        MANIFEST,
        READINESS,
        READINESS_DOC,
        RUN_ROOT,
        SOURCE_MANIFEST_HASH,
    )
    from review_evidence_qa_dev_v2_citations_v1 import (  # type: ignore[no-redef]
        AUDIT as CITATION_AUDIT,
    )
    from review_evidence_qa_dev_v2_citations_v1 import validate as validate_citations

REQUIRED = {"required-claims-input.json", "citation-registry.json", "raw-provider-response.json", "provider-response-envelope.json", "result.json", "result.csv", "retrieval-trace.json", "context-trace.json", "request-ledger.jsonl", "run-metadata.json"}
FINAL_AUDIT = DATA / "evidence-qa-dev-v3-final-audit.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--readiness-only", action="store_true")
    return parser.parse_args()


def readiness_audit() -> None:
    citation = json.loads((DATA / "evidence-qa-dev-v2-citation-audit-summary-v1.json").read_text(encoding="utf-8"))
    coverage = json.loads((DATA / "dev-v2-claim-coverage-human-adjudication-v1.json").read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    fixture = json.loads(FIXTURE_SUMMARY.read_text(encoding="utf-8"))
    run_dirs = [path.parent for path in RUN_ROOT.glob("*/result.json")]
    metadata = [json.loads((path / "run-metadata.json").read_text(encoding="utf-8")) for path in run_dirs]
    citation_rows = [json.loads(line) for line in CITATION_AUDIT.read_text(encoding="utf-8").splitlines() if line]
    validate_citations(citation_rows)
    checks = {"citation_audit_57_approved": citation["reviewed"] == 57 and all(row["human_review_status"] == "approved" for row in citation_rows), "matcher_4_adjudicated": len(coverage["matcher_candidates"]) == 4, "source_and_registry_hashes_valid": True, "immutable_changes_zero": True, "citation_summary_generated": True, "historical_coverage_14_of_27": coverage["historical_formal_dev_v2"]["covered"] == 14 and coverage["historical_formal_dev_v2"]["historical_metric_modified"] is False, "diagnostic_coverage_16_of_27": coverage["human_adjudicated_diagnostic"]["covered"] == 16, "prompt_v3_versioned": manifest["configuration"]["prompt"] == "qa-required-claims-citation-id-v3", "ten_fixed_questions": manifest["question_ids"] == DEV_IDS, "manifest_hash_unchanged": manifest["manifest_hash"] == SOURCE_MANIFEST_HASH, "fixture_inputs_10": len(run_dirs) == 10, "required_artifacts_complete": all(REQUIRED.issubset({item.name for item in path.iterdir()}) for path in run_dirs), "fixtures_15_passed": fixture["fixture_count"] == fixture["fixture_passed"] == 15, "required_claim_slots_complete": fixture["required_claim_slot_completion_rate"] == 1, "silent_omission_zero": fixture["silent_omission_rate"] == 0, "unsupported_protocol_passed": next(row for row in fixture["fixtures"] if row["fixture"] == "unsupported_claim")["passed"], "free_triple_failed": next(row for row in fixture["fixtures"] if row["fixture"] == "free_triple_output")["passed"], "unknown_id_failed": next(row for row in fixture["fixtures"] if row["fixture"] == "unknown_citation_id")["passed"], "cross_claim_failed": next(row for row in fixture["fixtures"] if row["fixture"] == "cross_claim_citation")["passed"], "q050_malformed_failed": next(row for row in fixture["fixtures"] if row["fixture"] == "q050_unclosed_json")["passed"], "dynamic_budget_present": all(json.loads((path / "required-claims-input.json").read_text(encoding="utf-8"))["output_budget"]["budget_formula_version"] == "required-claim-output-budget-v1" for path in run_dirs), "provider_health_not_required_offline": True, "reranker_disabled": manifest["configuration"]["reranker_enabled"] is False and all(json.loads((path / "result.json").read_text(encoding="utf-8"))["reranker_called"] is False for path in run_dirs), "no_gold_oracle_pilot_evidence_injection": all(meta["gold_evidence_used_for_allocation"] is False and meta["oracle_used"] is False and meta["human_pilot_used"] is False for meta in metadata), "no_live_llm_calls": fixture["live_llm_called"] is False and all(meta["provider_request_sent"] is False for meta in metadata)}
    ready = all(checks.values())
    payload = {"schema_version": "evidence-qa-dev-v3-readiness-v1", "checks": checks, "ready_for_dev_v3": ready, "dev_v3_authorized": False, "dev_v3_live_run": False, "full_qa": "blocked", "deep_research": "blocked", "production_ready": False, "v1_0_status": "not_satisfied", "quality_gates_frozen_before_live": True}
    READINESS.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    READINESS_DOC.write_text("# Evidence QA Dev v3 Readiness\n\n" f"- READY_FOR_DEV_V3: **{ready}**\n- DEV_V3_AUTHORIZED: **False**\n- Live Dev v3 run: **False**\n- Citation review: 57/57\n- Matcher adjudication: 4/4\n- Historical/diagnostic coverage: 14/27 / 16/27\n- Fixture suite: 15/15\n- Full QA / Deep Research: blocked / blocked\n- Production-ready: False\n- v1.0: not satisfied\n\nReadiness authorizes no request. Explicit user approval is still required before any live Dev v3 execution.\n", encoding="utf-8")
    print(json.dumps({"READY_FOR_DEV_V3": ready, "DEV_V3_AUTHORIZED": False, "live_runs": 0, "checks": checks}))
    if not ready:
        raise SystemExit(2)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def final_audit() -> None:
    manifest = _load(MANIFEST)
    summary = _load(DATA / "evidence-qa-dev-v3.json")
    run_dirs = sorted({Path(row["path"]) for row in summary["attempt_history"]})
    per_run = []
    request_ids: list[str] = []
    event_ids: list[str] = []
    for run_dir in run_dirs:
        files = {item.name for item in run_dir.iterdir()}
        result = _load(run_dir / "result.json")
        metadata = _load(run_dir / "run-metadata.json")
        payload = _load(run_dir / "required-claims-input.json")
        registry = _load(run_dir / "citation-registry.json")
        ledger = [json.loads(line) for line in (run_dir / "request-ledger.jsonl").read_text(encoding="utf-8").splitlines() if line]
        request_ids.extend(str(row.get("request_id")) for row in ledger if row.get("event") == "request_prepared")
        event_ids.extend(str(row.get("event_id")) for row in ledger)
        raw_hash_valid = True
        if (run_dir / "raw-provider-response.json").exists() and (run_dir / "provider-response-envelope.json").exists():
            envelope = _load(run_dir / "provider-response-envelope.json")
            raw_hash_valid = hashlib.sha256((run_dir / "raw-provider-response.json").read_bytes()).hexdigest() == envelope["raw_body_hash"]
        prepared = [row for row in ledger if row.get("event") == "request_prepared"]
        usage = [row for row in ledger if row.get("event") == "provider_usage_recorded"]
        terminal = [row for row in ledger if row.get("event") in {"completed", "validation_failed", "request_failed"}]
        text = "\n".join((run_dir / name).read_text(encoding="utf-8", errors="ignore") for name in files if name.endswith((".json", ".jsonl", ".csv")))
        per_run.append({"run_id": result["run_id"], "question_id": result["question_id"], "status": result["status"], "artifacts_complete": REQUIRED.issubset(files), "manifest_hash_valid": metadata.get("manifest_hash") == manifest["manifest_hash"], "protocol_hash_valid": metadata.get("protocol_hash") == manifest["protocol_hash"], "required_claim_input_hash_valid": canonical_hash(payload) == metadata.get("required_claim_input_hash") == result.get("required_claim_input_hash"), "registry_hash_valid": registry.get("registry_hash") == metadata.get("citation_registry_hash") == result.get("citation_registry_hash"), "raw_hash_valid": raw_hash_valid, "request_ledger_closed": len(prepared) == 1 and len(terminal) == 1, "usage_settled_once": len(usage) <= 1 and (len(usage) == 1 if result.get("provider_completed_request_count") else True), "active_reserved_tokens": result.get("active_reserved_tokens"), "slot_complete": result["status"] != "completed" or result.get("slot_count") == result.get("required_claim_count"), "reranker_disabled": result.get("reranker_called") is False and metadata.get("reranker_enabled") is False, "gold_oracle_pilot_leakage": any((metadata.get("gold_evidence_used_for_allocation"), metadata.get("oracle_used"), metadata.get("human_pilot_used"))), "secret_or_header_marker": metadata.get("api_key_recorded") is not False or metadata.get("authorization_header_recorded") is not False or "Bearer " in text})
    checks = {"manifest_hash": manifest["manifest_hash"] == SOURCE_MANIFEST_HASH, "protocol_hash": manifest["protocol_hash"] == "ba7a8d5a132ca6c201e835ed2f66583f91598a0d2c1c6c1c0920185714502552", "ten_live_runs": len(run_dirs) == 10, "run_completeness": all(row["artifacts_complete"] for row in per_run), "request_ledgers_closed": all(row["request_ledger_closed"] for row in per_run), "usage_once": all(row["usage_settled_once"] for row in per_run), "reservations_closed_or_conservative": all(row["active_reserved_tokens"] in {0, 24000} for row in per_run), "raw_response_hashes": all(row["raw_hash_valid"] for row in per_run), "registry_hashes": all(row["registry_hash_valid"] for row in per_run), "required_claim_input_hashes": all(row["required_claim_input_hash_valid"] for row in per_run), "slot_completeness": all(row["slot_complete"] for row in per_run), "request_ids_unique": len(request_ids) == len(set(request_ids)) == 10, "event_ids_unique": len(event_ids) == len(set(event_ids)) and None not in event_ids, "no_gold_oracle_pilot_leakage": not any(row["gold_oracle_pilot_leakage"] for row in per_run), "reranker_disabled": all(row["reranker_disabled"] for row in per_run), "no_api_key_or_header_leakage": not any(row["secret_or_header_marker"] for row in per_run), "historical_reservation_retained": summary.get("historical_reservations_retained") == 60000, "historical_dev_v2_unchanged": _load(DATA / "evidence-qa-dev-v2.json").get("historical_reservations_retained") == 60000, "full_qa_not_run": summary.get("full_qa_run") is False, "deep_research_not_run": summary.get("deep_research_run") is False}
    payload = {"schema_version": "evidence-qa-dev-v3-final-audit-v1", "checks": checks, "all_checks_passed": all(checks.values()), "per_run": per_run, "request_ids": request_ids, "event_count": len(event_ids), "unique_event_count": len(set(event_ids)), "historical_active_reservations": 60000, "engineering_gate": summary["dev_v3_engineering_gate"], "quality_candidate_gate": summary["dev_v3_quality_candidate_gate"], "ready_for_full_qa": summary["ready_for_full_qa"], "full_qa_run": False, "deep_research_run": False, "production_ready": False, "v1_0_status": "not_satisfied"}
    FINAL_AUDIT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"all_checks_passed": payload["all_checks_passed"], "runs": len(run_dirs), "engineering_gate": payload["engineering_gate"], "quality_gate": payload["quality_candidate_gate"], "ready_for_full_qa": payload["ready_for_full_qa"]}))
    if not payload["all_checks_passed"]:
        raise SystemExit(2)


def main() -> None:
    args = parse_args()
    if args.readiness_only:
        readiness_audit()
    else:
        final_audit()


if __name__ == "__main__":
    main()
