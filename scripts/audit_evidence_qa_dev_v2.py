# ruff: noqa: E501
"""Fail-closed artifact, accounting, registry, and secret audit for Dev v2."""

from __future__ import annotations

import hashlib
import json

from paper_research.config import Settings

try:
    from scripts.evidence_qa_dev_lib_v1 import DEV_IDS, read_jsonl
    from scripts.summarize_evidence_qa_dev_v2 import FINAL_AUDIT, RUN_ROOT, SUMMARY_JSON
except ModuleNotFoundError:
    from evidence_qa_dev_lib_v1 import DEV_IDS, read_jsonl  # type: ignore[no-redef]
    from summarize_evidence_qa_dev_v2 import (  # type: ignore[no-redef]
        FINAL_AUDIT,
        RUN_ROOT,
        SUMMARY_JSON,
    )

REQUIRED = {"raw-provider-response.json", "provider-response-envelope.json", "citation-registry.json", "result.json", "result.csv", "retrieval-trace.json", "context-trace.json", "request-ledger.jsonl", "run-metadata.json"}


def main() -> None:
    summary = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
    run_dirs = [RUN_ROOT / row["run_id"] for row in (json.loads(path.read_text(encoding="utf-8")) for path in RUN_ROOT.glob("*/result.json"))]
    results = [json.loads((path / "result.json").read_text(encoding="utf-8")) for path in run_dirs]
    request_ids, usage_ids, ledger_closed, raw_valid, registry_valid = [], [], True, True, True
    for path, row in zip(run_dirs, results, strict=True):
        events = read_jsonl(path / "request-ledger.jsonl")
        request_ids.extend(event["request_id"] for event in events if event["event"] == "request_prepared")
        usage_ids.extend(event["request_id"] for event in events if event["event"] == "provider_usage_recorded")
        ledger_closed &= sum(event["event"] == "request_prepared" for event in events) == 1 and sum(event["event"] == "request_started" for event in events) == 1 and sum(event["event"] in {"response_parsed", "post_processing_failed", "request_failed"} for event in events) == 1
        registry = json.loads((path / "citation-registry.json").read_text(encoding="utf-8"))
        registry_valid &= registry["registry_hash"] == row["citation_registry_hash"] and row["registry_hash_valid"]
        envelope = json.loads((path / "provider-response-envelope.json").read_text(encoding="utf-8"))
        raw = (path / "raw-provider-response.json").read_bytes()
        if envelope.get("response_received") is not False:
            raw_valid &= envelope["raw_body_hash"] == hashlib.sha256(raw).hexdigest()
    artifact_text = "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in RUN_ROOT.rglob("*") if path.is_file())
    key = Settings().llm_api_key or ""
    secret_leak = bool(key and key in artifact_text)
    header_leak = '"Authorization":' in artifact_text or "Bearer sk-" in artifact_text
    checks = {"fixed_manifest_questions": {row["question_id"] for row in results}.issubset(set(DEV_IDS)), "run_count_10": len(results) == 10, "required_artifacts_complete": all(REQUIRED.issubset({item.name for item in path.iterdir()}) for path in run_dirs), "unique_request_ids": len(request_ids) == len(set(request_ids)), "usage_at_most_once": len(usage_ids) == len(set(usage_ids)), "request_ledger_closed": ledger_closed, "registry_hash_valid": registry_valid, "raw_response_integrity": raw_valid, "api_key_not_leaked": not secret_leak, "authorization_header_not_leaked": not header_leak, "reranker_disabled": all(row["reranker_called"] is False for row in results), "no_gold_leakage": summary["gold_leakage"] is False, "no_oracle_leakage": summary["oracle_leakage"] is False, "no_pilot_injection": summary["human_pilot_evidence_used_for_selection"] is False, "historical_reservations_retained": summary["historical_reservations_retained"] == 60000, "full_qa_not_run": summary["full_qa_run"] is False, "deep_research_not_run": summary["deep_research_run"] is False}
    engineering = all(checks.values()) and summary["dev_v2_engineering_gate"]
    payload = {"schema_version": "evidence-qa-dev-v2-final-audit-v1", "checks": checks, "dev_v2_engineering_gate": engineering, "dev_v2_quality_candidate_gate": summary["dev_v2_quality_candidate_gate"], "ready_for_full_qa": engineering and summary["dev_v2_quality_candidate_gate"], "production_ready": False, "v1_0_status": "not_satisfied"}
    FINAL_AUDIT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload))
    if not engineering:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
