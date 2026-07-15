# ruff: noqa: E501
"""Audit Dev v3.1 artifacts, persistence order, hashes, and safety invariants."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from paper_research.config import Settings

try:
    from scripts.evidence_qa_dev_lib_v1 import DEV_IDS, canonical_hash
    from scripts.evidence_qa_dev_v3_1_lib import (
        CAPABILITY_HASH,
        FINAL_AUDIT,
        HEALTH,
        MANIFEST,
        OUTPUT,
        PROMPT_HASH,
        RUN_ROOT,
        SCHEMA_HASH,
        SOURCE_MANIFEST_HASH,
    )
except ModuleNotFoundError:
    from evidence_qa_dev_lib_v1 import DEV_IDS, canonical_hash  # type: ignore[no-redef]
    from evidence_qa_dev_v3_1_lib import (  # type: ignore[no-redef]
        CAPABILITY_HASH,
        FINAL_AUDIT,
        HEALTH,
        MANIFEST,
        OUTPUT,
        PROMPT_HASH,
        RUN_ROOT,
        SCHEMA_HASH,
        SOURCE_MANIFEST_HASH,
    )

REQUIRED_FILES = (
    "required-claims-input.json",
    "exact-json-schema.json",
    "provider-capability-snapshot.json",
    "response-format-parameters.json",
    "rendered-system-prompt.txt",
    "rendered-user-prompt.txt",
    "prompt-metadata.json",
    "citation-registry.json",
    "raw-provider-response.json",
    "provider-response-envelope.json",
    "result.json",
    "result.csv",
    "retrieval-trace.json",
    "context-trace.json",
    "request-ledger.jsonl",
    "run-metadata.json",
)
PRE_REQUEST_ORDER = (
    "manifest_validated", "required_claim_input_persisted",
    "required_claim_input_hash_recorded", "citation_registry_persisted",
    "citation_registry_hash_recorded", "exact_schema_persisted",
    "schema_hash_recorded", "prompt_rendered", "prompt_hash_recorded",
    "provider_capability_snapshot_persisted",
    "response_format_parameters_persisted", "request_id_allocated",
    "budget_reserved", "request_prepared", "request_started",
)
HISTORICAL_FILES = (
    "data/evaluation/evidence-qa-dev-v3.json",
    "data/evaluation/evidence-qa-dev-v3.csv",
    "data/evaluation/evidence-qa-dev-v3-final-audit.json",
    "data/evaluation/evidence-qa-dev-v3-1-readiness-v1.json",
    "data/evaluation/stage13-review-hash-migration-v1.json",
)


def tracked_unchanged(path: str) -> bool:
    current = Path(path).read_bytes()
    baseline = subprocess.run(
        ["git", "show", f"HEAD:{path}"], check=True, capture_output=True
    ).stdout
    return current == baseline


def load_ledger(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    summary = json.loads(OUTPUT.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    health = json.loads(HEALTH.read_text(encoding="utf-8"))
    settings = Settings()
    run_audits: list[dict[str, Any]] = []
    secret_hits = 0
    for run_id in summary["selected_runs"]:
        if not run_id:
            continue
        run_dir = RUN_ROOT / run_id
        result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
        metadata = json.loads((run_dir / "run-metadata.json").read_text(encoding="utf-8"))
        ledger = load_ledger(run_dir / "request-ledger.jsonl")
        names = [row["event"] for row in ledger]
        format_body = json.loads((run_dir / "response-format-parameters.json").read_text(encoding="utf-8"))
        corpus = "\n".join(
            path.read_text(encoding="utf-8", errors="ignore")
            for path in run_dir.iterdir()
            if path.is_file()
        )
        key = settings.llm_api_key or ""
        hits = int(bool(key and key in corpus))
        hits += int("Authorization: Bearer " in corpus or '"Authorization"' in corpus)
        secret_hits += hits
        unique_events = len({row["event_id"] for row in ledger}) == len(ledger)
        required_order = all(name in names for name in PRE_REQUEST_ORDER)
        required_order = required_order and [names.index(name) for name in PRE_REQUEST_ORDER] == sorted(names.index(name) for name in PRE_REQUEST_ORDER)
        post_expected = ["raw_response_received", "provider_usage_recorded", "raw_response_persisted", "response_parsing_started", "raw_schema_validation_started"]
        post_order = all(name in names for name in post_expected)
        post_order = post_order and [names.index(name) for name in post_expected] == sorted(names.index(name) for name in post_expected)
        request_allocations = [row.get("request_id") for row in ledger if row["event"] == "request_id_allocated"]
        run_audits.append(
            {
                "run_id": run_id,
                "question_id": result["question_id"],
                "all_artifacts_present": all((run_dir / name).exists() for name in REQUIRED_FILES),
                "event_ids_unique": unique_events,
                "pre_request_order_valid": required_order,
                "post_response_order_valid": post_order,
                "request_id_allocated_once": request_allocations == [metadata["request_id"]],
                "ledger_terminal": names[-1] in {"completed", "validation_failed", "request_failed"},
                "usage_before_parse": names.index("provider_usage_recorded") < names.index("response_parsing_started") if post_order else False,
                "response_format_exact": format_body == {"response_format": {"type": "json_object"}, "json_schema_sent": False, "tools_sent": False, "functions_sent": False},
                "input_hash_valid": canonical_hash(json.loads((run_dir / "required-claims-input.json").read_text(encoding="utf-8"))) == result["required_claim_input_hash"],
                "registry_hash_valid": result["citation_registry_hash_valid"],
                "schema_hash_valid": result["schema_hash_valid"],
                "prompt_hash_valid": result["prompt_hash_valid"],
                "capability_hash_valid": result["capability_snapshot_hash_valid"],
                "no_retry": result["retries"] == 0,
                "no_normalization": not result["formal_normalization_used"],
                "no_reranker": not result["reranker_called"],
                "no_leakage": not any(metadata[key] for key in ("gold_evidence_used_for_allocation", "oracle_used", "human_pilot_used")),
                "secret_hits": hits,
            }
        )
    q005 = next((row for row in summary["per_query"] if row["question_id"] == "q005"), {})
    history = {path: tracked_unchanged(path) for path in HISTORICAL_FILES}
    checks = {
        "health_safe": bool(health.get("safe_to_start_batch")),
        "manifest_hash": manifest["manifest_hash"] == SOURCE_MANIFEST_HASH,
        "prompt_hash": manifest["configuration"]["prompt_hash"] == PROMPT_HASH,
        "schema_hash": manifest["configuration"]["schema_hash"] == SCHEMA_HASH,
        "capability_hash": manifest["configuration"]["provider_capability_snapshot_hash"] == CAPABILITY_HASH,
        "fixed_question_order": manifest["question_ids"] == DEV_IDS,
        "all_run_audits_pass": len(run_audits) == 10 and all(all(value for key, value in row.items() if key not in {"run_id", "question_id", "secret_hits"}) and row["secret_hits"] == 0 for row in run_audits),
        "q005_refusal": bool(q005.get("refusal_correct")),
        "historical_files_unchanged": all(history.values()),
        "no_secret_hits": secret_hits == 0,
        "no_archive_artifacts": not any(path.suffix.lower() in {".zip", ".bak"} for path in RUN_ROOT.rglob("*")),
        "full_qa_not_run": not summary["full_qa_run"],
        "deep_research_not_run": not summary["deep_research_run"],
    }
    payload = {
        "schema_version": "evidence-qa-dev-v3-1-final-audit-v1",
        "checks": checks,
        "run_audits": run_audits,
        "historical_file_stability": history,
        "secret_hits": secret_hits,
        "engineering_gate": summary["dev_v3_1_engineering_gate"] and all(checks.values()),
        "quality_candidate_gate": summary["dev_v3_1_quality_candidate_gate"],
        "ready_for_full_qa": summary["ready_for_full_qa"] and all(checks.values()),
        "ready_for_dev_v3_1_checkpoint_commit": all(checks.values()),
        "full_qa_run": False,
        "deep_research_run": False,
        "production_ready": False,
        "v1_0_status": "not_satisfied",
        "current_release": "v0.9.0-rc3",
    }
    FINAL_AUDIT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"engineering_gate": payload["engineering_gate"], "quality_candidate_gate": payload["quality_candidate_gate"], "ready_for_full_qa": payload["ready_for_full_qa"], "secret_hits": secret_hits}))


if __name__ == "__main__":
    main()
