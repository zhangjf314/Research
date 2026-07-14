"""Offline final audit for the immutable Stage 11D live run evidence."""

from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path
from typing import Any

from paper_research.agents.smoke_artifacts import DEFAULT_RUN_ROOT, load_runs

EXPECTED_RUNS = {
    "live-q003-798ac68288e0",
    "live-q049-1d47dc6a1ab8",
    "live-q049-24a797337315",
    "live-q049-4c11db9a2c1d",
    "live-q005-03f669606bb7",
    "live-q049-03b2d6b68ca3",
    "live-q049-2e5704e44d91",
}
SELECTED_RUNS = {
    "live-q003-798ac68288e0",
    "live-q049-4c11db9a2c1d",
    "live-q005-03f669606bb7",
}
REQUIRED_FILES = {
    "result.json",
    "result.csv",
    "trace.jsonl",
    "request-ledger.jsonl",
    "checkpoint-summary.json",
    "run-metadata.json",
}
OUTPUT = Path("data/evaluation/stage11d-final-audit-v1.json")
CHECKPOINT = Path("data/checkpoints/deep-research-smoke-v1.sqlite3")


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def secret_values() -> list[str]:
    values = []
    for line in Path(".env").read_text(encoding="utf-8-sig").splitlines():
        if "=" not in line or line.lstrip().startswith("#"):
            continue
        key, value = line.split("=", 1)
        if value and ("API_KEY" in key or "PASSWORD" in key):
            values.append(value)
    return values


def main() -> int:
    runs = {run["metadata"]["run_id"]: run for run in load_runs(DEFAULT_RUN_ROOT)}
    if set(runs) != EXPECTED_RUNS:
        raise RuntimeError(f"unexpected run set: {sorted(set(runs) ^ EXPECTED_RUNS)}")
    connection = sqlite3.connect(CHECKPOINT)
    run_rows = []
    logical_request_ids = []
    all_event_ids = []
    completed_usage_ok = True
    failed_reservation_ok = True
    sqlite_consistency_ok = True
    for run_id in sorted(runs):
        run = runs[run_id]
        run_dir = run["run_dir"]
        result = run["result"]
        metadata = run["metadata"]
        trace = jsonl(run_dir / "trace.jsonl")
        ledger = jsonl(run_dir / "request-ledger.jsonl")
        checkpoint_summary = json.loads(
            (run_dir / "checkpoint-summary.json").read_text(encoding="utf-8")
        )
        sqlite_row = connection.execute(
            "SELECT state_json FROM smoke_runs WHERE run_id=?", (run_id,)
        ).fetchone()
        if sqlite_row is None:
            raise RuntimeError(f"missing SQLite checkpoint: {run_id}")
        state = json.loads(sqlite_row[0])
        migrated = metadata.get("historical_role") is not None
        sqlite_consistency_ok &= (
            state["question_id"] == result["question_id"]
            and state["total_tokens"] == result["total_tokens"]
            and state["llm_requests"] == result["request_attempt_count"]
        )
        if not migrated:
            sqlite_consistency_ok &= (
                state["status"] == result["graph_status"]
                and state.get("reserved_total_tokens", 0) == result["reserved_total_tokens"]
            )
        usage_total = sum(row["total_tokens"] for row in checkpoint_summary["usage_records"])
        if result["graph_status"] in {"completed", "refused"}:
            completed_usage_ok &= (
                usage_total == result["total_tokens"]
                and result["usage_record_count"] == 1
                and result["reserved_total_tokens"] == 0
            )
        if result["graph_status"] == "provider_failed":
            failed_reservation_ok &= (
                result["usage_record_count"] == 0
                and result["reserved_total_tokens"] > 0
                and "unavailable_after_send_attempt" in result["usage_source"]
            )
        request_ids = [value for value in result["request_ids"] if value]
        logical_request_ids.extend(request_ids)
        all_event_ids.extend(event["event_id"] for event in trace)
        run_rows.append(
            {
                "run_id": run_id,
                "question_id": metadata["question_id"],
                "attempt_number": metadata["attempt_number"],
                "parent_run_id": metadata["parent_run_id"],
                "status": result["graph_status"],
                "resume_count": result["resume_count"],
                "request_attempt_count": result["request_attempt_count"],
                "provider_completed_request_count": result[
                    "provider_completed_request_count"
                ],
                "usage_record_count": result["usage_record_count"],
                "settled_tokens": result["total_tokens"],
                "active_reservation": result["reserved_total_tokens"],
                "monetary_cost_usd": result["monetary_cost_usd"],
                "citation_validation": result["citation_validation"],
                "selected_by_latest_successful": run_id in SELECTED_RUNS,
                "failure_type": next(
                    (
                        row.get("failure_type")
                        for row in reversed(ledger)
                        if row.get("failure_type")
                    ),
                    None,
                ),
                "run_directory_complete": REQUIRED_FILES
                <= {path.name for path in run_dir.iterdir()},
                "trace_event_count": len(trace),
                "trace_event_ids_unique": len(trace)
                == len({event["event_id"] for event in trace}),
                "request_ledger_events": [row.get("event") for row in ledger],
            }
        )

    summary = json.loads(
        Path("data/evaluation/deep-research-smoke-v1.json").read_text(encoding="utf-8")
    )
    with Path("data/evaluation/deep-research-smoke-v1.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        csv_rows = list(csv.DictReader(handle))
    top_traces = jsonl(Path("data/evaluation/deep-research-smoke-traces-v1.jsonl"))
    markdown = Path("docs/deep-research-smoke-v1.md").read_text(encoding="utf-8")
    top_consistent = (
        set(summary["selected_run_ids"]) == SELECTED_RUNS
        and {row["run_id"] for row in csv_rows} == SELECTED_RUNS
        and {row["run_id"] for row in top_traces} == EXPECTED_RUNS
        and all(run_id in markdown for run_id in EXPECTED_RUNS)
    )

    selected_results = [runs[run_id]["result"] for run_id in SELECTED_RUNS]
    selected_totals = {
        "request_attempts": sum(row["request_attempt_count"] for row in selected_results),
        "provider_completed_requests": sum(
            row["provider_completed_request_count"] for row in selected_results
        ),
        "input_tokens": sum(row["input_tokens"] for row in selected_results),
        "output_tokens": sum(row["output_tokens"] for row in selected_results),
        "total_tokens": sum(row["total_tokens"] for row in selected_results),
        "monetary_cost_usd": "0",
        "elapsed_seconds": round(
            sum(row["elapsed_seconds"] for row in selected_results), 6
        ),
    }
    failed = [row for row in run_rows if row["status"] == "provider_failed"]
    artifact_files = [
        path
        for path in DEFAULT_RUN_ROOT.rglob("*")
        if path.is_file()
    ] + [
        Path("data/evaluation/deep-research-smoke-v1.json"),
        Path("data/evaluation/deep-research-smoke-v1.csv"),
        Path("data/evaluation/deep-research-smoke-traces-v1.jsonl"),
        Path("docs/deep-research-smoke-v1.md"),
    ]
    secrets = secret_values()
    secret_hits = 0
    auth_header_hits = 0
    for path in artifact_files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        secret_hits += int(any(secret in text for secret in secrets))
        auth_header_hits += int("Authorization" in text or "Bearer " in text)

    q005 = runs["live-q005-03f669606bb7"]["result"]
    audit = {
        "status": "STAGE11D_FINAL_OFFLINE_AUDIT_COMPLETE",
        "runs": run_rows,
        "selected_totals": selected_totals,
        "failed_request_attempts": sum(row["request_attempt_count"] for row in failed),
        "unresolved_active_reservations": len(failed),
        "conservative_reserved_tokens": sum(row["active_reservation"] for row in failed),
        "checks": {
            "all_run_directories_complete": all(
                row["run_directory_complete"] for row in run_rows
            ),
            "top_level_summary_consistent": top_consistent,
            "sqlite_export_consistent": bool(sqlite_consistency_ok),
            "completed_usage_sums_match": bool(completed_usage_ok),
            "failed_reservations_valid": bool(failed_reservation_ok),
            "logical_request_ids_unique": len(logical_request_ids)
            == len(set(logical_request_ids)),
            "trace_event_ids_unique": len(all_event_ids) == len(set(all_event_ids)),
            "q005_has_no_citations": q005["citation_count"] == 0,
            "reranker_never_called": all(
                not run["result"]["reranker_called"] for run in runs.values()
            ),
            "template_fallback_never_used": all(
                run["result"]["provider"] != "template" for run in runs.values()
            ),
            "api_key_value_hits": secret_hits,
            "authorization_header_hits": auth_header_hits,
        },
    }
    count_checks = {"api_key_value_hits", "authorization_header_hits"}
    boolean_checks = [
        value for key, value in audit["checks"].items() if key not in count_checks
    ]
    if not all(boolean_checks) or secret_hits != 0 or auth_header_hits != 0:
        raise RuntimeError(f"final audit failed: {audit['checks']}")
    OUTPUT.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": audit["status"], "checks": audit["checks"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
