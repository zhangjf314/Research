"""Isolated run artifacts and deterministic Stage 11D summary selection."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from paper_research.agents.bounded_smoke import BillingPolicy, SmokeState
from paper_research.config import Settings

DEFAULT_RUN_ROOT = Path("data/evaluation/deep-research-smoke-v1/runs")
SUCCESS_STATUSES = {"completed", "refused"}


def iso_timestamp(value: float) -> str:
    return datetime.fromtimestamp(value, UTC).isoformat()


def safe_config_fingerprint(settings: Settings, policy: BillingPolicy) -> str:
    safe = {
        "embedding_provider": settings.embedding_provider,
        "embedding_model": settings.embedding_model,
        "embedding_dimensions": settings.embedding_dimensions,
        "rerank_enabled": settings.rerank_enabled,
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
        "llm_temperature": settings.llm_temperature,
        "llm_max_retries": 0,
        "billing_mode": policy.mode,
        "input_price": str(policy.input_price),
        "output_price": str(policy.output_price),
        "max_cost": str(policy.max_cost),
        "deep_research_mode": settings.deep_research_mode,
    }
    payload = json.dumps(safe, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


class RequestLedger:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")


def state_result(
    state: SmokeState, *, provider: str, model: str, policy: BillingPolicy
) -> dict[str, Any]:
    answer = state.answer or {}
    claims = answer.get("claims") or []
    attempts = state.request_attempt_count or state.llm_requests
    return {
        "run_id": state.run_id,
        "question_id": state.question_id,
        "graph_status": state.status,
        "nodes_visited": state.nodes_visited,
        "iteration_count": state.iteration_count,
        "retrieval_calls": state.retrieval_calls,
        "llm_request_count": attempts,
        "request_attempt_count": attempts,
        "provider_completed_request_count": state.provider_completed_request_count,
        "usage_record_count": state.usage_record_count or len(state.usage_records),
        "request_ids": [row.get("request_id") for row in state.request_records],
        "input_tokens": state.input_tokens,
        "output_tokens": state.output_tokens,
        "total_tokens": state.total_tokens,
        "reserved_input_tokens": state.reserved_input_tokens,
        "reserved_output_tokens": state.reserved_output_tokens,
        "reserved_total_tokens": state.reserved_total_tokens,
        "budget_accounting_status": state.budget_accounting_status,
        "usage_source": sorted(
            {
                row.get("usage_source") or row.get("usage_status")
                for row in [*state.usage_records, *state.request_records]
                if row.get("usage_source") or row.get("usage_status")
            }
        ),
        "billing_mode": policy.mode,
        "monetary_cost_usd": state.monetary_cost_usd,
        "cost_basis": policy.cost_basis,
        "elapsed_seconds": state.elapsed_seconds,
        "checkpoint_ids": [state.run_id],
        "resume_count": state.resume_count,
        "final_answerable_state": answer.get("answerable"),
        "claim_count": len(claims),
        "citation_count": sum(len(claim.get("citations") or []) for claim in claims),
        "citation_validation": state.citation_validation,
        "refusal_reason": answer.get("refusal_reason"),
        "errors": state.errors,
        "budget_stop_reason": state.budget_stop_reason,
        "reranker_called": False,
        "provider": provider,
        "model": model,
        "prompt_versions": ["qa-production-v1"],
    }


def write_run_artifacts(
    root: Path,
    state: SmokeState,
    result: dict[str, Any],
    *,
    mode: str,
    attempt_number: int,
    parent_run_id: str | None,
    provider: str,
    model: str,
    policy: BillingPolicy,
    settings: Settings,
    historical_role: str | None = None,
    ended_at: float | None = None,
) -> Path:
    run_dir = root / state.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fields = [key for key, value in result.items() if not isinstance(value, (dict, list))]
    with (run_dir / "result.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({key: result[key] for key in fields})
    (run_dir / "trace.jsonl").write_text(
        "".join(
            json.dumps({"run_id": state.run_id, **event}, ensure_ascii=False) + "\n"
            for event in state.events
        ),
        encoding="utf-8",
    )
    ledger_path = run_dir / "request-ledger.jsonl"
    if historical_role or not ledger_path.exists():
        ledger_path.write_text(
            "".join(
                json.dumps({"event": "migrated_request", **row}, ensure_ascii=False) + "\n"
                for row in state.request_records
            ),
            encoding="utf-8",
        )
    checkpoint_summary = {
        "run_id": state.run_id,
        "status": state.status,
        "current_node": state.current_node,
        "request_records": state.request_records,
        "usage_records": state.usage_records,
        "reserved_total_tokens": state.reserved_total_tokens,
        "budget_accounting_status": state.budget_accounting_status,
        "state_sha256": hashlib.sha256(
            json.dumps(asdict(state), sort_keys=True).encode()
        ).hexdigest(),
    }
    (run_dir / "checkpoint-summary.json").write_text(
        json.dumps(checkpoint_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    ended = ended_at or (state.started_at + state.elapsed_seconds)
    paths = {
        name: str((run_dir / name).resolve())
        for name in (
            "result.json",
            "result.csv",
            "trace.jsonl",
            "request-ledger.jsonl",
            "checkpoint-summary.json",
            "run-metadata.json",
        )
    }
    metadata = {
        "run_id": state.run_id,
        "question_id": state.question_id,
        "mode": mode,
        "attempt_number": attempt_number,
        "parent_run_id": parent_run_id,
        "started_at": iso_timestamp(state.started_at),
        "ended_at": iso_timestamp(ended),
        "status": state.status,
        "provider": provider,
        "model": model,
        "billing_mode": policy.mode,
        "code_revision": git_revision(),
        "config_fingerprint": safe_config_fingerprint(settings, policy),
        "historical_role": historical_role,
        "output_paths": paths,
    }
    (run_dir / "run-metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return run_dir


def load_runs(root: Path = DEFAULT_RUN_ROOT) -> list[dict[str, Any]]:
    runs = []
    if not root.exists():
        return runs
    for metadata_path in sorted(root.glob("*/run-metadata.json")):
        run_dir = metadata_path.parent
        runs.append(
            {
                "metadata": json.loads(metadata_path.read_text(encoding="utf-8")),
                "result": json.loads((run_dir / "result.json").read_text(encoding="utf-8")),
                "run_dir": run_dir,
            }
        )
    return runs


def select_runs(
    runs: list[dict[str, Any]], policy: str, run_id: str | None = None
) -> list[dict[str, Any]]:
    if policy == "explicit-run-id":
        selected = [run for run in runs if run["metadata"]["run_id"] == run_id]
        if len(selected) != 1:
            raise ValueError("explicit-run-id requires one existing --run-id")
        return selected
    grouped: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        grouped.setdefault(run["metadata"]["question_id"], []).append(run)
    selected = []
    for question_runs in grouped.values():
        candidates = question_runs
        if policy == "latest-successful":
            candidates = [
                run for run in question_runs if run["result"]["graph_status"] in SUCCESS_STATUSES
            ]
        if candidates:
            selected.append(max(candidates, key=lambda run: run["metadata"]["ended_at"]))
    return sorted(selected, key=lambda run: run["metadata"]["question_id"])
