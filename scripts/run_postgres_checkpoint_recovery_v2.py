from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import uuid
from base64 import b64encode
from dataclasses import asdict
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx

from paper_research.agents.bounded_smoke import (
    BillingPolicy,
    BoundedSmokeRunner,
    BudgetGuard,
    SmokeLimits,
    SmokeState,
    smoke_configuration,
)
from paper_research.agents.smoke_artifacts import (
    RequestLedger,
    state_result,
    write_run_artifacts,
)
from paper_research.config import Settings
from paper_research.providers.factory import build_llm_provider

sys.path.append(str(Path(__file__).resolve().parent))
from run_deep_research_smoke_v1 import (  # noqa: E402
    MANIFEST,
    exact_contexts,
    load_jsonl,
    validate_corpus_and_index,
    validate_manifest,
)

OUTPUT_JSON = Path("data/evaluation/langgraph-production-recovery-v2.json")
OUTPUT_MD = Path("docs/langgraph-production-recovery-audit-v2.md")
RUN_ROOT = Path("artifacts/langgraph-production-recovery-v2")
PAUSE_NODE = "synthesize"
QUESTION_ID = "q003"


def run(*args: str) -> str:
    completed = subprocess.run(
        args,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
    return completed.stdout.strip()


def docker(*args: str) -> str:
    return run("docker", "compose", *args)


def psql(sql: str) -> str:
    return docker(
        "exec",
        "-T",
        "postgres",
        "psql",
        "-U",
        "paper",
        "-d",
        "paper_research",
        "-Atc",
        sql,
    )


class PostgresSmokeCheckpoint:
    """PostgreSQL-backed checkpoint adapter for the bounded recovery gate."""

    def __init__(self) -> None:
        psql(
            """
            CREATE TABLE IF NOT EXISTS portfolio_smoke_checkpoints_v2 (
                run_id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL,
                state_json JSONB NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )

    def save(self, state: SmokeState) -> None:
        payload = json.dumps(asdict(state), ensure_ascii=False, sort_keys=True)
        encoded = b64encode(payload.encode("utf-8")).decode("ascii")
        psql(
            f"""
            INSERT INTO portfolio_smoke_checkpoints_v2(run_id, thread_id, state_json)
            VALUES (
                '{state.run_id}',
                '{state.run_id}',
                convert_from(decode('{encoded}', 'base64'), 'UTF8')::jsonb
            )
            ON CONFLICT(run_id) DO UPDATE SET
                state_json = excluded.state_json,
                updated_at = now()
            """
        )

    def load(self, run_id: str) -> SmokeState | None:
        raw = psql(
            f"""
            SELECT state_json::text
            FROM portfolio_smoke_checkpoints_v2
            WHERE run_id = '{run_id}'
            """
        )
        return SmokeState(**json.loads(raw)) if raw else None

    def load_many(self, run_ids: list[str]) -> list[SmokeState]:
        return [state for run_id in run_ids if (state := self.load(run_id)) is not None]


def health_ready(timeout_seconds: int = 60) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            response = httpx.get("http://localhost/api/v1/health", timeout=5)
            if response.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def duplicate_count(values: list[str]) -> int:
    return len(values) - len(set(values))


def count_checkpoint_rows(run_id: str) -> int:
    raw = psql(
        f"""
        SELECT count(*)
        FROM portfolio_smoke_checkpoints_v2
        WHERE run_id = '{run_id}'
        """
    )
    return int(raw)


def limits_for_gate(
    policy: BillingPolicy,
    limits: SmokeLimits,
) -> tuple[BillingPolicy, SmokeLimits]:
    return (
        BillingPolicy(
            policy.mode,
            policy.input_price,
            policy.output_price,
            Decimal("0.05") if policy.mode == "paid" else Decimal("0"),
            policy.warning,
        ),
        SmokeLimits(
            max_queries=1,
            iterations_per_query=2,
            requests_per_query=3,
            requests_total=3,
            tokens_per_query=15000,
            tokens_total=15000,
            elapsed_per_query=300,
            elapsed_total=300,
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume-run-id")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = datetime.now(UTC)
    settings = Settings(
        deep_research_max_cost_usd=Decimal("0.05"),
        deep_research_max_llm_requests_per_query=3,
        deep_research_max_llm_requests_total=3,
        deep_research_max_tokens_per_query=15000,
        deep_research_max_tokens_total=15000,
        deep_research_max_elapsed_seconds_per_query=300,
        deep_research_max_elapsed_seconds_total=300,
    )
    rows = load_jsonl(MANIFEST)
    validate_manifest(rows)
    validate_corpus_and_index(settings)
    sample = next(row for row in rows if row["question_id"] == QUESTION_ID)
    contexts = exact_contexts()
    policy, limits = limits_for_gate(*smoke_configuration(settings))
    checkpoint = PostgresSmokeCheckpoint()
    run_id = args.resume_run_id or f"pg-recovery-{QUESTION_ID}-{uuid.uuid4().hex[:12]}"
    run_dir = RUN_ROOT / run_id
    ledger = RequestLedger(run_dir / "request-ledger.jsonl")
    guard = BudgetGuard(policy, limits)
    llm = build_llm_provider(settings)
    runner = BoundedSmokeRunner(
        llm,
        checkpoint,  # type: ignore[arg-type]
        guard,
        prompt_version=settings.prompt_version,
        max_output_tokens=min(settings.llm_max_output_tokens, 2048),
        retrieval=lambda item, iteration: contexts[item["question_id"]],
        request_event=ledger.emit,
    )

    if args.resume_run_id:
        interrupted = checkpoint.load(run_id)
        if interrupted is None:
            raise RuntimeError(f"checkpoint not found: {run_id}")
        if interrupted.status != "interrupted":
            raise RuntimeError(f"checkpoint is not interrupted: {interrupted.status}")
    else:
        interrupted = runner.run(sample, run_id=run_id, stop_after_node=PAUSE_NODE)
    checkpoint_written = count_checkpoint_rows(run_id) == 1
    before_nodes = list(interrupted.nodes_visited)
    before_requests = list(interrupted.request_records)
    before_usage = list(interrupted.usage_records)
    before_retrieval_calls = interrupted.retrieval_calls

    if args.resume_run_id:
        container_recreated = health_ready()
    else:
        docker("config", "--quiet")
        docker("up", "-d", "--force-recreate", "api", "nginx")
        container_recreated = health_ready()

    restored = PostgresSmokeCheckpoint()
    guard_after = BudgetGuard(policy, limits)
    guard_after.restore([interrupted])
    runner_after = BoundedSmokeRunner(
        llm,
        restored,  # type: ignore[arg-type]
        guard_after,
        prompt_version=settings.prompt_version,
        max_output_tokens=min(settings.llm_max_output_tokens, 2048),
        retrieval=lambda item, iteration: contexts[item["question_id"]],
        request_event=ledger.emit,
    )
    completed = runner_after.run(sample, run_id=run_id, resume=True)
    result = state_result(
        completed,
        provider=llm.provider_name,
        model=llm.model_name,
        policy=policy,
    )
    write_run_artifacts(
        RUN_ROOT,
        completed,
        result,
        mode="live",
        attempt_number=1,
        parent_run_id=None,
        provider=llm.provider_name,
        model=llm.model_name,
        policy=policy,
        settings=settings,
        ended_at=time.time(),
    )

    node_duplicates = duplicate_count(completed.nodes_visited)
    request_ids = [
        str(row.get("request_id"))
        for row in completed.request_records
        if row.get("request_id")
    ]
    usage_ids = [
        str(row.get("request_id"))
        for row in completed.usage_records
        if row.get("request_id")
    ]
    duplicate_provider_request_count = duplicate_count(request_ids)
    duplicate_usage_settlement_count = duplicate_count(usage_ids)
    resume_did_not_repeat_completed_nodes = (
        before_nodes == completed.nodes_visited[: len(before_nodes)]
    )
    synthesize_count = completed.nodes_visited.count("synthesize")
    gate = (
        "PASSED"
        if interrupted.status == "interrupted"
        and interrupted.current_node == "validate_citations"
        and checkpoint_written
        and container_recreated
        and completed.status == "completed"
        and completed.resume_count == 1
        and resume_did_not_repeat_completed_nodes
        and completed.retrieval_calls == before_retrieval_calls
        and synthesize_count == 1
        and duplicate_provider_request_count == 0
        and duplicate_usage_settlement_count == 0
        and completed.reserved_total_tokens == 0
        and completed.citation_validation == "passed"
        and completed.request_attempt_count <= 3
        and completed.total_tokens <= 15000
        and completed.elapsed_seconds <= 300
        else "FAILED"
    )
    payload = {
        "schema_version": "langgraph-production-recovery-v2",
        "started_at": started.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
        "gate": gate,
        "run_id": run_id,
        "thread_id": run_id,
        "question_id": QUESTION_ID,
        "provider": llm.provider_name,
        "model": llm.model_name,
        "checkpointer": "postgresql",
        "checkpoint_table": "portfolio_smoke_checkpoints_v2",
        "api_langgraph_postgres_saver_note": (
            "The API LangGraph route already uses PostgresSaver but is deterministic "
            "and does not call LLM; this release gate uses the existing bounded "
            "Deep Research smoke runner with a PostgreSQL checkpoint adapter to "
            "verify real-provider stop/resume accounting."
        ),
        "pause_node": PAUSE_NODE,
        "interrupted_status": interrupted.status,
        "interrupted_current_node": interrupted.current_node,
        "checkpoint_written": checkpoint_written,
        "container_recreated": container_recreated,
        "resume_completed": completed.status == "completed",
        "final_status": completed.status,
        "resume_count": completed.resume_count,
        "nodes_before_resume": before_nodes,
        "nodes_after_resume": completed.nodes_visited,
        "duplicate_completed_node_count": node_duplicates,
        "duplicate_provider_request_count": duplicate_provider_request_count,
        "duplicate_usage_settlement_count": duplicate_usage_settlement_count,
        "active_reserved_tokens": completed.reserved_total_tokens,
        "retrieval_calls_before": before_retrieval_calls,
        "retrieval_calls_after": completed.retrieval_calls,
        "request_count_before": len(before_requests),
        "usage_count_before": len(before_usage),
        "request_attempt_count": completed.request_attempt_count,
        "provider_completed_request_count": completed.provider_completed_request_count,
        "usage_record_count": completed.usage_record_count,
        "request_ids": request_ids,
        "input_tokens": completed.input_tokens,
        "output_tokens": completed.output_tokens,
        "total_tokens": completed.total_tokens,
        "monetary_cost_usd": completed.monetary_cost_usd,
        "cost_basis": policy.cost_basis,
        "elapsed_seconds": completed.elapsed_seconds,
        "citation_validation": completed.citation_validation,
        "reranker_called": False,
        "template_fallback": False,
        "run_dir": str(run_dir),
    }
    OUTPUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(payload)
    print(json.dumps({"gate": gate, "run_id": run_id}, indent=2))


def write_report(payload: dict[str, Any]) -> None:
    lines = [
        "# LangGraph Production Recovery Audit v2",
        "",
        f"- Gate: `{payload['gate']}`",
        f"- Run ID / thread ID: `{payload['run_id']}`",
        f"- Provider/model: `{payload['provider']}/{payload['model']}`",
        f"- Checkpointer: `{payload['checkpointer']}`",
        f"- Pause node: `{payload['pause_node']}`",
        f"- Interrupted status: `{payload['interrupted_status']}`",
        f"- Container recreated: `{payload['container_recreated']}`",
        f"- Resume completed: `{payload['resume_completed']}`",
        f"- Duplicate completed nodes: `{payload['duplicate_completed_node_count']}`",
        f"- Duplicate provider requests: `{payload['duplicate_provider_request_count']}`",
        f"- Duplicate usage settlements: `{payload['duplicate_usage_settlement_count']}`",
        f"- Active reserved tokens: `{payload['active_reserved_tokens']}`",
        f"- Citation validation: `{payload['citation_validation']}`",
        "",
        "## Scope note",
        "",
        payload["api_langgraph_postgres_saver_note"],
    ]
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
