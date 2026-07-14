"""Export immutable historical Stage 11D checkpoints into isolated run directories."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from paper_research.agents.bounded_smoke import SmokeState, smoke_configuration
from paper_research.agents.smoke_artifacts import (
    DEFAULT_RUN_ROOT,
    iso_timestamp,
    state_result,
    write_run_artifacts,
)
from paper_research.config import Settings

CHECKPOINT = Path("data/checkpoints/deep-research-smoke-v1.sqlite3")
AUDIT = Path("docs/stage11d1-live-run-reliability-audit.md")
RUNS = {
    "live-q003-798ac68288e0": (1, None, "historical_success"),
    "live-q049-1d47dc6a1ab8": (1, None, "historical_success"),
    "live-q049-24a797337315": (
        2,
        "live-q049-1d47dc6a1ab8",
        "historical_failed_attempt",
    ),
}


def migrated_state(raw: dict, historical_reservation_ceiling: int) -> SmokeState:
    state = SmokeState(**raw)
    state.request_attempt_count = state.llm_requests
    state.provider_completed_request_count = 1 if state.answer else 0
    state.usage_record_count = len(state.usage_records)
    if state.usage_records:
        usage = state.usage_records[0]
        state.request_records = [
            {
                "request_id": usage["request_id"],
                "node": "synthesize",
                "request_status": "completed",
                "usage_status": usage["usage_source"],
                "reservation_status": "not_retained",
                "actual_input_tokens": usage["input_tokens"],
                "actual_output_tokens": usage["output_tokens"],
                "actual_total_tokens": usage["total_tokens"],
                "historical_migration": True,
            }
        ]
    elif state.llm_requests:
        state.reserved_total_tokens = historical_reservation_ceiling
        state.reserved_output_tokens = min(2048, historical_reservation_ceiling)
        state.reserved_input_tokens = (
            historical_reservation_ceiling - state.reserved_output_tokens
        )
        state.request_records = [
            {
                "request_id": None,
                "request_id_status": "not_retained",
                "node": "synthesize",
                "request_status": "failed_after_send_unknown",
                "usage_status": "unavailable_after_send_attempt",
                "reservation_status": "not_retained",
                "conservative_reservation_tokens": historical_reservation_ceiling,
                "reservation_basis": "historical_per_query_budget_ceiling",
                "failure_type": "ConnectError",
                "historical_migration": True,
            }
        ]
        state.status = "provider_failed"
        state.budget_stop_reason = None
        state.budget_accounting_status = "historical_unknown_usage_conservative_reserved"
    return state


def main() -> int:
    settings = Settings()
    policy, _ = smoke_configuration(settings)
    connection = sqlite3.connect(CHECKPOINT)
    exported = []
    raw_states = {}
    for run_id, (attempt, parent, role) in RUNS.items():
        row = connection.execute(
            "SELECT state_json,updated_at FROM smoke_runs WHERE run_id=?", (run_id,)
        ).fetchone()
        if row is None:
            raise RuntimeError(f"missing historical checkpoint: {run_id}")
        raw = json.loads(row[0])
        raw_states[run_id] = raw
        state = migrated_state(raw, settings.deep_research_max_tokens_per_query)
        result = state_result(
            state, provider="siliconflow", model="Qwen/Qwen3-8B", policy=policy
        )
        run_dir = write_run_artifacts(
            DEFAULT_RUN_ROOT,
            state,
            result,
            mode="live",
            attempt_number=attempt,
            parent_run_id=parent,
            provider="siliconflow",
            model="Qwen/Qwen3-8B",
            policy=policy,
            settings=settings,
            historical_role=role,
            ended_at=float(row[1]),
        )
        exported.append({"run_id": run_id, "status": state.status, "run_dir": str(run_dir)})
    q003 = raw_states["live-q003-798ac68288e0"]
    success = raw_states["live-q049-1d47dc6a1ab8"]
    failed = raw_states["live-q049-24a797337315"]
    lines = [
        "# Stage 11D.1 Live Run Reliability Audit",
        "",
        "## Evidence and chronology",
        "",
        f"- q003 `{q003['run_id']}`: started `{iso_timestamp(q003['started_at'])}`, completed, "
        f"requests={q003['llm_requests']}, tokens={q003['total_tokens']}.",
        f"- q049 `{success['run_id']}`: started "
        f"`{iso_timestamp(success['started_at'])}`, completed first, "
        f"requests={success['llm_requests']}, tokens={success['total_tokens']}, citation=passed.",
        f"- q049 `{failed['run_id']}`: started `{iso_timestamp(failed['started_at'])}`, "
        "ran later, "
        "one provider attempt ended with ConnectError and unavailable usage.",
        "",
        "The later failed q049 invocation wrote the top-level JSON, CSV, trace and Markdown "
        "because the old runner called `write_outputs` after every run. The old selection "
        "semantics were therefore implicit latest-attempt-wins, with no question/run isolation.",
        "",
        "## Ledgers",
        "",
        "The completed q049 ledger contains one retained request ID, "
        f"input={success['input_tokens']}, output={success['output_tokens']}, "
        f"total={success['total_tokens']}, cost=0, and six unique events.",
        "",
        "The failed q049 checkpoint records one attempted request, zero settled tokens, no usage "
        "record, three completed graph events, ConnectError and the old `usage_unavailable` budget "
        "label. Its request ID was not retained. Migration marks that field `not_retained`; "
        "it does "
        "not invent an identifier or retroactively alter SQLite.",
        "",
        "A generic ConnectError cannot prove whether failure occurred before bytes were sent, "
        "while "
        "connecting, or before a response arrived. It is conservatively classified as "
        "`failed_after_send_unknown` with `unavailable_after_send_attempt`.",
        "",
        "## Root cause and risk",
        "",
        "The old synthesize node generated request_id only in a local variable immediately before "
        "the provider call and appended it only on successful usage settlement. Its exception path "
        "converted LLMProviderError into BudgetBlocked, producing the inaccurate budget_blocked "
        "status. Unknown usage was represented by absent usage plus numeric zero totals.",
        "",
        "Without isolated run directories, a later failure could overwrite successful evidence. "
        "Separate CLI invocations also initialized global accounting independently, creating a "
        "cross-run budget visibility risk. Stage 11D.1 isolates attempts, preserves conservative "
        "reservations, and makes summary selection explicit and auditable.",
    ]
    AUDIT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"exported": exported}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
