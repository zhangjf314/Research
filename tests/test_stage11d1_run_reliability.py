import json
from decimal import Decimal
from pathlib import Path

from paper_research.agents.bounded_smoke import (
    BillingPolicy,
    BoundedSmokeRunner,
    BudgetGuard,
    SmokeLimits,
    SQLiteSmokeCheckpoint,
)
from paper_research.agents.smoke_artifacts import load_runs, select_runs
from paper_research.providers.llm import LLMProviderError
from paper_research.retrieval.context_builder import ContextItem


def limits() -> SmokeLimits:
    return SmokeLimits(3, 2, 4, 12, 40000, 120000, 300, 900)


def sample() -> dict:
    return {
        "question_id": "q049",
        "question": "Compare two papers.",
        "retrieval_scope": "multi_paper",
        "retrieval_filter": {"paper_ids": ["p1", "p2"]},
    }


def context() -> list[ContextItem]:
    return [
        ContextItem(
            chunk_id="c1",
            paper_id="p1",
            block_ids=["b1"],
            block_page_map={"b1": 1},
            section_path=["Method"],
            page_start=1,
            page_end=1,
            evidence="evidence",
            score=1,
        )
    ]


class ConnectFailureLLM:
    provider_name = "siliconflow"
    model_name = "Qwen/Qwen3-8B"

    def __init__(self, checkpoint: SQLiteSmokeCheckpoint):
        self.checkpoint = checkpoint
        self.calls = 0
        self.pre_call_state = None

    def generate_claim_answer(self, question, contexts, prompt_version, audit_metadata=None):
        del question, contexts, prompt_version, audit_metadata
        self.calls += 1
        self.pre_call_state = self.checkpoint.load("run-new")
        raise LLMProviderError(
            "connect failed",
            api_request_count=1,
            retry_reasons=["ConnectError"],
            error_details={"classification": "DEEP_RESEARCH_TCP_CONNECT_ERROR"},
        )


def test_connect_error_has_pre_persisted_id_and_releases_reservation(tmp_path):
    checkpoint = SQLiteSmokeCheckpoint(tmp_path / "checkpoint.sqlite")
    llm = ConnectFailureLLM(checkpoint)
    ledger = []
    runner = BoundedSmokeRunner(
        llm,
        checkpoint,
        BudgetGuard(BillingPolicy("free", Decimal(0), Decimal(0), Decimal(0)), limits()),
        prompt_version="qa-production-v1",
        max_output_tokens=100,
        retrieval=lambda row, iteration: context(),
        request_event=ledger.append,
    )
    state = runner.run(sample(), run_id="run-new")
    assert llm.pre_call_state.request_records[0]["request_status"] == "started"
    assert llm.pre_call_state.request_records[0]["request_id"]
    assert state.status == "provider_failed"
    assert state.budget_stop_reason is None
    assert state.request_attempt_count == 1
    assert state.provider_completed_request_count == 0
    assert state.usage_record_count == 0
    assert state.usage_records == []
    assert state.request_records[0]["request_status"] == "failed_after_send_unknown"
    assert state.request_records[0]["usage_status"] == "released_after_provider_failure"
    assert state.request_records[0]["failure_type"] == "DEEP_RESEARCH_TCP_CONNECT_ERROR"
    assert state.reserved_total_tokens == 0
    assert state.budget_accounting_status == "settled"
    assert [event["event"] for event in ledger] == [
        "request_prepared",
        "TOKEN_BUDGET_RESERVED",
        "request_started",
        "TOKEN_BUDGET_RELEASED",
        "request_failed",
    ]

    resumed = runner.run(sample(), run_id="run-new", resume=True)
    assert resumed.request_attempt_count == 1
    assert resumed.reserved_total_tokens == 0
    assert llm.calls == 1


def create_run(root: Path, run_id: str, status: str, ended_at: str) -> None:
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    metadata = {
        "run_id": run_id,
        "question_id": "q049",
        "ended_at": ended_at,
        "attempt_number": 1 if run_id == "success" else 2,
        "parent_run_id": None if run_id == "success" else "success",
    }
    result = {"run_id": run_id, "question_id": "q049", "graph_status": status}
    (run_dir / "run-metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (run_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
    (run_dir / "trace.jsonl").write_text("", encoding="utf-8")


def test_selection_policies_do_not_let_failure_replace_success(tmp_path):
    create_run(tmp_path, "success", "completed", "2026-07-14T01:00:00+00:00")
    create_run(tmp_path, "failure", "provider_failed", "2026-07-14T02:00:00+00:00")
    runs = load_runs(tmp_path)
    assert select_runs(runs, "latest-successful")[0]["metadata"]["run_id"] == "success"
    assert select_runs(runs, "latest-attempt")[0]["metadata"]["run_id"] == "failure"
    assert select_runs(runs, "explicit-run-id", "failure")[0]["metadata"]["run_id"] == (
        "failure"
    )


def test_two_attempt_directories_are_isolated(tmp_path):
    create_run(tmp_path, "attempt-1", "completed", "2026-07-14T01:00:00+00:00")
    create_run(tmp_path, "attempt-2", "provider_failed", "2026-07-14T02:00:00+00:00")
    assert (tmp_path / "attempt-1" / "result.json").exists()
    assert (tmp_path / "attempt-2" / "result.json").exists()
    assert json.loads((tmp_path / "attempt-1" / "result.json").read_text())["graph_status"] == (
        "completed"
    )


def test_manifest_confirms_q005_has_not_run_in_stage11d1():
    # The reliability phase migrates only the three named historical runs and
    # does not create a q005 live attempt.
    source = Path("scripts/migrate_deep_research_smoke_runs_v1.py").read_text(encoding="utf-8")
    assert "live-q005" not in source
