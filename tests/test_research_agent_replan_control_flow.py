from __future__ import annotations

import json

from paper_research.agents.research_agent.models import (
    AgentStatus,
    VerificationResult,
    VerificationStatus,
)
from paper_research.agents.research_agent.state import AgentState
from tests.research_agent_helpers import MockEvidenceProvider, evidence, runner, small_budget


def _trace_events(tmp_path, task_id: str) -> list[dict]:
    path = tmp_path / "traces" / f"{task_id}.jsonl"
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_verifier_partial_reaches_replan_when_budget_available(tmp_path) -> None:
    provider = MockEvidenceProvider([[evidence("a")], [], [evidence("missing")]])
    agent = runner(tmp_path, provider)

    state = agent.run("alpha and beta", task_id="partial-replan")

    assert state.plan_version >= 2
    replan = next(item for item in state.tool_history if item.get("phase") == "REPLAN")
    assert replan["trigger"] == "VERIFICATION_PARTIAL_OR_FAILED"
    assert replan["effective_replan"] is True
    assert replan["plan_delta"]["added_subquestions"]


def test_replan_changes_plan_hash_and_next_decision_source(tmp_path) -> None:
    provider = MockEvidenceProvider([[evidence("a")], [], [evidence("missing")]])
    agent = runner(tmp_path, provider)

    state = agent.run("alpha and beta", task_id="replan-delta")
    events = _trace_events(tmp_path, state.task_id)
    replan = next(item for item in state.tool_history if item.get("phase") == "REPLAN")

    assert replan["old_plan_hash"] != replan["new_plan_hash"]
    assert replan["new_plan_version"] == replan["old_plan_version"] + 1
    replan_index = next(index for index, event in enumerate(events) if event["phase"] == "REPLAN")
    later_decisions = [
        event for event in events[replan_index + 1 :] if event["phase"] == "DECIDE"
    ]
    assert later_decisions
    assert later_decisions[0]["trigger_source"] == "REPLAN"


def test_partial_verification_with_budget_cannot_complete(tmp_path) -> None:
    provider = MockEvidenceProvider([])
    agent = runner(tmp_path, provider)
    state = AgentState(research_question="x", budget=small_budget(max_steps=1))
    state.task_id = "partial-no-budget"
    state.step_count = 1
    state.verification_state = VerificationResult(
        status=VerificationStatus.PARTIAL,
        unresolved_subquestions=["SQ1"],
        evidence_gaps=["missing evidence for SQ1"],
        recommended_next_action="REPLAN",
    )
    state.refresh_remaining_budget()

    terminal = agent._finish(state, None)

    assert terminal is True
    assert state.status == AgentStatus.PARTIAL
    assert state.stop_reason == "VERIFICATION_FAILED_NO_BUDGET"


def test_finish_runs_verify_before_completed_trace(tmp_path) -> None:
    provider = MockEvidenceProvider([[evidence("a")], [evidence("b")]])
    agent = runner(tmp_path, provider)

    state = agent.run("alpha and beta", task_id="verify-before-finish")
    events = _trace_events(tmp_path, state.task_id)
    verify_index = next(index for index, event in enumerate(events) if event["phase"] == "VERIFY")
    finish_index = next(index for index, event in enumerate(events) if event["phase"] == "FINISH")

    assert state.status == AgentStatus.COMPLETED
    assert verify_index < finish_index
    assert events[verify_index]["verification"]["status"] == "PASS"


def test_trace_causality_observation_verify_replan_decision_execution(tmp_path) -> None:
    provider = MockEvidenceProvider([[evidence("a")], [], [evidence("missing")]])
    agent = runner(tmp_path, provider)

    state = agent.run("alpha and beta", task_id="trace-causality")
    events = _trace_events(tmp_path, state.task_id)
    verify_index = next(
        index
        for index, event in enumerate(events)
        if event["phase"] == "VERIFY" and event["verification"]["status"] in {"PARTIAL", "FAIL"}
    )
    replan_index = next(index for index, event in enumerate(events) if event["phase"] == "REPLAN")
    decide_index = next(
        index
        for index, event in enumerate(events[replan_index + 1 :], start=replan_index + 1)
        if event["phase"] == "DECIDE"
    )
    observe_index = next(
        index
        for index, event in enumerate(events[decide_index + 1 :], start=decide_index + 1)
        if event["phase"] == "OBSERVE"
    )

    assert verify_index < replan_index < decide_index < observe_index
    assert events[verify_index]["based_on_observation_id"]
    assert events[replan_index]["replan"]["effective_replan"] is True
    assert events[decide_index]["trigger_source"] == "REPLAN"


def test_resume_preserves_replan_eligibility_without_duplicate_calls(tmp_path) -> None:
    provider = MockEvidenceProvider([[evidence("a")], [], [evidence("missing")]])
    agent = runner(tmp_path, provider)
    paused = agent.run(
        "alpha and beta",
        task_id="resume-replan-eligible",
        interrupt_after_phase="VERIFY",
    )
    calls_before = len(provider.calls)
    tokens_before = paused.token_usage.total_tokens

    resumed = agent.resume(paused.task_id)

    assert resumed.resume_count == 1
    assert len(provider.calls) >= calls_before
    assert resumed.token_usage.total_tokens >= tokens_before
    assert resumed.plan_version >= 2
    assert len(resumed.completed_tool_call_ids) == len(set(resumed.completed_tool_call_ids))
