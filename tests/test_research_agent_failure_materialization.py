from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from paper_research.agents.research_agent.decision_provider import AgentDecisionProviderError
from paper_research.agents.research_agent.state import AgentState
from paper_research.main import create_app


class _Settings:
    app_profile = "production"
    llm_provider = "deepseek"
    parsed_papers_dir = Path("data/parsed")


class _CheckpointStore:
    def __init__(self) -> None:
        self.state: AgentState | None = None
        self.saved_phases: list[str] = []

    def save(self, state: AgentState, phase: str) -> str:
        checkpoint_id = f"{state.task_id}-{len(state.checkpoint_chain) + 1:04d}-{phase}"
        state.checkpoint_id = checkpoint_id
        state.checkpoint_chain.append(checkpoint_id)
        self.state = state.model_copy(deep=True)
        self.saved_phases.append(phase)
        return checkpoint_id

    def load(self, task_id: str) -> AgentState:
        if self.state is None or self.state.task_id != task_id:
            raise KeyError(task_id)
        return self.state.model_copy(deep=True)


class _TraceWriter:
    def __init__(self) -> None:
        self.phases: list[str] = []

    def append(self, state: AgentState, *, phase: str, **kwargs) -> str:
        event_id = f"{state.task_id}:{len(state.trace_event_ids) + 1:04d}:{phase}"
        state.trace_event_ids.append(event_id)
        self.phases.append(phase)
        return event_id


class _FailingRunner:
    checkpoint_store = _CheckpointStore()
    trace_writer = _TraceWriter()

    def __init__(self, *args, **kwargs) -> None:
        self.checkpoints = self.__class__.checkpoint_store
        self.trace = self.__class__.trace_writer

    def run(
        self,
        query: str,
        *,
        task_id: str,
        paper_ids=None,
        budget,
        interrupt_after_phase=None,
    ):
        del paper_ids
        state = AgentState(research_question=query, task_id=task_id, budget=budget)
        state.provider_call_count = 1
        state.token_usage.input_tokens = 491
        state.token_usage.output_tokens = 599
        state.token_usage.total_tokens = 1090
        state.token_usage.usage_source = "provider_reported"
        state.estimated_cost = 0.00023646
        state.refresh_remaining_budget()
        self.checkpoints.save(state, "POST_USAGE_ACCOUNTING")
        self.trace.append(state, phase="POST_USAGE_ACCOUNTING")
        raise AgentDecisionProviderError("invalid provider plan payload: KeyError")


def test_agent_decision_provider_error_returns_terminal_provider_failure(monkeypatch) -> None:
    from paper_research.api.routes import research

    _FailingRunner.checkpoint_store = _CheckpointStore()
    _FailingRunner.trace_writer = _TraceWriter()
    monkeypatch.setattr(research, "get_settings", lambda: _Settings())
    monkeypatch.setattr(research, "HybridLocalResearchProvider", lambda settings: object())
    monkeypatch.setattr(research, "_agent_decision_provider", lambda settings: object())
    monkeypatch.setattr(research, "ResearchAgentRunner", _FailingRunner)

    response = TestClient(create_app()).post(
        "/api/v1/research/agent",
        json={
            "task_id": "stage4-rt-v1-002-agent",
            "query": "Compare RAG review papers.",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "FAILED"
    assert payload["terminal"] is True
    assert payload["stop_reason"] == "PROVIDER_FAILURE"
    assert payload["failure_code"] == "AGENT_DECISION_PROVIDER_ERROR"
    assert payload["provider_call_count"] == 1
    assert payload["token_usage"]["total_tokens"] == 1090
    assert payload["estimated_cost"] == 0.00023646
    assert payload["checkpoint_count"] >= 2
    assert "PROVIDER_FAILURE" in _FailingRunner.checkpoint_store.saved_phases
    assert "PROVIDER_FAILURE" in _FailingRunner.trace_writer.phases
