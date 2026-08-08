from paper_research.agents.research_agent.models import AgentStatus, StopReason
from tests.research_agent_helpers import MockEvidenceProvider, runner, small_budget


def test_tool_budget_exhaustion_stops_agent(tmp_path) -> None:
    state = runner(tmp_path, MockEvidenceProvider([[]])).run(
        "alpha and beta",
        budget=small_budget(max_tool_calls=1, max_steps=4),
    )
    assert state.status in {AgentStatus.PARTIAL, AgentStatus.FAILED}
    assert state.stop_reason in {
        StopReason.TOOL_BUDGET_EXHAUSTED,
        StopReason.NO_PROGRESS,
        StopReason.VERIFICATION_FAILED_NO_BUDGET,
    }

