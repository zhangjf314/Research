from paper_research.agents.research_agent.models import AgentStatus
from tests.research_agent_helpers import MockEvidenceProvider, evidence, runner


def test_agentic_paths_differ_for_complete_and_missing_observations(tmp_path) -> None:
    complete = runner(
        tmp_path / "complete",
        MockEvidenceProvider([[evidence("a")], [evidence("b")]]),
    ).run("alpha and beta")
    missing = runner(
        tmp_path / "missing",
        MockEvidenceProvider([[evidence("a")], [], [evidence("b")]]),
    ).run("alpha and beta")
    assert complete.status == AgentStatus.COMPLETED
    assert missing.plan_version > complete.plan_version
    assert len(missing.tool_history) >= 1


def test_failed_tool_is_retried_once(tmp_path) -> None:
    provider = MockEvidenceProvider([[evidence("a")], [evidence("b")]], fail_first=True)
    state = runner(tmp_path / "retry", provider).run("alpha and beta")
    assert len(provider.calls) >= 3
    assert state.retry_state.tool_retry_count == 1
