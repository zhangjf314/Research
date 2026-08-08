from paper_research.agents.research_agent.models import AgentStatus
from tests.research_agent_helpers import MockEvidenceProvider, evidence, runner


def test_resume_continues_without_reexecuting_completed_tool_calls(tmp_path) -> None:
    provider = MockEvidenceProvider([[evidence("a")], [evidence("b")]])
    agent = runner(tmp_path, provider)
    paused = agent.run(
        "alpha and beta",
        task_id="resume-case",
        interrupt_after_phase="STATE_UPDATED",
    )
    calls_before = len(provider.calls)
    assert paused.status == AgentStatus.PAUSED
    resumed = agent.resume("resume-case")
    assert resumed.resume_count == 1
    assert len(provider.calls) >= calls_before
    assert len(resumed.completed_tool_call_ids) == len(set(resumed.completed_tool_call_ids))
    assert resumed.status == AgentStatus.COMPLETED
