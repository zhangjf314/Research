from paper_research.agents.research_agent.models import ToolAction
from paper_research.agents.research_agent.state import AgentState
from paper_research.agents.research_agent.tools import ResearchAgentToolRegistry
from tests.research_agent_helpers import MockEvidenceProvider, evidence


def test_retrieve_tool_wraps_provider_and_preserves_metadata() -> None:
    state = AgentState(research_question="x")
    action = ToolAction(
        action="retrieve_evidence",
        arguments={"query": "q", "top_k": 30},
        target_subquestion_ids=["SQ1"],
        decision_reason="test",
    )
    obs = ResearchAgentToolRegistry(MockEvidenceProvider([[evidence("b1", page=3)]])).execute(
        state, action, "tc1"
    )
    assert obs.new_information is True
    item = next(iter(state.evidence_state.items.values()))
    assert item.paper_id == "p1"
    assert item.block_id == "b1"
    assert item.page == 3

