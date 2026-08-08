from paper_research.agents.research_agent.models import EvidenceItem
from paper_research.agents.research_agent.state import AgentState


def test_evidence_state_deduplicates_stable_block_identity() -> None:
    state = AgentState(research_question="x")
    item = EvidenceItem(
        evidence_id="b1",
        paper_id="p1",
        block_id="b1",
        page=2,
        text_or_reference="text",
        discovered_by_tool="t1",
        discovered_at_step=1,
    )
    assert state.evidence_state.add(item, ["SQ1"]) is True
    assert state.evidence_state.add(item, ["SQ1"]) is False
    assert len(state.evidence_state.items) == 1

