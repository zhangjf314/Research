from paper_research.agents.research_agent.models import VerificationStatus
from paper_research.agents.research_agent.planner import RuleBasedResearchPlanner
from paper_research.agents.research_agent.state import AgentState
from paper_research.agents.research_agent.verifier import DeterministicResearchVerifier


def test_verifier_requires_evidence_for_each_subquestion() -> None:
    state = AgentState(research_question="alpha and beta")
    plan = RuleBasedResearchPlanner().initial_plan(state.research_question)
    state.subquestions = plan.subquestions
    result = DeterministicResearchVerifier().verify(state)
    assert result.status == VerificationStatus.FAIL
    assert result.unresolved_subquestions == ["SQ1", "SQ2"]

