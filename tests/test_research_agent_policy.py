from paper_research.agents.research_agent.models import VerificationResult, VerificationStatus
from paper_research.agents.research_agent.planner import RuleBasedResearchPlanner
from paper_research.agents.research_agent.policy import ResearchAgentPolicy
from paper_research.agents.research_agent.state import AgentState


def test_policy_finishes_after_verification_pass() -> None:
    state = AgentState(research_question="x")
    state.verification_state = VerificationResult(
        status=VerificationStatus.PASS,
        recommended_next_action="FINISH",
    )
    action = ResearchAgentPolicy().decide(state)
    assert action.action == "finish"


def test_policy_retrieves_for_missing_evidence() -> None:
    state = AgentState(research_question="x")
    plan = RuleBasedResearchPlanner().initial_plan("alpha and beta")
    state.current_plan = plan
    state.subquestions = plan.subquestions
    action = ResearchAgentPolicy().decide(state)
    assert action.action == "retrieve_evidence"
    assert action.target_subquestion_ids == ["SQ1"]

