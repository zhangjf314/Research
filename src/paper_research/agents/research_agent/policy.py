from __future__ import annotations

from paper_research.agents.research_agent.models import (
    ObservationStatus,
    StopReason,
    ToolAction,
    VerificationStatus,
)
from paper_research.agents.research_agent.state import AgentState


class ResearchAgentPolicy:
    def decide(self, state: AgentState) -> ToolAction:
        if state.verification_state and state.verification_state.status == VerificationStatus.PASS:
            return ToolAction(action="finish", decision_reason="verification_passed")
        if state.tool_history and state.tool_history[-1].get("phase") == "REPLAN":
            target = state.unresolved_subquestions[:1]
            if target:
                return ToolAction(
                    action="retrieve_evidence",
                    arguments={"query": _query_for_target(state, target[0]), "top_k": 5},
                    target_subquestion_ids=target,
                    decision_reason="replan_selected_followup_retrieval",
                )
        if (
            state.observations
            and state.observations[-1].tool == "retrieve_evidence"
            and not state.observations[-1].new_information
            and state.unresolved_subquestions
        ):
            return ToolAction(
                action="verify_evidence",
                decision_reason="retrieve_returned_no_new_information",
            )
        if state.retry_state.retryable and state.retry_state.tool_retry_count <= 1:
            previous = state.last_action
            if previous is not None:
                return previous.model_copy(
                    update={"decision_reason": "retry_previous_retryable_tool_failure"}
                )
        if state.evidence_state.contradictions:
            return ToolAction(
                action="inspect_evidence",
                target_subquestion_ids=state.unresolved_subquestions[:1],
                decision_reason="contradiction_requires_inspection",
            )
        if state.no_progress_count >= state.budget.max_no_progress_actions:
            return ToolAction(action="finish", decision_reason=StopReason.NO_PROGRESS.value)
        if state.verification_state and state.verification_state.status in {
            VerificationStatus.FAIL,
            VerificationStatus.PARTIAL,
        }:
            target = state.verification_state.unresolved_subquestions[:1]
            if target and state.remaining_tool_budget > 0:
                return ToolAction(
                    action="retrieve_evidence",
                    arguments={"query": _query_for_target(state, target[0]), "top_k": 5},
                    target_subquestion_ids=target,
                    decision_reason="verification_found_unresolved_subquestion",
                )
        for subquestion in state.subquestions:
            if not state.evidence_state.evidence_for(subquestion.id):
                return ToolAction(
                    action="retrieve_evidence",
                    arguments={"query": subquestion.question, "top_k": 5},
                    target_subquestion_ids=[subquestion.id],
                    decision_reason=f"insufficient_evidence_for_{subquestion.id}",
                )
        return ToolAction(action="verify_evidence", decision_reason="candidate_evidence_available")

    def observe_progress(self, state: AgentState) -> None:
        if not state.observations:
            return
        observation = state.observations[-1]
        if observation.status == ObservationStatus.SUCCESS and observation.new_information:
            state.no_progress_count = 0
        elif observation.tool != "finish":
            state.no_progress_count += 1


def _query_for_target(state: AgentState, subquestion_id: str) -> str:
    for subquestion in state.subquestions:
        if subquestion.id == subquestion_id:
            return subquestion.question
    return state.research_question
