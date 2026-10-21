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
        if state.multimodal_enabled and state.current_plan:
            target = state.subquestions[0]
            # Explicit opt-in supplies each native modality one bounded opportunity.
            for modality in ("FIGURE", "TABLE", "EQUATION"):
                if not any(h.get("action", {}).get("action") == "retrieve_multimodal_evidence"
                           and h.get("action", {}).get("arguments", {}).get("modality") == modality
                           for h in state.tool_history):
                    return ToolAction(
                        action="retrieve_multimodal_evidence",
                        arguments={"query": {
                            "FIGURE": "Architecture overview diagram. ",
                            "TABLE": "Experimental results table, models, metrics and conditions. ",
                            "EQUATION": "Mathematical objective equation and loss function. ",
                        }[modality] + state.research_question, "modality": modality,
                                   "arm": "M1" if modality == "FIGURE" else "N1", "top_k": 5},
                        target_subquestion_ids=[target.id], decision_reason="explicit_native_modality_probe",
                    )
            equations = [e for e in state.evidence_state.items.values()
                         if getattr(e, "modality", None) == "EQUATION"]
            if equations and not any(
                h.get("action", {}).get("decision_reason") == "native_formula_canonical_context"
                for h in state.tool_history
            ):
                return ToolAction(action="retrieve_evidence", arguments={
                    "query": max(equations, key=lambda e: len(e.readable_text)).readable_text
                        + " " + state.research_question, "top_k": 12},
                    target_subquestion_ids=[target.id],
                    decision_reason="native_formula_canonical_context")
            if not any(h.get("action", {}).get("decision_reason") == "hybrid_text_evidence"
                       for h in state.tool_history):
                # Query canonical context using text actually returned by native
                # tools, rather than assuming a multilingual question alone
                # matches equation symbols or table model-group names.
                native_terms = " ".join(
                    item.readable_text[:900]
                    for modality in ("EQUATION", "TABLE")
                    for item in sorted(
                        (e for e in state.evidence_state.items.values()
                         if getattr(e, "modality", None) == modality),
                        key=lambda e: -len(e.readable_text),
                    )[:2]
                )
                return ToolAction(action="retrieve_evidence", arguments={"query":
                    "Method objective loss formula; experimental results table and settings. "
                    + native_terms + " " + state.research_question, "top_k": 20},
                                  target_subquestion_ids=[target.id], decision_reason="hybrid_text_evidence")
        if state.verification_state and state.verification_state.status == VerificationStatus.PASS:
            return ToolAction(action="finish", decision_reason="verification_passed")
        scoped = [sq for sq in state.subquestions
                  if sq.id.startswith("paper:") and sq.id[6:] in (state.paper_ids or [])]
        if scoped and not state.retry_state.retryable:
            attempted = {target for observation in state.observations
                         if observation.tool == "retrieve_evidence"
                         and observation.status == ObservationStatus.SUCCESS
                         for target in observation.target_subquestions}
            for sq in scoped:
                if sq.id not in attempted:
                    return ToolAction(
                        action="retrieve_evidence", target_subquestion_ids=[sq.id],
                        arguments={"query": sq.question, "paper_ids": [sq.id[6:]], "top_k": 12},
                        decision_reason="scoped_paper_method_retrieval",
                    )
            return ToolAction(action="verify_evidence", decision_reason="scoped_papers_attempted")
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
