from __future__ import annotations

from paper_research.agents.research_agent.models import VerificationResult, VerificationStatus
from paper_research.agents.research_agent.state import AgentState


class DeterministicResearchVerifier:
    def verify(self, state: AgentState) -> VerificationResult:
        unresolved = [
            subquestion.id
            for subquestion in state.subquestions
            if not state.evidence_state.evidence_for(subquestion.id)
        ]
        contradictions = list(state.evidence_state.contradictions)
        verified = [
            f"Evidence supports {subquestion_id}"
            for subquestion_id, evidence_ids in state.evidence_state.subquestion_evidence.items()
            if evidence_ids
        ]
        if contradictions:
            return VerificationResult(
                status=VerificationStatus.PARTIAL,
                verified_claims=verified,
                unresolved_subquestions=unresolved,
                contradictions=contradictions,
                recommended_next_action="REPLAN",
            )
        if not unresolved and verified:
            return VerificationResult(
                status=VerificationStatus.PASS,
                verified_claims=verified,
                recommended_next_action="FINISH",
            )
        return VerificationResult(
            status=VerificationStatus.PARTIAL if verified else VerificationStatus.FAIL,
            verified_claims=verified,
            unsupported_claims=[f"Missing evidence for {item}" for item in unresolved],
            unresolved_subquestions=unresolved,
            recommended_next_action="REPLAN",
        )

