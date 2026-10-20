from __future__ import annotations

import re

from paper_research.agents.research_agent.models import VerificationResult, VerificationStatus
from paper_research.agents.research_agent.state import AgentState

SEMANTIC_SUPPORT_NOT_ASSESSED = "SEMANTIC_SUPPORT_NOT_ASSESSED"
REPORT_SAFETY_NAMESPACE = "p1_agent_report_safety_v1"
REPORT_SAFETY_RECEIPT = {
    "mode": "CANDIDATE_ONLY_REPORT_SUPPRESSED",
    "schema_version": 1,
    "policy_id": "paperresearch-agent-report-safety-v1",
    "semantic_status": SEMANTIC_SUPPORT_NOT_ASSESSED,
    "source_identity_schema": "legacy-opaque-v0",
    "canonical_state_injection": False,
    "semantic_qualification_id": None,
    "report_eligibility": "DENIED",
}


class ResearchSafetyError(ValueError):
    def __init__(self, code: str, status_code: int = 503) -> None:
        self.code = code
        self.status_code = status_code
        super().__init__(code)


def validate_research_task_id(task_id: str) -> None:
    if not isinstance(task_id, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", task_id
    ):
        raise ResearchSafetyError("INVALID_RESEARCH_TASK_ID", 422)


class DeterministicResearchVerifier:
    def verify(self, state: AgentState) -> VerificationResult:
        # Retrieval presence is neither a positive nor a negative semantic judgment.
        return VerificationResult(
            status=VerificationStatus.PARTIAL,
            unresolved_subquestions=[item.id for item in state.subquestions],
            evidence_gaps=[SEMANTIC_SUPPORT_NOT_ASSESSED],
            recommended_next_action="FINISH",
        )
