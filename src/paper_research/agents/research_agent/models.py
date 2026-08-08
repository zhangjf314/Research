from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class SubquestionStatus(StrEnum):
    OPEN = "OPEN"
    RESOLVED = "RESOLVED"
    UNSUPPORTED = "UNSUPPORTED"


class EvidenceVerificationStatus(StrEnum):
    CANDIDATE = "CANDIDATE"
    VERIFIED = "VERIFIED"
    CONTRADICTED = "CONTRADICTED"


class ObservationStatus(StrEnum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    RETRIED = "RETRIED"


class VerificationStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    PARTIAL = "PARTIAL"


class AgentStatus(StrEnum):
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"


class StopReason(StrEnum):
    SUCCESS = "SUCCESS"
    EVIDENCE_SUFFICIENT = "EVIDENCE_SUFFICIENT"
    MAX_STEPS_REACHED = "MAX_STEPS_REACHED"
    TOOL_BUDGET_EXHAUSTED = "TOOL_BUDGET_EXHAUSTED"
    TOKEN_BUDGET_EXHAUSTED = "TOKEN_BUDGET_EXHAUSTED"
    COST_BUDGET_EXHAUSTED = "COST_BUDGET_EXHAUSTED"
    PROVIDER_FAILURE = "PROVIDER_FAILURE"
    TOOL_FAILURE = "TOOL_FAILURE"
    VERIFICATION_FAILED_NO_BUDGET = "VERIFICATION_FAILED_NO_BUDGET"
    NO_PROGRESS = "NO_PROGRESS"
    CHECKPOINT_FAILURE = "CHECKPOINT_FAILURE"


class Subquestion(BaseModel):
    id: str
    question: str
    status: SubquestionStatus = SubquestionStatus.OPEN
    priority: int = 1


class ResearchPlan(BaseModel):
    objective: str
    subquestions: list[Subquestion]
    completion_criteria: list[str]


class EvidenceItem(BaseModel):
    evidence_id: str
    paper_id: str
    block_id: str
    page: int
    section: str = ""
    text_or_reference: str
    source_type: str = "local_retrieval"
    supports: list[str] = Field(default_factory=list)
    contradicts: list[str] = Field(default_factory=list)
    discovered_by_tool: str
    discovered_at_step: int
    relevance: float | None = None
    verification_status: EvidenceVerificationStatus = EvidenceVerificationStatus.CANDIDATE

    @property
    def stable_key(self) -> str:
        return f"{self.paper_id}:{self.block_id}:{self.page}"


class ToolAction(BaseModel):
    action: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    target_subquestion_ids: list[str] = Field(default_factory=list)
    decision_reason: str


class ToolObservation(BaseModel):
    tool_call_id: str
    tool: str
    status: ObservationStatus
    target_subquestions: list[str] = Field(default_factory=list)
    evidence_added: list[str] = Field(default_factory=list)
    evidence_duplicates: list[str] = Field(default_factory=list)
    new_information: bool = False
    possible_gaps: list[str] = Field(default_factory=list)
    error: str | None = None
    retryable: bool = False


class VerificationResult(BaseModel):
    status: VerificationStatus
    verified_claims: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    unresolved_subquestions: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    recommended_next_action: str = "REPLAN"


class RetryState(BaseModel):
    provider_retry_count: int = 0
    tool_retry_count: int = 0
    last_error: str | None = None
    retryable: bool = False


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    usage_source: str = "not_used"

