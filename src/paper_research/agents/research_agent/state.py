from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, Field, SerializeAsAny

from paper_research.agents.research_agent.evidence_contract import (
    AgenticRetrievalConfig,
    EvidenceAssessment,
    EvidenceContract,
    EvidenceGap,
    InformationGain,
    ReplanRecord,
    RequirementCoverage,
    RetrievalAction,
)
from paper_research.agents.research_agent.models import (
    AgentStatus,
    CanonicalEvidenceItem,
    CanonicalToolObservation,
    EvidenceItem,
    NativeEvidenceItem,
    ResearchPlan,
    RetryState,
    StopReason,
    Subquestion,
    TokenUsage,
    ToolAction,
    ToolObservation,
    VerificationResult,
)
from paper_research.retrieval.graph import GraphAgentRecord


class AgentBudget(BaseModel):
    max_steps: int = Field(default=12, ge=1, le=100)
    max_tool_calls: int = Field(default=16, ge=0, le=200)
    max_provider_requests: int = Field(default=4, ge=0, le=50)
    max_tokens: int = Field(default=40000, ge=0)
    max_cost_usd: float = Field(default=0.05, ge=0)
    max_no_progress_actions: int = Field(default=2, ge=1, le=10)


class EvidenceState(BaseModel):
    items: dict[str, SerializeAsAny[NativeEvidenceItem | CanonicalEvidenceItem | EvidenceItem]] = Field(default_factory=dict)
    subquestion_evidence: dict[str, list[str]] = Field(default_factory=dict)
    evidence_gaps: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)

    def add(self, item: EvidenceItem, target_subquestions: list[str]) -> bool:
        if item.stable_key in self.items:
            return False
        self.items[item.stable_key] = item
        for subquestion_id in target_subquestions:
            linked = self.subquestion_evidence.setdefault(subquestion_id, [])
            if item.stable_key not in linked:
                linked.append(item.stable_key)
        return True

    def evidence_for(self, subquestion_id: str) -> list[EvidenceItem]:
        return [
            self.items[key]
            for key in self.subquestion_evidence.get(subquestion_id, [])
            if key in self.items
        ]

    def add_canonical_batch(
        self, items: list[CanonicalEvidenceItem], target_subquestions: list[str],
        known_subquestions: set[str],
    ) -> tuple[list[str], list[str], list[str]]:
        from paper_research.agents.research_agent.verifier import ResearchSafetyError

        if len(target_subquestions) != len(set(target_subquestions)) or any(
            sq not in known_subquestions for sq in target_subquestions
        ):
            raise ResearchSafetyError("ACTIVE_SUBQUESTION_INVALID")
        staged = self.model_copy(deep=True)
        added: list[str] = []
        duplicates: list[str] = []
        linked: list[str] = []
        for item in items:
            key = item.stable_key
            previous = staged.items.get(key)
            if previous is None:
                staged.items[key] = item.model_copy(deep=True)
                added.append(key)
            else:
                if not isinstance(previous, CanonicalEvidenceItem):
                    raise ResearchSafetyError("CANONICAL_PROVENANCE_INVALID")
                invariant = lambda value: (
                    value.provenance, value.retrieval_identity,
                    value.generation_binding_digest,
                    value.text_proof.canonical_chunk_sha256,
                    value.text_proof.canonical_text_sha256,
                    value.text_proof.canonical_source_sha256,
                    value.text_proof.source_artifact_sha256,
                )
                if invariant(previous) != invariant(item):
                    raise ResearchSafetyError("CANONICAL_PROVENANCE_CONFLICT")
                duplicates.append(key)
                observed = {occurrence.stable_key for occurrence in previous.occurrences}
                for occurrence in item.occurrences:
                    if occurrence.stable_key not in observed:
                        previous.occurrences.append(occurrence)
                        observed.add(occurrence.stable_key)
            for subquestion_id in target_subquestions:
                relation = staged.subquestion_evidence.setdefault(subquestion_id, [])
                if key not in relation:
                    relation.append(key)
                    linked.append(f"{subquestion_id}:{key}")
        self.items = staged.items
        self.subquestion_evidence = staged.subquestion_evidence
        return added, duplicates, linked


class AgentState(BaseModel):
    graph_rag: GraphAgentRecord | None = None
    multimodal_enabled: bool = Field(default=False, strict=True)
    task_id: str = Field(default_factory=lambda: f"agent-{uuid.uuid4().hex[:12]}")
    research_question: str
    paper_ids: list[str] | None = None
    current_plan: ResearchPlan | None = None
    plan_version: int = 0
    subquestions: list[Subquestion] = Field(default_factory=list)
    resolved_subquestions: list[str] = Field(default_factory=list)
    unresolved_subquestions: list[str] = Field(default_factory=list)
    evidence_state: EvidenceState = Field(default_factory=EvidenceState)
    observations: list[SerializeAsAny[CanonicalToolObservation | ToolObservation]] = Field(default_factory=list)
    tool_history: list[dict[str, Any]] = Field(default_factory=list)
    candidate_claims: list[str] = Field(default_factory=list)
    verified_claims: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    evidence_gaps: list[str] = Field(default_factory=list)
    step_count: int = 0
    tool_call_count: int = 0
    provider_call_count: int = 0
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    estimated_cost: float = 0.0
    budget: AgentBudget = Field(default_factory=AgentBudget)
    remaining_step_budget: int = 12
    remaining_tool_budget: int = 16
    remaining_token_budget: int = 40000
    remaining_cost_budget: float = 0.05
    retry_state: RetryState = Field(default_factory=RetryState)
    verification_state: VerificationResult | None = None
    last_action: ToolAction | None = None
    status: AgentStatus = AgentStatus.RUNNING
    stop_reason: StopReason | None = None
    checkpoint_id: str | None = None
    checkpoint_chain: list[str] = Field(default_factory=list)
    no_progress_count: int = 0
    resume_count: int = 0
    completed_tool_call_ids: list[str] = Field(default_factory=list)
    trace_event_ids: list[str] = Field(default_factory=list)
    agentic_config: AgenticRetrievalConfig | None = None
    retrieval_backend_id: str | None = None
    evidence_contract: EvidenceContract | None = None
    contract_history: list[EvidenceContract] = Field(default_factory=list)
    requirement_coverage: dict[str, RequirementCoverage] = Field(default_factory=dict)
    evidence_assessments: list[EvidenceAssessment] = Field(default_factory=list)
    contract_gaps: list[EvidenceGap] = Field(default_factory=list)
    retrieval_actions: list[RetrievalAction] = Field(default_factory=list)
    information_gain_history: list[InformationGain] = Field(default_factory=list)
    replan_history: list[ReplanRecord] = Field(default_factory=list)
    evidence_provenance: dict[str, dict[str, Any]] = Field(default_factory=dict)
    pending_retrieval_action: RetrievalAction | None = None
    pending_action_result: dict[str, Any] | None = None
    targeted_no_gain_count: int = 0
    retrieval_failures: dict[str, str] = Field(default_factory=dict)
    agentic_failure_reason: str | None = None

    def refresh_remaining_budget(self) -> None:
        self.remaining_step_budget = max(self.budget.max_steps - self.step_count, 0)
        self.remaining_tool_budget = max(self.budget.max_tool_calls - self.tool_call_count, 0)
        self.remaining_token_budget = max(
            self.budget.max_tokens - self.token_usage.total_tokens,
            0,
        )
        self.remaining_cost_budget = max(self.budget.max_cost_usd - self.estimated_cost, 0.0)
