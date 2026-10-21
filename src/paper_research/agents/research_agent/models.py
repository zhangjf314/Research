from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator
import hashlib
import json


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
    CONTRACT_SATISFIED = "CONTRACT_SATISFIED"
    NO_INFORMATION_GAIN = "NO_INFORMATION_GAIN"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    MAX_REPLANS = "MAX_REPLANS"
    UNRESOLVABLE_GAP = "UNRESOLVABLE_GAP"
    PLANNER_SCHEMA_FAILURE = "PLANNER_SCHEMA_FAILURE"
    CONTRACT_CREATION_FAILURE = "CONTRACT_CREATION_FAILURE"
    RETRIEVAL_FAILURE = "RETRIEVAL_FAILURE"
    VERIFIER_FAILURE = "VERIFIER_FAILURE"
    POLICY_FAILURE = "POLICY_FAILURE"
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
    chunk_id: str | None = None
    block_ids: list[str] = Field(default_factory=list)
    block_page_map: dict[str, int] = Field(default_factory=dict)
    page_end: int | None = None

    @property
    def stable_key(self) -> str:
        return f"{self.paper_id}:{self.block_id}:{self.page}"


class CanonicalRetrievalOccurrence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task_id: str = Field(strict=True, min_length=1)
    tool_call_id: str
    trace_id: str
    original_context_rank: int = Field(strict=True, ge=1)
    step: int = Field(strict=True, ge=0)
    query: str
    paper_scope: list[str]
    target_subquestion_ids: list[str]
    retrieval_score: float = Field(strict=True, allow_inf_nan=False)
    retrieval_sources: list[str]
    observed_text: str = Field(strict=True, min_length=1)
    display_text_sha256: str
    context_text_sha256: str
    context_text_length: int = Field(strict=True, ge=1)
    generation_binding_digest: str

    @property
    def stable_key(self) -> tuple[str, str, str, int]:
        return self.task_id, self.tool_call_id, self.trace_id, self.original_context_rank


class CanonicalTextProof(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = "active-text-proof-v1"
    canonical_chunk_sha256: str
    canonical_text_sha256: str
    canonical_text_length: int = Field(ge=1)
    context_text_sha256: str
    context_text_length: int = Field(ge=1)
    display_text_sha256: str
    display_text_length: int = Field(ge=1, le=800)
    prefix_offset: int = 0
    truncation_limit: int = 800
    canonical_source_sha256: str
    source_artifact_sha256: dict[str, str]


class CanonicalEvidenceItem(EvidenceItem):
    model_config = ConfigDict(extra="forbid")
    provenance: dict[str, Any]
    retrieval_identity: dict[str, Any]
    text_proof: CanonicalTextProof
    occurrences: list[CanonicalRetrievalOccurrence] = Field(min_length=1)
    generation_binding_digest: str

    @property
    def stable_key(self) -> str:
        return "canonical-v1:" + json.dumps(
            [self.paper_id, self.chunk_id], ensure_ascii=True, separators=(",", ":")
        )

    @model_validator(mode="after")
    def validate_canonical_aliases(self) -> "CanonicalEvidenceItem":
        from paper_research.generation.evidence_contract_adapter import (
            EvidenceProvenance, RetrievalIdentity,
        )
        from paper_research.agents.research_agent.verifier import ResearchSafetyError

        try:
            source = EvidenceProvenance.model_validate(self.provenance)
            identity = RetrievalIdentity.model_validate(self.retrieval_identity)
            primary = source.block_ids[0]
            valid = (
                self.paper_id == source.paper_id and self.chunk_id == source.chunk_id
                and self.evidence_id == f"{self.paper_id}:{self.chunk_id}"
                and self.block_id == primary and self.page == source.block_page_map[primary]
                and self.page_end == source.page_end and self.block_ids == source.block_ids
                and self.block_page_map == source.block_page_map
                and self.section == " > ".join(source.section_path)
                and self.verification_status == EvidenceVerificationStatus.CANDIDATE
                and not self.supports and not self.contradicts
                and self.text_proof.schema_version == "active-text-proof-v1"
                and self.text_proof.prefix_offset == 0
                and self.text_proof.truncation_limit == 800
                and len(self.text_or_reference) == self.text_proof.display_text_length
                and hashlib.sha256(self.text_or_reference.encode("utf8")).hexdigest()
                    == self.text_proof.display_text_sha256
                and all(o.generation_binding_digest == self.generation_binding_digest
                        and o.context_text_length <= self.text_proof.canonical_text_length
                        and len(o.observed_text) == min(800, o.context_text_length)
                        and hashlib.sha256(o.observed_text.encode("utf8")).hexdigest()
                            == o.display_text_sha256
                        and len(o.target_subquestion_ids) == len(set(o.target_subquestion_ids))
                        and len(o.retrieval_sources) == len(set(o.retrieval_sources))
                        and set(o.retrieval_sources) <= {"dense", "sparse"}
                        for o in self.occurrences)
                and self.occurrences[0].observed_text == self.text_or_reference
                and self.occurrences[0].context_text_sha256 == self.text_proof.context_text_sha256
                and self.occurrences[0].context_text_length == self.text_proof.context_text_length
                and len({o.stable_key for o in self.occurrences}) == len(self.occurrences)
                and bool(identity.collection)
            )
            if not valid:
                raise ValueError("canonical aliases disagree")
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ResearchSafetyError("CANONICAL_PROVENANCE_INVALID") from exc
        return self


class CanonicalRetrievalBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    items: list[CanonicalEvidenceItem]
    errors: list[dict[str, Any]] = Field(default_factory=list)
    gaps: list[dict[str, Any]] = Field(default_factory=list)
    coverage: list[dict[str, Any]] = Field(default_factory=list)
    trace_id: str
    generation_binding: dict[str, Any]
    generation_binding_digest: str


class NativeRetrievalOccurrence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task_id: str = Field(strict=True, min_length=1)
    tool_call_id: str = Field(strict=True, min_length=1)
    trace_id: str = Field(strict=True, min_length=1)
    step: int = Field(strict=True, ge=0)
    query: str = Field(strict=True, min_length=1)
    paper_scope: list[str]
    target_subquestion_ids: list[str]
    generation_binding_digest: str
    arm: str
    channels: dict[str, list[str]]
    rank: int = Field(strict=True, ge=1)
    native_profile: str = "agent-sparse-adapter-v1"
    native_index_sha256: str | None = None


class NativeEvidenceItem(EvidenceItem):
    """PDF-native candidate; never a semantic verification receipt."""

    model_config = ConfigDict(extra="forbid")
    unit_id: str
    modality: str
    source_pdf: str
    pdf_sha256: str
    native_paper_id: str
    native_unit: dict[str, Any]
    native_unit_sha256: str
    readable_kind: str
    readable_text: str
    bundle_sha256: str
    retrieval_identity: dict[str, Any]
    generation_binding_digest: str
    occurrences: list[NativeRetrievalOccurrence] = Field(min_length=1)

    @property
    def stable_key(self) -> str:
        return f"native-v1:{self.paper_id}:{self.unit_id}"

    @model_validator(mode="after")
    def validate_native_aliases(self):
        from paper_research.pdf_multimodal.identity import digest
        kinds = {"FIGURE_CANDIDATE": ("FIGURE", "caption_only"),
                 "TABLE_CANDIDATE": ("TABLE", "table_structure"),
                 "EQUATION_CANDIDATE": ("EQUATION", "native_equation_candidate_text")}
        u = self.native_unit
        if (kinds.get(u.get("unit_type")) != (self.modality, self.readable_kind)
                or u.get("unit_id") != self.unit_id or self.block_id != self.unit_id
                or self.evidence_id != f"{self.paper_id}:{self.unit_id}"
                or u.get("paper_id") != self.native_paper_id
                or u.get("pdf_sha256") != self.pdf_sha256
                or u.get("page_number") != self.page or self.page_end != self.page
                or digest(u) != self.native_unit_sha256
                or not self.readable_text or self.text_or_reference != self.readable_text[:800]
                or any(o.arm not in {"N1", "M1"}
                       or o.native_profile not in {"agent-sparse-adapter-v1", "phase4-full-v1"}
                       or set(o.channels) != (({"dense", "sparse"} if o.native_profile == "phase4-full-v1"
                                               else {"sparse"}) | ({"image"} if o.arm == "M1" else set()))
                       or any(len(ids) != len(set(ids)) for ids in o.channels.values())
                       or len(o.target_subquestion_ids) != len(set(o.target_subquestion_ids))
                       or o.generation_binding_digest != self.generation_binding_digest
                       for o in self.occurrences)
                or self.verification_status != EvidenceVerificationStatus.CANDIDATE
                or self.supports or self.contradicts):
            raise ValueError("NATIVE_PROVENANCE_INVALID")
        return self


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


class CanonicalToolObservation(ToolObservation):
    model_config = ConfigDict(extra="forbid")
    evidence_linked: list[str] = Field(default_factory=list)
    structural_errors: list[dict[str, Any]] = Field(default_factory=list)
    structural_gaps: list[dict[str, Any]] = Field(default_factory=list)
    structural_coverage: list[dict[str, Any]] = Field(default_factory=list)
    retrieval_trace_id: str
    structural_provenance_status: str = "STRUCTURAL_PROVENANCE_VALID"
    semantic_status: str = "SEMANTIC_SUPPORT_NOT_ASSESSED"


class VerificationResult(BaseModel):
    status: VerificationStatus
    verified_claims: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    unresolved_subquestions: list[str] = Field(default_factory=list)
    evidence_gaps: list[str] = Field(default_factory=list)
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
