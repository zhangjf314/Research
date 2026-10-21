"""Bounded, serializable evidence requirements and retrieval decisions."""

from enum import StrEnum
from hashlib import sha256

from pydantic import BaseModel, ConfigDict, Field, model_validator

from paper_research.agents.research_agent.models import ResearchPlan


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RequirementType(StrEnum):
    FACT = "FACT"
    DEFINITION = "DEFINITION"
    METHOD = "METHOD"
    RESULT = "RESULT"
    COMPARISON = "COMPARISON"
    LIMITATION = "LIMITATION"
    SOURCE = "SOURCE"


class CoverageStatus(StrEnum):
    UNSEARCHED = "UNSEARCHED"
    SEARCHED_NO_EVIDENCE = "SEARCHED_NO_EVIDENCE"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    SUPPORTED = "SUPPORTED"
    CONFLICTING = "CONFLICTING"


class EvidenceRequirement(StrictModel):
    requirement_id: str = Field(min_length=1, max_length=100)
    description: str = Field(min_length=1, max_length=600)
    requirement_type: RequirementType = RequirementType.FACT
    priority: int = Field(default=1, ge=1, le=8)
    required: bool = True
    source_scope: list[str] = Field(default_factory=list, max_length=38)
    min_sources: int = Field(default=1, ge=1, le=2)
    status: CoverageStatus = CoverageStatus.UNSEARCHED


class EvidenceContract(StrictModel):
    contract_id: str
    question: str = Field(min_length=1, max_length=2000)
    requirements: list[EvidenceRequirement] = Field(min_length=1, max_length=8)
    created_by: str
    contract_version: int = Field(default=1, ge=1)
    parent_contract_version: int | None = None
    change_reason: str | None = None
    added_requirements: list[str] = Field(default_factory=list)
    removed_requirements: list[str] = Field(default_factory=list)
    modified_requirements: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_ids(self):
        ids = [r.requirement_id for r in self.requirements]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate requirement IDs")
        if self.contract_version > 1 and (
            self.parent_contract_version != self.contract_version - 1 or not self.change_reason
        ):
            raise ValueError("contract revision requires previous version and reason")
        return self

    def revised(self, requirements, reason):
        old = {r.requirement_id: r for r in self.requirements}
        new = {r.requirement_id: r for r in requirements}
        return EvidenceContract(
            contract_id=self.contract_id,
            question=self.question,
            requirements=requirements,
            created_by=self.created_by,
            contract_version=self.contract_version + 1,
            parent_contract_version=self.contract_version,
            change_reason=reason,
            added_requirements=sorted(new.keys() - old.keys()),
            removed_requirements=sorted(old.keys() - new.keys()),
            modified_requirements=sorted(k for k in old.keys() & new.keys() if old[k] != new[k]),
        )


def stable_contract_id(question):
    return "EC-" + sha256(question.strip().encode("utf-8")).hexdigest()[:16]


class ContractPlan(StrictModel):
    plan: ResearchPlan
    contract: EvidenceContract


class RequirementCoverage(StrictModel):
    requirement_id: str
    status: CoverageStatus = CoverageStatus.UNSEARCHED
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0, ge=0, le=1)
    reason: str = Field(default="not searched", max_length=600)
    unresolvable: bool = False
    limitation: str | None = Field(default=None, max_length=600)


class Verdict(StrEnum):
    SUPPORTS = "SUPPORTS"
    PARTIAL = "PARTIAL"
    IRRELEVANT = "IRRELEVANT"
    CONTRADICTS = "CONTRADICTS"


class EvidenceAssessment(StrictModel):
    requirement_id: str
    evidence_id: str
    verdict: Verdict
    confidence: float = Field(ge=0, le=1)
    rationale: str = Field(min_length=1, max_length=600)


class AssessmentBatch(StrictModel):
    assessments: list[EvidenceAssessment] = Field(max_length=80)


class GapType(StrEnum):
    MISSING_EVIDENCE = "MISSING_EVIDENCE"
    PARTIAL_SUPPORT = "PARTIAL_SUPPORT"
    CONTRADICTORY_EVIDENCE = "CONTRADICTORY_EVIDENCE"
    INSUFFICIENT_SOURCE_DIVERSITY = "INSUFFICIENT_SOURCE_DIVERSITY"
    RETRIEVAL_FAILURE = "RETRIEVAL_FAILURE"


class EvidenceGap(StrictModel):
    gap_id: str
    requirement_id: str
    gap_type: GapType
    description: str = Field(max_length=600)
    evidence_ids: list[str] = Field(default_factory=list)
    severity: int = Field(default=1, ge=1, le=8)


class ActionType(StrEnum):
    INITIAL_SEARCH = "INITIAL_SEARCH"
    TARGETED_SEARCH = "TARGETED_SEARCH"
    REFORMULATE_QUERY = "REFORMULATE_QUERY"
    SEARCH_SUBQUESTION = "SEARCH_SUBQUESTION"
    STOP = "STOP"


class ReasonCode(StrEnum):
    INITIAL_REQUIREMENT = "INITIAL_REQUIREMENT"
    MISSING_EVIDENCE = "MISSING_EVIDENCE"
    PARTIAL_SUPPORT = "PARTIAL_SUPPORT"
    CONTRADICTION = "CONTRADICTION"
    LOW_INFORMATION_GAIN = "LOW_INFORMATION_GAIN"
    CONTRACT_SATISFIED = "CONTRACT_SATISFIED"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    MAX_REPLANS = "MAX_REPLANS"
    UNRESOLVABLE_GAP = "UNRESOLVABLE_GAP"
    INSUFFICIENT_SOURCE_DIVERSITY = "INSUFFICIENT_SOURCE_DIVERSITY"


class RetrievalAction(StrictModel):
    action_id: str
    action_type: ActionType
    query: str = Field(default="", max_length=800)
    requirement_ids: list[str] = Field(default_factory=list, max_length=8)
    reason_code: ReasonCode
    reason: str = Field(max_length=600)
    trigger_gap_ids: list[str] = Field(default_factory=list, max_length=8)
    parent_action_id: str | None = None


class InformationGain(StrictModel):
    new_evidence_count: int = 0
    newly_supported_requirements: list[str] = Field(default_factory=list)
    improved_requirements: list[str] = Field(default_factory=list)
    contradictions_added: list[str] = Field(default_factory=list)


class ReplanRecord(StrictModel):
    trigger_gap_ids: list[str]
    target_requirement_ids: list[str]
    previous_action: RetrievalAction
    new_action: RetrievalAction
    reason: str
    new_evidence_ids: list[str]
    improved_requirements: list[str]
    effective_replan: bool


class AgenticRetrievalConfig(StrictModel):
    agentic_retrieval_enabled: bool = False
    max_requirements: int = Field(default=8, ge=1, le=8)
    max_replans: int = Field(default=4, ge=0, le=12)
    no_information_gain_threshold: int = Field(default=2, ge=1, le=4)
    structured_max_attempts: int = Field(default=2, ge=1, le=2)
    max_assessment_evidence: int = Field(default=10, ge=5, le=10)
    evidence_chars: int = Field(default=2400, ge=400, le=4000)
    structured_max_output_tokens: int = Field(default=1600, ge=400, le=2400)
