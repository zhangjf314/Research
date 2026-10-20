"""Opt-in V1 serialization; the frozen semantic verifier/controller remain unchanged."""

from pydantic import Field, StrictInt, StrictStr

from paper_research.agents.research_agent.contract_verifier import (
    AgenticFailure,
    StructuredContractAdapter,
)
from paper_research.agents.research_agent.evidence_contract import (
    EvidenceAssessment,
    StrictModel,
    Verdict,
)

RATIONALE_NOT_REQUESTED = "V1_COMPACT: rationale not requested (reduced observability)"


class CompactAssessment(StrictModel):
    r: StrictStr = Field(description="Copy the original input requirement_id exactly.")
    e: StrictInt = Field(description="Zero-based row index in the original input evidence array.")
    v: Verdict
    c: float = Field(ge=0, le=1, description="Assessment confidence, unchanged from V0.")


class CompactAssessmentBatch(StrictModel):
    assessments: list[CompactAssessment] = Field(
        max_length=80,
        description="Exactly one row for EVERY requirement/evidence pair; no pairs may be omitted.",
    )


def assessment_input(state, requirement_ids, evidence_ids):
    """Same input contents and ordering as frozen StructuredContractAdapter.assess."""
    requirements = [r.model_dump(mode="json") for r in state.evidence_contract.requirements
                    if r.requirement_id in requirement_ids]
    by_id = {e.evidence_id: e for e in state.evidence_state.items.values()}
    evidence = [
        {"evidence_id": eid, "paper_id": by_id[eid].paper_id, "page": by_id[eid].page,
         "block_ids": by_id[eid].block_ids, "chunk_id": by_id[eid].chunk_id,
         "text": by_id[eid].text_or_reference[:state.agentic_config.evidence_chars]}
        for eid in evidence_ids
    ]
    return {"question": state.research_question, "requirements": requirements,
            "evidence": evidence,
            "instruction": "Return exactly one assessment per requirement/evidence pair."}


def decode_compact(batch, evidence_ids):
    batch = CompactAssessmentBatch.model_validate(batch)
    decoded = []
    for row in batch.assessments:
        if not 0 <= row.e < len(evidence_ids):
            raise AgenticFailure("VERIFIER_FAILURE", "unknown compact evidence ordinal")
        decoded.append(EvidenceAssessment(
            requirement_id=row.r, evidence_id=evidence_ids[row.e], verdict=row.v,
            confidence=row.c, rationale=RATIONALE_NOT_REQUESTED))
    # The unchanged controller validates full IDs, complete pair sets and scope before mutation.
    return decoded


class CompactContractAdapter(StructuredContractAdapter):
    def assess(self, state, requirement_ids, evidence_ids, runner):
        batch = self._request(
            state, runner, CompactAssessmentBatch,
            assessment_input(state, requirement_ids, evidence_ids), "VERIFIER_FAILURE")
        return decode_compact(batch, evidence_ids)


def build_contract_verifier(provider, *, verifier_protocol="v0"):
    """Default and all old callers remain V0; V1 requires explicit opt-in."""
    if verifier_protocol == "v0":
        return StructuredContractAdapter(provider)
    if verifier_protocol == "v1_compact":
        return CompactContractAdapter(provider)
    raise ValueError("unknown verifier_protocol")
