"""Strict model-output schema and deterministic resolution for citation-id-v2."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from paper_research.generation.citation_registry import CitationRegistry
from paper_research.providers.llm import GeneratedCitation


class CitationIdClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: str = Field(min_length=1)
    claim_text: str = Field(min_length=1)
    citation_ids: list[str] = Field(min_length=1)


class CitationIdQA(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answerable: bool
    answer: str | None
    claims: list[CitationIdClaim]
    refusal_reason: str | None

    @model_validator(mode="after")
    def validate_shape(self) -> CitationIdQA:
        if self.answerable:
            if not self.answer or not self.answer.strip() or not self.claims:
                raise ValueError("answerable output requires answer and claims")
            if self.refusal_reason is not None:
                raise ValueError("answerable output cannot contain refusal_reason")
        else:
            if self.answer is not None or self.claims:
                raise ValueError("refusal requires answer=null and claims=[]")
            if not self.refusal_reason or not self.refusal_reason.strip():
                raise ValueError("refusal requires a non-empty refusal_reason")
        identifiers = [claim.claim_id for claim in self.claims]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("claim_id values must be unique")
        return self


def resolve_citation_id_answer(
    answer: CitationIdQA,
    registry: CitationRegistry,
) -> tuple[dict, list[str]]:
    output_claims = []
    citations = []
    duplicate_ids: list[str] = []
    for claim in answer.claims:
        resolution = registry.resolve(claim.citation_ids, claim_id=claim.claim_id)
        duplicate_ids.extend(resolution.duplicate_ids)
        claim_citations = [
            GeneratedCitation(
                paper_id=entry.paper_id,
                page=entry.page,
                block_id=entry.block_id,
            ).model_dump()
            for entry in resolution.entries
        ]
        citations.extend(
            {
                "citation_id": entry.citation_id,
                "paper_id": entry.paper_id,
                "page": entry.page,
                "block_id": entry.block_id,
            }
            for entry in resolution.entries
        )
        output_claims.append(
            {
                "claim_id": claim.claim_id,
                "claim_text": claim.claim_text,
                "evidence_complete": bool(resolution.entries),
                "assigned_evidence_ids": [
                    entry.evidence_id for entry in resolution.entries
                ],
                "citation_ids": [entry.citation_id for entry in resolution.entries],
                "citations": claim_citations,
            }
        )
    return (
        {
            "answerable": answer.answerable,
            "answer": answer.answer,
            "claims": output_claims,
            "citations": citations,
            "refusal_reason": answer.refusal_reason,
        },
        duplicate_ids,
    )
