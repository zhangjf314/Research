"""Offline schema-reliability candidate with a minimal model payload."""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from paper_research.generation.required_claim_output import (
    RequiredClaimStatus,
    RequiredClaimValidationError,
)

SCHEMA_RELIABILITY_CANDIDATE = "schema-reliability-v1-candidate"
DEV_V3_3_PROMPT_VERSION = "qa-required-claims-minimal-payload-v3.3"
MODEL_PAYLOAD_SCHEMA_VERSION = "minimal-required-claim-payload-v1"
LOCAL_ENVELOPE_SCHEMA_VERSION = "locally-bound-required-claim-envelope-v1"


class MinimalRequiredClaimResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    required_claim_id: str = Field(min_length=1)
    status: RequiredClaimStatus
    claim_text: str | None
    omission_reason: str | None


class MinimalRequiredClaimsPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answerable: bool
    required_claim_results: list[MinimalRequiredClaimResult]
    refusal_reason: str | None


class LocallyBoundRequiredClaimResult(MinimalRequiredClaimResult):
    citation_ids: list[str]


class LocallyBoundRequiredClaimsEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question_id: str
    answerable: bool
    required_claim_results: list[LocallyBoundRequiredClaimResult]
    refusal_reason: str | None
    prompt_version: Literal["schema-reliability-v1-candidate"]
    citation_protocol: Literal["citation-id-v2"]


class DevV33RequiredClaimsEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question_id: str
    answerable: bool
    required_claim_results: list[LocallyBoundRequiredClaimResult]
    refusal_reason: str | None
    prompt_version: Literal["qa-required-claims-minimal-payload-v3.3"]
    citation_protocol: Literal["citation-id-v2"]


def schema_reliability_system_prompt() -> str:
    return (
        "Return exactly one JSON object with only answerable, "
        "required_claim_results, and refusal_reason. Each required claim ID must appear "
        "exactly once. Each result contains only required_claim_id, status, claim_text, "
        "and omission_reason. Do not output citation IDs or any other identifier. "
        "answered requires claim_text and no omission_reason. unsupported and "
        "not_applicable require no claim_text and a non-empty omission_reason. "
        "For an unanswerable question return answerable=false, an empty result list, "
        "and a non-empty refusal_reason. No Markdown, prose wrapper, or JSON repair."
    )


def dev_v3_3_system_prompt() -> str:
    return (
        "Return exactly one JSON object and nothing else. The object contains only "
        "answerable, required_claim_results, and refusal_reason. Each supplied "
        "required_claim_id must appear exactly once. Each result contains exactly "
        "required_claim_id, status, claim_text, and omission_reason. status is answered, "
        "unsupported, or not_applicable. answered requires a supported non-empty "
        "claim_text and omission_reason=null. unsupported and not_applicable require "
        "claim_text=null and a specific non-empty omission_reason. Do not output "
        "question_id, protocol fields, citation IDs, evidence labels, policy traces, or "
        "any other identifiers. Use evidence text only to draft claims; do not copy its "
        "display label. For an unanswerable question return answerable=false, "
        "required_claim_results=[], and a non-empty refusal_reason. No Markdown, no "
        "surrounding prose, no repair instructions, and no policy explanation."
    )


def parse_minimal_payload(
    raw_content: str,
    *,
    expected_claim_ids: list[str],
) -> MinimalRequiredClaimsPayload:
    try:
        raw = json.loads(raw_content)
    except json.JSONDecodeError as exc:
        raise RequiredClaimValidationError("malformed_json", str(exc)) from exc
    try:
        output = MinimalRequiredClaimsPayload.model_validate(raw)
    except Exception as exc:
        raise RequiredClaimValidationError("schema_validation_failure", str(exc)) from exc
    actual = [row.required_claim_id for row in output.required_claim_results]
    if len(actual) != len(set(actual)):
        raise RequiredClaimValidationError("duplicate_required_claim_id", str(actual))
    missing = sorted(set(expected_claim_ids) - set(actual))
    extra = sorted(set(actual) - set(expected_claim_ids))
    if missing:
        raise RequiredClaimValidationError("missing_required_claim_id", str(missing))
    if extra:
        raise RequiredClaimValidationError("extra_required_claim_id", str(extra))
    if output.answerable:
        if output.refusal_reason is not None:
            raise RequiredClaimValidationError("answerable_has_refusal_reason", "")
    else:
        if expected_claim_ids or output.required_claim_results or not output.refusal_reason:
            raise RequiredClaimValidationError("answerability_protocol_failure", "")
    for row in output.required_claim_results:
        if row.status == RequiredClaimStatus.ANSWERED:
            if not row.claim_text or row.omission_reason is not None:
                raise RequiredClaimValidationError("answered_status_invalid", row.required_claim_id)
        elif row.claim_text is not None or not row.omission_reason:
            raise RequiredClaimValidationError("unsupported_status_invalid", row.required_claim_id)
    return output


def bind_local_envelope(
    payload: MinimalRequiredClaimsPayload,
    *,
    question_id: str,
    citation_ids_by_claim: dict[str, list[str]],
) -> LocallyBoundRequiredClaimsEnvelope:
    slots = []
    for row in payload.required_claim_results:
        citation_ids = (
            list(citation_ids_by_claim.get(row.required_claim_id, []))[:3]
            if row.status == RequiredClaimStatus.ANSWERED
            else []
        )
        if row.status == RequiredClaimStatus.ANSWERED and not citation_ids:
            raise RequiredClaimValidationError(
                "local_citation_selection_empty", row.required_claim_id
            )
        slots.append({**row.model_dump(mode="json"), "citation_ids": citation_ids})
    return LocallyBoundRequiredClaimsEnvelope.model_validate(
        {
            "question_id": question_id,
            "answerable": payload.answerable,
            "required_claim_results": slots,
            "refusal_reason": payload.refusal_reason,
            "prompt_version": SCHEMA_RELIABILITY_CANDIDATE,
            "citation_protocol": "citation-id-v2",
        }
    )


def bind_dev_v3_3_envelope(
    payload: MinimalRequiredClaimsPayload,
    *,
    question_id: str,
    citation_ids_by_claim: dict[str, list[str]],
) -> DevV33RequiredClaimsEnvelope:
    candidate = bind_local_envelope(
        payload,
        question_id=question_id,
        citation_ids_by_claim=citation_ids_by_claim,
    ).model_dump(mode="json")
    candidate["prompt_version"] = DEV_V3_3_PROMPT_VERSION
    return DevV33RequiredClaimsEnvelope.model_validate(candidate)
