"""Offline schema-reliability candidate with a minimal model payload."""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from paper_research.generation.required_claim_output import (
    RequiredClaimStatus,
    RequiredClaimValidationError,
)

SCHEMA_RELIABILITY_CANDIDATE = "schema-reliability-v1-candidate"
DEV_V3_3_PROMPT_VERSION = "qa-required-claims-minimal-payload-v3.3"
SCHEMA_RELIABILITY_V2_CANDIDATE = "schema-reliability-v2-candidate"
DEV_V3_4_CANDIDATE_PROMPT_VERSION = "qa-required-claims-minimal-payload-v3.4-candidate"
REFUSAL_CANONICALIZATION_VERSION = "refusal-empty-to-null-canonicalization-v1"
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


class RequiredClaimLocalEnvelopeV2(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question_id: str
    answerable: bool
    required_claim_results: list[MinimalRequiredClaimResult]
    refusal_reason: str | None
    prompt_version: Literal["qa-required-claims-minimal-payload-v3.4-candidate"]
    citation_protocol: Literal["citation-id-v2"]


class RequiredClaimFinalResultV2(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question_id: str
    answerable: bool
    required_claim_results: list[LocallyBoundRequiredClaimResult]
    refusal_reason: str | None
    prompt_version: Literal["qa-required-claims-minimal-payload-v3.4-candidate"]
    citation_protocol: Literal["citation-id-v2"]


class ModelPayloadCanonicalizationV2(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw_payload: dict
    canonical_payload: MinimalRequiredClaimsPayload
    raw_payload_hash: str
    canonical_payload_hash: str
    canonicalization_applied: bool
    canonicalization_rule: str
    changed_paths: list[str]
    semantic_change: Literal[False]
    validation_before: str
    validation_after: str


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


def dev_v3_4_candidate_system_prompt() -> str:
    return (
        "Return exactly one JSON object containing only answerable, "
        "required_claim_results, and refusal_reason. Each supplied required_claim_id "
        "appears exactly once. Each result contains exactly required_claim_id, status, "
        "claim_text, and omission_reason. For answerable=true, refusal_reason should be "
        "null; an implementation that cannot represent absent text may use the exact "
        'empty string "". Do not use whitespace, N/A, none, not applicable, or '
        "explanatory refusal text for an answerable response. For answerable=false, "
        "required_claim_results must be empty and refusal_reason must be a specific "
        "non-empty string. Do not output citation IDs, protocol constants, internal IDs, "
        "or extra fields. No Markdown, surrounding prose, repair, or policy explanation."
    )


def _payload_hash(value: dict) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def canonicalize_model_payload_v2(
    raw_content: str,
    *,
    expected_claim_ids: list[str],
) -> ModelPayloadCanonicalizationV2:
    """Strictly validate, then canonicalize only answerable exact-empty refusal."""
    try:
        raw = json.loads(raw_content)
    except json.JSONDecodeError as exc:
        raise RequiredClaimValidationError("malformed_json", str(exc)) from exc
    if not isinstance(raw, dict):
        raise RequiredClaimValidationError("schema_validation_failure", "top-level object required")
    try:
        structural = MinimalRequiredClaimsPayload.model_validate(raw)
    except Exception as exc:
        raise RequiredClaimValidationError("schema_validation_failure", str(exc)) from exc
    actual = [row.required_claim_id for row in structural.required_claim_results]
    if len(actual) != len(set(actual)):
        raise RequiredClaimValidationError("duplicate_required_claim_id", str(actual))
    missing = sorted(set(expected_claim_ids) - set(actual))
    extra = sorted(set(actual) - set(expected_claim_ids))
    if missing:
        raise RequiredClaimValidationError("missing_required_claim_id", str(missing))
    if extra:
        raise RequiredClaimValidationError("extra_required_claim_id", str(extra))
    for row in structural.required_claim_results:
        if row.status == RequiredClaimStatus.ANSWERED:
            if not row.claim_text or row.omission_reason is not None:
                raise RequiredClaimValidationError("answered_status_invalid", row.required_claim_id)
        elif row.claim_text is not None or not row.omission_reason:
            raise RequiredClaimValidationError("unsupported_status_invalid", row.required_claim_id)
    raw_body = structural.model_dump(mode="json")
    if structural.answerable:
        if structural.refusal_reason not in {None, ""}:
            raise RequiredClaimValidationError(
                "answerable_has_semantic_refusal_reason",
                repr(structural.refusal_reason),
            )
        canonical_body = {
            **raw_body,
            "refusal_reason": None,
        }
        applied = structural.refusal_reason == ""
    else:
        reason = structural.refusal_reason
        if not isinstance(reason, str) or not reason.strip():
            raise RequiredClaimValidationError("unanswerable_missing_refusal_reason", repr(reason))
        if structural.required_claim_results:
            raise RequiredClaimValidationError("unanswerable_has_claim_slots", str(actual))
        canonical_body = raw_body
        applied = False
    canonical = MinimalRequiredClaimsPayload.model_validate(canonical_body)
    return ModelPayloadCanonicalizationV2(
        raw_payload=raw,
        canonical_payload=canonical,
        raw_payload_hash=_payload_hash(raw),
        canonical_payload_hash=_payload_hash(canonical.model_dump(mode="json")),
        canonicalization_applied=applied,
        canonicalization_rule=REFUSAL_CANONICALIZATION_VERSION,
        changed_paths=["$.refusal_reason"] if applied else [],
        semantic_change=False,
        validation_before="structural_and_slot_valid",
        validation_after="canonical_payload_valid",
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


def bind_local_envelope_v2(
    payload: MinimalRequiredClaimsPayload,
    *,
    question_id: str,
) -> RequiredClaimLocalEnvelopeV2:
    return RequiredClaimLocalEnvelopeV2.model_validate(
        {
            "question_id": question_id,
            **payload.model_dump(mode="json"),
            "prompt_version": DEV_V3_4_CANDIDATE_PROMPT_VERSION,
            "citation_protocol": "citation-id-v2",
        }
    )
