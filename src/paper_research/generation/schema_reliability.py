"""Offline schema-reliability candidate with a minimal model payload."""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator

from paper_research.generation.required_claim_output import (
    RequiredClaimStatus,
    RequiredClaimValidationError,
)

SCHEMA_RELIABILITY_CANDIDATE = "schema-reliability-v1-candidate"
DEV_V3_3_PROMPT_VERSION = "qa-required-claims-minimal-payload-v3.3"
SCHEMA_RELIABILITY_V2_CANDIDATE = "schema-reliability-v2-candidate"
DEV_V3_4_CANDIDATE_PROMPT_VERSION = "qa-required-claims-minimal-payload-v3.4-candidate"
DEV_V3_4_PROMPT_VERSION = "qa-required-claims-minimal-payload-v3.4"
REFUSAL_CANONICALIZATION_VERSION = "refusal-empty-to-null-canonicalization-v1"
MODEL_PAYLOAD_SCHEMA_VERSION = "minimal-required-claim-payload-v1"
LOCAL_ENVELOPE_SCHEMA_VERSION = "locally-bound-required-claim-envelope-v1"
SCHEMA_RELIABILITY_V3_CANDIDATE = "schema-reliability-v3-candidate"
MODEL_PAYLOAD_V3_VERSION = "required-claim-model-payload-v3"
LOCAL_ENVELOPE_V3_VERSION = "required-claim-local-envelope-v3"
DEV_V3_5_CANDIDATE_PROMPT_VERSION = "qa-required-claims-content-payload-v3.5-candidate"
SLOT_STATUS_DERIVATION_VERSION = "derive-slot-status-v1"


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


class DevV34LocalEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question_id: str
    answerable: bool
    required_claim_results: list[MinimalRequiredClaimResult]
    refusal_reason: str | None
    prompt_version: Literal["qa-required-claims-minimal-payload-v3.4"]
    citation_protocol: Literal["citation-id-v2"]


class DevV34FinalResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question_id: str
    answerable: bool
    required_claim_results: list[LocallyBoundRequiredClaimResult]
    refusal_reason: str | None
    prompt_version: Literal["qa-required-claims-minimal-payload-v3.4"]
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


class AnsweredContentSlotV3(BaseModel):
    model_config = ConfigDict(extra="forbid")

    required_claim_id: str = Field(min_length=1)
    claim_text: str
    omission_reason: Literal[None]

    @field_validator("claim_text")
    @classmethod
    def validate_claim_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("claim_text must be non-empty after trimming")
        return value


class UnsupportedContentSlotV3(BaseModel):
    model_config = ConfigDict(extra="forbid")

    required_claim_id: str = Field(min_length=1)
    claim_text: Literal[None]
    omission_reason: str

    @field_validator("omission_reason")
    @classmethod
    def validate_omission_reason(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("omission_reason must be non-empty after trimming")
        return value


ContentSlotV3 = AnsweredContentSlotV3 | UnsupportedContentSlotV3


class AnswerablePayloadV3(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answerable: Literal[True]
    required_claim_results: list[ContentSlotV3] = Field(min_length=1)


class UnanswerablePayloadV3(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answerable: Literal[False]
    required_claim_results: list[ContentSlotV3] = Field(max_length=0)
    refusal_reason: str

    @field_validator("refusal_reason")
    @classmethod
    def validate_refusal_reason(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("refusal_reason must be non-empty after trimming")
        return value


DiscriminatedPayloadV3 = Annotated[
    AnswerablePayloadV3 | UnanswerablePayloadV3,
    Field(discriminator="answerable"),
]
PAYLOAD_V3_ADAPTER = TypeAdapter(DiscriminatedPayloadV3)


class SlotStatusDerivationV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_payload_hash: str
    required_claim_id: str
    derived_status: Literal["answered", "unsupported"]
    derivation_rule: Literal[
        "nonempty_claim_and_null_omission",
        "null_claim_and_nonempty_omission",
    ]
    changed_semantic_fields: Literal[0]
    added_local_metadata_only: Literal[True]
    validation_before: Literal["content_slot_v3_valid"]
    validation_after: Literal["local_status_derived"]


class DerivedContentSlotV3(BaseModel):
    model_config = ConfigDict(extra="forbid")

    required_claim_id: str
    status: Literal["answered", "unsupported"]
    claim_text: str | None
    omission_reason: str | None
    citation_ids: list[str]
    policy_trace_reference: str | None


class LocalEnvelopeV3(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question_id: str
    answerable: bool
    required_claim_results: list[DerivedContentSlotV3]
    refusal_reason: str | None
    prompt_version: Literal["qa-required-claims-content-payload-v3.5-candidate"]
    citation_protocol: Literal["citation-id-v2"]
    payload_schema_version: Literal["required-claim-model-payload-v3"]
    status_derivation_version: Literal["derive-slot-status-v1"]


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


def dev_v3_4_system_prompt() -> str:
    return dev_v3_4_candidate_system_prompt()


def dev_v3_5_candidate_system_prompt() -> str:
    return (
        "Return exactly one JSON object and no Markdown or surrounding prose. "
        "Use exactly one of two top-level shapes. When the question can be addressed, "
        'return {"answerable":true,"required_claim_results":[{"required_claim_id":'
        '"RC1","claim_text":"A claim fully grounded in the supplied evidence.",'
        '"omission_reason":null}]}. Include every supplied required_claim_id exactly '
        "once and do not include refusal_reason. When the question cannot be addressed, "
        'return {"answerable":false,"required_claim_results":[],"refusal_reason":'
        '"The supplied evidence does not address the question."}. For an individual '
        "claim with insufficient evidence, use claim_text=null and a specific non-empty "
        "omission_reason. Do not output identifiers other than the supplied "
        "required_claim_id, and do not output protocol fields, policy traces, or extra "
        "fields."
    )


def _payload_hash(value: dict) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def derive_slot_status_v1(slot: dict) -> SlotStatusDerivationV1:
    """Derive local status from the unique valid content shape without repair."""
    try:
        validated = TypeAdapter(ContentSlotV3).validate_python(slot)
    except Exception as exc:
        raise RequiredClaimValidationError("slot_shape_failed", str(exc)) from exc
    if isinstance(validated, AnsweredContentSlotV3):
        status = "answered"
        rule = "nonempty_claim_and_null_omission"
    else:
        status = "unsupported"
        rule = "null_claim_and_nonempty_omission"
    return SlotStatusDerivationV1(
        source_payload_hash=_payload_hash(slot),
        required_claim_id=validated.required_claim_id,
        derived_status=status,
        derivation_rule=rule,
        changed_semantic_fields=0,
        added_local_metadata_only=True,
        validation_before="content_slot_v3_valid",
        validation_after="local_status_derived",
    )


def validate_payload_v3(
    raw_content: str,
    *,
    expected_claim_ids: list[str],
) -> AnswerablePayloadV3 | UnanswerablePayloadV3:
    try:
        raw = json.loads(raw_content)
    except json.JSONDecodeError as exc:
        raise RequiredClaimValidationError("malformed_json", str(exc)) from exc
    try:
        payload = PAYLOAD_V3_ADAPTER.validate_python(raw)
    except Exception as exc:
        raise RequiredClaimValidationError("branch_schema_failed", str(exc)) from exc
    actual = [row.required_claim_id for row in payload.required_claim_results]
    if len(actual) != len(set(actual)):
        raise RequiredClaimValidationError("duplicate_required_claim_id", str(actual))
    missing = sorted(set(expected_claim_ids) - set(actual))
    extra = sorted(set(actual) - set(expected_claim_ids))
    if missing:
        raise RequiredClaimValidationError("missing_required_claim_id", str(missing))
    if extra:
        raise RequiredClaimValidationError("extra_required_claim_id", str(extra))
    if expected_claim_ids and not payload.answerable:
        raise RequiredClaimValidationError(
            "answerability_protocol_failure",
            "answerable manifest item used unanswerable branch",
        )
    if not expected_claim_ids and payload.answerable:
        raise RequiredClaimValidationError(
            "answerability_protocol_failure",
            "unanswerable manifest item used answerable branch",
        )
    return payload


def bind_local_envelope_v3(
    payload: AnswerablePayloadV3 | UnanswerablePayloadV3,
    *,
    question_id: str,
    citation_ids_by_claim: dict[str, list[str]] | None = None,
    policy_trace_reference: str | None = None,
) -> LocalEnvelopeV3:
    citations = citation_ids_by_claim or {}
    slots = []
    for row in payload.required_claim_results:
        source = row.model_dump(mode="json")
        derivation = derive_slot_status_v1(source)
        slots.append(
            {
                **source,
                "status": derivation.derived_status,
                "citation_ids": list(citations.get(row.required_claim_id, []))
                if derivation.derived_status == "answered"
                else [],
                "policy_trace_reference": policy_trace_reference,
            }
        )
    return LocalEnvelopeV3.model_validate(
        {
            "question_id": question_id,
            "answerable": payload.answerable,
            "required_claim_results": slots,
            "refusal_reason": (
                payload.refusal_reason
                if isinstance(payload, UnanswerablePayloadV3)
                else None
            ),
            "prompt_version": DEV_V3_5_CANDIDATE_PROMPT_VERSION,
            "citation_protocol": "citation-id-v2",
            "payload_schema_version": MODEL_PAYLOAD_V3_VERSION,
            "status_derivation_version": SLOT_STATUS_DERIVATION_VERSION,
        }
    )


def payload_v3_as_minimal_payload(
    payload: AnswerablePayloadV3 | UnanswerablePayloadV3,
) -> MinimalRequiredClaimsPayload:
    """Bind derived status for reuse by the frozen local citation policy."""
    slots = []
    for row in payload.required_claim_results:
        source = row.model_dump(mode="json")
        derivation = derive_slot_status_v1(source)
        slots.append({**source, "status": derivation.derived_status})
    return MinimalRequiredClaimsPayload.model_validate(
        {
            "answerable": payload.answerable,
            "required_claim_results": slots,
            "refusal_reason": (
                payload.refusal_reason
                if isinstance(payload, UnanswerablePayloadV3)
                else None
            ),
        }
    )


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


def bind_dev_v3_4_envelope(
    payload: MinimalRequiredClaimsPayload,
    *,
    question_id: str,
) -> DevV34LocalEnvelope:
    return DevV34LocalEnvelope.model_validate(
        {
            "question_id": question_id,
            **payload.model_dump(mode="json"),
            "prompt_version": DEV_V3_4_PROMPT_VERSION,
            "citation_protocol": "citation-id-v2",
        }
    )
