"""Strict required-claim-slot protocol for qa-required-claims-citation-id-v3."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from paper_research.generation.citation_registry import CitationRegistry


class RequiredClaimValidationError(ValueError):
    """Auditable validation failure with a stable machine-readable code."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


class RequiredClaimStatus(StrEnum):
    ANSWERED = "answered"
    UNSUPPORTED = "unsupported"
    NOT_APPLICABLE = "not_applicable"


class AllocatedEvidenceSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    citation_ids: list[str]
    summary: str


class RequiredClaimInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    required_claim_id: str = Field(min_length=1)
    required_claim_text: str = Field(min_length=1)
    evidence_complete: bool
    allowed_citation_ids: list[str]
    allocated_evidence: list[AllocatedEvidenceSummary]
    omission_policy: str = Field(min_length=1)


class RequiredClaimResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    required_claim_id: str = Field(min_length=1)
    status: RequiredClaimStatus
    claim_text: str | None
    citation_ids: list[str]
    omission_reason: str | None


class RequiredClaimsQAResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question_id: str = Field(min_length=1)
    answerable: bool
    required_claim_results: list[RequiredClaimResult]
    refusal_reason: str | None
    prompt_version: Literal["qa-required-claims-citation-id-v3"]
    citation_protocol: Literal["citation-id-v2"]


class RequiredClaimsQAResponseV31(BaseModel):
    """Dev v3.1 transport-constrained response; business slot semantics are unchanged."""

    model_config = ConfigDict(extra="forbid")

    question_id: str = Field(min_length=1)
    answerable: bool
    required_claim_results: list[RequiredClaimResult]
    refusal_reason: str | None
    prompt_version: Literal["qa-required-claims-citation-id-v3.1"]
    citation_protocol: Literal["citation-id-v2"]


class RequiredClaimsQAResponseV32(BaseModel):
    """Dev v3.2 candidate response; top-level and slot cardinality stay at v1.1."""

    model_config = ConfigDict(extra="forbid")

    question_id: str = Field(min_length=1)
    answerable: bool
    required_claim_results: list[RequiredClaimResult]
    refusal_reason: str | None
    prompt_version: Literal["qa-required-claims-citation-id-v3.2-candidate"]
    citation_protocol: Literal["citation-id-v2"]


# Backward-compatible names for Phase A fixtures; new code uses the explicit names.
RequiredClaimSlot = RequiredClaimResult
RequiredClaimsQA = RequiredClaimsQAResponse


class OutputBudget(BaseModel):
    required_claim_count: int = Field(ge=0)
    calculated_max_output_tokens: int = Field(ge=1, le=4096)
    capped: bool
    budget_formula_version: str = "required-claim-output-budget-v1"


def validate_status_citation_consistency(result: RequiredClaimResult) -> None:
    if result.status == RequiredClaimStatus.ANSWERED:
        if not result.claim_text or not result.citation_ids:
            raise RequiredClaimValidationError(
                "answered_missing_content_or_citation",
                f"{result.required_claim_id} requires claim_text and citation_ids",
            )
        if result.omission_reason is not None:
            raise RequiredClaimValidationError(
                "answered_has_omission_reason",
                result.required_claim_id,
            )
    else:
        if result.citation_ids:
            raise RequiredClaimValidationError(
                "unsupported_or_na_has_citation",
                result.required_claim_id,
            )
        if not result.omission_reason:
            raise RequiredClaimValidationError(
                "missing_omission_reason",
                result.required_claim_id,
            )


def validate_no_free_triples(raw: Any) -> None:
    if isinstance(raw, dict):
        forbidden = {"paper_id", "page", "block_id"} & set(raw)
        if forbidden:
            raise RequiredClaimValidationError(
                "free_triple_forbidden", f"forbidden fields: {sorted(forbidden)}"
            )
        for value in raw.values():
            validate_no_free_triples(value)
    elif isinstance(raw, list):
        for value in raw:
            validate_no_free_triples(value)


def validate_answerability_protocol(
    output: RequiredClaimsQAResponse,
    expected_claim_ids: list[str],
) -> None:
    if output.answerable:
        if output.refusal_reason is not None:
            raise RequiredClaimValidationError(
                "answerable_has_refusal_reason", output.question_id
            )
    else:
        if not output.refusal_reason:
            raise RequiredClaimValidationError(
                "unanswerable_missing_refusal_reason", output.question_id
            )
        if any(
            result.status != RequiredClaimStatus.NOT_APPLICABLE
            or result.citation_ids
            for result in output.required_claim_results
        ):
            raise RequiredClaimValidationError(
                "unanswerable_has_answer_or_citation", output.question_id
            )
    actual = [result.required_claim_id for result in output.required_claim_results]
    if len(actual) != len(set(actual)):
        raise RequiredClaimValidationError("duplicate_required_claim_id", str(actual))
    missing = sorted(set(expected_claim_ids) - set(actual))
    extra = sorted(set(actual) - set(expected_claim_ids))
    if missing:
        raise RequiredClaimValidationError("missing_required_claim_id", str(missing))
    if extra:
        raise RequiredClaimValidationError("extra_required_claim_id", str(extra))


def validate_claim_local_citations(
    result: RequiredClaimResult,
    registry: CitationRegistry,
    allowed_by_claim: dict[str, set[str]],
) -> None:
    if result.status != RequiredClaimStatus.ANSWERED:
        return
    registry_ids = {entry.citation_id for entry in registry.entries}
    for citation_id in result.citation_ids:
        if citation_id not in registry_ids:
            raise RequiredClaimValidationError(
                "unknown_citation_id", citation_id
            )
        if citation_id not in allowed_by_claim.get(result.required_claim_id, set()):
            raise RequiredClaimValidationError(
                "cross_claim_citation",
                f"{citation_id} is not allocated to {result.required_claim_id}",
            )


def validate_registry_hash(registry: CitationRegistry, expected_hash: str) -> None:
    if registry.registry_hash != expected_hash:
        raise RequiredClaimValidationError(
            "registry_hash_mismatch",
            f"expected {expected_hash}, got {registry.registry_hash}",
        )


def validate_required_claim_slots(
    output: RequiredClaimsQAResponse,
    expected_claim_ids: list[str],
    registry: CitationRegistry,
    allowed_by_claim: dict[str, set[str]] | None = None,
) -> None:
    validate_answerability_protocol(output, expected_claim_ids)
    allocations = allowed_by_claim or {
        claim_id: {
            entry.citation_id
            for entry in registry.entries
            if not entry.claim_ids or claim_id in entry.claim_ids
        }
        for claim_id in expected_claim_ids
    }
    for result in output.required_claim_results:
        validate_status_citation_consistency(result)
        validate_claim_local_citations(result, registry, allocations)


def parse_and_validate_required_claim_response(
    raw_content: str,
    *,
    expected_claim_ids: list[str],
    registry: CitationRegistry,
    allowed_by_claim: dict[str, set[str]],
    expected_registry_hash: str,
) -> RequiredClaimsQAResponse:
    import json

    try:
        raw = json.loads(raw_content)
    except json.JSONDecodeError as exc:
        raise RequiredClaimValidationError("malformed_json", str(exc)) from exc
    validate_no_free_triples(raw)
    try:
        output = RequiredClaimsQAResponse.model_validate(raw)
    except Exception as exc:
        raise RequiredClaimValidationError("schema_validation_failure", str(exc)) from exc
    validate_registry_hash(registry, expected_registry_hash)
    validate_required_claim_slots(output, expected_claim_ids, registry, allowed_by_claim)
    return output


def parse_and_validate_required_claim_response_v31(
    raw_content: str,
    *,
    expected_question_id: str,
    expected_claim_ids: list[str],
    registry: CitationRegistry,
    allowed_by_claim: dict[str, set[str]],
    expected_registry_hash: str,
) -> RequiredClaimsQAResponseV31:
    """Validate raw Dev v3.1 output without normalization or correction."""
    import json

    try:
        raw = json.loads(raw_content)
    except json.JSONDecodeError as exc:
        raise RequiredClaimValidationError("malformed_json", str(exc)) from exc
    if not isinstance(raw, dict):
        raise RequiredClaimValidationError("valid_json_wrong_schema", "top level is not an object")
    if set(raw) == {expected_question_id} and isinstance(raw[expected_question_id], dict):
        raise RequiredClaimValidationError("question_wrapper_rejected", expected_question_id)
    if "claims" in raw:
        raise RequiredClaimValidationError("legacy_schema_rejected", "legacy claims field")
    if "required_claim_results" not in raw and set(raw) & set(expected_claim_ids):
        raise RequiredClaimValidationError("claim_map_rejected", "claim IDs used as top-level keys")
    try:
        validate_no_free_triples(raw)
    except RequiredClaimValidationError as exc:
        raise RequiredClaimValidationError("valid_json_wrong_schema", str(exc)) from exc
    try:
        output = RequiredClaimsQAResponseV31.model_validate(raw)
    except Exception as exc:
        raise RequiredClaimValidationError("valid_json_wrong_schema", str(exc)) from exc
    if output.question_id != expected_question_id:
        raise RequiredClaimValidationError(
            "answerability_protocol_failure",
            f"expected question_id {expected_question_id}, got {output.question_id}",
        )
    validate_registry_hash(registry, expected_registry_hash)
    try:
        validate_required_claim_slots(output, expected_claim_ids, registry, allowed_by_claim)
    except RequiredClaimValidationError as exc:
        mapping = {
            "missing_required_claim_id": "missing_slot",
            "duplicate_required_claim_id": "duplicate_slot",
            "extra_required_claim_id": "extra_slot",
            "answered_missing_content_or_citation": "status_citation_inconsistency",
            "answered_has_omission_reason": "status_citation_inconsistency",
            "unsupported_or_na_has_citation": "status_citation_inconsistency",
            "missing_omission_reason": "status_citation_inconsistency",
            "unanswerable_has_answer_or_citation": "answerability_protocol_failure",
            "unanswerable_missing_refusal_reason": "answerability_protocol_failure",
            "answerable_has_refusal_reason": "answerability_protocol_failure",
        }
        mapped = mapping.get(exc.code)
        if mapped:
            raise RequiredClaimValidationError(mapped, str(exc)) from exc
        raise
    return output


def parse_and_validate_required_claim_response_v32(
    raw_content: str,
    *,
    expected_question_id: str,
    expected_claim_ids: list[str],
    registry: CitationRegistry,
    allowed_by_claim: dict[str, set[str]],
    expected_registry_hash: str,
) -> RequiredClaimsQAResponseV32:
    """Validate raw/final Dev v3.2 output without normalization or correction."""
    import json

    try:
        raw = json.loads(raw_content)
    except json.JSONDecodeError as exc:
        raise RequiredClaimValidationError("malformed_json", str(exc)) from exc
    if not isinstance(raw, dict):
        raise RequiredClaimValidationError("valid_json_wrong_schema", "top level is not an object")
    if set(raw) == {expected_question_id} and isinstance(raw[expected_question_id], dict):
        raise RequiredClaimValidationError("question_wrapper_rejected", expected_question_id)
    if "claims" in raw:
        raise RequiredClaimValidationError("legacy_schema_rejected", "legacy claims field")
    if "required_claim_results" not in raw and set(raw) & set(expected_claim_ids):
        raise RequiredClaimValidationError("claim_map_rejected", "claim IDs used as top-level keys")
    try:
        validate_no_free_triples(raw)
        output = RequiredClaimsQAResponseV32.model_validate(raw)
    except RequiredClaimValidationError:
        raise
    except Exception as exc:
        raise RequiredClaimValidationError("valid_json_wrong_schema", str(exc)) from exc
    if output.question_id != expected_question_id:
        raise RequiredClaimValidationError(
            "answerability_protocol_failure",
            f"expected question_id {expected_question_id}, got {output.question_id}",
        )
    validate_registry_hash(registry, expected_registry_hash)
    validate_required_claim_slots(output, expected_claim_ids, registry, allowed_by_claim)
    if any(len(slot.citation_ids) > 3 for slot in output.required_claim_results):
        raise RequiredClaimValidationError("citation_cap_exceeded", expected_question_id)
    return output


def required_claim_output_token_budget(
    required_claim_count: int,
    *,
    base_output_tokens: int = 256,
    per_claim_output_tokens: int = 192,
) -> OutputBudget:
    if required_claim_count < 0:
        raise ValueError("required_claim_count cannot be negative")
    uncapped = base_output_tokens + per_claim_output_tokens * required_claim_count
    return OutputBudget(
        required_claim_count=required_claim_count,
        calculated_max_output_tokens=min(4096, uncapped),
        capped=uncapped > 4096,
    )
