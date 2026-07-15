# ruff: noqa: E501
"""Narrow, semantic-free normalization for diagnostic response replay only."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from paper_research.generation.required_claim_output import validate_no_free_triples

NORMALIZATION_SCHEMA_VERSION = "required-claim-response-normalization-v1"
CITATION_PATTERN = re.compile(r"^E\d{3}$")


@dataclass(frozen=True)
class NormalizationResult:
    accepted: bool
    status: str
    payload: dict[str, Any] | None
    operations: tuple[str, ...]
    reason: str
    semantic_information_loss: bool
    risk_level: str = "none"

    @property
    def normalized_payload_hash(self) -> str | None:
        if self.payload is None:
            return None
        encoded = json.dumps(self.payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


def schema_family(raw: Any, question_id: str, expected_claim_ids: set[str]) -> str:
    if not isinstance(raw, dict):
        return "arbitrary_json"
    keys = set(raw)
    exact = {"question_id", "answerable", "required_claim_results", "refusal_reason", "prompt_version", "citation_protocol"}
    if keys == exact:
        return "exact_v3"
    if len(keys) == 1 and question_id in keys and isinstance(raw[question_id], dict):
        return "question_id_wrapper"
    if keys and keys <= expected_claim_ids:
        return "required_claim_id_map"
    if "claims" in keys and raw.get("answerable") is False:
        return "legacy_refusal"
    if "claims" in keys:
        return "legacy_v2_claims"
    return "arbitrary_json"


def _slot_from_map(claim_id: str, value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    allowed = {"status", "claim_text", "citation_ids", "citations", "omission_reason"}
    if not set(value) <= allowed:
        return None
    citations = value.get("citation_ids", value.get("citations", []))
    if not isinstance(citations, list) or not all(isinstance(item, str) and CITATION_PATTERN.fullmatch(item) for item in citations):
        return None
    required = {"status", "claim_text", "omission_reason"}
    if not required <= set(value):
        return None
    return {"required_claim_id": claim_id, "status": value["status"], "claim_text": value["claim_text"], "citation_ids": citations, "omission_reason": value["omission_reason"]}


def normalize_response(raw: Any, *, question_id: str, expected_claim_ids: list[str]) -> NormalizationResult:
    """Normalize structure only; never add envelope fields or infer claim identity."""
    try:
        validate_no_free_triples(raw)
    except Exception as exc:
        return NormalizationResult(False, "normalization_rejected", None, (), str(exc), False)
    if not isinstance(raw, dict):
        return NormalizationResult(False, "normalization_rejected", None, (), "top level is not an object", False)
    expected = set(expected_claim_ids)
    candidate = raw
    operations: list[str] = []
    if len(raw) == 1 and question_id in raw and isinstance(raw[question_id], dict):
        candidate = raw[question_id]
        operations.append("single_exact_question_wrapper_unwrap")
    candidate_keys = set(candidate) if isinstance(candidate, dict) else set()
    if candidate_keys == expected and expected:
        slots = []
        alias_used = False
        for claim_id in expected_claim_ids:
            value = candidate[claim_id]
            alias_used = alias_used or isinstance(value, dict) and "citations" in value
            slot = _slot_from_map(claim_id, value)
            if slot is None:
                return NormalizationResult(False, "normalization_rejected", None, tuple(operations), "claim map values lack an unambiguous complete slot shape", False)
            slots.append(slot)
        operations.append("required_claim_id_map_to_slots")
        if alias_used:
            operations.append("citations_alias_to_citation_ids")
        return NormalizationResult(False, "normalization_rejected", None, tuple(operations), "conversion would require adding missing v3 envelope fields", False)
    required_envelope = {"question_id", "answerable", "required_claim_results", "refusal_reason", "prompt_version", "citation_protocol"}
    if set(candidate) == required_envelope:
        if not operations:
            return NormalizationResult(False, "raw_schema_passed", candidate, (), "raw payload already has the complete envelope", False, "none")
        return NormalizationResult(True, "normalized_schema_passed", candidate, tuple(operations), "structure-only transformation produced a complete envelope", False, "low")
    family = schema_family(raw, question_id, expected)
    if family in {"legacy_v2_claims", "legacy_refusal"}:
        return NormalizationResult(False, "legacy_semantic_salvage_only", None, tuple(operations), "legacy claims cannot be assigned to required-claim slots without semantic inference or adding fields", True)
    return NormalizationResult(False, "normalization_rejected", None, tuple(operations), "missing or extra envelope fields cannot be repaired deterministically", False)
