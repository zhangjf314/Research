"""Compatibility adapters applied after durable provider-response persistence."""

from __future__ import annotations

from typing import Any


class ClaimTextAdapterError(ValueError):
    pass


def normalized_claim_text(claim: dict[str, Any]) -> str:
    """Accept legacy text or v2 claim_text without silently resolving conflicts."""
    legacy = claim.get("text")
    current = claim.get("claim_text")
    if legacy is None and current is None:
        raise ClaimTextAdapterError("claim requires text or claim_text")
    if legacy is not None and current is not None and legacy != current:
        raise ClaimTextAdapterError("text and claim_text conflict")
    value = current if current is not None else legacy
    if not isinstance(value, str) or not value.strip():
        raise ClaimTextAdapterError("claim text must be a non-empty string")
    return value


def adapt_generated_claim(claim: dict[str, Any]) -> dict[str, Any]:
    output = dict(claim)
    output["claim_text"] = normalized_claim_text(claim)
    output.pop("text", None)
    return output
