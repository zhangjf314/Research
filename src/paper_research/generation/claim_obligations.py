"""Canonical claim obligation sets for offline evidence selection candidates."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from paper_research.generation.citation_selection import (
    ClaimObligation,
    analyze_claim_obligations,
)

CLAIM_OBLIGATION_SET_VERSION = "claim-obligation-set-v2-candidate"


@dataclass(frozen=True)
class ClaimObligationSet:
    version: str
    source_claim_hash: str
    obligations: tuple[ClaimObligation, ...]
    deterministic_hash: str


def build_claim_obligation_set(claim_text: str) -> ClaimObligationSet:
    analysis = analyze_claim_obligations(claim_text)
    source_hash = hashlib.sha256(" ".join(claim_text.split()).encode("utf-8")).hexdigest()
    payload = "|".join(
        "::".join(
            (
                obligation.obligation_id,
                obligation.obligation_type,
                obligation.obligation_text.lower(),
                ",".join(obligation.lexical_anchors),
                ",".join(obligation.numeric_anchors),
                obligation.comparison_side or "",
            )
        )
        for obligation in analysis.obligations
    )
    deterministic_hash = hashlib.sha256(
        f"{CLAIM_OBLIGATION_SET_VERSION}|{source_hash}|{payload}".encode()
    ).hexdigest()
    return ClaimObligationSet(
        version=CLAIM_OBLIGATION_SET_VERSION,
        source_claim_hash=source_hash,
        obligations=analysis.obligations,
        deterministic_hash=deterministic_hash,
    )
