"""Evaluation-only deterministic source-Gold attribution for RAGQ3."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SourceSpan:
    block_id: str
    start: int
    end: int


@dataclass(frozen=True)
class EvidenceProvenance:
    unit_id: str
    source_spans: tuple[SourceSpan, ...]


def covered_gold_blocks(unit: EvidenceProvenance, gold_blocks: set[str]) -> set[str]:
    """Frozen ANY_SPAN overlap: representation maps only through source provenance."""
    return {span.block_id for span in unit.source_spans if span.end > span.start} & gold_blocks


def covered_claims(
    unit: EvidenceProvenance, claims: list[set[str]] | None
) -> tuple[set[int] | None, str]:
    if claims is None:
        return None, "METRIC_NOT_COMPUTABLE"
    blocks = {span.block_id for span in unit.source_spans if span.end > span.start}
    return {index for index, claim in enumerate(claims) if blocks & claim}, "COMPUTABLE"
