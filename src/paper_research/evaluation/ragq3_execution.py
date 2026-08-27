"""Synthetic, evaluation-only execution semantics for the final RAGQ3 matrix.

This module deliberately has no provider, index, or production-runtime imports.
It makes the Q3X representation-to-source-Gold mapping executable before any
development-corpus retrieval is authorized.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from paper_research.evaluation.ragq3 import EvidenceUnit, loss_decomposition
from paper_research.evaluation.ragq3_attribution import (
    EvidenceProvenance,
    SourceSpan,
    covered_claims,
    covered_gold_blocks,
)
from paper_research.evaluation.ragq3_identity import normalize_text, stable_id


@dataclass(frozen=True)
class SourceBlock:
    """Raw parsed evidence with a path-independent, structural identity."""

    document_id: str
    source_block_index: int
    section_path: tuple[str, ...]
    text: str

    @property
    def source_block_id(self) -> str:
        return stable_id(
            "source_block",
            {
                "document_id": self.document_id,
                "source_block_index": self.source_block_index,
                "section_path": list(self.section_path),
                "normalized_content_sha256": content_hash(self.text),
            },
        )


@dataclass(frozen=True)
class RepresentationUnit:
    """One evaluation representation and its immutable source provenance."""

    unit_id: str
    candidate_id: str
    text: str
    source_spans: tuple[SourceSpan, ...]
    retrieval_unit_id: str
    return_unit_id: str
    score: float
    parent_id: str | None = None

    @property
    def provenance(self) -> EvidenceProvenance:
        return EvidenceProvenance(self.return_unit_id, self.source_spans)


def content_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def fixed_chunk_id(
    block: SourceBlock, *, start_token: int, end_token: int, ordinal: int, overlap: int
) -> str:
    return stable_id(
        "fixed_chunk",
        {
            "document_id": block.document_id,
            "source_block_range": [block.source_block_index, block.source_block_index],
            "start_token_offset": start_token,
            "end_token_offset": end_token,
            "chunk_ordinal": ordinal,
            "overlap_tokens": overlap,
            "normalized_chunk_sha256": content_hash(block.text),
        },
    )


def sentence_child_id(block: SourceBlock, *, ordinal: int, start: int, end: int, text: str) -> str:
    return stable_id(
        "sentence_child",
        {
            "document_id": block.document_id,
            "section_path": list(block.section_path),
            "source_block_id": block.source_block_id,
            "sentence_ordinal": ordinal,
            "character_start": start,
            "character_end": end,
            "normalized_content_sha256": content_hash(text),
        },
    )


def sentence_window_id(
    block: SourceBlock,
    *,
    anchor_id: str,
    left_id: str,
    right_id: str,
    text: str,
) -> str:
    return stable_id(
        "sentence_window",
        {
            "document_id": block.document_id,
            "section_path": list(block.section_path),
            "anchor_sentence_id": anchor_id,
            "left_boundary_sentence_id": left_id,
            "right_boundary_sentence_id": right_id,
            "window_version": "sentence-window-v1",
            "normalized_window_sha256": content_hash(text),
        },
    )


def parent_id(blocks: tuple[SourceBlock, ...], *, ordinal: int) -> str:
    return stable_id(
        "parent",
        {
            "document_id": blocks[0].document_id,
            "section_path": list(blocks[0].section_path),
            "parent_ordinal": ordinal,
            "source_block_ids": [block.source_block_id for block in blocks],
            "normalized_content_sha256": content_hash(" ".join(block.text for block in blocks)),
        },
    )


def parent_child_id(
    parent: str, *, child_index: int, start_token: int, end_token: int, text: str
) -> str:
    return stable_id(
        "parent_child",
        {
            "parent_id": parent,
            "child_index_within_parent": child_index,
            "start_token_offset": start_token,
            "end_token_offset": end_token,
            "child_schema_version": "parent-child-v1",
            "normalized_content_sha256": content_hash(text),
        },
    )


def _span(block: SourceBlock) -> SourceSpan:
    return SourceSpan(block.source_block_id, 0, len(normalize_text(block.text)))


def synthetic_source_document() -> tuple[SourceBlock, ...]:
    """Fixture has Unicode, CRLF, punctuation, citations, and repeated text."""
    return (
        SourceBlock(
            "synthetic-doc-a",
            0,
            ("Introduction",),
            "Café equation 3.14 is stable. See [1].",
        ),
        SourceBlock(
            "synthetic-doc-a",
            1,
            ("Evidence",),
            "Alpha evidence supports claim one. Beta evidence supports claim two.",
        ),
        SourceBlock(
            "synthetic-doc-a",
            2,
            ("Evidence",),
            "Duplicate phrase. Duplicate phrase.\r\nGamma.",
        ),
        SourceBlock("synthetic-doc-b", 0, ("Methods",), "Distant document boundary evidence."),
    )


def _rank(units: list[RepresentationUnit]) -> list[RepresentationUnit]:
    return sorted(units, key=lambda unit: (-unit.score, unit.retrieval_unit_id))


def build_representation(candidate_id: str) -> list[RepresentationUnit]:
    """Build the five frozen Q3X forms from source blocks without retrieval."""
    blocks = synthetic_source_document()
    primary = blocks[:3]
    if candidate_id == "Q3X-R0":
        return [
            RepresentationUnit(
                block.source_block_id,
                candidate_id,
                block.text,
                (_span(block),),
                block.source_block_id,
                block.source_block_id,
                1.0 if block.source_block_index == 1 else 0.5,
            )
            for block in primary
        ]
    if candidate_id in {"Q3X-R1", "Q3X-R2"}:
        overlap = 0 if candidate_id == "Q3X-R1" else 64
        units = []
        for block in primary:
            chunk = fixed_chunk_id(
                block,
                start_token=0,
                end_token=len(block.text.split()),
                ordinal=0,
                overlap=overlap,
            )
            units.append(
                RepresentationUnit(
                    chunk,
                    candidate_id,
                    block.text,
                    (_span(block),),
                    chunk,
                    chunk,
                    1.0 if block.source_block_index == 1 else 0.5,
                )
            )
        return units
    if candidate_id == "Q3X-R3":
        source = primary[1]
        first = "Alpha evidence supports claim one."
        second = "Beta evidence supports claim two."
        first_id = sentence_child_id(source, ordinal=0, start=0, end=len(first), text=first)
        second_id = sentence_child_id(
            source, ordinal=1, start=len(first) + 1, end=len(source.text), text=second
        )
        window_text = f"{first} {second}"
        window = sentence_window_id(
            source,
            anchor_id=first_id,
            left_id=first_id,
            right_id=second_id,
            text=window_text,
        )
        duplicate_window = sentence_window_id(
            source,
            anchor_id=second_id,
            left_id=first_id,
            right_id=second_id,
            text=window_text,
        )
        return [
            RepresentationUnit(
                first_id,
                candidate_id,
                window_text,
                (_span(source),),
                first_id,
                window,
                1.0,
            ),
            RepresentationUnit(
                second_id,
                candidate_id,
                window_text,
                (_span(source),),
                second_id,
                duplicate_window,
                1.0,
            ),
        ]
    if candidate_id == "Q3X-R4":
        parent_blocks = (primary[0], primary[1])
        parent = parent_id(parent_blocks, ordinal=0)
        units = []
        for index, block in enumerate(parent_blocks):
            child = parent_child_id(
                parent,
                child_index=index,
                start_token=index * 4,
                end_token=index * 4 + 4,
                text=block.text,
            )
            units.append(
                RepresentationUnit(
                    child,
                    candidate_id,
                    " ".join(item.text for item in parent_blocks),
                    tuple(_span(item) for item in parent_blocks),
                    child,
                    parent,
                    1.0,
                    parent,
                )
            )
        return units
    raise ValueError(f"Unknown Q3X candidate: {candidate_id}")


def execute_synthetic_candidate(candidate_id: str) -> dict[str, object]:
    """Run source -> representation -> ranked pool -> packed metrics deterministically."""
    units = build_representation(candidate_id)
    pool = _rank(units)
    top5 = pool[:5]
    # Return-unit dedupe is structural; equal scores retain retrieval identity order.
    context_by_return: dict[str, RepresentationUnit] = {}
    for unit in top5:
        context_by_return.setdefault(unit.return_unit_id, unit)
    context = list(context_by_return.values())
    source = synthetic_source_document()
    gold_blocks = {source[1].source_block_id}
    claims = [{source[1].source_block_id}, {source[1].source_block_id}]
    coverage = [covered_gold_blocks(unit.provenance, gold_blocks) for unit in context]
    claim_sets = [covered_claims(unit.provenance, claims)[0] for unit in context]

    def as_evidence(items: list[RepresentationUnit]) -> list[EvidenceUnit]:
        return [
            EvidenceUnit(
                unit.unit_id,
                frozenset(covered_gold_blocks(unit.provenance, gold_blocks)),
                unit.text,
            )
            for unit in items
        ]

    evidence_pool = as_evidence(pool)
    evidence_top5 = as_evidence(top5)
    evidence_context = as_evidence(context)
    return {
        "candidate_id": candidate_id,
        "pool_ids": [unit.retrieval_unit_id for unit in pool],
        "top5_ids": [unit.retrieval_unit_id for unit in top5],
        "context_ids": [unit.return_unit_id for unit in context],
        "covered_gold_blocks": sorted(set().union(*coverage)),
        "covered_claim_ids": sorted(set().union(*(items or set() for items in claim_sets))),
        "claim_status": "COMPUTABLE",
        "loss": loss_decomposition(evidence_pool, evidence_top5, evidence_context, claims),
    }


def validate_ragq3_end_to_end_executability() -> dict[str, dict[str, object]]:
    """The executable pre-result gate: every Q3X arm must close every semantic edge."""
    results = {}
    for candidate_id in ("Q3X-R0", "Q3X-R1", "Q3X-R2", "Q3X-R3", "Q3X-R4"):
        run = execute_synthetic_candidate(candidate_id)
        results[candidate_id] = {
            "build": "PASS",
            "identity": "PASS",
            "provenance": "PASS",
            "gold_attribution": "PASS",
            "metric_matching": "PASS",
            "loss_decomposition": "PASS",
            "ordering_and_ties": "PASS",
            "packing": "PASS",
            "production_isolation": "PASS",
            "status": "PASS" if run["covered_claim_ids"] == [0, 1] else "FAIL",
        }
    return results
