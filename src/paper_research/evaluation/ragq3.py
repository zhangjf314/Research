"""Deterministic, representation-only diagnostics for RAG Quality v3."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from hashlib import sha256


@dataclass(frozen=True)
class EvidenceUnit:
    unit_id: str
    block_ids: frozenset[str]
    text: str
    parent_id: str | None = None


def covered(unit_ids: Iterable[str], claim: set[str]) -> bool:
    return bool(set(unit_ids) & claim)


def claim_coverage(units: Iterable[EvidenceUnit], claims: list[set[str]]) -> tuple[int, int]:
    blocks = set().union(*(unit.block_ids for unit in units)) if claims else set()
    return sum(bool(blocks & claim) for claim in claims), len(claims)


def loss_decomposition(
    pool: Iterable[EvidenceUnit],
    top5: Iterable[EvidenceUnit],
    context: Iterable[EvidenceUnit],
    claims: list[set[str]],
) -> dict[str, int]:
    available, total = claim_coverage(pool, claims)
    selected, _ = claim_coverage(top5, claims)
    retained, _ = claim_coverage(context, claims)
    return {
        "required_claims_total": total,
        "claims_available_in_pool": available,
        "claims_selected_top5": selected,
        "claims_retained_after_packing": retained,
        "candidate_loss": total - available,
        "ranking_loss": available - selected,
        "packing_loss": selected - retained,
    }


def failure_labels(
    pool: Iterable[EvidenceUnit],
    top5: Iterable[EvidenceUnit],
    context: Iterable[EvidenceUnit],
    claims: list[set[str]],
) -> list[str]:
    loss = loss_decomposition(pool, top5, context, claims)
    labels: list[str] = []
    if loss["candidate_loss"]:
        labels.append("CANDIDATE_MISSING_REQUIRED_EVIDENCE")
    if loss["ranking_loss"]:
        labels.append("CORRECT_EVIDENCE_IN_POOL_NOT_TOP5")
    if loss["packing_loss"]:
        labels.append("CONTEXT_PACKING_TRUNCATION")
    if loss["claims_selected_top5"] < loss["required_claims_total"]:
        labels.append("MULTI_EVIDENCE_INCOMPLETE")
    return labels or ["PASS"]


def sentence_windows(block_id: str, text: str, *, radius: int = 1) -> list[EvidenceUnit]:
    """Split only on deterministic terminal punctuation and preserve source-block identity."""
    sentences = [item.strip() for item in re.split(r"(?<=[.!?])\s+", text) if item.strip()]
    result = []
    for index, _sentence in enumerate(sentences):
        lo, hi = max(0, index - radius), min(len(sentences), index + radius + 1)
        unit_id = f"sent:{block_id}:{index}"
        result.append(
            EvidenceUnit(unit_id, frozenset({block_id}), " ".join(sentences[lo:hi]), block_id)
        )
    return result


def parent_child_windows(
    parent_id: str, block_ids: list[str], text: str, *, width: int = 128, overlap: int = 32
) -> list[EvidenceUnit]:
    """Create deterministic child windows without consulting gold labels."""
    if width <= overlap:
        raise ValueError("width must exceed overlap")
    tokens = text.split()
    result = []
    for start in range(0, len(tokens), width - overlap):
        window = tokens[start : start + width]
        if not window:
            break
        digest = sha256(f"{parent_id}:{start}:{' '.join(window)}".encode()).hexdigest()[:16]
        result.append(
            EvidenceUnit(
                f"child:{parent_id}:{digest}", frozenset(block_ids), " ".join(window), parent_id
            )
        )
        if start + width >= len(tokens):
            break
    return result


def parent_dedupe(
    children: Iterable[tuple[EvidenceUnit, float]],
) -> list[tuple[EvidenceUnit, float]]:
    """Deterministic parent projection: maximum child score, first ranked child breaks ties."""
    chosen: dict[str, tuple[EvidenceUnit, float]] = {}
    for child, score in children:
        assert child.parent_id is not None
        current = chosen.get(child.parent_id)
        if current is None or score > current[1]:
            chosen[child.parent_id] = (child, score)
    return sorted(chosen.values(), key=lambda item: (-item[1], item[0].unit_id))


def fragmentation_summary(
    claims: list[set[str]],
    block_order: list[str],
    section_by_block: dict[str, str],
    parent_by_block: dict[str, str],
) -> dict[str, object]:
    positions = {block_id: index for index, block_id in enumerate(block_order)}
    rows = []
    for claim in claims:
        ordered = sorted((block for block in claim if block in positions), key=positions.get)
        adjacent = len(ordered) > 1 and all(
            b - a == 1
            for a, b in zip(
                map(positions.get, ordered), map(positions.get, ordered[1:]), strict=False
            )
        )
        rows.append(
            {
                "gold_blocks": len(claim),
                "observed_blocks": len(ordered),
                "adjacent": adjacent,
                "same_section": len({section_by_block.get(block) for block in ordered}) <= 1,
                "same_parent": len({parent_by_block.get(block) for block in ordered}) <= 1,
                "cross_chunk_required": len(ordered) > 1,
            }
        )
    counts = Counter()
    for row in rows:
        for key in ("adjacent", "same_section", "same_parent", "cross_chunk_required"):
            counts[key] += int(bool(row[key]))
    return {"claims": len(rows), "counts": dict(counts), "rows": rows}


def gate_decision(
    delta: dict[str, float], *, paper_tie_or_better_rate: float, gain_paper_count: int
) -> dict[str, bool]:
    """Frozen A1 gate; caller supplies aggregates sliced by the registered corpus."""
    return {
        "A": delta.get("multi_evidence_all_claims_present@pool", 0.0) >= 0.10
        or delta.get("required_claim_coverage@pool", 0.0) >= 0.05,
        "B": delta.get("candidate_pool_gold_recall", 0.0) >= -0.02,
        "C": delta.get("semantic_required_claim_coverage@pool", 0.0) >= -0.02
        and delta.get("semantic_gold_block_recall@5", 0.0) >= -0.02,
        "D": delta.get("MRR", 0.0) >= -0.03,
        "E": delta.get("context_gold_precision", 0.0) >= -0.02,
        "F": paper_tie_or_better_rate >= 0.75,
        "G": gain_paper_count > 2,
    }
