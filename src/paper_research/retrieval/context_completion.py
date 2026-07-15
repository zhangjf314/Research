from __future__ import annotations

from collections.abc import Iterable

from paper_research.evidence.schema import EvidenceUnit
from paper_research.retrieval.evidence_retriever import EvidenceCandidate

EXCLUDED_ADJACENT_ROLES = {"metadata", "citation_only"}


def complete_with_adjacent_same_page(
    selected: list[EvidenceCandidate],
    units: Iterable[EvidenceUnit],
    *,
    seed_limit: int = 5,
    window: int = 1,
) -> list[EvidenceCandidate]:
    """Add same-page neighbor blocks around high ranked selected evidence.

    This is a generic boundary-completion rule. It uses only the selected
    evidence positions and parsed corpus adjacency, never review labels.
    """
    by_position = {(unit.paper_id, unit.page, unit.ordinal): unit for unit in units}
    completed = list(selected)
    seen = {(item.evidence.paper_id, item.evidence.block_id) for item in completed}
    for parent in selected[:seed_limit]:
        unit = parent.evidence
        for delta in range(-window, window + 1):
            if delta == 0:
                continue
            neighbor = by_position.get((unit.paper_id, unit.page, unit.ordinal + delta))
            if neighbor is None:
                continue
            key = (neighbor.paper_id, neighbor.block_id)
            if key in seen:
                continue
            if EXCLUDED_ADJACENT_ROLES & set(neighbor.evidence_roles):
                continue
            candidate = parent.model_copy(deep=True)
            candidate.evidence = neighbor
            candidate.total_score = round(max(0.0, parent.total_score - 0.015 * abs(delta)), 8)
            candidate.filter_reasons = [
                *candidate.filter_reasons,
                "phase_b_adjacent_same_page_completion",
            ]
            candidate.rejection_reasons = []
            completed.append(candidate)
            seen.add(key)
    return completed
