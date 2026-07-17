"""Offline candidate for obligation-aware evidence selection.

The selector deliberately accepts only model claim text, local candidate evidence,
and local retrieval metadata. It does not accept Gold relations, human labels,
question IDs, required-claim IDs, or fixed paper/page/block special cases.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from paper_research.generation.citation_selection import (
    MAX_TOTAL,
    CitationCandidate,
    FallbackAction,
    analyze_claim_obligations,
    decide_claim_fallback,
    obligation_coverage,
    validate_comparison_evidence,
    validate_numeric_evidence,
)

EVIDENCE_SELECTION_V2_VERSION = "evidence-selection-v2-candidate"


@dataclass(frozen=True)
class EvidenceSelectionV2Result:
    version: str
    primary_citation_ids: tuple[str, ...]
    supporting_citation_ids: tuple[str, ...]
    rejected_citation_ids: tuple[str, ...]
    fallback_action: FallbackAction
    narrowed_claim_text: str | None
    removed_obligations: tuple[str, ...]
    decision_reasons: tuple[str, ...]
    selected_count: int
    citation_cap_blocked: bool


def _stable_score(claim_text: str, candidate: CitationCandidate) -> tuple[float, ...]:
    obligations = analyze_claim_obligations(claim_text).obligations
    coverages = tuple(obligation_coverage(obligation, candidate.text) for obligation in obligations)
    unique_obligations = sum(score >= 0.35 for score in coverages)
    numeric = validate_numeric_evidence(claim_text, [candidate])
    comparison = validate_comparison_evidence(claim_text, [candidate])
    role = 1.0 if set(candidate.evidence_role) & {
        "method",
        "result",
        "setup",
        "comparison",
        "limitation",
        "definition",
    } else 0.0
    adjacent_bonus = 0.15 if candidate.adjacent_completion and unique_obligations else 0.0
    original_bonus = 0.10 if candidate.original_selected else 0.0
    retrieval = candidate.retrieval_score if math.isfinite(candidate.retrieval_score) else 0.0
    return (
        float(numeric.complete),
        len(comparison.covered) / max(len(comparison.required), 1),
        float(unique_obligations),
        sum(coverages),
        max(coverages, default=0.0),
        role,
        adjacent_bonus,
        original_bonus,
        retrieval,
        -float(candidate.token_cost),
    )


def select_evidence_v2(
    claim_text: str,
    candidates: tuple[CitationCandidate, ...],
) -> EvidenceSelectionV2Result:
    """Select a minimal complementary evidence set without online Gold features."""
    deduped: dict[tuple[str, int, str], CitationCandidate] = {}
    for candidate in candidates:
        existing = deduped.get(candidate.triple)
        if existing is None or _stable_score(claim_text, candidate) > _stable_score(
            claim_text, existing
        ):
            deduped[candidate.triple] = candidate
    ordered = sorted(
        deduped.values(),
        key=lambda candidate: (
            tuple(-value for value in _stable_score(claim_text, candidate)),
            candidate.citation_id,
        ),
    )
    analysis = analyze_claim_obligations(claim_text)
    obligations = analysis.obligations
    if not ordered or not obligations:
        return EvidenceSelectionV2Result(
            version=EVIDENCE_SELECTION_V2_VERSION,
            primary_citation_ids=(),
            supporting_citation_ids=(),
            rejected_citation_ids=tuple(candidate.citation_id for candidate in ordered),
            fallback_action=FallbackAction.UNSUPPORTED,
            narrowed_claim_text=None,
            removed_obligations=(),
            decision_reasons=("no_usable_candidate_or_empty_claim",),
            selected_count=0,
            citation_cap_blocked=False,
        )
    selected = [ordered[0]]
    covered = {
        obligation.obligation_id
        for obligation in obligations
        if obligation_coverage(obligation, ordered[0].text) >= 0.35
    }
    for candidate in ordered[1:]:
        if len(selected) >= MAX_TOTAL:
            break
        newly_covered = {
            obligation.obligation_id
            for obligation in obligations
            if obligation.obligation_id not in covered
            and obligation_coverage(obligation, candidate.text) >= 0.35
        }
        numeric_needed = not validate_numeric_evidence(claim_text, selected).complete
        numeric_help = validate_numeric_evidence(claim_text, [*selected, candidate]).complete
        comparison_needed = not validate_comparison_evidence(claim_text, selected).complete
        comparison_help = validate_comparison_evidence(claim_text, [*selected, candidate]).complete
        if newly_covered or (numeric_needed and numeric_help) or (
            comparison_needed and comparison_help
        ):
            selected.append(candidate)
            covered.update(newly_covered)
    numeric = validate_numeric_evidence(claim_text, selected)
    comparison = validate_comparison_evidence(claim_text, selected)
    fallback, narrowed, removed = decide_claim_fallback(
        claim_text,
        obligations,
        covered,
        numeric,
        comparison,
    )
    selected_ids = tuple(candidate.citation_id for candidate in selected)
    rejected = tuple(candidate.citation_id for candidate in ordered if candidate not in selected)
    cap_blocked = len(selected) == MAX_TOTAL and any(
        obligation.obligation_id not in covered for obligation in obligations
    )
    return EvidenceSelectionV2Result(
        version=EVIDENCE_SELECTION_V2_VERSION,
        primary_citation_ids=selected_ids[:1],
        supporting_citation_ids=selected_ids[1:],
        rejected_citation_ids=rejected,
        fallback_action=fallback,
        narrowed_claim_text=narrowed,
        removed_obligations=removed,
        decision_reasons=(
            "obligation_weighted_set_coverage",
            "numeric_anchor_hard_requirement",
            "comparison_side_balanced_allocation",
            "adjacent_only_when_complementary",
        ),
        selected_count=len(selected),
        citation_cap_blocked=cap_blocked,
    )
