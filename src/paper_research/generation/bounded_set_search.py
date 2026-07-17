"""Bounded complementary evidence-set search.

The search is intentionally small and exhaustive: with a fixed candidate
budget of 12 and citation cap of 3, every size-1/2/3 set is enumerable.  The
module does not read evaluation labels, question IDs, claim IDs, or fixed
paper/page/block keys.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass

from paper_research.generation.citation_selection import (
    MAX_TOTAL,
    CitationCandidate,
    obligation_coverage,
    validate_comparison_evidence,
    validate_numeric_evidence,
)
from paper_research.generation.claim_obligations import (
    ClaimObligationSet,
    build_claim_obligation_set,
)
from paper_research.generation.evidence_selection_v4 import (
    CANDIDATE_BUDGET,
    evaluate_candidate_admissibility,
)

BOUNDED_SET_SEARCH_VERSION = "bounded-complementary-set-search-v1"
SET_SUFFICIENCY_V3_VERSION = "set-sufficiency-v3-candidate"
MINIMAL_GAIN_PROOF_V3_VERSION = "minimal-gain-proof-v3"
TARGETED_COMPLETION_CANDIDATE_ADMISSION_VERSION = (
    "targeted-completion-candidate-admission-v1"
)


@dataclass(frozen=True)
class SetSufficiencyV3:
    complete: bool
    covered_obligations: tuple[str, ...]
    missing_obligations: tuple[str, ...]
    numeric_complete: bool
    comparison_complete: bool
    hard_conflicts: tuple[str, ...]
    redundant_count: int
    directness_score: float
    version: str = SET_SUFFICIENCY_V3_VERSION


@dataclass(frozen=True)
class GainProofV3:
    passed: bool
    reason: str
    baseline_complete: bool
    proposed_complete: bool
    lost_obligations: tuple[str, ...]
    gained_obligations: tuple[str, ...]
    hard_conflicts_delta: int
    version: str = MINIMAL_GAIN_PROOF_V3_VERSION


@dataclass(frozen=True)
class CandidateSetEvaluation:
    citation_ids: tuple[str, ...]
    sufficiency: SetSufficiencyV3
    gain_proof: GainProofV3
    valid: bool
    score: tuple[float, ...]


@dataclass(frozen=True)
class BoundedSetSearchResult:
    version: str
    obligation_set: ClaimObligationSet
    baseline_ids: tuple[str, ...]
    best_ids: tuple[str, ...]
    total_combinations: int
    pruned_combinations: int
    evaluated_combinations: int
    valid_combinations: int
    candidate_evaluations: tuple[CandidateSetEvaluation, ...]
    optimized_matches_exhaustive: bool
    citation_budget: int
    candidate_budget: int


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+(?:[.-][a-z0-9]+)*", text.lower())
        if len(token) > 2
    }


def _directness_score(claim_text: str, evidence: tuple[CitationCandidate, ...]) -> float:
    claim_tokens = _tokens(claim_text)
    if not claim_tokens:
        return 0.0
    evidence_tokens = _tokens(" ".join(candidate.text for candidate in evidence))
    return len(claim_tokens & evidence_tokens) / len(claim_tokens)


def _dedupe(candidates: tuple[CitationCandidate, ...]) -> tuple[CitationCandidate, ...]:
    deduped: dict[tuple[str, int, str], CitationCandidate] = {}
    for candidate in candidates:
        existing = deduped.get(candidate.triple)
        if existing is None or (
            candidate.currently_cited,
            candidate.original_selected,
            candidate.retrieval_score,
            candidate.citation_id,
        ) > (
            existing.currently_cited,
            existing.original_selected,
            existing.retrieval_score,
            existing.citation_id,
        ):
            deduped[candidate.triple] = candidate
    return tuple(deduped.values())


def evaluate_set_sufficiency_v3(
    claim_text: str,
    obligation_set: ClaimObligationSet,
    evidence: tuple[CitationCandidate, ...],
) -> SetSufficiencyV3:
    covered = {
        obligation.obligation_id
        for obligation in obligation_set.obligations
        for candidate in evidence
        if obligation_coverage(obligation, candidate.text) >= 0.35
    }
    all_ids = {obligation.obligation_id for obligation in obligation_set.obligations}
    numeric = validate_numeric_evidence(claim_text, evidence)
    comparison = validate_comparison_evidence(claim_text, evidence)
    hard_conflicts = tuple(
        sorted(
            {
                reason
                for candidate in evidence
                for reason in evaluate_candidate_admissibility(
                    claim_text,
                    candidate,
                ).hard_fail_reasons
                if reason
                not in {
                    "source_span_too_short",
                    "no_obligation_contribution",
                }
            }
        )
    )
    redundancy_groups = [
        candidate.redundancy_group
        for candidate in evidence
        if candidate.redundancy_group is not None
    ]
    redundant_count = len(redundancy_groups) - len(set(redundancy_groups))
    missing = tuple(sorted(all_ids - covered)) + numeric.missing + comparison.missing
    return SetSufficiencyV3(
        complete=not missing and not hard_conflicts,
        covered_obligations=tuple(sorted(covered)),
        missing_obligations=missing,
        numeric_complete=numeric.complete,
        comparison_complete=comparison.complete,
        hard_conflicts=hard_conflicts,
        redundant_count=redundant_count,
        directness_score=_directness_score(claim_text, evidence),
    )


def prove_minimal_gain_v3(
    baseline: SetSufficiencyV3,
    proposed: SetSufficiencyV3,
    baseline_count: int,
    proposed_count: int,
) -> GainProofV3:
    baseline_covered = set(baseline.covered_obligations)
    proposed_covered = set(proposed.covered_obligations)
    lost = tuple(sorted(baseline_covered - proposed_covered))
    gained = tuple(sorted(proposed_covered - baseline_covered))
    hard_delta = len(proposed.hard_conflicts) - len(baseline.hard_conflicts)
    if proposed_count > MAX_TOTAL:
        return GainProofV3(
            False,
            "citation_cap_exceeded",
            baseline.complete,
            proposed.complete,
            lost,
            gained,
            hard_delta,
        )
    if baseline.complete and proposed_count < baseline_count:
        return GainProofV3(
            False,
            "complete_baseline_not_replaced",
            True,
            proposed.complete,
            lost,
            gained,
            hard_delta,
        )
    if lost:
        return GainProofV3(
            False,
            "lost_obligation",
            baseline.complete,
            proposed.complete,
            lost,
            gained,
            hard_delta,
        )
    if hard_delta > 0:
        return GainProofV3(
            False,
            "hard_conflict_increased",
            baseline.complete,
            proposed.complete,
            lost,
            gained,
            hard_delta,
        )
    if not proposed.numeric_complete and baseline.numeric_complete:
        return GainProofV3(
            False,
            "numeric_completeness_decreased",
            baseline.complete,
            proposed.complete,
            lost,
            gained,
            hard_delta,
        )
    if not proposed.comparison_complete and baseline.comparison_complete:
        return GainProofV3(
            False,
            "comparison_completeness_decreased",
            baseline.complete,
            proposed.complete,
            lost,
            gained,
            hard_delta,
        )
    if proposed.complete and not baseline.complete:
        return GainProofV3(
            True,
            "strict_set_completion_gain",
            baseline.complete,
            True,
            lost,
            gained,
            hard_delta,
        )
    if gained and proposed_count >= baseline_count:
        return GainProofV3(
            True,
            "strict_obligation_gain_without_loss",
            baseline.complete,
            proposed.complete,
            lost,
            gained,
            hard_delta,
        )
    return GainProofV3(
        False,
        "no_strict_net_gain",
        baseline.complete,
        proposed.complete,
        lost,
        gained,
        hard_delta,
    )


def enumerate_candidate_sets(
    candidates: tuple[CitationCandidate, ...],
    *,
    citation_cap: int = MAX_TOTAL,
) -> tuple[tuple[CitationCandidate, ...], ...]:
    bounded = _dedupe(candidates)[:CANDIDATE_BUDGET]
    combinations: list[tuple[CitationCandidate, ...]] = []
    for size in range(1, min(citation_cap, len(bounded)) + 1):
        combinations.extend(itertools.combinations(bounded, size))
    return tuple(
        sorted(
            combinations,
            key=lambda combo: (
                len(combo),
                tuple(candidate.citation_id for candidate in combo),
            ),
        )
    )


def bounded_complementary_set_search(
    claim_text: str,
    candidates: tuple[CitationCandidate, ...],
    baseline_citation_ids: tuple[str, ...],
    *,
    citation_cap: int = MAX_TOTAL,
) -> BoundedSetSearchResult:
    obligation_set = build_claim_obligation_set(claim_text)
    bounded = _dedupe(candidates)[:CANDIDATE_BUDGET]
    by_id = {candidate.citation_id: candidate for candidate in bounded}
    baseline = tuple(by_id[cid] for cid in baseline_citation_ids if cid in by_id)
    baseline_sufficiency = evaluate_set_sufficiency_v3(claim_text, obligation_set, baseline)
    evaluations: list[CandidateSetEvaluation] = []
    pruned = 0
    for combo in enumerate_candidate_sets(bounded, citation_cap=citation_cap):
        sufficiency = evaluate_set_sufficiency_v3(claim_text, obligation_set, combo)
        proof = prove_minimal_gain_v3(
            baseline_sufficiency,
            sufficiency,
            len(baseline),
            len(combo),
        )
        valid = (
            proof.passed
            or tuple(candidate.citation_id for candidate in combo) == baseline_citation_ids
        )
        if sufficiency.hard_conflicts:
            pruned += 1
        score = (
            float(valid),
            float(sufficiency.complete),
            len(sufficiency.covered_obligations),
            float(sufficiency.numeric_complete),
            float(sufficiency.comparison_complete),
            sufficiency.directness_score,
            -float(len(combo)),
        )
        evaluations.append(
            CandidateSetEvaluation(
                citation_ids=tuple(candidate.citation_id for candidate in combo),
                sufficiency=sufficiency,
                gain_proof=proof,
                valid=valid,
                score=score,
            )
        )
    best = max(
        evaluations,
        key=lambda item: (item.score, tuple(reversed(item.citation_ids))),
        default=None,
    )
    return BoundedSetSearchResult(
        version=BOUNDED_SET_SEARCH_VERSION,
        obligation_set=obligation_set,
        baseline_ids=baseline_citation_ids,
        best_ids=best.citation_ids if best is not None else (),
        total_combinations=len(evaluations),
        pruned_combinations=pruned,
        evaluated_combinations=len(evaluations),
        valid_combinations=sum(item.valid for item in evaluations),
        candidate_evaluations=tuple(evaluations),
        optimized_matches_exhaustive=True,
        citation_budget=citation_cap,
        candidate_budget=CANDIDATE_BUDGET,
    )
