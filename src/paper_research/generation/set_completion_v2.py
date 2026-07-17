"""Set-level obligation completion candidate.

This module composes local evidence candidates into small citation sets. It is
baseline-first, deterministic, and deliberately free of offline review labels,
question IDs, required-claim IDs, or fixed relation keys.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from enum import StrEnum

from paper_research.generation.citation_selection import (
    MAX_TOTAL,
    CitationCandidate,
    FallbackAction,
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

SET_COMPLETION_V2_VERSION = "set-completion-v2-candidate"
COMPLEMENTARY_SET_SEARCH_VERSION = "complementary-set-search-v2"
SET_SUFFICIENCY_V2_VERSION = "set-sufficiency-v2"
NUMERIC_SET_COVERAGE_VERSION = "numeric-set-coverage-v2"
COMPARISON_SET_COVERAGE_VERSION = "comparison-set-coverage-v2"
MINIMAL_GAIN_PROOF_VERSION = "minimal-gain-proof-v2"


class SetCompletionOperation(StrEnum):
    KEEP_BASELINE = "keep_baseline"
    ADD_COMPLEMENT = "add_complement"
    REPLACE_HARD_INVALID_BASELINE = "replace_hard_invalid_baseline"
    KEEP_UNSUPPORTED = "keep_unsupported"


@dataclass(frozen=True)
class SetCoverageV2:
    complete: bool
    covered_obligations: tuple[str, ...]
    missing_obligations: tuple[str, ...]
    numeric_applicable: bool
    numeric_complete: bool
    comparison_applicable: bool
    comparison_complete: bool
    directness_score: float
    version: str = SET_SUFFICIENCY_V2_VERSION


@dataclass(frozen=True)
class SetCompletionV2Result:
    version: str
    obligation_set: ClaimObligationSet
    primary_citation_ids: tuple[str, ...]
    supporting_citation_ids: tuple[str, ...]
    rejected_citation_ids: tuple[str, ...]
    operations: tuple[SetCompletionOperation, ...]
    baseline_coverage: SetCoverageV2
    final_coverage: SetCoverageV2
    combinations_considered: int
    combinations_pruned: int
    complete_sets_found: int
    valid_complementary_additions: tuple[str, ...]
    rejected_additions: tuple[str, ...]
    fallback_action: FallbackAction
    candidate_budget: int
    citation_budget: int
    candidate_admission_v4_required: bool
    claim_fallback_v4_required: bool


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(
        token
        for token in re.findall(r"[a-z0-9]+(?:[.-][a-z0-9]+)*", text.lower())
        if len(token) > 2
    )


def _bigrams(tokens: tuple[str, ...]) -> set[tuple[str, str]]:
    return set(zip(tokens, tokens[1:], strict=False))


def directness_score(claim_text: str, evidence: tuple[CitationCandidate, ...]) -> float:
    claim_tokens = _tokens(claim_text)
    evidence_tokens = _tokens(" ".join(candidate.text for candidate in evidence))
    if not claim_tokens:
        return 0.0
    token_score = len(set(claim_tokens) & set(evidence_tokens)) / len(set(claim_tokens))
    claim_bigrams = _bigrams(claim_tokens)
    evidence_bigrams = _bigrams(evidence_tokens)
    bigram_score = len(claim_bigrams & evidence_bigrams) / max(len(claim_bigrams), 1)
    return 0.65 * token_score + 0.35 * bigram_score


def evaluate_set_coverage_v2(
    claim_text: str,
    obligation_set: ClaimObligationSet,
    evidence: tuple[CitationCandidate, ...],
) -> SetCoverageV2:
    covered = {
        obligation.obligation_id
        for obligation in obligation_set.obligations
        for candidate in evidence
        if obligation_coverage(obligation, candidate.text) >= 0.35
    }
    all_ids = {obligation.obligation_id for obligation in obligation_set.obligations}
    numeric = validate_numeric_evidence(claim_text, evidence)
    comparison = validate_comparison_evidence(claim_text, evidence)
    numeric_applicable = bool(numeric.required)
    comparison_applicable = bool(comparison.required)
    missing = tuple(sorted(all_ids - covered))
    if numeric_applicable and not numeric.complete:
        missing += numeric.missing
    if comparison_applicable and not comparison.complete:
        missing += comparison.missing
    return SetCoverageV2(
        complete=all_ids.issubset(covered)
        and (numeric.complete if numeric_applicable else True)
        and (comparison.complete if comparison_applicable else True),
        covered_obligations=tuple(sorted(covered)),
        missing_obligations=missing,
        numeric_applicable=numeric_applicable,
        numeric_complete=numeric.complete if numeric_applicable else True,
        comparison_applicable=comparison_applicable,
        comparison_complete=comparison.complete if comparison_applicable else True,
        directness_score=directness_score(claim_text, evidence),
    )


def _dedupe(candidates: tuple[CitationCandidate, ...]) -> tuple[CitationCandidate, ...]:
    deduped: dict[tuple[str, int, str], CitationCandidate] = {}
    for candidate in candidates:
        existing = deduped.get(candidate.triple)
        if existing is None or candidate.retrieval_score > existing.retrieval_score:
            deduped[candidate.triple] = candidate
    return tuple(deduped.values())


def _rank_candidate(claim_text: str, candidate: CitationCandidate) -> tuple[float, ...]:
    obligation_set = build_claim_obligation_set(claim_text)
    single = (candidate,)
    coverage = evaluate_set_coverage_v2(claim_text, obligation_set, single)
    return (
        float(candidate.currently_cited),
        float(candidate.original_selected),
        float(coverage.complete),
        len(coverage.covered_obligations),
        coverage.directness_score,
        candidate.retrieval_score,
        -float(candidate.token_cost),
    )


def admit_candidates_v4(
    claim_text: str,
    candidates: tuple[CitationCandidate, ...],
    baseline_citation_ids: tuple[str, ...],
) -> tuple[CitationCandidate, ...]:
    deduped = list(_dedupe(candidates))
    by_id = {candidate.citation_id: candidate for candidate in deduped}
    selected = [by_id[cid] for cid in baseline_citation_ids if cid in by_id]
    for candidate in sorted(
        [candidate for candidate in deduped if candidate not in selected],
        key=lambda item: (
            tuple(-value for value in _rank_candidate(claim_text, item)),
            item.citation_id,
        ),
    ):
        if len(selected) >= CANDIDATE_BUDGET:
            break
        selected.append(candidate)
    return tuple(selected[:CANDIDATE_BUDGET])


def _candidate_conflicts(claim_text: str, candidate: CitationCandidate) -> bool:
    result = evaluate_candidate_admissibility(claim_text, candidate)
    hard_conflicts = {
        "polarity_conflict",
        "method_object_conflict",
        "source_directness_conflict",
        "source_span_too_short",
        "entity_or_method_unanchored",
    }
    return bool(set(result.hard_fail_reasons) & hard_conflicts)


def _best_set(
    claim_text: str,
    obligation_set: ClaimObligationSet,
    baseline: tuple[CitationCandidate, ...],
    candidates: tuple[CitationCandidate, ...],
) -> tuple[tuple[CitationCandidate, ...], int, int, int]:
    baseline_ids = {candidate.citation_id for candidate in baseline}
    pool = [candidate for candidate in candidates if candidate.citation_id not in baseline_ids]
    considered = 0
    pruned = 0
    complete_found = 0
    baseline_coverage = evaluate_set_coverage_v2(claim_text, obligation_set, baseline)
    best = baseline
    best_coverage = baseline_coverage
    for size in range(1, max(0, MAX_TOTAL - len(baseline)) + 1):
        for combo in itertools.combinations(pool, size):
            considered += 1
            if any(_candidate_conflicts(claim_text, candidate) for candidate in combo):
                pruned += 1
                continue
            trial = (*baseline, *combo)
            coverage = evaluate_set_coverage_v2(claim_text, obligation_set, trial)
            if coverage.complete:
                complete_found += 1
            gained = len(coverage.covered_obligations) > len(best_coverage.covered_obligations)
            completed_numeric = coverage.numeric_complete and not best_coverage.numeric_complete
            completed_comparison = (
                coverage.comparison_complete and not best_coverage.comparison_complete
            )
            direct_gain = coverage.directness_score >= best_coverage.directness_score + 0.20
            if (gained or completed_numeric or completed_comparison or direct_gain) and (
                coverage.directness_score >= best_coverage.directness_score
                or coverage.complete
            ):
                best = trial
                best_coverage = coverage
    return best, considered, pruned, complete_found


def select_set_completion_v2(
    claim_text: str,
    candidates: tuple[CitationCandidate, ...],
    baseline_citation_ids: tuple[str, ...],
    *,
    use_candidate_admission_v4: bool = True,
    use_fallback_v4: bool = False,
    baseline_first: bool = True,
    greedy: bool = False,
    use_numeric_rules: bool = True,
    use_comparison_rules: bool = True,
) -> SetCompletionV2Result:
    obligation_set = build_claim_obligation_set(claim_text)
    admitted = (
        admit_candidates_v4(claim_text, candidates, baseline_citation_ids)
        if use_candidate_admission_v4
        else _dedupe(candidates)[:CANDIDATE_BUDGET]
    )
    by_id = {candidate.citation_id: candidate for candidate in admitted}
    baseline = tuple(by_id[cid] for cid in baseline_citation_ids if cid in by_id)
    baseline_coverage = evaluate_set_coverage_v2(claim_text, obligation_set, baseline)
    if baseline_first and baseline_coverage.complete:
        selected = baseline
        considered = pruned = complete_found = 0
    elif greedy:
        selected = baseline
        considered = pruned = complete_found = 0
        while len(selected) < MAX_TOTAL:
            remaining = [candidate for candidate in admitted if candidate not in selected]
            if not remaining:
                break
            candidate = sorted(
                remaining,
                key=lambda item: (
                    tuple(-value for value in _rank_candidate(claim_text, item)),
                    item.citation_id,
                ),
            )[0]
            selected = (*selected, candidate)
            considered += 1
    else:
        selected, considered, pruned, complete_found = _best_set(
            claim_text, obligation_set, baseline, admitted
        )
    final_coverage = evaluate_set_coverage_v2(claim_text, obligation_set, selected)
    if not use_numeric_rules:
        final_coverage = SetCoverageV2(
            complete=final_coverage.complete or not final_coverage.missing_obligations,
            covered_obligations=final_coverage.covered_obligations,
            missing_obligations=final_coverage.missing_obligations,
            numeric_applicable=final_coverage.numeric_applicable,
            numeric_complete=True,
            comparison_applicable=final_coverage.comparison_applicable,
            comparison_complete=final_coverage.comparison_complete,
            directness_score=final_coverage.directness_score,
        )
    if not use_comparison_rules:
        final_coverage = SetCoverageV2(
            complete=final_coverage.complete or not final_coverage.missing_obligations,
            covered_obligations=final_coverage.covered_obligations,
            missing_obligations=final_coverage.missing_obligations,
            numeric_applicable=final_coverage.numeric_applicable,
            numeric_complete=final_coverage.numeric_complete,
            comparison_applicable=final_coverage.comparison_applicable,
            comparison_complete=True,
            directness_score=final_coverage.directness_score,
        )
    selected_ids = tuple(candidate.citation_id for candidate in selected)
    baseline_ids = {candidate.citation_id for candidate in baseline}
    operations: list[SetCompletionOperation] = [SetCompletionOperation.KEEP_BASELINE]
    if set(selected_ids) - baseline_ids:
        operations.append(SetCompletionOperation.ADD_COMPLEMENT)
    if not selected_ids:
        operations.append(SetCompletionOperation.KEEP_UNSUPPORTED)
    fallback = FallbackAction.ANSWERED_ORIGINAL if selected_ids else FallbackAction.UNSUPPORTED
    if use_fallback_v4 and final_coverage.complete and selected_ids:
        fallback = FallbackAction.ANSWERED_ORIGINAL
    return SetCompletionV2Result(
        version=SET_COMPLETION_V2_VERSION,
        obligation_set=obligation_set,
        primary_citation_ids=selected_ids[:1],
        supporting_citation_ids=selected_ids[1:],
        rejected_citation_ids=tuple(
            candidate.citation_id
            for candidate in admitted
            if candidate.citation_id not in selected_ids
        ),
        operations=tuple(dict.fromkeys(operations)),
        baseline_coverage=baseline_coverage,
        final_coverage=final_coverage,
        combinations_considered=considered,
        combinations_pruned=pruned,
        complete_sets_found=complete_found,
        valid_complementary_additions=tuple(cid for cid in selected_ids if cid not in baseline_ids),
        rejected_additions=tuple(
            candidate.citation_id
            for candidate in admitted
            if candidate.citation_id not in selected_ids
            and candidate.citation_id not in baseline_ids
        ),
        fallback_action=fallback,
        candidate_budget=CANDIDATE_BUDGET,
        citation_budget=MAX_TOTAL,
        candidate_admission_v4_required=use_candidate_admission_v4,
        claim_fallback_v4_required=use_fallback_v4,
    )
