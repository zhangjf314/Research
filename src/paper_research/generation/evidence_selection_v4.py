"""Baseline-first evidence selection candidate.

The selector is intentionally limited to claim text, local candidate evidence,
local retrieval metadata, and the existing baseline citation IDs. Offline
evaluation may score against curated labels, but this module must not branch on
evaluation identifiers or adjudication labels.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from enum import StrEnum

from paper_research.generation.citation_selection import (
    MAX_TOTAL,
    CitationCandidate,
    FallbackAction,
    analyze_claim_obligations,
    obligation_coverage,
    validate_comparison_evidence,
    validate_numeric_evidence,
)

EVIDENCE_SELECTION_V4_VERSION = "evidence-selection-v4-candidate"
CANDIDATE_ADMISSIBILITY_VERSION = "candidate-admissibility-v1"
ROLE_ELIGIBILITY_VERSION = "evidence-role-eligibility-v1"
SET_SUFFICIENCY_VERSION = "set-sufficiency-v1"
BASELINE_FIRST_VERSION = "baseline-first-minimal-intervention-v1"
COMPLEMENT_ALLOCATION_VERSION = "complementary-evidence-allocation-v1"
REPLACEMENT_PROOF_VERSION = "replacement-proof-v1"
CANDIDATE_ADMISSION_V3_VERSION = "candidate-admission-v3-candidate"
CLAIM_FALLBACK_V3_VERSION = "claim-fallback-v3-candidate"
CANDIDATE_BUDGET = 12
CITATION_BUDGET = MAX_TOTAL

_NEGATION = re.compile(r"\b(no|not|without|cannot|fails?|limitation|limited|except)\b", re.I)
_ADVANTAGE = re.compile(r"\b(improves?|better|superior|advantage|outperforms?|achieves?)\b", re.I)
_SETUP = re.compile(r"\b(trained|training|dataset|optimizer|batch|gpu|epochs?|steps?)\b", re.I)
_RESULT = re.compile(r"\b(score|accuracy|bleu|rouge|result|outperforms?|achieves?)\b", re.I)
_SURVEY = re.compile(r"\b(survey|overview|review|summarizes?)\b", re.I)


class CandidateRole(StrEnum):
    STANDALONE_PRIMARY = "standalone_primary"
    SIDE_SPECIFIC_PRIMARY = "side_specific_primary"
    COMPLEMENTARY_SUPPORT = "complementary_support"
    INELIGIBLE = "ineligible"


class SelectionOperation(StrEnum):
    KEEP_BASELINE = "keep_baseline"
    ADD_COMPLEMENT = "add_complement"
    REPLACE_INELIGIBLE_BASELINE = "replace_ineligible_baseline"
    REPLACE_WITH_STRICTLY_BETTER_SET = "replace_with_strictly_better_set"
    REMOVE_REDUNDANT_SUPPORT = "remove_redundant_support"
    KEEP_AND_NARROW = "keep_and_narrow"
    KEEP_UNSUPPORTED = "keep_unsupported"


@dataclass(frozen=True)
class CandidateAdmissibility:
    admissible: bool
    hard_fail_reasons: tuple[str, ...]
    covered_obligations: tuple[str, ...]
    numeric_partial: bool
    comparison_partial: bool
    text_hash: str
    version: str = CANDIDATE_ADMISSIBILITY_VERSION


@dataclass(frozen=True)
class RoleEligibility:
    role: CandidateRole
    reasons: tuple[str, ...]
    version: str = ROLE_ELIGIBILITY_VERSION


@dataclass(frozen=True)
class SetSufficiency:
    complete: bool
    covered_obligations: tuple[str, ...]
    missing_obligations: tuple[str, ...]
    numeric_complete: bool
    comparison_complete: bool
    redundant_count: int
    version: str = SET_SUFFICIENCY_VERSION


@dataclass(frozen=True)
class ReplacementProof:
    passed: bool
    reason: str
    baseline_complete: bool
    proposed_complete: bool
    lost_obligations: tuple[str, ...]
    gained_obligations: tuple[str, ...]
    version: str = REPLACEMENT_PROOF_VERSION


@dataclass(frozen=True)
class EvidenceSelectionV4Result:
    version: str
    primary_citation_ids: tuple[str, ...]
    supporting_citation_ids: tuple[str, ...]
    rejected_citation_ids: tuple[str, ...]
    operations: tuple[SelectionOperation, ...]
    candidate_admissibility: dict[str, CandidateAdmissibility]
    role_eligibility: dict[str, RoleEligibility]
    baseline_sufficiency: SetSufficiency
    final_sufficiency: SetSufficiency
    replacement_proof: ReplacementProof
    baseline_retained: bool
    baseline_added_to: bool
    baseline_replaced: bool
    baseline_removed: bool
    valid_additions: tuple[str, ...]
    rejected_additions: tuple[str, ...]
    fallback_action: FallbackAction
    narrowed_claim_text: str | None
    removed_obligations: tuple[str, ...]
    citation_cap_blocked: bool
    candidate_budget: int
    citation_budget: int


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+(?:[.-][a-z0-9]+)*", text.lower())
        if len(token) > 2
    }


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _finite(value: float) -> float:
    return value if math.isfinite(value) else 0.0


def _dedupe_candidates(candidates: tuple[CitationCandidate, ...]) -> tuple[CitationCandidate, ...]:
    deduped: dict[tuple[str, int, str], CitationCandidate] = {}
    for candidate in candidates:
        existing = deduped.get(candidate.triple)
        if existing is None:
            deduped[candidate.triple] = candidate
            continue
        if (
            candidate.currently_cited,
            candidate.original_selected,
            _finite(candidate.retrieval_score),
            candidate.citation_id,
        ) > (
            existing.currently_cited,
            existing.original_selected,
            _finite(existing.retrieval_score),
            existing.citation_id,
        ):
            deduped[candidate.triple] = candidate
    return tuple(deduped.values())


def _candidate_rank(claim_text: str, candidate: CitationCandidate) -> tuple[float, ...]:
    analysis = analyze_claim_obligations(claim_text)
    coverages = [
        obligation_coverage(obligation, candidate.text)
        for obligation in analysis.obligations
    ]
    numeric = validate_numeric_evidence(claim_text, [candidate])
    comparison = validate_comparison_evidence(claim_text, [candidate])
    claim_tokens = _tokens(claim_text)
    evidence_tokens = _tokens(candidate.text)
    lexical = len(claim_tokens & evidence_tokens) / max(len(claim_tokens), 1)
    return (
        float(candidate.currently_cited),
        float(candidate.original_selected),
        sum(score >= 0.35 for score in coverages),
        sum(coverages),
        max(coverages, default=0.0),
        float(numeric.complete),
        len(comparison.covered) / max(len(comparison.required), 1),
        lexical,
        _finite(candidate.retrieval_score),
        -float(candidate.token_cost),
    )


def _anchor_fraction(claim_text: str, evidence_text: str) -> float:
    claim_tokens = {
        token
        for token in _tokens(claim_text)
        if token
        not in {
            "paper",
            "uses",
            "use",
            "used",
            "model",
            "method",
            "result",
            "shows",
            "based",
        }
    }
    evidence_tokens = _tokens(evidence_text)
    return len(claim_tokens & evidence_tokens) / max(len(claim_tokens), 1)


def admit_candidates_v3(
    claim_text: str,
    candidates: tuple[CitationCandidate, ...],
    baseline_citation_ids: tuple[str, ...] = (),
) -> tuple[CitationCandidate, ...]:
    """Keep a fixed-budget, obligation-diverse local candidate pool."""
    deduped = list(_dedupe_candidates(candidates))
    by_id = {candidate.citation_id: candidate for candidate in deduped}
    selected: list[CitationCandidate] = [
        by_id[cid] for cid in baseline_citation_ids if cid in by_id
    ]
    analysis = analyze_claim_obligations(claim_text)
    for obligation in analysis.obligations:
        if len(selected) >= CANDIDATE_BUDGET:
            break
        ranked = sorted(
            [
                candidate
                for candidate in deduped
                if candidate not in selected
                and obligation_coverage(obligation, candidate.text) >= 0.25
            ],
            key=lambda candidate: (
                -obligation_coverage(obligation, candidate.text),
                tuple(-value for value in _candidate_rank(claim_text, candidate)),
                candidate.citation_id,
            ),
        )
        if ranked:
            selected.append(ranked[0])
    for candidate in sorted(
        [candidate for candidate in deduped if candidate not in selected],
        key=lambda candidate: (
            tuple(-value for value in _candidate_rank(claim_text, candidate)),
            candidate.citation_id,
        ),
    ):
        if len(selected) >= CANDIDATE_BUDGET:
            break
        selected.append(candidate)
    return tuple(selected[:CANDIDATE_BUDGET])


def evaluate_candidate_admissibility(
    claim_text: str, candidate: CitationCandidate
) -> CandidateAdmissibility:
    analysis = analyze_claim_obligations(claim_text)
    covered = tuple(
        obligation.obligation_id
        for obligation in analysis.obligations
        if obligation_coverage(obligation, candidate.text) >= 0.30
    )
    numeric = validate_numeric_evidence(claim_text, [candidate])
    comparison = validate_comparison_evidence(claim_text, [candidate])
    claim_tokens = _tokens(claim_text)
    evidence_tokens = _tokens(candidate.text)
    lexical = len(claim_tokens & evidence_tokens) / max(len(claim_tokens), 1)
    hard: list[str] = []
    if not covered and not numeric.covered and not comparison.covered:
        hard.append("no_obligation_contribution")
    if lexical < 0.06 and not candidate.currently_cited:
        hard.append("entity_or_method_unanchored")
    if _NEGATION.search(claim_text) and _ADVANTAGE.search(candidate.text) and not _NEGATION.search(
        candidate.text
    ):
        hard.append("polarity_conflict")
    if _RESULT.search(claim_text) and _SETUP.search(candidate.text) and not _RESULT.search(
        candidate.text
    ):
        hard.append("method_object_conflict")
    if _SURVEY.search(candidate.text) and not _SURVEY.search(claim_text):
        hard.append("source_directness_conflict")
    if len(candidate.text.strip()) < 35:
        hard.append("source_span_too_short")
    return CandidateAdmissibility(
        admissible=not hard,
        hard_fail_reasons=tuple(dict.fromkeys(hard)),
        covered_obligations=covered,
        numeric_partial=bool(numeric.covered) and not numeric.complete,
        comparison_partial=bool(comparison.covered) and not comparison.complete,
        text_hash=_hash_text(candidate.text),
    )


def evaluate_role_eligibility(
    claim_text: str,
    candidate: CitationCandidate,
    admissibility: CandidateAdmissibility,
) -> RoleEligibility:
    if not admissibility.admissible:
        return RoleEligibility(CandidateRole.INELIGIBLE, admissibility.hard_fail_reasons)
    numeric = validate_numeric_evidence(claim_text, [candidate])
    comparison = validate_comparison_evidence(claim_text, [candidate])
    analysis = analyze_claim_obligations(claim_text)
    all_obligations = {obligation.obligation_id for obligation in analysis.obligations}
    covered = set(admissibility.covered_obligations)
    if covered == all_obligations and numeric.complete and comparison.complete:
        return RoleEligibility(CandidateRole.STANDALONE_PRIMARY, ("complete_single_evidence",))
    if comparison.covered and not comparison.complete:
        return RoleEligibility(CandidateRole.SIDE_SPECIFIC_PRIMARY, ("comparison_side_specific",))
    if numeric.covered and not numeric.complete:
        return RoleEligibility(CandidateRole.COMPLEMENTARY_SUPPORT, ("numeric_partial_support",))
    return RoleEligibility(CandidateRole.COMPLEMENTARY_SUPPORT, ("partial_obligation_support",))


def evaluate_set_sufficiency(
    claim_text: str,
    selected: tuple[CitationCandidate, ...],
) -> SetSufficiency:
    analysis = analyze_claim_obligations(claim_text)
    covered = {
        obligation.obligation_id
        for obligation in analysis.obligations
        for candidate in selected
        if obligation_coverage(obligation, candidate.text) >= 0.35
    }
    all_obligations = {obligation.obligation_id for obligation in analysis.obligations}
    numeric = validate_numeric_evidence(claim_text, selected)
    comparison = validate_comparison_evidence(claim_text, selected)
    redundancy_groups = [
        candidate.redundancy_group
        for candidate in selected
        if candidate.redundancy_group is not None
    ]
    redundant_count = len(redundancy_groups) - len(set(redundancy_groups))
    return SetSufficiency(
        complete=all_obligations.issubset(covered) and numeric.complete and comparison.complete,
        covered_obligations=tuple(sorted(covered)),
        missing_obligations=(
            tuple(sorted(all_obligations - covered)) + numeric.missing + comparison.missing
        ),
        numeric_complete=numeric.complete,
        comparison_complete=comparison.complete,
        redundant_count=redundant_count,
    )


def prove_replacement(
    claim_text: str,
    baseline: tuple[CitationCandidate, ...],
    proposed: tuple[CitationCandidate, ...],
    *,
    allow_baseline_protection: bool = True,
) -> ReplacementProof:
    baseline_sufficiency = evaluate_set_sufficiency(claim_text, baseline)
    proposed_sufficiency = evaluate_set_sufficiency(claim_text, proposed)
    baseline_covered = set(baseline_sufficiency.covered_obligations)
    proposed_covered = set(proposed_sufficiency.covered_obligations)
    lost = tuple(sorted(baseline_covered - proposed_covered))
    gained = tuple(sorted(proposed_covered - baseline_covered))
    if not baseline:
        return ReplacementProof(
            True,
            "missing_baseline",
            False,
            proposed_sufficiency.complete,
            (),
            gained,
        )
    if allow_baseline_protection and baseline_sufficiency.complete:
        return ReplacementProof(
            False,
            "baseline_complete_no_replacement_allowed",
            True,
            proposed_sufficiency.complete,
            lost,
            gained,
        )
    if lost:
        return ReplacementProof(
            False,
            "replacement_loses_obligation",
            baseline_sufficiency.complete,
            proposed_sufficiency.complete,
            lost,
            gained,
        )
    if proposed_sufficiency.complete and not baseline_sufficiency.complete:
        return ReplacementProof(
            True,
            "strict_set_completion_gain",
            False,
            True,
            (),
            gained,
        )
    if len(gained) > 0 and len(proposed) <= min(MAX_TOTAL, len(baseline) + 1):
        return ReplacementProof(
            True,
            "strict_obligation_gain_without_loss",
            baseline_sufficiency.complete,
            proposed_sufficiency.complete,
            (),
            gained,
        )
    return ReplacementProof(
        False,
        "no_strict_net_gain",
        baseline_sufficiency.complete,
        proposed_sufficiency.complete,
        lost,
        gained,
    )


def _fallback_from_set(
    claim_text: str,
    sufficiency: SetSufficiency,
) -> tuple[FallbackAction, str | None, tuple[str, ...]]:
    if sufficiency.complete:
        return FallbackAction.ANSWERED_ORIGINAL, None, ()
    if sufficiency.covered_obligations:
        return FallbackAction.ANSWERED_NARROWED, None, sufficiency.missing_obligations
    return FallbackAction.UNSUPPORTED, None, sufficiency.missing_obligations


def select_evidence_v4(
    claim_text: str,
    candidates: tuple[CitationCandidate, ...],
    baseline_citation_ids: tuple[str, ...] = (),
    *,
    add_complements: bool = True,
    allow_replacement: bool = True,
    allow_baseline_protection: bool = True,
    use_candidate_admission_v3: bool = True,
    use_claim_fallback_v3: bool = False,
    use_old_candidate_veto: bool = False,
) -> EvidenceSelectionV4Result:
    admitted = (
        admit_candidates_v3(claim_text, candidates, baseline_citation_ids)
        if use_candidate_admission_v3
        else _dedupe_candidates(candidates)[:CANDIDATE_BUDGET]
    )
    by_id = {candidate.citation_id: candidate for candidate in admitted}
    baseline = tuple(by_id[cid] for cid in baseline_citation_ids if cid in by_id)
    admissibility = {
        candidate.citation_id: evaluate_candidate_admissibility(claim_text, candidate)
        for candidate in admitted
    }
    roles = {
        candidate.citation_id: evaluate_role_eligibility(
            claim_text, candidate, admissibility[candidate.citation_id]
        )
        for candidate in admitted
    }
    if use_old_candidate_veto:
        selectable = [
            candidate
            for candidate in admitted
            if roles[candidate.citation_id].role == CandidateRole.STANDALONE_PRIMARY
        ]
    else:
        selectable = [
            candidate
            for candidate in admitted
            if roles[candidate.citation_id].role != CandidateRole.INELIGIBLE
        ]
    selected = list(baseline[:MAX_TOTAL])
    baseline_sufficiency = evaluate_set_sufficiency(claim_text, tuple(selected))
    operations: list[SelectionOperation] = [SelectionOperation.KEEP_BASELINE]
    valid_additions: list[str] = []
    rejected_additions: list[str] = []
    if add_complements and selected and len(selected) < MAX_TOTAL:
        selected_ids = {candidate.citation_id for candidate in selected}
        before = evaluate_set_sufficiency(claim_text, tuple(selected))
        best_baseline_anchor = max(
            (_anchor_fraction(claim_text, candidate.text) for candidate in selected),
            default=0.0,
        )
        best_baseline_score = max(
            (_finite(candidate.retrieval_score) for candidate in selected),
            default=0.0,
        )
        for candidate in sorted(
            [candidate for candidate in selectable if candidate.citation_id not in selected_ids],
            key=lambda candidate: (
                tuple(-value for value in _candidate_rank(claim_text, candidate)),
                candidate.citation_id,
            ),
        ):
            if len(selected) >= MAX_TOTAL:
                break
            trial = (*selected, candidate)
            after = evaluate_set_sufficiency(claim_text, trial)
            gained = set(after.covered_obligations) - set(before.covered_obligations)
            completes_missing = len(after.missing_obligations) < len(before.missing_obligations)
            candidate_anchor = _anchor_fraction(claim_text, candidate.text)
            direct_support_gain = bool(selected) and (
                candidate_anchor >= best_baseline_anchor + 0.15
                or (
                    candidate_anchor >= best_baseline_anchor
                    and _finite(candidate.retrieval_score) >= max(best_baseline_score * 3.0, 0.75)
                )
            )
            numeric = validate_numeric_evidence(claim_text, [candidate])
            if numeric.required and not numeric.complete and not selected:
                direct_support_gain = False
            if (
                gained
                or completes_missing
                or (after.complete and not before.complete)
                or direct_support_gain
            ):
                selected.append(candidate)
                selected_ids.add(candidate.citation_id)
                valid_additions.append(candidate.citation_id)
                operations.append(SelectionOperation.ADD_COMPLEMENT)
                before = after
                best_baseline_anchor = max(best_baseline_anchor, candidate_anchor)
                best_baseline_score = max(best_baseline_score, _finite(candidate.retrieval_score))
            else:
                rejected_additions.append(candidate.citation_id)
    replacement = ReplacementProof(
        False,
        "replacement_not_attempted",
        baseline_sufficiency.complete,
        evaluate_set_sufficiency(claim_text, tuple(selected)).complete,
        (),
        (),
    )
    if allow_replacement and baseline and not set(baseline).issubset(set(selected)):
        selected_tuple = tuple(selected)
        replacement = prove_replacement(
            claim_text,
            baseline,
            selected_tuple,
            allow_baseline_protection=allow_baseline_protection,
        )
        if not replacement.passed:
            selected = list(baseline[:MAX_TOTAL])
            valid_additions = []
            operations = [SelectionOperation.KEEP_BASELINE]
        elif set(candidate.citation_id for candidate in selected) != set(baseline_citation_ids):
            operations.append(SelectionOperation.REPLACE_WITH_STRICTLY_BETTER_SET)
    final_sufficiency = evaluate_set_sufficiency(claim_text, tuple(selected))
    fallback, narrowed, removed = _fallback_from_set(claim_text, final_sufficiency)
    if not use_claim_fallback_v3:
        fallback = FallbackAction.ANSWERED_ORIGINAL if selected else FallbackAction.UNSUPPORTED
        narrowed = None
        removed = ()
    if fallback == FallbackAction.UNSUPPORTED:
        operations.append(SelectionOperation.KEEP_UNSUPPORTED)
    selected_ids = tuple(candidate.citation_id for candidate in selected)
    baseline_id_set = set(baseline_citation_ids)
    selected_id_set = set(selected_ids)
    rejected = tuple(
        candidate.citation_id
        for candidate in admitted
        if candidate.citation_id not in selected_id_set
    )
    return EvidenceSelectionV4Result(
        version=EVIDENCE_SELECTION_V4_VERSION,
        primary_citation_ids=selected_ids[:1],
        supporting_citation_ids=selected_ids[1:],
        rejected_citation_ids=rejected,
        operations=tuple(dict.fromkeys(operations)),
        candidate_admissibility=admissibility,
        role_eligibility=roles,
        baseline_sufficiency=baseline_sufficiency,
        final_sufficiency=final_sufficiency,
        replacement_proof=replacement,
        baseline_retained=bool(baseline_id_set & selected_id_set) or not baseline_id_set,
        baseline_added_to=bool(baseline_id_set) and baseline_id_set < selected_id_set,
        baseline_replaced=bool(baseline_id_set) and not bool(baseline_id_set & selected_id_set),
        baseline_removed=bool(baseline_id_set - selected_id_set),
        valid_additions=tuple(valid_additions),
        rejected_additions=tuple(rejected_additions),
        fallback_action=fallback,
        narrowed_claim_text=narrowed,
        removed_obligations=tuple(removed),
        citation_cap_blocked=len(selected_ids) >= MAX_TOTAL and not final_sufficiency.complete,
        candidate_budget=CANDIDATE_BUDGET,
        citation_budget=CITATION_BUDGET,
    )
