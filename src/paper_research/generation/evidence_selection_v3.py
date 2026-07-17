"""Precision-constrained evidence selection candidate.

This module is intentionally independent from offline Gold, human labels,
question IDs, required-claim IDs, failure-taxonomy labels, and fixed relation
keys. Those signals are only allowed in evaluation scripts.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from enum import StrEnum

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

EVIDENCE_SELECTION_V3_VERSION = "evidence-selection-v3-candidate"
ELIGIBILITY_POLICY_VERSION = "evidence-eligibility-v1"
BASELINE_PROTECTION_VERSION = "baseline-protection-v1"
CANDIDATE_ADMISSION_VERSION = "candidate-admission-v2-candidate"
CLAIM_FALLBACK_VERSION = "claim-fallback-v2-candidate"
CANDIDATE_BUDGET = 12
HARD_VETO_RULE_COUNT = 13

_NEGATION = re.compile(r"\b(no|not|without|cannot|fails?|limitation|limited|except)\b", re.I)
_ADVANTAGE = re.compile(r"\b(improves?|better|superior|advantage|outperforms?|achieves?)\b", re.I)
_SETUP = re.compile(r"\b(trained|training|dataset|optimizer|batch|gpu|epochs?|steps?)\b", re.I)
_RESULT = re.compile(r"\b(score|accuracy|bleu|rouge|result|outperforms?|achieves?)\b", re.I)
_SURVEY = re.compile(r"\b(survey|overview|review|summarizes?)\b", re.I)


class CompatibilityStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    PARTIAL = "partial"
    NOT_REQUIRED = "not_required"


@dataclass(frozen=True)
class EvidenceEligibilityResult:
    eligible: bool
    hard_fail_reasons: tuple[str, ...]
    soft_scores: dict[str, float]
    matched_obligations: tuple[str, ...]
    unmatched_obligations: tuple[str, ...]
    numeric_status: CompatibilityStatus
    comparison_status: CompatibilityStatus
    entity_status: CompatibilityStatus
    polarity_status: CompatibilityStatus
    method_object_status: CompatibilityStatus
    candidate_text_hash: str
    policy_version: str = ELIGIBILITY_POLICY_VERSION


@dataclass(frozen=True)
class EvidenceSelectionV3Result:
    version: str
    eligibility_results: dict[str, EvidenceEligibilityResult]
    primary_citation_ids: tuple[str, ...]
    supporting_citation_ids: tuple[str, ...]
    rejected_citation_ids: tuple[str, ...]
    baseline_retained: bool
    baseline_replaced: bool
    replacement_reason: str | None
    fallback_action: FallbackAction
    narrowed_claim_text: str | None
    removed_obligations: tuple[str, ...]
    citation_cap_blocked: bool


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+(?:[.-][a-z0-9]+)*", text.lower())
        if len(token) > 2
    }


def _numbers(text: str) -> set[str]:
    return set(re.findall(r"\b\d+(?:\.\d+)?%?\b", text.lower()))


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def evaluate_evidence_eligibility(
    claim_text: str,
    candidate: CitationCandidate,
) -> EvidenceEligibilityResult:
    analysis = analyze_claim_obligations(claim_text)
    obligations = analysis.obligations
    coverages = {
        obligation.obligation_id: obligation_coverage(obligation, candidate.text)
        for obligation in obligations
    }
    matched = tuple(key for key, score in coverages.items() if score >= 0.35)
    unmatched = tuple(key for key, score in coverages.items() if score < 0.35)
    numeric = validate_numeric_evidence(claim_text, [candidate])
    comparison = validate_comparison_evidence(claim_text, [candidate])
    claim_tokens = _tokens(claim_text)
    evidence_tokens = _tokens(candidate.text)
    lexical = len(claim_tokens & evidence_tokens) / max(len(claim_tokens), 1)
    hard: list[str] = []
    if _numbers(claim_text) and not numeric.complete:
        hard.append("numeric_anchor_missing_or_conflicting")
    if comparison.required and not comparison.complete and len(comparison.covered) < 2:
        hard.append("comparison_one_side_only")
    if not matched:
        hard.append("no_required_obligation_covered")
    if lexical < 0.10:
        hard.append("lexical_overlap_without_entailment")
    if _NEGATION.search(claim_text) and _ADVANTAGE.search(candidate.text) and not _NEGATION.search(
        candidate.text
    ):
        hard.append("limitation_advantage_polarity_conflict")
    if _RESULT.search(claim_text) and _SETUP.search(candidate.text) and not _RESULT.search(
        candidate.text
    ):
        hard.append("result_setup_mismatch")
    if _SURVEY.search(candidate.text) and not _SURVEY.search(claim_text):
        hard.append("survey_vs_primary_source_confusion")
    if len(candidate.text.strip()) < 40:
        hard.append("source_span_too_short")
    return EvidenceEligibilityResult(
        eligible=not hard,
        hard_fail_reasons=tuple(dict.fromkeys(hard)),
        soft_scores={
            "lexical_overlap": lexical,
            "obligation_coverage": sum(coverages.values()) / max(len(coverages), 1),
            "retrieval_score": candidate.retrieval_score,
        },
        matched_obligations=matched,
        unmatched_obligations=unmatched,
        numeric_status=CompatibilityStatus.PASS
        if numeric.complete
        else CompatibilityStatus.FAIL,
        comparison_status=CompatibilityStatus.PASS
        if comparison.complete
        else CompatibilityStatus.PARTIAL
        if comparison.covered
        else CompatibilityStatus.FAIL,
        entity_status=CompatibilityStatus.PASS if lexical >= 0.10 else CompatibilityStatus.FAIL,
        polarity_status=CompatibilityStatus.FAIL
        if "limitation_advantage_polarity_conflict" in hard
        else CompatibilityStatus.PASS,
        method_object_status=CompatibilityStatus.FAIL
        if "result_setup_mismatch" in hard or "survey_vs_primary_source_confusion" in hard
        else CompatibilityStatus.PASS,
        candidate_text_hash=_hash_text(candidate.text),
    )


def _score(result: EvidenceEligibilityResult, candidate: CitationCandidate) -> tuple[float, ...]:
    return (
        float(result.eligible),
        result.soft_scores["obligation_coverage"],
        result.soft_scores["lexical_overlap"],
        1.0 if candidate.original_selected else 0.0,
        0.25 if candidate.adjacent_completion else 0.0,
        candidate.retrieval_score,
        -float(candidate.token_cost),
    )


def _admit_candidates(
    claim_text: str,
    candidates: tuple[CitationCandidate, ...],
) -> tuple[CitationCandidate, ...]:
    deduped: dict[tuple[str, int, str], CitationCandidate] = {}
    for candidate in candidates:
        existing = deduped.get(candidate.triple)
        if existing is None or candidate.retrieval_score > existing.retrieval_score:
            deduped[candidate.triple] = candidate
    ranked = sorted(
        deduped.values(),
        key=lambda candidate: (
            -max(
                (
                    obligation_coverage(obligation, candidate.text)
                    for obligation in analyze_claim_obligations(claim_text).obligations
                ),
                default=0.0,
            ),
            -candidate.retrieval_score,
            candidate.citation_id,
        ),
    )
    return tuple(ranked[:CANDIDATE_BUDGET])


def select_evidence_v3(
    claim_text: str,
    candidates: tuple[CitationCandidate, ...],
    baseline_citation_ids: tuple[str, ...] = (),
) -> EvidenceSelectionV3Result:
    admitted = _admit_candidates(claim_text, candidates)
    by_id = {candidate.citation_id: candidate for candidate in admitted}
    eligibility = {
        candidate.citation_id: evaluate_evidence_eligibility(claim_text, candidate)
        for candidate in admitted
    }
    eligible = [
        candidate
        for candidate in admitted
        if eligibility[candidate.citation_id].eligible
    ]
    baseline = [by_id[cid] for cid in baseline_citation_ids if cid in by_id]
    baseline_eligible = baseline and all(
        eligibility[candidate.citation_id].eligible for candidate in baseline
    )
    selected: list[CitationCandidate]
    baseline_retained = False
    baseline_replaced = False
    replacement_reason: str | None = None
    if baseline_eligible:
        selected = list(baseline[:MAX_TOTAL])
        baseline_retained = True
    else:
        ordered = sorted(
            eligible,
            key=lambda candidate: (
                tuple(-value for value in _score(eligibility[candidate.citation_id], candidate)),
                candidate.citation_id,
            ),
        )
        selected = ordered[:1]
        baseline_replaced = bool(baseline)
        replacement_reason = "baseline_ineligible_or_missing"
    analysis = analyze_claim_obligations(claim_text)
    covered = {
        obligation.obligation_id
        for obligation in analysis.obligations
        for candidate in selected
        if obligation_coverage(obligation, candidate.text) >= 0.35
    }
    for candidate in sorted(
        [candidate for candidate in eligible if candidate not in selected],
        key=lambda candidate: (
            tuple(-value for value in _score(eligibility[candidate.citation_id], candidate)),
            candidate.citation_id,
        ),
    ):
        if len(selected) >= MAX_TOTAL:
            break
        newly = {
            obligation.obligation_id
            for obligation in analysis.obligations
            if obligation.obligation_id not in covered
            and obligation_coverage(obligation, candidate.text) >= 0.35
        }
        if newly:
            selected.append(candidate)
            covered.update(newly)
    numeric = validate_numeric_evidence(claim_text, selected)
    comparison = validate_comparison_evidence(claim_text, selected)
    fallback, narrowed, removed = decide_claim_fallback(
        claim_text,
        analysis.obligations,
        covered,
        numeric,
        comparison,
    )
    if fallback == FallbackAction.UNSUPPORTED:
        selected = []
    selected_ids = tuple(candidate.citation_id for candidate in selected)
    return EvidenceSelectionV3Result(
        version=EVIDENCE_SELECTION_V3_VERSION,
        eligibility_results=eligibility,
        primary_citation_ids=selected_ids[:1],
        supporting_citation_ids=selected_ids[1:],
        rejected_citation_ids=tuple(
            candidate.citation_id
            for candidate in admitted
            if candidate.citation_id not in selected_ids
        ),
        baseline_retained=baseline_retained,
        baseline_replaced=baseline_replaced,
        replacement_reason=replacement_reason,
        fallback_action=fallback,
        narrowed_claim_text=narrowed,
        removed_obligations=removed,
        citation_cap_blocked=len(selected) >= MAX_TOTAL and bool(removed),
    )
