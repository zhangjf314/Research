"""Deterministic, evidence-only citation selection for required-claim QA."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

CITATION_SELECTION_VERSION = "citation-selection-v1"
OBLIGATION_POLICY_VERSION = "claim-obligation-coverage-v1"
NUMERIC_VALIDATION_VERSION = "numeric-evidence-completeness-v1"
COMPARISON_VALIDATION_VERSION = "comparison-evidence-completeness-v1"
CITATION_BUDGET_VERSION = "citation-budget-v1"
EVIDENCE_ORIGIN_POLICY_VERSION = "citation-evidence-origin-policy-v1"

MAX_PRIMARY = 1
MAX_SUPPORTING = 2
MAX_TOTAL = 3

_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
    "as", "is", "are", "was", "were", "be", "been", "this", "that", "from",
    "by", "uses", "use", "using", "paper", "work", "model",
}
_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12",
}


class FallbackAction(StrEnum):
    ANSWERED_ORIGINAL = "answered_original"
    ANSWERED_NARROWED = "answered_narrowed"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class ClaimObligation:
    obligation_id: str
    obligation_text: str
    obligation_type: str
    required: bool
    lexical_anchors: tuple[str, ...]
    numeric_anchors: tuple[str, ...]
    comparison_side: str | None
    decomposition_confidence: float


@dataclass(frozen=True)
class ClaimObligationAnalysis:
    obligations: tuple[ClaimObligation, ...]
    needs_full_claim_support: bool
    decomposition_confidence: float


@dataclass(frozen=True)
class CitationCandidate:
    citation_id: str
    paper_id: str
    page: int
    block_id: str
    text: str
    neighboring_context: str = ""
    evidence_role: tuple[str, ...] = ()
    retrieval_origin: str = "original_selected"
    original_selected: bool = True
    adjacent_completion: bool = False
    currently_cited: bool = False
    retrieval_score: float = 0.0
    lexical_alignment: float = 0.0
    numeric_coverage: float = 0.0
    comparison_side_coverage: float = 0.0
    claim_scope_coverage: float = 0.0
    redundancy_group: str | None = None
    token_cost: int = 0

    @property
    def triple(self) -> tuple[str, int, str]:
        return self.paper_id, self.page, self.block_id


@dataclass(frozen=True)
class EvidenceValidation:
    complete: bool
    required: tuple[str, ...]
    covered: tuple[str, ...]
    missing: tuple[str, ...]
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class CitationSelectionResult:
    primary_citation_ids: tuple[str, ...]
    supporting_citation_ids: tuple[str, ...]
    rejected_citation_ids: tuple[str, ...]
    uncovered_requirements: tuple[str, ...]
    completeness_status: str
    decision_reasons: tuple[str, ...]
    selected_count: int
    citation_budget: dict[str, int | bool | str]
    fallback_action: FallbackAction
    narrowed_claim_text: str | None = None
    removed_obligations: tuple[str, ...] = ()


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(
        token
        for token in re.findall(r"[a-z0-9]+(?:[.-][a-z0-9]+)*", text.lower())
        if len(token) > 1 and token not in _STOPWORDS
    )


def _numeric_anchors(text: str) -> tuple[str, ...]:
    normalized = text.lower().replace("×", "x")
    for word, digit in _NUMBER_WORDS.items():
        normalized = re.sub(rf"\b{word}\b", digit, normalized)
    values = re.findall(
        r"\b\d+(?:\.\d+)?(?:e[+-]?\d+)?%?\b|\b\d+(?:\.\d+)?[mbk]\b",
        normalized,
    )
    return tuple(dict.fromkeys(values))


def _obligation(
    index: int,
    text: str,
    kind: str,
    confidence: float,
    side: str | None = None,
) -> ClaimObligation:
    return ClaimObligation(
        obligation_id=f"ob-{index:02d}",
        obligation_text=text.strip(),
        obligation_type=kind,
        required=True,
        lexical_anchors=tuple(dict.fromkeys(_tokens(text))),
        numeric_anchors=_numeric_anchors(text),
        comparison_side=side,
        decomposition_confidence=confidence,
    )


def analyze_claim_obligations(claim_text: str) -> ClaimObligationAnalysis:
    """Extract support obligations without generating new semantic content."""
    text = " ".join(claim_text.split()).strip()
    if not text:
        return ClaimObligationAnalysis((), True, 0.0)
    parts: list[tuple[str, str, str | None]] = []
    comparison = re.split(
        r"\b(?:while|whereas|compared with|in contrast to|unlike|rather than)\b",
        text,
        maxsplit=1,
        flags=re.I,
    )
    if len(comparison) == 2 and all(part.strip() for part in comparison):
        parts = [
            (comparison[0], "comparison_side", "side_a"),
            (comparison[1], "comparison_side", "side_b"),
        ]
    else:
        protected_and = bool(
            re.search(
                r"\b(?:research and development|trial and error|encoder and decoder)\b",
                text,
                re.I,
            )
        )
        clauses = (
            [text]
            if protected_and
            else [
                part.strip(" ,;.")
                for part in re.split(r"\s*;\s*|\s+and\s+", text, flags=re.I)
                if part.strip(" ,;.")
            ]
        )
        if len(clauses) > 1:
            parts = [(part, "parallel_clause", None) for part in clauses]
        else:
            parts = [(text, "atomic", None)]
    expanded: list[tuple[str, str, str | None]] = []
    for part, kind, side in parts:
        expanded.append((part, kind, side))
        if re.search(r"\b(?:schedule|optimizer)\b", part, re.I) and re.search(
            r"\bwarmup\b", part, re.I
        ):
            warmup = re.search(r"\bwarmup(?:\s+\w+){0,3}", part, re.I)
            if warmup and warmup.group(0).lower() != part.lower():
                expanded.append((warmup.group(0), "training_configuration", side))
    deduped: list[tuple[str, str, str | None]] = []
    seen: set[str] = set()
    for part in expanded:
        key = part[0].lower()
        if key not in seen:
            seen.add(key)
            deduped.append(part)
    confidence = 0.9 if len(deduped) > 1 else 0.55
    obligations = tuple(
        _obligation(index, part, kind, confidence, side)
        for index, (part, kind, side) in enumerate(deduped, 1)
    )
    return ClaimObligationAnalysis(
        obligations=obligations,
        needs_full_claim_support=len(obligations) <= 1,
        decomposition_confidence=confidence,
    )


def obligation_coverage(
    obligation: ClaimObligation, evidence_text: str
) -> float:
    evidence_tokens = set(_tokens(evidence_text))
    anchors = set(obligation.lexical_anchors)
    lexical = len(anchors & evidence_tokens) / len(anchors) if anchors else 0.0
    numeric = _numeric_anchors(evidence_text)
    numeric_score = (
        len(set(obligation.numeric_anchors) & set(numeric))
        / len(obligation.numeric_anchors)
        if obligation.numeric_anchors
        else 1.0
    )
    return 0.75 * lexical + 0.25 * numeric_score


def validate_numeric_evidence(
    claim_text: str, selected_evidence: Iterable[CitationCandidate]
) -> EvidenceValidation:
    required = _numeric_anchors(claim_text)
    if not required:
        return EvidenceValidation(True, (), (), ())
    formal_text = " ".join(candidate.text for candidate in selected_evidence)
    covered_values = set(_numeric_anchors(formal_text))
    covered = tuple(value for value in required if value in covered_values)
    missing = tuple(value for value in required if value not in covered_values)
    return EvidenceValidation(
        complete=not missing,
        required=required,
        covered=covered,
        missing=missing,
        reasons=(() if not missing else ("numeric_anchor_missing_from_citation_block",)),
    )


def _comparison_sides(claim_text: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    analysis = analyze_claim_obligations(claim_text)
    side_a = tuple(
        anchor
        for obligation in analysis.obligations
        if obligation.comparison_side == "side_a"
        for anchor in obligation.lexical_anchors
    )
    side_b = tuple(
        anchor
        for obligation in analysis.obligations
        if obligation.comparison_side == "side_b"
        for anchor in obligation.lexical_anchors
    )
    has_first_and_second = (
        re.search(r"\bfirst\b", claim_text, re.I)
        and re.search(r"\bsecond\b", claim_text, re.I)
    )
    if not side_a and not side_b and (
        has_first_and_second
        or re.search(r"\b(?:both|comparison|versus|better than)\b", claim_text, re.I)
    ):
        tokens = _tokens(claim_text)
        midpoint = max(1, len(tokens) // 2)
        side_a, side_b = tokens[:midpoint], tokens[midpoint:]
    return side_a, side_b


def validate_comparison_evidence(
    claim_text: str, selected_evidence: Iterable[CitationCandidate]
) -> EvidenceValidation:
    side_a, side_b = _comparison_sides(claim_text)
    if not side_a and not side_b:
        return EvidenceValidation(True, (), (), ())
    selected = list(selected_evidence)
    evidence_tokens = set(
        token
        for candidate in selected
        for token in _tokens(candidate.text)
    )
    required = ("side_a", "side_b")
    covered = []
    if side_a and set(side_a) & evidence_tokens:
        covered.append("side_a")
    if side_b and set(side_b) & evidence_tokens:
        covered.append("side_b")
    if len(covered) == 2 and len(selected) == 1:
        explicit_predicate = re.search(
            r"\b(?:while|whereas|unlike|rather than|compared with|in contrast|"
            r"better than|both)\b",
            selected[0].text,
            re.I,
        )
        if not explicit_predicate:
            covered.remove("side_b")
    missing = tuple(side for side in required if side not in covered)
    return EvidenceValidation(
        complete=not missing,
        required=required,
        covered=tuple(covered),
        missing=missing,
        reasons=(() if not missing else ("comparison_side_missing",)),
    )


def _candidate_score(
    claim_text: str,
    candidate: CitationCandidate,
    obligations: tuple[ClaimObligation, ...],
) -> tuple[float, ...]:
    coverages = [obligation_coverage(obligation, candidate.text) for obligation in obligations]
    claim_tokens = set(_tokens(claim_text))
    evidence_tokens = set(_tokens(candidate.text))
    lexical = len(claim_tokens & evidence_tokens) / max(len(claim_tokens), 1)
    numeric = validate_numeric_evidence(claim_text, [candidate]).complete
    side = validate_comparison_evidence(claim_text, [candidate])
    origin = 1.0 if candidate.original_selected else 0.0
    role = 1.0 if set(candidate.evidence_role) & {
        "method", "result", "setup", "comparison", "limitation", "definition"
    } else 0.0
    return (
        sum(score >= 0.35 for score in coverages),
        max(coverages, default=0.0),
        float(numeric),
        len(side.covered) / max(len(side.required), 1),
        lexical,
        origin,
        role,
        1.0 if candidate.currently_cited else 0.0,
        candidate.retrieval_score if math.isfinite(candidate.retrieval_score) else 0.0,
        -len(candidate.text),
    )


def select_citations(
    claim_text: str,
    candidates: Iterable[CitationCandidate],
) -> CitationSelectionResult:
    analysis = analyze_claim_obligations(claim_text)
    obligations = analysis.obligations
    deduped: dict[tuple[str, int, str], CitationCandidate] = {}
    for candidate in candidates:
        existing = deduped.get(candidate.triple)
        if existing is None or (
            candidate.original_selected and not existing.original_selected
        ):
            deduped[candidate.triple] = candidate
    ordered = sorted(
        deduped.values(),
        key=lambda candidate: (
            tuple(-value for value in _candidate_score(claim_text, candidate, obligations)),
            candidate.citation_id,
        ),
    )
    if not ordered or not obligations:
        return CitationSelectionResult(
            (), (), tuple(candidate.citation_id for candidate in ordered), (),
            "incomplete", ("no_usable_candidate_or_empty_claim",), 0,
            citation_budget(), FallbackAction.UNSUPPORTED,
        )
    primary = ordered[0]
    selected = [primary]
    covered = {
        obligation.obligation_id
        for obligation in obligations
        if obligation_coverage(obligation, primary.text) >= 0.35
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
        if newly_covered:
            selected.append(candidate)
            covered.update(newly_covered)
    numeric = validate_numeric_evidence(claim_text, selected)
    comparison = validate_comparison_evidence(claim_text, selected)
    uncovered = tuple(
        obligation.obligation_id
        for obligation in obligations
        if obligation.obligation_id not in covered
    )
    complete = not uncovered and numeric.complete and comparison.complete
    fallback, narrowed, removed = decide_claim_fallback(
        claim_text, obligations, covered, numeric, comparison
    )
    reasons = [
        "primary_ranked_by_obligation_numeric_comparison_origin",
        "supporting_added_only_for_new_obligation",
    ]
    reasons.extend(numeric.reasons)
    reasons.extend(comparison.reasons)
    rejected = tuple(
        candidate.citation_id for candidate in ordered if candidate not in selected
    )
    return CitationSelectionResult(
        primary_citation_ids=(primary.citation_id,),
        supporting_citation_ids=tuple(candidate.citation_id for candidate in selected[1:]),
        rejected_citation_ids=rejected,
        uncovered_requirements=uncovered + numeric.missing + comparison.missing,
        completeness_status="complete" if complete else "incomplete",
        decision_reasons=tuple(reasons),
        selected_count=len(selected),
        citation_budget=citation_budget(),
        fallback_action=fallback,
        narrowed_claim_text=narrowed,
        removed_obligations=removed,
    )


def decide_claim_fallback(
    original_claim: str,
    obligations: tuple[ClaimObligation, ...],
    covered_obligation_ids: set[str],
    numeric_validation: EvidenceValidation,
    comparison_validation: EvidenceValidation,
) -> tuple[FallbackAction, str | None, tuple[str, ...]]:
    all_ids = {obligation.obligation_id for obligation in obligations}
    missing = all_ids - covered_obligation_ids
    if not missing and numeric_validation.complete and comparison_validation.complete:
        return FallbackAction.ANSWERED_ORIGINAL, None, ()
    if not comparison_validation.complete and len(obligations) > 1:
        supported_side = next(
            (
                obligation
                for obligation in obligations
                if obligation.obligation_id in covered_obligation_ids
                and not obligation.numeric_anchors
            ),
            None,
        )
        if supported_side is not None:
            return (
                FallbackAction.ANSWERED_NARROWED,
                supported_side.obligation_text,
                tuple(
                    obligation.obligation_id
                    for obligation in obligations
                    if obligation.obligation_id != supported_side.obligation_id
                ),
            )
    covered = [
        obligation
        for obligation in obligations
        if obligation.obligation_id in covered_obligation_ids
        and not obligation.numeric_anchors
    ]
    if covered and len(covered) < len(obligations):
        narrowed = "; ".join(obligation.obligation_text for obligation in covered)
        return (
            FallbackAction.ANSWERED_NARROWED,
            narrowed,
            tuple(
                obligation.obligation_id
                for obligation in obligations
                if obligation.obligation_id not in covered_obligation_ids
                or obligation.numeric_anchors
            ),
        )
    return FallbackAction.UNSUPPORTED, None, tuple(sorted(missing))


def citation_budget() -> dict[str, int | bool | str]:
    return {
        "version": CITATION_BUDGET_VERSION,
        "max_primary": MAX_PRIMARY,
        "max_supporting": MAX_SUPPORTING,
        "max_total": MAX_TOTAL,
        "allow_zero_for_unsupported": True,
    }
