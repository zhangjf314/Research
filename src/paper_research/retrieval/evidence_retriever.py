from __future__ import annotations

import math
import re
from collections import Counter

from pydantic import BaseModel, Field

from paper_research.evidence.claims import ClaimUnit
from paper_research.evidence.schema import EvidenceUnit
from paper_research.retrieval.query_router import RoutingDecision

TERM_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.%-]+")


class EvidenceScoreComponents(BaseModel):
    query_relevance: float
    claim_term_coverage: float
    evidence_role_compatibility: float
    section_compatibility: float
    paper_filter_validity: float
    numeric_fact_compatibility: float
    comparison_dimension_coverage: float
    answerability_compatibility: float
    metadata_penalty: float
    citation_only_penalty: float
    duplication_penalty: float


class EvidenceCandidate(BaseModel):
    claim_id: str
    evidence: EvidenceUnit
    original_retrieval_rank: int | None = None
    dense_score: float | None = None
    lexical_score: float | None = None
    structural_score: float | None = None
    total_score: float
    score_components: EvidenceScoreComponents
    filter_reasons: list[str] = Field(default_factory=list)
    rejection_reasons: list[str] = Field(default_factory=list)


def _terms(text: str) -> Counter[str]:
    return Counter(term.casefold() for term in TERM_RE.findall(text))


def _coverage(wanted: list[str], text_terms: Counter[str]) -> float:
    if not wanted:
        return 0.0
    return sum(term.casefold() in text_terms for term in wanted) / len(wanted)


class EvidenceRetriever:
    def __init__(self, units: list[EvidenceUnit]) -> None:
        self.units = list(units)

    def score_candidates(
        self,
        claim: ClaimUnit,
        decision: RoutingDecision,
        *,
        source_scores: dict[str, dict[str, float | int | None]] | None = None,
    ) -> list[EvidenceCandidate]:
        source_scores = source_scores or {}
        target_papers = set(decision.retrieval_filter.paper_ids or [])
        candidates = []
        seen_text: set[tuple[str, str]] = set()
        for unit in self.units:
            if target_papers and unit.paper_id not in target_papers:
                continue
            source = source_scores.get(unit.source_chunk_id or "", {})
            text_terms = _terms(unit.normalized_text)
            claim_coverage = _coverage(claim.target_terms, text_terms)
            query_relevance = float(source.get("fused_score") or 0.0)
            if not source:
                query_relevance = claim_coverage * 0.5
            wanted_roles = set(
                claim.required_evidence_roles or decision.profile.evidence_role_filters
            )
            role_compatibility = (
                len(wanted_roles & set(unit.evidence_roles)) / len(wanted_roles)
                if wanted_roles
                else 0.5
            )
            section = (unit.section_title or "").casefold()
            section_score = sum(
                boost
                for term, boost in decision.profile.section_title_boosts.items()
                if term in section
            )
            numeric = 1.0 if claim.result_terms and unit.numeric_facts else 0.0
            comparison = _coverage(claim.comparison_dimensions, text_terms)
            metadata_penalty = 1.0 if "metadata" in unit.evidence_roles else 0.0
            citation_penalty = 1.0 if "citation_only" in unit.evidence_roles else 0.0
            duplicate_key = (unit.paper_id, unit.normalized_text)
            duplicate_penalty = 1.0 if duplicate_key in seen_text else 0.0
            seen_text.add(duplicate_key)
            components = EvidenceScoreComponents(
                query_relevance=query_relevance,
                claim_term_coverage=claim_coverage,
                evidence_role_compatibility=role_compatibility,
                section_compatibility=min(1.0, section_score),
                paper_filter_validity=1.0,
                numeric_fact_compatibility=numeric,
                comparison_dimension_coverage=comparison,
                answerability_compatibility=1.0 if claim.expected_answerability else 0.25,
                metadata_penalty=metadata_penalty,
                citation_only_penalty=citation_penalty,
                duplication_penalty=duplicate_penalty,
            )
            total = (
                0.30 * query_relevance
                + 0.28 * claim_coverage
                + 0.18 * role_compatibility
                + 0.08 * components.section_compatibility
                + 0.06 * numeric
                + 0.05 * comparison
                + 0.05 * components.answerability_compatibility
                - 0.45 * metadata_penalty
                - 0.45 * citation_penalty
                - 0.20 * duplicate_penalty
            )
            rejection = []
            if not unit.eligible_for_final_context:
                rejection.append("default_non_evidence_filter")
            if not math.isfinite(total):
                rejection.append("non_finite_score")
                total = -1.0
            candidates.append(
                EvidenceCandidate(
                    claim_id=claim.claim_id,
                    evidence=unit,
                    original_retrieval_rank=source.get("rank"),  # type: ignore[arg-type]
                    dense_score=source.get("dense_score"),  # type: ignore[arg-type]
                    lexical_score=source.get("lexical_score"),  # type: ignore[arg-type]
                    structural_score=source.get("structural_score"),  # type: ignore[arg-type]
                    total_score=round(total, 8),
                    score_components=components,
                    filter_reasons=["paper_filter_valid"],
                    rejection_reasons=rejection,
                )
            )
        return sorted(
            candidates,
            key=lambda item: (-item.total_score, item.evidence.paper_id, item.evidence.ordinal),
        )
