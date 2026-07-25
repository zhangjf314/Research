"""Deterministic quality gates for synthesized Deep Research reports."""

from __future__ import annotations

import re
from itertools import combinations
from typing import Any

from pydantic import BaseModel, Field


class ReportQualityMetrics(BaseModel):
    exact_duplicate_paragraph_count: int = 0
    normalized_duplicate_bullet_count: int = 0
    duplicate_reference_count: int = 0
    cross_section_similarity: float = 0.0
    section_pair_similarity: dict[str, float] = Field(default_factory=dict)
    unique_evidence_count: int = 0
    section_evidence_counts: dict[str, int] = Field(default_factory=dict)
    citation_id_validity: float = 1.0
    citation_context_validity: float = 1.0
    citation_page_validity: float = 1.0
    raw_quote_copy_ratio: float = 0.0
    passed: bool = True
    failures: list[str] = Field(default_factory=list)


_CITATION_RE = re.compile(r"\bE\d{2,3}\b|\[E\d+\]")


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def normalized_tokens(value: str) -> set[str]:
    without_citations = re.sub(r"\b[e]\d{2,3}\b|\[e\d+\]", " ", value.lower())
    return set(re.findall(r"[\w\u4e00-\u9fff]+", without_citations))


def jaccard_similarity(left: str, right: str) -> float:
    left_tokens = normalized_tokens(left)
    right_tokens = normalized_tokens(right)
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def evaluate_report_quality(
    report: str,
    *,
    sections: dict[str, str],
    evidence_catalog: dict[str, dict[str, Any]],
    section_evidence_ids: dict[str, list[str]],
    similarity_threshold: float = 0.80,
) -> ReportQualityMetrics:
    paragraphs = []
    for part in re.split(r"\n\s*\n", report):
        body = "\n".join(
            line for line in part.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
        normalized = normalize_text(body)
        if normalized:
            paragraphs.append(normalized)
    paragraph_counts: dict[str, int] = {}
    for paragraph in paragraphs:
        paragraph_counts[paragraph] = paragraph_counts.get(paragraph, 0) + 1
    duplicate_paragraphs = sum(count - 1 for count in paragraph_counts.values() if count > 1)

    bullets = [
        normalize_text(line[1:].strip())
        for line in report.splitlines()
        if line.strip().startswith("- ")
    ]
    bullet_counts: dict[str, int] = {}
    for bullet in bullets:
        bullet_counts[bullet] = bullet_counts.get(bullet, 0) + 1
    duplicate_bullets = sum(count - 1 for count in bullet_counts.values() if count > 1)

    references_started = False
    references: list[str] = []
    for line in report.splitlines():
        if line.startswith("## 9."):
            references_started = True
            continue
        if references_started and line.strip().startswith("- "):
            references.append(normalize_text(line))
    duplicate_references = len(references) - len(set(references))

    pair_similarity: dict[str, float] = {}
    max_similarity = 0.0
    for left, right in combinations(sections, 2):
        score = round(jaccard_similarity(sections[left], sections[right]), 6)
        pair_similarity[f"{left}:{right}"] = score
        max_similarity = max(max_similarity, score)

    allowed_citations = set(evidence_catalog)
    used_citations = set(_CITATION_RE.findall(report))
    unknown = used_citations - allowed_citations
    citation_validity = 1.0 if not unknown else 0.0

    section_counts = {
        section_id: len(dict.fromkeys(ids))
        for section_id, ids in section_evidence_ids.items()
    }
    raw_quote_copy_ratio = _raw_quote_copy_ratio(report, evidence_catalog)

    failures: list[str] = []
    if duplicate_paragraphs:
        failures.append("exact_duplicate_paragraph")
    if duplicate_bullets:
        failures.append("normalized_duplicate_bullet")
    if duplicate_references:
        failures.append("duplicate_reference")
    if citation_validity < 1.0:
        failures.append("unknown_citation_id")
    if max_similarity >= similarity_threshold:
        failures.append("cross_section_similarity")
    if raw_quote_copy_ratio > 0.80:
        failures.append("raw_quote_copy_ratio")

    return ReportQualityMetrics(
        exact_duplicate_paragraph_count=duplicate_paragraphs,
        normalized_duplicate_bullet_count=duplicate_bullets,
        duplicate_reference_count=duplicate_references,
        cross_section_similarity=round(max_similarity, 6),
        section_pair_similarity=pair_similarity,
        unique_evidence_count=len(evidence_catalog),
        section_evidence_counts=section_counts,
        citation_id_validity=citation_validity,
        citation_context_validity=citation_validity,
        citation_page_validity=citation_validity,
        raw_quote_copy_ratio=round(raw_quote_copy_ratio, 6),
        passed=not failures,
        failures=failures,
    )


def _raw_quote_copy_ratio(report: str, evidence_catalog: dict[str, dict[str, Any]]) -> float:
    normalized_report = normalize_text(report)
    if not normalized_report:
        return 0.0
    copied = 0
    total = 0
    for item in evidence_catalog.values():
        text = normalize_text(str(item.get("text", "")))
        if len(text) < 120:
            continue
        total += 1
        if text[:300] in normalized_report:
            copied += 1
    if not total:
        return 0.0
    return copied / total
