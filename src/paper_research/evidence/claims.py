from __future__ import annotations

import hashlib
import re
from typing import Literal

from pydantic import BaseModel, Field

from paper_research.evidence.schema import normalize_text

ClaimRole = Literal[
    "identify",
    "define",
    "explain_method",
    "explain_mechanism",
    "compare",
    "report_result",
    "report_limitation",
    "synthesize",
    "verify_absence",
    "unknown",
]

SCHEMA_VERSION = "claim-unit-v1"
TERM_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")
STOP = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "what",
    "which",
    "does",
    "paper",
    "target",
    "into",
    "their",
    "they",
    "are",
    "was",
    "were",
    "how",
}


class ClaimUnit(BaseModel):
    claim_id: str
    question_id: str
    question_type: str
    claim_text: str
    normalized_claim: str
    claim_role: ClaimRole
    target_paper_ids: list[str] = Field(default_factory=list)
    target_terms: list[str] = Field(default_factory=list)
    method_terms: list[str] = Field(default_factory=list)
    result_terms: list[str] = Field(default_factory=list)
    comparison_dimensions: list[str] = Field(default_factory=list)
    expected_answerability: bool
    required_evidence_roles: list[str] = Field(default_factory=list)
    gold_evidence_ids: list[str] = Field(default_factory=list)
    gold_block_ids: list[str] = Field(default_factory=list)
    gold_pages: list[int] = Field(default_factory=list)
    multi_block_required: bool | None = None
    negative_constraints: list[str] = Field(default_factory=list)
    schema_version: str = SCHEMA_VERSION
    derivation_trace: dict[str, str] = Field(default_factory=dict)


def stable_claim_id(question_id: str, index: int, claim_text: str) -> str:
    digest = hashlib.sha256(
        f"{SCHEMA_VERSION}|{question_id}|{index}|{claim_text}".encode()
    ).hexdigest()[:20]
    return f"cl-{question_id}-{digest}"


def classify_question_type(record: dict) -> str:
    if not record.get("answerable", True):
        return "unanswerable"
    category = str(record.get("category", "")).casefold()
    question = normalize_text(record.get("question", ""))
    if "multi" in category or len(record.get("gold_paper_ids", [])) > 1:
        return "multi_paper"
    for name, terms in (
        ("limitation", ("limitation", "weakness", "future work")),
        ("comparison", ("compare", "difference", "versus", "both papers")),
        ("result", ("result", "performance", "score", "experiment")),
        ("mechanism", ("mechanism", "how does", "why does")),
        ("method", ("method", "algorithm", "approach", "architecture")),
        ("definition", ("define", "what is", "meaning")),
    ):
        if name in category or any(term in question for term in terms):
            return name
    return "unknown"


def classify_claim_role(claim: str, question_type: str) -> ClaimRole:
    normalized = normalize_text(claim)
    if question_type == "unanswerable":
        return "verify_absence"
    rules: tuple[tuple[ClaimRole, tuple[str, ...]], ...] = (
        ("report_limitation", ("limitation", "limited", "future work", "fails")),
        ("compare", ("compared", "than", "versus", "difference", "both")),
        ("report_result", ("result", "achieve", "score", "accuracy", "improve")),
        ("explain_mechanism", ("mechanism", "because", "through", "by using")),
        ("explain_method", ("method", "model", "algorithm", "architecture", "training")),
        ("define", ("defined", "refers to", "is a")),
        ("identify", ("introduces", "proposes", "identifies")),
    )
    for role, terms in rules:
        if any(term in normalized for term in terms):
            return role
    if question_type == "multi_paper":
        return "synthesize"
    return "unknown"


def required_roles(role: ClaimRole) -> list[str]:
    mapping = {
        "identify": ["conclusion", "method"],
        "define": ["definition"],
        "explain_method": ["method"],
        "explain_mechanism": ["mechanism", "method"],
        "compare": ["comparison", "result"],
        "report_result": ["result", "metric"],
        "report_limitation": ["limitation"],
        "synthesize": ["comparison", "conclusion"],
        "verify_absence": [],
        "unknown": [],
    }
    return mapping[role]


def _terms(text: str) -> list[str]:
    return sorted(
        {term.casefold() for term in TERM_RE.findall(text) if term.casefold() not in STOP}
    )


def build_claim_units(gold: list[dict], retrieval: dict[str, dict]) -> list[ClaimUnit]:
    units: list[ClaimUnit] = []
    for record in gold:
        question_id = record["question_id"]
        protocol = retrieval.get(question_id, {})
        question_type = classify_question_type(record)
        target_papers = protocol.get("retrieval_filter", {}).get("paper_ids", [])
        if not target_papers:
            target_papers = record.get("gold_paper_ids", [])
        source_claims = list(record.get("required_claims", []))
        if not source_claims and not record.get("answerable", True):
            source_claims = [record["question"]]
        for index, text in enumerate(source_claims, 1):
            role = classify_claim_role(text, question_type)
            normalized = normalize_text(text)
            all_terms = _terms(text)
            method = [
                term
                for term in all_terms
                if term in {"method", "model", "algorithm", "training", "architecture"}
            ]
            result = [
                term
                for term in all_terms
                if term in {"result", "accuracy", "score", "performance", "improvement"}
            ]
            comparison = [
                term
                for term in all_terms
                if term in {"compare", "difference", "versus", "baseline", "than"}
            ]
            units.append(
                ClaimUnit(
                    claim_id=stable_claim_id(question_id, index, text),
                    question_id=question_id,
                    question_type=question_type,
                    claim_text=text,
                    normalized_claim=normalized,
                    claim_role=role,
                    target_paper_ids=list(target_papers),
                    target_terms=all_terms,
                    method_terms=method,
                    result_terms=result,
                    comparison_dimensions=comparison,
                    expected_answerability=bool(record.get("answerable", True)),
                    required_evidence_roles=required_roles(role),
                    gold_evidence_ids=[],
                    gold_block_ids=list(record.get("gold_block_ids", [])),
                    gold_pages=list(record.get("gold_pages", [])),
                    multi_block_required=None,
                    negative_constraints=(
                        ["No valid supporting evidence may be fabricated or inferred from absence."]
                        if not record.get("answerable", True)
                        else []
                    ),
                    derivation_trace={
                        "claim_text": "verbatim required_claims; unanswerable uses question text",
                        "question_type": "deterministic category/question rules",
                        "gold_mapping": (
                            "question-level candidates only; claim mapping pending review"
                        ),
                    },
                )
            )
    return units
