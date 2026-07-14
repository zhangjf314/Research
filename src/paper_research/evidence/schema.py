from __future__ import annotations

import hashlib
import re
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from paper_research.parsing.types import PaperBlock

EvidenceRole = Literal[
    "definition",
    "method",
    "mechanism",
    "assumption",
    "setup",
    "dataset",
    "metric",
    "result",
    "comparison",
    "limitation",
    "conclusion",
    "citation_only",
    "metadata",
    "non_evidence",
]

SCHEMA_VERSION = "evidence-unit-v1"
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")
SENTENCE_RE = re.compile(r"[^.!?。！？\n]+(?:[.!?。！？]+|$)")
NUMBER_RE = re.compile(r"(?<!\w)[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:%|\s*[A-Za-z]+)?")
EQUATION_RE = re.compile(r"(?:Eq(?:uation)?\.?\s*\(?\d+\)?|\([0-9]+\))", re.I)
TABLE_RE = re.compile(r"Table\s+[A-Za-z0-9.-]+", re.I)
FIGURE_RE = re.compile(r"(?:Figure|Fig\.)\s+[A-Za-z0-9.-]+", re.I)

ROLE_TERMS: dict[EvidenceRole, tuple[str, ...]] = {
    "definition": ("defined as", "refers to", "we define", "is a"),
    "method": ("method", "approach", "algorithm", "we propose", "architecture"),
    "mechanism": ("mechanism", "by using", "through", "consists of", "operates"),
    "assumption": ("assume", "assumption", "under the condition", "we consider"),
    "setup": ("setup", "implementation", "training", "hyperparameter", "batch size"),
    "dataset": ("dataset", "corpus", "benchmark", "training set", "test set"),
    "metric": ("metric", "accuracy", "precision", "recall", "f1", "bleu", "rouge"),
    "result": ("result", "outperform", "achieve", "improve", "score", "%"),
    "comparison": ("compared", "versus", "than", "baseline", "comparison"),
    "limitation": ("limitation", "limited", "however", "future work", "fails"),
    "conclusion": ("conclusion", "we show", "we demonstrate", "in summary"),
    "citation_only": (),
    "metadata": (),
    "non_evidence": (),
}

METHOD_TERMS = ("model", "method", "algorithm", "architecture", "encoder", "decoder", "training")
RESULT_TERMS = ("result", "accuracy", "score", "improve", "outperform", "performance")
COMPARISON_TERMS = ("compare", "versus", "than", "baseline", "difference", "similar")


class SentenceSpan(BaseModel):
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    text: str = Field(min_length=1)


class EvidenceUnit(BaseModel):
    evidence_id: str
    paper_id: str
    page: int = Field(ge=1)
    block_id: str
    source_chunk_id: str | None = None
    section_id: str | None = None
    section_title: str | None = None
    block_type: str
    ordinal: int = Field(ge=0)
    text: str
    normalized_text: str
    previous_block_id: str | None = None
    next_block_id: str | None = None
    parent_heading_ids: list[str] = Field(default_factory=list)
    sentence_spans: list[SentenceSpan] = Field(default_factory=list)
    entity_terms: list[str] = Field(default_factory=list)
    method_terms: list[str] = Field(default_factory=list)
    result_terms: list[str] = Field(default_factory=list)
    comparison_terms: list[str] = Field(default_factory=list)
    numeric_facts: list[str] = Field(default_factory=list)
    equation_refs: list[str] = Field(default_factory=list)
    table_refs: list[str] = Field(default_factory=list)
    figure_refs: list[str] = Field(default_factory=list)
    evidence_roles: list[EvidenceRole] = Field(default_factory=list)
    parse_confidence: float | None = Field(default=None, ge=0, le=1)
    source_version: str
    schema_version: str = SCHEMA_VERSION
    combination_group_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_traceability(self) -> EvidenceUnit:
        if not self.paper_id or not self.block_id:
            raise ValueError("paper_id and block_id are required for citation traceability")
        expected = stable_evidence_id(self.paper_id, self.page, self.block_id, self.source_version)
        if self.evidence_id != expected:
            raise ValueError("evidence_id does not match its source triple")
        return self

    @property
    def citation_triple(self) -> tuple[str, int, str]:
        return self.paper_id, self.page, self.block_id

    @property
    def eligible_for_final_context(self) -> bool:
        excluded = {"citation_only", "metadata", "non_evidence"}
        return bool(self.evidence_roles) and not set(self.evidence_roles).issubset(excluded)


def normalize_text(text: str) -> str:
    return " ".join(text.casefold().split())


def stable_evidence_id(paper_id: str, page: int, block_id: str, source_version: str) -> str:
    value = f"{SCHEMA_VERSION}|{source_version}|{paper_id}|{page}|{block_id}"
    return f"ev-{hashlib.sha256(value.encode()).hexdigest()[:24]}"


def sentence_spans(text: str) -> list[SentenceSpan]:
    spans = []
    for match in SENTENCE_RE.finditer(text):
        cleaned = match.group().strip()
        if not cleaned:
            continue
        left_trim = len(match.group()) - len(match.group().lstrip())
        start = match.start() + left_trim
        spans.append(SentenceSpan(start=start, end=start + len(cleaned), text=cleaned))
    return spans


def classify_roles(block: PaperBlock) -> list[EvidenceRole]:
    normalized = normalize_text(block.text)
    if block.block_type == "reference" or re.match(r"^\[?\d+\]?\s", normalized):
        return ["citation_only"]
    if block.block_type in {"title", "heading"}:
        return ["metadata"]
    if len(normalized) < 20 or normalized.isdigit():
        return ["non_evidence"]
    roles: list[EvidenceRole] = []
    if block.block_type == "table":
        roles.extend(["result", "comparison"])
    if block.block_type == "formula":
        roles.extend(["method", "mechanism"])
    for role, terms in ROLE_TERMS.items():
        if terms and any(term in normalized for term in terms):
            roles.append(role)
    if not roles:
        roles.append("non_evidence")
    return list(dict.fromkeys(roles))


def _matched_terms(normalized: str, terms: tuple[str, ...]) -> list[str]:
    return [term for term in terms if term in normalized]


def build_evidence_unit(
    paper_id: str,
    block: PaperBlock,
    *,
    source_chunk_id: str | None,
    source_version: str,
) -> EvidenceUnit:
    page = block.source_page or block.page_start
    normalized = normalize_text(block.text)
    words = WORD_RE.findall(block.text)
    entities = sorted({word for word in words if word[0].isupper()})[:32]
    confidence = block.ocr_confidence if block.is_ocr else 1.0
    section_title = block.section_path[-1] if block.section_path else None
    section_id = (
        f"sec-{hashlib.sha256(f'{paper_id}|{block.section_path}'.encode()).hexdigest()[:16]}"
        if block.section_path
        else None
    )
    return EvidenceUnit(
        evidence_id=stable_evidence_id(paper_id, page, block.block_id, source_version),
        paper_id=paper_id,
        page=page,
        block_id=block.block_id,
        source_chunk_id=source_chunk_id,
        section_id=section_id,
        section_title=section_title,
        block_type=block.block_type,
        ordinal=block.block_index,
        text=block.text,
        normalized_text=normalized,
        previous_block_id=block.previous_block_id,
        next_block_id=block.next_block_id,
        parent_heading_ids=[block.parent_block_id] if block.parent_block_id else [],
        sentence_spans=sentence_spans(block.text),
        entity_terms=entities,
        method_terms=_matched_terms(normalized, METHOD_TERMS),
        result_terms=_matched_terms(normalized, RESULT_TERMS),
        comparison_terms=_matched_terms(normalized, COMPARISON_TERMS),
        numeric_facts=NUMBER_RE.findall(block.text),
        equation_refs=EQUATION_RE.findall(block.text),
        table_refs=TABLE_RE.findall(block.text),
        figure_refs=FIGURE_RE.findall(block.text),
        evidence_roles=classify_roles(block),
        parse_confidence=confidence,
        source_version=source_version,
    )
