from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from paper_research.metadata.normalization import (
    comparable_title,
    extract_arxiv_id,
    identifier_as_title,
    normalize_title,
)
from paper_research.models.paper import Paper
from paper_research.search.models import PaperCandidate, SearchRequest


class MetadataSearchClient(Protocol):
    name: str

    def search(self, query: str, request: SearchRequest) -> list[PaperCandidate]: ...


@dataclass
class MetadataChange:
    field: str
    old_value: object
    new_value: object
    source: str
    confidence: float
    match_reason: str


@dataclass
class MetadataBackfillResult:
    paper_id: str
    before: dict[str, object]
    identifier_candidates: list[str] = field(default_factory=list)
    external_candidates: list[dict[str, object]] = field(default_factory=list)
    external_errors: list[dict[str, str]] = field(default_factory=list)
    selected_match: dict[str, object] | None = None
    confidence: float = 0.0
    proposed_changes: list[MetadataChange] = field(default_factory=list)
    status: str = "NO_CHANGE"
    error: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "paper_id": self.paper_id,
            "before": self.before,
            "identifier_candidates": self.identifier_candidates,
            "external_candidates": self.external_candidates,
            "external_errors": self.external_errors,
            "selected_match": self.selected_match,
            "confidence": self.confidence,
            "proposed_changes": [change.__dict__ for change in self.proposed_changes],
            "status": self.status,
            "error": self.error,
        }


class MetadataEnrichmentService:
    def __init__(
        self,
        *,
        arxiv_client: MetadataSearchClient | None = None,
        semantic_scholar_client: MetadataSearchClient | None = None,
    ) -> None:
        self.arxiv_client = arxiv_client
        self.semantic_scholar_client = semantic_scholar_client

    def enrich(self, paper: Paper, *, apply: bool = False) -> MetadataBackfillResult:
        result = MetadataBackfillResult(paper_id=str(paper.id), before=_paper_metadata(paper))
        if paper.source_type == "audit_fixture":
            result.status = "NO_CHANGE"
            return result
        identifier = extract_arxiv_id(paper.arxiv_id, paper.title, _filename_stem(paper))
        if identifier:
            result.identifier_candidates.append(identifier)
        candidates, external_errors = self._fetch_candidates(paper, identifier)
        result.external_candidates = [_candidate_summary(candidate) for candidate in candidates]
        result.external_errors = external_errors
        selected = self._select_match(paper, candidates, identifier)
        if selected is None:
            result.status = "NO_MATCH" if not candidates else "NEEDS_REVIEW"
            return result
        candidate, reason, confidence = selected
        result.selected_match = _candidate_summary(candidate)
        result.confidence = confidence
        result.proposed_changes = _safe_changes(paper, candidate, reason, confidence)
        result.status = "AUTO_UPDATE_SAFE" if result.proposed_changes else "NO_CHANGE"
        if apply and result.status == "AUTO_UPDATE_SAFE":
            for change in result.proposed_changes:
                setattr(paper, change.field, change.new_value)
            result.status = "UPDATED"
        return result

    def _fetch_candidates(
        self,
        paper: Paper,
        identifier: str | None,
    ) -> tuple[list[PaperCandidate], list[dict[str, str]]]:
        queries: list[str] = []
        if identifier:
            queries.append(identifier)
        normalized_title = normalize_title(paper.title)
        if normalized_title and not identifier_as_title(normalized_title):
            queries.append(normalized_title)
        candidates: list[PaperCandidate] = []
        errors: list[dict[str, str]] = []
        for query in dict.fromkeys(queries):
            request = SearchRequest(query=query, limit=5)
            if self.arxiv_client is not None:
                try:
                    candidates.extend(self.arxiv_client.search(query, request))
                except Exception as exc:  # noqa: BLE001 - audit and continue per paper
                    errors.append(_external_error(self.arxiv_client.name, query, exc))
            if self.semantic_scholar_client is not None:
                try:
                    candidates.extend(self.semantic_scholar_client.search(query, request))
                except Exception as exc:  # noqa: BLE001 - audit and continue per paper
                    errors.append(_external_error(self.semantic_scholar_client.name, query, exc))
        return candidates, errors

    def _select_match(
        self,
        paper: Paper,
        candidates: list[PaperCandidate],
        identifier: str | None,
    ) -> tuple[PaperCandidate, str, float] | None:
        if identifier:
            exact = [
                candidate
                for candidate in candidates
                if candidate.arxiv_id and extract_arxiv_id(candidate.arxiv_id) == identifier
            ]
            if len(exact) == 1:
                return exact[0], "exact_arxiv_id", 1.0
        title_key = comparable_title(paper.title)
        exact_title = [
            candidate
            for candidate in candidates
            if title_key and comparable_title(candidate.title) == title_key
        ]
        if len(exact_title) == 1:
            candidate = exact_title[0]
            if _metadata_supports_title_match(paper, candidate):
                return candidate, "unique_exact_normalized_title", 0.95
        deduplicated_title_match = _deduplicated_exact_title_match(exact_title)
        if deduplicated_title_match and _metadata_supports_title_match(
            paper,
            deduplicated_title_match,
        ):
            return deduplicated_title_match, "deduplicated_exact_normalized_title", 0.95
        return None


def _safe_changes(
    paper: Paper,
    candidate: PaperCandidate,
    reason: str,
    confidence: float,
) -> list[MetadataChange]:
    changes: list[MetadataChange] = []
    candidate_title = normalize_title(candidate.title)
    current_title = normalize_title(paper.title)
    if (
        candidate_title
        and (
            identifier_as_title(current_title)
            or comparable_title(current_title) == comparable_title(candidate_title)
        )
        and current_title != candidate_title
    ):
        changes.append(
            MetadataChange(
                "title",
                paper.title,
                candidate_title,
                candidate.source,
                confidence,
                reason,
            )
        )
    if candidate.authors and not paper.authors:
        changes.append(
            MetadataChange(
                "authors",
                paper.authors,
                candidate.authors,
                candidate.source,
                confidence,
                reason,
            )
        )
    if candidate.year and paper.year is None:
        changes.append(
            MetadataChange("year", None, candidate.year, candidate.source, confidence, reason)
        )
    if candidate.venue and not paper.venue:
        changes.append(
            MetadataChange("venue", None, candidate.venue, candidate.source, confidence, reason)
        )
    if candidate.doi and not paper.doi:
        changes.append(
            MetadataChange("doi", None, candidate.doi, candidate.source, confidence, reason)
        )
    if candidate.arxiv_id and not paper.arxiv_id:
        changes.append(
            MetadataChange(
                "arxiv_id",
                None,
                candidate.arxiv_id,
                candidate.source,
                confidence,
                reason,
            )
        )
    return changes


def _metadata_supports_title_match(paper: Paper, candidate: PaperCandidate) -> bool:
    if paper.year and candidate.year and paper.year == candidate.year:
        return True
    if paper.authors and candidate.authors:
        local = {author.casefold() for author in paper.authors}
        remote = {author.casefold() for author in candidate.authors}
        return bool(local & remote)
    return not paper.authors and paper.year is None


def _deduplicated_exact_title_match(candidates: list[PaperCandidate]) -> PaperCandidate | None:
    if len(candidates) < 2:
        return None
    identities = {_candidate_identity(candidate) for candidate in candidates}
    identities.discard("")
    if len(identities) != 1:
        return None
    return sorted(candidates, key=lambda candidate: candidate.source != "arxiv")[0]


def _candidate_identity(candidate: PaperCandidate) -> str:
    arxiv_id = extract_arxiv_id(candidate.arxiv_id)
    if arxiv_id:
        return f"arxiv:{arxiv_id}"
    if candidate.doi:
        return f"doi:{candidate.doi.casefold()}"
    return ""


def _paper_metadata(paper: Paper) -> dict[str, object]:
    return {
        "title": paper.title,
        "authors": list(paper.authors or []),
        "year": paper.year,
        "venue": paper.venue,
        "doi": paper.doi,
        "arxiv_id": paper.arxiv_id,
        "source_type": paper.source_type,
        "filename_stem": _filename_stem(paper),
        "created_at": paper.created_at.isoformat() if paper.created_at else None,
    }


def _filename_stem(paper: Paper) -> str | None:
    return Path(paper.pdf_path).stem if paper.pdf_path else None


def _candidate_summary(candidate: PaperCandidate) -> dict[str, object]:
    return {
        "source": candidate.source,
        "source_id": candidate.source_id,
        "title": candidate.title,
        "authors": candidate.authors,
        "year": candidate.year,
        "venue": candidate.venue,
        "doi": candidate.doi,
        "arxiv_id": candidate.arxiv_id,
    }


def _external_error(source: str, query: str, exc: Exception) -> dict[str, str]:
    return {
        "source": source,
        "query": query,
        "error_type": type(exc).__name__,
        "message": str(exc)[:500],
    }
