from __future__ import annotations

from uuid import uuid4

from paper_research.metadata.enrichment_service import MetadataEnrichmentService
from paper_research.metadata.normalization import extract_arxiv_id, normalize_title
from paper_research.models.paper import Paper
from paper_research.search.models import PaperCandidate


class FakeClient:
    name = "arxiv"

    def __init__(self, candidates: list[PaperCandidate]) -> None:
        self.candidates = candidates

    def search(self, query, request):  # noqa: ANN001
        return self.candidates


class FailingClient:
    name = "semantic_scholar"

    def search(self, query, request):  # noqa: ANN001
        raise RuntimeError("rate limited")


def test_extract_arxiv_id_variants() -> None:
    assert extract_arxiv_id("1910.10683") == "1910.10683"
    assert extract_arxiv_id("1910.10683v1") == "1910.10683"
    assert extract_arxiv_id("arXiv:1910.10683") == "1910.10683"
    assert extract_arxiv_id("2302.13971.pdf") == "2302.13971"
    assert extract_arxiv_id("hep-th/9901001") == "hep-th/9901001"


def test_normalize_ligature_mojibake() -> None:
    assert normalize_title("LLaMA: Open and Ef铿乧ient Foundation Language Models") == (
        "LLaMA: Open and Efficient Foundation Language Models"
    )


def test_exact_arxiv_match_auto_updates_missing_fields() -> None:
    paper = Paper(id=uuid4(), title="1910.10683", authors=[], source_type="upload")
    candidate = PaperCandidate(
        source="arxiv",
        source_id="1910.10683v1",
        title="Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer",
        authors=["Colin Raffel"],
        year=2019,
        arxiv_id="1910.10683",
    )
    result = MetadataEnrichmentService(arxiv_client=FakeClient([candidate])).enrich(paper)
    assert result.status == "AUTO_UPDATE_SAFE"
    assert {change.field for change in result.proposed_changes} >= {
        "title",
        "authors",
        "year",
        "arxiv_id",
    }


def test_fuzzy_match_needs_review_not_auto_update() -> None:
    paper = Paper(id=uuid4(), title="Some approximate title", authors=[], source_type="upload")
    candidate = PaperCandidate(
        source="arxiv",
        source_id="x",
        title="A different title",
        authors=[],
        year=2020,
    )
    result = MetadataEnrichmentService(arxiv_client=FakeClient([candidate])).enrich(paper)
    assert result.status == "NEEDS_REVIEW"


def test_audit_fixture_not_enriched() -> None:
    paper = Paper(id=uuid4(), title="fully-scanned", authors=[], source_type="audit_fixture")
    result = MetadataEnrichmentService(arxiv_client=FakeClient([])).enrich(paper)
    assert result.status == "NO_CHANGE"


def test_external_client_failure_is_audited_without_stopping() -> None:
    paper = Paper(id=uuid4(), title="1910.10683", authors=[], source_type="upload")
    candidate = PaperCandidate(
        source="arxiv",
        source_id="1910.10683v1",
        title="Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer",
        authors=["Colin Raffel"],
        year=2019,
        arxiv_id="1910.10683",
    )
    result = MetadataEnrichmentService(
        arxiv_client=FakeClient([candidate]),
        semantic_scholar_client=FailingClient(),
    ).enrich(paper)
    assert result.status == "AUTO_UPDATE_SAFE"
    assert result.external_errors[0]["source"] == "semantic_scholar"


def test_exact_title_duplicate_sources_same_arxiv_are_safe() -> None:
    paper = Paper(
        id=uuid4(),
        title="LLaMA: Open and Ef閾夸攻cient Foundation Language Models",
        authors=[],
        source_type="upload",
    )
    candidates = [
        PaperCandidate(
            source="arxiv",
            source_id="2302.13971v1",
            title="LLaMA: Open and Efficient Foundation Language Models",
            authors=["Hugo Touvron"],
            year=2023,
            arxiv_id="2302.13971",
        ),
        PaperCandidate(
            source="semantic_scholar",
            source_id="semantic-paper",
            title="LLaMA: Open and Efficient Foundation Language Models",
            authors=["Hugo Touvron"],
            year=2023,
            arxiv_id="2302.13971",
        ),
    ]
    result = MetadataEnrichmentService(arxiv_client=FakeClient(candidates)).enrich(paper)
    assert result.status == "AUTO_UPDATE_SAFE"
    assert result.selected_match is not None
    assert result.selected_match["source"] == "arxiv"
