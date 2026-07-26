from __future__ import annotations

from uuid import uuid4

from paper_research.metadata.enrichment_service import MetadataEnrichmentService
from paper_research.models.paper import Paper
from paper_research.search.models import PaperCandidate
from scripts.backfill_paper_metadata_v1 import classify


class FakeClient:
    name = "arxiv"

    def search(self, query, request):  # noqa: ANN001
        return [
            PaperCandidate(
                source="arxiv",
                source_id="1910.10683v1",
                title=(
                    "Exploring the Limits of Transfer Learning with a Unified "
                    "Text-to-Text Transformer"
                ),
                authors=["Colin Raffel"],
                year=2019,
                arxiv_id="1910.10683",
            )
        ]


def test_classify_missing_metadata() -> None:
    paper = Paper(id=uuid4(), title="Paper", authors=[], source_type="upload")
    assert classify(paper) == "MISSING_AUTHORS_AND_YEAR"


def test_dry_run_does_not_modify_paper() -> None:
    paper = Paper(id=uuid4(), title="1910.10683", authors=[], source_type="upload")
    result = MetadataEnrichmentService(arxiv_client=FakeClient()).enrich(paper, apply=False)
    assert result.status == "AUTO_UPDATE_SAFE"
    assert paper.title == "1910.10683"
    assert paper.year is None


def test_apply_only_safe_updates() -> None:
    paper = Paper(id=uuid4(), title="1910.10683", authors=[], source_type="upload")
    result = MetadataEnrichmentService(arxiv_client=FakeClient()).enrich(paper, apply=True)
    assert result.status == "UPDATED"
    assert paper.year == 2019
    assert paper.arxiv_id == "1910.10683"
