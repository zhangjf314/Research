from pathlib import Path

import httpx

from paper_research.search.clients import ArxivClient, PaperSourceClient, SemanticScholarClient
from paper_research.search.deduplication import CandidateDeduplicator
from paper_research.search.http import CachedRetryClient
from paper_research.search.models import PaperCandidate, SearchRequest
from paper_research.search.query_rewriter import QueryRewriter
from paper_research.search.service import PaperSearchService

ARXIV_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry><id>http://arxiv.org/abs/2106.09685v2</id><published>2021-06-17T17:59:49Z</published>
  <title>LoRA: Low-Rank Adaptation</title><summary>A parameter efficient method.</summary>
  <author><name>Edward Hu</name></author><arxiv:doi>10.0000/lora</arxiv:doi></entry>
</feed>"""


def transport_client(handler: object) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))  # type: ignore[arg-type]


def test_arxiv_atom_response_is_normalized(tmp_path: Path) -> None:
    http = CachedRetryClient(
        tmp_path, client=transport_client(lambda _: httpx.Response(200, text=ARXIV_XML))
    )
    result = ArxivClient(http).search("low rank", SearchRequest(query="low rank", limit=5))

    assert result[0].arxiv_id == "2106.09685"
    assert result[0].doi == "10.0000/lora"
    assert result[0].pdf_url == "https://arxiv.org/pdf/2106.09685"
    assert result[0].authors == ["Edward Hu"]


def test_semantic_scholar_response_is_normalized(tmp_path: Path) -> None:
    payload = {
        "data": [
            {
                "paperId": "s2-1",
                "title": "LoRA",
                "abstract": "Low rank adaptation",
                "authors": [{"name": "Ada"}],
                "year": 2021,
                "externalIds": {"DOI": "10.0000/lora", "ArXiv": "2106.09685"},
                "citationCount": 500,
                "openAccessPdf": {"url": "https://example.test/lora.pdf"},
            }
        ]
    }
    http = CachedRetryClient(
        tmp_path, client=transport_client(lambda _: httpx.Response(200, json=payload))
    )
    result = SemanticScholarClient(http).search(
        "lora", SearchRequest(query="lora", limit=5)
    )

    assert result[0].source_id == "s2-1"
    assert result[0].citation_count == 500
    assert result[0].is_open_access is True


def test_retry_and_cache_avoid_duplicate_network_calls(tmp_path: Path) -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503 if calls == 1 else 200, json={"data": []})

    http = CachedRetryClient(tmp_path, retries=2, client=transport_client(handler))
    first = http.get_json("https://example.test/search", params={"q": "paper"})
    second = http.get_json("https://example.test/search", params={"q": "paper"})

    assert first == second == {"data": []}
    assert calls == 2


def test_deduplication_merges_cross_source_identity() -> None:
    first = PaperCandidate(
        source="arxiv", source_id="a", title="A Paper", doi="10.1/ABC", pdf_url="a.pdf"
    )
    second = PaperCandidate(
        source="semantic_scholar",
        source_id="b",
        title="A Paper",
        doi="https://doi.org/10.1/abc",
        citation_count=20,
    )

    unique = CandidateDeduplicator().deduplicate([first, second])

    assert len(unique) == 1
    assert unique[0].citation_count == 20
    assert unique[0].pdf_url == "a.pdf"


class FakeSource(PaperSourceClient):
    name = "fake"

    def search(self, query: str, request: SearchRequest) -> list[PaperCandidate]:
        return [
            PaperCandidate(
                source=self.name,
                source_id=query,
                title="Transformer attention research",
                abstract="A transformer method",
                year=2025,
                pdf_url="https://example.test/paper.pdf",
            )
        ]


def test_multi_query_search_deduplicates_and_scores() -> None:
    response = PaperSearchService([FakeSource()]).search(
        SearchRequest(query="transformer-attention", limit=10, open_access_only=True)
    )

    assert QueryRewriter().rewrite("transformer-attention")[1] == "transformer attention"
    assert len(response.candidates) == 1
    assert response.candidates[0].relevance_score > 0
    assert len(response.candidates[0].matched_queries) >= 2
