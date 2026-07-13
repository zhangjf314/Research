import json
from pathlib import Path

from paper_research.config import get_settings
from paper_research.search.clients import ArxivClient, SemanticScholarClient
from paper_research.search.http import CachedRetryClient
from paper_research.search.models import SearchRequest
from paper_research.search.service import PaperSearchService


def main() -> None:
    settings = get_settings()
    http = CachedRetryClient(
        settings.search_cache_dir,
        ttl_seconds=settings.search_cache_ttl_seconds,
        retries=settings.external_request_retries,
    )
    response = PaperSearchService(
        [ArxivClient(http), SemanticScholarClient(http, settings.semantic_scholar_api_key)]
    ).search(SearchRequest(query="long-context large language models survey", limit=10))
    report = [
        "# External Paper Search Audit",
        "",
        f"- Rewritten queries: {len(response.rewritten_queries)}",
        f"- Deduplicated candidates: {len(response.candidates)}",
        f"- Source errors: {len(response.source_errors)}",
        "",
        "| Rank | Title | Source | Year | Citations | PDF | Score |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    report.extend(
        f"| {rank} | {item.title.replace('|', ' ')} | {item.source} | {item.year or ''} | "
        f"{item.citation_count} | {'yes' if item.pdf_url else 'no'} | {item.relevance_score:.3f} |"
        for rank, item in enumerate(response.candidates, start=1)
    )
    Path("data/reports/external-search-audit.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    Path("data/reports/external-search-audit.json").write_text(
        json.dumps(response.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
