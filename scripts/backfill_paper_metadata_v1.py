from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import httpx

from paper_research.config import get_settings
from paper_research.db import SessionLocal
from paper_research.metadata.enrichment_service import MetadataEnrichmentService
from paper_research.metadata.normalization import identifier_as_title
from paper_research.repositories.paper import PaperRepository
from paper_research.search.clients import ArxivClient, SemanticScholarClient
from paper_research.search.http import CachedRetryClient


def classify(paper) -> str:
    missing_authors = not bool(paper.authors)
    missing_year = paper.year is None
    if identifier_as_title(paper.title):
        return "ARXIV_ID_AS_TITLE"
    if missing_authors and missing_year:
        return "MISSING_AUTHORS_AND_YEAR"
    if missing_authors:
        return "MISSING_AUTHORS"
    if missing_year:
        return "MISSING_YEAR"
    if paper.title and paper.authors and paper.year:
        return "COMPLETE"
    return "AMBIGUOUS"


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--apply", action="store_true")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--external-timeout-seconds", type=float, default=8.0)
    parser.add_argument("--external-retries", type=int, default=1)
    args = parser.parse_args()
    apply = bool(args.apply)
    settings = get_settings()
    http = CachedRetryClient(
        settings.search_cache_dir,
        settings.search_cache_ttl_seconds,
        args.external_retries,
        client=httpx.Client(follow_redirects=True, timeout=args.external_timeout_seconds),
    )
    service = MetadataEnrichmentService(
        arxiv_client=ArxivClient(http),
        semantic_scholar_client=SemanticScholarClient(http, settings.semantic_scholar_api_key),
    )
    with SessionLocal() as session:
        repo = PaperRepository(session)
        papers = repo.list(limit=args.limit, include_fixtures=False)
        before = Counter(classify(paper) for paper in papers)
        rows = []
        for paper in papers:
            classification_before = classify(paper)
            if classification_before == "COMPLETE":
                row = {
                    "paper_id": str(paper.id),
                    "before": {
                        "title": paper.title,
                        "authors": list(paper.authors or []),
                        "year": paper.year,
                        "venue": paper.venue,
                        "doi": paper.doi,
                        "arxiv_id": paper.arxiv_id,
                        "source_type": paper.source_type,
                    },
                    "identifier_candidates": [],
                    "external_candidates": [],
                    "external_errors": [],
                    "selected_match": None,
                    "confidence": 0.0,
                    "proposed_changes": [],
                    "status": "SKIPPED_COMPLETE",
                    "error": None,
                }
            else:
                row = service.enrich(paper, apply=apply).as_dict()
                if apply and row["status"] == "UPDATED":
                    repo.save(paper)
            row["classification_before"] = classification_before
            row["classification_after"] = classify(paper)
            rows.append(row)
        after_papers = repo.list(limit=args.limit, include_fixtures=False)
        after = Counter(classify(paper) for paper in after_papers)
    out = {
        "schema_version": "paper-metadata-backfill-v1",
        "dry_run": not apply,
        "apply": apply,
        "total_papers": len(papers),
        "before": dict(before),
        "after": dict(after),
        "limit": args.limit,
        "external_timeout_seconds": args.external_timeout_seconds,
        "external_retries": args.external_retries,
        "records": rows,
        "llm_used": False,
        "raw_external_responses_persisted": False,
        "vector_payload_metadata_sync": "NOT_APPLICABLE",
    }
    Path("data/evaluation").mkdir(parents=True, exist_ok=True)
    Path("docs").mkdir(parents=True, exist_ok=True)
    Path("data/evaluation/paper-metadata-backfill-v1.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    Path("docs/paper-metadata-backfill-v1.md").write_text(
        "# Paper metadata backfill v1\n\n"
        f"- Mode: {'apply' if apply else 'dry-run'}\n"
        f"- Total papers: {len(papers)}\n"
        f"- Before: `{dict(before)}`\n"
        f"- After: `{dict(after)}`\n"
        "- LLM used: false\n"
        "- Vector payload metadata sync: NOT_APPLICABLE\n",
        encoding="utf-8",
    )
    Path("data/evaluation/library-metadata-completeness-v1.json").write_text(
        json.dumps(
            {
                "schema_version": "library-metadata-completeness-v1",
                "total_papers": len(after_papers),
                "counts": dict(after),
                "complete_count": after.get("COMPLETE", 0),
                "missing_title_count": 0,
                "identifier_as_title_count": after.get("ARXIV_ID_AS_TITLE", 0),
                "missing_authors_count": after.get("MISSING_AUTHORS", 0)
                + after.get("MISSING_AUTHORS_AND_YEAR", 0),
                "missing_year_count": after.get("MISSING_YEAR", 0)
                + after.get("MISSING_AUTHORS_AND_YEAR", 0),
                "ambiguous_count": after.get("AMBIGUOUS", 0),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    Path("docs/library-metadata-completeness-v1.md").write_text(
        "# Library metadata completeness v1\n\n"
        f"- Total papers: {len(after_papers)}\n"
        f"- Counts: `{dict(after)}`\n",
        encoding="utf-8",
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
