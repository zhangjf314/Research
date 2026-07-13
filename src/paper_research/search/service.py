from concurrent.futures import ThreadPoolExecutor, as_completed

from paper_research.search.clients import PaperSourceClient
from paper_research.search.deduplication import CandidateDeduplicator
from paper_research.search.models import PaperCandidate, SearchRequest, SearchResponse
from paper_research.search.query_rewriter import QueryRewriter
from paper_research.search.scoring import CandidateScorer


class PaperSearchService:
    def __init__(
        self,
        clients: list[PaperSourceClient],
        rewriter: QueryRewriter | None = None,
        deduplicator: CandidateDeduplicator | None = None,
        scorer: CandidateScorer | None = None,
    ) -> None:
        self.clients = clients
        self.rewriter = rewriter or QueryRewriter()
        self.deduplicator = deduplicator or CandidateDeduplicator()
        self.scorer = scorer or CandidateScorer()

    def search(self, request: SearchRequest) -> SearchResponse:
        queries = self.rewriter.rewrite(request.query)
        candidates: list[PaperCandidate] = []
        errors: dict[str, str] = {}
        worker_count = max(1, min(8, len(self.clients) * len(queries)))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(client.search, query, request): (client.name, query)
                for client in self.clients
                for query in queries
            }
            for future in as_completed(futures):
                source, query = futures[future]
                try:
                    results = future.result()
                    for candidate in results:
                        candidate.matched_queries.append(query)
                    candidates.extend(results)
                except Exception as exc:
                    errors[source] = f"{type(exc).__name__}: {exc}"
        filtered = [candidate for candidate in candidates if self._matches(request, candidate)]
        unique = self.deduplicator.deduplicate(filtered)
        for candidate in unique:
            candidate.relevance_score = self.scorer.score(request.query, candidate)
        unique.sort(key=lambda item: item.relevance_score, reverse=True)
        return SearchResponse(
            original_query=request.query,
            rewritten_queries=queries,
            candidates=unique[: request.limit],
            source_errors=errors,
            telemetry={
                "fallback_used": bool(errors and unique),
                "rate_limited": any("429" in message for message in errors.values()),
            },
        )

    @staticmethod
    def _matches(request: SearchRequest, candidate: PaperCandidate) -> bool:
        if request.year_from and (candidate.year is None or candidate.year < request.year_from):
            return False
        if request.year_to and (candidate.year is None or candidate.year > request.year_to):
            return False
        return not (request.open_access_only and not candidate.pdf_url)
