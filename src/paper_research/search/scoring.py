import math
from datetime import UTC, datetime

from paper_research.chunking.tokenizer import tokenize
from paper_research.search.models import PaperCandidate


class CandidateScorer:
    def score(self, query: str, candidate: PaperCandidate) -> float:
        query_terms = self._terms(query)
        title_terms = self._terms(candidate.title)
        abstract_terms = self._terms(candidate.abstract or "")
        title_overlap = len(query_terms & title_terms) / max(1, len(query_terms))
        abstract_overlap = len(query_terms & abstract_terms) / max(1, len(query_terms))
        current_year = datetime.now(UTC).year
        recency = max(0.0, 1 - (current_year - candidate.year) / 20) if candidate.year else 0.0
        citation = min(1.0, math.log1p(candidate.citation_count) / math.log1p(10000))
        open_access = 1.0 if candidate.pdf_url else 0.0
        return round(
            0.45 * title_overlap
            + 0.25 * abstract_overlap
            + 0.1 * recency
            + 0.1 * citation
            + 0.1 * open_access,
            6,
        )

    @staticmethod
    def _terms(text: str) -> set[str]:
        return {token.lower() for token in tokenize(text) if token.isalnum()}
