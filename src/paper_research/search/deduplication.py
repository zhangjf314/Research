import re

from paper_research.search.models import PaperCandidate


class CandidateDeduplicator:
    def deduplicate(self, candidates: list[PaperCandidate]) -> list[PaperCandidate]:
        unique: list[PaperCandidate] = []
        for candidate in candidates:
            match_index = next(
                (
                    index
                    for index, existing in enumerate(unique)
                    if self._same_identity(existing, candidate)
                ),
                None,
            )
            if match_index is None:
                unique.append(candidate.model_copy(deep=True))
            else:
                unique[match_index] = self._merge(unique[match_index], candidate)
        return unique

    def _same_identity(self, first: PaperCandidate, second: PaperCandidate) -> bool:
        first_doi, second_doi = self._doi(first.doi), self._doi(second.doi)
        if first_doi and second_doi and first_doi == second_doi:
            return True
        first_arxiv, second_arxiv = self._arxiv(first.arxiv_id), self._arxiv(second.arxiv_id)
        if first_arxiv and second_arxiv and first_arxiv == second_arxiv:
            return True
        return self._title(first.title) == self._title(second.title)

    @staticmethod
    def _doi(value: str | None) -> str | None:
        return value.lower().removeprefix("https://doi.org/") if value else None

    @staticmethod
    def _arxiv(value: str | None) -> str | None:
        return re.sub(r"v\d+$", "", value.lower()) if value else None

    @staticmethod
    def _title(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", value.lower())

    @staticmethod
    def _merge(first: PaperCandidate, second: PaperCandidate) -> PaperCandidate:
        preferred = first if len(first.abstract or "") >= len(second.abstract or "") else second
        merged = preferred.model_copy(deep=True)
        merged.citation_count = max(first.citation_count, second.citation_count)
        merged.pdf_url = first.pdf_url or second.pdf_url
        merged.source_url = first.source_url or second.source_url
        merged.doi = first.doi or second.doi
        merged.arxiv_id = first.arxiv_id or second.arxiv_id
        merged.is_open_access = first.is_open_access or second.is_open_access
        merged.matched_queries = list(dict.fromkeys(first.matched_queries + second.matched_queries))
        merged.source = "+".join(sorted(set(first.source.split("+") + second.source.split("+"))))
        return merged
