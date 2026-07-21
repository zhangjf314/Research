"""Deterministic local lexical retrieval over evidence blocks."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

LOCAL_LEXICAL_INDEX_VERSION = "local-lexical-index-v1"


@dataclass(frozen=True)
class LexicalDocument:
    doc_id: str
    paper_id: str
    page: int
    block_id: str
    text: str
    block_type: str = ""


@dataclass(frozen=True)
class LexicalSearchResult:
    doc_id: str
    score: float
    rank: int
    paper_id: str
    page: int
    block_id: str
    matched_terms: tuple[str, ...]


def tokenize(text: str) -> tuple[str, ...]:
    return tuple(
        token
        for token in re.findall(r"[a-z0-9]+(?:[.-][a-z0-9]+)*", text.lower())
        if len(token) > 1
    )


def numeric_tokens(text: str) -> tuple[str, ...]:
    normalized = text.lower().replace(",", "")
    return tuple(dict.fromkeys(re.findall(r"\b\d+(?:\.\d+)?(?:[mbk]|%)?\b", normalized)))


class LocalLexicalIndex:
    def __init__(self, documents: Iterable[LexicalDocument]) -> None:
        self.documents = tuple(documents)
        self._term_freqs = {doc.doc_id: Counter(tokenize(doc.text)) for doc in self.documents}
        self._doc_by_id = {doc.doc_id: doc for doc in self.documents}
        self._doc_freq: Counter[str] = Counter()
        for freqs in self._term_freqs.values():
            for token in freqs:
                self._doc_freq[token] += 1
        self._avg_len = sum(sum(freqs.values()) for freqs in self._term_freqs.values()) / max(
            len(self._term_freqs),
            1,
        )
        self._by_paper_page: dict[tuple[str, int], list[LexicalDocument]] = defaultdict(list)
        for doc in self.documents:
            self._by_paper_page[(doc.paper_id, doc.page)].append(doc)

    def bm25(
        self,
        query: str,
        *,
        top_k: int = 12,
        paper_ids: set[str] | None = None,
    ) -> tuple[LexicalSearchResult, ...]:
        query_terms = tokenize(query)
        results: list[LexicalSearchResult] = []
        total_docs = max(len(self.documents), 1)
        for doc in self.documents:
            if paper_ids and doc.paper_id not in paper_ids:
                continue
            freqs = self._term_freqs[doc.doc_id]
            doc_len = max(sum(freqs.values()), 1)
            score = 0.0
            matched: list[str] = []
            for term in query_terms:
                tf = freqs.get(term, 0)
                if not tf:
                    continue
                matched.append(term)
                idf = math.log(
                    1
                    + (total_docs - self._doc_freq[term] + 0.5)
                    / (self._doc_freq[term] + 0.5)
                )
                score += idf * (tf * 2.2) / (
                    tf + 1.2 * (1 - 0.75 + 0.75 * doc_len / self._avg_len)
                )
            if score > 0:
                results.append(
                    LexicalSearchResult(
                        doc_id=doc.doc_id,
                        score=score,
                        rank=0,
                        paper_id=doc.paper_id,
                        page=doc.page,
                        block_id=doc.block_id,
                        matched_terms=tuple(dict.fromkeys(matched)),
                    )
                )
        return _rank(results, top_k)

    def exact_numeric(
        self,
        query: str,
        *,
        top_k: int = 12,
        paper_ids: set[str] | None = None,
    ) -> tuple[LexicalSearchResult, ...]:
        numbers = set(numeric_tokens(query))
        anchors = set(tokenize(query)) - numbers
        results: list[LexicalSearchResult] = []
        for doc in self.documents:
            if paper_ids and doc.paper_id not in paper_ids:
                continue
            doc_numbers = set(numeric_tokens(doc.text))
            if not numbers or not numbers & doc_numbers:
                continue
            doc_terms = set(tokenize(doc.text))
            score = 10.0 * len(numbers & doc_numbers) + len(anchors & doc_terms)
            results.append(
                LexicalSearchResult(
                    doc_id=doc.doc_id,
                    score=score,
                    rank=0,
                    paper_id=doc.paper_id,
                    page=doc.page,
                    block_id=doc.block_id,
                    matched_terms=tuple(sorted((numbers & doc_numbers) | (anchors & doc_terms))),
                )
            )
        return _rank(results, top_k)

    def same_page_expand(
        self,
        results: Iterable[LexicalSearchResult],
        *,
        top_k: int = 12,
    ) -> tuple[LexicalSearchResult, ...]:
        by_id = {result.doc_id: result for result in results}
        for result in list(by_id.values()):
            for doc in self._by_paper_page.get((result.paper_id, result.page), []):
                by_id.setdefault(
                    doc.doc_id,
                    LexicalSearchResult(
                        doc_id=doc.doc_id,
                        score=max(result.score - 0.01, 0.01),
                        rank=0,
                        paper_id=doc.paper_id,
                        page=doc.page,
                        block_id=doc.block_id,
                        matched_terms=result.matched_terms,
                    ),
                )
        return _rank(list(by_id.values()), top_k)


def _rank(results: Iterable[LexicalSearchResult], top_k: int) -> tuple[LexicalSearchResult, ...]:
    ordered = sorted(results, key=lambda item: (-item.score, item.doc_id))[:top_k]
    return tuple(
        LexicalSearchResult(
            doc_id=item.doc_id,
            score=item.score,
            rank=rank,
            paper_id=item.paper_id,
            page=item.page,
            block_id=item.block_id,
            matched_terms=item.matched_terms,
        )
        for rank, item in enumerate(ordered, 1)
    )
