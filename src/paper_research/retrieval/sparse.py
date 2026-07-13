import math
from collections import Counter

from paper_research.chunking.tokenizer import tokenize
from paper_research.chunking.types import Chunk
from paper_research.retrieval.dense import RetrievalResult
from paper_research.retrieval.filters import RetrievalFilter


class BM25Retriever:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.term_frequencies = [Counter(self._tokens(chunk.chunk_text)) for chunk in chunks]
        self.lengths = [sum(counts.values()) for counts in self.term_frequencies]
        self.average_length = sum(self.lengths) / len(self.lengths) if self.lengths else 1.0
        document_frequency: Counter[str] = Counter()
        for counts in self.term_frequencies:
            document_frequency.update(counts.keys())
        self.document_frequency = document_frequency

    def retrieve(
        self,
        query: str,
        *,
        retrieval_filter: RetrievalFilter | None = None,
        top_k: int = 10,
    ) -> list[RetrievalResult]:
        query_terms = self._tokens(query)
        scores: list[RetrievalResult] = []
        for chunk, frequencies, length in zip(
            self.chunks, self.term_frequencies, self.lengths, strict=True
        ):
            if retrieval_filter and not retrieval_filter.matches(chunk):
                continue
            score = sum(
                self._term_score(term, frequencies[term], length) for term in set(query_terms)
            )
            if score > 0:
                scores.append(RetrievalResult(chunk=chunk, score=score))
        return sorted(scores, key=lambda item: item.score, reverse=True)[:top_k]

    def _term_score(self, term: str, frequency: int, document_length: int) -> float:
        if frequency == 0:
            return 0.0
        total = len(self.chunks)
        document_frequency = self.document_frequency[term]
        inverse_document_frequency = math.log(
            1 + (total - document_frequency + 0.5) / (document_frequency + 0.5)
        )
        denominator = frequency + self.k1 * (
            1 - self.b + self.b * document_length / self.average_length
        )
        return inverse_document_frequency * frequency * (self.k1 + 1) / denominator

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return [token.lower() for token in tokenize(text) if token.isalnum()]
