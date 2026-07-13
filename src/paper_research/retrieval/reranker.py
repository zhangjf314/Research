from abc import ABC, abstractmethod

import httpx

from paper_research.chunking.tokenizer import tokenize
from paper_research.retrieval.fusion import FusedResult


class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: str, results: list[FusedResult], top_k: int) -> list[FusedResult]:
        """Rerank fused candidates."""


class LexicalReranker(Reranker):
    """Local deterministic baseline; replace with a Cross-Encoder in production."""

    def rerank(self, query: str, results: list[FusedResult], top_k: int) -> list[FusedResult]:
        query_terms = {token.lower() for token in tokenize(query) if token.isalnum()}

        def score(result: FusedResult) -> float:
            text_terms = {
                token.lower() for token in tokenize(result.chunk.chunk_text) if token.isalnum()
            }
            overlap = len(query_terms & text_terms) / max(1, len(query_terms))
            phrase_bonus = 0.25 if query.lower() in result.chunk.chunk_text.lower() else 0.0
            return overlap + phrase_bonus + result.score

        return sorted(results, key=score, reverse=True)[:top_k]


class DisabledReranker(Reranker):
    def rerank(self, query: str, results: list[FusedResult], top_k: int) -> list[FusedResult]:
        del query
        return results[:top_k]


class CrossEncoderReranker(Reranker):
    """HTTP Cross-Encoder adapter using a common ``/v1/rerank`` JSON contract."""

    def __init__(self, base_url: str, api_key: str | None, model: str, timeout: float = 60) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def rerank(self, query: str, results: list[FusedResult], top_k: int) -> list[FusedResult]:
        if not results:
            return []
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        response = httpx.post(
            f"{self.base_url}/v1/rerank",
            headers=headers,
            json={
                "model": self.model,
                "query": query,
                "documents": [item.chunk.chunk_text for item in results],
                "top_n": top_k,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        ranked = []
        for item in response.json().get("results", []):
            source = results[int(item["index"])]
            ranked.append(
                FusedResult(
                    chunk=source.chunk,
                    score=float(item.get("relevance_score", item.get("score", 0))),
                    dense_rank=source.dense_rank,
                    sparse_rank=source.sparse_rank,
                )
            )
        if not ranked:
            raise RuntimeError("rerank provider returned no results")
        return ranked[:top_k]
