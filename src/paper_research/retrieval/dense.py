from dataclasses import dataclass

from paper_research.chunking.types import Chunk
from paper_research.indexing.embedding import EmbeddingProvider
from paper_research.indexing.vector_store import QdrantVectorStore
from paper_research.retrieval.filters import RetrievalFilter


@dataclass(frozen=True)
class RetrievalResult:
    chunk: Chunk
    score: float


class DenseRetriever:
    def __init__(self, embedding: EmbeddingProvider, vector_store: QdrantVectorStore) -> None:
        self.embedding = embedding
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        *,
        paper_ids: list[str] | None = None,
        retrieval_filter: RetrievalFilter | None = None,
        top_k: int = 5,
        score_threshold: float | None = None,
    ) -> list[RetrievalResult]:
        if not query.strip():
            return []
        vector = self.embedding.embed_query(query)
        effective_paper_ids = retrieval_filter.paper_ids if retrieval_filter else paper_ids
        candidates = [
            RetrievalResult(chunk=chunk, score=score)
            for chunk, score in self.vector_store.search(
                vector,
                paper_ids=effective_paper_ids,
                limit=top_k * 5 if retrieval_filter else top_k,
                score_threshold=score_threshold,
            )
        ]
        if retrieval_filter:
            candidates = [item for item in candidates if retrieval_filter.matches(item.chunk)]
        return candidates[:top_k]
