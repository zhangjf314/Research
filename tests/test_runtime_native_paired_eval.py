from __future__ import annotations

from paper_research.chunking.types import Chunk
from paper_research.evaluation.runtime_native import C4RuntimeNativeFinalizer
from paper_research.retrieval.context_builder import ContextBuilder
from paper_research.retrieval.dense import RetrievalResult
from paper_research.retrieval.fusion import FusedResult
from paper_research.retrieval.hybrid import HybridRetriever
from paper_research.retrieval.reranker import (
    DisabledReranker,
    Reranker,
    RerankerProviderError,
)
from paper_research.retrieval.sparse import BM25Retriever


def make_chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        paper_id="p1",
        block_ids=[f"b-{chunk_id}"],
        block_type="paragraph",
        page_start=1,
        page_end=1,
        chunk_text=text,
        token_count=len(text.split()),
    )


class CountingDense:
    def __init__(self, results: list[RetrievalResult]) -> None:
        self.results = results
        self.calls = 0

    def retrieve(self, *_args: object, **_kwargs: object) -> list[RetrievalResult]:
        self.calls += 1
        return self.results


class ReverseReranker(Reranker):
    provider_name = "test"
    model_name = "test-reranker"

    def __init__(self) -> None:
        self.queries: list[str] = []
        self.inputs: list[list[str]] = []

    def rerank(self, query: str, results: list[FusedResult], top_k: int) -> list[FusedResult]:
        self.queries.append(query)
        self.inputs.append([item.chunk.chunk_id for item in results])
        return list(reversed(results))[:top_k]


class FailingReranker(Reranker):
    provider_name = "test"
    model_name = "test-reranker"

    def rerank(self, query: str, results: list[FusedResult], top_k: int) -> list[FusedResult]:
        del query, results, top_k
        raise RerankerProviderError("test failure", api_request_count=1)


def test_paired_finalizers_execute_common_trunk_once_and_share_candidates() -> None:
    first, second = make_chunk("a", "alpha evidence"), make_chunk("b", "beta evidence")
    dense = CountingDense([RetrievalResult(first, 0.9), RetrievalResult(second, 0.8)])
    p0 = HybridRetriever(
        dense,  # type: ignore[arg-type]
        BM25Retriever([first, second]),
        DisabledReranker(),
        ContextBuilder(include_neighbors=False),
    )
    trunk = p0.common_trunk("alpha", recall_k=2, retrieval_scope="paper")
    baseline = p0.finalize(trunk, top_k=2)
    reranker = ReverseReranker()
    c4 = C4RuntimeNativeFinalizer(reranker, ContextBuilder(include_neighbors=False))
    candidate = c4.finalize(trunk, top_k=2)

    assert dense.calls == 1
    assert reranker.queries == ["alpha"]
    assert reranker.inputs == [[item.chunk.chunk_id for item in trunk.candidates]]
    assert {item.chunk.chunk_id for item in candidate.ranked} == {
        item.chunk.chunk_id for item in trunk.candidates
    }
    assert [item.chunk_id for item in baseline.context] == [
        item.chunk.chunk_id for item in trunk.candidates
    ]


def test_c4_terminal_failure_falls_back_to_original_p0_order() -> None:
    first, second = make_chunk("a", "alpha evidence"), make_chunk("b", "beta evidence")
    p0 = HybridRetriever(
        CountingDense([RetrievalResult(first, 0.9), RetrievalResult(second, 0.8)]),  # type: ignore[arg-type]
        BM25Retriever([first, second]),
        DisabledReranker(),
        ContextBuilder(include_neighbors=False),
    )
    trunk = p0.common_trunk("alpha", recall_k=2, retrieval_scope="paper")
    c4 = C4RuntimeNativeFinalizer(FailingReranker(), ContextBuilder(include_neighbors=False))
    result = c4.finalize(trunk, top_k=2)

    assert result.fallback is True
    assert result.api_request_count == 1
    assert [item.chunk.chunk_id for item in result.ranked] == [
        item.chunk.chunk_id for item in trunk.candidates
    ]
