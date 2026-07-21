import json
from pathlib import Path

from paper_research.chunking.types import Chunk
from paper_research.retrieval.context_builder import ContextBuilder
from paper_research.retrieval.dense import RetrievalResult
from paper_research.retrieval.filters import RetrievalFilter
from paper_research.retrieval.fusion import reciprocal_rank_fusion
from paper_research.retrieval.hybrid import HybridRetriever
from paper_research.retrieval.reranker import LexicalReranker
from paper_research.retrieval.sparse import BM25Retriever
from paper_research.retrieval.trace import JsonlTraceRepository


def chunk(
    chunk_id: str, text: str, *, paper: str = "p1", section: str = "Method", page: int = 1
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        paper_id=paper,
        block_ids=[f"b-{chunk_id}"],
        section_path=[section],
        block_type="paragraph",
        page_start=page,
        page_end=page,
        chunk_text=text,
        previous_context="previous evidence",
        next_context="next evidence",
        token_count=len(text.split()),
    )


class FakeDenseRetriever:
    def __init__(self, results: list[RetrievalResult]) -> None:
        self.results = results
        self.queries: list[str] = []

    def retrieve(self, query: str, *_args: object, **_kwargs: object) -> list[RetrievalResult]:
        self.queries.append(query)
        return self.results


def test_bm25_ranks_exact_terms_and_applies_metadata_filter() -> None:
    chunks = [
        chunk("a", "transformer attention mechanism", section="Method", page=2),
        chunk("b", "translation evaluation benchmark", section="Experiments", page=7),
    ]
    retriever = BM25Retriever(chunks)

    results = retriever.retrieve(
        "attention mechanism",
        retrieval_filter=RetrievalFilter(sections=["method"], page_to=3),
    )

    assert [result.chunk.chunk_id for result in results] == ["a"]
    assert results[0].score > 0


def test_rrf_records_dense_and_sparse_ranks() -> None:
    first = chunk("a", "alpha")
    second = chunk("b", "beta")
    fused = reciprocal_rank_fusion(
        [RetrievalResult(first, 0.9), RetrievalResult(second, 0.8)],
        [RetrievalResult(second, 4.0), RetrievalResult(first, 2.0)],
    )

    assert {item.chunk.chunk_id for item in fused} == {"a", "b"}
    assert all(item.dense_rank is not None and item.sparse_rank is not None for item in fused)


def test_rrf_prefers_fresh_sparse_chunk_metadata_without_changing_ranks() -> None:
    dense_chunk = chunk("multi", "dense payload", page=10).model_copy(
        update={
            "block_ids": ["b000113", "b000115"],
            "page_start": 10,
            "page_end": 11,
            "block_page_map": {"b000113": 10, "b000115": 10},
        }
    )
    sparse_chunk = chunk("multi", "sparse payload", page=10).model_copy(
        update={
            "block_ids": ["b000113", "b000115"],
            "page_start": 10,
            "page_end": 11,
            "block_page_map": {"b000113": 10, "b000115": 11},
        }
    )

    fused = reciprocal_rank_fusion(
        [RetrievalResult(dense_chunk, 0.9)],
        [RetrievalResult(sparse_chunk, 4.0)],
    )

    assert fused[0].dense_rank == 1
    assert fused[0].sparse_rank == 1
    assert fused[0].chunk.block_page_map == {"b000113": 10, "b000115": 11}


def test_hybrid_pipeline_persists_complete_trace_and_neighbors(tmp_path: Path) -> None:
    first = chunk("a", "generic language model background")
    second = chunk("b", "low rank adaptation lora method")
    sparse = BM25Retriever([first, second])
    trace_path = tmp_path / "traces.jsonl"
    retriever = HybridRetriever(
        FakeDenseRetriever([RetrievalResult(first, 0.9), RetrievalResult(second, 0.8)]),  # type: ignore[arg-type]
        sparse,
        LexicalReranker(),
        ContextBuilder(include_neighbors=True),
        JsonlTraceRepository(trace_path),
    )

    result = retriever.retrieve("lora low rank", RetrievalFilter(paper_ids=["p1"]), top_k=1)

    assert result.context[0].chunk_id == "b"
    assert "previous evidence" in result.context[0].evidence
    assert result.trace.dense_results
    assert result.trace.sparse_results
    assert result.trace.fusion_results
    assert result.trace.rerank_results
    saved = json.loads(trace_path.read_text(encoding="utf-8"))
    assert saved["trace_id"] == result.trace.trace_id
    assert saved["filters"]["paper_ids"] == ["p1"]


def test_paper_experiment_design_query_records_deterministic_routing_signals() -> None:
    result_chunk = chunk(
        "results",
        "Below, we evaluate the 8 models on a wide range of datasets "
        "and group the datasets into task categories.",
        section="3 Results",
        page=10,
    )
    dense = FakeDenseRetriever([RetrievalResult(result_chunk, 0.9)])
    sparse = BM25Retriever([result_chunk])
    retriever = HybridRetriever(
        dense,  # type: ignore[arg-type]
        sparse,
        LexicalReranker(),
        ContextBuilder(include_neighbors=False),
    )

    result = retriever.retrieve(
        "How are the target paper's experiments designed and evaluated?",
        RetrievalFilter(paper_ids=["p1"]),
        top_k=1,
        retrieval_scope="paper",
    )

    assert dense.queries
    assert "models compared" in dense.queries[0]
    assert "task categories" in dense.queries[0]
    assert result.trace.query == "How are the target paper's experiments designed and evaluated?"
    assert result.trace.routed_query == dense.queries[0]
    assert result.trace.query_routing_signals
