import hashlib
from pathlib import Path

import pytest

import scripts.run_retrieval_context_optimization_v1 as optimization
from paper_research.chunking.types import Chunk
from paper_research.retrieval.context_strategy import ContextStrategy, StrategicContextBuilder
from paper_research.retrieval.dense import RetrievalResult
from paper_research.retrieval.fusion import FusedResult, reciprocal_rank_fusion


def chunk(
    identifier: str,
    *,
    paper: str = "paper-a",
    page: int = 1,
    section: str = "method",
    tokens: int = 10,
) -> Chunk:
    number = int(identifier.removeprefix("c"))
    return Chunk(
        chunk_id=identifier,
        paper_id=paper,
        block_ids=[f"b{number:06d}"],
        section_path=[section],
        block_type="paragraph",
        page_start=page,
        page_end=page,
        chunk_text=(identifier + " ") * tokens,
        token_count=tokens,
    )


def fused(items: list[Chunk]) -> list[FusedResult]:
    return [
        FusedResult(chunk=item, score=1 / rank, dense_rank=rank, sparse_rank=rank)
        for rank, item in enumerate(items, start=1)
    ]


def test_neighbor_expansion_is_ordered_and_never_crosses_paper() -> None:
    a1, a2, a3 = [chunk(f"c{index}") for index in range(1, 4)]
    other = chunk("c4", paper="paper-b")
    strategy = ContextStrategy(retrieval_k=1, context_k=1, neighbor_window=1)
    result = StrategicContextBuilder([a1, a2, a3, other], strategy).build(fused([a2]))

    assert [item.chunk_id for item in result.context] == ["c2", "c1", "c3"]
    assert {item.paper_id for item in result.context} == {"paper-a"}
    expanded = [item for item in result.trace.candidate_trace if item.expansion_source_chunk_id]
    assert [item.expansion_reason for item in expanded] == ["neighbor_window_1"] * 2


def test_page_expansion_deduplicates_and_keeps_original_trace() -> None:
    a1, a2 = chunk("c1"), chunk("c2")
    result = StrategicContextBuilder(
        [a1, a2],
        ContextStrategy(retrieval_k=2, context_k=2, page_expansion=True),
    ).build(fused([a2, a1]))

    assert [item.chunk_id for item in result.context] == ["c2", "c1"]
    assert result.trace.duplicate_chunk_ids
    retained = [item for item in result.trace.candidate_trace if item.final_context_rank]
    assert [(item.original_rank, item.original_score) for item in retained] == [(1, 1.0), (2, 0.5)]


def test_page_and_section_caps_are_applied_independently() -> None:
    chunks = [chunk(f"c{index}", page=1, section="same") for index in range(1, 5)]
    page = StrategicContextBuilder(
        chunks,
        ContextStrategy(retrieval_k=4, context_k=4, max_blocks_per_page=2),
    ).build(fused(chunks))
    section = StrategicContextBuilder(
        chunks,
        ContextStrategy(retrieval_k=4, context_k=4, max_blocks_per_section=3),
    ).build(fused(chunks))

    assert len(page.context) == 2
    assert len(section.context) == 3
    assert sum(item.excluded_reason == "page_cap" for item in page.trace.candidate_trace) == 2
    assert sum(item.excluded_reason == "section_cap" for item in section.trace.candidate_trace) == 1


def test_token_budget_truncation_is_stable() -> None:
    chunks = [chunk("c1", tokens=8), chunk("c2", tokens=8)]
    builder = StrategicContextBuilder(
        chunks,
        ContextStrategy(retrieval_k=2, context_k=2, max_context_tokens=10),
    )
    first = builder.build(fused(chunks))
    second = builder.build(fused(chunks))

    assert first.trace == second.trace
    assert first.trace.estimated_tokens == 10
    assert first.trace.truncated_chunk_ids == ["c2"]
    assert first.trace.candidate_trace[-1].token_truncated is True


def test_weighted_rrf_validates_weights_and_changes_ranking() -> None:
    dense_first, sparse_first = chunk("c1"), chunk("c2")
    dense = [
        RetrievalResult(chunk=dense_first, score=1),
        RetrievalResult(chunk=sparse_first, score=0.5),
    ]
    sparse = [
        RetrievalResult(chunk=sparse_first, score=1),
        RetrievalResult(chunk=dense_first, score=0.5),
    ]

    dense_weighted = reciprocal_rank_fusion(
        dense, sparse, dense_weight=0.7, lexical_weight=0.3
    )
    sparse_weighted = reciprocal_rank_fusion(
        dense, sparse, dense_weight=0.3, lexical_weight=0.7
    )
    assert dense_weighted[0].chunk.chunk_id == "c1"
    assert sparse_weighted[0].chunk.chunk_id == "c2"
    with pytest.raises(ValueError):
        reciprocal_rank_fusion(dense, sparse, dense_weight=0, lexical_weight=0)


def test_gold_availability_and_scope_metrics_are_exact() -> None:
    context = [
        optimization.ContextItem(
            chunk_id="c1",
            paper_id="paper-public",
            block_ids=["b000001"],
            section_path=["s"],
            page_start=2,
            page_end=2,
            evidence="evidence",
            score=1,
        )
    ]
    trace = {
        "page_counts": {"raw:2": 1},
        "section_counts": {"raw:s": 1},
        "duplicate_chunk_ids": [],
        "estimated_tokens": 5,
        "truncated_chunk_ids": [],
    }
    metrics = optimization.query_metrics(
        context,
        trace,
        {"gold_paper_ids": ["paper-public"], "gold_block_ids": ["b000001"], "gold_pages": [2]},
    )

    assert metrics["exact_gold_block_available"] is True
    assert metrics["gold_page_available"] is True
    assert metrics["context_duplication_rate"] == 0


def test_frozen_gold_files_are_not_write_targets_and_llm_is_qa_only() -> None:
    source = Path("scripts/run_retrieval_context_optimization_v1.py").read_text(encoding="utf-8")
    before = {
        path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for path in (optimization.GOLD, optimization.PROTOCOL)
    }

    assert "build_llm_provider(settings)" in source
    assert 'if args.phase == "retrieval"' in source
    assert 'require_llm=args.phase == "qa"' in source
    assert "deep_research" not in source.lower() or '"deep_research_called": False' in source
    assert all(
        hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
        for path, digest in before.items()
    )


def test_audit_schema_defaults_are_pending_and_never_contain_api_key_fields() -> None:
    source = Path("scripts/run_retrieval_context_optimization_v1.py").read_text(encoding="utf-8")

    assert '"human_review_status": "pending"' in source
    assert '"human_label": None' in source
    assert '"review_notes": None' in source
    assert "api_key\"" not in source
    assert "DisabledReranker" not in source
