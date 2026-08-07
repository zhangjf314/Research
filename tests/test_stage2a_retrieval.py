from __future__ import annotations

from paper_research.chunking.types import Chunk
from paper_research.evaluation.rag_stage2a import (
    bootstrap_delta_ci,
    complementarity,
    context_from_candidates,
    evaluate_ranked_candidates,
    paired_comparison,
    retrieval_headroom,
)
from paper_research.retrieval.fusion import FusedResult, reciprocal_rank_fusion
from paper_research.retrieval.reranker import LexicalReranker


def _chunk(chunk_id: str, paper_id: str = "p1", text: str = "alpha beta") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        paper_id=paper_id,
        block_ids=[chunk_id],
        block_type="text",
        page_start=1,
        page_end=1,
        chunk_text=text,
        token_count=2,
    )


def _row(question_id: str, value: float, *, category: str = "single_hop_factual") -> dict:
    return {
        "question_id": question_id,
        "answerable": True,
        "category": category,
        "difficulty": "easy",
        "metrics": {
            "recall_at_5": value,
            "recall_at_10": value,
            "recall_at_20": value,
            "evidence_coverage_at_5": value,
            "evidence_coverage_at_10": value,
            "evidence_coverage_at_20": value,
            "mrr_at_10": value,
        },
        "ranked_results": [{"block_ids": ["b1"], "paper_id": "p1"}] if value else [],
    }


def test_stage2a_dense_sparse_hybrid_modes_are_isolated() -> None:
    dense = [FusedResult(_chunk("dense"), 1.0, dense_rank=1)]
    sparse = [FusedResult(_chunk("sparse"), 1.0, sparse_rank=1)]
    hybrid = reciprocal_rank_fusion(dense, sparse)

    assert [item.chunk.chunk_id for item in dense] == ["dense"]
    assert [item.chunk.chunk_id for item in sparse] == ["sparse"]
    assert {item.chunk.chunk_id for item in hybrid} == {"dense", "sparse"}


def test_stage2a_reranker_preserves_identifiers_and_top_n() -> None:
    candidates = [
        FusedResult(_chunk("a", text="unrelated"), 0.1, dense_rank=1),
        FusedResult(_chunk("b", text="target alpha"), 0.01, sparse_rank=2),
    ]

    reranked = LexicalReranker().rerank("target alpha", candidates, top_k=1)

    assert len(reranked) == 1
    assert reranked[0].chunk.chunk_id == "b"
    assert reranked[0].chunk.block_ids == ["b"]


def test_stage2a_context_preserves_candidate_identifiers() -> None:
    candidate = FusedResult(_chunk("b1", paper_id="paper"), 0.5, dense_rank=2, sparse_rank=1)
    context = context_from_candidates(
        [
            type(
                "Ranked",
                (),
                {
                    "chunk": candidate.chunk,
                    "score": candidate.score,
                    "dense_rank": candidate.dense_rank,
                    "sparse_rank": candidate.sparse_rank,
                },
            )()
        ]
    )

    assert context[0]["paper_id"] == "paper"
    assert context[0]["block_ids"] == ["b1"]
    assert context[0]["dense_rank"] == 2
    assert context[0]["sparse_rank"] == 1


def test_stage2a_headroom_and_same_paper_wrong_block() -> None:
    rows = [
        {
            "question_id": "q1",
            "answerable": True,
            "metrics": {
                "recall_at_5": 0,
                "recall_at_10": 0,
                "recall_at_20": 1,
                "paper_recall_at_10": 1,
                "evidence_coverage_at_10": 0,
                "evidence_coverage_at_20": 1,
            },
            "ranked_results": [{"block_ids": ["x"]}],
        }
    ]

    headroom = retrieval_headroom(rows)

    assert headroom["top20_to_top10_rerank_headroom"] == 1.0
    assert headroom["same_paper_wrong_block_rate"] == 1.0


def test_stage2a_complementarity_accounting() -> None:
    dense = [_row("q1", 1), _row("q2", 0), _row("q3", 1), _row("q4", 0)]
    sparse = [_row("q1", 1), _row("q2", 1), _row("q3", 0), _row("q4", 0)]
    hybrid = [_row("q1", 1), _row("q2", 1), _row("q3", 1), _row("q4", 0)]

    result = complementarity(dense, sparse, hybrid)

    assert result["summary"]["both_success"] == 1
    assert result["summary"]["sparse_only_success"] == 1
    assert result["summary"]["dense_only_success"] == 1
    assert result["summary"]["both_failure"] == 1


def test_stage2a_paired_comparison_and_bootstrap_are_reproducible() -> None:
    baseline = [_row("q1", 0), _row("q2", 1)]
    candidate = [_row("q1", 1), _row("q2", 1)]

    paired = paired_comparison(baseline, candidate)
    left = bootstrap_delta_ci(baseline, candidate, metric="recall_at_10", resamples=100)
    right = bootstrap_delta_ci(baseline, candidate, metric="recall_at_10", resamples=100)

    assert paired["win_count"] == 1
    assert paired["tie_count"] == 1
    assert paired["loss_count"] == 0
    assert left == right


def test_stage2a_evaluate_ranked_candidates_uses_gold_only() -> None:
    gold = {
        "question_id": "q1",
        "answerable": True,
        "gold_block_ids": ["b1"],
        "gold_paper_ids": ["p1"],
    }
    row = evaluate_ranked_candidates(
        gold,
        [
            type(
                "Ranked",
                (),
                {
                    "chunk": _chunk("b1", "p1"),
                    "score": 1.0,
                    "dense_rank": 1,
                    "sparse_rank": None,
                },
            )()
        ],
        latency_ms=1.0,
    )

    assert row["metrics"]["recall_at_10"] == 1.0
