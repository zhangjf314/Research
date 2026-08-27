"""Synthetic deterministic tests for RAG v2 diagnosis: taxonomy, metrics, packing."""

from __future__ import annotations

from paper_research.evaluation.rag_v2_diagnosis import (
    ClaimTrace,
    ContextPackingResult,
    StageResult,
    classify_failure,
    ndcg_at,
    pack_context,
    precision_at,
    recall_at,
    reciprocal_rank,
    summarize_traces,
)


def _trace(
    dense: list[str],
    sparse: list[str],
    fusion: list[str],
    context: list[str],
    excluded: list[tuple[str, str]] | None = None,
    gold: set[str] | None = None,
    answer_correct: bool | None = None,
) -> ClaimTrace:
    return ClaimTrace(
        benchmark_sample_id="synthetic-001",
        claim_text="synthetic claim",
        gold={"G1"} if gold is None else gold,
        dense=StageResult("dense", dense),
        sparse=StageResult("sparse", sparse),
        fusion=StageResult("fusion", fusion),
        rerank=None,
        context=ContextPackingResult(
            selected=context,
            excluded=excluded or [],
            token_count=10,
            budget=100,
            truncated_boundary_chunk=False,
        ),
        answer_correct=answer_correct,
    )


def test_case1_candidate_pool_failure() -> None:
    trace = _trace(
        dense=["N1", "N2"], sparse=["N3", "N4"], fusion=["N1", "N3"], context=["N1"]
    )
    codes = classify_failure(trace)
    assert "CANDIDATE_POOL_FAILURE" in codes
    assert "DENSE_RECALL_FAILURE" in codes
    assert "SPARSE_RECALL_FAILURE" in codes


def test_case2_fusion_ranking_failure() -> None:
    trace = _trace(
        dense=["G1", "N1"],
        sparse=["N2", "G1"],
        # Gold was in both input pools but falls out of the fused pool.
        fusion=["N1", "N2", "N3", "N4", "N5", "N6", "N7", "N8", "N9", "N10", "N11", "N12"],
        context=["N1", "N2"],
    )
    codes = classify_failure(trace)
    assert codes == ["FUSION_RANKING_FAILURE"]


def test_case3_context_selection_failure() -> None:
    trace = _trace(
        dense=["G1", "N1"],
        sparse=["G1", "N2"],
        fusion=["G1", "N1", "N2"],
        context=["N1", "N2"],
        excluded=[("G1", "TOP_K_EXCEEDED")],
    )
    codes = classify_failure(trace)
    assert codes == ["CONTEXT_SELECTION_FAILURE"]


def test_case3b_budget_eviction() -> None:
    trace = _trace(
        dense=["G1"],
        sparse=["G1"],
        fusion=["N1", "N2", "G1"],
        context=["N1"],
        excluded=[("N2", "CONTEXT_BUDGET_EXCEEDED"), ("G1", "CONTEXT_BUDGET_EXCEEDED")],
    )
    codes = classify_failure(trace)
    assert "CONTEXT_BUDGET_EVICTION" in codes
    assert "CONTEXT_SELECTION_FAILURE" in codes


def test_case4_generation_failure() -> None:
    trace = _trace(
        dense=["G1"],
        sparse=["G1", "N1"],
        fusion=["G1", "N1"],
        context=["G1", "N1"],
        answer_correct=False,
    )
    assert classify_failure(trace) == ["GENERATION_FAILURE"]


def test_success_has_no_codes() -> None:
    trace = _trace(
        dense=["G1"], sparse=["G1"], fusion=["G1"], context=["G1"], answer_correct=True
    )
    assert classify_failure(trace) == []


def test_empty_gold_is_annotation_mismatch() -> None:
    trace = _trace(dense=["G1"], sparse=["G1"], fusion=["G1"], context=["G1"], gold=set())
    assert classify_failure(trace) == ["GOLD_ANNOTATION_MISMATCH"]


def test_metrics_deterministic() -> None:
    ranked = ["a", "b", "c", "d"]
    gold = {"b", "d", "z"}
    assert recall_at(ranked, gold, 2) == 1 / 3
    assert recall_at(ranked, gold, 4) == 2 / 3
    assert reciprocal_rank(ranked, gold) == 0.5
    assert precision_at(ranked, gold, 2) == 0.5
    assert 0.0 < ndcg_at(ranked, gold, 4) <= 1.0
    assert recall_at(ranked, set(), 1) == 1.0


def test_pack_context_budget_break() -> None:
    packed = pack_context(
        ["a", "b", "c"],
        token_counts={"a": 50, "b": 60, "c": 10},
        budget=100,
        top_k=5,
    )
    assert packed.selected == ["a"]
    reasons = {doc: reason for doc, reason in packed.excluded}
    assert reasons["b"] == "CONTEXT_BUDGET_EXCEEDED"
    assert reasons["c"] == "CONTEXT_BUDGET_EXCEEDED"
    assert packed.truncated_boundary_chunk


def test_pack_context_top_k_and_dedupe() -> None:
    packed = pack_context(
        ["a", "a", "b", "c", "d", "e"],
        token_counts={k: 1 for k in "abcde"},
        budget=100,
        top_k=3,
    )
    assert packed.selected == ["a", "b", "c"]
    assert ("d", "TOP_K_EXCEEDED") in packed.excluded


def test_summarize_traces_and_serialization() -> None:
    traces = [
        _trace(dense=["G1"], sparse=["G1"], fusion=["G1"], context=["G1"]),
        _trace(dense=["N1"], sparse=["N2"], fusion=["N1"], context=["N1"]),
    ]
    summary = summarize_traces(traces)
    assert summary["sample_count"] == 2
    assert summary["failure_buckets"]["candidate_recall"] == 1
    assert summary["metrics"]["context_recall"] == 0.5
    records = [t.to_record() for t in traces]
    assert records[0]["gold_in_final_context"] is True
    assert records[1]["failure_codes"] == [
        "DENSE_RECALL_FAILURE",
        "SPARSE_RECALL_FAILURE",
        "CANDIDATE_POOL_FAILURE",
    ]
    # Metrics beyond the pool depth must be explicitly not computable.
    assert summary["metrics"]["context_claim_hit_rate@20"] == "METRIC_NOT_COMPUTABLE"
    assert summary["metrics"]["context_gold_block_recall@20"] == "METRIC_NOT_COMPUTABLE"


def test_metric_semantics_hit_rate_vs_block_fraction() -> None:
    # One claim with two gold blocks, only one recalled: hit-rate 1.0 but
    # block fraction 0.5 — the two contracts must never collapse into one key.
    trace = _trace(
        dense=["G1", "N1"],
        sparse=["G1"],
        fusion=["G1", "N1", "N2", "N3", "N4"],
        context=["G1"],
        gold={"G1", "G2"},
    )
    summary = summarize_traces([trace])
    assert summary["metrics"]["fusion_claim_hit_rate@5"] == 1.0
    assert summary["metrics"]["fusion_gold_block_recall@5"] == 0.5


def test_metric_name_contract_rejects_ambiguous_recall() -> None:
    from paper_research.evaluation.rag_v2_diagnosis import validate_metric_names

    assert validate_metric_names({"fusion_claim_hit_rate@5": 1.0}) == []
    assert validate_metric_names({"fusion_gold_block_recall@5": 1.0}) == []
    assert validate_metric_names({"fusion_Recall@5": 0.9}) == ["fusion_Recall@5"]
    assert validate_metric_names({"Recall@5": 0.9}) == ["Recall@5"]
