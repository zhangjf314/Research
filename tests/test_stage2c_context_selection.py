from __future__ import annotations

from paper_research.evaluation.rag_stage2c import (
    aggregate_traces,
    build_trace,
    context_selection_hypothesis_supported,
    diversity_aware_context,
    exclusive_failure_funnel,
    offline_selector_gate,
    reconstruct_baseline_context,
    score_budgeted_deduplicated_context,
)


def _gold() -> dict:
    return {
        "question_id": "q001",
        "answerable": True,
        "category": "single_hop_factual",
        "difficulty": "medium",
        "gold_block_ids": ["b1"],
        "required_claims": [
            {"claim_id": "C1", "text": "uses Adam optimizer", "gold_block_ids": ["b1"]}
        ],
    }


def _retrieval(blocks: list[str]) -> dict:
    return {
        "question_id": "q001",
        "ranked_results": [
            {"chunk_id": f"c{i}", "paper_id": f"p{i}", "block_ids": [block], "score": 1.0 / i}
            for i, block in enumerate(blocks, start=1)
        ],
        "retrieved_block_ids": blocks,
    }


def test_context_trace_distinguishes_retrieved_and_final_context() -> None:
    trace = build_trace(
        _gold(),
        _retrieval(["b2", "b1"]),
        {"answer": {"claims": []}},
        [{"chunk_id": "c1", "paper_id": "p1", "block_ids": ["b2"], "estimated_tokens": 10}],
    )
    assert "b1" in trace["retrieved_top20_block_ids"]
    assert "b1" not in trace["final_context_block_ids"]
    assert trace["required_claims"][0]["retrieved"] is True
    assert trace["required_claims"][0]["in_final_context"] is False


def test_exclusive_failure_funnel_retrieval_miss() -> None:
    trace = build_trace(_gold(), _retrieval(["b2"]), {"answer": {"claims": []}}, [])
    funnel = exclusive_failure_funnel([trace])
    assert funnel["counts"]["R0_RETRIEVAL_MISS"] == 1


def test_exclusive_failure_funnel_context_drop() -> None:
    trace = build_trace(
        _gold(),
        _retrieval(["b1"]),
        {"answer": {"claims": []}},
        [{"chunk_id": "c2", "paper_id": "p2", "block_ids": ["b2"], "estimated_tokens": 10}],
    )
    funnel = exclusive_failure_funnel([trace])
    assert funnel["counts"]["C0_CONTEXT_SELECTION_DROP"] == 1


def test_exclusive_failure_funnel_generation_omission() -> None:
    trace = build_trace(
        _gold(),
        _retrieval(["b1"]),
        {"answer": {"claims": []}},
        [{"chunk_id": "c1", "paper_id": "p1", "block_ids": ["b1"], "estimated_tokens": 10}],
    )
    funnel = exclusive_failure_funnel([trace])
    assert funnel["counts"]["G0_GENERATION_OMISSION"] == 1


def test_exclusive_failure_funnel_citation_failure() -> None:
    answer = {"answer": {"claims": [{"text": "uses Adam optimizer", "citations": []}]}}
    trace = build_trace(
        _gold(),
        _retrieval(["b1"]),
        answer,
        [{"chunk_id": "c1", "paper_id": "p1", "block_ids": ["b1"], "estimated_tokens": 10}],
    )
    funnel = exclusive_failure_funnel([trace])
    assert funnel["counts"]["CITATION_FAILURE"] == 1


def test_required_claim_context_retention_metric() -> None:
    trace = build_trace(
        _gold(),
        _retrieval(["b1"]),
        {"answer": {"claims": []}},
        [{"chunk_id": "c1", "paper_id": "p1", "block_ids": ["b1"], "estimated_tokens": 10}],
    )
    metrics = aggregate_traces([trace])
    assert metrics["required_claim_context_retention"] == 1.0
    assert metrics["required_claim_evidence_coverage_in_final_context"] == 1.0


def test_hypothesis_gate_uses_context_drop_or_retention() -> None:
    metrics = {
        "answerable_count": 10,
        "exclusive_failure_funnel": {"counts": {"C0_CONTEXT_SELECTION_DROP": 2}},
        "required_claim_context_retention": 0.95,
        "required_claim_evidence_coverage_in_final_context": 0.8,
    }
    assert context_selection_hypothesis_supported(metrics, 0.8) is True


def test_score_budgeted_selector_deduplicates_blocks_and_respects_budget() -> None:
    ranked = [
        {"chunk_id": "c1", "paper_id": "p1", "block_ids": ["b1"], "score": 1.0},
        {"chunk_id": "c2", "paper_id": "p1", "block_ids": ["b1"], "score": 0.9},
        {"chunk_id": "c3", "paper_id": "p2", "block_ids": ["b2"], "score": 0.8},
    ]
    selected = score_budgeted_deduplicated_context(
        ranked, {"c1": "a" * 20, "c2": "b" * 20, "c3": "c" * 20}, token_budget=20
    )
    assert [item["chunk_id"] for item in selected] == ["c1", "c3"]


def test_diversity_selector_limits_initial_paper_concentration() -> None:
    ranked = [
        {"chunk_id": f"c{i}", "paper_id": "p1", "block_ids": [f"b{i}"], "score": 1.0}
        for i in range(4)
    ] + [{"chunk_id": "c4", "paper_id": "p2", "block_ids": ["b4"], "score": 0.5}]
    selected = diversity_aware_context(
        ranked, {f"c{i}": "x" * 20 for i in range(5)}, token_budget=100, paper_cap=2
    )
    assert "p2" in [item["paper_id"] for item in selected]


def test_offline_selector_gate_requires_coverage_gain_and_token_bound() -> None:
    c0 = {
        "required_claim_evidence_coverage_in_final_context": 0.5,
        "full_required_claim_evidence_coverage_in_final_context": 0.2,
        "context_token_p95": 100,
    }
    candidate = {
        "required_claim_evidence_coverage_in_final_context": 0.56,
        "full_required_claim_evidence_coverage_in_final_context": 0.2,
        "context_token_p95": 109,
    }
    assert offline_selector_gate(c0, candidate) is True


def test_reconstruct_baseline_context_uses_top_five_only() -> None:
    ranked = [
        {"chunk_id": f"c{i}", "paper_id": "p", "block_ids": [f"b{i}"], "score": 1}
        for i in range(10)
    ]
    selected = reconstruct_baseline_context(ranked, {}, top_k=5)
    assert len(selected) == 5
