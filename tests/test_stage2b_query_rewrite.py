from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from paper_research.evaluation import rag_stage2b
from paper_research.evaluation.rag_stage2b import (
    Decomposition,
    add_extended_row_metrics,
    assert_no_gold_leakage,
    cache_key,
    decomposition_user_prompt,
    deduplicate_queries,
    drift_labels,
    fuse_ranked_contexts,
    queries_for_config,
    read_cache,
    single_rewrite_user_prompt,
    write_cache,
)


def test_rewrite_schema_accepts_single_query() -> None:
    assert rag_stage2b.SingleRewrite.model_validate(
        {"rewritten_query": "Transformer attention-only sequence transduction experiments"}
    ).rewritten_query


def test_decomposition_schema_rejects_more_than_three_queries() -> None:
    with pytest.raises(ValidationError):
        Decomposition.model_validate(
            {
                "queries": [
                    {"query": "q1", "purpose": "p1"},
                    {"query": "q2", "purpose": "p2"},
                    {"query": "q3", "purpose": "p3"},
                    {"query": "q4", "purpose": "p4"},
                ]
            }
        )


def test_rewrite_prompt_does_not_require_gold_fields() -> None:
    gold = {
        "question": "What optimizer did the paper use?",
        "gold_answer": "Adam",
        "gold_block_ids": ["b000025"],
        "gold_pages": [2],
        "required_claims": [{"text": "The optimizer is Adam."}],
    }
    assert_no_gold_leakage(single_rewrite_user_prompt(gold["question"]), gold)
    assert_no_gold_leakage(decomposition_user_prompt(gold["question"]), gold)


def test_rewrite_prompt_leakage_is_rejected() -> None:
    gold = {"gold_answer": "Adam", "gold_block_ids": ["b000025"]}
    with pytest.raises(ValueError, match="leaks gold"):
        assert_no_gold_leakage("Use Adam from b000025", gold)


def test_cache_roundtrip_uses_sanitized_payload(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(rag_stage2b, "STAGE2B_CACHE", tmp_path)
    key = cache_key("q001", "prompt-v1", "abc", "model")
    payload = {"question_id": "q001", "single_rewrite": "attention model"}
    write_cache("single", key, payload)
    assert read_cache("single", key) == payload


def test_query_config_original_retention_and_failure_fallback() -> None:
    original = "What did the target paper compare?"
    item = {
        "single_status": "failed",
        "decomposition_status": "failed",
        "single_rewrite": None,
        "decomposition_queries": [],
    }
    assert queries_for_config("Q1_SINGLE_REWRITE_REPLACE", original, item) == ([], True)
    assert queries_for_config("Q2_ORIGINAL_PLUS_SINGLE_REWRITE", original, item) == (
        [original],
        True,
    )
    assert queries_for_config("Q3_ORIGINAL_PLUS_DECOMPOSITION", original, item) == (
        [original],
        True,
    )


def test_query_config_decomposition_keeps_original_and_caps_to_four() -> None:
    original = "Compare GPT-3 scaling results."
    item = {
        "decomposition_status": "success",
        "decomposition_queries": ["scaling laws", "datasets", "few-shot tasks"],
    }
    queries, fallback = queries_for_config("Q3_ORIGINAL_PLUS_DECOMPOSITION", original, item)
    assert fallback is False
    assert queries == [original, "scaling laws", "datasets", "few-shot tasks"]
    assert len(queries) <= 4


def test_deduplicate_queries_is_case_and_space_insensitive() -> None:
    assert deduplicate_queries(["Alpha Beta", " alpha   beta ", "Gamma"]) == [
        "Alpha Beta",
        "Gamma",
    ]


def test_multi_query_fusion_is_deterministic_and_deduplicates() -> None:
    left = [
        {"paper_id": "p1", "chunk_id": "c1", "block_ids": ["b1"], "score": 0.9},
        {"paper_id": "p2", "chunk_id": "c2", "block_ids": ["b2"], "score": 0.8},
    ]
    right = [
        {"paper_id": "p2", "chunk_id": "c2b", "block_ids": ["b2"], "score": 0.7},
        {"paper_id": "p3", "chunk_id": "c3", "block_ids": ["b3"], "score": 0.6},
    ]
    first = fuse_ranked_contexts([left, right], top_k=3)
    second = fuse_ranked_contexts([left, right], top_k=3)
    assert first == second
    assert [item["paper_id"] for item in first] == ["p2", "p1", "p3"]
    assert len(first) == 3


def test_drift_detector_flags_numeric_constraint_drop() -> None:
    assert "NUMERIC_CONSTRAINT_DROPPED" in drift_labels(
        "What is the 2048-token context result?",
        ["context result"],
    )


def test_add_extended_row_metrics_supports_required_claim_bootstrap() -> None:
    row = {
        "answerable": True,
        "retrieval_failure": None,
        "metrics": {"evidence_coverage_at_10": 1.0},
        "ranked_results": [{"block_ids": ["b1"]}],
    }
    gold = {"required_claims": [{"gold_block_ids": ["b1"]}]}
    add_extended_row_metrics(row, gold)
    assert row["metrics"]["required_claim_evidence_coverage_at_10"] == 1.0
    assert row["metrics"]["full_evidence_coverage_at_10"] == 1.0


def test_plan_has_test_split_forbidden() -> None:
    from scripts.run_rag_stage2b_query_rewrite_v1 import plan_payload

    payload = plan_payload()
    assert payload["split"] == "dev"
    assert payload["test_questions_allowed"] is False
    assert payload["test_questions_evaluated"] == 0


def test_report_records_reranker_disabled_and_no_generation() -> None:
    from scripts.run_rag_stage2b_query_rewrite_v1 import CONFIGS, report_markdown

    metrics = {
        config_id: {
            "recall_at_10": 0,
            "evidence_coverage_at_10": 0,
            "required_claim_evidence_coverage_at_10": 0,
            "full_evidence_coverage_at_10": 0,
            "mrr_at_10": 0,
            "ndcg_at_10": 0,
            "paper_recall_at_10": 0,
            "latency_ms": {"p95": 0},
        }
        for config_id in CONFIGS
    }
    text = report_markdown(
        {
            "split": "dev",
            "dev_question_count": 98,
            "test_questions_evaluated": 0,
            "test_protocol_violation": False,
            "selected_stage2a_candidate": "Current Hybrid",
            "selected_candidate": "Q0_CURRENT_HYBRID",
            "reranker_enabled": False,
            "llm_generation_requests": 0,
            "metrics": metrics,
            "rewrite_usage": {},
            "selection_gate": {},
        }
    )
    assert "reranker_enabled: `False`" in text
    assert "llm_generation_requests: `0`" in text


def test_no_raw_provider_response_fields_in_public_audit() -> None:
    from scripts.run_rag_stage2b_query_rewrite_v1 import public_rewrite_audit

    audit = public_rewrite_audit(
        [
            {
                "question_id": "q001",
                "single_rewrite": "query",
                "decomposition_queries": [],
                "provider": "deepseek",
                "model": "deepseek-v4-flash",
                "single_status": "success",
                "decomposition_status": "success",
                "raw_provider_response": {"secret": True},
            }
        ]
    )
    dumped = json.dumps(audit)
    assert "raw_provider_response" not in dumped
    assert "secret" not in dumped
