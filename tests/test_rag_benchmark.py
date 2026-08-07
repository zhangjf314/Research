from __future__ import annotations

from paper_research.evaluation.rag_benchmark import (
    aggregate_generation,
    aggregate_retrieval,
    audit_gold,
    canonical_json_hash,
    classify_bad_case,
    dataset_hash,
    evaluate_generation_item,
    evaluate_retrieval_question,
    ndcg_at,
)


def test_retrieval_metric_correctness() -> None:
    gold = {
        "question_id": "q1",
        "answerable": True,
        "gold_block_ids": ["b1", "b2"],
        "gold_paper_ids": ["p1"],
    }
    rows = [
        {"paper_id": "p2", "block_id": "x"},
        {"paper_id": "p1", "block_id": "b2"},
        {"paper_id": "p1", "block_id": "b1"},
    ]

    result = evaluate_retrieval_question(gold, rows)

    assert result["recall_at_5"] == 1.0
    assert result["paper_recall_at_5"] == 1.0
    assert result["evidence_coverage_at_5"] == 1.0
    assert result["mrr_at_10"] == 0.5
    assert round(result["ndcg_at_10"], 6) == round(ndcg_at([False, True, True], 2, 10), 6)


def test_retrieval_unanswerable_irrelevant_rate() -> None:
    result = evaluate_retrieval_question(
        {"question_id": "q2", "answerable": False, "gold_block_ids": []},
        [{"paper_id": "p1", "block_id": "b1"}],
    )

    assert result["irrelevant_retrieval_rate"] == 1.0
    assert result["answerable"] is False


def test_aggregate_retrieval_keeps_unanswerable_separate() -> None:
    rows = [
        {
            "question_id": "q1",
            "answerable": True,
            "recall_at_5": 1.0,
            "recall_at_10": 1.0,
            "recall_at_20": 1.0,
            "mrr_at_10": 1.0,
            "ndcg_at_10": 1.0,
            "paper_recall_at_5": 1.0,
            "paper_recall_at_10": 1.0,
            "evidence_coverage_at_5": 1.0,
            "evidence_coverage_at_10": 1.0,
            "evidence_coverage_at_20": 1.0,
        },
        {"question_id": "q2", "answerable": False, "irrelevant_retrieval_rate": 1.0},
    ]

    aggregate = aggregate_retrieval(rows)

    assert aggregate["answerable_count"] == 1
    assert aggregate["unanswerable_count"] == 1
    assert aggregate["recall_at_5"] == 1.0
    assert aggregate["irrelevant_retrieval_rate"] == 1.0


def test_generation_metrics_and_failure_stage() -> None:
    item = {
        "question_id": "q1",
        "status": "COMPLETED",
        "gold": {"answerable": True},
        "gold_block_present": False,
        "answer": {
            "answerable": True,
            "model_usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
                "estimated_cost_usd": 0.1,
            },
        },
        "metrics": {
            "required_claim_coverage": 0.5,
            "claim_citation_binding_rate": 1.0,
            "citation_precision": 0.25,
            "citation_recall": 0.2,
        },
        "wall_ms": 123.0,
    }

    row = evaluate_generation_item(item)
    aggregate = aggregate_generation([row])

    assert row["failure_stage"] == "retrieval failed"
    assert aggregate["retrieval_failed_count"] == 1
    assert aggregate["required_claim_coverage"] == 0.5
    assert aggregate["total_tokens"] == 15
    assert aggregate["cost"] == 0.1


def test_hashes_are_reproducible_and_ordered_by_question_id() -> None:
    left = [{"question_id": "b", "value": 2}, {"question_id": "a", "value": 1}]
    right = [{"question_id": "a", "value": 1}, {"question_id": "b", "value": 2}]

    assert dataset_hash(left) == dataset_hash(right)
    assert canonical_json_hash({"b": 2, "a": 1}) == canonical_json_hash({"a": 1, "b": 2})


def test_gold_audit_schema_and_gap_plan() -> None:
    audit = audit_gold(
        [
            {
                "question_id": "q1",
                "review_status": "approved",
                "answerable": True,
                "category": "method",
                "difficulty": "easy",
                "required_claims": ["c1"],
                "gold_paper_ids": ["p1"],
                "gold_pages": [1],
                "gold_block_ids": ["b1"],
            },
            {
                "question_id": "q2",
                "review_status": "approved",
                "answerable": False,
                "category": "unanswerable",
                "difficulty": "hard",
                "required_claims": [],
                "gold_paper_ids": [],
                "gold_pages": [],
                "gold_block_ids": [],
            },
        ]
    )

    assert audit["total"] == 2
    assert audit["answerable"] == 1
    assert audit["unanswerable"] == 1
    assert audit["gold_evidence_coverage"]["complete_for_answerable"] is True
    assert audit["gap_plan"]["recommended_target_gold_count"] == 150


def test_bad_case_classification() -> None:
    failure_stage, bad_type = classify_bad_case(
        {
            "question_id": "q1",
            "status": "COMPLETED",
            "gold": {"answerable": True},
            "gold_block_present": False,
            "metrics": {"required_claim_coverage": 1.0},
        }
    )

    assert failure_stage == "retrieval"
    assert bad_type == "RETRIEVAL_MISS"
