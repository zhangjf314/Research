from __future__ import annotations

from paper_research.evaluation.rag_gold import (
    expansion_plan,
    normalize_gold_record,
    overlap_ratio,
    stratified_split,
    validate_gold_records,
)


def _answerable(question_id: str, *, category: str = "single_hop_factual") -> dict:
    return {
        "question_id": question_id,
        "question": f"What does paper {question_id} claim about retrieval?",
        "category": category,
        "difficulty": "medium",
        "answerable": True,
        "gold_answer": "It claims retrieval must be evaluated with explicit evidence.",
        "required_claims": [
            {
                "claim_id": "C1",
                "text": "Retrieval must be evidence-backed.",
                "gold_block_ids": ["b1"],
            }
        ],
        "gold_paper_ids": ["p1"],
        "gold_pages": [1],
        "gold_block_ids": ["b1"],
        "review_status": "approved",
    }


def test_answerable_contract_requires_structured_claim_evidence() -> None:
    record = _answerable("q001")
    report = validate_gold_records([record], evidence_index={("p1", "b1"): {"page": 1}})
    assert report["valid"]
    assert report["error_count"] == 0


def test_legacy_claim_strings_are_rejected_in_strict_final_mode() -> None:
    record = _answerable("q001")
    record["required_claims"] = ["Retrieval must be evidence-backed."]
    report = validate_gold_records([record], evidence_index={("p1", "b1"): {"page": 1}})
    assert not report["valid"]
    assert any(error["type"] == "required_claims_not_structured" for error in report["errors"])


def test_unanswerable_contract_rejects_gold_evidence() -> None:
    record = {
        "question_id": "q005",
        "question": "What unreported metric is given?",
        "category": "unanswerable",
        "difficulty": "hard",
        "answerable": False,
        "gold_answer": "",
        "required_claims": [],
        "gold_paper_ids": [],
        "gold_pages": [],
        "gold_block_ids": [],
        "unanswerable_reason": "UNREPORTED_METRIC",
        "review_status": "approved",
    }
    report = validate_gold_records([record], evidence_index={})
    assert report["valid"]

    record["gold_block_ids"] = ["b1"]
    report = validate_gold_records([record], evidence_index={})
    assert not report["valid"]
    assert any(error["type"] == "unanswerable_has_gold_block_ids" for error in report["errors"])


def test_duplicate_and_near_duplicate_detection() -> None:
    left = _answerable("q001")
    right = _answerable("q002")
    right["question"] = "What does the paper q001 claim about retrieval?"
    assert overlap_ratio(left["question"], right["question"]) > 0.8
    report = validate_gold_records(
        [left, right],
        evidence_index={("p1", "b1"): {"page": 1}},
        near_duplicate_threshold=0.8,
    )
    assert report["near_duplicate_question_count"] == 1


def test_expansion_plan_counts_only_deficits() -> None:
    records = [
        {"category": "research_background"},
        {"category": "paper_contributions"},
        {"category": "multi_paper_comparison"},
        {"category": "unanswerable"},
    ]
    plan = expansion_plan(records)
    rows = {row["category"]: row for row in plan["category_plan"]}
    assert rows["single_hop_factual"]["current"] == 2
    assert rows["cross_paper_comparison"]["current"] == 1
    assert rows["unanswerable"]["deficit"] == 14


def test_split_is_deterministic_disjoint_and_complete() -> None:
    records = [
        normalize_gold_record(_answerable(f"q{index:03d}", category="single_hop_factual"))
        for index in range(1, 11)
    ]
    dev_a, test_a = stratified_split(records, seed=13)
    dev_b, test_b = stratified_split(records, seed=13)
    assert [row["question_id"] for row in dev_a] == [row["question_id"] for row in dev_b]
    assert [row["question_id"] for row in test_a] == [row["question_id"] for row in test_b]
    dev_ids = {row["question_id"] for row in dev_a}
    test_ids = {row["question_id"] for row in test_a}
    assert dev_ids.isdisjoint(test_ids)
    assert dev_ids | test_ids == {row["question_id"] for row in records}
