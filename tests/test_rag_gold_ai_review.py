from __future__ import annotations

from paper_research.evaluation.gold_reviewer import review_gold_record
from paper_research.evaluation.rag_gold import validate_gold_records


def _valid_record() -> dict:
    return {
        "question_id": "q001",
        "question": "What evidence-backed method is reported?",
        "category": "methods_and_experiments",
        "difficulty": "medium",
        "answerable": True,
        "gold_answer": "The method is evidence-backed.",
        "required_claims": [
            {
                "claim_id": "C1",
                "text": "The method is evidence-backed.",
                "gold_block_ids": ["b1"],
                "gold_evidence": [{"paper_id": "p1", "block_id": "b1"}],
            }
        ],
        "gold_paper_ids": ["p1"],
        "gold_pages": [1],
        "gold_block_ids": ["b1"],
        "gold_evidence": [{"paper_id": "p1", "block_id": "b1"}],
        "review_status": "approved",
    }


def test_ai_reviewer_approves_supported_record() -> None:
    review = review_gold_record(_valid_record(), evidence_index={("p1", "b1"): {"page": 1}})
    assert review.decision == "APPROVE"
    assert review.claim_reviews[0]["support"] == "SUPPORTED"


def test_ai_reviewer_rejects_unsupported_claim() -> None:
    record = _valid_record()
    record["required_claims"][0]["gold_evidence"] = [{"paper_id": "p1", "block_id": "missing"}]
    review = review_gold_record(record, evidence_index={("p1", "b1"): {"page": 1}})
    assert review.decision == "REJECT"


def test_ai_reviewer_rejects_duplicate_risk() -> None:
    review = review_gold_record(
        _valid_record(),
        evidence_index={("p1", "b1"): {"page": 1}},
        duplicate_risk=True,
    )
    assert review.decision == "REJECT"
    assert review.duplicate_risk


def test_benchmark_admission_requires_review_and_validator() -> None:
    record = _valid_record()
    review = review_gold_record(record, evidence_index={("p1", "b1"): {"page": 1}})
    validation = validate_gold_records([record], evidence_index={("p1", "b1"): {"page": 1}})
    assert review.decision == "APPROVE"
    assert validation["valid"]


def test_cross_paper_contract_requires_two_papers() -> None:
    record = _valid_record()
    record["category"] = "cross_paper_comparison"
    validation = validate_gold_records([record], evidence_index={("p1", "b1"): {"page": 1}})
    assert not validation["valid"]
    assert any(error["type"] == "cross_paper_requires_two_papers" for error in validation["errors"])


def test_multi_evidence_contract_requires_two_blocks() -> None:
    record = _valid_record()
    record["category"] = "multi_evidence_synthesis"
    validation = validate_gold_records([record], evidence_index={("p1", "b1"): {"page": 1}})
    assert not validation["valid"]
    assert any(
        error["type"] == "multi_evidence_requires_two_blocks"
        for error in validation["errors"]
    )
