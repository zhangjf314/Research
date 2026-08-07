from __future__ import annotations

# ruff: noqa: E501
from dataclasses import dataclass
from typing import Any

from paper_research.evaluation.rag_gold import evidence_pairs, validate_gold_records


@dataclass(frozen=True)
class ReviewResult:
    question_id: str
    decision: str
    question_quality: str
    answer_quality: str
    claim_reviews: list[dict[str, Any]]
    ambiguity: bool
    duplicate_risk: bool
    suggested_revision: dict[str, Any] | None
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "decision": self.decision,
            "question_quality": self.question_quality,
            "answer_quality": self.answer_quality,
            "claim_reviews": self.claim_reviews,
            "ambiguity": self.ambiguity,
            "duplicate_risk": self.duplicate_risk,
            "suggested_revision": self.suggested_revision,
            "reason": self.reason,
        }


def review_gold_record(
    record: dict[str, Any],
    *,
    evidence_index: dict[tuple[str, str], dict[str, Any]],
    duplicate_risk: bool = False,
) -> ReviewResult:
    question_id = str(record.get("question_id", ""))
    if duplicate_risk:
        return ReviewResult(
            question_id=question_id,
            decision="REJECT",
            question_quality="FAIL",
            answer_quality="NOT_APPLICABLE",
            claim_reviews=[],
            ambiguity=False,
            duplicate_risk=True,
            suggested_revision=None,
            reason="Duplicate or near-duplicate benchmark capability cluster.",
        )

    deterministic = validate_gold_records(
        [record],
        evidence_index=evidence_index,
        strict_structured_claims=True,
        near_duplicate_threshold=1.1,
    )
    if deterministic["error_count"]:
        return ReviewResult(
            question_id=question_id,
            decision="REJECT",
            question_quality="FAIL",
            answer_quality="FAIL",
            claim_reviews=[],
            ambiguity=False,
            duplicate_risk=False,
            suggested_revision=None,
            reason=f"Deterministic validation failed: {deterministic['errors'][:3]}",
        )

    if not record.get("answerable"):
        return ReviewResult(
            question_id=question_id,
            decision="APPROVE",
            question_quality="PASS",
            answer_quality="PASS",
            claim_reviews=[],
            ambiguity=False,
            duplicate_risk=False,
            suggested_revision=None,
            reason="Unanswerable hard negative has no gold answer or gold blocks and includes a verified unanswerable reason.",
        )

    available_pairs = set(evidence_pairs(record))
    claim_reviews: list[dict[str, Any]] = []
    for claim in record.get("required_claims", []):
        claim_pairs = []
        for item in claim.get("gold_evidence", []):
            claim_pairs.append((str(item.get("paper_id", "")), str(item.get("block_id", ""))))
        if not claim_pairs:
            paper_ids = [str(pid) for pid in record.get("gold_paper_ids", [])]
            if len(paper_ids) == 1:
                claim_pairs = [(paper_ids[0], str(block_id)) for block_id in claim.get("gold_block_ids", [])]
        support = "SUPPORTED" if claim_pairs and all(pair in available_pairs and pair in evidence_index for pair in claim_pairs) else "NOT_SUPPORTED"
        claim_reviews.append(
            {
                "claim_id": claim.get("claim_id"),
                "support": support,
                "evidence_ids": [f"{paper_id}:{block_id}" for paper_id, block_id in claim_pairs],
                "reason": "Claim has explicit evidence pairs that exist in the frozen evidence corpus."
                if support == "SUPPORTED"
                else "Claim is missing explicit valid evidence pairs.",
            }
        )

    if not claim_reviews or any(item["support"] != "SUPPORTED" for item in claim_reviews):
        return ReviewResult(
            question_id=question_id,
            decision="REJECT",
            question_quality="PASS",
            answer_quality="FAIL",
            claim_reviews=claim_reviews,
            ambiguity=False,
            duplicate_risk=False,
            suggested_revision=None,
            reason="At least one required claim is not fully supported by explicit evidence pairs.",
        )

    return ReviewResult(
        question_id=question_id,
        decision="APPROVE",
        question_quality="PASS",
        answer_quality="PASS",
        claim_reviews=claim_reviews,
        ambiguity=False,
        duplicate_risk=False,
        suggested_revision=None,
        reason="Question, answer, claims, pages, blocks, and explicit evidence pairs passed evidence-grounded review.",
    )
