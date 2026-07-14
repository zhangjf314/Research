from __future__ import annotations

from pydantic import BaseModel, Field

from paper_research.evidence.claims import ClaimUnit
from paper_research.retrieval.evidence_retriever import EvidenceCandidate


class ClaimEvidenceAllocation(BaseModel):
    claim_id: str
    candidate_evidence: list[EvidenceCandidate] = Field(default_factory=list)
    selected_evidence: list[EvidenceCandidate] = Field(default_factory=list)
    rejected_evidence: list[EvidenceCandidate] = Field(default_factory=list)
    evidence_complete: bool = False
    evidence_confidence: float = 0.0
    token_cost: int = 0
    missing_evidence_reason: str | None = None
    unsupported_before_generation: bool = False


class ClaimFirstEvidenceSelector:
    def __init__(self, *, minimum_score: float = 0.18, max_per_claim: int = 2) -> None:
        self.minimum_score = minimum_score
        self.max_per_claim = max_per_claim

    def select(
        self,
        claim: ClaimUnit,
        candidates: list[EvidenceCandidate],
    ) -> ClaimEvidenceAllocation:
        eligible = [
            item
            for item in candidates
            if not item.rejection_reasons and item.total_score >= self.minimum_score
        ]
        target_papers = list(dict.fromkeys(claim.target_paper_ids))
        multi_paper = claim.question_type == "multi_paper" and len(target_papers) > 1
        needed = len(target_papers) if multi_paper else 2 if claim.multi_block_required else 1
        selected: list[EvidenceCandidate] = []
        used_blocks: set[tuple[str, str]] = set()
        if multi_paper:
            for paper_id in target_papers:
                paper_candidate = next(
                    (item for item in eligible if item.evidence.paper_id == paper_id), None
                )
                if paper_candidate is not None:
                    selected.append(paper_candidate)
                    used_blocks.add((paper_id, paper_candidate.evidence.block_id))
        for candidate in eligible:
            if len(selected) >= needed:
                break
            key = (candidate.evidence.paper_id, candidate.evidence.block_id)
            if key in used_blocks:
                continue
            selected.append(candidate)
            used_blocks.add(key)
        paper_complete = not multi_paper or set(target_papers).issubset(
            {item.evidence.paper_id for item in selected}
        )
        complete = bool(selected) and len(selected) >= needed and paper_complete
        if not claim.expected_answerability:
            selected = []
            complete = False
        selected_ids = {item.evidence.evidence_id for item in selected}
        rejected = [item for item in candidates if item.evidence.evidence_id not in selected_ids]
        confidence = min((item.total_score for item in selected), default=0.0)
        return ClaimEvidenceAllocation(
            claim_id=claim.claim_id,
            candidate_evidence=candidates,
            selected_evidence=selected,
            rejected_evidence=rejected,
            evidence_complete=complete,
            evidence_confidence=round(confidence, 8),
            token_cost=sum(max(1, (len(item.evidence.text) + 3) // 4) for item in selected),
            missing_evidence_reason=(
                None
                if complete
                else "unanswerable_requires_refusal"
                if not claim.expected_answerability
                else "no_candidate_meets_evidence_threshold"
            ),
            unsupported_before_generation=not complete,
        )
