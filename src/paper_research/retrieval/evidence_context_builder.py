from __future__ import annotations

from collections import Counter

from pydantic import BaseModel

from paper_research.generation.claim_evidence_selector import ClaimEvidenceAllocation


class EvidenceContextTraceItem(BaseModel):
    claim_id: str
    evidence_id: str
    original_retrieval_rank: int | None
    evidence_score: float
    selected_reason: str
    token_count: int
    final_order: int | None = None
    truncated: bool = False
    citation_triple: tuple[str, int, str]
    evidence_role: list[str]


class EvidenceContextItem(BaseModel):
    evidence_id: str
    claim_ids: list[str]
    paper_id: str
    page: int
    block_id: str
    text: str
    allowed_citation_triple: tuple[str, int, str]
    evidence_roles: list[str]


class EvidenceContextResult(BaseModel):
    context: list[EvidenceContextItem]
    trace: list[EvidenceContextTraceItem]
    total_tokens: int
    truncated_claim_ids: list[str]


class EvidenceContextBuilder:
    def __init__(
        self,
        *,
        max_tokens: int = 3000,
        max_units_per_section: int = 3,
    ) -> None:
        self.max_tokens = max_tokens
        self.max_units_per_section = max_units_per_section

    def build(self, allocations: list[ClaimEvidenceAllocation]) -> EvidenceContextResult:
        context: list[EvidenceContextItem] = []
        traces: list[EvidenceContextTraceItem] = []
        by_evidence: dict[str, EvidenceContextItem] = {}
        section_counts: Counter[tuple[str, str | None]] = Counter()
        used_tokens = 0
        truncated_claims: set[str] = set()
        ordered = sorted(
            (
                (allocation.claim_id, candidate)
                for allocation in allocations
                for candidate in allocation.selected_evidence
            ),
            key=lambda item: (-item[1].total_score, item[0], item[1].evidence.evidence_id),
        )
        for claim_id, candidate in ordered:
            unit = candidate.evidence
            tokens = max(1, (len(unit.text) + 3) // 4)
            section_key = (unit.paper_id, unit.section_id)
            trace = EvidenceContextTraceItem(
                claim_id=claim_id,
                evidence_id=unit.evidence_id,
                original_retrieval_rank=candidate.original_retrieval_rank,
                evidence_score=candidate.total_score,
                selected_reason="minimum_claim_evidence_set",
                token_count=tokens,
                citation_triple=unit.citation_triple,
                evidence_role=list(unit.evidence_roles),
            )
            if unit.evidence_id in by_evidence:
                by_evidence[unit.evidence_id].claim_ids.append(claim_id)
                trace.selected_reason = "deduplicated_shared_evidence"
                trace.final_order = next(
                    index + 1
                    for index, item in enumerate(context)
                    if item.evidence_id == unit.evidence_id
                )
                traces.append(trace)
                continue
            if section_counts[section_key] >= self.max_units_per_section:
                trace.selected_reason = "section_cap"
                trace.truncated = True
                truncated_claims.add(claim_id)
                traces.append(trace)
                continue
            if used_tokens + tokens > self.max_tokens:
                trace.selected_reason = "token_budget"
                trace.truncated = True
                truncated_claims.add(claim_id)
                traces.append(trace)
                continue
            item = EvidenceContextItem(
                evidence_id=unit.evidence_id,
                claim_ids=[claim_id],
                paper_id=unit.paper_id,
                page=unit.page,
                block_id=unit.block_id,
                text=unit.text,
                allowed_citation_triple=unit.citation_triple,
                evidence_roles=list(unit.evidence_roles),
            )
            context.append(item)
            by_evidence[unit.evidence_id] = item
            section_counts[section_key] += 1
            used_tokens += tokens
            trace.final_order = len(context)
            traces.append(trace)
        return EvidenceContextResult(
            context=context,
            trace=traces,
            total_tokens=used_tokens,
            truncated_claim_ids=sorted(truncated_claims),
        )
