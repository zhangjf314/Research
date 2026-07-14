import time

from pydantic import BaseModel, Field

from paper_research.providers.llm import (
    GeneratedCitation,
    LLMProvider,
    ModelUsage,
    TemplateLLMProvider,
)
from paper_research.retrieval.context_builder import ContextItem
from paper_research.retrieval.dense import DenseRetriever


class Citation(BaseModel):
    paper_id: str
    section: str | None
    page_start: int
    page_end: int
    quote: str
    score: float
    pdf_url: str
    block_ids: list[str] = Field(default_factory=list)


class AnswerClaim(BaseModel):
    claim_id: str
    text: str
    citations: list[GeneratedCitation]
    block_ids: list[str]
    pages: list[int]
    supported: bool = True
    support_note: str | None = None


class AnswerLatency(BaseModel):
    retrieval_latency_ms: float = 0
    rerank_latency_ms: float = 0
    context_build_latency_ms: float = 0
    llm_first_token_latency_ms: float | None = None
    llm_total_latency_ms: float = 0
    total_latency_ms: float = 0


class Answer(BaseModel):
    answerable: bool
    answer: str | None
    claims: list[AnswerClaim] = Field(default_factory=list)
    refusal_reason: str | None = None
    refused: bool
    citations: list[Citation] = Field(default_factory=list)
    uncertainty: str | None = None
    insufficient_evidence: bool = False
    model_usage: ModelUsage = Field(default_factory=ModelUsage)
    latency: AnswerLatency = Field(default_factory=AnswerLatency)
    provider: str = "template"
    model: str = "template-v1"
    prompt_version: str = "claim-qa-v1"
    api_request_count: int = 0
    retry_count: int = 0
    retry_reasons: list[str] = Field(default_factory=list)
    rate_limit_events: int = 0


class ClaimValidationError(ValueError):
    pass


class ClaimEvidenceValidator:
    def validate(self, claims: list, context: list[ContextItem]) -> list[AnswerClaim]:
        allowed = set()
        for item in context:
            block_ids = item.block_ids or [item.chunk_id]
            if item.block_page_map:
                allowed.update(
                    (item.paper_id, item.block_page_map[block_id], block_id)
                    for block_id in block_ids
                )
            else:
                allowed.update(
                    (item.paper_id, item.page_start, block_id)
                    for block_id in block_ids
                )
        accepted = []
        for claim in claims:
            if not claim.citations:
                raise ClaimValidationError(f"claim {claim.claim_id} has no citation")
            for citation in claim.citations:
                key = (citation.paper_id, citation.page, citation.block_id)
                if key not in allowed:
                    raise ClaimValidationError(
                        f"claim {claim.claim_id} cites evidence outside supplied context"
                    )
            accepted.append(
                AnswerClaim(
                    claim_id=claim.claim_id,
                    text=claim.text,
                    citations=claim.citations,
                    block_ids=list(dict.fromkeys(c.block_id for c in claim.citations)),
                    pages=list(dict.fromkeys(c.page for c in claim.citations)),
                    support_note="citation identifiers validated against supplied context",
                )
            )
        return accepted


class QAService:
    """Compatibility dense QA plus strict claim-level cited generation."""

    def __init__(
        self,
        retriever: DenseRetriever | None = None,
        score_threshold: float = 0.12,
        llm: LLMProvider | None = None,
        prompt_version: str = "claim-qa-v1",
    ) -> None:
        self.retriever = retriever
        self.score_threshold = score_threshold
        self.llm = llm or TemplateLLMProvider()
        self.prompt_version = prompt_version
        self.validator = ClaimEvidenceValidator()

    def answer(self, question: str, paper_ids: list[str] | None, top_k: int = 5) -> Answer:
        if self.retriever is None:
            raise RuntimeError("dense retriever is not configured")
        started = time.perf_counter()
        results = self.retriever.retrieve(
            question,
            paper_ids=paper_ids,
            top_k=top_k,
            score_threshold=self.score_threshold,
        )
        retrieval_ms = round((time.perf_counter() - started) * 1000, 3)
        context = [
            ContextItem(
                chunk_id=result.chunk.chunk_id,
                paper_id=result.chunk.paper_id,
                block_ids=result.chunk.block_ids,
                section_path=result.chunk.section_path,
                page_start=result.chunk.page_start,
                page_end=result.chunk.page_end,
                evidence=result.chunk.chunk_text,
                score=result.score,
            )
            for result in results
        ]
        return self.answer_from_context(
            question,
            context,
            retrieval_latency_ms=retrieval_ms,
            total_started=started,
        )

    def answer_from_context(
        self,
        question: str,
        context: list[ContextItem],
        *,
        retrieval_latency_ms: float = 0,
        rerank_latency_ms: float = 0,
        context_build_latency_ms: float = 0,
        total_started: float | None = None,
    ) -> Answer:
        started = total_started or time.perf_counter()
        generation = self.llm.generate_claim_answer(question, context, self.prompt_version)
        accepted = self.validator.validate(generation.claims, context)
        if generation.answerable and not accepted:
            raise ClaimValidationError("answerable response has no validated claims")
        if not generation.answerable and accepted:
            raise ClaimValidationError("unanswerable response cannot have claims")
        cited_ids = {citation.block_id for claim in accepted for citation in claim.citations}
        citations = [
            Citation(
                paper_id=item.paper_id,
                section=" > ".join(item.section_path) or None,
                page_start=item.page_start,
                page_end=item.page_end,
                quote=item.evidence[:500],
                score=round(item.score, 6),
                pdf_url=f"/api/v1/papers/{item.paper_id}/pdf#page={item.page_start}",
                block_ids=[block_id for block_id in item.block_ids if block_id in cited_ids],
            )
            for item in context
            if set(item.block_ids or [item.chunk_id]) & cited_ids
        ]
        return Answer(
            answerable=generation.answerable,
            answer=generation.answer,
            claims=accepted,
            refusal_reason=generation.refusal_reason,
            refused=not generation.answerable,
            citations=citations,
            insufficient_evidence=not generation.answerable,
            model_usage=generation.usage,
            latency=AnswerLatency(
                retrieval_latency_ms=retrieval_latency_ms,
                rerank_latency_ms=rerank_latency_ms,
                context_build_latency_ms=context_build_latency_ms,
                llm_first_token_latency_ms=generation.first_token_latency_ms,
                llm_total_latency_ms=generation.total_latency_ms,
                total_latency_ms=round((time.perf_counter() - started) * 1000, 3),
            ),
            provider=self.llm.provider_name,
            model=self.llm.model_name,
            prompt_version=self.prompt_version,
            api_request_count=generation.api_request_count,
            retry_count=generation.retry_count,
            retry_reasons=generation.retry_reasons,
            rate_limit_events=generation.rate_limit_events,
        )
