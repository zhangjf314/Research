import time

from pydantic import BaseModel, Field

from paper_research.chunking.tokenizer import tokenize
from paper_research.providers.llm import LLMProvider, ModelUsage, TemplateLLMProvider
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
    text: str
    block_ids: list[str]
    pages: list[int]
    supported: bool
    support_note: str | None = None


class AnswerLatency(BaseModel):
    retrieval_latency_ms: float = 0
    rerank_latency_ms: float = 0
    context_build_latency_ms: float = 0
    llm_first_token_latency_ms: float | None = None
    llm_total_latency_ms: float = 0
    total_latency_ms: float = 0


class Answer(BaseModel):
    answer: str
    refused: bool
    citations: list[Citation] = Field(default_factory=list)
    uncertainty: str | None = None
    claims: list[AnswerClaim] = Field(default_factory=list)
    insufficient_evidence: bool = False
    model_usage: ModelUsage = Field(default_factory=ModelUsage)
    latency: AnswerLatency = Field(default_factory=AnswerLatency)
    provider: str = "template"
    model: str = "template-v1"
    prompt_version: str = "claim-qa-v1"


class ClaimEvidenceValidator:
    def validate(
        self, claims: list, context: list[ContextItem]
    ) -> tuple[list[AnswerClaim], list[str]]:
        by_id = {item.chunk_id: item for item in context}
        accepted: list[AnswerClaim] = []
        removed: list[str] = []
        for claim in claims:
            bound = [by_id[block_id] for block_id in claim.block_ids if block_id in by_id]
            allowed_pages = {
                page
                for item in bound
                for page in range(item.page_start, item.page_end + 1)
            }
            terms = {token.lower() for token in tokenize(claim.text) if token.isalnum()}
            evidence_terms = {
                token.lower()
                for item in bound
                for token in tokenize(item.evidence)
                if token.isalnum()
            }
            overlap = len(terms & evidence_terms) / max(1, len(terms))
            supported = bool(bound) and set(claim.pages).issubset(allowed_pages) and overlap >= 0.35
            if not supported:
                removed.append(claim.text)
                continue
            accepted.append(
                AnswerClaim(
                    text=claim.text,
                    block_ids=claim.block_ids,
                    pages=claim.pages,
                    supported=True,
                    support_note=f"lexical evidence overlap={overlap:.3f}",
                )
            )
        return accepted, removed


class QAService:
    """Compatibility dense QA plus claim-aware generation from retrieved context."""

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
        accepted, removed = self.validator.validate(generation.claims, context)
        insufficient = generation.insufficient_evidence or not accepted
        answer_text = (
            "The available evidence is insufficient."
            if insufficient
            else "\n\n".join(claim.text for claim in accepted)
        )
        cited_ids = {block_id for claim in accepted for block_id in claim.block_ids}
        citations = [
            Citation(
                paper_id=item.paper_id,
                section=" > ".join(item.section_path) or None,
                page_start=item.page_start,
                page_end=item.page_end,
                quote=item.evidence[:500],
                score=round(item.score, 6),
                pdf_url=f"/api/v1/papers/{item.paper_id}/pdf#page={item.page_start}",
                block_ids=[item.chunk_id],
            )
            for item in context
            if item.chunk_id in cited_ids
        ]
        return Answer(
            answer=answer_text,
            refused=insufficient,
            citations=citations,
            uncertainty=(
                f"Removed {len(removed)} unsupported claim(s)." if removed else None
            ),
            claims=accepted,
            insufficient_evidence=insufficient,
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
        )
