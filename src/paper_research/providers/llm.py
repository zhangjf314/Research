import json
import math
import time
from abc import ABC, abstractmethod
from typing import Any

import httpx
from pydantic import BaseModel, Field, ValidationError, model_validator

from paper_research.generation.prompts import qa_system_prompt
from paper_research.retrieval.context_builder import ContextItem


class ModelUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float | None = None


class GeneratedCitation(BaseModel):
    paper_id: str = Field(min_length=1)
    page: int = Field(ge=1)
    block_id: str = Field(min_length=1)


class GeneratedClaim(BaseModel):
    claim_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    citations: list[GeneratedCitation] = Field(min_length=1)


class StructuredQA(BaseModel):
    answerable: bool
    answer: str | None
    claims: list[GeneratedClaim]
    refusal_reason: str | None

    @model_validator(mode="after")
    def validate_answer_shape(self) -> "StructuredQA":
        if self.answerable:
            if not self.answer or not self.answer.strip():
                raise ValueError("answerable output requires answer")
            if not self.claims:
                raise ValueError("answerable output requires claims")
            if self.refusal_reason is not None:
                raise ValueError("answerable output cannot contain refusal_reason")
            claim_ids = [claim.claim_id for claim in self.claims]
            if len(claim_ids) != len(set(claim_ids)):
                raise ValueError("claim_id values must be unique")
        else:
            if self.answer is not None:
                raise ValueError("unanswerable output requires answer=null")
            if self.claims:
                raise ValueError("unanswerable output cannot contain claims or citations")
            if not self.refusal_reason or not self.refusal_reason.strip():
                raise ValueError("unanswerable output requires refusal_reason")
        return self


class GenerationResult(StructuredQA):
    usage: ModelUsage = Field(default_factory=ModelUsage)
    first_token_latency_ms: float | None = None
    total_latency_ms: float = 0
    raw_model: str
    api_request_count: int = 0
    retry_count: int = 0
    retry_reasons: list[str] = Field(default_factory=list)
    rate_limit_events: int = 0

    @property
    def insufficient_evidence(self) -> bool:
        return not self.answerable


class LLMProviderError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        api_request_count: int = 0,
        retry_reasons: list[str] | None = None,
        rate_limit_events: int = 0,
    ) -> None:
        super().__init__(message)
        self.api_request_count = api_request_count
        self.retry_reasons = list(retry_reasons or [])
        self.rate_limit_events = rate_limit_events


class LLMProvider(ABC):
    provider_name: str
    model_name: str

    @abstractmethod
    def generate_claim_answer(
        self, question: str, context: list[ContextItem], prompt_version: str
    ) -> GenerationResult:
        """Return a schema-validated answer bound to supplied evidence IDs."""


class TemplateLLMProvider(LLMProvider):
    provider_name = "template"
    model_name = "template-v1"

    def generate_claim_answer(
        self, question: str, context: list[ContextItem], prompt_version: str
    ) -> GenerationResult:
        del question, prompt_version
        started = time.perf_counter()
        if not context:
            return GenerationResult(
                answerable=False,
                answer=None,
                claims=[],
                refusal_reason="The available evidence is insufficient.",
                raw_model=self.model_name,
                total_latency_ms=round((time.perf_counter() - started) * 1000, 3),
            )
        claims = []
        for index, item in enumerate(context[:3], start=1):
            block_id = item.block_ids[0] if item.block_ids else item.chunk_id
            claims.append(
                GeneratedClaim(
                    claim_id=f"c{index}",
                    text=item.evidence[:500],
                    citations=[
                        GeneratedCitation(
                            paper_id=item.paper_id,
                            page=item.page_start,
                            block_id=block_id,
                        )
                    ],
                )
            )
        return GenerationResult(
            answerable=True,
            answer="\n\n".join(claim.text for claim in claims),
            claims=claims,
            refusal_reason=None,
            raw_model=self.model_name,
            total_latency_ms=round((time.perf_counter() - started) * 1000, 3),
        )


class SiliconFlowLLMProvider(LLMProvider):
    provider_name = "siliconflow"

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        temperature: float = 0,
        timeout_seconds: float = 120,
        max_output_tokens: int = 2048,
        max_retries: int = 2,
        input_cost_per_million: float | None = None,
        output_cost_per_million: float | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("SiliconFlow API key is required")
        if not model:
            raise ValueError("SiliconFlow model is required")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_name = model
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self.max_output_tokens = max_output_tokens
        self.max_retries = max_retries
        self.input_cost_per_million = input_cost_per_million
        self.output_cost_per_million = output_cost_per_million
        self.client = client or httpx.Client()

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/chat/completions"

    def generate_claim_answer(
        self, question: str, context: list[ContextItem], prompt_version: str
    ) -> GenerationResult:
        started = time.perf_counter()
        evidence = self._evidence_payload(context)
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": qa_system_prompt(prompt_version)},
                {
                    "role": "user",
                    "content": json.dumps(
                        {"question": question, "evidence": evidence}, ensure_ascii=False
                    ),
                },
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_output_tokens,
            "stream": False,
            "enable_thinking": False,
            "response_format": {"type": "json_object"},
        }
        requests = 0
        rate_limits = 0
        retry_reasons: list[str] = []
        for attempt in range(self.max_retries + 1):
            requests += 1
            try:
                response = self.client.post(
                    self.endpoint,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self.timeout_seconds,
                )
                if response.status_code == 429:
                    rate_limits += 1
                if response.status_code >= 400:
                    reason = f"HTTP {response.status_code}"
                    if response.status_code == 429 or response.status_code >= 500:
                        raise _RetryableLLMError(reason)
                    raise LLMProviderError(reason, api_request_count=requests)
                body = response.json()
                content = body["choices"][0]["message"]["content"]
                parsed = StructuredQA.model_validate(json.loads(content))
                self._validate_context_citations(parsed, context)
                usage_body = body.get("usage") or {}
                usage = self._usage(usage_body)
                return GenerationResult(
                    **parsed.model_dump(),
                    usage=usage,
                    first_token_latency_ms=None,
                    total_latency_ms=round((time.perf_counter() - started) * 1000, 3),
                    raw_model=str(body.get("model") or self.model_name),
                    api_request_count=requests,
                    retry_count=len(retry_reasons),
                    retry_reasons=retry_reasons,
                    rate_limit_events=rate_limits,
                )
            except LLMProviderError:
                raise
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                reason = type(exc).__name__
            except _RetryableLLMError as exc:
                reason = str(exc)
            except json.JSONDecodeError:
                reason = "malformed_json"
            except ValidationError:
                reason = "schema_validation"
            except (KeyError, TypeError):
                reason = "malformed_response"
            except CitationContextError as exc:
                reason = f"citation_validation:{exc.code}"
            except ValueError:
                reason = "citation_validation"
            if attempt >= self.max_retries:
                raise LLMProviderError(
                    f"SiliconFlow QA failed after {requests} request(s): {reason}",
                    api_request_count=requests,
                    retry_reasons=[*retry_reasons, reason],
                    rate_limit_events=rate_limits,
                ) from None
            retry_reasons.append(reason)
            time.sleep(min(2**attempt, 4))
        raise AssertionError("unreachable")

    def _usage(self, body: dict[str, Any]) -> ModelUsage:
        input_tokens = int(body.get("prompt_tokens", 0))
        output_tokens = int(body.get("completion_tokens", 0))
        total_tokens = int(body.get("total_tokens", input_tokens + output_tokens))
        cost = None
        if self.input_cost_per_million is not None and self.output_cost_per_million is not None:
            cost = (
                input_tokens * self.input_cost_per_million
                + output_tokens * self.output_cost_per_million
            ) / 1_000_000
            if not math.isfinite(cost):
                cost = None
        return ModelUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=cost,
        )

    @staticmethod
    def _evidence_payload(context: list[ContextItem]) -> list[dict[str, Any]]:
        output = []
        for item in context:
            output.append(
                {
                    "paper_id": item.paper_id,
                    "block_ids": item.block_ids or [item.chunk_id],
                    "pages": list(range(item.page_start, item.page_end + 1)),
                    "block_page_map": {
                        block_id: item.page_start
                        for block_id in (item.block_ids or [item.chunk_id])
                    },
                    "text": item.evidence,
                }
            )
        return output

    @staticmethod
    def _validate_context_citations(answer: StructuredQA, context: list[ContextItem]) -> None:
        allowed = {
            (item.paper_id, page, block_id)
            for item in context
            for page in range(item.page_start, item.page_end + 1)
            for block_id in (item.block_ids or [item.chunk_id])
        }
        for claim in answer.claims:
            for citation in claim.citations:
                if (citation.paper_id, citation.page, citation.block_id) not in allowed:
                    paper_items = [item for item in context if item.paper_id == citation.paper_id]
                    if not paper_items:
                        raise CitationContextError("paper_id")
                    block_items = [
                        item
                        for item in paper_items
                        if citation.block_id in (item.block_ids or [item.chunk_id])
                    ]
                    if not block_items:
                        raise CitationContextError("block_id")
                    raise CitationContextError("page")


class _RetryableLLMError(RuntimeError):
    pass


class CitationContextError(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__("citation is not present in supplied context")
        self.code = code


class OpenAICompatibleLLMProvider(SiliconFlowLLMProvider):
    """Backward-compatible strict JSON provider using the same chat-completions contract."""

    provider_name = "openai_compatible"

    def __init__(
        self, base_url: str, api_key: str, model: str, temperature: float = 0, timeout: float = 120
    ) -> None:
        super().__init__(
            base_url,
            api_key,
            model,
            temperature=temperature,
            timeout_seconds=timeout,
        )
