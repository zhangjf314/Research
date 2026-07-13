import json
import time
from abc import ABC, abstractmethod
from typing import Any

import httpx
from pydantic import BaseModel, Field

from paper_research.retrieval.context_builder import ContextItem


class ModelUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float | None = None


class GeneratedClaim(BaseModel):
    text: str
    block_ids: list[str] = Field(default_factory=list)
    pages: list[int] = Field(default_factory=list)
    supported: bool = True
    support_note: str | None = None


class GenerationResult(BaseModel):
    answer: str
    claims: list[GeneratedClaim] = Field(default_factory=list)
    insufficient_evidence: bool = False
    usage: ModelUsage = Field(default_factory=ModelUsage)
    first_token_latency_ms: float | None = None
    total_latency_ms: float = 0
    raw_model: str


class LLMProvider(ABC):
    provider_name: str
    model_name: str

    @abstractmethod
    def generate_claim_answer(
        self, question: str, context: list[ContextItem], prompt_version: str
    ) -> GenerationResult:
        """Return claims already bound to supplied block IDs and pages."""


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
                answer="The available evidence is insufficient.",
                insufficient_evidence=True,
                raw_model=self.model_name,
                total_latency_ms=round((time.perf_counter() - started) * 1000, 3),
            )
        claims = [
            GeneratedClaim(
                text=item.evidence[:500],
                block_ids=[item.chunk_id],
                pages=list(range(item.page_start, item.page_end + 1)),
            )
            for item in context[:3]
        ]
        return GenerationResult(
            answer="\n\n".join(claim.text for claim in claims),
            claims=claims,
            raw_model=self.model_name,
            total_latency_ms=round((time.perf_counter() - started) * 1000, 3),
        )


class OpenAICompatibleLLMProvider(LLMProvider):
    provider_name = "openai_compatible"

    def __init__(
        self, base_url: str, api_key: str, model: str, temperature: float = 0, timeout: float = 120
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_name = model
        self.temperature = temperature
        self.timeout = timeout

    def generate_claim_answer(
        self, question: str, context: list[ContextItem], prompt_version: str
    ) -> GenerationResult:
        started = time.perf_counter()
        evidence = [
            {
                "block_id": item.chunk_id,
                "pages": list(range(item.page_start, item.page_end + 1)),
                "text": item.evidence,
            }
            for item in context
        ]
        schema = {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "insufficient_evidence": {"type": "boolean"},
                "claims": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "block_ids": {"type": "array", "items": {"type": "string"}},
                            "pages": {"type": "array", "items": {"type": "integer"}},
                        },
                        "required": ["text", "block_ids", "pages"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["answer", "claims", "insufficient_evidence"],
            "additionalProperties": False,
        }
        payload: dict[str, Any] = {
            "model": self.model_name,
            "temperature": self.temperature,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        f"Prompt version {prompt_version}. Answer only from supplied evidence. "
                        "Every atomic claim must cite exact supplied block_ids and pages."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {"question": question, "evidence": evidence}, ensure_ascii=False
                    ),
                },
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "claim_answer", "strict": True, "schema": schema},
            },
        }
        response = httpx.post(
            f"{self.base_url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        body = response.json()
        content = body["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        usage = body.get("usage") or {}
        elapsed = round((time.perf_counter() - started) * 1000, 3)
        return GenerationResult(
            **parsed,
            usage=ModelUsage(
                input_tokens=int(usage.get("prompt_tokens", 0)),
                output_tokens=int(usage.get("completion_tokens", 0)),
                total_tokens=int(usage.get("total_tokens", 0)),
            ),
            first_token_latency_ms=None,
            total_latency_ms=elapsed,
            raw_model=self.model_name,
        )
