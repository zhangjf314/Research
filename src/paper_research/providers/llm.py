import json
import math
import re
import socket
import ssl
import time
import uuid
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path
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
    diagnostic_attempts: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def insufficient_evidence(self) -> bool:
        return not self.answerable


class LLMProviderError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        error_code: str = "CLAIM_QA_PROVIDER_ERROR",
        stage: str = "LLM_PROVIDER_CALL",
        api_request_count: int = 0,
        retry_reasons: list[str] | None = None,
        rate_limit_events: int = 0,
        diagnostic_attempts: list[dict[str, Any]] | None = None,
        response_audit_path: str | None = None,
        error_details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.stage = stage
        self.api_request_count = api_request_count
        self.retry_reasons = list(retry_reasons or [])
        self.rate_limit_events = rate_limit_events
        self.diagnostic_attempts = list(diagnostic_attempts or [])
        self.response_audit_path = response_audit_path
        self.error_details = dict(error_details or {})


class LLMProvider(ABC):
    provider_name: str
    model_name: str

    @abstractmethod
    def generate_claim_answer(
        self,
        question: str,
        context: list[ContextItem],
        prompt_version: str,
        audit_metadata: dict[str, Any] | None = None,
    ) -> GenerationResult:
        """Return a schema-validated answer bound to supplied evidence IDs."""


class TemplateLLMProvider(LLMProvider):
    provider_name = "template"
    model_name = "template-v1"

    def generate_claim_answer(
        self,
        question: str,
        context: list[ContextItem],
        prompt_version: str,
        audit_metadata: dict[str, Any] | None = None,
    ) -> GenerationResult:
        del question, prompt_version, audit_metadata
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
        response_audit_enabled: bool = False,
        response_audit_dir: Path | str = Path("artifacts/private/qa-response-audits"),
        response_audit_max_prefix_chars: int = 500,
        response_audit_max_suffix_chars: int = 500,
        response_audit_max_error_window_chars: int = 500,
        response_audit_store_full_payload: bool = False,
        response_format: str = "json_object",
        thinking_enabled: bool = False,
        stream: bool = False,
        provider_name: str | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("LLM API key is required")
        if not model:
            raise ValueError("LLM model is required")
        if response_format != "json_object":
            raise ValueError("LLM response_format must be json_object")
        if stream:
            raise ValueError("LLM stream must be false for structured QA")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_name = model
        if provider_name:
            self.provider_name = provider_name
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self.max_output_tokens = max_output_tokens
        self.max_retries = max_retries
        self.response_format = response_format
        self.thinking_enabled = thinking_enabled
        self.stream = stream
        self.input_cost_per_million = input_cost_per_million
        self.output_cost_per_million = output_cost_per_million
        self.client = client or httpx.Client()
        self.response_audit_enabled = response_audit_enabled
        self.response_audit_dir = Path(response_audit_dir)
        self.response_audit_max_prefix_chars = response_audit_max_prefix_chars
        self.response_audit_max_suffix_chars = response_audit_max_suffix_chars
        self.response_audit_max_error_window_chars = response_audit_max_error_window_chars
        self.response_audit_store_full_payload = response_audit_store_full_payload

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/chat/completions"

    def _endpoint_hostname(self) -> str:
        return self.base_url.split("://", 1)[-1].split("/", 1)[0].split(":", 1)[0]

    def generate_claim_answer(
        self,
        question: str,
        context: list[ContextItem],
        prompt_version: str,
        audit_metadata: dict[str, Any] | None = None,
    ) -> GenerationResult:
        started = time.perf_counter()
        citation_key_map = self._citation_key_map(context)
        evidence = self._evidence_payload(context, citation_key_map)
        try:
            system_prompt = qa_system_prompt(prompt_version)
        except ValueError as exc:
            raise LLMProviderError(
                "The configured QA prompt version is unsupported.",
                error_code="CLAIM_QA_CONFIGURATION_ERROR",
                stage="LLM_REQUEST_BUILD",
            ) from exc
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(
                        {"question": question, "evidence": evidence}, ensure_ascii=False
                    ),
                },
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_output_tokens,
            "stream": self.stream,
            "response_format": {"type": "json_object"},
        }
        payload.update(self._provider_payload_overrides())
        requests = 0
        rate_limits = 0
        retry_reasons: list[str] = []
        diagnostic_attempts: list[dict[str, Any]] = []
        for attempt in range(self.max_retries + 1):
            requests += 1
            error_code = "CLAIM_QA_PROVIDER_ERROR"
            stage = "LLM_PROVIDER_CALL"
            response_audit_path: str | None = None
            request_started = time.perf_counter()
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
                choice = body["choices"][0]
                finish_reason = choice.get("finish_reason")
                if finish_reason == "length":
                    raise _ProviderResponseWithAudit("finish_reason:length", None)
                message = choice["message"]
                content = message["content"]
                response_audit = self._response_audit(response, body, content)
                audit_payload = self._build_response_audit(
                    response=response,
                    body=body,
                    content=content,
                    prompt_version=prompt_version,
                    audit_metadata=audit_metadata,
                    normalization_events=[],
                    parse_error=None,
                )
                audit_path = self._persist_response_audit(audit_payload)
                if not isinstance(content, str):
                    audit_path = self._persist_response_audit(
                        audit_payload, audit_path, force=True
                    )
                    raise _ProviderResponseWithAudit("non_string_content", audit_path)
                try:
                    parsed_payload, normalization_events = normalize_structured_qa_content(
                        content,
                        citation_key_map=citation_key_map,
                    )
                except json.JSONDecodeError as exc:
                    audit_payload = self._build_response_audit(
                        response=response,
                        body=body,
                        content=content,
                        prompt_version=prompt_version,
                        audit_metadata=audit_metadata,
                        normalization_events=[],
                        parse_error=exc,
                    )
                    audit_path = self._persist_response_audit(
                        audit_payload, audit_path, force=True
                    )
                    raise _MalformedJSONWithAudit(audit_path) from exc
                except ValueError as exc:
                    audit_payload = self._build_response_audit(
                        response=response,
                        body=body,
                        content=content,
                        prompt_version=prompt_version,
                        audit_metadata=audit_metadata,
                        normalization_events=[],
                        parse_error=exc,
                    )
                    audit_path = self._persist_response_audit(
                        audit_payload, audit_path, force=True
                    )
                    raise _MalformedJSONWithAudit(audit_path) from exc
                diagnostic_attempts.append(
                    {
                        "attempt": requests,
                        "model": str(body.get("model") or self.model_name),
                        "finish_reason": finish_reason,
                        "normalization_events": normalization_events,
                        "response_audit": response_audit,
                        "sanitized_output": self._sanitize_model_output(content),
                    }
                )
                parsed = StructuredQA.model_validate(parsed_payload)
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
                    diagnostic_attempts=diagnostic_attempts,
                )
            except LLMProviderError:
                raise
            except httpx.TimeoutException as exc:
                reason = type(exc).__name__
                error_details = classify_provider_exception(
                    exc, hostname=self._endpoint_hostname()
                )
                error_code = "CLAIM_QA_PROVIDER_TIMEOUT"
                stage = "LLM_PROVIDER_TIMEOUT"
                response_audit_path = str(
                    self._persist_provider_failure_audit(
                        exception=exc,
                        prompt_version=prompt_version,
                        audit_metadata=audit_metadata,
                        attempt=requests,
                        elapsed_ms=round((time.perf_counter() - request_started) * 1000, 3),
                    )
                )
            except httpx.NetworkError as exc:
                reason = type(exc).__name__
                error_details = classify_provider_exception(
                    exc, hostname=self._endpoint_hostname()
                )
                error_code = "CLAIM_QA_PROVIDER_NETWORK_ERROR"
                stage = "LLM_PROVIDER_NETWORK"
                response_audit_path = str(
                    self._persist_provider_failure_audit(
                        exception=exc,
                        prompt_version=prompt_version,
                        audit_metadata=audit_metadata,
                        attempt=requests,
                        elapsed_ms=round((time.perf_counter() - request_started) * 1000, 3),
                    )
                )
            except _RetryableLLMError as exc:
                reason = str(exc)
            except _MalformedJSONWithAudit as exc:
                reason = "malformed_json"
                error_code = "CLAIM_QA_JSON_PARSE_ERROR"
                stage = "LLM_JSON_PARSE"
                response_audit_path = str(exc.audit_path) if exc.audit_path else None
            except ValidationError as exc:
                reason = "schema_validation"
                error_code = "CLAIM_QA_SCHEMA_VALIDATION_ERROR"
                stage = "CLAIM_SCHEMA_VALIDATE"
                response_audit_path = self._persist_current_response_error_audit(
                    response=locals().get("response"),
                    body=locals().get("body"),
                    content=locals().get("content"),
                    prompt_version=prompt_version,
                    audit_metadata=audit_metadata,
                    normalization_events=locals().get("normalization_events", []),
                    parse_error=exc,
                    audit_path=locals().get("audit_path"),
                )
            except (AttributeError, KeyError, TypeError):
                reason = "malformed_response"
                error_code = "CLAIM_QA_PROVIDER_RESPONSE_ERROR"
                stage = "LLM_RESPONSE_EXTRACT"
            except _ProviderResponseWithAudit as exc:
                reason = exc.reason
                error_code = "CLAIM_QA_PROVIDER_RESPONSE_ERROR"
                stage = "LLM_RESPONSE_EXTRACT"
                response_audit_path = str(exc.audit_path) if exc.audit_path else None
            except CitationContextError as exc:
                reason = f"citation_validation:{exc.code}"
                error_code = "CLAIM_QA_CITATION_VALIDATION_ERROR"
                stage = "CLAIM_CITATION_VALIDATE"
                response_audit_path = self._persist_current_response_error_audit(
                    response=locals().get("response"),
                    body=locals().get("body"),
                    content=locals().get("content"),
                    prompt_version=prompt_version,
                    audit_metadata=audit_metadata,
                    normalization_events=locals().get("normalization_events", []),
                    parse_error=exc,
                    audit_path=locals().get("audit_path"),
                )
            except ValueError:
                reason = "citation_validation"
                error_code = "CLAIM_QA_CITATION_VALIDATION_ERROR"
                stage = "CLAIM_CITATION_VALIDATE"
                response_audit_path = self._persist_current_response_error_audit(
                    response=locals().get("response"),
                    body=locals().get("body"),
                    content=locals().get("content"),
                    prompt_version=prompt_version,
                    audit_metadata=audit_metadata,
                    normalization_events=locals().get("normalization_events", []),
                    parse_error=None,
                    audit_path=locals().get("audit_path"),
                )
            if attempt >= self.max_retries or not _is_transport_retry_reason(reason):
                raise LLMProviderError(
                    f"{self.provider_name} QA failed after {requests} request(s): {reason}",
                    error_code=error_code,
                    stage=stage,
                    api_request_count=requests,
                    retry_reasons=[*retry_reasons, reason],
                    rate_limit_events=rate_limits,
                    diagnostic_attempts=diagnostic_attempts,
                    response_audit_path=response_audit_path
                    if "response_audit_path" in locals()
                    else None,
                    error_details=locals().get("error_details"),
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

    def _provider_payload_overrides(self) -> dict[str, Any]:
        if self.provider_name == "deepseek":
            return {
                "thinking": {
                    "type": "enabled" if self.thinking_enabled else "disabled",
                }
            }
        return {"enable_thinking": bool(self.thinking_enabled)}

    @staticmethod
    def _citation_key_map(context: list[ContextItem]) -> dict[str, dict[str, Any]]:
        mapping: dict[str, dict[str, Any]] = {}
        index = 1
        for item in context:
            block_ids = item.block_ids or [item.chunk_id]
            block_page_map = (
                item.block_page_map
                if item.block_page_map
                else {block_id: item.page_start for block_id in block_ids}
            )
            for block_id in block_ids:
                mapping[f"C{index}"] = {
                    "paper_id": item.paper_id,
                    "page": int(block_page_map[block_id]),
                    "block_id": block_id,
                    "chunk_id": item.chunk_id,
                }
                index += 1
        return mapping

    @staticmethod
    def _evidence_payload(
        context: list[ContextItem],
        citation_key_map: dict[str, dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        citation_key_map = citation_key_map or SiliconFlowLLMProvider._citation_key_map(context)
        by_chunk: dict[str, list[dict[str, Any]]] = {}
        for key, citation in citation_key_map.items():
            by_chunk.setdefault(str(citation["chunk_id"]), []).append(
                {
                    "key": key,
                    "paper_id": citation["paper_id"],
                    "block_id": citation["block_id"],
                    "page": citation["page"],
                }
            )
        output = []
        for item in context:
            output.append(
                {
                    "chunk_id": item.chunk_id,
                    "section_path": item.section_path,
                    "citation_keys": by_chunk.get(item.chunk_id, []),
                    "text": item.evidence,
                }
            )
        return output

    @staticmethod
    def _validate_context_citations(answer: StructuredQA, context: list[ContextItem]) -> None:
        allowed = SiliconFlowLLMProvider._allowed_citations(context)
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

    @staticmethod
    def _allowed_citations(context: list[ContextItem]) -> set[tuple[str, int, str]]:
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
        return allowed

    @classmethod
    def _citation_retry_prompt(cls, reason: str, context: list[ContextItem]) -> str:
        triples = [
            {"paper_id": paper_id, "page": page, "block_id": block_id}
            for paper_id, page, block_id in sorted(cls._allowed_citations(context))
        ]
        return (
            "The previous JSON failed strict citation validation with reason "
            f"{reason}. Return the complete corrected JSON object. Do not change or infer "
            "citation identifiers. Every citation must exactly match one of these allowed "
            f"triples: {json.dumps(triples, ensure_ascii=False)}"
        )

    @staticmethod
    def _sanitize_model_output(content: Any) -> Any:
        if isinstance(content, str):
            return content[:20000]
        return str(content)[:20000]

    @staticmethod
    def _response_audit(
        response: httpx.Response, body: dict[str, Any], content: Any
    ) -> dict[str, Any]:
        content_text = content if isinstance(content, str) else str(content)
        usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
        choice = (body.get("choices") or [{}])[0] if isinstance(body.get("choices"), list) else {}
        message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
        safe_headers = {
            key.lower(): value
            for key, value in response.headers.items()
            if key.lower() in {"content-type", "x-request-id", "date"}
        }
        return {
            "http_status": response.status_code,
            "safe_headers": safe_headers,
            "top_level_fields": sorted(str(key) for key in body.keys()),
            "choices_present": "choices" in body,
            "message_present": bool(message),
            "content_present": isinstance(content, str),
            "content_length": len(content_text),
            "content_sha256": _sha256_text(content_text),
            "content_first_500": content_text[:500],
            "content_last_500": content_text[-500:],
            "reasoning_content_present": "reasoning_content" in message,
            "tool_calls_present": bool(message.get("tool_calls")),
            "finish_reason": choice.get("finish_reason"),
            "usage_fields": sorted(str(key) for key in usage.keys()),
        }

    def _build_response_audit(
        self,
        *,
        response: httpx.Response,
        body: dict[str, Any],
        content: Any,
        prompt_version: str,
        audit_metadata: dict[str, Any] | None,
        normalization_events: list[str],
        parse_error: BaseException | None,
    ) -> dict[str, Any]:
        metadata = audit_metadata or {}
        content_text = content if isinstance(content, str) else str(content)
        usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
        choice = (body.get("choices") or [{}])[0] if isinstance(body.get("choices"), list) else {}
        message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
        reasoning = message.get("reasoning_content")
        parse_details = classify_json_parse_failure(content_text, parse_error)
        payload: dict[str, Any] = {
            "schema_version": "qa-response-audit-v1",
            "request_id": metadata.get("request_id"),
            "run_id": metadata.get("run_id"),
            "sample_id": metadata.get("sample_id"),
            "provider": self.provider_name,
            "model": self.model_name,
            "prompt_version": prompt_version,
            "response_format_mode": "json_object",
            "http_status": response.status_code,
            "finish_reason": choice.get("finish_reason"),
            "content_present": isinstance(content, str),
            "content_type": type(content).__name__,
            "content_length": len(content_text),
            "content_sha256": _sha256_text(content_text),
            "think_tag_present": bool(re.search(r"</?think>", content_text, re.IGNORECASE)),
            "markdown_fence_present": "```" in content_text,
            "content_prefix_sanitized": self._sanitized_slice(
                content_text, self.response_audit_max_prefix_chars, start=True
            ),
            "content_suffix_sanitized": self._sanitized_slice(
                content_text, self.response_audit_max_suffix_chars, start=False
            ),
            "content_error_window_sanitized": self._error_window(content_text, parse_error),
            "reasoning_content_present": isinstance(reasoning, str),
            "reasoning_content_length": len(reasoning) if isinstance(reasoning, str) else 0,
            "tool_calls_present": bool(message.get("tool_calls")),
            "usage": {
                "input_tokens": usage.get("prompt_tokens"),
                "output_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
            },
            "parse_error_type": parse_details["parse_error_type"],
            "parse_error_message": parse_details["parse_error_message"],
            "parse_error_offset": parse_details["parse_error_offset"],
            "parse_error_line": parse_details["parse_error_line"],
            "parse_error_column": parse_details["parse_error_column"],
            "normalization_events": normalization_events,
            "captured_at": datetime.now(UTC).isoformat(),
            "store_full_payload": self.response_audit_store_full_payload,
            "full_payload_sanitized": None,
        }
        if self.response_audit_store_full_payload:
            payload["full_payload_sanitized"] = redact_sensitive_text(content_text)
        return payload

    def _persist_response_audit(
        self, audit: dict[str, Any], path: Path | None = None, *, force: bool = False
    ) -> Path | None:
        if not self.response_audit_enabled and not force:
            return None
        self.response_audit_dir.mkdir(parents=True, exist_ok=True)
        sample_id = str(audit.get("sample_id") or "unknown")
        run_id = str(audit.get("run_id") or uuid.uuid4().hex[:12])
        target = path or self.response_audit_dir / f"{sample_id}-{run_id}.json"
        target.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if sample_id == "q017":
            public = Path("artifacts/q017-live-retry-response-audit-sanitized-v1.json")
            public.parent.mkdir(parents=True, exist_ok=True)
            public.write_text(
                json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        return target

    def _persist_current_response_error_audit(
        self,
        *,
        response: object,
        body: object,
        content: object,
        prompt_version: str,
        audit_metadata: dict[str, Any] | None,
        normalization_events: list[str],
        parse_error: BaseException | None,
        audit_path: object,
    ) -> str | None:
        if not isinstance(response, httpx.Response) or not isinstance(body, dict):
            return None
        audit_payload = self._build_response_audit(
            response=response,
            body=body,
            content=content,
            prompt_version=prompt_version,
            audit_metadata=audit_metadata,
            normalization_events=normalization_events,
            parse_error=parse_error,
        )
        path = self._persist_response_audit(
            audit_payload,
            audit_path if isinstance(audit_path, Path) else None,
            force=True,
        )
        return str(path) if path else None

    def _persist_provider_failure_audit(
        self,
        *,
        exception: BaseException,
        prompt_version: str,
        audit_metadata: dict[str, Any] | None,
        attempt: int,
        elapsed_ms: float,
    ) -> Path:
        metadata = audit_metadata or {}
        self.response_audit_dir.mkdir(parents=True, exist_ok=True)
        sample_id = str(metadata.get("sample_id") or "unknown")
        run_id = str(metadata.get("run_id") or uuid.uuid4().hex[:12])
        audit = {
            "schema_version": "qa-provider-failure-audit-v1",
            "request_id": metadata.get("request_id"),
            "run_id": metadata.get("run_id"),
            "sample_id": metadata.get("sample_id"),
            "provider": self.provider_name,
            "model": self.model_name,
            "prompt_version": prompt_version,
            "endpoint_host": self.base_url,
            "attempt": attempt,
            "timeout_seconds": self.timeout_seconds,
            "elapsed_ms": elapsed_ms,
            "exception_type": type(exception).__name__,
            "exception_message_sanitized": redact_sensitive_text(str(exception)),
            "exception_classification": classify_provider_exception(
                exception, hostname=self._endpoint_hostname()
            ),
            "captured_at": datetime.now(UTC).isoformat(),
            "api_key_persisted": False,
            "authorization_header_persisted": False,
            "request_payload_persisted": False,
        }
        target = self.response_audit_dir / f"{sample_id}-{run_id}-provider-failure.json"
        target.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return target

    def _sanitized_slice(self, text: str, limit: int, *, start: bool) -> str:
        if limit <= 0:
            return ""
        value = text[:limit] if start else text[-limit:]
        return redact_sensitive_text(value)

    def _error_window(self, text: str, parse_error: BaseException | None) -> str:
        if self.response_audit_max_error_window_chars <= 0:
            return ""
        position = getattr(parse_error, "pos", None)
        if not isinstance(position, int):
            position = 0
        half = self.response_audit_max_error_window_chars // 2
        start = max(0, position - half)
        end = min(len(text), start + self.response_audit_max_error_window_chars)
        return redact_sensitive_text(text[start:end])


class _RetryableLLMError(RuntimeError):
    pass


class _MalformedJSONWithAudit(RuntimeError):
    def __init__(self, audit_path: Path | None) -> None:
        super().__init__("malformed_json")
        self.audit_path = audit_path


class _ProviderResponseWithAudit(RuntimeError):
    def __init__(self, reason: str, audit_path: Path | None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.audit_path = audit_path


class CitationContextError(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__("citation is not present in supplied context")
        self.code = code


class OpenAICompatibleLLMProvider(SiliconFlowLLMProvider):
    """Backward-compatible strict JSON provider using the same chat-completions contract."""

    provider_name = "openai_compatible"

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0,
        timeout: float = 120,
        *,
        max_output_tokens: int = 2048,
        max_retries: int = 2,
        input_cost_per_million: float | None = None,
        output_cost_per_million: float | None = None,
        provider_name: str | None = None,
        response_format: str = "json_object",
        thinking_enabled: bool = False,
        stream: bool = False,
        client: httpx.Client | None = None,
        response_audit_enabled: bool = False,
        response_audit_dir: Path | str = Path("artifacts/private/qa-response-audits"),
        response_audit_max_prefix_chars: int = 500,
        response_audit_max_suffix_chars: int = 500,
        response_audit_max_error_window_chars: int = 500,
        response_audit_store_full_payload: bool = False,
    ) -> None:
        super().__init__(
            base_url,
            api_key,
            model,
            temperature=temperature,
            timeout_seconds=timeout,
            max_output_tokens=max_output_tokens,
            max_retries=max_retries,
            input_cost_per_million=input_cost_per_million,
            output_cost_per_million=output_cost_per_million,
            client=client,
            response_audit_enabled=response_audit_enabled,
            response_audit_dir=response_audit_dir,
            response_audit_max_prefix_chars=response_audit_max_prefix_chars,
            response_audit_max_suffix_chars=response_audit_max_suffix_chars,
            response_audit_max_error_window_chars=response_audit_max_error_window_chars,
            response_audit_store_full_payload=response_audit_store_full_payload,
            response_format=response_format,
            thinking_enabled=thinking_enabled,
            stream=stream,
            provider_name=provider_name,
        )


FORBIDDEN_MODEL_CLAIM_FIELDS = {
    "claim_id",
    "paper_id",
    "block_id",
    "page",
    "support_status",
    "model_usage",
    "latency",
    "request_id",
    "citations",
    "citation_ids",
}


def normalize_structured_qa_content(
    content: str,
    *,
    citation_key_map: dict[str, dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Apply bounded, deterministic JSON normalization before schema validation.

    This deliberately avoids fuzzy citation repair: it only removes common transport
    wrappers and coerces obviously numeric page strings.
    """
    events: list[str] = []
    text = content.strip()
    text_without_think = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if text_without_think != text:
        text = text_without_think
        events.append("removed_think_block")
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
        events.append("removed_markdown_fence")
    first = text.find("{")
    last = text.rfind("}")
    if first > 0 or (last != -1 and last < len(text) - 1):
        if first == -1 or last == -1 or last < first:
            raise json.JSONDecodeError("no complete JSON object found", text, 0)
        text = text[first : last + 1]
        events.append("extracted_single_json_object")
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("structured QA response must be a JSON object")
    if "insufficient_evidence" in parsed:
        parsed = normalize_citation_key_qa_payload(parsed, citation_key_map or {})
        events.append("mapped_citation_keys_to_server_triples")
    claims = parsed.get("claims")
    if isinstance(claims, dict):
        parsed["claims"] = [claims]
        events.append("wrapped_single_claim_object")
    claims = parsed.get("claims") or []
    if not isinstance(claims, list):
        raise ValueError("structured QA response claims must be a list")
    for claim in claims:
        if not isinstance(claim, dict):
            raise ValueError("structured QA response claims entries must be objects")
        citations = claim.get("citations") or []
        if not isinstance(citations, list):
            raise ValueError("structured QA response claim citations must be a list")
        for citation in citations:
            if not isinstance(citation, dict):
                raise ValueError("structured QA response citation entries must be objects")
            page = citation.get("page")
            if isinstance(page, str) and page.isdigit():
                citation["page"] = int(page)
                events.append("coerced_page_string_to_int")
    return parsed, events


def normalize_citation_key_qa_payload(
    parsed: dict[str, Any],
    citation_key_map: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Convert the Stage 13.34 model-only citation-key shape into API triples.

    This is deterministic mapping, not citation repair: unknown or malformed keys
    fail validation instead of fuzzy matching or falling back to free triples.
    """
    allowed_top_level = {"answer", "insufficient_evidence", "claims"}
    extra_top_level = set(parsed) - allowed_top_level
    if extra_top_level:
        raise ValueError(
            "structured QA response contains forbidden top-level fields: "
            + ", ".join(sorted(extra_top_level))
        )
    insufficient = parsed.get("insufficient_evidence")
    if not isinstance(insufficient, bool):
        raise ValueError("structured QA response insufficient_evidence must be boolean")
    claims = parsed.get("claims")
    if not isinstance(claims, list):
        raise ValueError("structured QA response claims must be a list")
    answer = parsed.get("answer")
    if insufficient:
        if claims:
            raise ValueError("insufficient_evidence=true requires claims=[]")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("insufficient_evidence=true requires non-empty answer/refusal")
        return {
            "answerable": False,
            "answer": None,
            "claims": [],
            "refusal_reason": answer.strip(),
        }
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("answerable response requires non-empty answer")
    converted_claims: list[dict[str, Any]] = []
    for index, claim in enumerate(claims, start=1):
        if not isinstance(claim, dict):
            raise ValueError("structured QA response claims entries must be objects")
        forbidden = set(claim) & FORBIDDEN_MODEL_CLAIM_FIELDS
        if forbidden:
            raise ValueError(
                "structured QA response claim contains service-owned fields: "
                + ", ".join(sorted(forbidden))
            )
        extra_claim_fields = set(claim) - {"text", "citation_keys"}
        if extra_claim_fields:
            raise ValueError(
                "structured QA response claim contains unknown fields: "
                + ", ".join(sorted(extra_claim_fields))
            )
        text_value = claim.get("text")
        if not isinstance(text_value, str) or not text_value.strip():
            raise ValueError("structured QA response claim text must be non-empty string")
        citation_keys = claim.get("citation_keys")
        if not isinstance(citation_keys, list):
            raise ValueError("structured QA response citation_keys must be a list")
        if not citation_keys:
            raise ValueError("answerable claim requires citation_keys")
        citations: list[dict[str, Any]] = []
        for citation_key in citation_keys:
            if not isinstance(citation_key, str):
                raise ValueError("structured QA response citation key must be a string")
            if citation_key not in citation_key_map:
                raise ValueError(f"unknown citation key: {citation_key}")
            citation = citation_key_map[citation_key]
            citations.append(
                {
                    "paper_id": citation["paper_id"],
                    "page": citation["page"],
                    "block_id": citation["block_id"],
                }
            )
        converted_claims.append(
            {
                "claim_id": f"c{index}",
                "text": text_value.strip(),
                "citations": citations,
            }
        )
    return {
        "answerable": True,
        "answer": answer.strip(),
        "claims": converted_claims,
        "refusal_reason": None,
    }


def _is_transport_retry_reason(reason: str) -> bool:
    return reason in {
        "ReadTimeout",
        "ConnectTimeout",
        "ConnectError",
        "NetworkError",
        "HTTP 429",
        "HTTP 502",
        "HTTP 503",
        "HTTP 504",
    }


def classify_json_parse_failure(
    content: str, parse_error: BaseException | None
) -> dict[str, Any]:
    if parse_error is None:
        return {
            "parse_error_type": None,
            "parse_error_message": None,
            "parse_error_offset": None,
            "parse_error_line": None,
            "parse_error_column": None,
        }
    text = content or ""
    stripped = text.strip()
    message = str(parse_error) if parse_error else None
    error_type = "UNKNOWN_MALFORMED_JSON"
    if text == "":
        error_type = "EMPTY_CONTENT"
    elif stripped == "":
        error_type = "WHITESPACE_ONLY"
    elif re.fullmatch(r"```(?:json)?\s*.*?\s*```", stripped, flags=re.DOTALL | re.IGNORECASE):
        error_type = "MARKDOWN_FENCE"
    elif "<think>" in stripped.lower() and "</think>" in stripped.lower():
        error_type = "THINK_TAG_PRESENT"
    elif _looks_like_multiple_json_objects(stripped):
        error_type = "MULTIPLE_JSON_OBJECTS"
    elif stripped.startswith("{") and not stripped.endswith("}"):
        error_type = "TRUNCATED_JSON"
    elif stripped.startswith("["):
        error_type = "NON_OBJECT_TOP_LEVEL"
    elif stripped.startswith("{'") or "':" in stripped:
        error_type = "PYTHON_LITERAL"
    if isinstance(parse_error, json.JSONDecodeError):
        lower = parse_error.msg.lower()
        if "unterminated string" in lower:
            error_type = "UNTERMINATED_STRING"
        elif "invalid \\escape" in lower or "invalid escape" in lower:
            error_type = "INVALID_ESCAPE"
        elif "invalid control character" in lower:
            error_type = "INVALID_CONTROL_CHARACTER"
        elif "expecting property name enclosed in double quotes" in lower and re.search(
            r",\s*[}\]]", stripped
        ):
            error_type = "TRAILING_COMMA"
        elif "extra data" in lower and _looks_like_multiple_json_objects(stripped):
            error_type = "MULTIPLE_JSON_OBJECTS"
    if stripped and not stripped.startswith(("{", "[", "```", "<think>")):
        first = stripped.find("{")
        last = stripped.rfind("}")
        if first > 0 and last > first:
            error_type = "LEADING_PROSE"
    if stripped.startswith("{") and stripped.endswith("}") and parse_error is None:
        error_type = "UNKNOWN_MALFORMED_JSON"
    return {
        "parse_error_type": error_type,
        "parse_error_message": message,
        "parse_error_offset": getattr(parse_error, "pos", None),
        "parse_error_line": getattr(parse_error, "lineno", None),
        "parse_error_column": getattr(parse_error, "colno", None),
    }


def redact_sensitive_text(value: str) -> str:
    redacted = re.sub(
        r"(?i)(authorization\s*[:=]\s*bearer\s+)[A-Za-z0-9._~+/=-]+",
        r"\1[REDACTED]",
        value,
    )
    redacted = re.sub(r"(?i)(api[_-]?key\s*[:=]\s*)[A-Za-z0-9._~+/=-]+", r"\1[REDACTED]", redacted)
    redacted = re.sub(r"(?i)(cookie\s*[:=]\s*)[^\s;]+", r"\1[REDACTED]", redacted)
    redacted = re.sub(
        r"(?i)(postgres(?:ql)?://)[^\s'\"]+",
        r"\1[REDACTED]",
        redacted,
    )
    redacted = re.sub(r"[A-Za-z]:\\[^\s'\"]+", "[REDACTED_PATH]", redacted)
    return redacted


def classify_provider_exception(
    exception: BaseException,
    *,
    hostname: str | None = None,
    port: int = 443,
) -> dict[str, Any]:
    chain = _exception_chain(exception)
    classes = {str(item.get("class")).lower() for item in chain}
    messages = " ".join(str(item.get("repr_sanitized") or "") for item in chain).lower()
    winerrors = {item.get("winerror") for item in chain if item.get("winerror") is not None}
    errnos = {item.get("errno") for item in chain if item.get("errno") is not None}
    classification = "DEEP_RESEARCH_UNKNOWN_CONNECT_ERROR"
    if isinstance(exception, httpx.ConnectTimeout) or "connecttimeout" in classes:
        classification = "DEEP_RESEARCH_CONNECT_TIMEOUT"
    elif any(item.get("is_ssl_error") for item in chain) or "certificate" in messages:
        classification = "DEEP_RESEARCH_TLS_ERROR"
    elif isinstance(exception, httpx.ProxyError) or "proxy" in messages:
        classification = "DEEP_RESEARCH_PROXY_ERROR"
    elif 10013 in winerrors or "permissionerror" in classes or "winerror 10013" in messages:
        classification = "DEEP_RESEARCH_SOCKET_PERMISSION_ERROR"
    elif any(item.get("is_gaierror") for item in chain) or "name or service" in messages:
        classification = "DEEP_RESEARCH_DNS_ERROR"
    elif isinstance(exception, httpx.ConnectError):
        classification = "DEEP_RESEARCH_TCP_CONNECT_ERROR"
    elif isinstance(exception, httpx.TimeoutException):
        classification = "DEEP_RESEARCH_CONNECT_TIMEOUT"
    elif isinstance(exception, httpx.HTTPStatusError):
        status = exception.response.status_code
        if status in {401, 403}:
            classification = "DEEP_RESEARCH_PROVIDER_AUTH_ERROR"
        elif status == 429:
            classification = "DEEP_RESEARCH_PROVIDER_RATE_LIMIT"
        else:
            classification = "DEEP_RESEARCH_PROVIDER_HTTP_ERROR"
    elif errnos:
        classification = "DEEP_RESEARCH_TCP_CONNECT_ERROR"
    return {
        "classification": classification,
        "exception_class": type(exception).__name__,
        "exception_repr_sanitized": redact_sensitive_text(repr(exception)),
        "cause_chain": chain,
        "errno": next(iter(errnos), None),
        "winerror": next(iter(winerrors), None),
        "hostname": hostname,
        "port": port,
    }


def _exception_chain(exception: BaseException) -> list[dict[str, Any]]:
    chain: list[dict[str, Any]] = []
    seen: set[int] = set()
    current: BaseException | None = exception
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(
            {
                "class": type(current).__name__,
                "repr_sanitized": redact_sensitive_text(repr(current)),
                "errno": getattr(current, "errno", None),
                "winerror": getattr(current, "winerror", None),
                "is_gaierror": isinstance(current, socket.gaierror),
                "is_ssl_error": isinstance(current, ssl.SSLError),
            }
        )
        current = current.__cause__ or current.__context__
    return chain


def _looks_like_multiple_json_objects(text: str) -> bool:
    first_end = text.find("}")
    if first_end == -1:
        return False
    return text[first_end + 1 :].lstrip().startswith("{")


def _sha256_text(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()
