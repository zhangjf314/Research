import math
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import httpx

from paper_research.chunking.tokenizer import tokenize
from paper_research.retrieval.fusion import FusedResult


class RerankerProviderError(RuntimeError):
    """Sanitized provider error that never includes credentials or response bodies."""

    def __init__(self, message: str, *, api_request_count: int = 0) -> None:
        super().__init__(message)
        self.api_request_count = api_request_count


@dataclass(frozen=True)
class RerankOutcome:
    results: list[FusedResult]
    provider: str
    model: str
    input_count: int
    output_count: int
    latency_ms: float
    fallback_occurred: bool = False
    failure_reason: str | None = None
    api_request_count: int = 0


class Reranker(ABC):
    provider_name = "unknown"
    model_name = "unknown"

    @abstractmethod
    def rerank(self, query: str, results: list[FusedResult], top_k: int) -> list[FusedResult]:
        """Rerank only the supplied candidate list."""

    def rerank_with_trace(
        self, query: str, results: list[FusedResult], top_k: int
    ) -> RerankOutcome:
        started = time.perf_counter()
        reranked = self.rerank(query, results, top_k)
        return RerankOutcome(
            results=reranked,
            provider=self.provider_name,
            model=self.model_name,
            input_count=len(results),
            output_count=len(reranked),
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
        )


class LexicalReranker(Reranker):
    """Local deterministic ablation baseline using lexical overlap plus RRF score."""

    provider_name = "lexical"
    model_name = "lexical-v1"

    def rerank(self, query: str, results: list[FusedResult], top_k: int) -> list[FusedResult]:
        query_terms = {token.lower() for token in tokenize(query) if token.isalnum()}

        def score(result: FusedResult) -> float:
            text_terms = {
                token.lower() for token in tokenize(result.chunk.chunk_text) if token.isalnum()
            }
            overlap = len(query_terms & text_terms) / max(1, len(query_terms))
            phrase_bonus = 0.25 if query.lower() in result.chunk.chunk_text.lower() else 0.0
            return overlap + phrase_bonus + result.score

        ranked = sorted(results, key=score, reverse=True)[:top_k]
        return [
            FusedResult(
                chunk=item.chunk,
                score=score(item),
                dense_rank=item.dense_rank,
                sparse_rank=item.sparse_rank,
            )
            for item in ranked
        ]


class DisabledReranker(Reranker):
    provider_name = "disabled"
    model_name = "none"

    def rerank(self, query: str, results: list[FusedResult], top_k: int) -> list[FusedResult]:
        del query
        return results[:top_k]


class JinaReranker(Reranker):
    """Strict Jina ``/v1/rerank`` adapter with optional explicit fallback."""

    provider_name = "jina"

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: float = 60,
        max_retries: int = 2,
        allow_fallback: bool = False,
        fallback: Reranker | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("RERANK_API_KEY is required for Jina")
        if not model:
            raise ValueError("RERANK_MODEL is required for Jina")
        if timeout_seconds <= 0:
            raise ValueError("RERANK_TIMEOUT_SECONDS must be positive")
        if max_retries < 0:
            raise ValueError("RERANK_MAX_RETRIES must be non-negative")
        if allow_fallback and fallback is None:
            raise ValueError("an explicit fallback provider is required when fallback is enabled")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_name = model
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.allow_fallback = allow_fallback
        self.fallback = fallback
        self.client = client or httpx.Client(timeout=timeout_seconds)

    def rerank(self, query: str, results: list[FusedResult], top_k: int) -> list[FusedResult]:
        return self.rerank_with_trace(query, results, top_k).results

    def rerank_with_trace(
        self, query: str, results: list[FusedResult], top_k: int
    ) -> RerankOutcome:
        if not query.strip():
            raise ValueError("rerank query must not be blank")
        if not results:
            return RerankOutcome([], self.provider_name, self.model_name, 0, 0, 0.0)
        if top_k < 1 or top_k > len(results):
            raise ValueError("rerank top_k must be between 1 and candidate count")
        started = time.perf_counter()
        requests = 0
        try:
            ranked, requests = self._request(query, results, top_k)
            return RerankOutcome(
                results=ranked,
                provider=self.provider_name,
                model=self.model_name,
                input_count=len(results),
                output_count=len(ranked),
                latency_ms=round((time.perf_counter() - started) * 1000, 3),
                api_request_count=requests,
            )
        except RerankerProviderError as exc:
            requests = exc.api_request_count
            if not self.allow_fallback or self.fallback is None:
                raise
            fallback_results = self.fallback.rerank(query, results, top_k)
            return RerankOutcome(
                results=fallback_results,
                provider=self.provider_name,
                model=self.model_name,
                input_count=len(results),
                output_count=len(fallback_results),
                latency_ms=round((time.perf_counter() - started) * 1000, 3),
                fallback_occurred=True,
                failure_reason=str(exc),
                api_request_count=requests,
            )

    def _request(
        self, query: str, results: list[FusedResult], top_k: int
    ) -> tuple[list[FusedResult], int]:
        payload = {
            "model": self.model_name,
            "query": query,
            "documents": [item.chunk.chunk_text for item in results],
            "top_n": top_k,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        requests = 0
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            requests += 1
            try:
                response = self.client.post(
                    _rerank_endpoint(self.base_url), headers=headers, json=payload
                )
                response.raise_for_status()
                return _validated_results(response.json(), results, top_k), requests
            except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError) as exc:
                last_error = exc
                retryable = not isinstance(exc, httpx.HTTPStatusError) or (
                    exc.response.status_code == 429 or exc.response.status_code >= 500
                )
                if not retryable or attempt >= self.max_retries:
                    break
                time.sleep(_retry_delay(exc, attempt))
            except (KeyError, TypeError, ValueError) as exc:
                raise RerankerProviderError(
                    f"Jina rerank response validation failed: {exc}",
                    api_request_count=requests,
                ) from exc
        assert last_error is not None
        detail = (
            f"HTTP {last_error.response.status_code}"
            if isinstance(last_error, httpx.HTTPStatusError)
            else type(last_error).__name__
        )
        raise RerankerProviderError(
            f"Jina rerank request failed: {detail}", api_request_count=requests
        ) from last_error


class CrossEncoderReranker(JinaReranker):
    """Backward-compatible strict adapter for an OpenAI-style rerank endpoint."""

    provider_name = "cross_encoder"

    def __init__(
        self, base_url: str, api_key: str | None, model: str, timeout: float = 60
    ) -> None:
        super().__init__(
            base_url=base_url,
            api_key=api_key or "",
            model=model,
            timeout_seconds=timeout,
            max_retries=0,
        )


def _rerank_endpoint(base_url: str) -> str:
    return f"{base_url}/rerank" if base_url.endswith("/v1") else f"{base_url}/v1/rerank"


def _validated_results(
    payload: dict, candidates: list[FusedResult], expected: int
) -> list[FusedResult]:
    raw = payload.get("results")
    if not isinstance(raw, list):
        raise ValueError("results must be a list")
    if len(raw) != expected:
        raise ValueError(f"result count mismatch: expected {expected}, returned {len(raw)}")
    seen: set[int] = set()
    ranked = []
    for item in raw:
        index = item["index"]
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError("result index must be an integer")
        if index < 0 or index >= len(candidates):
            raise ValueError(f"result index out of range: {index}")
        if index in seen:
            raise ValueError(f"duplicate result index: {index}")
        seen.add(index)
        score = item.get("relevance_score", item.get("score"))
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError("rerank score must be numeric")
        if not math.isfinite(score):
            raise ValueError("rerank score must be finite")
        source = candidates[index]
        ranked.append(
            FusedResult(
                chunk=source.chunk,
                score=float(score),
                dense_rank=source.dense_rank,
                sparse_rank=source.sparse_rank,
            )
        )
    return ranked


def _retry_delay(exc: Exception, attempt: int) -> float:
    if not isinstance(exc, httpx.HTTPStatusError):
        return min(2.0, 0.25 * (2**attempt))
    for value in (
        exc.response.headers.get("Retry-After"),
        exc.response.headers.get("x-ratelimit-reset-tokens"),
        exc.response.headers.get("x-ratelimit-reset-requests"),
    ):
        seconds = _duration_seconds(value)
        if seconds is not None:
            return min(120.0, max(0.25, seconds))
    return 60.0 if exc.response.status_code == 429 else min(2.0, 0.25 * (2**attempt))


def _duration_seconds(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        pass
    matches = re.findall(r"([0-9]+(?:\.[0-9]+)?)(ms|s|m|h)", value.lower())
    if not matches:
        return None
    scale = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}
    return sum(float(number) * scale[unit] for number, unit in matches)
