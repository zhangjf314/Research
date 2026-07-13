import hashlib
import math
import re
import time
from abc import ABC, abstractmethod

import httpx

from paper_research.chunking.tokenizer import tokenize


class EmbeddingProviderError(RuntimeError):
    """Sanitized provider failure which never includes credentials or response bodies."""


class EmbeddingProvider(ABC):
    dimensions: int
    provider_name: str
    model_name: str
    revision: str

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed passages/documents for storage."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a query using retrieval-query semantics."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Backward-compatible document embedding alias."""
        return self.embed_documents(texts)


class HashEmbeddingProvider(EmbeddingProvider):
    """Dependency-free deterministic baseline for local development and tests."""

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions
        self.provider_name = "hash"
        self.model_name = "hash-v1"
        self.revision = "v1"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in tokenize(text.lower()):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "little")
            index = value % self.dimensions
            vector[index] += 1.0 if value & 1 else -1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


class OpenAICompatibleEmbeddingProvider(EmbeddingProvider):
    """Generic OpenAI-compatible embedding provider with symmetric input semantics."""

    provider_name = "openai_compatible"

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None,
        model: str,
        dimensions: int,
        revision: str = "unknown",
        timeout: float = 60.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_name = model
        self.dimensions = dimensions
        self.revision = revision
        self.timeout = timeout
        self.client = client or httpx.Client(timeout=timeout)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._request(texts)

    def embed_query(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("embedding query must not be blank")
        return self._request([text])[0]

    def _request(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        response = self.client.post(
            _embedding_endpoint(self.base_url),
            headers=headers,
            json={"model": self.model_name, "input": texts},
        )
        response.raise_for_status()
        return _validated_vectors(response.json(), len(texts), self.dimensions)


class JinaEmbeddingProvider(EmbeddingProvider):
    """Jina embeddings with asymmetric query/passage retrieval tasks."""

    provider_name = "jina"

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        dimensions: int,
        revision: str,
        batch_size: int = 32,
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
        client: httpx.Client | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("EMBEDDING_API_KEY is required for Jina")
        if not model:
            raise ValueError("EMBEDDING_MODEL is required for Jina")
        if dimensions <= 0:
            raise ValueError("EMBEDDING_DIMENSIONS must be positive for Jina")
        if batch_size <= 0:
            raise ValueError("EMBEDDING_BATCH_SIZE must be positive")
        if max_retries < 0:
            raise ValueError("EMBEDDING_MAX_RETRIES must be non-negative")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_name = model
        self.dimensions = dimensions
        self.revision = revision
        self.batch_size = batch_size
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.client = client or httpx.Client(timeout=timeout_seconds)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if any(not text.strip() for text in texts):
            raise ValueError("embedding documents must not contain blank text")
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            vectors.extend(
                self._request(texts[start : start + self.batch_size], "retrieval.passage")
            )
        return vectors

    def embed_query(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("embedding query must not be blank")
        return self._request([text], "retrieval.query")[0]

    def _request(self, texts: list[str], task: str) -> list[list[float]]:
        payload = {
            "model": self.model_name,
            "task": task,
            "dimensions": self.dimensions,
            "input": texts,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.post(
                    _embedding_endpoint(self.base_url), headers=headers, json=payload
                )
                if response.status_code == 429 or response.status_code >= 500:
                    response.raise_for_status()
                response.raise_for_status()
                return _validated_vectors(response.json(), len(texts), self.dimensions)
            except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError) as exc:
                last_error = exc
                retryable = not isinstance(exc, httpx.HTTPStatusError) or (
                    exc.response.status_code == 429 or exc.response.status_code >= 500
                )
                if not retryable or attempt >= self.max_retries:
                    break
                time.sleep(_retry_delay(exc, attempt))
            except (KeyError, TypeError, ValueError) as exc:
                raise EmbeddingProviderError(
                    f"Jina embedding response validation failed: {exc}"
                ) from exc
        assert last_error is not None
        detail = (
            f"HTTP {last_error.response.status_code}"
            if isinstance(last_error, httpx.HTTPStatusError)
            else type(last_error).__name__
        )
        raise EmbeddingProviderError(f"Jina embedding request failed: {detail}") from last_error


def _embedding_endpoint(base_url: str) -> str:
    return f"{base_url}/embeddings" if base_url.endswith("/v1") else f"{base_url}/v1/embeddings"


def _retry_delay(exc: Exception, attempt: int) -> float:
    if not isinstance(exc, httpx.HTTPStatusError):
        return min(2.0, 0.25 * (2**attempt))
    response = exc.response
    candidates = [
        response.headers.get("Retry-After"),
        response.headers.get("x-ratelimit-reset-tokens"),
        response.headers.get("x-ratelimit-reset-requests"),
    ]
    for value in candidates:
        seconds = _duration_seconds(value)
        if seconds is not None:
            return min(120.0, max(0.25, seconds))
    if response.status_code == 429:
        return 60.0
    return min(2.0, 0.25 * (2**attempt))


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


def _validated_vectors(payload: dict, expected: int, dimensions: int) -> list[list[float]]:
    data = sorted(payload.get("data", []), key=lambda item: item["index"])
    vectors = [item["embedding"] for item in data]
    if len(vectors) != expected:
        raise ValueError(
            f"embedding vector count mismatch: expected {expected}, returned {len(vectors)}"
        )
    for vector in vectors:
        if len(vector) != dimensions:
            raise ValueError(
                f"embedding dimension mismatch: configured {dimensions}, returned {len(vector)}"
            )
        if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in vector):
            raise ValueError("embedding vector contains a non-finite or non-numeric value")
    return [[float(value) for value in vector] for vector in vectors]
