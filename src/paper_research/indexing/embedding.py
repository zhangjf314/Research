import hashlib
import math
from abc import ABC, abstractmethod

import httpx

from paper_research.chunking.tokenizer import tokenize


class EmbeddingProvider(ABC):
    dimensions: int
    provider_name: str
    model_name: str
    revision: str

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts into normalized dense vectors."""


class HashEmbeddingProvider(EmbeddingProvider):
    """Dependency-free deterministic baseline for local development and tests."""

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions
        self.provider_name = "hash"
        self.model_name = "hash-v1"
        self.revision = "v1"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

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
    """Embedding provider using the OpenAI-compatible ``/v1/embeddings`` contract."""

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
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_name = model
        self.dimensions = dimensions
        self.revision = revision
        self.timeout = timeout

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        response = httpx.post(
            f"{self.base_url}/v1/embeddings",
            headers=headers,
            json={"model": self.model_name, "input": texts},
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = sorted(response.json().get("data", []), key=lambda item: item["index"])
        vectors = [item["embedding"] for item in data]
        if len(vectors) != len(texts):
            raise RuntimeError("embedding provider returned an unexpected vector count")
        wrong = [len(vector) for vector in vectors if len(vector) != self.dimensions]
        if wrong:
            raise ValueError(
                f"embedding dimension mismatch: configured {self.dimensions}, returned {wrong[0]}"
            )
        return vectors
