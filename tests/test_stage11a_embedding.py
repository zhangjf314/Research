import json
import math
from pathlib import Path

import httpx
import pytest
from qdrant_client import QdrantClient

from paper_research.chunking.types import Chunk
from paper_research.config import Settings
from paper_research.evaluation.dataset import EvaluationItem
from paper_research.indexing.embedding import (
    EmbeddingProvider,
    EmbeddingProviderError,
    HashEmbeddingProvider,
    JinaEmbeddingProvider,
)
from paper_research.indexing.vector_store import QdrantVectorStore
from paper_research.providers.factory import (
    ProviderConfigurationError,
    build_embedding_provider,
)


def mock_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def jina(client: httpx.Client, **overrides) -> JinaEmbeddingProvider:
    values = {
        "base_url": "https://api.jina.ai/v1",
        "api_key": "secret-test-key",
        "model": "jina-embeddings-v5-text-small",
        "dimensions": 4,
        "revision": "2026-02-18",
        "batch_size": 2,
        "timeout_seconds": 0.1,
        "max_retries": 1,
        "client": client,
    }
    values.update(overrides)
    return JinaEmbeddingProvider(**values)


def response_for(request: httpx.Request) -> httpx.Response:
    payload = json.loads(request.content)
    return httpx.Response(
        200,
        json={
            "data": [
                {"index": index, "embedding": [float(index + 1)] * payload["dimensions"]}
                for index, _ in enumerate(payload["input"])
            ]
        },
    )


def raw_json_response(payload: dict) -> httpx.Response:
    return httpx.Response(
        200,
        content=json.dumps(payload, allow_nan=True).encode(),
        headers={"Content-Type": "application/json"},
    )


def test_hash_baseline_supports_query_and_documents() -> None:
    provider = HashEmbeddingProvider(384)
    assert len(provider.embed_query("query")) == 384
    assert len(provider.embed_documents(["one", "two"])) == 2
    assert provider.embed(["legacy"]) == provider.embed_documents(["legacy"])


def test_jina_uses_asymmetric_tasks_batches_and_dimensions() -> None:
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://api.jina.ai/v1/embeddings"
        assert request.headers["Authorization"] == "Bearer secret-test-key"
        requests.append(json.loads(request.content))
        return response_for(request)

    provider = jina(mock_client(handler))
    documents = provider.embed_documents(["doc one", "doc two", "doc three"])
    query = provider.embed_query("query")

    assert len(documents) == 3
    assert len(query) == 4
    assert [request["task"] for request in requests] == [
        "retrieval.passage",
        "retrieval.passage",
        "retrieval.query",
    ]
    assert all(request["dimensions"] == 4 for request in requests)
    assert all(request["model"] == "jina-embeddings-v5-text-small" for request in requests)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"data": []}, "vector count mismatch"),
        ({"data": [{"index": 0, "embedding": [1.0]}]}, "dimension mismatch"),
        (
            {"data": [{"index": 0, "embedding": [math.nan] * 4}]},
            "non-finite",
        ),
        (
            {"data": [{"index": 0, "embedding": [math.inf] * 4}]},
            "non-finite",
        ),
    ],
)
def test_jina_rejects_invalid_responses(payload: dict, message: str) -> None:
    provider = jina(mock_client(lambda _: raw_json_response(payload)))
    with pytest.raises(EmbeddingProviderError, match="response validation failed") as exc:
        provider.embed_query("query")
    assert message in str(exc.value)
    assert "secret-test-key" not in str(exc.value)


def test_jina_http_error_does_not_include_response_body_or_key() -> None:
    provider = jina(
        mock_client(
            lambda _: httpx.Response(
                401,
                text="secret-test-key and upstream private diagnostics",
            )
        )
    )
    with pytest.raises(EmbeddingProviderError, match="HTTP 401") as exc:
        provider.embed_query("query")
    assert "secret-test-key" not in str(exc.value)
    assert "private diagnostics" not in str(exc.value)


def test_jina_timeout_retries_and_sanitizes_error() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("secret-test-key must not escape", request=request)

    provider = jina(mock_client(handler), max_retries=2)
    with pytest.raises(EmbeddingProviderError, match="ReadTimeout") as exc:
        provider.embed_query("query")
    assert calls == 3
    assert "secret-test-key" not in str(exc.value)


def test_jina_honors_rate_limit_reset_header(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, headers={"x-ratelimit-reset-tokens": "1m2s"})
        return response_for(request)

    monkeypatch.setattr("paper_research.indexing.embedding.time.sleep", sleeps.append)
    provider = jina(mock_client(handler), max_retries=1)
    assert len(provider.embed_query("query")) == 4
    assert sleeps == [62.0]


def test_jina_empty_inputs_are_explicit() -> None:
    provider = jina(mock_client(response_for))
    assert provider.embed_documents([]) == []
    with pytest.raises(ValueError, match="blank"):
        provider.embed_documents([" "])
    with pytest.raises(ValueError, match="blank"):
        provider.embed_query("")


def test_production_jina_requires_key_model_and_dimensions() -> None:
    base = {"app_profile": "production", "embedding_provider": "jina", "_env_file": None}
    missing_all = Settings(**base)
    assert set(missing_all.embedding_configuration_issues) == {
        "EMBEDDING_API_KEY",
        "EMBEDDING_DIMENSIONS",
        "EMBEDDING_MODEL",
    }
    missing_model = Settings(
        **base, embedding_api_key="x", embedding_dimensions=1024
    )
    assert missing_model.embedding_configuration_issues == ["EMBEDDING_MODEL"]
    missing_dimensions = Settings(
        **base,
        embedding_api_key="x",
        embedding_model="jina-embeddings-v5-text-small",
    )
    assert missing_dimensions.embedding_configuration_issues == ["EMBEDDING_DIMENSIONS"]


def test_production_never_falls_back_to_hash() -> None:
    settings = Settings(app_profile="production", _env_file=None)
    with pytest.raises(ProviderConfigurationError, match="EMBEDDING_PROVIDER"):
        build_embedding_provider(settings)


def test_new_1024_collection_does_not_modify_hash_collection() -> None:
    client = QdrantClient(":memory:")
    hash_store = QdrantVectorStore(client, "papers_hash_v1", 384)
    hash_store.ensure_collection()
    production_store = QdrantVectorStore(client, "papers_jina_v5_small_1024", 1024)
    production_store.ensure_collection()
    assert client.get_collection("papers_hash_v1").config.params.vectors.size == 384
    assert client.get_collection("papers_jina_v5_small_1024").config.params.vectors.size == 1024


def test_ablation_loader_rejects_pending_and_excludes_approved_unanswerable(
    tmp_path: Path,
) -> None:
    from scripts.run_retrieval_ablation_v1 import load_items

    path = tmp_path / "gold.jsonl"
    answerable = {
        "question_id": "q1",
        "question": "question",
        "category": "method",
        "difficulty": "easy",
        "answerable": True,
        "gold_paper_ids": ["paper"],
        "gold_block_ids": [],
        "gold_pages": [1],
        "required_claims": [],
        "review_status": "approved",
    }
    unanswerable = {**answerable, "question_id": "q2", "answerable": False}
    path.write_text(
        "\n".join(json.dumps(item) for item in [answerable, unanswerable]) + "\n",
        encoding="utf-8",
    )
    raw, items = load_items(path)
    assert len(raw) == len(items) == 1
    pending = {**answerable, "question_id": "q3", "review_status": "pending"}
    path.write_text(json.dumps(pending) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="non-approved"):
        load_items(path)


class FailingQueryEmbedding(EmbeddingProvider):
    provider_name = "failing"
    model_name = "failing-v1"
    revision = "test"
    dimensions = 384

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return HashEmbeddingProvider(384).embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        raise EmbeddingProviderError("controlled query failure")


def test_ablation_counts_failed_queries() -> None:
    from scripts.run_retrieval_ablation_v1 import evaluate_variant

    client = QdrantClient(":memory:")
    chunk = Chunk(
        chunk_id="chunk-1",
        paper_id="paper-1",
        block_ids=["block-1"],
        block_type="paragraph",
        page_start=1,
        page_end=1,
        chunk_text="retrieval evidence",
        token_count=2,
    )
    store = QdrantVectorStore(client, "failing-eval", 384)
    store.upsert([chunk], HashEmbeddingProvider(384).embed_documents([chunk.chunk_text]))
    raw = {
        "question_id": "q1",
        "question": "question",
        "category": "method",
        "difficulty": "easy",
    }
    item = EvaluationItem(
        id="q1",
        question="question",
        question_type="method",
        relevant_paper_ids=["paper-1"],
    )
    result = evaluate_variant(
        name="failure_test",
        profile="production",
        provider=FailingQueryEmbedding(),
        retriever_type="dense",
        collection="failing-eval",
        collection_metadata={},
        chunks=[chunk],
        client=client,
        raw_items=[raw],
        eval_items=[item],
        canonical_ids={},
        settings=Settings(_env_file=None),
    )
    assert result["metrics"]["failure_count"] == 1
    assert result["queries"][0]["failure_reason"].startswith("EmbeddingProviderError")


def test_stage11a_defaults_keep_rerank_disabled_and_template_llm() -> None:
    settings = Settings(_env_file=None)
    assert settings.rerank_enabled is False
    assert settings.llm_provider == "template"
