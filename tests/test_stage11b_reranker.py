import json
import math

import httpx
import pytest

import scripts.run_reranker_ablation_v1 as reranker_ablation
from paper_research.chunking.types import Chunk
from paper_research.config import Settings
from paper_research.providers.factory import ProviderConfigurationError, build_reranker
from paper_research.retrieval.context_builder import ContextBuilder
from paper_research.retrieval.dense import RetrievalResult
from paper_research.retrieval.fusion import FusedResult
from paper_research.retrieval.hybrid import HybridRetriever
from paper_research.retrieval.reranker import (
    JinaReranker,
    Reranker,
    RerankerProviderError,
)
from paper_research.retrieval.sparse import BM25Retriever


def chunk(identifier: str, text: str | None = None) -> Chunk:
    value = text or f"query evidence candidate {identifier}"
    return Chunk(
        chunk_id=identifier,
        paper_id="paper-1",
        block_ids=[f"block-{identifier}"],
        section_path=["Method"],
        block_type="paragraph",
        page_start=1,
        page_end=1,
        chunk_text=value,
        token_count=len(value.split()),
    )


def candidates(count: int = 3) -> list[FusedResult]:
    return [
        FusedResult(chunk(str(index)), 1 / (index + 1), index + 1, index + 2)
        for index in range(count)
    ]


def client_for(payload: dict, status: int = 200) -> httpx.Client:
    return httpx.Client(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                status,
                content=json.dumps(payload, allow_nan=True).encode(),
                headers={"Content-Type": "application/json"},
            )
        )
    )


def test_jina_constructs_request_and_preserves_candidate_metadata() -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["Authorization"]
        seen["payload"] = __import__("json").loads(request.content)
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 2, "relevance_score": 0.9},
                    {"index": 0, "relevance_score": 0.7},
                ]
            },
        )

    source = candidates()
    provider = JinaReranker(
        base_url="https://api.jina.ai/v1",
        api_key="secret-value",
        model="jina-reranker-v3",
        max_retries=0,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    outcome = provider.rerank_with_trace("query", source, 2)

    assert seen["url"] == "https://api.jina.ai/v1/rerank"
    assert seen["auth"] == "Bearer secret-value"
    assert seen["payload"]["top_n"] == 2
    assert len(seen["payload"]["documents"]) == 3
    assert [item.chunk.chunk_id for item in outcome.results] == ["2", "0"]
    assert outcome.results[0].chunk is source[2].chunk
    assert outcome.results[0].dense_rank == source[2].dense_rank
    assert outcome.input_count == 3
    assert outcome.output_count == 2
    assert outcome.api_request_count == 1


@pytest.mark.parametrize(
    "results, message",
    [
        ([{"index": 4, "relevance_score": 0.9}], "out of range"),
        (
            [
                {"index": 0, "relevance_score": 0.9},
                {"index": 0, "relevance_score": 0.8},
            ],
            "duplicate",
        ),
        ([{"index": 0, "relevance_score": math.nan}], "finite"),
        ([{"index": 0, "relevance_score": math.inf}], "finite"),
    ],
)
def test_jina_rejects_invalid_results(results: list[dict], message: str) -> None:
    provider = JinaReranker(
        base_url="https://api.jina.ai/v1",
        api_key="secret",
        model="jina-reranker-v3",
        max_retries=0,
        client=client_for({"results": results}),
    )
    with pytest.raises(RerankerProviderError, match=message):
        provider.rerank("query", candidates(4), len(results))


def test_jina_requires_exact_output_count() -> None:
    provider = JinaReranker(
        base_url="https://api.jina.ai/v1",
        api_key="secret",
        model="jina-reranker-v3",
        max_retries=0,
        client=client_for({"results": [{"index": 0, "relevance_score": 1.0}]}),
    )
    with pytest.raises(RerankerProviderError, match="count mismatch"):
        provider.rerank("query", candidates(3), 2)


def test_provider_errors_are_sanitized() -> None:
    provider = JinaReranker(
        base_url="https://api.jina.ai/v1",
        api_key="secret-value",
        model="jina-reranker-v3",
        max_retries=0,
        client=client_for({"error": "secret-value response body"}, status=401),
    )
    with pytest.raises(RerankerProviderError) as captured:
        provider.rerank("query", candidates(), 3)
    assert "secret-value" not in str(captured.value)
    assert "response body" not in str(captured.value)
    assert "HTTP 401" in str(captured.value)


def test_production_jina_requires_key_and_model_without_fallback() -> None:
    missing_key = Settings(
        app_profile="production",
        rerank_enabled=True,
        rerank_provider="jina",
        rerank_model="jina-reranker-v3",
        _env_file=None,
    )
    assert missing_key.rerank_configuration_issues == ["RERANK_API_KEY"]
    with pytest.raises(ProviderConfigurationError, match="RERANK_API_KEY"):
        build_reranker(missing_key)

    missing_model = Settings(
        app_profile="production",
        rerank_enabled=True,
        rerank_provider="jina",
        rerank_api_key="secret",
        _env_file=None,
    )
    assert "RERANK_MODEL" in missing_model.rerank_configuration_issues

    configured = Settings(
        app_profile="production",
        rerank_enabled=True,
        rerank_provider="jina",
        rerank_model="jina-reranker-v3",
        rerank_api_key="secret",
        rerank_allow_fallback=False,
        _env_file=None,
    )
    provider = build_reranker(configured)
    assert isinstance(provider, JinaReranker)
    assert provider.allow_fallback is False


class FakeDense:
    def __init__(self, results: list[RetrievalResult]) -> None:
        self.results = results

    def retrieve(self, *_args, **_kwargs) -> list[RetrievalResult]:
        return self.results


class SpyReverseReranker(Reranker):
    provider_name = "spy"
    model_name = "reverse"

    def __init__(self) -> None:
        self.input_counts = []

    def rerank(self, query: str, results: list[FusedResult], top_k: int) -> list[FusedResult]:
        del query
        self.input_counts.append(len(results))
        return list(reversed(results))[:top_k]


def test_hybrid_only_reranks_top30_and_traces_all_rank_changes() -> None:
    chunks = [chunk(str(index)) for index in range(40)]
    dense = [RetrievalResult(item, 1 - index / 100) for index, item in enumerate(chunks)]
    spy = SpyReverseReranker()
    retriever = HybridRetriever(
        FakeDense(dense),  # type: ignore[arg-type]
        BM25Retriever(chunks),
        spy,
        ContextBuilder(include_neighbors=False),
        rerank_input_k=30,
        rerank_output_k=30,
    )
    result = retriever.retrieve("query evidence", recall_k=40, top_k=10, retrieval_scope="paper")

    assert spy.input_counts == [30]
    assert len(result.trace.rerank_candidates) == 30
    assert result.trace.pre_rerank_candidate_count == 30
    assert result.trace.rerank_output_count == 30
    assert result.trace.retrieval_scope == "paper"
    assert result.trace.rerank_candidates[0].pre_rerank_rank == 1
    assert result.trace.rerank_candidates[0].post_rerank_rank == 30
    assert len(result.context) == 10


def test_disabled_pipeline_does_not_call_reranker() -> None:
    class BombDisabled(Reranker):
        provider_name = "disabled"
        model_name = "none"

        def rerank(self, query, results, top_k):
            raise AssertionError("disabled reranker must not be called")

    items = [chunk("a"), chunk("b")]
    retriever = HybridRetriever(
        FakeDense([RetrievalResult(item, 1.0) for item in items]),  # type: ignore[arg-type]
        BM25Retriever(items),
        BombDisabled(),
        ContextBuilder(include_neighbors=False),
    )
    result = retriever.retrieve("query", recall_k=2, top_k=1)
    assert result.trace.rerank_api_request_count == 0
    assert result.trace.rerank_fallback_occurred is False


def ablation_record(identifier: str, scope: str = "paper") -> dict:
    papers = ["paper-public", "paper-two"] if scope == "multi_paper" else ["paper-public"]
    return {
        "question_id": identifier,
        "original_question": "question",
        "retrieval_query": "query evidence",
        "retrieval_scope": scope,
        "retrieval_filter": {"paper_ids": papers},
        "gold_paper_ids": papers,
        "gold_pages": [1],
        "gold_block_ids": ["block-0"],
        "category": "method",
        "difficulty": "easy",
        "review_status": "approved",
        "query_revision_reason": "test",
        "query_revision_version": "retrieval-query-v2",
        "query_revision_author": "test",
        "query_revision_review_status": "not_required_scope_only",
    }


def snapshot(record: dict) -> dict:
    items = candidates(3)
    return {
        "record": record,
        "filter_database_ids": ["paper-raw"],
        "dense_count": 3,
        "sparse_count": 3,
        "candidates": items,
        "candidate_signature": reranker_ablation.initial_signature(items),
        "retrieval_latency_ms": 4.0,
        "failure_reason": None,
    }


def test_ablation_variants_share_initial_candidates_and_do_not_mix_scopes() -> None:
    snapshots = [
        snapshot(ablation_record("paper-question")),
        snapshot(ablation_record("multi-question", "multi_paper")),
    ]
    mapping = {"paper-1": "paper-public"}
    no_rerank = reranker_ablation.apply_variant(
        name="no_rerank",
        reranker=reranker_ablation.DisabledReranker(),
        snapshots=snapshots,
        raw_to_public=mapping,
    )
    lexical = reranker_ablation.apply_variant(
        name="lexical",
        reranker=reranker_ablation.LexicalReranker(),
        snapshots=snapshots,
        raw_to_public=mapping,
    )
    assert [query["initial_candidate_signature"] for query in no_rerank["queries"]] == [
        query["initial_candidate_signature"] for query in lexical["queries"]
    ]
    assert no_rerank["metrics"]["paper"]["query_count"] == 1
    assert no_rerank["metrics"]["multi_paper"]["query_count"] == 1
    assert no_rerank["configuration"]["llm_called"] is False
    assert no_rerank["configuration"]["deep_research_called"] is False


def test_ablation_api_error_is_counted_without_fallback() -> None:
    class FailingReranker(Reranker):
        provider_name = "jina"
        model_name = "jina-reranker-v3"

        def rerank(self, query, results, top_k):
            raise AssertionError("rerank_with_trace is overridden")

        def rerank_with_trace(self, query, results, top_k):
            raise RerankerProviderError("Jina rerank request failed: HTTP 503", api_request_count=3)

    result = reranker_ablation.apply_variant(
        name="jina",
        reranker=FailingReranker(),
        snapshots=[snapshot(ablation_record("failure"))],
        raw_to_public={"paper-1": "paper-public"},
    )
    assert result["metrics"]["failure_count"] == 1
    assert result["metrics"]["fallback_count"] == 0
    assert result["metrics"]["api_request_count"] == 3
    assert result["queries"][0]["failure_reason"] == "Jina rerank request failed: HTTP 503"


def test_formal_ablation_does_not_mutate_collections() -> None:
    script = (
        reranker_ablation.Path(__file__).parents[1]
        / "scripts/run_reranker_ablation_v1.py"
    )
    source = script.read_text(encoding="utf-8")
    assert ".upsert(" not in source
    assert ".create_collection(" not in source
    assert reranker_ablation.INPUT_K == 30
    assert reranker_ablation.RERANK_OUTPUT_K == 30
    assert reranker_ablation.EVALUATION_K == 10
