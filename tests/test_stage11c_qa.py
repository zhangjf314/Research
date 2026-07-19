import json
import math
from pathlib import Path

import httpx
import pytest

import scripts.run_qa_production_v1 as qa_eval
from paper_research.chunking.types import Chunk
from paper_research.config import Settings
from paper_research.generation.qa_service import ClaimValidationError, QAService
from paper_research.providers.factory import ProviderConfigurationError, build_llm_provider
from paper_research.providers.llm import LLMProviderError, SiliconFlowLLMProvider
from paper_research.retrieval.context_builder import ContextBuilder, ContextItem
from paper_research.retrieval.fusion import FusedResult


def context() -> list[ContextItem]:
    return [
        ContextItem(
            chunk_id="chunk-1",
            paper_id="paper-1",
            block_ids=["block-1"],
            section_path=["Method"],
            page_start=2,
            page_end=2,
            evidence="The model uses an attention mechanism.",
            score=1.0,
        )
    ]


def valid_answer() -> dict:
    return {
        "answer": "The model uses attention.",
        "insufficient_evidence": False,
        "claims": [
            {
                "text": "The model uses attention.",
                "citation_keys": ["C1"],
            }
        ],
    }


def legacy_valid_answer() -> dict:
    return {
        "answerable": True,
        "answer": "The model uses attention.",
        "claims": [
            {
                "claim_id": "c1",
                "text": "The model uses attention.",
                "citations": [
                    {"paper_id": "paper-1", "page": 2, "block_id": "block-1"}
                ],
            }
        ],
        "refusal_reason": None,
    }


def response(payload: dict | str, *, status: int = 200) -> httpx.Response:
    content = payload if isinstance(payload, str) else json.dumps(payload)
    return httpx.Response(
        status,
        json={
            "model": "Qwen/Qwen3-8B",
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        if status == 200
        else {"error": content},
    )


def provider(handler, *, retries: int = 2) -> SiliconFlowLLMProvider:
    return SiliconFlowLLMProvider(
        "https://api.siliconflow.cn/v1",
        "secret-value",
        "Qwen/Qwen3-8B",
        max_retries=retries,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def test_siliconflow_request_and_structured_result() -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["Authorization"]
        seen["payload"] = json.loads(request.content)
        return response(valid_answer())

    result = provider(handler).generate_claim_answer("method?", context(), "qa-production-v1")
    assert seen["url"] == "https://api.siliconflow.cn/v1/chat/completions"
    assert seen["auth"] == "Bearer secret-value"
    assert seen["payload"]["response_format"] == {"type": "json_object"}
    assert seen["payload"]["enable_thinking"] is False
    assert seen["payload"]["temperature"] == 0
    assert result.answerable is True
    assert result.usage.total_tokens == 15
    assert result.first_token_latency_ms is None
    assert result.api_request_count == 1


def test_malformed_json_does_not_retry_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return response("not-json")

    monkeypatch.setattr("paper_research.providers.llm.time.sleep", lambda _seconds: None)
    with pytest.raises(LLMProviderError) as captured:
        provider(handler).generate_claim_answer("method?", context(), "qa-production-v1")
    assert calls == 1
    assert captured.value.api_request_count == 1
    assert captured.value.retry_reasons == ["malformed_json"]


@pytest.mark.parametrize(
    "mutation, reason",
    [
        (lambda body: body["claims"][0].update(citations=[]), "malformed_json"),
        (
            lambda body: body["claims"][0].update(citation_keys=["C99"]),
            "malformed_json",
        ),
        (
            lambda body: body.update(insufficient_evidence="false"),
            "malformed_json",
        ),
    ],
)
def test_invalid_claim_outputs_fail_without_generation_retry(monkeypatch, mutation, reason) -> None:
    body = valid_answer()
    mutation(body)
    monkeypatch.setattr("paper_research.providers.llm.time.sleep", lambda _seconds: None)
    with pytest.raises(LLMProviderError) as captured:
        provider(lambda _request: response(body)).generate_claim_answer(
            "method?", context(), "qa-production-v1"
        )
    assert captured.value.api_request_count == 1
    assert captured.value.retry_reasons == [reason]


def test_transport_timeout_retries_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ReadTimeout("timeout")
        return response(valid_answer())

    monkeypatch.setattr("paper_research.providers.llm.time.sleep", lambda _seconds: None)
    result = provider(handler, retries=1).generate_claim_answer(
        "method?", context(), "qa-production-v1"
    )
    assert result.api_request_count == 2
    assert result.retry_count == 1
    assert result.retry_reasons == ["ReadTimeout"]


def test_provider_error_is_sanitized_and_never_falls_back() -> None:
    llm = provider(lambda _request: response("secret-value leaked body", status=401))
    with pytest.raises(LLMProviderError) as captured:
        llm.generate_claim_answer("method?", context(), "qa-production-v1")
    assert "secret-value" not in str(captured.value)
    assert "leaked body" not in str(captured.value)
    assert "HTTP 401" in str(captured.value)


def test_production_siliconflow_requires_key_and_model() -> None:
    missing_key = Settings(
        app_profile="production",
        embedding_provider="jina",
        embedding_model="jina-embeddings-v5-text-small",
        embedding_dimensions=1024,
        embedding_api_key="embedding",
        llm_provider="siliconflow",
        llm_model="Qwen/Qwen3-8B",
        prompt_version="qa-production-v1",
        _env_file=None,
    )
    assert "LLM_API_KEY" in missing_key.llm_configuration_issues
    with pytest.raises(ProviderConfigurationError, match="LLM_API_KEY"):
        build_llm_provider(missing_key)
    missing_model = missing_key.model_copy(update={"llm_api_key": "key", "llm_model": ""})
    assert "LLM_MODEL" in missing_model.llm_configuration_issues


def test_qa_service_rejects_citation_outside_context() -> None:
    class UnsafeProvider:
        provider_name = "unsafe"
        model_name = "unsafe"

        def generate_claim_answer(self, *_args):
            from paper_research.providers.llm import GenerationResult

            body = legacy_valid_answer()
            body["claims"][0]["citations"][0]["block_id"] = "invented"
            return GenerationResult(**body, raw_model="unsafe")

    with pytest.raises(ClaimValidationError):
        QAService(llm=UnsafeProvider(), prompt_version="qa-production-v1").answer_from_context(
            "method?", context()
        )


def test_context_dedup_and_token_budget_trace() -> None:
    chunk = Chunk(
        chunk_id="chunk-1",
        paper_id="paper-1",
        block_ids=["block-1"],
        section_path=["Method"],
        block_type="paragraph",
        page_start=2,
        page_end=2,
        chunk_text="x" * 100,
        token_count=25,
    )
    builder = ContextBuilder(include_neighbors=False, max_tokens=10)
    output = builder.build([FusedResult(chunk, 1.0), FusedResult(chunk, 0.5)])
    assert len(output) == 1
    assert len(output[0].evidence) == 40
    assert builder.last_trace.truncated_chunk_id == "chunk-1"
    assert builder.last_trace.token_budget == 10


def test_mode_selection_and_formal_protocol_guards() -> None:
    records = [{"question_id": f"q{index:03d}"} for index in range(1, 51)]
    assert [row["question_id"] for row in qa_eval.select_records(records, "smoke")] == [
        "q001",
        "q005",
        "q030",
    ]
    assert len(qa_eval.select_records(records, "dev")) == 10
    assert len(qa_eval.select_records(records, "full")) == 50
    resumed = qa_eval.order_rows(
        qa_eval.select_records(records, "smoke"),
        [
            {"question_id": "q030"},
            {"question_id": "q001"},
            {"question_id": "q005"},
            {"question_id": "q001"},
        ],
    )
    assert [row["question_id"] for row in resumed] == ["q001", "q005", "q030"]
    source = Path(qa_eval.__file__).read_text(encoding="utf-8")
    assert "if settings.rerank_enabled" in source
    assert '"rerank_enabled": False' in source
    assert '"deep_research_called": False' in source
    assert ".upsert(" not in source


def test_rule_metrics_do_not_use_llm_judge() -> None:
    score = qa_eval.overlap("uses attention mechanism", "The model uses an attention mechanism")
    assert math.isclose(score, 1.0)
    assert qa_eval.CLAIM_MATCH_THRESHOLD == 0.35
