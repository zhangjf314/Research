import json

import httpx
import pytest

from paper_research.chunking.types import Chunk
from paper_research.providers.llm import (
    LLMProviderError,
    SiliconFlowLLMProvider,
    classify_json_parse_failure,
    normalize_structured_qa_content,
)
from paper_research.retrieval.fusion import FusedResult
from paper_research.retrieval.hybrid import HybridRetriever


def test_json_parse_failure_persists_sanitized_response_audit(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        assert "secret-test-key" in request.headers["authorization"]
        return httpx.Response(
            200,
            json={
                "model": "Qwen/Qwen3-8B",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": '{"answerable": true'},
                    }
                ],
                "usage": {
                    "prompt_tokens": 11,
                    "completion_tokens": 7,
                    "total_tokens": 18,
                },
            },
        )

    provider = SiliconFlowLLMProvider(
        "https://example.test/v1",
        "secret-test-key",
        "Qwen/Qwen3-8B",
        max_retries=0,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        response_audit_enabled=True,
        response_audit_dir=tmp_path,
    )
    with pytest.raises(LLMProviderError) as exc_info:
        provider.generate_claim_answer(
            "question",
            [],
            "qa-production-v1",
            audit_metadata={"sample_id": "qtest", "run_id": "test-run"},
        )
    error = exc_info.value
    assert error.error_code == "CLAIM_QA_JSON_PARSE_ERROR"
    assert error.stage == "LLM_JSON_PARSE"
    assert error.api_request_count == 1
    assert error.response_audit_path is not None
    audit = json.loads((tmp_path / "qtest-test-run.json").read_text(encoding="utf-8"))
    assert audit["parse_error_type"] == "TRUNCATED_JSON"
    assert audit["content_length"] == len('{"answerable": true')
    assert audit["content_sha256"]
    assert audit["usage"]["total_tokens"] == 18
    serialized = json.dumps(audit)
    assert "secret-test-key" not in serialized
    assert "Authorization" not in serialized


def test_provider_timeout_persists_sanitized_failure_audit(tmp_path):
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out while waiting for provider")

    provider = SiliconFlowLLMProvider(
        "https://example.test/v1",
        "secret-test-key",
        "Qwen/Qwen3-8B",
        max_retries=0,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        response_audit_dir=tmp_path,
    )

    with pytest.raises(LLMProviderError) as exc_info:
        provider.generate_claim_answer(
            "question",
            [],
            "qa-production-v1",
            audit_metadata={"sample_id": "qtimeout", "run_id": "timeout-run"},
        )

    error = exc_info.value
    assert error.error_code == "CLAIM_QA_PROVIDER_TIMEOUT"
    assert error.stage == "LLM_PROVIDER_TIMEOUT"
    assert error.api_request_count == 1
    assert error.response_audit_path is not None
    audit_path = tmp_path / "qtimeout-timeout-run-provider-failure.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["schema_version"] == "qa-provider-failure-audit-v1"
    assert audit["exception_type"] == "ReadTimeout"
    assert audit["request_payload_persisted"] is False
    serialized = json.dumps(audit)
    assert "secret-test-key" not in serialized
    assert "Authorization" not in serialized


def test_schema_validation_failure_forces_sanitized_response_audit(tmp_path):
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "Qwen/Qwen3-8B",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                {
                                    "answerable": True,
                                    "answer": "A schema-invalid answer.",
                                    "claims": [],
                                }
                            )
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 11,
                    "completion_tokens": 7,
                    "total_tokens": 18,
                },
            },
        )

    provider = SiliconFlowLLMProvider(
        "https://example.test/v1",
        "secret-test-key",
        "Qwen/Qwen3-8B",
        max_retries=0,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        response_audit_dir=tmp_path,
    )

    with pytest.raises(LLMProviderError) as exc_info:
        provider.generate_claim_answer(
            "question",
            [],
            "qa-production-v1",
            audit_metadata={"sample_id": "qschema", "run_id": "schema-run"},
        )

    error = exc_info.value
    assert error.error_code == "CLAIM_QA_SCHEMA_VALIDATION_ERROR"
    assert error.stage == "CLAIM_SCHEMA_VALIDATE"
    assert error.response_audit_path is not None
    audit = json.loads((tmp_path / "qschema-schema-run.json").read_text(encoding="utf-8"))
    assert audit["schema_version"] == "qa-response-audit-v1"
    assert audit["content_present"] is True
    assert audit["usage"]["total_tokens"] == 18
    serialized = json.dumps(audit)
    assert "secret-test-key" not in serialized
    assert "Authorization" not in serialized


def test_non_string_provider_content_is_audited_as_provider_response_error(tmp_path):
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "Qwen/Qwen3-8B",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": None},
                    }
                ],
                "usage": {
                    "prompt_tokens": 11,
                    "completion_tokens": 0,
                    "total_tokens": 11,
                },
            },
        )

    provider = SiliconFlowLLMProvider(
        "https://example.test/v1",
        "secret-test-key",
        "Qwen/Qwen3-8B",
        max_retries=0,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        response_audit_dir=tmp_path,
    )

    with pytest.raises(LLMProviderError) as exc_info:
        provider.generate_claim_answer(
            "question",
            [],
            "qa-production-v1",
            audit_metadata={"sample_id": "qnull", "run_id": "null-run"},
        )

    error = exc_info.value
    assert error.error_code == "CLAIM_QA_PROVIDER_RESPONSE_ERROR"
    assert error.stage == "LLM_RESPONSE_EXTRACT"
    assert error.retry_reasons == ["non_string_content"]
    assert error.response_audit_path is not None
    audit = json.loads((tmp_path / "qnull-null-run.json").read_text(encoding="utf-8"))
    assert audit["content_present"] is False
    assert audit["content_type"] == "NoneType"
    serialized = json.dumps(audit)
    assert "secret-test-key" not in serialized
    assert "Authorization" not in serialized


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("", "EMPTY_CONTENT"),
        ("   \n", "WHITESPACE_ONLY"),
        ('{"a": 1', "TRUNCATED_JSON"),
        ('{"a": 1} {"b": 2}', "MULTIPLE_JSON_OBJECTS"),
        ("{'a': 1}", "PYTHON_LITERAL"),
    ],
)
def test_malformed_json_classification(content, expected):
    try:
        json.loads(content)
    except json.JSONDecodeError as exc:
        details = classify_json_parse_failure(content, exc)
    else:
        details = classify_json_parse_failure(content, None)
    assert details["parse_error_type"] == expected


def test_structured_qa_normalization_rejects_non_object_claims():
    content = json.dumps(
        {
            "answerable": True,
            "answer": "answer",
            "claims": ["not an object"],
            "refusal_reason": None,
        }
    )

    with pytest.raises(ValueError, match="claims entries must be objects"):
        normalize_structured_qa_content(content)


def _chunk(chunk_id: str, section: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        paper_id="paper",
        block_ids=[chunk_id],
        section_path=[section],
        block_type="paragraph",
        page_start=1,
        page_end=1,
        chunk_text=f"{section} text",
        token_count=10,
    )


def test_contribution_context_selection_uses_section_prior_without_ids():
    results = [
        FusedResult(_chunk("later", "Results"), 0.9, dense_rank=1, sparse_rank=1),
        FusedResult(_chunk("abstract", "Abstract"), 0.1, dense_rank=43, sparse_rank=29),
    ]
    selected = HybridRetriever._context_candidates(
        "What are the target paper's main contributions?",
        results,
        top_k=1,
        retrieval_scope="paper",
    )
    assert selected[0].chunk.chunk_id == "abstract"
