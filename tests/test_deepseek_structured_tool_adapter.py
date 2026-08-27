from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from paper_research.providers.llm import LLMProviderError, OpenAICompatibleLLMProvider


def _body(message: dict[str, Any], *, finish_reason: str = "tool_calls") -> dict[str, Any]:
    return {
        "model": "deepseek-v4-flash",
        "choices": [{"finish_reason": finish_reason, "message": message}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _tool_message(name: str, arguments: str) -> dict[str, Any]:
    return {
        "content": "",
        "tool_calls": [
            {"id": "call-1", "type": "function", "function": {"name": name, "arguments": arguments}}
        ],
    }


def _provider(handler: Any, *, retries: int = 0) -> OpenAICompatibleLLMProvider:
    return OpenAICompatibleLLMProvider(
        "https://api.deepseek.com",
        "secret-value",
        "deepseek-v4-flash",
        provider_name="deepseek",
        max_retries=retries,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def _call(provider: OpenAICompatibleLLMProvider):
    return provider.generate_structured_json(
        system_prompt="Return structured output.",
        user_prompt="Plan a paper summary.",
        schema_name="research-agent-decision-v1",
        request_context={"task_id": "adapter-test", "agent_phase": "PLAN"},
        max_output_tokens=900,
    )


def test_deepseek_uses_forced_normal_function_call_and_parses_arguments() -> None:
    seen: dict[str, Any] = {}
    arguments = json.dumps(
        {
            "objective": "Summarize the paper.",
            "subquestions": [{"id": "SQ1", "question": "What is contributed?", "status": "OPEN"}],
            "completion_criteria": ["Use evidence."],
        }
    )

    def handler(request: httpx.Request) -> httpx.Response:
        seen["payload"] = json.loads(request.content)
        return httpx.Response(200, json=_body(_tool_message("submit_research_plan", arguments)))

    result = _call(_provider(handler))

    payload = seen["payload"]
    assert "response_format" not in payload
    assert payload["tool_choice"] == {
        "type": "function",
        "function": {"name": "submit_research_plan"},
    }
    assert payload["tools"][0]["function"]["name"] == "submit_research_plan"
    assert result.payload["subquestions"][0]["id"] == "SQ1"
    assert result.usage_records[0].finish_reason == "tool_calls"


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        (_tool_message("submit_research_plan", "{"), "STRUCTURED_JSON_RESPONSE_ERROR"),
        (_tool_message("unexpected_tool", "{}"), "STRUCTURED_JSON_RESPONSE_ERROR"),
        (_tool_message("submit_research_plan", ""), "STRUCTURED_JSON_RESPONSE_ERROR"),
    ],
)
def test_deepseek_fails_closed_for_invalid_or_unknown_tool_arguments(
    message: dict[str, Any], expected: str
) -> None:
    with pytest.raises(LLMProviderError) as exc:
        _call(_provider(lambda _request: httpx.Response(200, json=_body(message))))
    assert exc.value.error_code == expected


def test_deepseek_length_finish_reason_fails_closed() -> None:
    arguments = json.dumps(
        {
            "objective": "x",
            "subquestions": [{"id": "SQ1", "question": "x", "status": "OPEN"}],
            "completion_criteria": ["x"],
        }
    )
    with pytest.raises(LLMProviderError) as exc:
        _call(
            _provider(
                lambda _request: httpx.Response(
                    200,
                    json=_body(
                        _tool_message("submit_research_plan", arguments),
                        finish_reason="length",
                    ),
                )
            )
        )
    assert exc.value.error_code == "STRUCTURED_JSON_RESPONSE_ERROR"
    assert exc.value.error_details["finish_reason"] == "length"


def test_deepseek_provider_http_failure_is_not_treated_as_schema_success() -> None:
    with pytest.raises(LLMProviderError) as exc:
        _call(
            _provider(
                lambda _request: httpx.Response(401, json={"error": {"message": "denied"}})
            )
        )
    assert exc.value.error_code == "STRUCTURED_JSON_PROVIDER_ERROR"
    assert "denied" not in str(exc.value)


def test_non_deepseek_keeps_json_object_and_rejects_malformed_json() -> None:
    provider = OpenAICompatibleLLMProvider(
        "https://example.test/v1",
        "secret-value",
        "compatible-model",
        provider_name="compatible",
        max_retries=0,
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda _request: httpx.Response(
                    200,
                    json={
                        "model": "compatible-model",
                        "choices": [{"finish_reason": "stop", "message": {"content": "not-json"}}],
                        "usage": {},
                    },
                )
            )
        ),
    )
    with pytest.raises(LLMProviderError) as exc:
        provider.generate_structured_json(
            system_prompt="system", user_prompt="user", schema_name="generic", request_context={}
        )
    assert exc.value.error_code == "STRUCTURED_JSON_PARSE_ERROR"
