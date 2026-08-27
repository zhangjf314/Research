from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from paper_research.agents.research_synthesis_provider import DeepSeekResearchSynthesisProvider
from paper_research.providers.llm import LLMProviderError, SiliconFlowLLMProvider
from scripts.replay_research_synthesis_schema_v1 import replay_record
from tests.test_research_synthesis_provider import (
    evidence_catalog,
    section_evidence_ids,
    valid_payload,
)


class FakeChatClient:
    def __init__(self, contents: list[str]) -> None:
        self.contents = list(contents)
        self.calls = 0

    def post(self, *args: Any, **kwargs: Any) -> httpx.Response:
        del args, kwargs
        self.calls += 1
        content = self.contents.pop(0)
        return httpx.Response(
            200,
            headers={"x-request-id": f"req-{self.calls}"},
            json={
                "id": f"resp-{self.calls}",
                "model": "deepseek-v4-flash",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": content},
                    }
                ],
                "usage": {
                    "prompt_tokens": 11,
                    "completion_tokens": 22,
                    "total_tokens": 33,
                },
            },
        )


def provider(tmp_path: Path, contents: list[str]) -> SiliconFlowLLMProvider:
    return SiliconFlowLLMProvider(
        base_url="https://api.deepseek.com/v1",
        api_key="secret-test-key",
        model="deepseek-v4-flash",
        client=FakeChatClient(contents),
        input_cost_per_million=1.0,
        output_cost_per_million=1.0,
        # This fixture verifies JSON-object raw-response persistence.  DeepSeek
        # has its own exercised function-call adapter tests.
        provider_name="compat-test",
    )


def context() -> dict[str, Any]:
    return {
        "task_id": "obs-task",
        "run_id": "obs-task",
        "section_evidence_ids": section_evidence_ids(),
        "allowed_citation_ids": list(evidence_catalog()),
    }


def test_raw_response_persisted_on_schema_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    llm = provider(tmp_path, [json.dumps(valid_payload())])

    result = llm.generate_structured_json(
        system_prompt="system",
        user_prompt="user",
        schema_name="deep-research-synthesis-v1",
        request_context=context(),
    )

    files = sorted(Path(".runtime/research-synthesis-provider/obs-task").glob("attempt-*.json"))
    assert len(files) == 1
    raw = json.loads(files[0].read_text(encoding="utf-8"))
    assert raw["content_sha256"]
    assert raw["raw_content"]
    assert "secret-test-key" not in files[0].read_text(encoding="utf-8")
    assert "Authorization" not in files[0].read_text(encoding="utf-8")
    assert result.usage_records[0].raw_response_path == str(files[0])


def test_raw_response_persisted_on_json_parse_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    llm = provider(tmp_path, ["not json"])

    with pytest.raises(LLMProviderError) as exc:
        llm.generate_structured_json(
            system_prompt="system",
            user_prompt="user",
            schema_name="deep-research-synthesis-v1",
            request_context=context(),
        )

    files = sorted(Path(".runtime/research-synthesis-provider/obs-task").glob("attempt-*.json"))
    assert len(files) == 1
    assert len(exc.value.usage_records) == 1
    assert exc.value.usage_records[0].usage.total_tokens == 33
    assert exc.value.usage_records[0].raw_response_path == str(files[0])


def test_two_completed_attempts_create_two_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    llm = provider(tmp_path, [json.dumps({"title": "bad"}), json.dumps({"title": "bad2"})])
    adapter = DeepSeekResearchSynthesisProvider(llm)

    with pytest.raises(LLMProviderError):
        adapter.synthesize(
            question="question",
            section_queries={},
            evidence_catalog=evidence_catalog(),
            section_evidence_ids=section_evidence_ids(),
            contradictions=[],
            request_context=context(),
        )

    files = sorted(Path(".runtime/research-synthesis-provider/obs-task").glob("attempt-*.json"))
    assert [path.name for path in files] == ["attempt-01.json", "attempt-02.json"]
    assert files[0].read_text(encoding="utf-8") != files[1].read_text(encoding="utf-8")


def test_persisted_raw_response_is_replay_compatible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    llm = provider(tmp_path, ["```json\n" + json.dumps(valid_payload()) + "\n```"])

    llm.generate_structured_json(
        system_prompt="system",
        user_prompt="user",
        schema_name="deep-research-synthesis-v1",
        request_context=context(),
    )

    raw = json.loads(
        Path(".runtime/research-synthesis-provider/obs-task/attempt-01.json").read_text(
            encoding="utf-8"
        )
    )
    result = replay_record(raw)
    assert result["json_parse_status"] == "passed"
    assert result["research_synthesis_schema"] == "passed"
    assert result["report_quality_gate"] == "passed"
