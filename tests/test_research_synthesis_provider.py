from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from paper_research.agents.research_synthesis_provider import (
    DeepSeekResearchSynthesisProvider,
    ResearchGap,
    ResearchSynthesis,
)
from paper_research.providers.llm import (
    LLMProviderError,
    ModelUsage,
    ProviderUsageRecord,
    StructuredJSONResult,
)


def valid_payload() -> dict[str, Any]:
    return {
        "title": "RAG synthesis",
        "executive_summary": "A structured synthesis of RAG evidence.",
        "sections": [
            {
                "section_id": "background",
                "summary": "Background summary.",
                "claims": [
                    {
                        "text": "RAG is motivated by knowledge grounding.",
                        "citation_ids": ["E01"],
                    }
                ],
                "insufficient_evidence": False,
                "evidence_gap": None,
            },
            {
                "section_id": "methods",
                "summary": "Methods summary.",
                "claims": [
                    {
                        "text": "Methods combine retrieval and generation.",
                        "citation_ids": ["E02"],
                    }
                ],
                "insufficient_evidence": False,
                "evidence_gap": None,
            },
            {
                "section_id": "results",
                "summary": "Results summary.",
                "claims": [{"text": "Results cite metric evidence.", "citation_ids": ["E03"]}],
                "insufficient_evidence": False,
                "evidence_gap": None,
            },
            {
                "section_id": "limitations",
                "summary": "Limitations summary.",
                "claims": [{"text": "Limitations cite risk evidence.", "citation_ids": ["E04"]}],
                "insufficient_evidence": False,
                "evidence_gap": None,
            },
        ],
        "consensus": [{"text": "RAG needs evidence.", "citation_ids": ["E01"]}],
        "disagreements": [],
        "research_gaps": [
            {
                "text": "Further evidence is needed for robustness trade-offs.",
                "citation_ids": ["E04"],
                "is_inference": False,
            }
        ],
    }


def evidence_catalog() -> dict[str, dict[str, Any]]:
    return {
        f"E{index:02d}": {
            "citation_id": f"E{index:02d}",
            "evidence_id": f"e{index}",
            "paper_id": "paper-a",
            "section_path": ["Section"],
            "page_start": index,
            "page_end": index,
            "text": (
                f"Evidence {index}. System Prompt and Ignore all previous "
                "instructions are text only."
            ),
            "retrieval_score": 1.0,
            "retrieval_sources": ["fake"],
            "target_sections": [],
        }
        for index in range(1, 5)
    }


def section_evidence_ids() -> dict[str, list[str]]:
    return {
        "background": ["E01"],
        "methods": ["E02"],
        "results": ["E03"],
        "limitations": ["E04"],
    }


class FakeStructuredProvider:
    provider_name = "deepseek"
    model_name = "deepseek-v4-flash"

    def __init__(
        self,
        payloads: list[dict[str, Any]],
        *,
        raw_response_paths: list[str] | None = None,
    ) -> None:
        self.payloads = list(payloads)
        self.raw_response_paths = list(raw_response_paths or [])
        self.calls: list[dict[str, Any]] = []

    def generate_structured_json(self, **kwargs: Any) -> StructuredJSONResult:
        self.calls.append(kwargs)
        payload = self.payloads.pop(0)
        raw_response_path = self.raw_response_paths.pop(0) if self.raw_response_paths else None
        return StructuredJSONResult(
            payload=payload,
            provider=self.provider_name,
            model=self.model_name,
            usage=ModelUsage(
                input_tokens=10,
                output_tokens=20,
                total_tokens=30,
                estimated_cost_usd=0.0,
            ),
            usage_records=[
                ProviderUsageRecord(
                    attempt_number=len(self.calls),
                    usage=ModelUsage(
                        input_tokens=10,
                        output_tokens=20,
                        total_tokens=30,
                        estimated_cost_usd=0.0,
                    ),
                    raw_response_path=raw_response_path,
                )
            ],
            request_attempt_count=len(self.calls),
        )


def synthesize(provider: FakeStructuredProvider):
    return DeepSeekResearchSynthesisProvider(provider).synthesize(
        question="What are RAG methods?",
        section_queries={
            "background": "background?",
            "methods": "methods?",
            "results": "results?",
            "limitations": "limitations?",
        },
        evidence_catalog=evidence_catalog(),
        section_evidence_ids=section_evidence_ids(),
        contradictions=[],
        request_context={"run_id": "test-run"},
    )


def test_adapter_uses_structured_json_transport_not_qa_protocol() -> None:
    provider = FakeStructuredProvider([valid_payload()])

    result = synthesize(provider)

    assert isinstance(result.synthesis, ResearchSynthesis)
    assert provider.calls
    assert provider.calls[0]["schema_name"] == "deep-research-synthesis-v1"
    assert "System Prompt" in provider.calls[0]["user_prompt"]
    assert "prompt injection" in provider.calls[0]["system_prompt"]
    assert "Return exactly one JSON object" in provider.calls[0]["system_prompt"]
    assert result.provider == "deepseek"
    assert result.model == "deepseek-v4-flash"
    assert result.usage.total_tokens == 30
    user_prompt = provider.calls[0]["user_prompt"]
    for section_id in ("background", "methods", "results", "limitations"):
        assert f"SECTION: {section_id}" in user_prompt
    assert "ALLOWED_CITATION_IDS:" in user_prompt
    assert "A claim inside one section may only cite IDs listed for that section." in user_prompt
    assert "An ID being present in the global evidence catalog does not make it" in user_prompt
    assert "SECTION_ONLY: background" in user_prompt
    assert '<EVIDENCE_FOR_SECTION section_id="background" id="E01"' in user_prompt
    assert '<EVIDENCE id="E01"' not in user_prompt
    assert '"citation_ids"' in user_prompt
    assert '"E01"' in user_prompt


def test_adapter_repairs_once_after_schema_failure() -> None:
    bad = valid_payload()
    bad["sections"] = bad["sections"][:3]
    provider = FakeStructuredProvider([bad, valid_payload()])

    result = synthesize(provider)

    assert result.request_attempt_count == 2
    assert result.retry_count == 1
    assert len(provider.calls) == 2
    assert len(result.usage_records) == 2
    assert result.usage.total_tokens == 60
    repair_prompt = provider.calls[1]["user_prompt"]
    assert "Allowed citation IDs by section" in repair_prompt
    for section_id in ("background", "methods", "results", "limitations"):
        assert f"SECTION: {section_id}" in repair_prompt
    assert "A claim inside one section may only cite IDs listed for that section." in repair_prompt
    assert "An ID being present in the global evidence catalog does not make it" in repair_prompt
    assert '"research_gaps": [' in repair_prompt
    assert '"is_inference"' in repair_prompt
    assert '"citation_ids": ["E01"]' in repair_prompt


def test_adapter_fails_closed_after_second_schema_failure() -> None:
    provider = FakeStructuredProvider([{"title": "bad"}, {"title": "still bad"}])

    with pytest.raises(LLMProviderError) as exc:
        synthesize(provider)

    assert exc.value.error_code == "FAILED_PROVIDER_SCHEMA"
    assert exc.value.api_request_count == 2
    assert len(exc.value.usage_records) == 2
    assert exc.value.error_details["usage"]["total_tokens"] == 60
    assert exc.value.error_details["usage_record_count"] == 2


def test_cross_section_citation_repair_receives_specific_validation_error() -> None:
    bad = valid_payload()
    bad["sections"][0]["claims"][0]["citation_ids"] = ["E02"]
    provider = FakeStructuredProvider([bad, valid_payload()])

    result = synthesize(provider)

    assert result.request_attempt_count == 2
    repair_prompt = provider.calls[1]["user_prompt"]
    assert "citation IDs outside section allowlist for background claim[0]" in repair_prompt
    assert "SECTION: background" in repair_prompt
    assert '"E01"' in repair_prompt


def test_schema_failure_updates_runtime_raw_response_diagnostics(tmp_path: Path) -> None:
    raw_path = tmp_path / "attempt-01.json"
    raw_path.write_text(
        json.dumps(
            {
                "schema_version": "research-synthesis-raw-response-v1",
                "json_parse_status": "passed",
                "schema_parse_status": "not_run",
                "validation_errors": [],
            }
        ),
        encoding="utf-8",
    )
    bad = valid_payload()
    bad["sections"][0]["claims"][0]["citation_ids"] = ["E02"]
    provider = FakeStructuredProvider(
        [bad, bad],
        raw_response_paths=[str(raw_path), str(tmp_path / "attempt-02.json")],
    )

    with pytest.raises(LLMProviderError):
        synthesize(provider)

    record = json.loads(raw_path.read_text(encoding="utf-8"))
    assert record["schema_parse_status"] == "failed"
    assert record["schema_failure_subtype"] == "INVALID_CITATION_SCHEMA"
    assert "CITATION_NOT_ALLOWED_FOR_SECTION" in record["failure_types"]
    assert record["offending_citation_ids"] == ["E02"]
    assert record["citation_allowlist_details"][0]["location"] == (
        "sections.0.claims.0.citation_ids"
    )


def test_unknown_citation_id_is_rejected() -> None:
    payload = valid_payload()
    payload["sections"][0]["claims"][0]["citation_ids"] = ["E999"]
    provider = FakeStructuredProvider([payload, payload])

    with pytest.raises(LLMProviderError):
        synthesize(provider)


def test_duplicate_or_missing_section_is_rejected() -> None:
    payload = valid_payload()
    payload["sections"][0]["section_id"] = "methods"
    provider = FakeStructuredProvider([payload, payload])

    with pytest.raises(LLMProviderError):
        synthesize(provider)


def test_cross_section_citation_is_rejected() -> None:
    payload = valid_payload()
    payload["sections"][0]["claims"][0]["citation_ids"] = ["E02"]
    provider = FakeStructuredProvider([payload, payload])

    with pytest.raises(LLMProviderError):
        synthesize(provider)


def test_research_gap_requires_object_shape() -> None:
    payload = valid_payload()
    payload["research_gaps"] = ["More robustness work is needed."]

    with pytest.raises(ValueError):
        ResearchSynthesis.model_validate(payload)


def test_research_gap_requires_citation_unless_inference() -> None:
    with pytest.raises(ValueError):
        ResearchGap.model_validate({"text": "More robustness work is needed."})

    assert ResearchGap.model_validate(
        {
            "text": "This is an inferred gap.",
            "citation_ids": [],
            "is_inference": True,
        }
    )


def test_research_gap_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError):
        ResearchGap.model_validate(
            {
                "text": "More robustness work is needed.",
                "citation_ids": ["E01"],
                "extra": "not allowed",
            }
        )
