from __future__ import annotations

import json

from paper_research.agents.research_synthesis_provider import ResearchSynthesis
from paper_research.providers.llm import normalize_structured_json_content
from scripts.replay_research_synthesis_schema_v1 import replay_record
from tests.test_research_synthesis_provider import section_evidence_ids, valid_payload


def record(content: str):
    return {
        "attempt_number": 1,
        "content": content,
        "section_evidence_ids": section_evidence_ids(),
    }


def test_code_fence_json_is_normalized() -> None:
    payload = valid_payload()
    content = "```json\n" + json.dumps(payload) + "\n```"

    result = replay_record(record(content))

    assert result["json_parse_status"] == "passed"
    assert result["research_synthesis_schema"] == "passed"
    assert "removed_markdown_fence" in result["normalization_actions"]


def test_single_json_object_with_prefix_suffix_can_be_extracted() -> None:
    payload = valid_payload()
    content = "Here is the JSON:\n" + json.dumps(payload) + "\nDone."

    result = replay_record(record(content))

    assert result["json_parse_status"] == "passed"
    assert "extracted_single_top_level_json_object" in result["normalization_actions"]


def test_camel_case_field_aliases_are_normalized() -> None:
    payload = valid_payload()
    payload["executiveSummary"] = payload.pop("executive_summary")
    payload["researchGaps"] = payload.pop("research_gaps")
    payload["sections"][0]["sectionId"] = payload["sections"][0].pop("section_id")
    payload["sections"][0]["insufficientEvidence"] = payload["sections"][0].pop(
        "insufficient_evidence"
    )
    payload["sections"][0]["evidenceGap"] = payload["sections"][0].pop("evidence_gap")
    payload["sections"][0]["claims"][0]["citationIds"] = payload["sections"][0][
        "claims"
    ][0].pop("citation_ids")
    payload["researchGaps"][0]["description"] = payload["researchGaps"][0].pop("text")
    payload["researchGaps"][0]["citations"] = payload["researchGaps"][0].pop("citation_ids")
    payload["researchGaps"][0]["isInference"] = payload["researchGaps"][0].pop(
        "is_inference"
    )

    normalized, actions = normalize_structured_json_content(json.dumps(payload))

    assert ResearchSynthesis.model_validate(normalized)
    assert "mapped_field_alias:executiveSummary->executive_summary" in actions
    assert "mapped_field_alias:sectionId->section_id" in actions
    assert "mapped_field_alias:description->text" in actions


def test_missing_section_is_not_autofilled() -> None:
    payload = valid_payload()
    payload["sections"] = payload["sections"][:3]

    result = replay_record(record(json.dumps(payload)))

    assert result["research_synthesis_schema"] == "failed"
    assert "MISSING_SECTION" in result["failure_types"]


def test_unknown_citation_is_not_repaired() -> None:
    payload = valid_payload()
    payload["sections"][0]["claims"][0]["citation_ids"] = ["E999"]

    result = replay_record(record(json.dumps(payload)))

    assert result["research_synthesis_schema"] == "failed"
    assert "UNKNOWN_CITATION_ID" in result["failure_types"]


def test_insufficient_section_may_have_empty_claims() -> None:
    payload = valid_payload()
    payload["sections"][2]["claims"] = []
    payload["sections"][2]["insufficient_evidence"] = True
    payload["sections"][2]["evidence_gap"] = "No quantitative results were retrieved."

    result = replay_record(record(json.dumps(payload)))

    assert result["research_synthesis_schema"] == "passed"


def test_sufficient_section_cannot_have_empty_claims() -> None:
    payload = valid_payload()
    payload["sections"][2]["claims"] = []
    payload["sections"][2]["insufficient_evidence"] = False

    result = replay_record(record(json.dumps(payload)))

    assert result["research_synthesis_schema"] == "failed"
    assert "CROSS_FIELD_VALIDATION_FAILURE" in result["failure_types"]


def test_section_citation_diagnostics_survive_unrelated_schema_failure() -> None:
    payload = valid_payload()
    payload["research_gaps"] = ["Wrong gap shape."]
    payload["sections"][0]["claims"][0]["citation_ids"] = ["E02"]

    result = replay_record(record(json.dumps(payload)))

    assert result["research_synthesis_schema"] == "failed"
    assert "WRONG_FIELD_TYPE" in result["failure_types"]
    assert "CITATION_NOT_ALLOWED_FOR_SECTION" in result["failure_types"]
    assert result["offending_citation_ids"] == ["E02"]
