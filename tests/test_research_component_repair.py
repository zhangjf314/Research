from __future__ import annotations

from paper_research.agents.research_synthesis_provider import (
    CitationScopeValidationError,
    CitationScopeViolation,
    _invalid_section_ids,
    _merge_component_repair_payload,
    _research_component_repair_prompt,
)


def test_component_repair_targets_only_invalid_section_evidence() -> None:
    payload = {
        "sections": [
            {"section_id": "background", "claims": [{"citation_ids": ["E09"]}]},
            {"section_id": "methods", "claims": [{"citation_ids": ["E02"]}]},
            {"section_id": "results", "claims": [{"citation_ids": ["E03"]}]},
            {"section_id": "limitations", "claims": [{"citation_ids": ["E04"]}]},
        ]
    }
    section_ids = {
        "background": ["E01"],
        "methods": ["E02"],
        "results": ["E03"],
        "limitations": ["E04"],
    }
    error = CitationScopeValidationError(
        [
            CitationScopeViolation(
                section_id="background",
                section_title="Background",
                claim_path="sections[0].claims[0]",
                claim_index=0,
                offending_citation_ids=["E09"],
                allowed_citation_ids=["E01"],
                globally_known_but_section_disallowed_ids=["E09"],
                unknown_citation_ids=[],
                validation_code="CITATION_NOT_ALLOWED_FOR_SECTION",
                validation_message="bad citation",
            )
        ]
    )
    invalid = _invalid_section_ids(
        payload,
        validation_error=error,
        section_evidence_ids=section_ids,
    )
    assert invalid == {"background"}
    prompt = _research_component_repair_prompt(
        previous_payload=payload,
        validation_error=error,
        evidence_catalog={
            "E01": {
                "paper_id": "p1",
                "text": "Background evidence",
                "section_path": ["Intro"],
                "page_start": 1,
                "page_end": 1,
                "evidence_id": "b1",
            },
            "E02": {
                "paper_id": "p2",
                "text": "Methods evidence",
                "section_path": ["Methods"],
                "page_start": 2,
                "page_end": 2,
                "evidence_id": "b2",
            },
        },
        section_evidence_ids=section_ids,
        target_sections=invalid,
    )
    assert "SECTION: background" in prompt
    assert "SECTION: methods" not in prompt
    assert "<EVIDENCE id=\"E01\"" in prompt
    assert "<EVIDENCE id=\"E02\"" not in prompt
    assert "Structured citation violations" in prompt
    assert "CITATION_NOT_ALLOWED_FOR_SECTION" in prompt


def test_component_repair_merge_replaces_only_target_section() -> None:
    previous = {
        "sections": [
            {"section_id": "background", "summary": "bad"},
            {"section_id": "methods", "summary": "keep"},
        ]
    }
    repair = {"sections": [{"section_id": "background", "summary": "fixed"}]}
    merged = _merge_component_repair_payload(
        previous_payload=previous,
        repair_payload=repair,
        target_sections={"background"},
    )
    assert merged["sections"][0]["summary"] == "fixed"
    assert merged["sections"][1]["summary"] == "keep"


def test_component_repair_multi_section_keeps_non_offending_sections() -> None:
    previous = {
        "sections": [
            {"section_id": "background", "summary": "bad background"},
            {"section_id": "methods", "summary": "keep methods"},
            {"section_id": "results", "summary": "bad results"},
            {"section_id": "limitations", "summary": "keep limitations"},
        ]
    }
    repair = {
        "sections": [
            {"section_id": "background", "summary": "fixed background"},
            {"section_id": "results", "summary": "fixed results"},
        ]
    }

    merged = _merge_component_repair_payload(
        previous_payload=previous,
        repair_payload=repair,
        target_sections={"background", "results"},
    )

    assert merged["sections"][0]["summary"] == "fixed background"
    assert merged["sections"][1] == previous["sections"][1]
    assert merged["sections"][2]["summary"] == "fixed results"
    assert merged["sections"][3] == previous["sections"][3]


def test_component_repair_rejects_missing_or_extra_sections() -> None:
    previous = {
        "sections": [
            {"section_id": "background", "summary": "bad"},
            {"section_id": "methods", "summary": "bad"},
        ]
    }
    repair = {"sections": [{"section_id": "background", "summary": "fixed"}]}

    try:
        _merge_component_repair_payload(
            previous_payload=previous,
            repair_payload=repair,
            target_sections={"background", "methods"},
        )
    except ValueError as exc:
        assert "wrong section set" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected targeted repair failure")


def test_non_citation_error_uses_full_repair_target() -> None:
    payload = {
        "sections": [
            {"section_id": "background", "claims": [{"citation_ids": ["E09"]}]},
            {"section_id": "methods", "claims": [{"citation_ids": ["E02"]}]},
            {"section_id": "results", "claims": [{"citation_ids": ["E03"]}]},
            {"section_id": "limitations", "claims": [{"citation_ids": ["E04"]}]},
        ]
    }
    section_ids = {
        "background": ["E01"],
        "methods": ["E02"],
        "results": ["E03"],
        "limitations": ["E04"],
    }

    invalid = _invalid_section_ids(
        payload,
        validation_error=ValueError("generic schema failure"),
        section_evidence_ids=section_ids,
    )

    assert invalid == {"background", "methods", "results", "limitations"}
