from __future__ import annotations

from paper_research.agents.research_synthesis_provider import (
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
    invalid = _invalid_section_ids(
        payload,
        validation_error=ValueError("citation IDs outside section allowlist"),
        section_evidence_ids=section_ids,
    )
    assert invalid == {"background"}
    prompt = _research_component_repair_prompt(
        previous_payload=payload,
        validation_error=ValueError("bad citation"),
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
