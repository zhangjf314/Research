from __future__ import annotations

import json

import pytest

from paper_research.generation.required_claim_output import RequiredClaimValidationError
from paper_research.generation.schema_reliability import (
    dev_v3_7_candidate_system_prompt,
    validate_payload_v4,
)
from scripts.audit_prompt_output_field_contamination_v1 import build_audit as build_prompt_audit
from scripts.evidence_presentation_v2_lib import (
    BEGIN,
    END,
    build_protocol,
    control_occurrences,
    presentation_input,
    render_user_prompt,
    rendered_messages,
    source_content_occurrences,
)
from scripts.render_dev_v3_6_prompt_preflight_v1 import render_rows


def test_evidence_presentation_v2_protocol_keeps_payload_v4() -> None:
    protocol = build_protocol()
    assert protocol["version"] == "evidence-presentation-v2-candidate"
    assert protocol["selected_format"] == "uniform-unnumbered-delimiter"
    assert protocol["payload_schema_version"] == "required-claim-model-payload-v4"
    assert protocol["next_live_authorized"] is False
    assert protocol["model_visible_candidate_metadata"] == []


def test_prompt_v37_does_not_contain_metadata_or_label_terms() -> None:
    prompt = dev_v3_7_candidate_system_prompt().lower()
    for token in (
        "evidence_label",
        "evidence_id",
        "citation_id",
        "block_id",
        "paper_id",
        "status",
        "gold",
        "human",
        "label",
    ):
        assert token not in prompt


def test_rendered_prompt_removes_candidate_metadata_but_keeps_passage_text() -> None:
    model_input, local = presentation_input("q013")
    rendered = render_user_prompt(model_input)
    assert '"label"' not in rendered
    assert "Evidence A" not in rendered
    assert BEGIN in rendered
    assert END in rendered
    assert local["local_mapping"]
    assert all(rows for rows in local["local_mapping"].values())


def test_source_content_evidence_label_literal_is_not_control_leakage() -> None:
    model_input = {
        "question": "Does the source mention evidence_label?",
        "answerability_expectation": True,
        "required_claims": [
            {
                "required_claim_id": "c1",
                "required_claim_text": "The source discusses evidence_label as text.",
                "passages": ["The literal token evidence_label appears in source content."],
                "omission_policy": "Use omission_reason when missing.",
            }
        ],
        "output_budget": {"required_claim_count": 1},
    }
    messages = [
        {"role": "system", "content": dev_v3_7_candidate_system_prompt()},
        {"role": "user", "content": render_user_prompt(model_input)},
    ]
    assert source_content_occurrences(model_input, "evidence_label") == 3
    assert control_occurrences(messages, model_input, "evidence_label") == 0


@pytest.mark.parametrize(
    "payload",
    [
        {
            "answerable": True,
            "required_claim_results": [
                {
                    "required_claim_id": "c1",
                    "claim_text": "Claim.",
                    "evidence_label": "Evidence A",
                }
            ],
        },
        {
            "answerable": True,
            "required_claim_results": [
                {
                    "required_claim_id": "c1",
                    "claim_text": "Claim.",
                    "arbitrary_extra": "x",
                }
            ],
        },
        {
            "answerable": True,
            "required_claim_results": [
                {
                    "required_claim_id": "c1",
                    "claim_text": "Claim.",
                    "omission_reason": "Conflict.",
                }
            ],
        },
    ],
)
def test_payload_v4_extra_and_dual_fields_still_fail(payload: dict) -> None:
    with pytest.raises(RequiredClaimValidationError):
        validate_payload_v4(json.dumps(payload), expected_claim_ids=["c1"])


def test_q005_unanswerable_payload_still_passes() -> None:
    payload = {
        "answerable": False,
        "required_claim_results": [],
        "refusal_reason": "The supplied passages do not address the question.",
    }
    assert validate_payload_v4(json.dumps(payload), expected_claim_ids=[]).answerable is False


def test_prompt_contamination_audit_passes_and_render_hashes_are_stable() -> None:
    audit = build_prompt_audit()
    assert audit["gate"] == "PASSED"
    rows = render_rows()
    assert len(rows) == 10
    assert all(row["render_hash_consistent"] for row in rows)
    assert all(row["passage_text_hashes_unchanged"] for row in rows)


def test_rendered_messages_keep_local_mapping_without_model_visible_ids() -> None:
    messages, _model_input, local = rendered_messages("q001")
    rendered = "\n".join(message["content"] for message in messages)
    assert "evidence_id" not in rendered
    assert "citation_id" not in rendered
    assert "block_id" not in rendered
    assert local["citation_registry_hash"]
    assert local["local_mapping"]
