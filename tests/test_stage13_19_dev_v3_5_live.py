from __future__ import annotations

import json

import pytest

from scripts import run_evidence_qa_dev_v3_5
from scripts.evidence_qa_dev_lib_v1 import DEV_IDS
from scripts.evidence_qa_dev_v3_5_lib import (
    build_prompt_delivery_freeze,
    build_protocol_freeze,
)
from scripts.run_evidence_qa_dev_v3_5 import _shape_counts, preflight


def test_dev_v3_5_protocol_freeze_uses_payload_v4_without_repair() -> None:
    protocol = build_protocol_freeze()
    assert protocol["question_ids"] == DEV_IDS
    assert protocol["model_payload_schema_version"] == "required-claim-model-payload-v4"
    assert protocol["prompt_version"] == "qa-required-claims-discriminated-slots-v3.6-candidate"
    assert protocol["transport"]["response_format"] == {"type": "json_object"}
    assert protocol["transport"]["json_schema"] is False
    assert protocol["retry_policy"] == {"provider": 0, "json": 0, "citation": 0}
    assert protocol["canonicalization"] == "none"
    assert protocol["normalization"] == "none"
    assert protocol["repair"] == "none"
    assert protocol["fallback"] == "none"
    assert protocol["reranker_enabled"] is False
    assert protocol["gold_used_online"] is False
    assert protocol["human_labels_used_online"] is False
    assert protocol["next_live_authorized"] is False


def test_prompt_delivery_freeze_records_per_question_hashes() -> None:
    freeze = build_prompt_delivery_freeze()
    assert freeze["question_count"] == 10
    assert freeze["old_prompt_mixed_in"] is False
    assert freeze["payload_schema_mismatch"] is False
    assert freeze["schema_hash"] == build_protocol_freeze()["model_payload_schema_hash"]
    rows = freeze["questions"]
    assert [row["question_id"] for row in rows] == DEV_IDS
    for row in rows:
        assert row["delivered_system_prompt_hash"]
        assert row["delivered_user_payload_hash"]
        assert row["protocol_signature"] == freeze["protocol_signature"]
        assert row["schema_hash"] == freeze["schema_hash"]


def test_preflight_passes_without_provider_network(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeSettings:
        llm_provider = "siliconflow"
        llm_model = "Qwen/Qwen3-8B"
        llm_temperature = 0
        llm_max_retries = 0
        rerank_enabled = False
        llm_billing_mode = "free"

    monkeypatch.setattr(run_evidence_qa_dev_v3_5, "Settings", FakeSettings)
    result = preflight()
    assert result["checks"]["prompt_delivery"] is True
    assert result["checks"]["protocol_v4"] is True


@pytest.mark.parametrize(
    ("slot", "expected"),
    [
        ({"required_claim_id": "c1", "claim_text": "Claim."}, (1, 0, 0)),
        (
            {"required_claim_id": "c1", "omission_reason": "Evidence missing."},
            (0, 1, 0),
        ),
        (
            {
                "required_claim_id": "c1",
                "claim_text": "Claim.",
                "omission_reason": "Conflict.",
            },
            (0, 0, 1),
        ),
        ({"required_claim_id": "c1", "claim_text": ""}, (0, 0, 1)),
        ({"required_claim_id": "c1", "status": "answered"}, (0, 0, 1)),
    ],
)
def test_shape_counts_distinguish_answered_unsupported_and_conflict(
    slot: dict, expected: tuple[int, int, int]
) -> None:
    raw = {"answerable": True, "required_claim_results": [slot]}
    counts = _shape_counts(raw)
    assert (
        counts["answered_shape"],
        counts["unsupported_shape"],
        counts["invalid_shape"],
    ) == expected


def test_protocol_schema_does_not_expose_status_or_citations() -> None:
    schema = json.dumps(build_protocol_freeze(), ensure_ascii=False).lower()
    assert "json_schema" in schema
    assert '"status"' not in schema
    assert "citation_ids" not in schema
    assert "gold_used_online" in schema
