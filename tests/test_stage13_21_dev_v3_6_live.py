"""Offline safeguards for Stage 13.21 Dev v3.6 controlled live artifacts."""

from __future__ import annotations

import json

from scripts import audit_evidence_qa_dev_v3_6 as audit_v36
from scripts import run_evidence_qa_dev_v3_6 as run_v36
from scripts.evidence_qa_dev_v3_6_lib import FINAL_AUDIT, OUTPUT, build_protocol_freeze


def test_dev_v3_6_protocol_freeze_uses_frozen_payload_and_presentation() -> None:
    protocol = build_protocol_freeze()

    assert protocol["evaluation_version"] == "evidence-qa-dev-v3.6"
    assert protocol["prompt_version"] == "qa-required-claims-discriminated-slots-v3.7-candidate"
    assert protocol["payload_v4_version"] == "required-claim-model-payload-v4"
    assert protocol["envelope_v4_version"] == "required-claim-local-envelope-v4"
    assert protocol["evidence_presentation_version"] == "evidence-presentation-v2-candidate"
    assert protocol["selected_rendering_format"] == "uniform-unnumbered-delimiter"
    assert protocol["retry_policy"] == {
        "provider": 0,
        "json": 0,
        "citation": 0,
        "repair": 0,
    }
    assert protocol["frozen_before_live"] is True
    assert protocol["historical_results_immutable"] is True


def test_dev_v3_6_shape_and_sentinel_detection_is_strict() -> None:
    raw_payload = {
        "answerable": True,
        "required_claim_results": [
            {"required_claim_id": "cl-1", "claim_text": "supported"},
            {"required_claim_id": "cl-2", "omission_reason": "unsupported"},
            {
                "required_claim_id": "cl-3",
                "claim_text": "conflict",
                "omission_reason": "also conflict",
            },
        ],
        "status": "answerable",
        "citations": [],
        "extra": {"empty": "", "null_value": None},
    }

    shape = run_v36._shape_counts(raw_payload)

    assert shape["answered_shape"] == 1
    assert shape["unsupported_shape"] == 1
    assert shape["invalid_shape"] == 1
    assert shape["dual_semantic_conflict"] == 1
    assert run_v36._contains_key(raw_payload, {"status"}) == 1
    assert run_v36._contains_key(raw_payload, {"citations"}) == 1
    assert run_v36._contains_null(raw_payload) == 1
    assert run_v36._contains_empty_string(raw_payload) == 1


def test_dev_v3_6_final_audit_gates_match_current_summary() -> None:
    audit_v36.main()
    audit = json.loads(FINAL_AUDIT.read_text(encoding="utf-8"))
    summary = json.loads(OUTPUT.read_text(encoding="utf-8"))

    assert audit["DEV_V3_6_PROVIDER_HEALTH"] == "PASSED"
    assert audit["DEV_V3_6_PROMPT_CONTAMINATION_GATE"] == "PASSED"
    assert audit["DEV_V3_6_RAW_PAYLOAD_GATE"] == "PASSED"
    assert audit["DEV_V3_6_FINAL_POLICY_ENGINEERING_GATE"] == "PASSED"
    assert audit["DEV_V3_6_ENGINEERING_GATE"] == "PASSED"
    assert audit["DEV_V3_6_AUTOMATED_QUALITY_GATE"] == "FAILED"
    assert audit["DEV_V3_6_HUMAN_SUPPORT_GATE"] == "PENDING"
    assert audit["READY_FOR_FULL_QA"] is False
    assert audit["READY_FOR_DEV_V3_6_CHECKPOINT_COMMIT"] is True
    assert summary["raw_payload_layer"]["evidence_label_leakage"] == 0
    assert summary["all_manifest_conservative"]["retries"] == 0
