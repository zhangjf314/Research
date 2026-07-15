from __future__ import annotations

import json
from pathlib import Path

import pytest

from paper_research.generation.prompts import (
    QA_REQUIRED_CLAIMS_CITATION_ID_V3_1,
    qa_system_prompt,
)
from paper_research.generation.response_normalization import normalize_response
from paper_research.providers.capabilities import (
    ProviderCapabilities,
    VerificationStatus,
    siliconflow_qwen3_8b_stage13_5_snapshot,
)

ROOT = Path(__file__).parents[1]
DATA = ROOT / "data/evaluation"


def complete_payload() -> dict:
    return {
        "question_id": "q001",
        "answerable": True,
        "required_claim_results": [],
        "refusal_reason": None,
        "prompt_version": "qa-required-claims-citation-id-v3.1",
        "citation_protocol": "citation-id-v2",
    }


def test_freeze_is_stable_and_has_ten_runs() -> None:
    body = json.loads((DATA / "stage13-5-schema-failure-freeze-v1.json").read_text())
    assert body["record_count"] == 10
    assert len(body["freeze_signature"]) == 64
    assert all(row["official_status"] == "validation_failed" for row in body["records"])


def test_actual_shape_families_and_q050() -> None:
    lines = (DATA / "dev-v3-response-shape-audit-v1.jsonl").read_text().splitlines()
    rows = [json.loads(line) for line in lines if line]
    families = {row["question_id"]: row["detected_schema_family"] for row in rows}
    assert families["q050"] == "question_id_wrapper"
    assert families["q002"] == "required_claim_id_map"
    assert families["q005"] == "legacy_refusal"
    assert all(row["raw_json_valid"] for row in rows)


def test_exact_question_wrapper_is_the_only_allowed_rewrite() -> None:
    result = normalize_response(
        {"q001": complete_payload()}, question_id="q001", expected_claim_ids=[]
    )
    assert result.accepted
    assert result.operations == ("single_exact_question_wrapper_unwrap",)


@pytest.mark.parametrize(
    "raw",
    [
        {"q002": complete_payload()},
        {"q001": complete_payload(), "extra": {}},
        {"claims": []},
        "not-json",
    ],
)
def test_unsafe_normalization_is_rejected(raw: object) -> None:
    assert not normalize_response(raw, question_id="q001", expected_claim_ids=[]).accepted


def test_claim_map_missing_envelope_is_rejected() -> None:
    raw = {
        "cl-a": {
            "status": "answered",
            "claim_text": "x",
            "citation_ids": ["E001"],
            "omission_reason": None,
        }
    }
    result = normalize_response(raw, question_id="q001", expected_claim_ids=["cl-a"])
    assert not result.accepted
    assert "adding missing v3 envelope" in result.reason


def test_free_triple_is_rejected() -> None:
    result = normalize_response(
        {"q001": {**complete_payload(), "paper_id": "x"}},
        question_id="q001",
        expected_claim_ids=[],
    )
    assert not result.accepted


def test_unknown_capability_fails_closed() -> None:
    capability = ProviderCapabilities(
        provider="x",
        model="y",
        supports_json_object=None,
        supports_json_schema=None,
        supports_tool_calling=None,
        supports_strict_schema=None,
        capability_source="none",
        verified_at=None,
        verification_status=VerificationStatus.UNKNOWN,
    )
    with pytest.raises(RuntimeError):
        capability.require("supports_json_schema")


def test_verified_json_object_snapshot_and_hash() -> None:
    capability = siliconflow_qwen3_8b_stage13_5_snapshot()
    capability.require("supports_json_object")
    assert capability.supports_json_schema is None
    assert len(capability.snapshot_hash) == 64


def test_prompt_v31_is_explicit_and_versioned() -> None:
    prompt = qa_system_prompt(QA_REQUIRED_CLAIMS_CITATION_ID_V3_1)
    assert "Never wrap" in prompt
    assert "Never use a legacy claims field" in prompt
    assert "Answerable example" in prompt
    assert "Unanswerable example" in prompt
    assert "Do not use Markdown" in prompt


def test_replay_does_not_replace_official_result() -> None:
    replay = json.loads(
        (DATA / "dev-v3-response-normalization-replay-v1.json").read_text()
    )
    official = json.loads((DATA / "evidence-qa-dev-v3.json").read_text())
    assert replay["official_stage13_5_modified"] is False
    assert official["dev_v3_engineering_gate"] is False


def test_readiness_is_offline_and_not_authorized() -> None:
    body = json.loads(
        (DATA / "evidence-qa-dev-v3-1-readiness-v1.json").read_text()
    )
    assert body["ready_for_dev_v3_1"] is True
    assert body["dev_v3_1_authorized"] is False
    assert body["dev_v3_1_live_run"] is False
    assert body["formal_live_normalization_policy"] == "raw_schema_passed_only"
