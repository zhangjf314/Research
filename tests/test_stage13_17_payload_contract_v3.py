from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from paper_research.evaluation.request_accounting import (
    RequestTerminalState,
    close_reservation_for_terminal_run,
)
from paper_research.generation.required_claim_output import RequiredClaimValidationError
from paper_research.generation.schema_reliability import (
    DEV_V3_5_CANDIDATE_PROMPT_VERSION,
    PAYLOAD_V3_ADAPTER,
    bind_local_envelope_v3,
    derive_slot_status_v1,
    dev_v3_5_candidate_system_prompt,
    validate_payload_v3,
)
from scripts.payload_contract_v3_lib import (
    build_protocol,
    project_raw_payload_v3,
)
from scripts.replay_dev_v3_4_payload_contract_v3 import (
    accounting_gate,
    build_replay,
    fixture_payload,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/evaluation"
RUN_ROOT = DATA / "evidence-qa-dev-v3-4/runs"


def _answerable_slot() -> dict:
    return {
        "required_claim_id": "c1",
        "claim_text": "A grounded claim.",
        "omission_reason": None,
    }


def _answerable_payload() -> dict:
    return {"answerable": True, "required_claim_results": [_answerable_slot()]}


def _unanswerable_payload() -> dict:
    return {
        "answerable": False,
        "required_claim_results": [],
        "refusal_reason": "The evidence does not address the question.",
    }


def _validate(body: dict, expected: list[str]):
    return validate_payload_v3(json.dumps(body), expected_claim_ids=expected)


def test_valid_discriminated_answerable_and_unanswerable_branches() -> None:
    answerable = _validate(_answerable_payload(), ["c1"])
    unanswerable = _validate(_unanswerable_payload(), [])
    assert answerable.answerable is True
    assert not hasattr(answerable, "refusal_reason")
    assert unanswerable.answerable is False
    assert unanswerable.refusal_reason


@pytest.mark.parametrize(
    "body",
    [
        {**_answerable_payload(), "refusal_reason": None},
        {"answerable": False, "required_claim_results": []},
        {
            "answerable": False,
            "required_claim_results": [],
            "refusal_reason": None,
        },
        {
            "answerable": False,
            "required_claim_results": [],
            "refusal_reason": "",
        },
        {
            "answerable": False,
            "required_claim_results": [_answerable_slot()],
            "refusal_reason": "No.",
        },
        {"answerable": True, "required_claim_results": []},
        {**_answerable_payload(), "extra": 1},
    ],
)
def test_invalid_top_level_branch_shapes_fail(body: dict) -> None:
    expected = ["c1"] if body.get("answerable") is True else []
    with pytest.raises(RequiredClaimValidationError) as exc:
        _validate(body, expected)
    assert exc.value.code == "branch_schema_failed"


def test_slot_content_shapes_derive_unique_local_status() -> None:
    answered = derive_slot_status_v1(_answerable_slot())
    unsupported = derive_slot_status_v1(
        {
            "required_claim_id": "c1",
            "claim_text": None,
            "omission_reason": "The evidence is insufficient.",
        }
    )
    assert answered.derived_status == "answered"
    assert answered.derivation_rule == "nonempty_claim_and_null_omission"
    assert unsupported.derived_status == "unsupported"
    assert unsupported.derivation_rule == "null_claim_and_nonempty_omission"
    assert answered.changed_semantic_fields == 0
    assert answered.added_local_metadata_only is True


@pytest.mark.parametrize(
    "slot",
    [
        {
            "required_claim_id": "c1",
            "claim_text": "Claim.",
            "omission_reason": "Conflict.",
        },
        {
            "required_claim_id": "c1",
            "claim_text": None,
            "omission_reason": None,
        },
        {
            "required_claim_id": "c1",
            "claim_text": "",
            "omission_reason": None,
        },
        {"required_claim_id": "c1", "omission_reason": None},
        {"required_claim_id": "c1", "claim_text": "Claim."},
        {
            "required_claim_id": "c1",
            "claim_text": 1,
            "omission_reason": None,
        },
        {
            "required_claim_id": "c1",
            "claim_text": "Claim.",
            "omission_reason": None,
            "extra": 1,
        },
        {
            "required_claim_id": "c1",
            "status": "supported",
            "claim_text": "Claim.",
            "omission_reason": None,
        },
    ],
)
def test_invalid_slot_content_shapes_fail_without_repair(slot: dict) -> None:
    with pytest.raises(RequiredClaimValidationError) as exc:
        derive_slot_status_v1(slot)
    assert exc.value.code == "slot_shape_failed"


def test_missing_duplicate_and_extra_required_claim_ids_fail() -> None:
    valid = _answerable_payload()
    duplicate = {
        **valid,
        "required_claim_results": [
            valid["required_claim_results"][0],
            valid["required_claim_results"][0],
        ],
    }
    with pytest.raises(RequiredClaimValidationError) as exc:
        _validate(duplicate, ["c1"])
    assert exc.value.code == "duplicate_required_claim_id"
    with pytest.raises(RequiredClaimValidationError) as exc:
        _validate(valid, ["c1", "c2"])
    assert exc.value.code == "missing_required_claim_id"
    with pytest.raises(RequiredClaimValidationError) as exc:
        _validate(valid, [])
    assert exc.value.code == "extra_required_claim_id"


def test_malformed_json_strictly_fails() -> None:
    with pytest.raises(RequiredClaimValidationError) as exc:
        validate_payload_v3("{", expected_claim_ids=[])
    assert exc.value.code == "malformed_json"


def test_local_envelope_only_adds_protocol_metadata_and_status() -> None:
    payload = _validate(_answerable_payload(), ["c1"])
    envelope = bind_local_envelope_v3(payload, question_id="q")
    slot = envelope.required_claim_results[0]
    assert slot.claim_text == payload.required_claim_results[0].claim_text
    assert slot.omission_reason is payload.required_claim_results[0].omission_reason
    assert slot.status == "answered"
    assert slot.citation_ids == []
    assert envelope.prompt_version == DEV_V3_5_CANDIDATE_PROMPT_VERSION


def test_prompt_and_schema_remove_model_status_namespace() -> None:
    prompt = dev_v3_5_candidate_system_prompt().lower()
    for token in (
        "status",
        "answered",
        "supported",
        "unsupported",
        "citation_ids",
        "evidence_id",
        "block_id",
        "question_id",
        "gold",
        "human_label",
    ):
        assert token not in prompt
    schema = json.dumps(PAYLOAD_V3_ADAPTER.json_schema()).lower()
    assert '"status"' not in schema
    protocol = build_protocol()
    assert protocol["canonicalization"] == "none"
    assert protocol["model_outputs_status"] is False
    assert protocol["model_outputs_citation"] is False
    assert protocol["next_live_authorized"] is False


def test_projection_only_removes_deprecated_fields() -> None:
    raw = {
        "answerable": True,
        "required_claim_results": [
            {
                "required_claim_id": "c1",
                "status": "supported",
                "claim_text": "Claim.",
                "omission_reason": "",
            }
        ],
        "refusal_reason": "",
    }
    projection = project_raw_payload_v3(raw)
    projected = projection["projected_payload"]
    assert projection["semantic_modifications"] == 0
    assert len(projection["operations"]) == 2
    assert "status" not in projected["required_claim_results"][0]
    assert "refusal_reason" not in projected
    assert projected["required_claim_results"][0]["claim_text"] == "Claim."
    assert projected["required_claim_results"][0]["omission_reason"] == ""
    assert projected["answerable"] is True
    assert len(projected["required_claim_results"]) == 1
    assert projection["operations"][0]["operation"] == "remove_deprecated_field"


def test_projection_q001_is_generic_but_content_shape_remains_invalid() -> None:
    run_dir = next(RUN_ROOT.glob("live-dev-v3-4-q001-*"))
    raw = json.loads((run_dir / "raw-model-payload.json").read_text(encoding="utf-8"))
    projection = project_raw_payload_v3(raw)
    assert projection["projectable"] is True
    assert len(projection["operations"]) == 3
    with pytest.raises(RequiredClaimValidationError) as exc:
        validate_payload_v3(
            json.dumps(projection["projected_payload"]),
            expected_claim_ids=[
                row["required_claim_id"]
                for row in raw["required_claim_results"]
            ],
        )
    assert exc.value.code == "branch_schema_failed"


def test_projection_q005_preserves_refusal_and_is_valid() -> None:
    run_dir = next(RUN_ROOT.glob("live-dev-v3-4-q005-*"))
    raw = json.loads((run_dir / "raw-model-payload.json").read_text(encoding="utf-8"))
    projection = project_raw_payload_v3(raw)
    assert projection["operations"] == []
    assert projection["projected_payload"]["refusal_reason"] == raw["refusal_reason"]
    assert _validate(projection["projected_payload"], []).answerable is False


def test_projection_unknown_extra_field_is_blocked() -> None:
    raw = {**_answerable_payload(), "unknown": 1}
    projection = project_raw_payload_v3(raw)
    assert projection["projectable"] is False
    assert projection["failure"] == "unknown_top_level_extra_field"


def test_fixture_replay_covers_all_slots_and_is_deterministic() -> None:
    first = build_replay()
    second = build_replay()
    assert first["replay_hash"] == second["replay_hash"]
    assert first["historical_strict_dev_v3_4"] == {
        "structural_pass": 1,
        "final_slots": 0,
        "gate": "FAILED",
    }
    assert first["diagnostic_projection"]["projectable_questions"] == 1
    assert first["diagnostic_projection"]["unprojectable_questions"] == [
        "q001",
        "q002",
        "q004",
        "q007",
        "q008",
        "q013",
        "q015",
        "q019",
        "q050",
    ]
    assert first["diagnostic_projection"]["semantic_field_changes"] == 0
    assert first["fixture_layer"]["payload_schema_success"] == 10
    assert first["fixture_layer"]["slot_derivation_success"] == 27
    assert first["fixture_layer"]["envelope_binding_success"] == 10
    assert first["fixture_layer"]["final_policy_success"] == 10
    assert first["fixture_layer"]["final_slot_success"] == 27
    assert first["fixture_layer"]["derivation_deterministic"] is True
    assert fixture_payload("q005")["refusal_reason"]


@pytest.mark.parametrize(
    "terminal",
    [
        RequestTerminalState.SCHEMA_FAILED,
        RequestTerminalState.POLICY_FAILED,
        RequestTerminalState.MALFORMED_JSON,
    ],
)
def test_candidate_failure_accounting_settles_and_is_idempotent(
    terminal: RequestTerminalState,
) -> None:
    kwargs = {
        "reservation_id": "r",
        "request_id": "req",
        "reserved_tokens": 100,
        "terminal_state": terminal,
        "provider_usage": {"total_tokens": 20, "usage_source": "provider_reported"},
        "request_sent": True,
    }
    events, close = close_reservation_for_terminal_run([], **kwargs)
    repeated, repeated_close = close_reservation_for_terminal_run(events, **kwargs)
    assert events[-1]["event"] == "reservation_settled"
    assert close["effective_active_tokens"] == 0
    assert repeated == events
    assert repeated_close["effective_active_tokens"] == 0


def test_all_candidate_terminal_accounting_is_closed() -> None:
    accounting = accounting_gate()
    assert accounting["effective_active_reservations"] == 0
    assert accounting["double_settlement"] == 0


def test_stage13_16_formal_and_raw_evidence_remain_immutable() -> None:
    freeze = json.loads(
        (DATA / "stage13-16-dev-v3-4-failure-freeze-v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert hashlib.sha256(
        (DATA / "evidence-qa-dev-v3-4.json").read_bytes()
    ).hexdigest() == freeze["summary_sha256"]
    assert hashlib.sha256(
        (DATA / "evidence-qa-dev-v3-4-final-audit.json").read_bytes()
    ).hexdigest() == freeze["final_audit_sha256"]
    assert freeze["gate_results"]["engineering"] == "FAILED"
    for row in freeze["runs"]:
        run_dir = RUN_ROOT / row["run_id"]
        assert hashlib.sha256(
            (run_dir / "raw-provider-response.json").read_bytes()
        ).hexdigest() == row["raw_response_sha256"]
        assert hashlib.sha256(
            (run_dir / "final-result.json").read_bytes()
        ).hexdigest() == row["final_result_sha256"]
