from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from paper_research.evaluation.canonical_hash import verify_legacy_raw_hash
from paper_research.evaluation.request_accounting import (
    RequestTerminalState,
    close_reservation_for_terminal_run,
)
from paper_research.generation.required_claim_output import RequiredClaimValidationError
from paper_research.generation.schema_reliability import (
    PAYLOAD_V4_ADAPTER,
    bind_local_envelope_v4,
    derive_slot_status_v2,
    dev_v3_6_candidate_system_prompt,
    validate_payload_v4,
)
from scripts.payload_contract_v4_lib import (
    PLACEHOLDER_REMOVAL_VERSION,
    build_preflight,
    build_protocol,
    project_raw_payload_v4,
)
from scripts.replay_dev_v3_4_payload_contract_v4 import (
    accounting_gate,
    build_replay,
    fixture_payload,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/evaluation"
RUN_ROOT = DATA / "evidence-qa-dev-v3-4/runs"


def _answered_slot() -> dict:
    return {"required_claim_id": "c1", "claim_text": "A grounded claim."}


def _unsupported_slot() -> dict:
    return {
        "required_claim_id": "c1",
        "omission_reason": "The evidence does not establish the claim.",
    }


def _answerable_payload(slot: dict | None = None) -> dict:
    return {
        "answerable": True,
        "required_claim_results": [slot or _answered_slot()],
    }


def _unanswerable_payload() -> dict:
    return {
        "answerable": False,
        "required_claim_results": [],
        "refusal_reason": "The evidence does not address the question.",
    }


def _validate(body: dict, expected: list[str]):
    return validate_payload_v4(json.dumps(body), expected_claim_ids=expected)


def test_valid_answerable_and_unanswerable_top_level_branches() -> None:
    assert _validate(_answerable_payload(), ["c1"]).answerable is True
    refusal = _validate(_unanswerable_payload(), [])
    assert refusal.answerable is False
    assert refusal.refusal_reason


@pytest.mark.parametrize(
    "body",
    [
        {**_answerable_payload(), "refusal_reason": ""},
        {"answerable": False, "required_claim_results": []},
        {
            "answerable": False,
            "required_claim_results": [_answered_slot()],
            "refusal_reason": "No.",
        },
        {"answerable": True, "required_claim_results": []},
        {**_answerable_payload(), "extra": 1},
    ],
)
def test_invalid_top_level_shapes_fail(body: dict) -> None:
    expected = ["c1"] if body.get("answerable") is True else []
    with pytest.raises(RequiredClaimValidationError) as exc:
        _validate(body, expected)
    assert exc.value.code == "top_level_or_slot_shape_failed"


def test_valid_answered_and_unsupported_slot_shapes() -> None:
    answered = _validate(_answerable_payload(_answered_slot()), ["c1"])
    unsupported = _validate(_answerable_payload(_unsupported_slot()), ["c1"])
    assert answered.required_claim_results[0].claim_text
    assert unsupported.required_claim_results[0].omission_reason


@pytest.mark.parametrize(
    "extra",
    [
        {"omission_reason": None},
        {"omission_reason": ""},
        {"omission_reason": "Conflicting reason."},
        {"status": "answered"},
        {"citation_ids": ["E1"]},
        {"extra": 1},
    ],
)
def test_answered_shape_forbids_every_non_shape_field(extra: dict) -> None:
    slot = {**_answered_slot(), **extra}
    with pytest.raises(RequiredClaimValidationError) as exc:
        _validate(_answerable_payload(slot), ["c1"])
    assert exc.value.code == "top_level_or_slot_shape_failed"


@pytest.mark.parametrize("claim", ["", " ", "\n", None, 1])
def test_answered_shape_requires_nonempty_string(claim: object) -> None:
    slot = {"required_claim_id": "c1", "claim_text": claim}
    with pytest.raises(RequiredClaimValidationError) as exc:
        _validate(_answerable_payload(slot), ["c1"])
    assert exc.value.code == "top_level_or_slot_shape_failed"


@pytest.mark.parametrize(
    "extra",
    [
        {"claim_text": None},
        {"claim_text": ""},
        {"claim_text": "Conflicting claim."},
        {"status": "unsupported"},
        {"citation_ids": ["E1"]},
        {"extra": 1},
    ],
)
def test_unsupported_shape_forbids_every_non_shape_field(extra: dict) -> None:
    slot = {**_unsupported_slot(), **extra}
    with pytest.raises(RequiredClaimValidationError) as exc:
        _validate(_answerable_payload(slot), ["c1"])
    assert exc.value.code == "top_level_or_slot_shape_failed"


@pytest.mark.parametrize("reason", ["", " ", "\n", None, 1])
def test_unsupported_shape_requires_nonempty_string(reason: object) -> None:
    slot = {"required_claim_id": "c1", "omission_reason": reason}
    with pytest.raises(RequiredClaimValidationError) as exc:
        _validate(_answerable_payload(slot), ["c1"])
    assert exc.value.code == "top_level_or_slot_shape_failed"


def test_both_or_neither_content_fields_fail() -> None:
    for slot in (
        {
            "required_claim_id": "c1",
            "claim_text": "Claim.",
            "omission_reason": "Reason.",
        },
        {"required_claim_id": "c1"},
    ):
        with pytest.raises(RequiredClaimValidationError):
            _validate(_answerable_payload(slot), ["c1"])


def test_required_claim_cardinality_remains_strict() -> None:
    duplicate = _answerable_payload()
    duplicate["required_claim_results"].append(_answered_slot())
    with pytest.raises(RequiredClaimValidationError) as exc:
        _validate(duplicate, ["c1"])
    assert exc.value.code == "duplicate_required_claim_id"
    with pytest.raises(RequiredClaimValidationError) as exc:
        _validate(_answerable_payload(), ["c1", "c2"])
    assert exc.value.code == "missing_required_claim_id"
    with pytest.raises(RequiredClaimValidationError) as exc:
        _validate(_answerable_payload(), [])
    assert exc.value.code == "extra_required_claim_id"


def test_status_derivation_v2_is_unique_and_non_mutating() -> None:
    answered_source = _answered_slot()
    unsupported_source = _unsupported_slot()
    answered_copy = dict(answered_source)
    unsupported_copy = dict(unsupported_source)
    answered = derive_slot_status_v2(answered_source)
    unsupported = derive_slot_status_v2(unsupported_source)
    assert answered.detected_shape == "claim_text_only"
    assert answered.derived_status == "answered"
    assert unsupported.detected_shape == "omission_reason_only"
    assert unsupported.derived_status == "unsupported"
    assert answered.ambiguity is False
    assert answered.model_field_changes == 0
    assert answered.semantic_field_changes == 0
    assert answered_source == answered_copy
    assert unsupported_source == unsupported_copy


def test_local_envelope_adds_only_local_metadata() -> None:
    payload = _validate(_answerable_payload(), ["c1"])
    envelope = bind_local_envelope_v4(payload, question_id="q")
    slot = envelope.required_claim_results[0]
    assert slot.claim_text == "A grounded claim."
    assert slot.omission_reason is None
    assert slot.status == "answered"
    assert slot.citation_ids == []


def test_prompt_schema_and_protocol_remove_status_null_and_empty_sentinels() -> None:
    prompt = dev_v3_6_candidate_system_prompt().lower()
    for token in (
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
    schema = json.dumps(PAYLOAD_V4_ADAPTER.json_schema()).lower()
    assert '"status"' not in schema
    assert '"citation_ids"' not in schema
    protocol = build_protocol()
    assert protocol["canonicalization"] == "none"
    assert protocol["null_sentinel_required"] is False
    assert protocol["empty_sentinel_required"] is False
    assert protocol["model_outputs_status"] is False
    assert protocol["model_outputs_citation"] is False
    assert protocol["next_live_authorized"] is False


def test_preflight_inputs_are_stable_and_immutable() -> None:
    first = build_preflight()
    second = build_preflight()
    assert first == second
    assert first["immutable"] is True
    assert first["stage13_17_head"] == "1cbeb094a4c3a2466cb4c9079fb328114e670d4f"


def test_projection_removes_only_deprecated_and_empty_opposite_fields() -> None:
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
    projection = project_raw_payload_v4(raw)
    projected = projection["projected_payload"]
    assert projection["field_projection_completed"] is True
    assert projection["placeholder_fields_removed"] == 1
    assert projection["semantic_field_modifications"] == 0
    assert len(projection["operations"]) == 3
    assert all(
        row["version"] == PLACEHOLDER_REMOVAL_VERSION
        for row in projection["operations"]
    )
    assert projected == {
        "answerable": True,
        "required_claim_results": [
            {"required_claim_id": "c1", "claim_text": "Claim."}
        ],
    }
    assert "null" not in json.dumps(projected)


def test_projection_can_remove_exact_empty_claim_placeholder() -> None:
    raw = {
        "answerable": True,
        "required_claim_results": [
            {
                "required_claim_id": "c1",
                "status": "unsupported",
                "claim_text": "",
                "omission_reason": "Evidence is insufficient.",
            }
        ],
    }
    projection = project_raw_payload_v4(raw)
    assert projection["projected_payload"]["required_claim_results"] == [
        {
            "required_claim_id": "c1",
            "omission_reason": "Evidence is insufficient.",
        }
    ]


def test_projection_preserves_nonempty_semantic_values() -> None:
    raw = {
        "answerable": True,
        "required_claim_results": [
            {
                "required_claim_id": "c1",
                "status": "supported",
                "claim_text": "Exact claim text.",
                "omission_reason": "",
            }
        ],
    }
    projected = project_raw_payload_v4(raw)["projected_payload"]
    assert projected["required_claim_results"][0]["claim_text"] == "Exact claim text."


def test_projection_preserves_semantic_conflicts_as_failures() -> None:
    raw = {
        "answerable": True,
        "required_claim_results": [
            {
                "required_claim_id": "c1",
                "status": "supported",
                "claim_text": "Claim.",
                "omission_reason": "Conflicting reason.",
            }
        ],
    }
    projection = project_raw_payload_v4(raw)
    assert projection["field_projection_completed"] is False
    assert projection["failure"] == "unprojectable_semantic_conflict"
    assert projection["semantic_conflict_count"] == 1


@pytest.mark.parametrize(
    "claim,omission",
    [
        ("", ""),
        (None, "Reason."),
        ("Claim.", None),
    ],
)
def test_projection_does_not_remove_dual_empty_or_null_sentinels(
    claim: object, omission: object
) -> None:
    raw = {
        "answerable": True,
        "required_claim_results": [
            {
                "required_claim_id": "c1",
                "status": "supported",
                "claim_text": claim,
                "omission_reason": omission,
            }
        ],
    }
    projection = project_raw_payload_v4(raw)
    assert projection["field_projection_completed"] is False


def test_projection_does_not_replace_status_or_change_cardinality() -> None:
    raw = {
        "answerable": True,
        "required_claim_results": [
            {
                "required_claim_id": "c1",
                "status": "answerable",
                "claim_text": "Claim.",
                "omission_reason": "",
            }
        ],
    }
    projection = project_raw_payload_v4(raw)
    assert projection["operations"][0]["old_value"] == "answerable"
    assert "status" not in projection["projected_payload"]["required_claim_results"][0]
    assert len(projection["projected_payload"]["required_claim_results"]) == 1
    assert projection["answerability_modifications"] == 0
    assert projection["slot_count_modifications"] == 0


def test_projection_q005_unchanged_and_projection_is_idempotent() -> None:
    run_dir = next(RUN_ROOT.glob("live-dev-v3-4-q005-*"))
    raw = json.loads((run_dir / "raw-model-payload.json").read_text(encoding="utf-8"))
    first = project_raw_payload_v4(raw)
    second = project_raw_payload_v4(first["projected_payload"])
    assert first["operations"] == []
    assert first["projected_payload"] == raw
    assert second["operations"] == []
    assert second["projected_payload"] == raw


def test_historical_replay_improves_projection_without_rewriting_history() -> None:
    first = build_replay()
    second = build_replay()
    assert first["replay_hash"] == second["replay_hash"]
    assert first["historical_strict_dev_v3_4"] == {
        "structural_success": 1,
        "final_slots": 0,
        "gate": "FAILED",
    }
    assert first["payload_v3_diagnostic"]["projectable_questions"] == 1
    assert first["payload_v4_diagnostic"]["projectable_questions"] == 9
    assert first["payload_v4_diagnostic"]["unprojectable_questions"] == ["q015"]
    assert first["payload_v4_diagnostic"]["placeholder_fields_removed"] == 24
    assert first["payload_v4_diagnostic"]["semantic_conflicts"] == 3
    assert first["payload_v4_diagnostic"]["projected_payload_schema_success"] == 9
    assert first["payload_v4_diagnostic"]["slot_shape_success"] == 24
    assert first["payload_v4_diagnostic"]["envelope_binding_success"] == 9
    assert first["payload_v4_diagnostic"]["final_policy_success"] == 9
    assert first["payload_v4_diagnostic"]["final_slot_success"] == 24
    assert first["payload_v4_diagnostic"]["semantic_field_changes"] == 0
    assert first["fixture_layer"]["payload_schema_success"] == 10
    assert first["fixture_layer"]["slot_shape_success"] == 27
    assert first["fixture_layer"]["final_slot_success"] == 27
    assert fixture_payload("q005")["refusal_reason"]


@pytest.mark.parametrize(
    "terminal",
    [
        RequestTerminalState.MALFORMED_JSON,
        RequestTerminalState.SCHEMA_FAILED,
        RequestTerminalState.POLICY_FAILED,
    ],
)
def test_shape_and_policy_failures_settle_idempotently(
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


def test_all_v4_accounting_paths_close() -> None:
    accounting = accounting_gate()
    assert accounting["effective_active_reservations"] == 0
    assert accounting["double_settlement"] == 0


def test_stage13_16_and_stage13_17_inputs_remain_immutable() -> None:
    preflight = build_preflight()
    freeze = json.loads(
        (DATA / "stage13-16-dev-v3-4-failure-freeze-v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert verify_legacy_raw_hash(
        DATA / "evidence-qa-dev-v3-4.json", freeze["summary_sha256"]
    )
    assert verify_legacy_raw_hash(
        DATA / "evidence-qa-dev-v3-4-final-audit.json",
        freeze["final_audit_sha256"],
    )
    assert freeze["gate_results"]["engineering"] == "FAILED"
    for row in freeze["runs"]:
        run_dir = RUN_ROOT / row["run_id"]
        assert hashlib.sha256(
            (run_dir / "raw-provider-response.json").read_bytes()
        ).hexdigest() == row["raw_response_sha256"]
    assert verify_legacy_raw_hash(
        DATA / "dev-v3-4-payload-contract-v3-replay.json",
        preflight["inputs"]["stage13_17_replay"]["sha256"],
    )
    assert verify_legacy_raw_hash(
        DATA / "slot-status-derivation-safety-audit-v1.json",
        preflight["inputs"]["stage13_17_safety"]["sha256"],
    )
