from __future__ import annotations

import json
from pathlib import Path

import pytest

from paper_research.evaluation.request_accounting import (
    RequestTerminalState,
    close_reservation_for_terminal_run,
)
from paper_research.generation.required_claim_output import RequiredClaimValidationError
from paper_research.generation.schema_reliability import (
    REFUSAL_CANONICALIZATION_VERSION,
    canonicalize_model_payload_v2,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/evaluation"


def body(refusal: object, *, answerable: bool = True) -> dict:
    return {
        "answerable": answerable,
        "required_claim_results": (
            [
                {
                    "required_claim_id": "c1",
                    "status": "answered",
                    "claim_text": "supported",
                    "omission_reason": None,
                }
            ]
            if answerable
            else []
        ),
        "refusal_reason": refusal,
    }


def test_answerable_null_is_valid_and_unchanged() -> None:
    result = canonicalize_model_payload_v2(
        json.dumps(body(None)), expected_claim_ids=["c1"]
    )
    assert result.canonicalization_applied is False
    assert result.changed_paths == []
    assert result.raw_payload_hash == result.canonical_payload_hash


def test_answerable_exact_empty_is_the_only_canonicalized_value() -> None:
    result = canonicalize_model_payload_v2(
        json.dumps(body("")), expected_claim_ids=["c1"]
    )
    assert result.canonicalization_applied is True
    assert result.canonicalization_rule == REFUSAL_CANONICALIZATION_VERSION
    assert result.changed_paths == ["$.refusal_reason"]
    assert result.canonical_payload.refusal_reason is None
    assert result.semantic_change is False


@pytest.mark.parametrize("value", [" ", "\n", "N/A", "none", "not applicable", "explanation"])
def test_answerable_other_strings_are_strictly_rejected(value: str) -> None:
    with pytest.raises(RequiredClaimValidationError) as exc:
        canonicalize_model_payload_v2(
            json.dumps(body(value)), expected_claim_ids=["c1"]
        )
    assert exc.value.code == "answerable_has_semantic_refusal_reason"


def test_unanswerable_requires_real_reason_and_is_never_canonicalized() -> None:
    valid = canonicalize_model_payload_v2(
        json.dumps(body("Evidence is insufficient.", answerable=False)),
        expected_claim_ids=[],
    )
    assert valid.canonicalization_applied is False
    for value in ("", None, " \n"):
        with pytest.raises(RequiredClaimValidationError) as exc:
            canonicalize_model_payload_v2(
                json.dumps(body(value, answerable=False)),
                expected_claim_ids=[],
            )
        assert exc.value.code == "unanswerable_missing_refusal_reason"


@pytest.mark.parametrize("value", [1, False, [], {}])
def test_wrong_refusal_type_fails_before_canonicalization(value: object) -> None:
    with pytest.raises(RequiredClaimValidationError) as exc:
        canonicalize_model_payload_v2(
            json.dumps(body(value)), expected_claim_ids=["c1"]
        )
    assert exc.value.code == "schema_validation_failure"


def test_missing_or_extra_top_level_field_fails() -> None:
    missing = body(None)
    missing.pop("refusal_reason")
    extra = {**body(None), "extra": 1}
    for value in (missing, extra):
        with pytest.raises(RequiredClaimValidationError) as exc:
            canonicalize_model_payload_v2(
                json.dumps(value), expected_claim_ids=["c1"]
            )
        assert exc.value.code == "schema_validation_failure"


def test_malformed_json_never_reaches_canonicalization() -> None:
    with pytest.raises(RequiredClaimValidationError) as exc:
        canonicalize_model_payload_v2("{", expected_claim_ids=[])
    assert exc.value.code == "malformed_json"


def test_missing_duplicate_extra_unknown_and_illegal_status_fail() -> None:
    valid = body(None)
    duplicate = {
        **valid,
        "required_claim_results": [
            *valid["required_claim_results"],
            *valid["required_claim_results"],
        ],
    }
    cases = [
        ({**valid, "required_claim_results": []}, ["c1"], "missing_required_claim_id"),
        (duplicate, ["c1"], "duplicate_required_claim_id"),
        (valid, [], "extra_required_claim_id"),
        (valid, ["unknown"], "missing_required_claim_id"),
        (
            {
                **valid,
                "required_claim_results": [
                    {**valid["required_claim_results"][0], "status": "bad"}
                ],
            },
            ["c1"],
            "schema_validation_failure",
        ),
    ]
    for value, expected, code in cases:
        with pytest.raises(RequiredClaimValidationError) as exc:
            canonicalize_model_payload_v2(
                json.dumps(value), expected_claim_ids=expected
            )
        assert exc.value.code == code


def test_canonicalization_is_idempotent_deterministic_and_semantically_safe() -> None:
    first = canonicalize_model_payload_v2(
        json.dumps(body("")), expected_claim_ids=["c1"]
    )
    second = canonicalize_model_payload_v2(
        json.dumps(first.canonical_payload.model_dump(mode="json")),
        expected_claim_ids=["c1"],
    )
    assert second.canonicalization_applied is False
    assert first.canonical_payload_hash == second.canonical_payload_hash
    raw_slot = first.raw_payload["required_claim_results"][0]
    final_slot = first.canonical_payload.required_claim_results[0].model_dump(
        mode="json"
    )
    assert raw_slot == final_slot


def test_stage13_14_replay_is_separate_complete_and_stable() -> None:
    replay = json.loads(
        (DATA / "dev-v3-3-payload-contract-v2-replay.json").read_text(
            encoding="utf-8"
        )
    )
    audit = json.loads(
        (DATA / "dev-v3-3-payload-contract-v2-final-audit.json").read_text(
            encoding="utf-8"
        )
    )
    assert replay["metric_status"] == "diagnostic_new_protocol_replay"
    assert replay["historical_strict_v3_3"] == {
        "payload_pass": 2,
        "slot_pass": 3,
        "gate": "FAILED",
    }
    assert replay["diagnostic_v2_contract"] == {
        "json_valid": 10,
        "structural_schema_success": 10,
        "canonicalization_applied": 8,
        "canonical_payload_success": 10,
        "required_slot_success": 27,
        "envelope_binding_success": 10,
        "final_policy_success": 10,
        "final_slot_success": 27,
    }
    assert replay["semantic_field_changes"] == 0
    assert replay["canonicalization_path_violations"] == 0
    assert replay["q005_changed"] is False
    assert audit["payload_contract_v2_engineering_gate"] == "PASSED"
    assert audit["payload_contract_v2_ready"] is True
    assert audit["next_live_authorized"] is False


def test_stage13_14_formal_result_remains_failed() -> None:
    formal = json.loads(
        (DATA / "evidence-qa-dev-v3-3-final-audit.json").read_text(
            encoding="utf-8"
        )
    )
    freeze = json.loads(
        (DATA / "stage13-14-dev-v3-3-failure-freeze-v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert formal["dev_v3_3_engineering_gate"] == "FAILED"
    assert freeze["gate_results"]["engineering"] == "FAILED"


@pytest.mark.parametrize(
    "terminal",
    [
        RequestTerminalState.COMPLETED,
        RequestTerminalState.SCHEMA_FAILED,
        RequestTerminalState.MALFORMED_JSON,
    ],
)
def test_post_usage_contract_outcomes_settle_and_close(
    terminal: RequestTerminalState,
) -> None:
    events, close = close_reservation_for_terminal_run(
        [],
        reservation_id="r",
        request_id="req",
        reserved_tokens=100,
        terminal_state=terminal,
        provider_usage={"total_tokens": 42, "usage_source": "provider_reported"},
        request_sent=True,
    )
    assert events[-1]["event"] == "reservation_settled"
    assert close["effective_active_tokens"] == 0


def test_accounting_close_is_idempotent_without_double_settlement() -> None:
    kwargs = {
        "reservation_id": "r",
        "request_id": "req",
        "reserved_tokens": 100,
        "terminal_state": RequestTerminalState.SCHEMA_FAILED,
        "provider_usage": {"total_tokens": 42},
        "request_sent": True,
    }
    first, _ = close_reservation_for_terminal_run([], **kwargs)
    second, _ = close_reservation_for_terminal_run(first, **kwargs)
    assert first == second
