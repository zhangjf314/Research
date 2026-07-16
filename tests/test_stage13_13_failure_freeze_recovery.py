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
    bind_local_envelope,
    parse_minimal_payload,
    schema_reliability_system_prompt,
)
from scripts.run_evidence_qa_dev_v3_2 import exact_delivered_messages_hash

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/evaluation"


def load(name: str) -> dict:
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def test_exact_delivered_hash_is_order_stable_and_content_sensitive() -> None:
    messages = [{"role": "system", "content": "a"}, {"role": "user", "content": "b"}]
    assert exact_delivered_messages_hash(messages) == exact_delivered_messages_hash(messages)
    assert exact_delivered_messages_hash(messages) != exact_delivered_messages_hash(
        [{"role": "system", "content": "a"}, {"role": "user", "content": "c"}]
    )


def test_candidate_prompt_has_no_protocol_constant_or_copyable_evidence_id() -> None:
    prompt = schema_reliability_system_prompt()
    assert "qa-required-claims-citation-id-v3.1" not in prompt
    assert "qa-required-claims-citation-id-v3.2" not in prompt
    assert "evidence_id" not in prompt
    assert "block_id" not in prompt


@pytest.mark.parametrize(
    ("raw", "code"),
    [
        ("{", "malformed_json"),
        ("```json\n{}\n```", "malformed_json"),
        ('{"answerable":true} trailing', "malformed_json"),
        ('{"answerable":"\\q"}', "malformed_json"),
    ],
)
def test_candidate_never_repairs_malformed_json(raw: str, code: str) -> None:
    with pytest.raises(RequiredClaimValidationError) as exc:
        parse_minimal_payload(raw, expected_claim_ids=[])
    assert exc.value.code == code


def test_candidate_rejects_missing_duplicate_and_extra_slots() -> None:
    base = {
        "answerable": True,
        "required_claim_results": [
            {
                "required_claim_id": "c1",
                "status": "answered",
                "claim_text": "x",
                "omission_reason": None,
            }
        ],
        "refusal_reason": None,
    }
    cases = [
        ({**base, "required_claim_results": []}, "missing_required_claim_id"),
        (
            {
                **base,
                "required_claim_results": [
                    *base["required_claim_results"],
                    *base["required_claim_results"],
                ],
            },
            "duplicate_required_claim_id",
        ),
        (base, "extra_required_claim_id"),
    ]
    for body, code in cases:
        expected = ["c1"] if code != "extra_required_claim_id" else []
        with pytest.raises(RequiredClaimValidationError) as exc:
            parse_minimal_payload(json.dumps(body), expected_claim_ids=expected)
        assert exc.value.code == code


def test_local_envelope_owns_constants_and_citations() -> None:
    raw = {
        "answerable": True,
        "required_claim_results": [
            {
                "required_claim_id": "c1",
                "status": "answered",
                "claim_text": "supported",
                "omission_reason": None,
            }
        ],
        "refusal_reason": None,
    }
    parsed = parse_minimal_payload(json.dumps(raw), expected_claim_ids=["c1"])
    envelope = bind_local_envelope(
        parsed, question_id="q", citation_ids_by_claim={"c1": ["E001", "E002"]}
    )
    assert envelope.prompt_version == "schema-reliability-v1-candidate"
    assert envelope.required_claim_results[0].citation_ids == ["E001", "E002"]


@pytest.mark.parametrize(
    "state",
    [
        RequestTerminalState.COMPLETED,
        RequestTerminalState.MALFORMED_JSON,
        RequestTerminalState.SCHEMA_FAILED,
        RequestTerminalState.CITATION_VALIDATION_FAILED,
        RequestTerminalState.POLICY_FAILED,
    ],
)
def test_provider_usage_settles_every_downstream_terminal_state(
    state: RequestTerminalState,
) -> None:
    events, result = close_reservation_for_terminal_run(
        [],
        reservation_id="r",
        request_id="req",
        reserved_tokens=100,
        terminal_state=state,
        provider_usage={"total_tokens": 42, "usage_source": "provider_reported"},
        request_sent=True,
    )
    assert events[-1]["event"] == "reservation_settled"
    assert events[-1]["settled_tokens"] == 42
    assert result["effective_active_tokens"] == 0


def test_unsent_provider_failure_releases_and_timeout_is_terminal_unknown() -> None:
    released, _ = close_reservation_for_terminal_run(
        [],
        reservation_id="release",
        request_id="req",
        reserved_tokens=100,
        terminal_state=RequestTerminalState.PROVIDER_FAILED,
        provider_usage=None,
        request_sent=False,
    )
    unknown, result = close_reservation_for_terminal_run(
        [],
        reservation_id="timeout",
        request_id="req2",
        reserved_tokens=100,
        terminal_state=RequestTerminalState.TIMED_OUT,
        provider_usage=None,
        request_sent=True,
    )
    assert released[-1]["event"] == "reservation_released"
    assert unknown[-1]["event"] == "billing_unknown_terminal"
    assert result["effective_active_tokens"] == 0


def test_terminal_closure_is_idempotent_and_rejects_duplicate_settlement() -> None:
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
    with pytest.raises(ValueError):
        close_reservation_for_terminal_run([first[-1], dict(first[-1])], **kwargs)


def test_historical_gate_and_formal_files_remain_frozen() -> None:
    summary = load("evidence-qa-dev-v3-2.json")
    final_audit = load("evidence-qa-dev-v3-2-final-audit.json")
    freeze = load("stage13-12-dev-v3-2-failure-freeze-v1.json")
    reconciliation = load("stage13-12-reservation-reconciliation-v1.json")
    assert summary["all_manifest_conservative"]["active_reserved_tokens"] == 96000
    assert final_audit["dev_v3_2_engineering_gate"] == "FAILED"
    assert freeze["freeze_signature"] == (
        "1b24de0ac01829477b5dbc1c00a5349116f099004dea911851095f5110a6ca2f"
    )
    assert reconciliation["historical_ledgers_modified"] is False
    assert reconciliation["effective_active_reservations"] == 0


def test_readiness_is_offline_and_never_authorizes_live() -> None:
    readiness = load("schema-reliability-v1-readiness.json")
    assert readiness["schema_reliability_vnext_ready"] is True
    assert readiness["next_live_authorized"] is False
    assert readiness["ready_for_full_qa"] is False
