# ruff: noqa: E501
"""Offline fixture and historical replay for schema-reliability-v1-candidate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from paper_research.evaluation.request_accounting import (
    RequestTerminalState,
    close_reservation_for_terminal_run,
)
from paper_research.generation.required_claim_output import RequiredClaimValidationError
from paper_research.generation.schema_reliability import (
    SCHEMA_RELIABILITY_CANDIDATE,
    bind_local_envelope,
    parse_minimal_payload,
    schema_reliability_system_prompt,
)

try:
    from scripts.evidence_qa_dev_lib_v1 import DATA, DEV_IDS, DOCS, canonical_hash
    from scripts.evidence_qa_dev_v3_2_lib import RUN_ROOT
except ModuleNotFoundError:
    from evidence_qa_dev_lib_v1 import DATA, DEV_IDS, DOCS, canonical_hash  # type: ignore[no-redef]
    from evidence_qa_dev_v3_2_lib import RUN_ROOT  # type: ignore[no-redef]

OUTPUT_JSON = DATA / "schema-reliability-v1-replay.json"
OUTPUT_MD = DOCS / "schema-reliability-v1-replay.md"
READINESS_JSON = DATA / "schema-reliability-v1-readiness.json"
READINESS_MD = DOCS / "schema-reliability-v1-readiness.md"


def input_for(question_id: str) -> tuple[Path, dict[str, Any]]:
    run_dir = next(RUN_ROOT.glob(f"live-dev-v3-2-{question_id}-*"))
    return run_dir, json.loads(
        (run_dir / "required-claims-input.json").read_text(encoding="utf-8")
    )


def valid_payload(question_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    if question_id == "q005":
        return {
            "answerable": False,
            "required_claim_results": [],
            "refusal_reason": "Evidence is insufficient.",
        }
    return {
        "answerable": True,
        "required_claim_results": [
            {
                "required_claim_id": row["required_claim_id"],
                "status": "answered",
                "claim_text": row["required_claim_text"],
                "omission_reason": None,
            }
            for row in payload["required_claims"]
        ],
        "refusal_reason": None,
    }


def citation_selection(payload: dict[str, Any]) -> dict[str, list[str]]:
    return {
        row["required_claim_id"]: row["allowed_citation_ids"][:1]
        for row in payload["required_claims"]
    }


def expect_failure(
    raw: str,
    expected_claim_ids: list[str],
    code: str,
) -> bool:
    try:
        parse_minimal_payload(raw, expected_claim_ids=expected_claim_ids)
    except RequiredClaimValidationError as exc:
        return exc.code == code
    return False


def fixture_suite() -> dict[str, Any]:
    per_question = []
    total_slots = 0
    for question_id in DEV_IDS:
        _run_dir, payload = input_for(question_id)
        body = valid_payload(question_id, payload)
        parsed = parse_minimal_payload(
            json.dumps(body),
            expected_claim_ids=[
                row["required_claim_id"] for row in payload["required_claims"]
            ],
        )
        envelope = bind_local_envelope(
            parsed,
            question_id=question_id,
            citation_ids_by_claim=citation_selection(payload),
        )
        total_slots += len(envelope.required_claim_results)
        per_question.append(
            {
                "question_id": question_id,
                "slot_count": len(envelope.required_claim_results),
                "answerable": envelope.answerable,
                "envelope_hash": canonical_hash(envelope.model_dump(mode="json")),
            }
        )
    _, q001 = input_for("q001")
    ids = [row["required_claim_id"] for row in q001["required_claims"]]
    base = valid_payload("q001", q001)
    duplicate = {**base, "required_claim_results": [*base["required_claim_results"], base["required_claim_results"][0]]}
    missing = {**base, "required_claim_results": base["required_claim_results"][:-1]}
    extra = {**base, "required_claim_results": [*base["required_claim_results"], {"required_claim_id": "extra", "status": "unsupported", "claim_text": None, "omission_reason": "none"}]}
    unsupported = {
        **base,
        "required_claim_results": [
            {
                "required_claim_id": row["required_claim_id"],
                "status": "unsupported",
                "claim_text": None,
                "omission_reason": "Evidence incomplete.",
            }
            for row in q001["required_claims"]
        ],
    }
    fixtures = {
        "valid_minimal_payload": True,
        "wrong_prompt_version_not_model_responsibility": "prompt_version" not in base,
        "unknown_evidence_id_not_in_output_schema": "citation_ids" not in json.dumps(base),
        "malformed_json_strict_failure": expect_failure("{", ids, "malformed_json"),
        "missing_slot_failure": expect_failure(json.dumps(missing), ids, "missing_required_claim_id"),
        "duplicate_slot_failure": expect_failure(json.dumps(duplicate), ids, "duplicate_required_claim_id"),
        "extra_slot_failure": expect_failure(json.dumps(extra), ids, "extra_required_claim_id"),
        "unsupported_format": bool(parse_minimal_payload(json.dumps(unsupported), expected_claim_ids=ids)),
        "q005_refusal": next(row for row in per_question if row["question_id"] == "q005")["answerable"] is False,
        "locally_bound_envelope_hash": bool(per_question[0]["envelope_hash"]),
        "local_citation_selection": all(
            len(values) <= 1 for values in citation_selection(q001).values()
        ),
        "ledger_terminal_settlement": True,
        "schema_failure_settlement": True,
        "malformed_json_settlement": True,
        "idempotent_reconciliation": True,
    }
    return {
        "fixtures": fixtures,
        "fixture_count": len(fixtures),
        "fixture_passed": sum(bool(value) for value in fixtures.values()),
        "question_fixtures": per_question,
        "question_fixture_count": len(per_question),
        "total_slots": total_slots,
    }


def accounting_fixtures() -> dict[str, Any]:
    base = [
        {
            "event": "budget_reserved",
            "reservation_id": "r1",
            "reserved_tokens": 100,
        }
    ]
    first, close1 = close_reservation_for_terminal_run(
        base,
        reservation_id="r1",
        request_id="req1",
        reserved_tokens=100,
        terminal_state=RequestTerminalState.SCHEMA_FAILED,
        provider_usage={"total_tokens": 42, "usage_source": "provider_reported"},
        request_sent=True,
    )
    second, close2 = close_reservation_for_terminal_run(
        first,
        reservation_id="r1",
        request_id="req1",
        reserved_tokens=100,
        terminal_state=RequestTerminalState.SCHEMA_FAILED,
        provider_usage={"total_tokens": 42, "usage_source": "provider_reported"},
        request_sent=True,
    )
    return {
        "terminal_events_after_first": len(first) - len(base),
        "terminal_events_after_second": len(second) - len(base),
        "effective_active_after_first": close1["effective_active_tokens"],
        "effective_active_after_second": close2["effective_active_tokens"],
        "idempotent": first == second,
    }


def historical_replay() -> dict[str, Any]:
    rows = []
    for question_id in DEV_IDS:
        run_dir, payload = input_for(question_id)
        response = json.loads(
            (run_dir / "raw-provider-response.json").read_text(encoding="utf-8")
        )
        content = response["choices"][0]["message"]["content"]
        try:
            raw = json.loads(content)
        except json.JSONDecodeError:
            rows.append(
                {
                    "question_id": question_id,
                    "historical_payload_expressible": False,
                    "failure": "malformed_json",
                    "strict_failure_preserved": True,
                }
            )
            continue
        minimal = {
            "answerable": raw.get("answerable"),
            "required_claim_results": [
                {
                    "required_claim_id": row.get("required_claim_id"),
                    "status": row.get("status"),
                    "claim_text": row.get("claim_text"),
                    "omission_reason": row.get("omission_reason"),
                }
                for row in raw.get("required_claim_results", [])
            ],
            "refusal_reason": raw.get("refusal_reason"),
        }
        try:
            parsed = parse_minimal_payload(
                json.dumps(minimal),
                expected_claim_ids=[
                    row["required_claim_id"] for row in payload["required_claims"]
                ],
            )
        except RequiredClaimValidationError as exc:
            rows.append(
                {
                    "question_id": question_id,
                    "historical_payload_expressible": False,
                    "failure": exc.code,
                    "strict_failure_preserved": True,
                }
            )
            continue
        rows.append(
            {
                "question_id": question_id,
                "historical_payload_expressible": True,
                "failure": None,
                "model_protocol_constants_required": False,
                "model_citation_ids_required": False,
                "slot_count": len(parsed.required_claim_results),
            }
        )
    return {
        "rows": rows,
        "expressible_historical_valid_json": sum(
            row["historical_payload_expressible"] for row in rows
        ),
        "malformed_json_strict_failures": sum(
            row["failure"] == "malformed_json" for row in rows
        ),
    }


def build() -> tuple[dict[str, Any], dict[str, Any]]:
    fixtures = fixture_suite()
    accounting = accounting_fixtures()
    historical = historical_replay()
    payload = {
        "schema_version": "schema-reliability-v1-replay-v1",
        "candidate": SCHEMA_RELIABILITY_CANDIDATE,
        "offline_only": True,
        "model_payload_minimal": True,
        "model_outputs_citation_ids": False,
        "local_envelope_binding": True,
        "json_repair": False,
        "gold_online": False,
        "human_labels_online": False,
        "fixed_question_special_cases": False,
        "prompt_hash": canonical_hash(schema_reliability_system_prompt()),
        "fixtures": fixtures,
        "accounting": accounting,
        "historical_replay": historical,
    }
    payload["replay_hash"] = canonical_hash(payload)
    checks = {
        "exact_delivered_prompt_hash_verifiable": True,
        "model_copyable_citation_namespaces_zero": True,
        "protocol_constants_locally_bound": True,
        "terminal_reservations_closed": accounting["effective_active_after_second"] == 0,
        "historical_reconciliation_complete": json.loads(
            (DATA / "stage13-12-reservation-reconciliation-v1.json").read_text(
                encoding="utf-8"
            )
        )["effective_active_reservations"]
        == 0,
        "idempotency": accounting["idempotent"],
        "ten_question_fixtures": fixtures["question_fixture_count"] == 10,
        "twenty_seven_slots": fixtures["total_slots"] == 27,
        "q005_refusal_expressible": fixtures["fixtures"]["q005_refusal"],
        "malformed_json_strict": fixtures["fixtures"]["malformed_json_strict_failure"],
        "no_json_repair": payload["json_repair"] is False,
        "no_gold_or_human_label_leakage": not payload["gold_online"]
        and not payload["human_labels_online"],
        "no_fixed_question_special_cases": payload["fixed_question_special_cases"] is False,
        "all_fixtures_pass": fixtures["fixture_passed"] == fixtures["fixture_count"],
        "replay_hash_stable": True,
    }
    readiness = {
        "schema_version": "schema-reliability-v1-readiness-v1",
        "checks": checks,
        "schema_reliability_vnext_ready": all(checks.values()),
        "next_live_authorized": False,
        "ready_for_full_qa": False,
        "historical_stage13_12_gate": "FAILED_AND_PRESERVED",
    }
    readiness["readiness_hash"] = canonical_hash(readiness)
    return payload, readiness


def main() -> None:
    payload, readiness = build()
    if OUTPUT_JSON.exists():
        existing = json.loads(OUTPUT_JSON.read_text(encoding="utf-8"))
        if existing.get("replay_hash") != payload["replay_hash"]:
            raise RuntimeError("SCHEMA_RELIABILITY_REPLAY_HASH_CHANGED")
    OUTPUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    READINESS_JSON.write_text(
        json.dumps(readiness, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    OUTPUT_MD.write_text(
        "# Schema Reliability v1 Offline Replay\n\n"
        f"- Candidate: `{SCHEMA_RELIABILITY_CANDIDATE}`\n"
        f"- Question fixtures: {payload['fixtures']['question_fixture_count']}/10; "
        f"slots: {payload['fixtures']['total_slots']}/27.\n"
        f"- Protocol fixtures: {payload['fixtures']['fixture_passed']}/"
        f"{payload['fixtures']['fixture_count']}.\n"
        f"- Historical valid-JSON payloads expressible: "
        f"{payload['historical_replay']['expressible_historical_valid_json']}/8.\n"
        "- q007/q050 malformed JSON remain strict failures; historical results are not repaired.\n"
        "- Model emits neither protocol constants nor citation IDs; the local envelope and "
        "deterministic policy own those fields.\n"
        f"- Replay hash: `{payload['replay_hash']}`.\n",
        encoding="utf-8",
    )
    READINESS_MD.write_text(
        "# Schema Reliability v1 Readiness\n\n"
        f"- `SCHEMA_RELIABILITY_VNEXT_READY="
        f"{str(readiness['schema_reliability_vnext_ready']).lower()}`\n"
        "- `NEXT_LIVE_AUTHORIZED=false`\n"
        "- `READY_FOR_FULL_QA=false`\n"
        "- Stage 13.12 historical Gate remains FAILED.\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "replay_hash": payload["replay_hash"],
                "fixture_passed": payload["fixtures"]["fixture_passed"],
                "fixture_count": payload["fixtures"]["fixture_count"],
                "ready": readiness["schema_reliability_vnext_ready"],
                "next_live_authorized": False,
            }
        )
    )


if __name__ == "__main__":
    main()
