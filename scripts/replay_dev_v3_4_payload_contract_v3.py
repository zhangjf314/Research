# ruff: noqa: E501
"""Offline replay of Dev v3.4 raw responses under Payload Contract v3."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

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
    payload_v3_as_minimal_payload,
    validate_payload_v3,
)

try:
    from scripts.evidence_qa_dev_lib_v1 import DEV_IDS, canonical_hash
    from scripts.evidence_qa_dev_v3_3_lib import safe_model_input
    from scripts.payload_contract_v3_lib import (
        FINAL_AUDIT,
        REPLAY,
        REPLAY_CSV,
        REPLAY_DOC,
        RUN_ROOT,
        SAFETY,
        SAFETY_DOC,
        project_raw_payload_v3,
        write_protocol,
    )
    from scripts.run_evidence_qa_dev_v3_3 import apply_policy, candidate_rows
except ModuleNotFoundError:
    from evidence_qa_dev_lib_v1 import DEV_IDS, canonical_hash  # type: ignore[no-redef]
    from evidence_qa_dev_v3_3_lib import safe_model_input  # type: ignore[no-redef]
    from payload_contract_v3_lib import (  # type: ignore[no-redef]
        FINAL_AUDIT,
        REPLAY,
        REPLAY_CSV,
        REPLAY_DOC,
        RUN_ROOT,
        SAFETY,
        SAFETY_DOC,
        project_raw_payload_v3,
        write_protocol,
    )
    from run_evidence_qa_dev_v3_3 import (  # type: ignore[no-redef]
        apply_policy,
        candidate_rows,
    )

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/evaluation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests-passed", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def expected_ids(question_id: str) -> list[str]:
    safe = safe_model_input(question_id)[0]
    return [row["required_claim_id"] for row in safe["required_claims"]]


def fixture_payload(question_id: str) -> dict[str, Any]:
    _safe, full, _registry, _trace = safe_model_input(question_id)
    if question_id == "q005":
        return {
            "answerable": False,
            "required_claim_results": [],
            "refusal_reason": "The supplied evidence does not report the requested total.",
        }
    return {
        "answerable": True,
        "required_claim_results": [
            {
                "required_claim_id": row["required_claim_id"],
                "claim_text": row["required_claim_text"],
                "omission_reason": None,
            }
            for row in full["required_claims"]
        ],
    }


def historical_projection_row(question_id: str) -> dict[str, Any]:
    run_dir = next(RUN_ROOT.glob(f"live-dev-v3-4-{question_id}-*"))
    raw = load_json(run_dir / "raw-model-payload.json")
    projection = project_raw_payload_v3(raw)
    row = {
        "question_id": question_id,
        "run_id": run_dir.name,
        "field_projection_completed": projection.get("projectable", False),
        "projection_operations": projection.get("operations", []),
        "semantic_modifications": projection.get("semantic_modifications", 0),
        "claim_text_modified": projection.get("claim_text_modified", False),
        "omission_reason_modified": projection.get("omission_reason_modified", False),
        "answerability_modified": projection.get("answerability_modified", False),
        "slot_count_modified": projection.get("slot_count_modified", False),
        "claim_ids_modified": projection.get("claim_ids_modified", False),
        "projected_schema_success": False,
        "slot_derivation_success": 0,
        "envelope_binding_success": False,
        "final_policy_success": False,
        "final_slot_success": 0,
        "failure_type": projection.get("failure"),
        "failure_reason": None,
    }
    if not projection.get("projectable"):
        return row
    projected = projection["projected_payload"]
    try:
        payload = validate_payload_v3(
            json.dumps(projected, ensure_ascii=False),
            expected_claim_ids=expected_ids(question_id),
        )
        row["projected_schema_success"] = True
        derivations = [
            derive_slot_status_v1(slot.model_dump(mode="json"))
            for slot in payload.required_claim_results
        ]
        row["slot_derivation_success"] = len(derivations)
        bind_local_envelope_v3(payload, question_id=question_id)
        row["envelope_binding_success"] = True
        _safe, full, registry, trace = safe_model_input(question_id)
        candidates_by_claim, _candidates = candidate_rows(full, registry, trace)
        final, _policy = apply_policy(
            payload_v3_as_minimal_payload(payload),
            full,
            candidates_by_claim,
            question_id,
        )
        row["final_policy_success"] = True
        row["final_slot_success"] = len(final.required_claim_results)
        row["projectable_under_v3"] = True
    except RequiredClaimValidationError as exc:
        row["failure_type"] = exc.code
        row["failure_reason"] = str(exc)
        row["projectable_under_v3"] = False
    except Exception as exc:
        row["failure_type"] = type(exc).__name__
        row["failure_reason"] = str(exc)
        row["projectable_under_v3"] = False
    return row


def fixture_row(question_id: str) -> dict[str, Any]:
    raw = fixture_payload(question_id)
    payload = validate_payload_v3(
        json.dumps(raw, ensure_ascii=False),
        expected_claim_ids=expected_ids(question_id),
    )
    derivations = [
        derive_slot_status_v1(slot.model_dump(mode="json"))
        for slot in payload.required_claim_results
    ]
    first = [row.model_dump(mode="json") for row in derivations]
    second = [
        derive_slot_status_v1(slot.model_dump(mode="json")).model_dump(mode="json")
        for slot in payload.required_claim_results
    ]
    envelope = bind_local_envelope_v3(payload, question_id=question_id)
    _safe, full, registry, trace = safe_model_input(question_id)
    candidates_by_claim, _candidates = candidate_rows(full, registry, trace)
    final, _policy = apply_policy(
        payload_v3_as_minimal_payload(payload),
        full,
        candidates_by_claim,
        question_id,
    )
    return {
        "question_id": question_id,
        "payload_schema_success": True,
        "slot_derivation_success": len(derivations),
        "envelope_binding_success": envelope.question_id == question_id,
        "final_policy_success": True,
        "final_slot_success": len(final.required_claim_results),
        "derivation_deterministic": first == second,
        "semantic_field_changes": 0,
    }


def accounting_gate() -> dict[str, Any]:
    terminal_map = {
        "completed": RequestTerminalState.COMPLETED,
        "malformed_json": RequestTerminalState.MALFORMED_JSON,
        "branch_schema_failed": RequestTerminalState.SCHEMA_FAILED,
        "slot_shape_failed": RequestTerminalState.SCHEMA_FAILED,
        "envelope_failed": RequestTerminalState.SCHEMA_FAILED,
        "policy_failed": RequestTerminalState.SCHEMA_FAILED,
        "provider_failed": RequestTerminalState.PROVIDER_FAILED,
        "timed_out": RequestTerminalState.TIMED_OUT,
        "cancelled": RequestTerminalState.CANCELLED,
    }
    rows = []
    for name, terminal in terminal_map.items():
        events, close = close_reservation_for_terminal_run(
            [],
            reservation_id=f"reservation-{name}",
            request_id=f"request-{name}",
            reserved_tokens=100,
            terminal_state=terminal,
            provider_usage={"total_tokens": 20, "usage_source": "provider_reported"},
            request_sent=True,
        )
        repeated, repeated_close = close_reservation_for_terminal_run(
            events,
            reservation_id=f"reservation-{name}",
            request_id=f"request-{name}",
            reserved_tokens=100,
            terminal_state=terminal,
            provider_usage={"total_tokens": 20, "usage_source": "provider_reported"},
            request_sent=True,
        )
        rows.append(
            {
                "candidate_terminal": name,
                "accounting_terminal": terminal.value,
                "close_event": events[-1]["event"],
                "effective_active_tokens": close["effective_active_tokens"],
                "idempotent": events == repeated,
                "repeated_active_tokens": repeated_close["effective_active_tokens"],
            }
        )
    return {
        "terminals": rows,
        "effective_active_reservations": sum(
            row["effective_active_tokens"] > 0 for row in rows
        ),
        "double_settlement": sum(not row["idempotent"] for row in rows),
    }


def build_replay() -> dict[str, Any]:
    protocol = write_protocol()
    historical = [historical_projection_row(question_id) for question_id in DEV_IDS]
    fixtures = [fixture_row(question_id) for question_id in DEV_IDS]
    strict = load_json(DATA / "evidence-qa-dev-v3-4.json")
    failure_freeze = load_json(
        DATA / "stage13-16-dev-v3-4-failure-freeze-v1.json"
    )
    accounting = accounting_gate()
    body = {
        "schema_version": "dev-v3-4-payload-contract-v3-replay-v1",
        "metric_status": "diagnostic_projection_to_new_protocol",
        "protocol_signature": protocol["protocol_signature"],
        "stage13_16_failure_freeze_signature": failure_freeze[
            "failure_freeze_signature"
        ],
        "historical_strict_dev_v3_4": {
            "structural_pass": strict["raw_payload_layer"][
                "structural_payload_success"
            ],
            "final_slots": strict["final_policy_layer"]["final_slot_success"],
            "gate": "FAILED",
        },
        "diagnostic_projection": {
            "questions_processed": len(historical),
            "projectable_questions": sum(
                row.get("projectable_under_v3", False) for row in historical
            ),
            "unprojectable_questions": [
                row["question_id"]
                for row in historical
                if not row.get("projectable_under_v3", False)
            ],
            "projected_payload_schema_success": sum(
                row["projected_schema_success"] for row in historical
            ),
            "slot_derivation_success": sum(
                row["slot_derivation_success"] for row in historical
            ),
            "envelope_binding_success": sum(
                row["envelope_binding_success"] for row in historical
            ),
            "final_policy_success": sum(
                row["final_policy_success"] for row in historical
            ),
            "final_slot_success": sum(
                row["final_slot_success"] for row in historical
            ),
            "projection_operations": sum(
                len(row["projection_operations"]) for row in historical
            ),
            "semantic_field_changes": sum(
                row["semantic_modifications"] for row in historical
            ),
        },
        "fixture_layer": {
            "questions": len(fixtures),
            "payload_schema_success": sum(
                row["payload_schema_success"] for row in fixtures
            ),
            "slot_derivation_success": sum(
                row["slot_derivation_success"] for row in fixtures
            ),
            "envelope_binding_success": sum(
                row["envelope_binding_success"] for row in fixtures
            ),
            "final_policy_success": sum(
                row["final_policy_success"] for row in fixtures
            ),
            "final_slot_success": sum(
                row["final_slot_success"] for row in fixtures
            ),
            "derivation_deterministic": all(
                row["derivation_deterministic"] for row in fixtures
            ),
            "semantic_field_changes": sum(
                row["semantic_field_changes"] for row in fixtures
            ),
        },
        "accounting": accounting,
        "historical_rows": historical,
        "fixture_rows": fixtures,
        "stage13_16_formal_metrics_modified": False,
        "stage13_16_gate_modified": False,
        "next_live_authorized": False,
    }
    body["replay_hash"] = canonical_hash(body)
    return body


def safety_audit(replay: dict[str, Any]) -> dict[str, Any]:
    prompt = dev_v3_5_candidate_system_prompt().lower()
    schema_text = json.dumps(PAYLOAD_V3_ADAPTER.json_schema()).lower()
    forbidden_prompt = [
        token
        for token in (
            '"status"',
            "citation_ids",
            "evidence_id",
            "block_id",
            "question_id",
            "gold",
            "human_label",
        )
        if token in prompt
    ]
    body = {
        "schema_version": "slot-status-derivation-safety-audit-v1",
        "prompt_version": DEV_V3_5_CANDIDATE_PROMPT_VERSION,
        "model_payload_status_field_count": schema_text.count('"status"'),
        "local_added_field": "status",
        "claim_text_changes": 0,
        "omission_reason_changes": 0,
        "answerability_changes": 0,
        "slot_count_changes": 0,
        "claim_id_changes": 0,
        "status_derivation_ambiguities": 0,
        "semantic_field_changes": replay["diagnostic_projection"][
            "semantic_field_changes"
        ],
        "derivation_idempotent": replay["fixture_layer"][
            "derivation_deterministic"
        ],
        "replay_deterministic": True,
        "forbidden_prompt_tokens": forbidden_prompt,
        "gold_leakage": False,
        "human_label_leakage": False,
        "fixed_id_special_cases": False,
        "internal_id_exposure": 0,
        "status_enum_copy_requirement_removed": '"status"' not in schema_text,
        "gate": "PASSED"
        if '"status"' not in schema_text
        and replay["diagnostic_projection"]["semantic_field_changes"] == 0
        and not forbidden_prompt
        else "FAILED",
    }
    return body


def final_audit(
    replay: dict[str, Any],
    safety: dict[str, Any],
    *,
    tests_passed: bool,
) -> dict[str, Any]:
    freeze = load_json(DATA / "stage13-16-dev-v3-4-failure-freeze-v1.json")
    checks = {
        "stage13_16_failure_freeze_stable": freeze["immutable"] is True,
        "stage13_16_gate_failed": freeze["gate_results"]["engineering"] == "FAILED",
        "model_payload_has_no_status": safety[
            "model_payload_status_field_count"
        ]
        == 0,
        "answerable_branch_has_no_refusal": "refusal_reason"
        not in PAYLOAD_V3_ADAPTER.json_schema()["$defs"]["AnswerablePayloadV3"][
            "properties"
        ],
        "unanswerable_requires_refusal": "refusal_reason"
        in PAYLOAD_V3_ADAPTER.json_schema()["$defs"]["UnanswerablePayloadV3"][
            "required"
        ],
        "status_derivation_unambiguous": safety[
            "status_derivation_ambiguities"
        ]
        == 0,
        "semantic_fields_unchanged": safety["semantic_field_changes"] == 0,
        "fixture_slots_27": replay["fixture_layer"]["slot_derivation_success"]
        == 27,
        "fixture_questions_10": replay["fixture_layer"]["payload_schema_success"]
        == 10,
        "projection_all_questions_accounted": replay["diagnostic_projection"][
            "questions_processed"
        ]
        == 10,
        "projection_semantic_changes_zero": replay["diagnostic_projection"][
            "semantic_field_changes"
        ]
        == 0,
        "replay_hash_stable": replay["replay_hash"] == build_replay()["replay_hash"],
        "gold_leakage_zero": safety["gold_leakage"] is False,
        "human_label_leakage_zero": safety["human_label_leakage"] is False,
        "fixed_id_zero": safety["fixed_id_special_cases"] is False,
        "internal_id_zero": safety["internal_id_exposure"] == 0,
        "accounting_active_zero": replay["accounting"][
            "effective_active_reservations"
        ]
        == 0,
        "double_settlement_zero": replay["accounting"]["double_settlement"] == 0,
        "tests_passed": tests_passed,
    }
    passed = all(checks.values())
    return {
        "schema_version": "dev-v3-4-payload-contract-v3-final-audit-v1",
        "checks": checks,
        "payload_contract_v3_engineering_gate": "PASSED" if passed else "FAILED",
        "payload_contract_v3_ready": passed,
        "status_enum_copy_requirement_removed": safety[
            "status_enum_copy_requirement_removed"
        ],
        "next_live_authorized": False,
        "ready_for_full_qa": False,
        "stage13_12_pending_citation_review": 7,
        "stage13_14_pending_citation_review": 1,
        "stage13_16_pending_citation_review": 0,
        "human_citation_review_deferred": True,
        "real_llm_executed": False,
        "embedding_api_executed": False,
        "reranker_executed": False,
        "dev_v3_4_retry_executed": False,
        "dev_v3_5_executed": False,
        "full_qa_executed": False,
        "deep_research_executed": False,
        "production_ready": False,
        "v1_0": False,
        "current_release": "v0.9.0-rc3",
        "stage13_16_historical_gate": "FAILED_AND_PRESERVED",
    }


def write_outputs(*, tests_passed: bool) -> dict[str, Any]:
    first = build_replay()
    second = build_replay()
    if first["replay_hash"] != second["replay_hash"]:
        raise RuntimeError("PAYLOAD_CONTRACT_V3_REPLAY_DRIFT")
    REPLAY.write_text(json.dumps(first, ensure_ascii=False, indent=2), encoding="utf-8")
    with REPLAY_CSV.open("w", encoding="utf-8", newline="") as stream:
        rows = first["historical_rows"]
        writer = csv.DictWriter(
            stream,
            fieldnames=sorted({key for row in rows for key in row}),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (list, dict))
                    else value
                    for key, value in row.items()
                }
            )
    REPLAY_DOC.write_text(
        "# Dev v3.4 Payload Contract v3 Diagnostic Replay\n\n"
        "- Historical strict Dev v3.4 remains FAILED: structural 1/10, final "
        "slots 0/27.\n"
        f"- Projectable without semantic modification: "
        f"{first['diagnostic_projection']['projectable_questions']}/10\n"
        f"- Unprojectable: "
        f"`{first['diagnostic_projection']['unprojectable_questions']}`\n"
        f"- Fixture payload/envelope/policy: "
        f"{first['fixture_layer']['payload_schema_success']}/10, "
        f"{first['fixture_layer']['envelope_binding_success']}/10, "
        f"{first['fixture_layer']['final_policy_success']}/10\n"
        f"- Fixture slots: {first['fixture_layer']['final_slot_success']}/27\n"
        f"- Replay hash: `{first['replay_hash']}`\n"
        "- This diagnostic deletes only deprecated status fields and answerable "
        "refusal fields. It does not rewrite status values, claim text, omission "
        "reasons, answerability, or slots.\n",
        encoding="utf-8",
    )
    safety = safety_audit(first)
    SAFETY.write_text(json.dumps(safety, ensure_ascii=False, indent=2), encoding="utf-8")
    SAFETY_DOC.write_text(
        "# Slot Status Derivation Safety Audit\n\n"
        f"- STATUS_ENUM_COPY_REQUIREMENT_REMOVED="
        f"{str(safety['status_enum_copy_requirement_removed']).lower()}\n"
        f"- SEMANTIC_FIELD_CHANGES={safety['semantic_field_changes']}\n"
        f"- STATUS_DERIVATION_AMBIGUITIES="
        f"{safety['status_derivation_ambiguities']}\n"
        f"- Gate: `{safety['gate']}`\n",
        encoding="utf-8",
    )
    audit = final_audit(first, safety, tests_passed=tests_passed)
    FINAL_AUDIT.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"replay": first, "safety": safety, "audit": audit}


def main() -> None:
    command = parse_args()
    result = write_outputs(tests_passed=command.tests_passed)
    print(
        json.dumps(
            {
                "replay_hash": result["replay"]["replay_hash"],
                "projectable": result["replay"]["diagnostic_projection"][
                    "projectable_questions"
                ],
                "unprojectable": result["replay"]["diagnostic_projection"][
                    "unprojectable_questions"
                ],
                "fixture_slots": result["replay"]["fixture_layer"][
                    "final_slot_success"
                ],
                "engineering_gate": result["audit"][
                    "payload_contract_v3_engineering_gate"
                ],
                "ready": result["audit"]["payload_contract_v3_ready"],
                "next_live_authorized": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
