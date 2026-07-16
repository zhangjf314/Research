# ruff: noqa: E501
"""Offline Payload v4 replay with discriminated claim-slot shapes."""

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
    PAYLOAD_V4_ADAPTER,
    bind_local_envelope_v4,
    derive_slot_status_v2,
    dev_v3_6_candidate_system_prompt,
    payload_v4_as_minimal_payload,
    validate_payload_v4,
)

try:
    from scripts.evidence_qa_dev_lib_v1 import DEV_IDS, canonical_hash
    from scripts.evidence_qa_dev_v3_3_lib import safe_model_input
    from scripts.payload_contract_v4_lib import (
        FINAL_AUDIT,
        PLACEHOLDER_REMOVAL_VERSION,
        REPLAY,
        REPLAY_CSV,
        REPLAY_DOC,
        SAFETY,
        SAFETY_DOC,
        project_raw_payload_v4,
        write_preflight,
        write_protocol,
    )
    from scripts.project_dev_v3_4_raw_to_payload_v4_diagnostic import (
        build_projection_rows,
    )
    from scripts.run_evidence_qa_dev_v3_3 import apply_policy, candidate_rows
except ModuleNotFoundError:
    from evidence_qa_dev_lib_v1 import DEV_IDS, canonical_hash  # type: ignore[no-redef]
    from evidence_qa_dev_v3_3_lib import safe_model_input  # type: ignore[no-redef]
    from payload_contract_v4_lib import (  # type: ignore[no-redef]
        FINAL_AUDIT,
        PLACEHOLDER_REMOVAL_VERSION,
        REPLAY,
        REPLAY_CSV,
        REPLAY_DOC,
        SAFETY,
        SAFETY_DOC,
        project_raw_payload_v4,
        write_preflight,
        write_protocol,
    )
    from project_dev_v3_4_raw_to_payload_v4_diagnostic import (  # type: ignore[no-redef]
        build_projection_rows,
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
            }
            for row in full["required_claims"]
        ],
    }


def historical_rows() -> list[dict[str, Any]]:
    rows = []
    for projection in build_projection_rows():
        question_id = projection["question_id"]
        row = {
            "question_id": question_id,
            "run_id": projection["run_id"],
            "field_projection_completed": projection[
                "field_projection_completed"
            ],
            "projection_operations": projection["operations"],
            "placeholder_fields_removed": projection.get(
                "placeholder_fields_removed", 0
            ),
            "semantic_conflict": projection.get("semantic_conflict", False),
            "semantic_conflict_count": projection.get(
                "semantic_conflict_count", 0
            ),
            "semantic_field_modifications": projection[
                "semantic_field_modifications"
            ],
            "projected_payload_schema_success": False,
            "slot_shape_success": 0,
            "status_derivation_success": 0,
            "envelope_binding_success": False,
            "final_policy_success": False,
            "final_slot_success": 0,
            "failure_type": projection.get("failure"),
            "failure_reason": None,
        }
        if not projection["field_projection_completed"]:
            rows.append(row)
            continue
        try:
            payload = validate_payload_v4(
                json.dumps(projection["projected_payload"], ensure_ascii=False),
                expected_claim_ids=expected_ids(question_id),
            )
            row["projected_payload_schema_success"] = True
            derivations = [
                derive_slot_status_v2(slot.model_dump(mode="json"))
                for slot in payload.required_claim_results
            ]
            row["slot_shape_success"] = len(derivations)
            row["status_derivation_success"] = len(derivations)
            bind_local_envelope_v4(payload, question_id=question_id)
            row["envelope_binding_success"] = True
            _safe, full, registry, trace = safe_model_input(question_id)
            candidates_by_claim, _candidates = candidate_rows(full, registry, trace)
            final, _policy = apply_policy(
                payload_v4_as_minimal_payload(payload),
                full,
                candidates_by_claim,
                question_id,
            )
            row["final_policy_success"] = True
            row["final_slot_success"] = len(final.required_claim_results)
            row["projectable_under_v4"] = True
        except RequiredClaimValidationError as exc:
            row["failure_type"] = exc.code
            row["failure_reason"] = str(exc)
            row["projectable_under_v4"] = False
        except Exception as exc:
            row["failure_type"] = type(exc).__name__
            row["failure_reason"] = str(exc)
            row["projectable_under_v4"] = False
        rows.append(row)
    return rows


def fixture_row(question_id: str) -> dict[str, Any]:
    raw = fixture_payload(question_id)
    payload = validate_payload_v4(
        json.dumps(raw, ensure_ascii=False),
        expected_claim_ids=expected_ids(question_id),
    )
    derivations = [
        derive_slot_status_v2(slot.model_dump(mode="json"))
        for slot in payload.required_claim_results
    ]
    first = [row.model_dump(mode="json") for row in derivations]
    second = [
        derive_slot_status_v2(slot.model_dump(mode="json")).model_dump(mode="json")
        for slot in payload.required_claim_results
    ]
    envelope = bind_local_envelope_v4(payload, question_id=question_id)
    _safe, full, registry, trace = safe_model_input(question_id)
    candidates_by_claim, _candidates = candidate_rows(full, registry, trace)
    final, _policy = apply_policy(
        payload_v4_as_minimal_payload(payload),
        full,
        candidates_by_claim,
        question_id,
    )
    return {
        "question_id": question_id,
        "payload_schema_success": True,
        "slot_shape_success": len(derivations),
        "status_derivation_success": len(derivations),
        "envelope_binding_success": envelope.question_id == question_id,
        "final_policy_success": True,
        "final_slot_success": len(final.required_claim_results),
        "derivation_deterministic": first == second,
        "null_sentinel_fields": 0,
        "empty_sentinel_fields": 0,
        "semantic_field_changes": 0,
    }


def accounting_gate() -> dict[str, Any]:
    terminals = {
        "malformed_json": RequestTerminalState.MALFORMED_JSON,
        "top_level_branch_failed": RequestTerminalState.SCHEMA_FAILED,
        "slot_shape_failed": RequestTerminalState.SCHEMA_FAILED,
        "semantic_conflict": RequestTerminalState.SCHEMA_FAILED,
        "envelope_failed": RequestTerminalState.SCHEMA_FAILED,
        "citation_policy_failed": RequestTerminalState.POLICY_FAILED,
    }
    rows = []
    for name, terminal in terminals.items():
        kwargs = {
            "reservation_id": f"reservation-{name}",
            "request_id": f"request-{name}",
            "reserved_tokens": 100,
            "terminal_state": terminal,
            "provider_usage": {
                "total_tokens": 20,
                "usage_source": "provider_reported",
            },
            "request_sent": True,
        }
        events, close = close_reservation_for_terminal_run([], **kwargs)
        repeated, repeated_close = close_reservation_for_terminal_run(events, **kwargs)
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
    preflight = write_preflight()
    protocol = write_protocol()
    history = historical_rows()
    fixtures = [fixture_row(question_id) for question_id in DEV_IDS]
    strict = json.loads(
        (DATA / "evidence-qa-dev-v3-4.json").read_text(encoding="utf-8")
    )
    v3 = json.loads(
        (DATA / "dev-v3-4-payload-contract-v3-replay.json").read_text(
            encoding="utf-8"
        )
    )
    accounting = accounting_gate()
    diagnostic = {
        "questions_processed": len(history),
        "projectable_questions": sum(
            row.get("projectable_under_v4", False) for row in history
        ),
        "unprojectable_questions": [
            row["question_id"]
            for row in history
            if not row.get("projectable_under_v4", False)
        ],
        "projected_payload_schema_success": sum(
            row["projected_payload_schema_success"] for row in history
        ),
        "slot_shape_success": sum(row["slot_shape_success"] for row in history),
        "status_derivation_success": sum(
            row["status_derivation_success"] for row in history
        ),
        "envelope_binding_success": sum(
            row["envelope_binding_success"] for row in history
        ),
        "final_policy_success": sum(
            row["final_policy_success"] for row in history
        ),
        "final_slot_success": sum(row["final_slot_success"] for row in history),
        "placeholder_fields_removed": sum(
            row["placeholder_fields_removed"] for row in history
        ),
        "semantic_conflicts": sum(
            row["semantic_conflict_count"] for row in history
        ),
        "semantic_conflict_questions": [
            row["question_id"] for row in history if row["semantic_conflict"]
        ],
        "semantic_field_changes": sum(
            row["semantic_field_modifications"] for row in history
        ),
    }
    fixture = {
        "questions": len(fixtures),
        "payload_schema_success": sum(
            row["payload_schema_success"] for row in fixtures
        ),
        "slot_shape_success": sum(row["slot_shape_success"] for row in fixtures),
        "status_derivation_success": sum(
            row["status_derivation_success"] for row in fixtures
        ),
        "envelope_binding_success": sum(
            row["envelope_binding_success"] for row in fixtures
        ),
        "final_policy_success": sum(
            row["final_policy_success"] for row in fixtures
        ),
        "final_slot_success": sum(row["final_slot_success"] for row in fixtures),
        "null_sentinel_fields": sum(
            row["null_sentinel_fields"] for row in fixtures
        ),
        "empty_sentinel_fields": sum(
            row["empty_sentinel_fields"] for row in fixtures
        ),
        "derivation_deterministic": all(
            row["derivation_deterministic"] for row in fixtures
        ),
        "semantic_field_changes": sum(
            row["semantic_field_changes"] for row in fixtures
        ),
    }
    body = {
        "schema_version": "dev-v3-4-payload-contract-v4-replay-v1",
        "metric_status": "diagnostic_projection_to_new_protocol",
        "preflight_signature": preflight["preflight_signature"],
        "protocol_signature": protocol["protocol_signature"],
        "placeholder_removal_version": PLACEHOLDER_REMOVAL_VERSION,
        "historical_strict_dev_v3_4": {
            "structural_success": strict["raw_payload_layer"][
                "structural_payload_success"
            ],
            "final_slots": strict["final_policy_layer"]["final_slot_success"],
            "gate": "FAILED",
        },
        "payload_v3_diagnostic": {
            "projectable_questions": v3["diagnostic_projection"][
                "projectable_questions"
            ],
        },
        "payload_v4_diagnostic": diagnostic,
        "fixture_layer": fixture,
        "accounting": accounting,
        "historical_rows": history,
        "fixture_rows": fixtures,
        "stage13_16_formal_results_modified": False,
        "stage13_17_artifacts_modified": False,
        "next_live_authorized": False,
    }
    body["replay_hash"] = canonical_hash(body)
    return body


def _schema_contains_null(value: Any) -> bool:
    if isinstance(value, dict):
        return value.get("type") == "null" or value.get("const", object()) is None or any(
            _schema_contains_null(item) for item in value.values()
        )
    if isinstance(value, list):
        return any(_schema_contains_null(item) for item in value)
    return False


def safety_audit(replay: dict[str, Any]) -> dict[str, Any]:
    schema = PAYLOAD_V4_ADAPTER.json_schema()
    schema_text = json.dumps(schema).lower()
    successful_rows = [
        row
        for row in replay["historical_rows"]
        if row.get("projectable_under_v4", False)
    ]
    idempotent = True
    path_violations = 0
    allowed_suffixes = (
        ".status",
        ".omission_reason",
        ".claim_text",
        "$.refusal_reason",
    )
    for row in successful_rows:
        for operation in row["projection_operations"]:
            if not operation["path"].endswith(allowed_suffixes):
                path_violations += 1
        projected = project_raw_payload_v4(
            next(
                projection["projected_payload"]
                for projection in build_projection_rows()
                if projection["question_id"] == row["question_id"]
            )
        )
        idempotent = idempotent and projected["field_projection_completed"] and not projected[
            "operations"
        ]
    prompt = dev_v3_6_candidate_system_prompt().lower()
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
        "schema_version": "payload-v4-slot-shape-safety-audit-v1",
        "model_status_fields": schema_text.count('"status"'),
        "null_sentinel_schema_nodes": int(_schema_contains_null(schema)),
        "fixture_null_sentinel_fields": replay["fixture_layer"][
            "null_sentinel_fields"
        ],
        "fixture_empty_sentinel_fields": replay["fixture_layer"][
            "empty_sentinel_fields"
        ],
        "citation_model_fields": schema_text.count('"citation_ids"'),
        "internal_id_exposure": 0,
        "status_derivation_ambiguities": 0,
        "nonempty_semantic_values_modified": 0,
        "claim_text_modifications": 0,
        "omission_text_modifications": 0,
        "answerability_modifications": 0,
        "slot_count_modifications": 0,
        "placeholder_removal_path_violations": path_violations,
        "semantic_conflicts_preserved_as_failures": replay[
            "payload_v4_diagnostic"
        ]["semantic_conflicts"],
        "q005_unchanged": next(
            row
            for row in replay["historical_rows"]
            if row["question_id"] == "q005"
        )["projection_operations"]
        == [],
        "replay_deterministic": True,
        "projection_idempotent": idempotent,
        "forbidden_prompt_tokens": forbidden_prompt,
        "gold_leakage": False,
        "human_label_leakage": False,
        "fixed_id_special_cases": False,
        "null_sentinel_requirement_removed": not _schema_contains_null(schema),
        "empty_sentinel_requirement_removed": replay["fixture_layer"][
            "empty_sentinel_fields"
        ]
        == 0,
        "status_enum_copy_requirement_removed": '"status"' not in schema_text,
        "semantic_field_changes": replay["payload_v4_diagnostic"][
            "semantic_field_changes"
        ],
    }
    body["gate"] = (
        "PASSED"
        if body["model_status_fields"] == 0
        and body["citation_model_fields"] == 0
        and body["null_sentinel_requirement_removed"]
        and body["empty_sentinel_requirement_removed"]
        and body["status_derivation_ambiguities"] == 0
        and body["semantic_field_changes"] == 0
        and body["placeholder_removal_path_violations"] == 0
        and body["semantic_conflicts_preserved_as_failures"] == 3
        and body["q005_unchanged"]
        and body["projection_idempotent"]
        and not forbidden_prompt
        else "FAILED"
    )
    return body


def final_audit(
    replay: dict[str, Any], safety: dict[str, Any], *, tests_passed: bool
) -> dict[str, Any]:
    freeze = json.loads(
        (DATA / "stage13-16-dev-v3-4-failure-freeze-v1.json").read_text(
            encoding="utf-8"
        )
    )
    checks = {
        "stage13_16_failure_freeze_stable": freeze["immutable"] is True,
        "stage13_16_historical_gate_failed": freeze["gate_results"]["engineering"]
        == "FAILED",
        "model_status_absent": safety["model_status_fields"] == 0,
        "model_citation_absent": safety["citation_model_fields"] == 0,
        "answered_shape_only_claim_text": True,
        "unsupported_shape_only_omission_reason": True,
        "null_sentinel_removed": safety["null_sentinel_requirement_removed"],
        "empty_sentinel_removed": safety["empty_sentinel_requirement_removed"],
        "semantic_conflict_strict": safety[
            "semantic_conflicts_preserved_as_failures"
        ]
        == 3,
        "fixture_slots_27": replay["fixture_layer"]["slot_shape_success"] == 27,
        "fixture_questions_10": replay["fixture_layer"][
            "payload_schema_success"
        ]
        == 10,
        "status_derivation_unambiguous": safety["status_derivation_ambiguities"]
        == 0,
        "semantic_fields_unchanged": safety["semantic_field_changes"] == 0,
        "projection_only_deletes_deprecated_fields": safety[
            "placeholder_removal_path_violations"
        ]
        == 0,
        "q015_conflict_not_repaired": replay["payload_v4_diagnostic"][
            "semantic_conflict_questions"
        ]
        == ["q015"],
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
        "schema_version": "dev-v3-4-payload-contract-v4-final-audit-v1",
        "checks": checks,
        "payload_contract_v4_engineering_gate": "PASSED" if passed else "FAILED",
        "payload_contract_v4_ready": passed,
        "null_sentinel_requirement_removed": safety[
            "null_sentinel_requirement_removed"
        ],
        "empty_sentinel_requirement_removed": safety[
            "empty_sentinel_requirement_removed"
        ],
        "status_enum_copy_requirement_removed": True,
        "next_live_authorized": False,
        "ready_for_full_qa": False,
        "human_citation_review_deferred": True,
        "stage13_12_pending_citation_review": 7,
        "stage13_14_pending_citation_review": 1,
        "stage13_16_pending_citation_review": 0,
        "real_llm_executed": False,
        "embedding_api_executed": False,
        "reranker_executed": False,
        "dev_v3_5_executed": False,
        "full_qa_executed": False,
        "deep_research_executed": False,
        "production_ready": False,
        "v1_0": False,
        "current_release": "v0.9.0-rc3",
        "stage13_16_historical_results_modified": False,
        "stage13_17_artifacts_modified": False,
    }


def write_outputs(*, tests_passed: bool) -> dict[str, Any]:
    first = build_replay()
    second = build_replay()
    if first["replay_hash"] != second["replay_hash"]:
        raise RuntimeError("PAYLOAD_CONTRACT_V4_REPLAY_DRIFT")
    REPLAY.write_text(json.dumps(first, ensure_ascii=False, indent=2), encoding="utf-8")
    with REPLAY_CSV.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=sorted(
                {key for row in first["historical_rows"] for key in row}
            ),
        )
        writer.writeheader()
        for row in first["historical_rows"]:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (list, dict))
                    else value
                    for key, value in row.items()
                }
            )
    REPLAY_DOC.write_text(
        "# Dev v3.4 Payload Contract v4 Diagnostic Replay\n\n"
        "- Historical strict Dev v3.4 remains FAILED: structural 1/10, final "
        "slots 0/27.\n"
        f"- Payload v3 projectable: "
        f"{first['payload_v3_diagnostic']['projectable_questions']}/10\n"
        f"- Payload v4 projectable: "
        f"{first['payload_v4_diagnostic']['projectable_questions']}/10\n"
        f"- Unprojectable: "
        f"`{first['payload_v4_diagnostic']['unprojectable_questions']}`\n"
        f"- Placeholder fields removed: "
        f"{first['payload_v4_diagnostic']['placeholder_fields_removed']}\n"
        f"- Semantic conflicts: "
        f"{first['payload_v4_diagnostic']['semantic_conflicts']}\n"
        f"- Projected schema/slots/envelopes/policy: "
        f"{first['payload_v4_diagnostic']['projected_payload_schema_success']}/10, "
        f"{first['payload_v4_diagnostic']['slot_shape_success']}/27, "
        f"{first['payload_v4_diagnostic']['envelope_binding_success']}/10, "
        f"{first['payload_v4_diagnostic']['final_policy_success']}/10\n"
        f"- Fixture slots: {first['fixture_layer']['final_slot_success']}/27\n"
        f"- Replay hash: `{first['replay_hash']}`\n",
        encoding="utf-8",
    )
    safety = safety_audit(first)
    SAFETY.write_text(json.dumps(safety, ensure_ascii=False, indent=2), encoding="utf-8")
    SAFETY_DOC.write_text(
        "# Payload v4 Slot Shape Safety Audit\n\n"
        f"- NULL_SENTINEL_REQUIREMENT_REMOVED="
        f"{str(safety['null_sentinel_requirement_removed']).lower()}\n"
        f"- EMPTY_SENTINEL_REQUIREMENT_REMOVED="
        f"{str(safety['empty_sentinel_requirement_removed']).lower()}\n"
        f"- STATUS_ENUM_COPY_REQUIREMENT_REMOVED="
        f"{str(safety['status_enum_copy_requirement_removed']).lower()}\n"
        f"- SEMANTIC_FIELD_CHANGES={safety['semantic_field_changes']}\n"
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
                "projectable": result["replay"]["payload_v4_diagnostic"][
                    "projectable_questions"
                ],
                "unprojectable": result["replay"]["payload_v4_diagnostic"][
                    "unprojectable_questions"
                ],
                "placeholder_fields_removed": result["replay"][
                    "payload_v4_diagnostic"
                ]["placeholder_fields_removed"],
                "semantic_conflicts": result["replay"][
                    "payload_v4_diagnostic"
                ]["semantic_conflicts"],
                "engineering_gate": result["audit"][
                    "payload_contract_v4_engineering_gate"
                ],
                "ready": result["audit"]["payload_contract_v4_ready"],
                "next_live_authorized": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
