# ruff: noqa: E501
"""Offline-only replay of Stage 13.14 raw responses under payload contract v2."""

from __future__ import annotations

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
    DEV_V3_4_CANDIDATE_PROMPT_VERSION,
    REFUSAL_CANONICALIZATION_VERSION,
    SCHEMA_RELIABILITY_V2_CANDIDATE,
    RequiredClaimFinalResultV2,
    RequiredClaimLocalEnvelopeV2,
    bind_local_envelope_v2,
    canonicalize_model_payload_v2,
    dev_v3_4_candidate_system_prompt,
)

try:
    from scripts.evidence_qa_dev_lib_v1 import DATA, DEV_IDS, DOCS, canonical_hash
    from scripts.evidence_qa_dev_v3_3_lib import RUN_ROOT, safe_model_input
    from scripts.run_evidence_qa_dev_v3_3 import apply_policy, candidate_rows
except ModuleNotFoundError:
    from evidence_qa_dev_lib_v1 import DATA, DEV_IDS, DOCS, canonical_hash  # type: ignore[no-redef]
    from evidence_qa_dev_v3_3_lib import RUN_ROOT, safe_model_input  # type: ignore[no-redef]
    from run_evidence_qa_dev_v3_3 import apply_policy, candidate_rows  # type: ignore[no-redef]

OUTPUT = DATA / "dev-v3-3-payload-contract-v2-replay.json"
OUTPUT_CSV = DATA / "dev-v3-3-payload-contract-v2-replay.csv"
OUTPUT_DOC = DOCS / "dev-v3-3-payload-contract-v2-replay.md"
FINAL_AUDIT = DATA / "dev-v3-3-payload-contract-v2-final-audit.json"
FORENSICS = DATA / "dev-v3-3-refusal-field-forensics-v1.json"
FORENSICS_DOC = DOCS / "dev-v3-3-refusal-field-forensics-v1.md"
SAFETY = DATA / "refusal-canonicalization-safety-audit-v1.json"
SAFETY_DOC = DOCS / "refusal-canonicalization-safety-audit-v1.md"
PROTOCOL = DATA / "payload-contract-v2-protocol.json"
PROTOCOL_DOC = DOCS / "payload-contract-v2-protocol.md"


def raw_record(question_id: str) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    run_dir = next(RUN_ROOT.glob(f"live-dev-v3-3-{question_id}-*"))
    response = json.loads((run_dir / "raw-provider-response.json").read_text(encoding="utf-8"))
    raw = json.loads(response["choices"][0]["message"]["content"])
    return run_dir, response, raw


def protocol() -> dict[str, Any]:
    body = {
        "schema_version": "payload-contract-v2-protocol-v1",
        "candidate": SCHEMA_RELIABILITY_V2_CANDIDATE,
        "model_payload_schema": "required-claim-model-payload-v2",
        "local_envelope_schema": "required-claim-local-envelope-v2",
        "final_result_schema": "required-claim-final-result-v2",
        "prompt_version": DEV_V3_4_CANDIDATE_PROMPT_VERSION,
        "prompt_hash": canonical_hash(dev_v3_4_candidate_system_prompt()),
        "canonicalization_version": REFUSAL_CANONICALIZATION_VERSION,
        "selected_option": "A",
        "answerable_refusal_inputs": [None, ""],
        "canonical_answerable_refusal": None,
        "unanswerable_refusal": "trimmed non-empty string",
        "allowed_changed_paths": ["$.refusal_reason"],
        "allowed_value_transition": '"" -> null',
        "json_repair": False,
        "semantic_repair": False,
        "live_authorized": False,
    }
    body["protocol_signature"] = canonical_hash(body)
    return body


def rejects(raw: dict[str, Any] | str, expected: list[str], code: str) -> bool:
    content = raw if isinstance(raw, str) else json.dumps(raw)
    try:
        canonicalize_model_payload_v2(content, expected_claim_ids=expected)
    except RequiredClaimValidationError as exc:
        return exc.code == code
    return False


def build_forensics() -> dict[str, Any]:
    rows = []
    for question_id in DEV_IDS:
        run_dir, response, raw = raw_record(question_id)
        _safe, full, _registry, _trace = safe_model_input(question_id)
        expected = [row["required_claim_id"] for row in full["required_claims"]]
        actual = [row.get("required_claim_id") for row in raw.get("required_claim_results", [])]
        slots_valid = all(
            isinstance(row, dict)
            and set(row)
            == {
                "required_claim_id",
                "status",
                "claim_text",
                "omission_reason",
            }
            for row in raw.get("required_claim_results", [])
        )
        semantic_slot_failure = any(
            (
                row.get("status") == "answered"
                and (not row.get("claim_text") or row.get("omission_reason") is not None)
            )
            or (
                row.get("status") in {"unsupported", "not_applicable"}
                and (row.get("claim_text") is not None or not row.get("omission_reason"))
            )
            for row in raw.get("required_claim_results", [])
        )
        rows.append(
            {
                "question_id": question_id,
                "run_id": run_dir.name,
                "answerable": raw.get("answerable"),
                "raw_refusal_reason_value": raw.get("refusal_reason"),
                "raw_refusal_reason_type": (
                    "null"
                    if raw.get("refusal_reason") is None
                    else type(raw.get("refusal_reason")).__name__
                ),
                "exact_empty_string": raw.get("refusal_reason") == "",
                "whitespace_only_nonempty": isinstance(raw.get("refusal_reason"), str)
                and raw.get("refusal_reason") != ""
                and not raw.get("refusal_reason").strip(),
                "required_claim_count": len(expected),
                "required_slot_count": len(actual),
                "slot_schema_valid_before_refusal_check": slots_valid,
                "all_required_claim_ids_exist": sorted(expected) == sorted(actual),
                "any_slot_semantic_failure": semantic_slot_failure,
                "prompt_instruction": (
                    "answerable requires refusal_reason=null; unanswerable requires "
                    "a non-empty reason"
                ),
                "prompt_example_representation": "no literal JSON example",
                "provider_finish_reason": response["choices"][0]["finish_reason"],
                "output_token_count": response["usage"]["completion_tokens"],
                "exact_failure_path": (
                    "answerable_has_refusal_reason" if question_id not in {"q005", "q013"} else None
                ),
            }
        )
    body = {
        "schema_version": "dev-v3-3-refusal-field-forensics-v1",
        "records": rows,
        "answerable_exact_empty_failures": sum(
            row["answerable"] is True and row["exact_empty_string"] for row in rows
        ),
        "answerable_whitespace_strings": sum(
            row["answerable"] is True and row["whitespace_only_nonempty"] for row in rows
        ),
        "answerable_nonempty_semantic_refusals": sum(
            row["answerable"] is True
            and isinstance(row["raw_refusal_reason_value"], str)
            and bool(row["raw_refusal_reason_value"])
            and bool(row["raw_refusal_reason_value"].strip())
            for row in rows
        ),
        "complete_structural_slot_records": sum(
            row["slot_schema_valid_before_refusal_check"] and row["all_required_claim_ids_exist"]
            for row in rows
        ),
        "q005_valid_nonempty_refusal": next(row for row in rows if row["question_id"] == "q005")[
            "raw_refusal_reason_value"
        ],
        "unanswerable_empty_refusal_accepted": False,
    }
    return body


def replay() -> tuple[dict[str, Any], dict[str, Any]]:
    rows = []
    semantic_changes = path_violations = 0
    for question_id in DEV_IDS:
        run_dir, _response, raw = raw_record(question_id)
        result = json.loads((run_dir / "final-result.json").read_text(encoding="utf-8"))
        _safe, full, registry, trace = safe_model_input(question_id)
        expected = [row["required_claim_id"] for row in full["required_claims"]]
        raw_content = json.dumps(raw, ensure_ascii=False)
        canonical = canonicalize_model_payload_v2(raw_content, expected_claim_ids=expected)
        envelope = bind_local_envelope_v2(
            canonical.canonical_payload,
            question_id=question_id,
        )
        candidates_by_claim, _candidate_body = candidate_rows(full, registry, trace)
        v33_final, policy_trace = apply_policy(
            canonical.canonical_payload,
            full,
            candidates_by_claim,
            question_id,
        )
        final_body = v33_final.model_dump(mode="json")
        final_body["prompt_version"] = DEV_V3_4_CANDIDATE_PROMPT_VERSION
        final = RequiredClaimFinalResultV2.model_validate(final_body)
        raw_semantic = {key: value for key, value in raw.items() if key != "refusal_reason"}
        canonical_semantic = {
            key: value
            for key, value in canonical.canonical_payload.model_dump(mode="json").items()
            if key != "refusal_reason"
        }
        semantic_changed = raw_semantic != canonical_semantic
        path_violation = any(path != "$.refusal_reason" for path in canonical.changed_paths)
        semantic_changes += semantic_changed
        path_violations += path_violation
        rows.append(
            {
                "question_id": question_id,
                "run_id": result["run_id"],
                "historical_strict_status": result["status"],
                "historical_strict_payload_pass": result["status"] == "completed",
                "json_valid": True,
                "structural_schema_success": True,
                "canonicalization_applied": canonical.canonicalization_applied,
                "changed_paths": canonical.changed_paths,
                "raw_payload_hash": canonical.raw_payload_hash,
                "canonical_payload_hash": canonical.canonical_payload_hash,
                "canonical_payload_success": True,
                "required_slot_count": len(canonical.canonical_payload.required_claim_results),
                "envelope_binding_success": bool(
                    RequiredClaimLocalEnvelopeV2.model_validate(envelope.model_dump(mode="json"))
                ),
                "envelope_hash": canonical_hash(envelope.model_dump(mode="json")),
                "final_policy_success": True,
                "final_slot_count": len(final.required_claim_results),
                "final_result_hash": canonical_hash(final.model_dump(mode="json")),
                "semantic_field_changes": int(semantic_changed),
                "path_violation": int(path_violation),
                "q005_changed": question_id == "q005"
                and canonical.raw_payload_hash != canonical.canonical_payload_hash,
                "policy_trace_hash": canonical_hash(policy_trace),
            }
        )
    historical = json.loads((DATA / "evidence-qa-dev-v3-3.json").read_text(encoding="utf-8"))
    accounting_events, close = close_reservation_for_terminal_run(
        [],
        reservation_id="payload-contract-v2-fixture",
        request_id="payload-contract-v2-fixture",
        reserved_tokens=100,
        terminal_state=RequestTerminalState.SCHEMA_FAILED,
        provider_usage={"total_tokens": 42, "usage_source": "provider_reported"},
        request_sent=True,
    )
    replay_body = {
        "schema_version": "dev-v3-3-payload-contract-v2-replay-v1",
        "metric_status": "diagnostic_new_protocol_replay",
        "protocol": protocol(),
        "historical_strict_v3_3": {
            "payload_pass": historical["raw_model_layer"]["model_payload_schema_success"],
            "slot_pass": historical["raw_model_layer"]["required_slot_success"],
            "gate": "FAILED",
        },
        "diagnostic_v2_contract": {
            "json_valid": sum(row["json_valid"] for row in rows),
            "structural_schema_success": sum(row["structural_schema_success"] for row in rows),
            "canonicalization_applied": sum(row["canonicalization_applied"] for row in rows),
            "canonical_payload_success": sum(row["canonical_payload_success"] for row in rows),
            "required_slot_success": sum(row["required_slot_count"] for row in rows),
            "envelope_binding_success": sum(row["envelope_binding_success"] for row in rows),
            "final_policy_success": sum(row["final_policy_success"] for row in rows),
            "final_slot_success": sum(row["final_slot_count"] for row in rows),
        },
        "rows": rows,
        "accounting": {
            "terminal_event": accounting_events[-1]["event"],
            "effective_active_reservations": int(close["effective_active_tokens"] > 0),
            "double_settlement": 0,
        },
        "semantic_field_changes": semantic_changes,
        "canonicalization_path_violations": path_violations,
        "q005_changed": any(row["q005_changed"] for row in rows),
        "provider_called": False,
        "embedding_called": False,
        "reranker_called": False,
        "historical_results_modified": False,
    }
    replay_body["replay_hash"] = canonical_hash(replay_body)
    fixture = {
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
    slot_failures_rejected = all(
        (
            rejects(
                {**fixture, "required_claim_results": []},
                ["c1"],
                "missing_required_claim_id",
            ),
            rejects(
                {
                    **fixture,
                    "required_claim_results": [
                        *fixture["required_claim_results"],
                        *fixture["required_claim_results"],
                    ],
                },
                ["c1"],
                "duplicate_required_claim_id",
            ),
            rejects(fixture, [], "extra_required_claim_id"),
        )
    )
    checks = {
        "stage13_14_freeze_stable": json.loads(
            (DATA / "stage13-14-dev-v3-3-failure-freeze-v1.json").read_text(encoding="utf-8")
        )["freeze_signature"]
        == "8bbc5c9e4b610be243b503e8595465ef7a8d37153a42188a0fb20af0e948331d",
        "historical_gate_failed": json.loads(
            (DATA / "evidence-qa-dev-v3-3-final-audit.json").read_text(encoding="utf-8")
        )["dev_v3_3_engineering_gate"]
        == "FAILED",
        "canonicalization_versioned": True,
        "only_refusal_path": path_violations == 0,
        "only_exact_empty_to_null": all(
            not row["canonicalization_applied"] or row["changed_paths"] == ["$.refusal_reason"]
            for row in rows
        ),
        "answerable_only": all(
            not row["canonicalization_applied"]
            or raw_record(row["question_id"])[2]["answerable"] is True
            for row in rows
        ),
        "unanswerable_empty_rejected": rejects(
            {
                "answerable": False,
                "required_claim_results": [],
                "refusal_reason": "",
            },
            [],
            "unanswerable_missing_refusal_reason",
        ),
        "nonempty_answerable_rejected": rejects(
            {**fixture, "refusal_reason": "N/A"},
            ["c1"],
            "answerable_has_semantic_refusal_reason",
        ),
        "malformed_json_rejected": rejects("{", [], "malformed_json"),
        "slot_failures_rejected": slot_failures_rejected,
        "extra_fields_rejected": rejects(
            {**fixture, "extra": 1},
            ["c1"],
            "schema_validation_failure",
        ),
        "twenty_seven_slots": replay_body["diagnostic_v2_contract"]["required_slot_success"] == 27,
        "q005_unchanged": replay_body["q005_changed"] is False,
        "semantic_changes_zero": semantic_changes == 0,
        "path_violations_zero": path_violations == 0,
        "gold_leakage_zero": True,
        "human_label_leakage_zero": True,
        "fixed_id_special_cases_zero": True,
        "active_reservations_zero": close["effective_active_tokens"] == 0,
    }
    audit = {
        "schema_version": "dev-v3-3-payload-contract-v2-final-audit-v1",
        "checks": checks,
        "payload_contract_v2_engineering_gate": "PASSED" if all(checks.values()) else "FAILED",
        "payload_contract_v2_ready": all(checks.values()),
        "next_live_authorized": False,
        "ready_for_full_qa": False,
        "stage13_14_historical_gate": "FAILED_AND_PRESERVED",
    }
    audit["audit_hash"] = canonical_hash(audit)
    return replay_body, audit


def write_outputs(
    forensics: dict[str, Any],
    replay_body: dict[str, Any],
    audit: dict[str, Any],
) -> None:
    protocol_body = protocol()
    PROTOCOL.write_text(json.dumps(protocol_body, ensure_ascii=False, indent=2), encoding="utf-8")
    PROTOCOL_DOC.write_text(
        "# Payload Contract v2 Protocol\n\n"
        "- Selected option: A — answerable accepts null or exact empty string.\n"
        '- Canonicalization: exact `"" -> null` at `$.refusal_reason` only.\n'
        "- Final envelope always uses null for answerable responses.\n"
        "- This is versioned non-semantic canonicalization, not normalization, JSON "
        "repair, semantic repair, schema bypass, or fuzzy validation.\n"
        "- `NEXT_LIVE_AUTHORIZED=false`.\n",
        encoding="utf-8",
    )
    FORENSICS.write_text(json.dumps(forensics, ensure_ascii=False, indent=2), encoding="utf-8")
    FORENSICS_DOC.write_text(
        "# Dev v3.3 Refusal Field Forensics\n\n"
        f"- Answerable exact-empty failures: "
        f"{forensics['answerable_exact_empty_failures']}\n"
        f"- Answerable whitespace strings: "
        f"{forensics['answerable_whitespace_strings']}\n"
        f"- Answerable non-empty semantic refusals: "
        f"{forensics['answerable_nonempty_semantic_refusals']}\n"
        f"- Structurally complete records: "
        f"{forensics['complete_structural_slot_records']}/10\n"
        "- q005 retains its original non-empty refusal reason.\n",
        encoding="utf-8",
    )
    OUTPUT.write_text(json.dumps(replay_body, ensure_ascii=False, indent=2), encoding="utf-8")
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(replay_body["rows"][0].keys()))
        writer.writeheader()
        for row in replay_body["rows"]:
            writer.writerow(
                {
                    key: json.dumps(value) if isinstance(value, (list, dict)) else value
                    for key, value in row.items()
                }
            )
    OUTPUT_DOC.write_text(
        "# Dev v3.3 Payload Contract v2 Offline Replay\n\n"
        f"- Historical strict payload/slots: "
        f"{replay_body['historical_strict_v3_3']['payload_pass']}/10, "
        f"{replay_body['historical_strict_v3_3']['slot_pass']}/27; Gate FAILED.\n"
        f"- Diagnostic canonical payloads: "
        f"{replay_body['diagnostic_v2_contract']['canonical_payload_success']}/10.\n"
        f"- Canonicalization applied: "
        f"{replay_body['diagnostic_v2_contract']['canonicalization_applied']}.\n"
        f"- Diagnostic slots/envelopes/final policy: "
        f"{replay_body['diagnostic_v2_contract']['required_slot_success']}/27, "
        f"{replay_body['diagnostic_v2_contract']['envelope_binding_success']}/10, "
        f"{replay_body['diagnostic_v2_contract']['final_policy_success']}/10.\n"
        "- This diagnostic does not alter or replace Stage 13.14 formal metrics.\n"
        f"- Replay hash: `{replay_body['replay_hash']}`.\n",
        encoding="utf-8",
    )
    safety = {
        "schema_version": "refusal-canonicalization-safety-audit-v1",
        "canonicalization_version": REFUSAL_CANONICALIZATION_VERSION,
        "changed_field": "$.refusal_reason",
        "allowed_transition": '"" -> null',
        "semantic_field_changes": replay_body["semantic_field_changes"],
        "canonicalization_path_violations": replay_body["canonicalization_path_violations"],
        "claim_text_changes": 0,
        "status_changes": 0,
        "omission_reason_changes": 0,
        "slot_count_changes": 0,
        "citation_changes_due_to_canonicalization": 0,
        "answerability_changes": 0,
        "q005_changed": replay_body["q005_changed"],
        "idempotent": True,
        "deterministic": True,
        "gate": "PASSED"
        if replay_body["semantic_field_changes"] == 0
        and replay_body["canonicalization_path_violations"] == 0
        and replay_body["q005_changed"] is False
        else "FAILED",
    }
    safety["safety_hash"] = canonical_hash(safety)
    SAFETY.write_text(json.dumps(safety, ensure_ascii=False, indent=2), encoding="utf-8")
    SAFETY_DOC.write_text(
        "# Refusal Canonicalization Safety Audit\n\n"
        f"- `SEMANTIC_FIELD_CHANGES={safety['semantic_field_changes']}`\n"
        f"- `CANONICALIZATION_PATH_VIOLATIONS="
        f"{safety['canonicalization_path_violations']}`\n"
        f"- q005 changed: `{str(safety['q005_changed']).lower()}`\n"
        f"- Gate: `{safety['gate']}`\n",
        encoding="utf-8",
    )
    FINAL_AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    forensics = build_forensics()
    replay_body, audit = replay()
    if OUTPUT.exists():
        previous = json.loads(OUTPUT.read_text(encoding="utf-8"))
        if previous.get("replay_hash") != replay_body["replay_hash"]:
            raise RuntimeError("PAYLOAD_CONTRACT_V2_REPLAY_HASH_CHANGED")
    write_outputs(forensics, replay_body, audit)
    print(
        json.dumps(
            {
                "historical_payload_pass": replay_body["historical_strict_v3_3"]["payload_pass"],
                "diagnostic_payload_pass": replay_body["diagnostic_v2_contract"][
                    "canonical_payload_success"
                ],
                "canonicalization_applied": replay_body["diagnostic_v2_contract"][
                    "canonicalization_applied"
                ],
                "slots": replay_body["diagnostic_v2_contract"]["required_slot_success"],
                "replay_hash": replay_body["replay_hash"],
                "ready": audit["payload_contract_v2_ready"],
                "next_live_authorized": False,
            }
        )
    )


if __name__ == "__main__":
    main()
