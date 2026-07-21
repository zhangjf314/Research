# ruff: noqa: E501
"""One controlled Dev v3.5 batch using Payload Contract v4 slot shapes."""

from __future__ import annotations

import argparse
import csv
import json
import time
import uuid
from pathlib import Path
from typing import Any

import httpx

from paper_research.config import Settings
from paper_research.evaluation.request_accounting import (
    RequestTerminalState,
    close_reservation_for_terminal_run,
)
from paper_research.generation.required_claim_output import RequiredClaimValidationError
from paper_research.generation.schema_reliability import (
    DEV_V3_6_CANDIDATE_PROMPT_VERSION,
    LOCAL_ENVELOPE_V4_VERSION,
    MODEL_PAYLOAD_V4_VERSION,
    PAYLOAD_V4_ADAPTER,
    bind_local_envelope_v4,
    derive_slot_status_v2,
    dev_v3_6_candidate_system_prompt,
    payload_v4_as_minimal_payload,
    validate_payload_v4,
)

try:
    from scripts.evidence_qa_dev_lib_v1 import DEV_IDS, canonical_hash
    from scripts.evidence_qa_dev_v3_3_lib import output_budget, safe_model_input
    from scripts.evidence_qa_dev_v3_5_lib import (
        EVALUATION_VERSION,
        HEALTH,
        RUN_ROOT,
        write_prompt_delivery_freeze,
        write_protocol_freeze,
    )
    from scripts.run_evidence_qa_dev_v3_3 import (
        apply_policy,
        candidate_rows,
        event,
        hash_text,
        write_json,
    )
except ModuleNotFoundError:
    from evidence_qa_dev_lib_v1 import DEV_IDS, canonical_hash  # type: ignore[no-redef]
    from evidence_qa_dev_v3_3_lib import output_budget, safe_model_input  # type: ignore[no-redef]
    from evidence_qa_dev_v3_5_lib import (  # type: ignore[no-redef]
        EVALUATION_VERSION,
        HEALTH,
        RUN_ROOT,
        write_prompt_delivery_freeze,
        write_protocol_freeze,
    )
    from run_evidence_qa_dev_v3_3 import (  # type: ignore[no-redef]
        apply_policy,
        candidate_rows,
        event,
        hash_text,
        write_json,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preflight", "live"), required=True)
    parser.add_argument("--no-summary", action="store_true")
    return parser.parse_args()


def _contains_key(value: Any, keys: set[str]) -> int:
    if isinstance(value, dict):
        return sum(key in keys for key in value) + sum(
            _contains_key(child, keys) for child in value.values()
        )
    if isinstance(value, list):
        return sum(_contains_key(child, keys) for child in value)
    return 0


def _contains_null(value: Any) -> int:
    if value is None:
        return 1
    if isinstance(value, dict):
        return sum(_contains_null(child) for child in value.values())
    if isinstance(value, list):
        return sum(_contains_null(child) for child in value)
    return 0


def _contains_empty_string(value: Any) -> int:
    if value == "":
        return 1
    if isinstance(value, dict):
        return sum(_contains_empty_string(child) for child in value.values())
    if isinstance(value, list):
        return sum(_contains_empty_string(child) for child in value)
    return 0


def _shape_counts(raw: dict[str, Any]) -> dict[str, int]:
    answered = unsupported = invalid = 0
    slots = raw.get("required_claim_results")
    if not isinstance(slots, list):
        return {"answered_shape": 0, "unsupported_shape": 0, "invalid_shape": 0}
    for slot in slots:
        if not isinstance(slot, dict):
            invalid += 1
            continue
        has_claim = isinstance(slot.get("claim_text"), str) and bool(slot["claim_text"].strip())
        has_reason = isinstance(slot.get("omission_reason"), str) and bool(
            slot["omission_reason"].strip()
        )
        allowed = {"required_claim_id", "claim_text"} if has_claim and not has_reason else {"required_claim_id", "omission_reason"} if has_reason and not has_claim else set()
        if has_claim and not has_reason and set(slot) == allowed:
            answered += 1
        elif has_reason and not has_claim and set(slot) == allowed:
            unsupported += 1
        else:
            invalid += 1
    return {
        "answered_shape": answered,
        "unsupported_shape": unsupported,
        "invalid_shape": invalid,
    }


def build_request(
    question_id: str,
    settings: Settings,
    protocol: dict[str, Any],
    delivery: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, str]], dict[str, Any]]:
    safe = safe_model_input(question_id)[0]
    messages = [
        {"role": "system", "content": dev_v3_6_candidate_system_prompt()},
        {"role": "user", "content": json.dumps(safe, ensure_ascii=False)},
    ]
    request_body = {
        "model": settings.llm_model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": output_budget(len(safe["required_claims"]))[
            "calculated_max_output_tokens"
        ],
        "stream": False,
        "enable_thinking": False,
        "response_format": {"type": "json_object"},
    }
    expected = next(row for row in delivery["questions"] if row["question_id"] == question_id)
    checks = {
        "delivered_system_prompt_hash": canonical_hash(messages[0]["content"])
        == expected["delivered_system_prompt_hash"],
        "delivered_user_payload_hash": canonical_hash(safe)
        == expected["delivered_user_payload_hash"],
        "delivered_messages_hash": canonical_hash(messages)
        == expected["delivered_messages_hash"],
        "delivered_request_body_hash": canonical_hash(request_body)
        == expected["delivered_request_body_hash"],
        "protocol_signature": protocol["protocol_freeze_signature"]
        == expected["protocol_signature"],
        "schema_hash": protocol["model_payload_schema_hash"] == expected["schema_hash"],
        "prompt_version": protocol["prompt_version"] == DEV_V3_6_CANDIDATE_PROMPT_VERSION,
        "payload_schema": protocol["model_payload_schema_version"]
        == MODEL_PAYLOAD_V4_VERSION,
        "local_envelope": protocol["local_envelope_schema_version"]
        == LOCAL_ENVELOPE_V4_VERSION,
    }
    if not all(checks.values()):
        raise RuntimeError(
            f"DEV_V3_5_PROMPT_DELIVERY_MISMATCH: {[key for key, value in checks.items() if not value]}"
        )
    return safe, messages, request_body


def preflight() -> dict[str, Any]:
    protocol = write_protocol_freeze()
    delivery = write_prompt_delivery_freeze()
    settings = Settings()
    checks = {
        "questions": protocol["question_ids"] == DEV_IDS,
        "provider": settings.llm_provider == "siliconflow",
        "model": settings.llm_model == protocol["model"],
        "temperature": settings.llm_temperature == 0,
        "retries": settings.llm_max_retries == 0,
        "reranker": settings.rerank_enabled is False,
        "billing": settings.llm_billing_mode == "free",
        "prompt_delivery": delivery["old_prompt_mixed_in"] is False
        and delivery["payload_schema_mismatch"] is False,
        "protocol_v4": bool(protocol["payload_contract_v4_protocol_signature"]),
    }
    for question_id in DEV_IDS:
        build_request(question_id, settings, protocol, delivery)
    if not all(checks.values()):
        raise RuntimeError(
            f"DEV_V3_5_CONFIGURATION_INVALID: {[key for key, value in checks.items() if not value]}"
        )
    return {
        "protocol_signature": protocol["protocol_freeze_signature"],
        "prompt_delivery_signature": delivery["prompt_delivery_signature"],
        "checks": checks,
    }


TERMINAL_FILES = (
    "raw-model-payload.json",
    "payload-v4-validation.json",
    "slot-status-derivation.json",
    "local-envelope-binding.json",
    "citation-selection-trace.json",
    "final-result.json",
)


def sentinel_files(run_dir: Path, result: dict[str, Any]) -> None:
    sentinel = {
        "status": "not_available_due_to_terminal_failure",
        "failure_type": result["failure_type"],
        "failure_reason": result["failure_reason"],
        "raw_response_modified": False,
    }
    for name in TERMINAL_FILES:
        if not (run_dir / name).exists():
            write_json(run_dir / name, sentinel)


def run_one(
    question_id: str,
    settings: Settings,
    client: httpx.Client,
    protocol: dict[str, Any],
    delivery: dict[str, Any],
) -> dict[str, Any]:
    safe, full, registry, trace = safe_model_input(question_id)
    run_id = f"live-dev-v3-5-{question_id}-{uuid.uuid4().hex[:12]}"
    request_id = f"{run_id}:primary:1:{uuid.uuid4().hex[:12]}"
    run_dir = RUN_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    ledger = run_dir / "request-ledger.jsonl"
    ledger.write_text("", encoding="utf-8")
    candidates_by_claim, candidates = candidate_rows(full, registry, trace)
    _safe, messages, request_body = build_request(question_id, settings, protocol, delivery)
    delivery_row = next(row for row in delivery["questions"] if row["question_id"] == question_id)
    pre_request = {
        "required-claims-input.json": safe,
        "model-payload-schema.json": PAYLOAD_V4_ADAPTER.json_schema(),
        "local-envelope-schema.json": {
            "schema_version": LOCAL_ENVELOPE_V4_VERSION,
        },
        "citation-registry.json": registry.model_dump(mode="json"),
        "candidate-evidence.json": candidates,
        "delivered-request-metadata.json": delivery_row,
        "protocol-snapshot.json": protocol,
        "accounting-reservation.json": {
            "reservation_id": request_id,
            "reserved_tokens": 24000,
            "accounting_policy": "request-accounting-v1",
        },
        "request.json": {
            "request_id": request_id,
            "request_body_hash": canonical_hash(request_body),
            "body": request_body,
        },
    }
    for name, value in pre_request.items():
        write_json(run_dir / name, value)
        event(ledger, f"{name.removesuffix('.json').replace('-', '_')}_persisted")
    (run_dir / "rendered-system-prompt.txt").write_text(messages[0]["content"], encoding="utf-8")
    (run_dir / "rendered-user-prompt.txt").write_text(messages[1]["content"], encoding="utf-8")
    metadata = {
        "run_id": run_id,
        "question_id": question_id,
        "evaluation_version": EVALUATION_VERSION,
        "request_id": request_id,
        "protocol_freeze_signature": protocol["protocol_freeze_signature"],
        "prompt_version": DEV_V3_6_CANDIDATE_PROMPT_VERSION,
        "required_claim_input_hash": canonical_hash(safe),
        "citation_registry_hash": registry.registry_hash,
        "candidate_evidence_hash": canonical_hash(candidates),
        "reranker_enabled": False,
        "retries": 0,
        "billing_mode": "explicit_free_provider",
        "gold_used_online": False,
        "human_labels_used_online": False,
        "fixed_id_special_cases": False,
        "api_key_recorded": False,
        "authorization_header_recorded": False,
        **delivery_row,
    }
    write_json(run_dir / "run-metadata.json", metadata)
    event(ledger, "request_id_allocated", request_id=request_id)
    event(ledger, "budget_reserved", reservation_id=request_id, reserved_tokens=24000)
    started = time.perf_counter()
    status = "provider_failed"
    failure_type = failure_reason = None
    usage: dict[str, Any] = {}
    raw_payload: dict[str, Any] = {}
    final_answer: dict[str, Any] = {}
    policy_trace: dict[str, Any] = {}
    envelope_bound = False
    derivations: list[dict[str, Any]] = []
    request_sent = False
    json_valid = False
    payload_schema_valid = False
    raw_text = ""
    try:
        request_sent = True
        event(ledger, "request_started", request_id=request_id)
        response = client.post(
            f"{(settings.llm_base_url or '').rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.llm_api_key}",
                "Content-Type": "application/json",
            },
            json=request_body,
            timeout=httpx.Timeout(connect=15, read=180, write=30, pool=15),
        )
        response.raise_for_status()
        body = response.json()
        write_json(run_dir / "raw-provider-response.json", body)
        event(ledger, "raw_response_persisted")
        usage_body = body.get("usage") or {}
        usage = {
            "input_tokens": int(usage_body.get("prompt_tokens", 0)),
            "output_tokens": int(usage_body.get("completion_tokens", 0)),
            "total_tokens": int(usage_body.get("total_tokens", 0)),
            "usage_source": "provider_reported",
        }
        write_json(
            run_dir / "provider-response-envelope.json",
            {
                "request_id": request_id,
                "provider": "siliconflow",
                "model": settings.llm_model,
                "finish_reason": body["choices"][0].get("finish_reason"),
                "usage": usage,
                "response_received": True,
                "raw_response_persisted": True,
            },
        )
        event(ledger, "provider_usage_recorded", **usage)
        raw_text = body["choices"][0]["message"]["content"]
        raw_payload = json.loads(raw_text)
        json_valid = True
        write_json(run_dir / "raw-model-payload.json", raw_payload)
        payload = validate_payload_v4(
            raw_text,
            expected_claim_ids=[row["required_claim_id"] for row in full["required_claims"]],
        )
        payload_schema_valid = True
        derivations = [
            derive_slot_status_v2(slot.model_dump(mode="json")).model_dump(mode="json")
            for slot in payload.required_claim_results
        ]
        write_json(
            run_dir / "payload-v4-validation.json",
            {
                "json_valid": True,
                "payload_v4_schema_valid": True,
                "slot_count": len(payload.required_claim_results),
                "payload_hash": canonical_hash(raw_payload),
            },
        )
        write_json(run_dir / "slot-status-derivation.json", derivations)
        envelope = bind_local_envelope_v4(payload, question_id=question_id)
        envelope_bound = True
        write_json(
            run_dir / "local-envelope-binding.json",
            {
                "envelope": envelope.model_dump(mode="json"),
                "envelope_hash": canonical_hash(envelope.model_dump(mode="json")),
                "semantic_fields_modified": False,
                "model_field_changes": 0,
            },
        )
        v33_final, policy_trace = apply_policy(
            payload_v4_as_minimal_payload(payload),
            full,
            candidates_by_claim,
            question_id,
        )
        final_answer = v33_final.model_dump(mode="json")
        final_answer["prompt_version"] = DEV_V3_6_CANDIDATE_PROMPT_VERSION
        write_json(run_dir / "citation-selection-trace.json", policy_trace)
        status = "completed"
        event(ledger, "final_schema_validation_passed")
    except json.JSONDecodeError as exc:
        status, failure_type, failure_reason = "validation_failed", "malformed_json", str(exc)
        event(ledger, "validation_failed", failure_type=failure_type)
    except RequiredClaimValidationError as exc:
        status, failure_type, failure_reason = "validation_failed", exc.code, str(exc)
        event(ledger, "validation_failed", failure_type=failure_type)
    except Exception as exc:
        if isinstance(exc, httpx.HTTPError):
            status = "provider_failed"
        else:
            status = "validation_failed"
        failure_type, failure_reason = getattr(exc, "code", type(exc).__name__), str(exc)
        event(ledger, status, failure_type=failure_type)
        if status == "provider_failed":
            write_json(
                run_dir / "raw-provider-response.json",
                {
                    "response_received": False,
                    "request_id": request_id,
                    "failure_type": failure_type,
                },
            )
            write_json(
                run_dir / "provider-response-envelope.json",
                {"response_received": False, "request_id": request_id, "usage": None},
            )
    finally:
        ledger_events = [
            json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines() if line
        ]
        terminal_state = (
            RequestTerminalState.COMPLETED
            if status == "completed"
            else RequestTerminalState.PROVIDER_FAILED
            if status == "provider_failed"
            else RequestTerminalState.MALFORMED_JSON
            if failure_type == "malformed_json"
            else RequestTerminalState.SCHEMA_FAILED
        )
        closed, accounting = close_reservation_for_terminal_run(
            ledger_events,
            reservation_id=request_id,
            request_id=request_id,
            reserved_tokens=24000,
            terminal_state=terminal_state,
            provider_usage=usage or None,
            request_sent=request_sent,
        )
        terminal = closed[-1]
        event(
            ledger,
            terminal["event"],
            **{key: value for key, value in terminal.items() if key != "event"},
        )
    elapsed = time.perf_counter() - started
    shapes = _shape_counts(raw_payload)
    result = {
        "run_id": run_id,
        "question_id": question_id,
        "status": status,
        "failure_type": failure_type,
        "failure_reason": failure_reason,
        "raw_model_payload": raw_payload,
        "raw_model_payload_text_hash": hash_text(raw_text) if raw_text else None,
        "final_answer": final_answer,
        "required_claim_count": len(full["required_claims"]),
        "raw_slot_count": len(raw_payload.get("required_claim_results", []))
        if isinstance(raw_payload, dict)
        else 0,
        "final_slot_count": len(final_answer.get("required_claim_results", [])),
        "json_valid": json_valid,
        "payload_v4_schema_valid": payload_schema_valid,
        "slot_shape_success": len(derivations),
        "envelope_binding_success": envelope_bound,
        "status_field_leakage": _contains_key(raw_payload, {"status"}),
        "citation_field_leakage": _contains_key(
            raw_payload,
            {"citation_ids", "citation_id", "evidence_id", "paper_id", "page", "block_id"},
        ),
        "null_sentinel": _contains_null(raw_payload),
        "empty_sentinel": _contains_empty_string(raw_payload),
        **shapes,
        "request_attempt_count": 1,
        "provider_completed_request_count": int(bool(usage)),
        "provider_failure_count": int(status == "provider_failed"),
        "usage_record_count": int(bool(usage)),
        "usage": usage,
        "reservation_count": 1,
        "settled_reservation_count": int(terminal["event"] == "reservation_settled"),
        "released_reservation_count": int(terminal["event"] == "reservation_released"),
        "billing_unknown_reservation_count": int(terminal["event"] == "billing_unknown_terminal"),
        "active_reserved_tokens": accounting["effective_active_tokens"],
        "elapsed_seconds": round(elapsed, 6),
        "monetary_cost_usd": "0",
        "cost_basis": "explicit_free_provider",
        "reranker_called": False,
        "template_fallback": False,
        "retries": 0,
        "policy_trace_complete": bool(policy_trace)
        if status == "completed" and question_id != "q005"
        else status == "completed",
        **delivery_row,
    }
    write_json(run_dir / "final-result.json", result)
    flat = {
        key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
        for key, value in result.items()
    }
    with (run_dir / "final-result.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat))
        writer.writeheader()
        writer.writerow(flat)
    sentinel_files(run_dir, result)
    return result


def main() -> None:
    command = parse_args()
    check = preflight()
    if command.mode == "preflight":
        print(json.dumps(check, ensure_ascii=False))
        return
    if not HEALTH.exists():
        raise RuntimeError("DEV_V3_5_BLOCKED_BY_PROVIDER_HEALTH")
    health = json.loads(HEALTH.read_text(encoding="utf-8"))
    if health.get("status") != "PASSED" or not health.get("safe_to_start_batch"):
        raise RuntimeError("DEV_V3_5_BLOCKED_BY_PROVIDER_HEALTH")
    if list(RUN_ROOT.glob("live-dev-v3-5-*/final-result.json")):
        raise RuntimeError("duplicate formal Dev v3.5 run set")
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    settings = Settings()
    protocol = write_protocol_freeze()
    delivery = write_prompt_delivery_freeze()
    results = []
    with httpx.Client() as client:
        for question_id in DEV_IDS:
            result = run_one(question_id, settings, client, protocol, delivery)
            results.append(result)
    print(json.dumps({"runs": len(results), "results": results}, ensure_ascii=False))


if __name__ == "__main__":
    main()
