# ruff: noqa: E501
"""One controlled Dev v3.4 batch using payload contract v2."""

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
from paper_research.generation.schema_reliability import (
    DEV_V3_4_PROMPT_VERSION,
    REFUSAL_CANONICALIZATION_VERSION,
    DevV34FinalResult,
    DevV34LocalEnvelope,
    MinimalRequiredClaimsPayload,
    bind_dev_v3_4_envelope,
    canonicalize_model_payload_v2,
    dev_v3_4_system_prompt,
)

try:
    from scripts.evidence_qa_dev_lib_v1 import DEV_IDS, canonical_hash
    from scripts.evidence_qa_dev_v3_3_lib import output_budget, safe_model_input
    from scripts.evidence_qa_dev_v3_4_lib import (
        HEALTH,
        RUN_ROOT,
        write_freeze,
        write_visible_id_audit,
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
    from evidence_qa_dev_v3_4_lib import (  # type: ignore[no-redef]
        HEALTH,
        RUN_ROOT,
        write_freeze,
        write_visible_id_audit,
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


def verify_request(
    safe: dict[str, Any],
    messages: list[dict[str, str]],
    request_body: dict[str, Any],
    freeze: dict[str, Any],
) -> dict[str, str]:
    combined = (messages[0]["content"] + messages[1]["content"]).lower()
    forbidden = (
        "qa-required-claims-citation-id-v3",
        "citation_ids",
        "evidence_id",
        "block_id",
        "paper_id",
        "relation_id",
        "gold_",
        "human_label",
    )
    found = [token for token in forbidden if token in combined]
    if found:
        raise RuntimeError(f"DEV_V3_4_CONFIGURATION_INVALID: {found}")
    if canonical_hash(dev_v3_4_system_prompt()) != freeze["prompt_template_hash"]:
        raise RuntimeError("DEV_V3_4_CONFIGURATION_INVALID: prompt hash")
    if request_body["max_tokens"] != safe["output_budget"]["calculated_max_output_tokens"]:
        raise RuntimeError("DEV_V3_4_CONFIGURATION_INVALID: budget")
    return {
        "prompt_template_hash": freeze["prompt_template_hash"],
        "rendered_system_prompt_hash": hash_text(messages[0]["content"]),
        "rendered_user_prompt_hash": hash_text(messages[1]["content"]),
        "delivered_messages_hash": canonical_hash(messages),
        "exact_delivered_request_body_hash": canonical_hash(request_body),
    }


def preflight() -> dict[str, Any]:
    freeze = write_freeze()
    visible = write_visible_id_audit()
    settings = Settings()
    checks = {
        "freeze": freeze["frozen_before_live"],
        "visible_ids": visible["gate"] == "PASSED",
        "questions": freeze["question_ids"] == DEV_IDS,
        "claims": freeze["required_claims"] == 27,
        "q005": freeze["q005_required_claims"] == 0,
        "collection": settings.qdrant_collection == freeze["collection"],
        "embedding": settings.embedding_model == freeze["embedding"],
        "dimensions": settings.embedding_dimensions == 1024,
        "provider": settings.llm_provider == "siliconflow",
        "model": settings.llm_model == "Qwen/Qwen3-8B",
        "temperature": settings.llm_temperature == 0,
        "retries": settings.llm_max_retries == 0,
        "reranker": settings.rerank_enabled is False,
        "billing": settings.llm_billing_mode == "free",
        "payload_contract_v2_ready": json.loads(
            (Path("data/evaluation") / "dev-v3-3-payload-contract-v2-final-audit.json").read_text(
                encoding="utf-8"
            )
        )["payload_contract_v2_ready"],
    }
    for question_id in DEV_IDS:
        safe = safe_model_input(question_id)[0]
        messages = [
            {"role": "system", "content": dev_v3_4_system_prompt()},
            {"role": "user", "content": json.dumps(safe, ensure_ascii=False)},
        ]
        body = {
            "model": freeze["model"],
            "messages": messages,
            "temperature": 0,
            "max_tokens": safe["output_budget"]["calculated_max_output_tokens"],
            "stream": False,
            "enable_thinking": False,
            "response_format": {"type": "json_object"},
        }
        verify_request(safe, messages, body, freeze)
    if not all(checks.values()):
        raise RuntimeError(
            f"DEV_V3_4_CONFIGURATION_INVALID: {[key for key, value in checks.items() if not value]}"
        )
    return {"freeze_signature": freeze["protocol_freeze_signature"], "checks": checks}


TERMINAL_FILES = (
    "raw-model-payload.json",
    "structural-validation.json",
    "canonicalization-trace.json",
    "canonical-payload.json",
    "payload-validation.json",
    "local-envelope-binding.json",
    "obligation-analysis.json",
    "citation-selection-trace.json",
    "numeric-validation.json",
    "comparison-validation.json",
    "claim-fallback-trace.json",
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
    freeze: dict[str, Any],
) -> dict[str, Any]:
    safe, full, registry, trace = safe_model_input(question_id)
    run_id = f"live-dev-v3-4-{question_id}-{uuid.uuid4().hex[:12]}"
    request_id = f"{run_id}:primary:1:{uuid.uuid4().hex[:12]}"
    run_dir = RUN_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    ledger = run_dir / "request-ledger.jsonl"
    ledger.write_text("", encoding="utf-8")
    candidates_by_claim, candidates = candidate_rows(full, registry, trace)
    messages = [
        {"role": "system", "content": dev_v3_4_system_prompt()},
        {"role": "user", "content": json.dumps(safe, ensure_ascii=False)},
    ]
    request_body = {
        "model": settings.llm_model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": output_budget(len(full["required_claims"]))["calculated_max_output_tokens"],
        "stream": False,
        "enable_thinking": False,
        "response_format": {"type": "json_object"},
    }
    delivered = verify_request(safe, messages, request_body, freeze)
    pre_request = {
        "required-claims-input.json": safe,
        "model-payload-schema.json": MinimalRequiredClaimsPayload.model_json_schema(),
        "local-envelope-schema.json": DevV34LocalEnvelope.model_json_schema(),
        "canonicalization-policy.json": {
            "version": REFUSAL_CANONICALIZATION_VERSION,
            "allowed_path": "$.refusal_reason",
            "allowed_transition": '"" -> null',
        },
        "citation-registry.json": registry.model_dump(mode="json"),
        "candidate-evidence.json": candidates,
        "delivered-request-metadata.json": delivered,
        "protocol-snapshot.json": freeze,
        "accounting-reservation.json": {
            "reservation_id": request_id,
            "reserved_tokens": 24000,
            "accounting_policy": "request-accounting-v1",
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
        "evaluation_version": "evidence-qa-dev-v3.4",
        "request_id": request_id,
        "protocol_freeze_signature": freeze["protocol_freeze_signature"],
        "prompt_version": DEV_V3_4_PROMPT_VERSION,
        **delivered,
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
    }
    write_json(run_dir / "run-metadata.json", metadata)
    event(ledger, "request_id_allocated", request_id=request_id)
    event(ledger, "budget_reserved", reservation_id=request_id, reserved_tokens=24000)
    started = time.perf_counter()
    status = "provider_failed"
    failure_type = failure_reason = None
    usage: dict[str, Any] = {}
    raw_payload: dict[str, Any] = {}
    canonical_payload: dict[str, Any] = {}
    final_answer: dict[str, Any] = {}
    policy_trace: dict[str, Any] = {}
    canonicalization_applied = False
    changed_paths: list[str] = []
    request_sent = False
    try:
        if canonical_hash(request_body) != delivered["exact_delivered_request_body_hash"]:
            raise RuntimeError("DEV_V3_4_CONFIGURATION_INVALID: delivered body")
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
        content = body["choices"][0]["message"]["content"]
        decoded = json.loads(content)
        raw_payload = decoded
        write_json(run_dir / "raw-model-payload.json", raw_payload)
        structural = MinimalRequiredClaimsPayload.model_validate(decoded)
        write_json(
            run_dir / "structural-validation.json",
            {
                "json_valid": True,
                "structural_schema_valid": True,
                "slot_count": len(structural.required_claim_results),
                "payload_hash": canonical_hash(decoded),
            },
        )
        canonical = canonicalize_model_payload_v2(
            content,
            expected_claim_ids=[row["required_claim_id"] for row in full["required_claims"]],
        )
        canonical_payload = canonical.canonical_payload.model_dump(mode="json")
        canonicalization_applied = canonical.canonicalization_applied
        changed_paths = canonical.changed_paths
        write_json(
            run_dir / "canonicalization-trace.json",
            {
                **canonical.model_dump(mode="json", exclude={"raw_payload", "canonical_payload"}),
                "old_value_type": (
                    "string"
                    if canonical.canonicalization_applied
                    else type(decoded.get("refusal_reason")).__name__
                ),
                "old_value_summary": "exact_empty_string"
                if canonical.canonicalization_applied
                else "unchanged",
                "new_value": None
                if canonical.canonicalization_applied
                else decoded.get("refusal_reason"),
            },
        )
        write_json(run_dir / "canonical-payload.json", canonical_payload)
        write_json(
            run_dir / "payload-validation.json",
            {
                "canonical_payload_valid": True,
                "slot_cardinality_valid": True,
                "semantic_change": False,
                "changed_paths": changed_paths,
            },
        )
        envelope = bind_dev_v3_4_envelope(
            canonical.canonical_payload,
            question_id=question_id,
        )
        write_json(
            run_dir / "local-envelope-binding.json",
            {
                "canonical_payload_hash": canonical.canonical_payload_hash,
                "envelope_hash": canonical_hash(envelope.model_dump(mode="json")),
                "binding_sources": {
                    "question_id": "fixed_manifest",
                    "prompt_version": "protocol_freeze",
                    "citation_protocol": "protocol_freeze",
                },
                "semantic_fields_modified": False,
            },
        )
        v33_final, policy_trace = apply_policy(
            canonical.canonical_payload,
            full,
            candidates_by_claim,
            question_id,
        )
        final_answer = v33_final.model_dump(mode="json")
        final_answer["prompt_version"] = DEV_V3_4_PROMPT_VERSION
        final = DevV34FinalResult.model_validate(final_answer)
        final_answer = final.model_dump(mode="json")
        write_json(run_dir / "citation-selection-trace.json", policy_trace)
        write_json(run_dir / "obligation-analysis.json", {"slots": policy_trace["slots"]})
        write_json(
            run_dir / "numeric-validation.json",
            {"slots": [row.get("numeric_validation") for row in policy_trace["slots"]]},
        )
        write_json(
            run_dir / "comparison-validation.json",
            {"slots": [row.get("comparison_validation") for row in policy_trace["slots"]]},
        )
        write_json(run_dir / "claim-fallback-trace.json", policy_trace)
        status = "completed"
        event(ledger, "final_schema_validation_passed")
    except json.JSONDecodeError as exc:
        status, failure_type, failure_reason = "validation_failed", "malformed_json", str(exc)
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
    result = {
        "run_id": run_id,
        "question_id": question_id,
        "status": status,
        "failure_type": failure_type,
        "failure_reason": failure_reason,
        "raw_model_payload": raw_payload,
        "canonical_payload": canonical_payload,
        "final_answer": final_answer,
        "required_claim_count": len(full["required_claims"]),
        "raw_slot_count": len(raw_payload.get("required_claim_results", [])),
        "canonical_slot_count": len(canonical_payload.get("required_claim_results", [])),
        "final_slot_count": len(final_answer.get("required_claim_results", [])),
        "canonicalization_applied": canonicalization_applied,
        "canonicalization_changed_paths": changed_paths,
        "semantic_field_changes": 0,
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
        **delivered,
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
        raise RuntimeError("DEV_V3_4_BLOCKED_BY_PROVIDER_HEALTH")
    health = json.loads(HEALTH.read_text(encoding="utf-8"))
    if not health.get("safe_to_start_batch"):
        raise RuntimeError("DEV_V3_4_BLOCKED_BY_PROVIDER_HEALTH")
    if list(RUN_ROOT.glob("live-dev-v3-4-*/final-result.json")):
        raise RuntimeError("duplicate formal Dev v3.4 run set")
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    settings = Settings()
    freeze = write_freeze()
    results = []
    provider_failures = consecutive_failures = 0
    with httpx.Client() as client:
        for question_id in DEV_IDS:
            result = run_one(question_id, settings, client, freeze)
            results.append(result)
            if result["status"] == "provider_failed":
                provider_failures += 1
                consecutive_failures += 1
            else:
                consecutive_failures = 0
            if provider_failures >= 2 or consecutive_failures >= 2:
                break
            if result["active_reserved_tokens"] != 0:
                break
    print(json.dumps({"runs": len(results), "results": results}, ensure_ascii=False))


if __name__ == "__main__":
    main()
