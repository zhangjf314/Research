# ruff: noqa: E501
"""One controlled Dev v3.3 batch using the minimal-payload schema."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from paper_research.config import Settings
from paper_research.evaluation.request_accounting import (
    RequestTerminalState,
    close_reservation_for_terminal_run,
)
from paper_research.generation.citation_selection import (
    CitationCandidate,
    FallbackAction,
    select_citations,
    validate_comparison_evidence,
    validate_numeric_evidence,
)
from paper_research.generation.required_claim_output import RequiredClaimValidationError
from paper_research.generation.schema_reliability import (
    DEV_V3_3_PROMPT_VERSION,
    DevV33RequiredClaimsEnvelope,
    MinimalRequiredClaimsPayload,
    bind_dev_v3_3_envelope,
    dev_v3_3_system_prompt,
    parse_minimal_payload,
)

try:
    from scripts.evidence_qa_dev_lib_v1 import DEV_IDS, canonical_hash
    from scripts.evidence_qa_dev_v3_3_lib import (
        HEALTH,
        RUN_ROOT,
        safe_model_input,
        write_freeze,
        write_visible_id_audit,
    )
    from scripts.replay_dev_v3_2_citation_policy_v1 import build_candidates
except ModuleNotFoundError:
    from evidence_qa_dev_lib_v1 import DEV_IDS, canonical_hash  # type: ignore[no-redef]
    from evidence_qa_dev_v3_3_lib import (  # type: ignore[no-redef]
        HEALTH,
        RUN_ROOT,
        safe_model_input,
        write_freeze,
        write_visible_id_audit,
    )
    from replay_dev_v3_2_citation_policy_v1 import build_candidates  # type: ignore[no-redef]


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preflight", "live"), required=True)
    parser.add_argument("--no-summary", action="store_true")
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def event(path: Path, name: str, **values: Any) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "event_id": uuid.uuid4().hex,
                    "event": name,
                    "timestamp": datetime.now(UTC).isoformat(),
                    **values,
                },
                ensure_ascii=False,
            )
            + "\n"
        )


def verify_safe_request(
    safe_input: dict[str, Any],
    messages: list[dict[str, str]],
    request_body: dict[str, Any],
    freeze: dict[str, Any],
) -> dict[str, str]:
    system = messages[0]["content"]
    user = messages[1]["content"]
    combined = (system + user).lower()
    forbidden = (
        "qa-required-claims-citation-id-v3.1",
        "qa-required-claims-citation-id-v3.2",
        "citation_ids",
        "evidence_id",
        "block_id",
        "paper_id",
        "relation_id",
        "human_label",
        "core_gold",
        "gold_block",
    )
    found = [token for token in forbidden if token in combined]
    if found:
        raise RuntimeError(f"DEV_V3_3_CONFIGURATION_INVALID: {found}")
    if canonical_hash(dev_v3_3_system_prompt()) != freeze["prompt_hash"]:
        raise RuntimeError("DEV_V3_3_CONFIGURATION_INVALID: prompt hash")
    if safe_input["output_budget"]["calculated_max_output_tokens"] != request_body["max_tokens"]:
        raise RuntimeError("DEV_V3_3_CONFIGURATION_INVALID: output budget")
    return {
        "template_hash": freeze["prompt_hash"],
        "rendered_system_prompt_hash": hash_text(system),
        "rendered_user_prompt_hash": hash_text(user),
        "normalized_messages_hash": canonical_hash(messages),
        "exact_delivered_request_body_hash": canonical_hash(request_body),
    }


def preflight() -> dict[str, Any]:
    freeze = write_freeze()
    visible = write_visible_id_audit()
    settings = Settings()
    checks = {
        "branch_protocol_frozen": freeze["frozen_before_live"],
        "visible_id_audit": visible["gate"] == "PASSED",
        "questions": freeze["fixed_manifest"]["question_ids"] == DEV_IDS,
        "claims": freeze["fixed_manifest"]["required_claims"] == 27,
        "q005_zero": freeze["fixed_manifest"]["q005_required_claims"] == 0,
        "collection": settings.qdrant_collection == freeze["collection"],
        "embedding": settings.embedding_model == freeze["embedding"],
        "dimensions": settings.embedding_dimensions == freeze["embedding_dimensions"],
        "provider": settings.llm_provider == freeze["provider"],
        "model": settings.llm_model == freeze["model"],
        "temperature": settings.llm_temperature == 0,
        "retries": settings.llm_max_retries == 0,
        "reranker": settings.rerank_enabled is False,
        "billing": settings.llm_billing_mode == "free",
    }
    for question_id in DEV_IDS:
        safe, _full, _registry, _trace = safe_model_input(question_id)
        messages = [
            {"role": "system", "content": dev_v3_3_system_prompt()},
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
        verify_safe_request(safe, messages, body, freeze)
    if not all(checks.values()):
        raise RuntimeError(
            f"DEV_V3_3_CONFIGURATION_INVALID: {[key for key, value in checks.items() if not value]}"
        )
    return {"freeze_signature": freeze["protocol_freeze_signature"], "checks": checks}


def candidate_rows(
    full_input: dict[str, Any], registry: Any, trace: dict[str, Any]
) -> tuple[dict[str, list[CitationCandidate]], list[dict[str, Any]]]:
    by_claim = {}
    body = []
    registry_body = registry.model_dump(mode="json")
    for claim in full_input["required_claims"]:
        candidates = build_candidates(claim, registry_body, trace, set())
        by_claim[claim["required_claim_id"]] = candidates
        body.append(
            {
                "required_claim_id": claim["required_claim_id"],
                "candidates": [candidate.__dict__ for candidate in candidates],
            }
        )
    return by_claim, body


def apply_policy(
    raw: MinimalRequiredClaimsPayload,
    full_input: dict[str, Any],
    candidates_by_claim: dict[str, list[CitationCandidate]],
    question_id: str,
) -> tuple[DevV33RequiredClaimsEnvelope, dict[str, Any]]:
    inputs = {row["required_claim_id"]: row for row in full_input["required_claims"]}
    revised = []
    citations = {}
    traces = []
    for slot in raw.required_claim_results:
        body = slot.model_dump(mode="json")
        if slot.status.value != "answered":
            revised.append(body)
            citations[slot.required_claim_id] = []
            traces.append(
                {"required_claim_id": slot.required_claim_id, "fallback_action": slot.status.value}
            )
            continue
        candidates = candidates_by_claim[slot.required_claim_id]
        selection = select_citations(
            slot.claim_text or inputs[slot.required_claim_id]["required_claim_text"],
            candidates,
        )
        selected = list(selection.primary_citation_ids + selection.supporting_citation_ids)[:3]
        selected_candidates = [row for row in candidates if row.citation_id in selected]
        if selection.fallback_action == FallbackAction.UNSUPPORTED or not selected:
            body = {
                "required_claim_id": slot.required_claim_id,
                "status": "unsupported",
                "claim_text": None,
                "omission_reason": "Evidence gap: "
                + ", ".join(selection.uncovered_requirements or ("complete support unavailable",)),
            }
            selected = []
        elif selection.fallback_action == FallbackAction.ANSWERED_NARROWED:
            body["claim_text"] = selection.narrowed_claim_text
        revised.append(body)
        citations[slot.required_claim_id] = selected
        traces.append(
            {
                "required_claim_id": slot.required_claim_id,
                "original_claim_text": slot.claim_text,
                "final_claim_text": body["claim_text"],
                "fallback_action": selection.fallback_action.value,
                "citation_ids": selected,
                "primary_citation_ids": list(selection.primary_citation_ids),
                "supporting_citation_ids": list(selection.supporting_citation_ids),
                "uncovered_requirements": list(selection.uncovered_requirements),
                "removed_obligations": list(selection.removed_obligations),
                "numeric_validation": validate_numeric_evidence(
                    slot.claim_text or "", selected_candidates
                ).__dict__,
                "comparison_validation": validate_comparison_evidence(
                    slot.claim_text or "", selected_candidates
                ).__dict__,
                "decision_reasons": list(selection.decision_reasons),
            }
        )
    revised_payload = MinimalRequiredClaimsPayload.model_validate(
        {
            "answerable": raw.answerable,
            "required_claim_results": revised,
            "refusal_reason": raw.refusal_reason,
        }
    )
    envelope = bind_dev_v3_3_envelope(
        revised_payload,
        question_id=question_id,
        citation_ids_by_claim=citations,
    )
    return envelope, {"slots": traces}


TERMINAL_ARTIFACTS = (
    "raw-model-payload.json",
    "payload-validation.json",
    "local-envelope-binding.json",
    "obligation-analysis.json",
    "citation-selection-trace.json",
    "numeric-validation.json",
    "comparison-validation.json",
    "claim-fallback-trace.json",
)


def persist_terminal_sentinels(run_dir: Path, result: dict[str, Any]) -> None:
    sentinel = {
        "status": "not_available_due_to_terminal_failure",
        "failure_type": result["failure_type"],
        "failure_reason": result["failure_reason"],
        "raw_response_modified": False,
    }
    for name in TERMINAL_ARTIFACTS:
        path = run_dir / name
        if not path.exists():
            write_json(path, sentinel)


def run_one(
    question_id: str,
    settings: Settings,
    client: httpx.Client,
    freeze: dict[str, Any],
) -> dict[str, Any]:
    safe, full, registry, trace = safe_model_input(question_id)
    run_id = f"live-dev-v3-3-{question_id}-{uuid.uuid4().hex[:12]}"
    request_id = f"{run_id}:primary:1:{uuid.uuid4().hex[:12]}"
    run_dir = RUN_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    ledger = run_dir / "request-ledger.jsonl"
    ledger.write_text("", encoding="utf-8")
    candidates_by_claim, candidates = candidate_rows(full, registry, trace)
    system_prompt = dev_v3_3_system_prompt()
    user_prompt = json.dumps(safe, ensure_ascii=False)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    request_body = {
        "model": settings.llm_model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": safe["output_budget"]["calculated_max_output_tokens"],
        "stream": False,
        "enable_thinking": False,
        "response_format": {"type": "json_object"},
    }
    delivered = verify_safe_request(safe, messages, request_body, freeze)
    artifacts = {
        "required-claims-input.json": safe,
        "model-payload-schema.json": MinimalRequiredClaimsPayload.model_json_schema(),
        "local-envelope-schema.json": DevV33RequiredClaimsEnvelope.model_json_schema(),
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
    for name, value in artifacts.items():
        write_json(run_dir / name, value)
        event(ledger, f"{name.removesuffix('.json').replace('-', '_')}_persisted")
    (run_dir / "rendered-system-prompt.txt").write_text(system_prompt, encoding="utf-8")
    (run_dir / "rendered-user-prompt.txt").write_text(user_prompt, encoding="utf-8")
    metadata = {
        "run_id": run_id,
        "question_id": question_id,
        "evaluation_version": "evidence-qa-dev-v3.3",
        "request_id": request_id,
        "protocol_freeze_signature": freeze["protocol_freeze_signature"],
        **delivered,
        "required_claim_input_hash": canonical_hash(safe),
        "full_local_input_hash": canonical_hash(full),
        "citation_registry_hash": registry.registry_hash,
        "candidate_evidence_hash": canonical_hash(candidates),
        "prompt_version": DEV_V3_3_PROMPT_VERSION,
        "reranker_enabled": False,
        "retries": 0,
        "billing_mode": "explicit_free_provider",
        "gold_used_online": False,
        "human_labels_used_online": False,
        "internal_ids_in_model_prompt": False,
        "authorization_header_recorded": False,
        "api_key_recorded": False,
    }
    write_json(run_dir / "run-metadata.json", metadata)
    event(ledger, "request_id_allocated", request_id=request_id)
    event(ledger, "budget_reserved", reservation_id=request_id, reserved_tokens=24000)
    status = "provider_failed"
    failure_type = failure_reason = None
    usage: dict[str, Any] = {}
    raw_payload: dict[str, Any] = {}
    final_answer: dict[str, Any] = {}
    policy_trace: dict[str, Any] = {}
    request_sent = False
    started = time.perf_counter()
    try:
        persisted = json.loads(
            (run_dir / "delivered-request-metadata.json").read_text(encoding="utf-8")
        )
        if canonical_hash(request_body) != persisted["exact_delivered_request_body_hash"]:
            raise RuntimeError("DEV_V3_3_CONFIGURATION_INVALID: delivered body drift")
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
        event(ledger, "raw_response_persisted", request_id=request_id)
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
        event(ledger, "provider_usage_recorded", request_id=request_id, **usage)
        content = body["choices"][0]["message"]["content"]
        raw = parse_minimal_payload(
            content,
            expected_claim_ids=[row["required_claim_id"] for row in full["required_claims"]],
        )
        raw_payload = raw.model_dump(mode="json")
        write_json(run_dir / "raw-model-payload.json", raw_payload)
        write_json(
            run_dir / "payload-validation.json",
            {
                "json_valid": True,
                "schema_valid": True,
                "slot_valid": True,
                "payload_hash": canonical_hash(raw_payload),
            },
        )
        event(ledger, "model_payload_validated")
        final, policy_trace = apply_policy(raw, full, candidates_by_claim, question_id)
        final_answer = final.model_dump(mode="json")
        write_json(
            run_dir / "local-envelope-binding.json",
            {
                "model_payload_hash": canonical_hash(raw_payload),
                "envelope_hash": canonical_hash(final_answer),
                "question_id_source": "frozen_manifest",
                "prompt_version_source": "protocol_freeze",
                "citation_protocol_source": "protocol_freeze",
                "semantic_fields_modified_by_binding": False,
            },
        )
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
        DevV33RequiredClaimsEnvelope.model_validate(final_answer)
        status = "completed"
        event(ledger, "final_schema_validation_passed")
    except RequiredClaimValidationError as exc:
        status = "validation_failed"
        failure_type, failure_reason = exc.code, str(exc)
        event(ledger, "validation_failed", failure_type=exc.code)
    except httpx.HTTPError as exc:
        status = "provider_failed"
        failure_type, failure_reason = type(exc).__name__, str(exc)
        event(ledger, "provider_failed", failure_type=failure_type)
        write_json(
            run_dir / "raw-provider-response.json",
            {"response_received": False, "request_id": request_id, "failure_type": failure_type},
        )
        write_json(
            run_dir / "provider-response-envelope.json",
            {"response_received": False, "request_id": request_id, "usage": None},
        )
    except Exception as exc:
        status = "policy_failed"
        failure_type, failure_reason = type(exc).__name__, str(exc)
        event(ledger, "policy_failed", failure_type=failure_type)
    finally:
        ledger_events = [
            json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines() if line
        ]
        terminal_state = {
            "completed": RequestTerminalState.COMPLETED,
            "validation_failed": (
                RequestTerminalState.MALFORMED_JSON
                if failure_type == "malformed_json"
                else RequestTerminalState.SCHEMA_FAILED
            ),
            "provider_failed": RequestTerminalState.PROVIDER_FAILED,
            "policy_failed": RequestTerminalState.POLICY_FAILED,
        }[status]
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
        "final_answer": final_answer,
        "required_claim_count": len(full["required_claims"]),
        "raw_slot_count": len(raw_payload.get("required_claim_results", [])),
        "final_slot_count": len(final_answer.get("required_claim_results", [])),
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
    persist_terminal_sentinels(run_dir, result)
    return result


def main() -> None:
    command = args()
    check = preflight()
    if command.mode == "preflight":
        print(json.dumps(check, ensure_ascii=False))
        return
    if not HEALTH.exists():
        raise RuntimeError("DEV_V3_3_BLOCKED_BY_PROVIDER_HEALTH")
    health = json.loads(HEALTH.read_text(encoding="utf-8"))
    if not health.get("safe_to_start_batch"):
        raise RuntimeError("DEV_V3_3_BLOCKED_BY_PROVIDER_HEALTH")
    if list(RUN_ROOT.glob("live-dev-v3-3-*/final-result.json")):
        raise RuntimeError("duplicate formal Dev v3.3 run set")
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    settings = Settings()
    freeze = write_freeze()
    results = []
    provider_failures = consecutive_failures = 0
    with httpx.Client() as client:
        for question_id in DEV_IDS:
            row = run_one(question_id, settings, client, freeze)
            results.append(row)
            if row["status"] == "provider_failed":
                provider_failures += 1
                consecutive_failures += 1
            else:
                consecutive_failures = 0
            if provider_failures >= 2 or consecutive_failures >= 2:
                break
            if row["active_reserved_tokens"] != 0:
                break
    print(json.dumps({"runs": len(results), "results": results}, ensure_ascii=False))


if __name__ == "__main__":
    main()
