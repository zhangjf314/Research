"""One-shot schema-constrained Dev v3.1 evaluation runner."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from paper_research.config import Settings
from paper_research.generation.prompts import (
    QA_REQUIRED_CLAIMS_CITATION_ID_V3_1,
    qa_system_prompt,
)
from paper_research.generation.required_claim_output import (
    RequiredClaimsQAResponseV31,
    RequiredClaimValidationError,
    parse_and_validate_required_claim_response_v31,
    validate_no_free_triples,
)
from paper_research.providers.capabilities import siliconflow_qwen3_8b_stage13_5_snapshot
from paper_research.providers.response_envelope import ProviderResponseEnvelopeStore

try:
    from scripts.evidence_qa_dev_lib_v1 import DEV_IDS, canonical_hash
    from scripts.evidence_qa_dev_v3_1_lib import (
        CAPABILITY_HASH,
        HEALTH,
        PROMPT_HASH,
        RUN_ROOT,
        SCHEMA_HASH,
        SOURCE_MANIFEST_HASH,
        build_required_claim_input,
        write_manifest,
    )
except ModuleNotFoundError:
    from evidence_qa_dev_lib_v1 import DEV_IDS, canonical_hash  # type: ignore[no-redef]
    from evidence_qa_dev_v3_1_lib import (  # type: ignore[no-redef]
        CAPABILITY_HASH,
        HEALTH,
        PROMPT_HASH,
        RUN_ROOT,
        SCHEMA_HASH,
        SOURCE_MANIFEST_HASH,
        build_required_claim_input,
        write_manifest,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("dry-run", "live"), required=True)
    parser.add_argument("--no-summary", action="store_true")
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


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


def preflight() -> dict[str, Any]:
    settings = Settings()
    manifest = write_manifest()
    prompt = qa_system_prompt(QA_REQUIRED_CLAIMS_CITATION_ID_V3_1)
    schema = RequiredClaimsQAResponseV31.model_json_schema()
    capability = siliconflow_qwen3_8b_stage13_5_snapshot()
    observed = {
        "manifest_hash": manifest["manifest_hash"],
        "question_ids": manifest["question_ids"],
        "total_required_claims": manifest["total_required_claims"],
        "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest(),
        "schema_hash": canonical_hash(schema),
        "capability_hash": capability.snapshot_hash,
        "collection": settings.qdrant_collection,
        "embedding_model": settings.embedding_model,
        "embedding_dimensions": settings.embedding_dimensions,
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
        "temperature": settings.llm_temperature,
        "reranker_enabled": settings.rerank_enabled,
        "llm_max_retries": settings.llm_max_retries,
        "billing_mode": settings.llm_billing_mode,
    }
    expected = {
        "manifest_hash": SOURCE_MANIFEST_HASH,
        "question_ids": DEV_IDS,
        "total_required_claims": 27,
        "prompt_hash": PROMPT_HASH,
        "schema_hash": SCHEMA_HASH,
        "capability_hash": CAPABILITY_HASH,
        "collection": "papers_jina_eval34_v2__20260713152149",
        "embedding_model": "jina-embeddings-v5-text-small",
        "embedding_dimensions": 1024,
        "llm_provider": "siliconflow",
        "llm_model": "Qwen/Qwen3-8B",
        "temperature": 0,
        "reranker_enabled": False,
        "llm_max_retries": 0,
        "billing_mode": "free",
    }
    failures = [key for key, value in expected.items() if observed[key] != value]
    config = manifest["configuration"]
    protocol_checks = {
        "required_claim_protocol": config["required_claim_protocol"]
        == "required-claim-slots-v1.1",
        "citation_protocol": config["citation_protocol"] == "citation-id-v2",
        "transport": config["transport"]
        == "provider_json_object_plus_strict_local_schema",
        "normalization": config["formal_normalization_policy"] == "raw_schema_passed_only",
        "response_format": config["response_format"] == "json_object",
        "no_json_schema": True,
        "no_tools_or_functions": True,
    }
    failures.extend(key for key, passed in protocol_checks.items() if not passed)
    if failures:
        raise RuntimeError(f"DEV_V3_1_CONFIGURATION_INVALID: {sorted(failures)}")
    return {"observed": observed, "protocol_checks": protocol_checks}


def assert_live_authorized() -> dict[str, Any]:
    if os.getenv("DEV_V3_1_LIVE_AUTHORIZED") != "true":
        raise RuntimeError("DEV_V3_1_LIVE_NOT_AUTHORIZED")
    if not HEALTH.exists():
        raise RuntimeError("DEV_V3_1_BLOCKED_BY_PROVIDER_HEALTH")
    health = json.loads(HEALTH.read_text(encoding="utf-8"))
    if not health.get("safe_to_start_batch"):
        raise RuntimeError("DEV_V3_1_BLOCKED_BY_PROVIDER_HEALTH")
    if list(RUN_ROOT.glob("live-dev-v3-1-*/result.json")):
        raise RuntimeError("duplicate formal Dev v3.1 run set")
    return health


def classify_failure(error: RequiredClaimValidationError) -> str:
    allowed = {
        "malformed_json",
        "valid_json_wrong_schema",
        "question_wrapper_rejected",
        "claim_map_rejected",
        "legacy_schema_rejected",
        "missing_slot",
        "duplicate_slot",
        "extra_slot",
        "unknown_claim_id",
        "unknown_citation_id",
        "cross_claim_citation",
        "status_citation_inconsistency",
        "answerability_protocol_failure",
    }
    return error.code if error.code in allowed else error.code


def validate_raw_schema(content: str, expected_question_id: str, claim_ids: list[str]) -> None:
    """Validate the frozen raw envelope without applying business-slot semantics."""
    try:
        raw = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RequiredClaimValidationError("malformed_json", str(exc)) from exc
    if not isinstance(raw, dict):
        raise RequiredClaimValidationError("valid_json_wrong_schema", "top level is not an object")
    if set(raw) == {expected_question_id} and isinstance(raw[expected_question_id], dict):
        raise RequiredClaimValidationError("question_wrapper_rejected", expected_question_id)
    if "claims" in raw:
        raise RequiredClaimValidationError("legacy_schema_rejected", "legacy claims field")
    if "required_claim_results" not in raw and set(raw) & set(claim_ids):
        raise RequiredClaimValidationError("claim_map_rejected", "claim IDs used as top-level keys")
    try:
        validate_no_free_triples(raw)
        RequiredClaimsQAResponseV31.model_validate(raw)
    except Exception as exc:
        raise RequiredClaimValidationError("valid_json_wrong_schema", str(exc)) from exc


def ensure_failure_response_artifacts(
    run_dir: Path, request_id: str, error: Exception
) -> None:
    """Create failure sentinels only when no provider evidence was persisted."""
    sentinel = {
        "response_received": False,
        "request_id": request_id,
        "failure_type": type(error).__name__,
    }
    raw_path = run_dir / "raw-provider-response.json"
    response_received = raw_path.exists()
    if not raw_path.exists():
        write_json(raw_path, sentinel)
    envelope_path = run_dir / "provider-response-envelope.json"
    if not envelope_path.exists():
        write_json(
            envelope_path,
            {
                "schema_version": "provider-response-envelope-v1-failure",
                **sentinel,
                "response_received": response_received,
                "usage": None,
                "usage_status": "unknown_after_send_reserved_conservative",
                "parse_status": "not_started",
            },
        )


def persist_pre_request_artifacts(
    question_id: str,
    run_dir: Path,
    request_id: str,
    payload: dict[str, Any],
    registry: Any,
    trace: dict[str, Any],
    settings: Settings,
) -> tuple[Path, dict[str, Any]]:
    ledger = run_dir / "request-ledger.jsonl"
    ledger.write_text("", encoding="utf-8")
    event(ledger, "manifest_validated", manifest_hash=SOURCE_MANIFEST_HASH)
    write_json(run_dir / "required-claims-input.json", payload)
    event(ledger, "required_claim_input_persisted")
    payload_hash = canonical_hash(payload)
    event(ledger, "required_claim_input_hash_recorded", value=payload_hash)
    write_json(run_dir / "citation-registry.json", registry.model_dump(mode="json"))
    event(ledger, "citation_registry_persisted")
    event(ledger, "citation_registry_hash_recorded", value=registry.registry_hash)
    schema = RequiredClaimsQAResponseV31.model_json_schema()
    write_json(run_dir / "exact-json-schema.json", schema)
    event(ledger, "exact_schema_persisted")
    event(ledger, "schema_hash_recorded", value=SCHEMA_HASH)
    system_prompt = qa_system_prompt(QA_REQUIRED_CLAIMS_CITATION_ID_V3_1)
    user_prompt = json.dumps(payload, ensure_ascii=False)
    (run_dir / "rendered-system-prompt.txt").write_text(system_prompt, encoding="utf-8")
    (run_dir / "rendered-user-prompt.txt").write_text(user_prompt, encoding="utf-8")
    event(ledger, "prompt_rendered")
    event(ledger, "prompt_hash_recorded", value=PROMPT_HASH)
    capability = siliconflow_qwen3_8b_stage13_5_snapshot()
    write_json(
        run_dir / "provider-capability-snapshot.json",
        {"snapshot": capability.model_dump(mode="json"), "snapshot_hash": capability.snapshot_hash},
    )
    event(ledger, "provider_capability_snapshot_persisted")
    response_format = {
        "response_format": {"type": "json_object"},
        "json_schema_sent": False,
        "tools_sent": False,
        "functions_sent": False,
    }
    write_json(run_dir / "response-format-parameters.json", response_format)
    event(ledger, "response_format_parameters_persisted")
    prompt_metadata = {
        "prompt_version": QA_REQUIRED_CLAIMS_CITATION_ID_V3_1,
        "prompt_hash": PROMPT_HASH,
        "schema_hash": SCHEMA_HASH,
        "provider_capability_snapshot_hash": CAPABILITY_HASH,
        "formal_normalization_policy": "raw_schema_passed_only",
    }
    write_json(run_dir / "prompt-metadata.json", prompt_metadata)
    write_json(run_dir / "retrieval-trace.json", trace)
    write_json(run_dir / "context-trace.json", trace)
    event(ledger, "request_id_allocated", request_id=request_id)
    event(ledger, "budget_reserved", request_id=request_id, reserved_tokens=24000)
    metadata = {
        "run_id": run_dir.name,
        "question_id": question_id,
        "evaluation_version": "evidence-qa-dev-v3.1",
        "mode": "live",
        "manifest_hash": SOURCE_MANIFEST_HASH,
        "required_claim_input_hash": payload_hash,
        "citation_registry_hash": registry.registry_hash,
        "schema_hash": SCHEMA_HASH,
        "prompt_hash": PROMPT_HASH,
        "provider_capability_snapshot_hash": CAPABILITY_HASH,
        "request_id": request_id,
        "collection": settings.qdrant_collection,
        "embedding_model": settings.embedding_model,
        "reranker_enabled": False,
        "retries": 0,
        "json_correction_retries": 0,
        "citation_correction_retries": 0,
        "billing_mode": "free",
        "historical_active_reservations_retained": 60000,
        "api_key_recorded": False,
        "authorization_header_recorded": False,
        "gold_evidence_used_for_allocation": False,
        "oracle_used": False,
        "human_pilot_used": False,
        "formal_normalization_used": False,
    }
    write_json(run_dir / "run-metadata.json", metadata)
    event(
        ledger,
        "request_prepared",
        request_id=request_id,
        response_format="json_object",
        retries=0,
    )
    return ledger, {
        "payload_hash": payload_hash,
        "registry_hash": registry.registry_hash,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "response_format": response_format,
    }


def write_result(run_dir: Path, result: dict[str, Any]) -> None:
    write_json(run_dir / "result.json", result)
    flat = {
        key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
        for key, value in result.items()
    }
    with (run_dir / "result.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat))
        writer.writeheader()
        writer.writerow(flat)


def run_one(question_id: str, settings: Settings, client: httpx.Client) -> dict[str, Any]:
    payload, registry, _contexts, trace = build_required_claim_input(question_id)
    run_id = f"live-dev-v3-1-{question_id}-{uuid.uuid4().hex[:12]}"
    request_id = f"{run_id}:primary:1:{uuid.uuid4().hex[:12]}"
    run_dir = RUN_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    ledger, prepared = persist_pre_request_artifacts(
        question_id,
        run_dir,
        request_id,
        payload,
        registry,
        trace,
        settings,
    )
    event(ledger, "request_started", request_id=request_id)
    started = time.perf_counter()
    status = "provider_failed"
    failure_type: str | None = None
    failure_reason: str | None = None
    answer: dict[str, Any] = {}
    usage: dict[str, Any] = {}
    active_reserved_tokens = 24000
    raw_json_valid = False
    raw_schema_valid = False
    slot_validation_success = False
    provider_completed = 0
    try:
        response = client.post(
            f"{(settings.llm_base_url or '').rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.llm_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": settings.llm_model,
                "messages": [
                    {"role": "system", "content": prepared["system_prompt"]},
                    {"role": "user", "content": prepared["user_prompt"]},
                ],
                "temperature": 0,
                "max_tokens": payload["output_budget"]["calculated_max_output_tokens"],
                "stream": False,
                "enable_thinking": False,
                "response_format": {"type": "json_object"},
            },
            timeout=httpx.Timeout(connect=15, read=180, write=30, pool=15),
        )
        response.raise_for_status()
        provider_completed = 1
        store = ProviderResponseEnvelopeStore(run_dir, ledger)
        envelope = store.record_received(
            request_id=request_id,
            provider="siliconflow",
            model=settings.llm_model,
            raw_body=response.content,
        )
        usage = envelope.usage.model_dump()
        active_reserved_tokens = 0
        envelope = store.parsing_started(envelope)
        content = envelope.parsed_provider_payload["choices"][0]["message"]["content"]
        event(ledger, "raw_schema_validation_started", request_id=request_id)
        claim_ids = [claim["required_claim_id"] for claim in payload["required_claims"]]
        try:
            json.loads(content)
            raw_json_valid = True
        except json.JSONDecodeError:
            raw_json_valid = False
        validate_raw_schema(content, question_id, claim_ids)
        raw_schema_valid = True
        event(ledger, "raw_schema_passed", request_id=request_id)
        allowed = {
            claim["required_claim_id"]: set(claim["allowed_citation_ids"])
            for claim in payload["required_claims"]
        }
        try:
            output = parse_and_validate_required_claim_response_v31(
                content,
                expected_question_id=question_id,
                expected_claim_ids=claim_ids,
                registry=registry,
                allowed_by_claim=allowed,
                expected_registry_hash=registry.registry_hash,
            )
            slot_validation_success = True
            answer = output.model_dump(mode="json")
            store.parsed(envelope)
            event(ledger, "required_claim_validation", request_id=request_id, passed=True)
            event(ledger, "citation_validation", request_id=request_id, passed=True)
            event(ledger, "completed", request_id=request_id, active_reserved_tokens=0)
            status = "completed"
        except RequiredClaimValidationError as exc:
            store.post_processing_failed(envelope, exc)
            failure_type = classify_failure(exc)
            failure_reason = str(exc)
            event(
                ledger,
                "validation_failed",
                request_id=request_id,
                failure_type=failure_type,
                active_reserved_tokens=0,
            )
            status = "validation_failed"
    except RequiredClaimValidationError as exc:
        store.post_processing_failed(envelope, exc)
        failure_type = classify_failure(exc)
        failure_reason = str(exc)
        active_reserved_tokens = 0 if provider_completed else active_reserved_tokens
        event(
            ledger,
            "validation_failed",
            request_id=request_id,
            failure_type=failure_type,
            active_reserved_tokens=active_reserved_tokens,
        )
        status = "validation_failed"
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        failure_type = "valid_json_wrong_schema" if provider_completed else "provider_failed"
        failure_reason = f"{type(exc).__name__}: {exc}"
        active_reserved_tokens = 0 if provider_completed else active_reserved_tokens
        event(
            ledger,
            "validation_failed" if provider_completed else "request_failed",
            request_id=request_id,
            failure_type=failure_type,
            active_reserved_tokens=active_reserved_tokens,
        )
        ensure_failure_response_artifacts(run_dir, request_id, exc)
        status = "validation_failed" if provider_completed else "provider_failed"
    except httpx.HTTPError as exc:
        failure_type = "provider_failed"
        failure_reason = f"{type(exc).__name__}: {exc}"
        event(
            ledger,
            "request_failed",
            request_id=request_id,
            failure_type=type(exc).__name__,
            active_reserved_tokens=24000,
        )
        ensure_failure_response_artifacts(run_dir, request_id, exc)
    elapsed = time.perf_counter() - started
    results = answer.get("required_claim_results", [])
    result = {
        "run_id": run_id,
        "question_id": question_id,
        "status": status,
        "failure_type": failure_type,
        "failure_reason": failure_reason,
        "answer": answer,
        "required_claim_count": len(payload["required_claims"]),
        "slot_count": len(results),
        "answered_slots": sum(row.get("status") == "answered" for row in results),
        "unsupported_slots": sum(row.get("status") == "unsupported" for row in results),
        "not_applicable_slots": sum(row.get("status") == "not_applicable" for row in results),
        "silent_omission_count": max(0, len(payload["required_claims"]) - len(results)),
        "request_attempt_count": 1,
        "provider_completed_request_count": provider_completed,
        "provider_failure_count": int(failure_type == "provider_failed"),
        "usage_record_count": int(bool(usage)),
        "usage": usage,
        "active_reserved_tokens": active_reserved_tokens,
        "elapsed_seconds": round(elapsed, 6),
        "raw_json_valid": raw_json_valid,
        "raw_schema_valid": raw_schema_valid,
        "slot_validation_success": slot_validation_success,
        "required_claim_input_hash": prepared["payload_hash"],
        "required_claim_input_hash_valid": canonical_hash(
            json.loads((run_dir / "required-claims-input.json").read_text(encoding="utf-8"))
        )
        == prepared["payload_hash"],
        "citation_registry_hash": prepared["registry_hash"],
        "citation_registry_hash_valid": registry.registry_hash == prepared["registry_hash"],
        "schema_hash_valid": canonical_hash(
            json.loads((run_dir / "exact-json-schema.json").read_text(encoding="utf-8"))
        )
        == SCHEMA_HASH,
        "prompt_hash_valid": hashlib.sha256(
            (run_dir / "rendered-system-prompt.txt").read_bytes()
        ).hexdigest()
        == PROMPT_HASH,
        "capability_snapshot_hash_valid": json.loads(
            (run_dir / "provider-capability-snapshot.json").read_text(encoding="utf-8")
        )["snapshot_hash"]
        == CAPABILITY_HASH,
        "response_format_sent": True,
        "formal_normalization_used": False,
        "monetary_cost_usd": "0",
        "cost_basis": "explicit_free_provider",
        "reranker_called": False,
        "template_fallback": False,
        "retries": 0,
        "live_llm_called": True,
    }
    write_result(run_dir, result)
    return result


def main() -> int:
    args = parse_args()
    try:
        audit = preflight()
    except RuntimeError as exc:
        print(str(exc))
        return 2
    if args.mode == "dry-run":
        print(
            json.dumps(
                {
                    "status": "DEV_V3_1_DRY_RUN_PASSED",
                    "questions": 10,
                    "required_claims": 27,
                    "audit": audit,
                    "live_llm_called": False,
                }
            )
        )
        return 0
    try:
        assert_live_authorized()
    except RuntimeError as exc:
        print(str(exc))
        return 2
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    settings = Settings()
    rows: list[dict[str, Any]] = []
    provider_failures = 0
    consecutive_provider_failures = 0
    started = time.perf_counter()
    transport = httpx.HTTPTransport(retries=0)
    with httpx.Client(transport=transport) as client:
        for question_id in DEV_IDS:
            if provider_failures >= 2 or consecutive_provider_failures >= 2:
                break
            if len(rows) >= 10 or sum(
                int(row.get("usage", {}).get("total_tokens", 0)) for row in rows
            ) >= 240000:
                break
            row = run_one(question_id, settings, client)
            rows.append(row)
            if row["failure_type"] == "provider_failed":
                provider_failures += 1
                consecutive_provider_failures += 1
            else:
                consecutive_provider_failures = 0
            if time.perf_counter() - started > 1800:
                break
    print(
        json.dumps(
            {
                "status": "DEV_V3_1_LIVE_BATCH_FINISHED",
                "runs": len(rows),
                "provider_failures": provider_failures,
                "early_stop": len(rows) < 10,
                "requests": sum(row["request_attempt_count"] for row in rows),
                "tokens": sum(
                    int(row.get("usage", {}).get("total_tokens", 0)) for row in rows
                ),
                "elapsed_seconds": round(time.perf_counter() - started, 6),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
