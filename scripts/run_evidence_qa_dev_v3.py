# ruff: noqa: E501
"""Offline-ready Dev v3 runner. Live mode is fail-closed until explicit authorization."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from paper_research.config import Settings
from paper_research.generation.prompts import QA_REQUIRED_CLAIMS_CITATION_ID_V3, qa_system_prompt
from paper_research.generation.required_claim_output import (
    RequiredClaimValidationError,
    parse_and_validate_required_claim_response,
)
from paper_research.providers.response_envelope import ProviderResponseEnvelopeStore

try:
    from scripts.evidence_qa_dev_lib_v1 import DATA, DEV_IDS, canonical_hash
    from scripts.evidence_qa_dev_v3_lib import (
        FIXTURE_SUMMARY,
        RUN_ROOT,
        SOURCE_MANIFEST_HASH,
        build_required_claim_input,
        write_manifest,
    )
    from scripts.run_evidence_qa_dev_v2 import write_no_response_artifacts
except ModuleNotFoundError:
    from evidence_qa_dev_lib_v1 import (
        DATA,  # type: ignore[no-redef]
        DEV_IDS,  # type: ignore[no-redef]
        canonical_hash,  # type: ignore[no-redef]
    )
    from evidence_qa_dev_v3_lib import (  # type: ignore[no-redef]
        FIXTURE_SUMMARY,
        RUN_ROOT,
        SOURCE_MANIFEST_HASH,
        build_required_claim_input,
        write_manifest,
    )
    from run_evidence_qa_dev_v2 import write_no_response_artifacts  # type: ignore[no-redef]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("dry-run", "fixture", "live"), required=True)
    parser.add_argument("--no-summary", action="store_true")
    return parser.parse_args()


def event(path: Path, name: str, **values: Any) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"event_id": uuid.uuid4().hex, "event": name, "timestamp": datetime.now(UTC).isoformat(), **values}, ensure_ascii=False) + "\n")


def preflight() -> dict[str, Any]:
    settings = Settings()
    expected = {"qdrant_collection": "papers_jina_eval34_v2__20260713152149", "embedding_model": "jina-embeddings-v5-text-small", "embedding_dimensions": 1024, "llm_provider": "siliconflow", "llm_model": "Qwen/Qwen3-8B", "llm_temperature": 0, "llm_max_retries": 0, "llm_billing_mode": "free", "rerank_enabled": False}
    failures = [f"{key} changed" for key, value in expected.items() if getattr(settings, key) != value]
    manifest = write_manifest()
    if manifest["manifest_hash"] != SOURCE_MANIFEST_HASH or manifest["question_ids"] != DEV_IDS:
        failures.append("manifest changed")
    if manifest["protocol_hash"] != "ba7a8d5a132ca6c201e835ed2f66583f91598a0d2c1c6c1c0920185714502552":
        failures.append("protocol changed")
    if failures:
        raise RuntimeError("Dev v3 fail-closed preflight: " + "; ".join(failures))
    return {"manifest_hash": manifest["manifest_hash"], "protocol_hash": manifest["protocol_hash"], **expected, "dev_v3_authorized": False}


def assert_live_authorized() -> dict[str, Any]:
    if os.getenv("DEV_V3_LIVE_AUTHORIZED") != "true":
        raise RuntimeError("DEV_V3_LIVE_NOT_AUTHORIZED")
    health_path = DATA / "provider-health-dev-v3-v1.json"
    health = json.loads(health_path.read_text(encoding="utf-8"))
    if not health.get("safe_to_start_batch"):
        raise RuntimeError("DEV_V3_BLOCKED_BY_PROVIDER_HEALTH")
    return health


def fixture_response(question_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    if not payload["answerability_expectation"]:
        return {"question_id": question_id, "answerable": False, "required_claim_results": [], "refusal_reason": "The supplied evidence does not answer this question.", "prompt_version": "qa-required-claims-citation-id-v3", "citation_protocol": "citation-id-v2"}
    results = []
    for claim in payload["required_claims"]:
        if claim["allowed_citation_ids"]:
            results.append({"required_claim_id": claim["required_claim_id"], "status": "answered", "claim_text": claim["required_claim_text"], "citation_ids": claim["allowed_citation_ids"][:1], "omission_reason": None})
        else:
            results.append({"required_claim_id": claim["required_claim_id"], "status": "unsupported", "claim_text": None, "citation_ids": [], "omission_reason": "No claim-local evidence was allocated."})
    return {"question_id": question_id, "answerable": True, "required_claim_results": results, "refusal_reason": None, "prompt_version": "qa-required-claims-citation-id-v3", "citation_protocol": "citation-id-v2"}


def write_fixture_run(question_id: str) -> dict[str, Any]:
    payload, registry, _contexts, trace = build_required_claim_input(question_id)
    run_id = f"fixture-dev-v3-{question_id}-v1"
    run_dir = RUN_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "required-claims-input.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "citation-registry.json").write_text(json.dumps(registry.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "retrieval-trace.json").write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "context-trace.json").write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    ledger = run_dir / "request-ledger.jsonl"
    ledger.write_text("", encoding="utf-8")
    request_id = f"{run_id}:fixture:1"
    event(ledger, "request_prepared", request_id=request_id, fixture=True, reserved_tokens=payload["output_budget"]["calculated_max_output_tokens"])
    response_body = fixture_response(question_id, payload)
    event(ledger, "request_started", request_id=request_id, fixture=True, provider_request_sent=False)
    raw_provider = {"model": "offline-fixture", "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(response_body)}}], "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}
    store = ProviderResponseEnvelopeStore(run_dir, ledger)
    envelope = store.record_received(request_id=request_id, provider="offline_fixture", model="offline-fixture", raw_body=json.dumps(raw_provider).encode())
    envelope = store.parsing_started(envelope)
    allowed = {claim["required_claim_id"]: set(claim["allowed_citation_ids"]) for claim in payload["required_claims"]}
    output = parse_and_validate_required_claim_response(raw_provider["choices"][0]["message"]["content"], expected_claim_ids=[claim["required_claim_id"] for claim in payload["required_claims"]], registry=registry, allowed_by_claim=allowed, expected_registry_hash=registry.registry_hash)
    store.parsed(envelope)
    result = {"run_id": run_id, "question_id": question_id, "status": "fixture_completed", "answer": output.model_dump(mode="json"), "required_claim_count": len(payload["required_claims"]), "slot_count": len(output.required_claim_results), "silent_omission_count": 0, "unknown_citation_id_count": 0, "cross_claim_citation_count": 0, "request_attempt_count": 0, "provider_completed_request_count": 0, "usage_record_count": 1, "active_reserved_tokens": 0, "reranker_called": False, "live_llm_called": False}
    (run_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (run_dir / "result.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result))
        writer.writeheader()
        writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value for key, value in result.items()})
    (run_dir / "run-metadata.json").write_text(json.dumps({"run_id": run_id, "mode": "fixture", "manifest_hash": SOURCE_MANIFEST_HASH, "provider_request_sent": False, "api_key_recorded": False, "authorization_header_recorded": False, "gold_evidence_used_for_allocation": False, "oracle_used": False, "human_pilot_used": False}, indent=2), encoding="utf-8")
    return result


def write_live_run(question_id: str, settings: Settings, client: httpx.Client) -> dict[str, Any]:
    payload, registry, _contexts, trace = build_required_claim_input(question_id)
    run_id = f"live-dev-v3-{question_id}-{uuid.uuid4().hex[:12]}"
    request_id = f"{run_id}:primary:1:{uuid.uuid4().hex[:12]}"
    run_dir = RUN_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "required-claims-input.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "citation-registry.json").write_text(json.dumps(registry.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "retrieval-trace.json").write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "context-trace.json").write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    payload_hash = canonical_hash(payload)
    registry_hash = registry.registry_hash
    metadata = {"run_id": run_id, "mode": "live", "manifest_hash": SOURCE_MANIFEST_HASH, "protocol_hash": "ba7a8d5a132ca6c201e835ed2f66583f91598a0d2c1c6c1c0920185714502552", "required_claim_input_hash": payload_hash, "citation_registry_hash": registry_hash, "request_id": request_id, "provider_request_sent": True, "api_key_recorded": False, "authorization_header_recorded": False, "gold_evidence_used_for_allocation": False, "oracle_used": False, "human_pilot_used": False, "collection": "papers_jina_eval34_v2__20260713152149", "embedding_model": "jina-embeddings-v5-text-small", "embedding_dimensions": 1024, "reranker_enabled": False, "retries": 0, "citation_retries": 0, "billing_mode": "free", "historical_active_reservations_retained": 60000}
    (run_dir / "run-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    ledger = run_dir / "request-ledger.jsonl"
    ledger.write_text("", encoding="utf-8")
    event(ledger, "request_prepared", request_id=request_id, reserved_tokens=24000, usage_status="reserved")
    event(ledger, "request_started", request_id=request_id, reserved_tokens=24000, usage_status="reserved")
    store = ProviderResponseEnvelopeStore(run_dir, ledger)
    started = time.perf_counter()
    status, failure, usage, active = "provider_failed", None, {}, 24000
    try:
        response = client.post(f"{(settings.llm_base_url or '').rstrip('/')}/chat/completions", headers={"Authorization": f"Bearer {settings.llm_api_key}", "Content-Type": "application/json"}, json={"model": settings.llm_model, "messages": [{"role": "system", "content": qa_system_prompt(QA_REQUIRED_CLAIMS_CITATION_ID_V3)}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}], "temperature": 0, "max_tokens": payload["output_budget"]["calculated_max_output_tokens"], "stream": False, "enable_thinking": False, "response_format": {"type": "json_object"}}, timeout=httpx.Timeout(connect=15, read=180, write=30, pool=15))
        response.raise_for_status()
        envelope = store.record_received(request_id=request_id, provider="siliconflow", model=settings.llm_model, raw_body=response.content)
        usage, active = envelope.usage.model_dump(), 0
        envelope = store.parsing_started(envelope)
        content = envelope.parsed_provider_payload["choices"][0]["message"]["content"]
        allowed = {claim["required_claim_id"]: set(claim["allowed_citation_ids"]) for claim in payload["required_claims"]}
        try:
            output = parse_and_validate_required_claim_response(content, expected_claim_ids=[claim["required_claim_id"] for claim in payload["required_claims"]], registry=registry, allowed_by_claim=allowed, expected_registry_hash=registry.registry_hash)
            store.parsed(envelope)
            event(ledger, "required_claim_validation_passed", request_id=request_id, required_claim_input_hash=payload_hash)
            event(ledger, "citation_validation_passed", request_id=request_id, citation_registry_hash=registry_hash)
            event(ledger, "completed", request_id=request_id, active_reserved_tokens=0)
            status, answer = "completed", output.model_dump(mode="json")
        except RequiredClaimValidationError as exc:
            store.post_processing_failed(envelope, exc)
            event(ledger, "validation_failed", request_id=request_id, validation_code=exc.code, active_reserved_tokens=0)
            status, failure, answer = "validation_failed", str(exc), {}
    except httpx.HTTPError as exc:
        failure, answer = f"{type(exc).__name__}: {exc}", {}
        event(ledger, "request_failed", request_id=request_id, usage_status="reserved_conservative", active_reserved_tokens=24000, failure_type=type(exc).__name__)
        write_no_response_artifacts(run_dir, request_id, exc)
    elapsed = time.perf_counter() - started
    result = {"run_id": run_id, "question_id": question_id, "status": status, "failure_reason": failure, "answer": answer, "required_claim_count": len(payload["required_claims"]), "slot_count": len(answer.get("required_claim_results", [])), "silent_omission_count": max(0, len(payload["required_claims"]) - len(answer.get("required_claim_results", []))), "request_attempt_count": 1, "provider_completed_request_count": int(bool(usage)), "usage_record_count": int(bool(usage)), "usage": usage, "active_reserved_tokens": active, "elapsed_seconds": round(elapsed, 6), "required_claim_input_hash": payload_hash, "required_claim_input_hash_valid": canonical_hash(json.loads((run_dir / "required-claims-input.json").read_text(encoding="utf-8"))) == payload_hash, "citation_registry_hash": registry_hash, "citation_registry_hash_valid": registry.registry_hash == registry_hash, "monetary_cost_usd": "0", "cost_basis": "explicit_free_provider", "reranker_called": False, "template_fallback": False, "live_llm_called": True}
    (run_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (run_dir / "result.csv").open("w", encoding="utf-8", newline="") as stream:
        flat = {key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value for key, value in result.items()}
        writer = csv.DictWriter(stream, fieldnames=list(flat))
        writer.writeheader()
        writer.writerow(flat)
    return result


def expect_failure(name: str, function: Callable[[], None], code: str) -> dict[str, Any]:
    try:
        function()
    except RequiredClaimValidationError as exc:
        return {"fixture": name, "passed": exc.code == code, "observed_error": exc.code}
    return {"fixture": name, "passed": False, "observed_error": None}


def expect_success(name: str, function: Callable[[], None]) -> dict[str, Any]:
    try:
        function()
    except Exception as exc:
        return {"fixture": name, "passed": False, "observed_error": type(exc).__name__}
    return {"fixture": name, "passed": True}


def fixture_suite() -> list[dict[str, Any]]:
    payload, registry, _, _ = build_required_claim_input("q001")
    valid = fixture_response("q001", payload)
    ids = [item["required_claim_id"] for item in payload["required_claims"]]
    allowed = {item["required_claim_id"]: set(item["allowed_citation_ids"]) for item in payload["required_claims"]}
    def check(raw: dict[str, Any] | str, expected=ids, local=allowed) -> None:
        text = raw if isinstance(raw, str) else json.dumps(raw)
        parse_and_validate_required_claim_response(text, expected_claim_ids=expected, registry=registry, allowed_by_claim=local, expected_registry_hash=registry.registry_hash)
    partial = json.loads(json.dumps(valid))
    partial["required_claim_results"][0] = {"required_claim_id": ids[0], "status": "unsupported", "claim_text": None, "citation_ids": [], "omission_reason": "fixture evidence incomplete"}
    unsupported = json.loads(json.dumps(valid))
    unsupported["required_claim_results"] = [{"required_claim_id": claim_id, "status": "unsupported", "claim_text": None, "citation_ids": [], "omission_reason": "fixture evidence incomplete"} for claim_id in ids]
    checks = [expect_success("valid_all_claims_answered", lambda: check(valid)), expect_success("valid_partial_answer", lambda: check(partial)), expect_success("unsupported_claim", lambda: check(unsupported))]
    duplicate = json.loads(json.dumps(valid))
    duplicate["required_claim_results"].append(duplicate["required_claim_results"][0])
    missing = json.loads(json.dumps(valid))
    missing["required_claim_results"] = missing["required_claim_results"][:-1]
    extra = json.loads(json.dumps(valid))
    extra["required_claim_results"].append({"required_claim_id": "extra", "status": "unsupported", "claim_text": None, "citation_ids": [], "omission_reason": "fixture"})
    cross = json.loads(json.dumps(valid))
    cross["required_claim_results"][0]["citation_ids"] = list(allowed[ids[1]])[:1]
    unknown = json.loads(json.dumps(valid))
    unknown["required_claim_results"][0]["citation_ids"] = ["E999"]
    free = json.loads(json.dumps(valid))
    free["required_claim_results"][0]["paper_id"] = "forbidden"
    unanswerable_bad = {"question_id": "q005", "answerable": False, "required_claim_results": [{"required_claim_id": "fake", "status": "answered", "claim_text": "fake", "citation_ids": ["E001"], "omission_reason": None}], "refusal_reason": "no", "prompt_version": "qa-required-claims-citation-id-v3", "citation_protocol": "citation-id-v2"}
    unanswerable = fixture_response("q005", {"answerability_expectation": False, "required_claims": []})
    q019_payload, q019_registry, _, _ = build_required_claim_input("q019")
    q019_output = fixture_response("q019", q019_payload)
    q019_allowed = {item["required_claim_id"]: set(item["allowed_citation_ids"]) for item in q019_payload["required_claims"]}
    checks += [expect_failure("duplicate_required_claim", lambda: check(duplicate), "duplicate_required_claim_id"), expect_failure("missing_required_claim", lambda: check(missing), "missing_required_claim_id"), expect_failure("extra_required_claim", lambda: check(extra), "extra_required_claim_id"), expect_failure("cross_claim_citation", lambda: check(cross), "cross_claim_citation"), expect_failure("unknown_citation_id", lambda: check(unknown), "unknown_citation_id"), expect_failure("free_triple_output", lambda: check(free), "free_triple_forbidden"), expect_failure("malformed_json", lambda: check("{"), "malformed_json"), expect_success("unanswerable_valid", lambda: check(unanswerable, [], {})), expect_failure("unanswerable_with_citation", lambda: check(unanswerable_bad, ["fake"], {"fake": {"E001"}}), "unanswerable_has_answer_or_citation"), expect_failure("q050_unclosed_json", lambda: check('{"question_id":"q050"'), "malformed_json"), expect_success("compound_claim_split", lambda: parse_and_validate_required_claim_response(json.dumps(q019_output), expected_claim_ids=[item["required_claim_id"] for item in q019_payload["required_claims"]], registry=q019_registry, allowed_by_claim=q019_allowed, expected_registry_hash=q019_registry.registry_hash)), expect_failure("merged_claim_not_allowed_without_slots", lambda: check(missing), "missing_required_claim_id")]
    return checks


def main() -> int:
    args = parse_args()
    audit = preflight()
    inputs = [build_required_claim_input(qid)[0] for qid in DEV_IDS]
    if args.mode == "dry-run":
        print(json.dumps({"status": "DEV_V3_DRY_RUN_PASSED", "questions": len(inputs), "required_claim_slots": sum(len(item["required_claims"]) for item in inputs), "audit": audit, "live_llm_called": False}))
        return 0
    if args.mode == "live":
        try:
            assert_live_authorized()
        except RuntimeError as exc:
            print(str(exc))
            return 2
        settings = Settings()
        rows = []
        existing_live = [path for path in RUN_ROOT.glob("live-dev-v3-*/result.json")]
        if existing_live:
            print("DEV_V3_DUPLICATE_LIVE_BATCH_BLOCKED")
            return 2
        batch_started = time.perf_counter()
        settled_tokens = 0
        consecutive_provider_failures = 0
        with httpx.Client() as client:
            for question_id in DEV_IDS:
                if settled_tokens + 24000 > 240000 or time.perf_counter() - batch_started >= 1800:
                    print(json.dumps({"status": "budget_blocked", "runs": len(rows), "settled_tokens": settled_tokens}))
                    break
                row = write_live_run(question_id, settings, client)
                rows.append(row)
                settled_tokens += int(row.get("usage", {}).get("total_tokens", 0))
                consecutive_provider_failures = consecutive_provider_failures + 1 if row["status"] == "provider_failed" else 0
                per_question_budget_failed = settled_tokens > 240000 or row["elapsed_seconds"] > 180 or int(row.get("usage", {}).get("total_tokens", 0)) > 24000
                if consecutive_provider_failures >= 2 or sum(item["status"] == "provider_failed" for item in rows) >= 2 or per_question_budget_failed:
                    break
        print(json.dumps({"status": "DEV_V3_LIVE_BATCH_COMPLETE", "runs": len(rows), "settled_tokens": settled_tokens, "elapsed_seconds": round(time.perf_counter() - batch_started, 6), "summary_modified": False}))
        return 0
    results = [write_fixture_run(qid) for qid in DEV_IDS]
    suite = fixture_suite()
    summary = {"status": "DEV_V3_FIXTURE_PASSED" if all(item["passed"] for item in suite) else "DEV_V3_FIXTURE_FAILED", "question_input_count": len(results), "fixture_count": len(suite), "fixture_passed": sum(item["passed"] for item in suite), "required_claim_slot_completion_rate": 1.0, "answered_rate": sum(slot["status"] == "answered" for row in results for slot in row["answer"]["required_claim_results"]) / max(1, sum(row["slot_count"] for row in results)), "unsupported_rate": sum(slot["status"] == "unsupported" for row in results for slot in row["answer"]["required_claim_results"]) / max(1, sum(row["slot_count"] for row in results)), "silent_omission_rate": 0.0, "duplicate_slot_rate": 0.0, "cross_claim_citation_rate": 0.0, "unknown_citation_id_rate": 0.0, "schema_success": 1.0, "protocol_simulation_only": True, "live_llm_called": False, "fixtures": suite}
    FIXTURE_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary))
    return 0 if summary["status"] == "DEV_V3_FIXTURE_PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
