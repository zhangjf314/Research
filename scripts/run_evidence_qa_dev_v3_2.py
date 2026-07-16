# ruff: noqa: E501
"""One-shot controlled live Dev v3.2 runner with deterministic post-policy processing."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from paper_research.config import Settings
from paper_research.generation.citation_selection import (
    FallbackAction,
    select_citations,
    validate_comparison_evidence,
    validate_numeric_evidence,
)
from paper_research.generation.prompts import (
    QA_REQUIRED_CLAIMS_CITATION_ID_V3_2_CANDIDATE,
    qa_system_prompt,
)
from paper_research.generation.required_claim_output import (
    RequiredClaimsQAResponseV31,
    RequiredClaimsQAResponseV32,
    RequiredClaimValidationError,
    parse_and_validate_required_claim_response_v32,
)
from paper_research.providers.capabilities import siliconflow_qwen3_8b_stage13_5_snapshot

try:
    from scripts.evidence_qa_dev_lib_v1 import DEV_IDS, canonical_hash
    from scripts.evidence_qa_dev_v3_2_lib import (
        CAPABILITY_HASH,
        HEALTH,
        MANIFEST,
        POLICY_HASH,
        PROMPT_HASH,
        RUN_ROOT,
        SCHEMA_HASH,
        SOURCE_MANIFEST_HASH,
        build_required_claim_input,
        write_manifest,
    )
    from scripts.replay_dev_v3_2_citation_policy_v1 import build_candidates
except ModuleNotFoundError:
    from evidence_qa_dev_lib_v1 import DEV_IDS, canonical_hash  # type: ignore[no-redef]
    from evidence_qa_dev_v3_2_lib import (  # type: ignore[no-redef]
        CAPABILITY_HASH,
        HEALTH,
        MANIFEST,
        POLICY_HASH,
        PROMPT_HASH,
        RUN_ROOT,
        SCHEMA_HASH,
        SOURCE_MANIFEST_HASH,
        build_required_claim_input,
        write_manifest,
    )
    from replay_dev_v3_2_citation_policy_v1 import build_candidates  # type: ignore[no-redef]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preflight", "live"), required=True)
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
    protocol = json.loads(
        (MANIFEST.parent / "dev-v3-2-protocol-candidate-v1.json").read_text(encoding="utf-8")
    )
    prompt = qa_system_prompt(QA_REQUIRED_CLAIMS_CITATION_ID_V3_2_CANDIDATE)
    schema = RequiredClaimsQAResponseV31.model_json_schema()
    observed = {
        "manifest_hash": manifest["manifest_hash"],
        "question_ids": manifest["question_ids"],
        "required_claims": manifest["total_required_claims"],
        "prompt_hash": canonical_hash(prompt),
        "schema_hash": canonical_hash(schema),
        "policy_hash": protocol["policy_hash"],
        "collection": settings.qdrant_collection,
        "embedding_model": settings.embedding_model,
        "embedding_dimensions": settings.embedding_dimensions,
        "provider": settings.llm_provider,
        "model": settings.llm_model,
        "temperature": settings.llm_temperature,
        "retries": settings.llm_max_retries,
        "billing": settings.llm_billing_mode,
        "reranker": settings.rerank_enabled,
        "citation_cap": protocol["policy_versions"]["citation_budget"]["max_total"],
    }
    expected = {
        "manifest_hash": SOURCE_MANIFEST_HASH,
        "question_ids": DEV_IDS,
        "required_claims": 27,
        "prompt_hash": PROMPT_HASH,
        "schema_hash": SCHEMA_HASH,
        "policy_hash": POLICY_HASH,
        "collection": "papers_jina_eval34_v2__20260713152149",
        "embedding_model": "jina-embeddings-v5-text-small",
        "embedding_dimensions": 1024,
        "provider": "siliconflow",
        "model": "Qwen/Qwen3-8B",
        "temperature": 0,
        "retries": 0,
        "billing": "free",
        "reranker": False,
        "citation_cap": 3,
    }
    failures = [key for key in expected if observed[key] != expected[key]]
    config = manifest["configuration"]
    checks = {
        "response_format_json_object": config["response_format"] == "json_object",
        "no_json_schema": config["json_schema_sent"] is False,
        "no_tools": config["tools_or_functions_sent"] is False,
        "raw_schema_only": config["formal_normalization_policy"] == "raw_schema_passed_only",
        "q005_zero_claims": manifest["required_claim_counts"]["q005"] == 0,
        "gold_absent_from_payload": True,
        "human_labels_absent_from_payload": True,
    }
    for question_id in DEV_IDS:
        payload, _registry, _contexts, _trace = build_required_claim_input(question_id)
        encoded = json.dumps(payload, ensure_ascii=False).lower()
        if any(token in encoded for token in ("core_gold", "human_label", "gold_block")):
            checks["gold_absent_from_payload"] = False
    failures.extend(key for key, value in checks.items() if not value)
    if failures:
        raise RuntimeError(f"DEV_V3_2_CONFIGURATION_INVALID: {sorted(failures)}")
    return {"observed": observed, "checks": checks}


def assert_live_authorized() -> dict[str, Any]:
    if os.getenv("DEV_V3_2_LIVE_AUTHORIZED") != "true":
        raise RuntimeError("DEV_V3_2_LIVE_NOT_AUTHORIZED")
    if not HEALTH.exists():
        raise RuntimeError("DEV_V3_2_BLOCKED_BY_PROVIDER_HEALTH")
    health = json.loads(HEALTH.read_text(encoding="utf-8"))
    if not health.get("safe_to_start_batch"):
        raise RuntimeError("DEV_V3_2_BLOCKED_BY_PROVIDER_HEALTH")
    if list(RUN_ROOT.glob("live-dev-v3-2-*/final-result.json")):
        raise RuntimeError("duplicate formal Dev v3.2 run set")
    return health


def persist_pre_request(
    question_id: str,
    run_dir: Path,
    request_id: str,
    payload: dict[str, Any],
    registry: Any,
    trace: dict[str, Any],
    settings: Settings,
) -> tuple[Path, list[Any], dict[str, Any]]:
    ledger = run_dir / "request-ledger.jsonl"
    ledger.write_text("", encoding="utf-8")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    protocol = json.loads(
        (MANIFEST.parent / "dev-v3-2-protocol-candidate-v1.json").read_text(encoding="utf-8")
    )
    schema = RequiredClaimsQAResponseV31.model_json_schema()
    system_prompt = qa_system_prompt(QA_REQUIRED_CLAIMS_CITATION_ID_V3_2_CANDIDATE)
    user_prompt = json.dumps(payload, ensure_ascii=False)
    candidates_by_claim = {}
    all_candidates = []
    for claim in payload["required_claims"]:
        candidates = build_candidates(claim, registry.model_dump(mode="json"), trace, set())
        candidates_by_claim[claim["required_claim_id"]] = candidates
        all_candidates.extend(candidates)
    candidate_body = [
        {
            "required_claim_id": claim_id,
            "candidates": [candidate.__dict__ for candidate in candidates],
        }
        for claim_id, candidates in candidates_by_claim.items()
    ]
    artifacts = {
        "required-claims-input.json": payload,
        "exact-json-schema.json": schema,
        "prompt-metadata.json": {
            "prompt_version": QA_REQUIRED_CLAIMS_CITATION_ID_V3_2_CANDIDATE,
            "prompt_hash": PROMPT_HASH,
            "schema_hash": SCHEMA_HASH,
            "policy_hash": POLICY_HASH,
        },
        "provider-capability-snapshot.json": {
            "snapshot": siliconflow_qwen3_8b_stage13_5_snapshot().model_dump(mode="json"),
            "snapshot_hash": CAPABILITY_HASH,
        },
        "response-format-parameters.json": {
            "response_format": {"type": "json_object"},
            "json_schema_sent": False,
            "tools_sent": False,
            "functions_sent": False,
        },
        "citation-registry.json": registry.model_dump(mode="json"),
        "candidate-evidence.json": candidate_body,
        "citation-policy-input.json": {
            "policy_versions": protocol["policy_versions"],
            "combined_policy_hash": POLICY_HASH,
            "citation_budget": protocol["policy_versions"]["citation_budget"],
        },
        "retrieval-trace.json": trace,
        "context-trace.json": trace,
    }
    event(ledger, "manifest_validated", manifest_hash=manifest["manifest_hash"])
    for name, value in artifacts.items():
        write_json(run_dir / name, value)
        event(ledger, f"{name.removesuffix('.json').replace('-', '_')}_persisted")
        event(ledger, f"{name.removesuffix('.json').replace('-', '_')}_hash_recorded", value=canonical_hash(value))
    (run_dir / "rendered-system-prompt.txt").write_text(system_prompt, encoding="utf-8")
    (run_dir / "rendered-user-prompt.txt").write_text(user_prompt, encoding="utf-8")
    event(ledger, "prompt_rendered")
    event(ledger, "prompt_hash_recorded", value=PROMPT_HASH)
    event(ledger, "request_id_allocated", request_id=request_id)
    event(ledger, "budget_reserved", request_id=request_id, reserved_tokens=24000)
    event(ledger, "request_prepared", request_id=request_id, response_format="json_object")
    metadata = {
        "run_id": run_dir.name,
        "question_id": question_id,
        "evaluation_version": "evidence-qa-dev-v3.2",
        "manifest_hash": SOURCE_MANIFEST_HASH,
        "required_claim_input_hash": canonical_hash(payload),
        "citation_registry_hash": registry.registry_hash,
        "candidate_evidence_hash": canonical_hash(candidate_body),
        "prompt_hash": PROMPT_HASH,
        "schema_hash": SCHEMA_HASH,
        "policy_hash": POLICY_HASH,
        "request_id": request_id,
        "collection": settings.qdrant_collection,
        "embedding_model": settings.embedding_model,
        "reranker_enabled": False,
        "retries": 0,
        "billing_mode": "free",
        "api_key_recorded": False,
        "authorization_header_recorded": False,
        "gold_used_online": False,
        "human_labels_used_online": False,
        "formal_normalization_used": False,
    }
    write_json(run_dir / "run-metadata.json", metadata)
    return ledger, all_candidates, {"by_claim": candidates_by_claim, "body": candidate_body}


def apply_policy(
    raw: RequiredClaimsQAResponseV32,
    payload: dict[str, Any],
    candidates_by_claim: dict[str, list[Any]],
) -> tuple[RequiredClaimsQAResponseV32, dict[str, Any]]:
    inputs = {item["required_claim_id"]: item for item in payload["required_claims"]}
    final_slots = []
    traces = []
    for slot in raw.required_claim_results:
        if slot.status.value != "answered":
            final_slots.append(slot.model_dump(mode="json"))
            traces.append({"required_claim_id": slot.required_claim_id, "unchanged": True})
            continue
        candidates = [
            candidate.__class__(
                **{
                    **candidate.__dict__,
                    "currently_cited": candidate.citation_id in slot.citation_ids,
                }
            )
            for candidate in candidates_by_claim[slot.required_claim_id]
        ]
        selection = select_citations(slot.claim_text or inputs[slot.required_claim_id]["required_claim_text"], candidates)
        selected_ids = list(selection.primary_citation_ids + selection.supporting_citation_ids)
        if selection.fallback_action == FallbackAction.UNSUPPORTED:
            final = {
                "required_claim_id": slot.required_claim_id,
                "status": "unsupported",
                "claim_text": None,
                "citation_ids": [],
                "omission_reason": (
                    "Evidence gap: " + ", ".join(selection.uncovered_requirements or ("claim obligations incomplete",))
                ),
            }
        else:
            final_text = (
                selection.narrowed_claim_text
                if selection.fallback_action == FallbackAction.ANSWERED_NARROWED
                else slot.claim_text
            )
            final = {
                "required_claim_id": slot.required_claim_id,
                "status": "answered",
                "claim_text": final_text,
                "citation_ids": selected_ids[:3],
                "omission_reason": None,
            }
        final_slots.append(final)
        selected_candidates = [candidate for candidate in candidates if candidate.citation_id in final["citation_ids"]]
        traces.append(
            {
                "required_claim_id": slot.required_claim_id,
                "original_claim_text": slot.claim_text,
                "final_claim_text": final["claim_text"],
                "original_citation_ids": slot.citation_ids,
                "final_citation_ids": final["citation_ids"],
                "fallback_action": selection.fallback_action.value,
                "removed_obligations": list(selection.removed_obligations),
                "uncovered_requirements": list(selection.uncovered_requirements),
                "numeric_validation": validate_numeric_evidence(slot.claim_text or "", selected_candidates).__dict__,
                "comparison_validation": validate_comparison_evidence(slot.claim_text or "", selected_candidates).__dict__,
                "decision_reasons": list(selection.decision_reasons),
            }
        )
    body = {
        "question_id": raw.question_id,
        "answerable": raw.answerable,
        "required_claim_results": final_slots,
        "refusal_reason": raw.refusal_reason,
        "prompt_version": QA_REQUIRED_CLAIMS_CITATION_ID_V3_2_CANDIDATE,
        "citation_protocol": "citation-id-v2",
    }
    return RequiredClaimsQAResponseV32.model_validate(body), {"slots": traces}


def run_one(question_id: str, settings: Settings, client: httpx.Client) -> dict[str, Any]:
    payload, registry, _contexts, trace = build_required_claim_input(question_id)
    run_id = f"live-dev-v3-2-{question_id}-{uuid.uuid4().hex[:12]}"
    request_id = f"{run_id}:primary:1:{uuid.uuid4().hex[:12]}"
    run_dir = RUN_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    ledger, _all_candidates, candidate_data = persist_pre_request(
        question_id, run_dir, request_id, payload, registry, trace, settings
    )
    event(ledger, "request_started", request_id=request_id)
    started = time.perf_counter()
    status = "provider_failed"
    failure_type = failure_reason = None
    usage: dict[str, Any] = {}
    raw_answer: dict[str, Any] = {}
    final_answer: dict[str, Any] = {}
    active_reserved_tokens = 24000
    try:
        response = client.post(
            f"{(settings.llm_base_url or '').rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {settings.llm_api_key}", "Content-Type": "application/json"},
            json={
                "model": settings.llm_model,
                "messages": [
                    {"role": "system", "content": qa_system_prompt(QA_REQUIRED_CLAIMS_CITATION_ID_V3_2_CANDIDATE)},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
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
        body = response.json()
        usage_body = body.get("usage") or {}
        usage = {
            "input_tokens": int(usage_body.get("prompt_tokens", 0)),
            "output_tokens": int(usage_body.get("completion_tokens", 0)),
            "total_tokens": int(usage_body.get("total_tokens", 0)),
            "usage_source": "provider_reported",
        }
        write_json(run_dir / "raw-provider-response.json", body)
        write_json(
            run_dir / "provider-response-envelope.json",
            {
                "request_id": request_id,
                "provider": "siliconflow",
                "model": settings.llm_model,
                "usage": usage,
                "response_received": True,
                "raw_response_persisted": True,
            },
        )
        event(ledger, "raw_response_received", request_id=request_id)
        event(ledger, "provider_usage_recorded", request_id=request_id, **usage)
        event(ledger, "raw_response_persisted", request_id=request_id)
        content = body["choices"][0]["message"]["content"]
        allowed = {
            claim["required_claim_id"]: set(claim["allowed_citation_ids"])
            for claim in payload["required_claims"]
        }
        raw = parse_and_validate_required_claim_response_v32(
            content,
            expected_question_id=question_id,
            expected_claim_ids=[claim["required_claim_id"] for claim in payload["required_claims"]],
            registry=registry,
            allowed_by_claim=allowed,
            expected_registry_hash=registry.registry_hash,
        )
        raw_answer = raw.model_dump(mode="json")
        write_json(run_dir / "parsed-v3-2-output.json", raw_answer)
        event(ledger, "raw_schema_validation_passed")
        event(ledger, "required_claim_slot_validation_passed")
        final, policy_trace = apply_policy(raw, payload, candidate_data["by_claim"])
        final_answer = final.model_dump(mode="json")
        write_json(run_dir / "citation-selection-trace.json", policy_trace)
        write_json(run_dir / "obligation-analysis.json", {"slots": policy_trace["slots"]})
        write_json(run_dir / "numeric-validation.json", {"slots": [item.get("numeric_validation") for item in policy_trace["slots"]]})
        write_json(run_dir / "comparison-validation.json", {"slots": [item.get("comparison_validation") for item in policy_trace["slots"]]})
        write_json(run_dir / "claim-fallback-trace.json", {"slots": policy_trace["slots"]})
        event(ledger, "deterministic_citation_policy_applied")
        event(ledger, "obligation_numeric_comparison_validation_completed")
        parse_and_validate_required_claim_response_v32(
            json.dumps(final_answer),
            expected_question_id=question_id,
            expected_claim_ids=[claim["required_claim_id"] for claim in payload["required_claims"]],
            registry=registry,
            allowed_by_claim=allowed,
            expected_registry_hash=registry.registry_hash,
        )
        event(ledger, "final_schema_revalidation_passed")
        status = "completed"
        active_reserved_tokens = 0
        event(ledger, "completed", active_reserved_tokens=0)
    except RequiredClaimValidationError as exc:
        status, failure_type, failure_reason = "validation_failed", exc.code, str(exc)
        event(ledger, "validation_failed", failure_type=exc.code)
    except httpx.HTTPError as exc:
        failure_type, failure_reason = type(exc).__name__, str(exc)
        event(ledger, "request_failed", failure_type=failure_type)
        write_json(run_dir / "raw-provider-response.json", {"response_received": False, "request_id": request_id, "failure_type": failure_type})
        write_json(run_dir / "provider-response-envelope.json", {"response_received": False, "request_id": request_id, "usage": None})
    elapsed = time.perf_counter() - started
    result = {
        "run_id": run_id,
        "question_id": question_id,
        "status": status,
        "failure_type": failure_type,
        "failure_reason": failure_reason,
        "raw_answer": raw_answer,
        "final_answer": final_answer,
        "required_claim_count": len(payload["required_claims"]),
        "raw_slot_count": len(raw_answer.get("required_claim_results", [])),
        "final_slot_count": len(final_answer.get("required_claim_results", [])),
        "request_attempt_count": 1,
        "provider_completed_request_count": int(bool(usage)),
        "provider_failure_count": int(status == "provider_failed"),
        "usage_record_count": int(bool(usage)),
        "usage": usage,
        "active_reserved_tokens": active_reserved_tokens,
        "elapsed_seconds": round(elapsed, 6),
        "monetary_cost_usd": "0",
        "cost_basis": "explicit_free_provider",
        "reranker_called": False,
        "template_fallback": False,
        "retries": 0,
        "prompt_hash": PROMPT_HASH,
        "schema_hash": SCHEMA_HASH,
        "policy_hash": POLICY_HASH,
        "citation_registry_hash": registry.registry_hash,
        "required_claim_input_hash": canonical_hash(payload),
        "candidate_evidence_hash": canonical_hash(candidate_data["body"]),
    }
    write_json(run_dir / "final-result.json", result)
    flat = {key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value for key, value in result.items()}
    with (run_dir / "final-result.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat))
        writer.writeheader()
        writer.writerow(flat)
    return result


def main() -> None:
    args = parse_args()
    result = preflight()
    if args.mode == "preflight":
        print(json.dumps(result, indent=2))
        return
    assert_live_authorized()
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    settings = Settings()
    results = []
    provider_failures = consecutive_connection_failures = 0
    with httpx.Client() as client:
        for question_id in DEV_IDS:
            row = run_one(question_id, settings, client)
            results.append(row)
            if row["status"] == "provider_failed":
                provider_failures += 1
                if row["failure_type"] in {"ConnectError", "ConnectTimeout", "ReadTimeout"}:
                    consecutive_connection_failures += 1
                else:
                    consecutive_connection_failures = 0
            else:
                consecutive_connection_failures = 0
            if provider_failures >= 2 or consecutive_connection_failures >= 2:
                break
    print(json.dumps({"runs": len(results), "results": results}, ensure_ascii=False))


if __name__ == "__main__":
    main()
