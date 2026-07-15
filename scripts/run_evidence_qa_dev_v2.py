# ruff: noqa: E501
"""Run one isolated citation-id-v2 evaluation over the frozen Stage 13 Dev manifest."""

from __future__ import annotations

import argparse
import csv
import json
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from pydantic import ValidationError

from paper_research.config import Settings
from paper_research.generation.citation_id_output import (
    CitationIdQA,
    resolve_citation_id_answer,
)
from paper_research.generation.citation_registry import (
    CitationRegistry,
    CitationRegistryError,
)
from paper_research.generation.prompts import (
    QA_PRODUCTION_CITATION_ID_V2,
    qa_system_prompt,
)
from paper_research.providers.response_envelope import ProviderResponseEnvelopeStore

try:
    from scripts.evidence_qa_dev_lib_v1 import (
        DATA,
        DEV_IDS,
        MANIFEST,
        evaluate_answer,
        read_jsonl,
    )
    from scripts.run_evidence_qa_dev_v1 import load_contexts
except ModuleNotFoundError:
    from evidence_qa_dev_lib_v1 import (  # type: ignore[no-redef]
        DATA,
        DEV_IDS,
        MANIFEST,
        evaluate_answer,
        read_jsonl,
    )
    from run_evidence_qa_dev_v1 import load_contexts  # type: ignore[no-redef]

RUN_ROOT = DATA / "evidence-qa-dev-v2/runs"
HEALTH = DATA / "provider-health-v1.json"
EVALUATION_VERSION = "evidence-qa-dev-v2"
HISTORICAL_RESERVATIONS = 60000
GLOBAL_TOKEN_BUDGET = 200000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("live", "dry-run"), default="dry-run")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--no-summary", action="store_true")
    return parser.parse_args()


def append_event(path: Path, event: str, **values: Any) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {"event": event, "timestamp": datetime.now(UTC).isoformat(), **values},
                ensure_ascii=False,
            )
            + "\n"
        )


def preflight(settings: Settings, manifest: dict[str, Any]) -> dict[str, Any]:
    failures = []
    expected = {
        "app_profile": "production",
        "embedding_provider": "jina",
        "embedding_model": "jina-embeddings-v5-text-small",
        "embedding_dimensions": 1024,
        "llm_provider": "siliconflow",
        "llm_model": "Qwen/Qwen3-8B",
        "llm_temperature": 0,
        "llm_max_retries": 0,
        "llm_billing_mode": "free",
        "rerank_enabled": False,
    }
    for field, value in expected.items():
        if getattr(settings, field) != value:
            failures.append(f"{field} must equal {value!r}")
    if not settings.llm_api_key:
        failures.append("LLM_API_KEY must be configured locally")
    if manifest.get("manifest_hash") != "fcb59b71fc68549479c24f6475f7d18ad9e382aace93e70e93594ee355ffb988":
        failures.append("frozen manifest hash changed")
    if manifest.get("question_ids") != DEV_IDS:
        failures.append("frozen manifest questions changed")
    if failures:
        raise RuntimeError("fail-closed Dev v2 preflight: " + "; ".join(failures))
    return {
        "collection": settings.active_collection,
        "embedding_model": settings.embedding_model,
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
        "effective_prompt_version": QA_PRODUCTION_CITATION_ID_V2,
        "citation_protocol": "citation-id-v2",
        "temperature": 0,
        "rerank_enabled": False,
        "llm_max_retries": 0,
        "billing_mode": "free",
        "gold_used_for_selection": False,
        "oracle_used_for_selection": False,
        "human_pilot_used_for_selection": False,
        "api_key_configured": True,
    }


def request_payload(question: str, contexts, registry: CitationRegistry, model: str) -> dict[str, Any]:
    evidence = []
    entries_by_evidence: dict[str, list[str]] = {}
    for entry in registry.entries:
        entries_by_evidence.setdefault(entry.evidence_id, []).append(entry.citation_id)
    for item in contexts:
        evidence.append(
            {
                "evidence_id": item.chunk_id,
                "text": item.evidence,
                "citation_ids": entries_by_evidence[item.chunk_id],
            }
        )
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": qa_system_prompt(QA_PRODUCTION_CITATION_ID_V2)},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "question": question,
                        "evidence": evidence,
                        "citation_registry": registry.prompt_entries(),
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        "temperature": 0,
        "max_tokens": 2048,
        "stream": False,
        "enable_thinking": False,
        "response_format": {"type": "json_object"},
    }


def write_result(run_dir: Path, row: dict[str, Any]) -> None:
    (run_dir / "result.json").write_text(
        json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    flat = {
        "run_id": row["run_id"],
        "question_id": row["question_id"],
        "status": row["status"],
        "request_attempt_count": row["request_attempt_count"],
        "provider_completed_request_count": row["provider_completed_request_count"],
        "usage_record_count": row["usage_record_count"],
        "total_tokens": row.get("usage", {}).get("total_tokens"),
        "active_reserved_tokens": row["active_reserved_tokens"],
        "elapsed_seconds": row["elapsed_seconds"],
    }
    with (run_dir / "result.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat))
        writer.writeheader()
        writer.writerow(flat)


def write_no_response_artifacts(run_dir: Path, request_id: str, error: Exception) -> None:
    """Keep a complete run directory without pretending a response or usage exists."""
    sentinel = {
        "response_received": False,
        "request_id": request_id,
        "failure_type": type(error).__name__,
    }
    (run_dir / "raw-provider-response.json").write_text(
        json.dumps(sentinel, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "provider-response-envelope.json").write_text(
        json.dumps(
            {
                "schema_version": "provider-response-envelope-v1-failure",
                **sentinel,
                "usage": None,
                "usage_status": "unknown_after_send_reserved_conservative",
                "parse_status": "not_started",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def run_one(
    client: httpx.Client,
    settings: Settings,
    manifest: dict[str, Any],
    protocol: dict[str, Any],
    gold: dict[str, Any],
    question_id: str,
    audit: dict[str, Any],
) -> dict[str, Any]:
    run_id = f"live-dev-v2-{question_id}-{uuid.uuid4().hex[:12]}"
    request_id = f"{run_id}:primary:1:{uuid.uuid4().hex[:12]}"
    run_dir = RUN_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    contexts, context_trace = load_contexts(question_id)
    registry = CitationRegistry.from_context(contexts)
    registry_path = run_dir / "citation-registry.json"
    registry_path.write_text(
        json.dumps(registry.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    retrieval_trace = {
        "question_id": question_id,
        "retrieval_scope": protocol["retrieval_scope"],
        "retrieval_filter": protocol["retrieval_filter"],
        **context_trace,
    }
    (run_dir / "retrieval-trace.json").write_text(
        json.dumps(retrieval_trace, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "context-trace.json").write_text(
        json.dumps(context_trace, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    ledger = run_dir / "request-ledger.jsonl"
    append_event(
        ledger,
        "request_prepared",
        request_id=request_id,
        reserved_tokens=20000,
        usage_status="reserved",
    )
    append_event(
        ledger,
        "request_started",
        request_id=request_id,
        reserved_tokens=20000,
        usage_status="reserved",
    )
    metadata = {
        "schema_version": "evidence-qa-dev-v2-run-v1",
        "evaluation_version": EVALUATION_VERSION,
        "run_id": run_id,
        "request_id": request_id,
        "question_id": question_id,
        "manifest_hash": manifest["manifest_hash"],
        "citation_registry_hash": registry.registry_hash,
        "citation_registry_schema_version": registry.schema_version,
        "provider_response_envelope_schema_version": "provider-response-envelope-v1",
        "configuration": audit,
        "api_key_recorded": False,
        "authorization_header_recorded": False,
        "deep_research_called": False,
    }
    (run_dir / "run-metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    base = {
        "run_id": run_id,
        "evaluation_version": EVALUATION_VERSION,
        "question_id": question_id,
        "category": gold["category"],
        "difficulty": gold["difficulty"],
        "retrieval_scope": protocol["retrieval_scope"],
        "retrieval_filter": protocol["retrieval_filter"],
        "prompt_version": QA_PRODUCTION_CITATION_ID_V2,
        "citation_protocol": "citation-id-v2",
        "retrieval_variant": "phase_b_adjacent_same_page_completion",
        "context_version": "phase-b-adjacent-same-page-v1",
        "citation_registry_hash": registry.registry_hash,
        "gold": {
            "answerable": gold["answerable"],
            "gold_paper_ids": gold["gold_paper_ids"],
            "gold_pages": gold["gold_pages"],
            "gold_block_ids": gold["gold_block_ids"],
            "required_claims": gold["required_claims"],
        },
    }
    started = time.perf_counter()
    provider_completed = 0
    usage_record_count = 0
    usage: dict[str, Any] = {}
    active_reservation = 20000
    store = ProviderResponseEnvelopeStore(run_dir, ledger)
    try:
        response = client.post(
            f"{(settings.llm_base_url or '').rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.llm_api_key}",
                "Content-Type": "application/json",
            },
            json=request_payload(protocol["retrieval_query"], contexts, registry, settings.llm_model),
            timeout=httpx.Timeout(connect=15, read=180, write=30, pool=15),
        )
        if response.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"HTTP {response.status_code}", request=response.request, response=response
            )
        envelope = store.record_received(
            request_id=request_id,
            provider="siliconflow",
            model=settings.llm_model,
            raw_body=response.content,
        )
        provider_completed = 1
        usage_record_count = 1
        usage = envelope.usage.model_dump()
        active_reservation = 0
        envelope = store.parsing_started(envelope)
        try:
            payload = envelope.parsed_provider_payload
            content = payload["choices"][0]["message"]["content"]
            parsed = CitationIdQA.model_validate(json.loads(content))
            answer, duplicate_ids = resolve_citation_id_answer(parsed, registry)
            allowed = {entry.triple for entry in registry.entries}
            metrics = evaluate_answer(answer, gold, allowed)
            metrics.update(
                {
                    "unknown_citation_id_count": 0,
                    "duplicate_citation_ids": duplicate_ids,
                    "unknown_citation_id_rate": 0.0,
                }
            )
            envelope = store.parsed(envelope)
            elapsed = time.perf_counter() - started
            status = "completed"
            failure_reason = None
        except (json.JSONDecodeError, KeyError, TypeError, ValidationError, CitationRegistryError, ValueError) as exc:
            envelope = store.post_processing_failed(envelope, exc)
            elapsed = time.perf_counter() - started
            answer = {}
            metrics = {
                "unknown_citation_id_count": int(
                    isinstance(exc, CitationRegistryError)
                    and "unknown citation_id" in str(exc)
                ),
                "unknown_citation_id_rate": 1.0
                if isinstance(exc, CitationRegistryError)
                and "unknown citation_id" in str(exc)
                else 0.0,
            }
            status = "validation_failed"
            failure_reason = f"{type(exc).__name__}: {exc}"
    except (httpx.HTTPError, json.JSONDecodeError, ValueError) as exc:
        elapsed = time.perf_counter() - started
        append_event(
            ledger,
            "request_failed",
            request_id=request_id,
            request_status="failed_after_send_unknown",
            usage_status="reserved_conservative",
            active_reserved_tokens=20000,
            failure_type=type(exc).__name__,
            failure_message=str(exc)[:500],
        )
        answer = {}
        metrics = {}
        status = "provider_failed"
        failure_reason = f"{type(exc).__name__}: {exc}"
        write_no_response_artifacts(run_dir, request_id, exc)
    row = {
        **base,
        "status": status,
        "failure_reason": failure_reason,
        "answer": answer,
        "metrics": metrics,
        "request_attempt_count": 1,
        "provider_completed_request_count": provider_completed,
        "provider_failure_count": int(status == "provider_failed"),
        "post_processing_failure_count": int(status == "validation_failed"),
        "usage_record_count": usage_record_count,
        "usage": usage,
        "usage_source": usage.get("usage_source", "unavailable_after_send_attempt"),
        "active_reserved_tokens": active_reservation,
        "citation_retry_count": 0,
        "elapsed_seconds": round(elapsed, 6),
        "monetary_cost_usd": "0",
        "cost_basis": "explicit_free_provider",
        "citation_validation": "passed" if status == "completed" else "failed",
        "registry_hash_valid": CitationRegistry.model_validate(
            json.loads(registry_path.read_text(encoding="utf-8"))
        ).registry_hash
        == registry.registry_hash,
        "reranker_called": False,
        "template_fallback": False,
    }
    write_result(run_dir, row)
    return row


def main() -> int:
    args = parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    settings = Settings()
    audit = preflight(settings, manifest)
    if args.mode == "dry-run":
        print(json.dumps({"status": "dry_run_preflight_passed", "audit": audit}))
        return 0
    health = json.loads(HEALTH.read_text(encoding="utf-8"))
    if not health.get("safe_to_start_batch"):
        print("DEV_V2_BLOCKED_BY_PROVIDER_HEALTH")
        return 2
    existing = list(RUN_ROOT.glob("*/result.json")) if RUN_ROOT.exists() else []
    if existing:
        raise RuntimeError("Dev v2 already has attempts; this evaluation may run only once")
    protocol = {row["question_id"]: row for row in read_jsonl(DATA / "retrieval-gold-v2.jsonl")}
    gold = {row["question_id"]: row for row in read_jsonl(DATA / "gold-set-v1.jsonl")}
    rows = []
    provider_failures = 0
    consecutive_failures = 0
    stop_reason = None
    with httpx.Client() as client:
        for question_id in DEV_IDS:
            accounted = HISTORICAL_RESERVATIONS + sum(
                row.get("usage", {}).get("total_tokens", 0)
                + row.get("active_reserved_tokens", 0)
                for row in rows
            )
            if accounted + 20000 > GLOBAL_TOKEN_BUDGET:
                stop_reason = "active_reservation_budget_insufficient"
                break
            row = run_one(
                client,
                settings,
                manifest,
                protocol[question_id],
                gold[question_id],
                question_id,
                audit,
            )
            rows.append(row)
            if row["status"] == "provider_failed":
                provider_failures += 1
                consecutive_failures += 1
            else:
                consecutive_failures = 0
            if provider_failures >= 2 or consecutive_failures >= 2:
                stop_reason = "DEV_V2_BLOCKED_BY_PROVIDER_CONNECTIVITY"
                break
            if row.get("usage", {}).get("total_tokens", 0) > 20000:
                stop_reason = "per_question_token_budget_exceeded"
                break
            if row["elapsed_seconds"] > 180:
                stop_reason = "per_question_elapsed_budget_exceeded"
                break
            if not row["registry_hash_valid"]:
                stop_reason = "registry_hash_mismatch"
                break
            if sum(item["elapsed_seconds"] for item in rows) > 1800:
                stop_reason = "elapsed_budget_exceeded"
                break
    print(
        json.dumps(
            {
                "status": stop_reason or "dev_v2_batch_completed",
                "run_count": len(rows),
                "provider_failures": provider_failures,
                "top_level_summary_modified": False,
            }
        )
    )
    return 2 if stop_reason else 0


if __name__ == "__main__":
    raise SystemExit(main())
