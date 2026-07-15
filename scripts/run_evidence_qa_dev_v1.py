# ruff: noqa: E501
"""Run isolated Stage 13.2 Dev QA attempts without modifying top-level summaries."""

from __future__ import annotations

import argparse
import csv
import json
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from paper_research.config import Settings
from paper_research.providers.llm import LLMProviderError, SiliconFlowLLMProvider
from paper_research.retrieval.context_builder import ContextItem

try:
    from scripts.evidence_qa_dev_lib_v1 import (
        BLOCKED_C_REASON,
        DATA,
        DEV_IDS,
        MANIFEST,
        RUN_ROOT,
        VARIANT_B,
        VARIANT_C,
        evaluate_answer,
        phase_b_rows,
        read_jsonl,
        write_manifest,
    )
except ModuleNotFoundError:
    from evidence_qa_dev_lib_v1 import (  # type: ignore[no-redef]
        BLOCKED_C_REASON,
        DATA,
        DEV_IDS,
        MANIFEST,
        RUN_ROOT,
        VARIANT_B,
        VARIANT_C,
        evaluate_answer,
        phase_b_rows,
        read_jsonl,
        write_manifest,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=(VARIANT_B, VARIANT_C))
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--mode", choices=("live", "dry-run"), default="dry-run")
    parser.add_argument("--no-summary", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def safe_preflight(settings: Settings, variant: str) -> dict[str, Any]:
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
    for key, value in expected.items():
        if getattr(settings, key) != value:
            failures.append(f"{key} must equal {value!r}")
    if not settings.llm_api_key:
        failures.append("LLM_API_KEY must be configured locally")
    if variant != VARIANT_B:
        failures.append(BLOCKED_C_REASON)
    if failures:
        raise RuntimeError("fail-closed preflight: " + "; ".join(failures))
    return {
        "app_profile": settings.app_profile,
        "collection": settings.active_collection,
        "embedding_provider": settings.embedding_provider,
        "embedding_model": settings.embedding_model,
        "embedding_dimension": settings.embedding_dimensions,
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
        "llm_temperature": settings.llm_temperature,
        "llm_max_retries": settings.llm_max_retries,
        "billing_mode": settings.llm_billing_mode,
        "rerank_enabled": settings.rerank_enabled,
        "api_key_configured": True,
    }


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def recover_orphaned_attempts(
    manifest: dict[str, Any],
    protocol: dict[str, dict[str, Any]],
    gold: dict[str, dict[str, Any]],
) -> list[str]:
    """Close interrupted client attempts conservatively without issuing a request."""
    recovered = []
    for run_dir in (RUN_ROOT / VARIANT_B).iterdir() if (RUN_ROOT / VARIANT_B).exists() else []:
        if not run_dir.is_dir() or (run_dir / "result.json").exists():
            continue
        ledger = run_dir / "request-ledger.jsonl"
        if not ledger.exists():
            continue
        events = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
        if not events or events[-1]["event"] != "request_started":
            continue
        run_id = run_dir.name
        question_id = next((qid for qid in DEV_IDS if f"-{qid}-" in run_id), None)
        if question_id is None:
            raise RuntimeError(f"cannot recover orphaned run identity: {run_id}")
        request_id = events[-1]["request_id"]
        append_jsonl(
            ledger,
            {
                "event": "request_failed",
                "request_id": request_id,
                "request_status": "client_postprocessing_failed_after_send",
                "usage_status": "reserved_conservative",
                "active_reserved_tokens": 20000,
                "failure_type": "ClientPostprocessingError",
                "failure_message": "Stage 13.2 claim field adapter emitted claim_text while evaluator read text",
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )
        row = {
            "run_id": run_id,
            "variant": VARIANT_B,
            "question_id": question_id,
            "category": gold[question_id]["category"],
            "difficulty": gold[question_id]["difficulty"],
            "retrieval_scope": protocol[question_id]["retrieval_scope"],
            "retrieval_filter": protocol[question_id]["retrieval_filter"],
            "prompt_version": "qa-production-v1",
            "retrieval_variant": "phase_b_adjacent_same_page_completion",
            "context_version": "phase-b-adjacent-same-page-v1",
            "gold": {
                "answerable": gold[question_id]["answerable"],
                "gold_paper_ids": gold[question_id]["gold_paper_ids"],
                "gold_pages": gold[question_id]["gold_pages"],
                "gold_block_ids": gold[question_id]["gold_block_ids"],
                "required_claims": gold[question_id]["required_claims"],
            },
            "status": "client_postprocessing_failed",
            "failure_reason": "claim_text/text deterministic adapter mismatch",
            "metrics": {},
            "request_attempt_count": 1,
            "provider_completed_request_count": None,
            "provider_completion_status": "unknown_not_durably_recorded",
            "usage_record_count": 0,
            "request_ids": [request_id],
            "usage": {},
            "usage_source": "unavailable_after_send_attempt",
            "active_reserved_tokens": 20000,
            "citation_retry_count": 0,
            "elapsed_seconds": 0,
            "monetary_cost_usd": "0",
            "cost_basis": "explicit_free_provider",
            "citation_validation": "not_durably_recorded",
            "reranker_called": False,
            "template_fallback": False,
        }
        (run_dir / "result.json").write_text(
            json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        with (run_dir / "result.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "run_id",
                    "question_id",
                    "variant",
                    "status",
                    "request_attempt_count",
                    "provider_completed_request_count",
                    "total_tokens",
                    "active_reserved_tokens",
                    "elapsed_seconds",
                    "monetary_cost_usd",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "run_id": run_id,
                    "question_id": question_id,
                    "variant": VARIANT_B,
                    "status": row["status"],
                    "request_attempt_count": 1,
                    "provider_completed_request_count": "unknown",
                    "total_tokens": "unknown",
                    "active_reserved_tokens": 20000,
                    "elapsed_seconds": 0,
                    "monetary_cost_usd": "0",
                }
            )
        metadata = {
            "schema_version": "evidence-qa-dev-run-v1",
            "run_id": run_id,
            "manifest_hash": manifest["manifest_hash"],
            "question_id": question_id,
            "variant": VARIANT_B,
            "recovered_orphan": True,
            "recovery_did_not_issue_request": True,
            "api_key_recorded": False,
            "headers_recorded": False,
            "deep_research_called": False,
        }
        (run_dir / "run-metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        recovered.append(question_id)
    return recovered


def load_contexts(question_id: str) -> tuple[list[ContextItem], dict[str, Any]]:
    baseline, candidate = phase_b_rows()
    base_triples = {tuple(item) for item in baseline[question_id]["metrics"]["citation_triples"]}
    selected_triples = [tuple(item) for item in candidate[question_id]["metrics"]["citation_triples"]]
    units = {
        (row["paper_id"], int(row["page"]), row["block_id"]): row
        for row in read_jsonl(DATA / "evidence-corpus-v1.jsonl")
    }
    contexts = []
    adjacent = []
    for rank, triple in enumerate(selected_triples, start=1):
        unit = units[triple]
        contexts.append(
            ContextItem(
                chunk_id=unit["evidence_id"],
                paper_id=unit["paper_id"],
                block_ids=[unit["block_id"]],
                block_page_map={unit["block_id"]: int(unit["page"])},
                section_path=[unit.get("section_title") or unit.get("section_id") or ""],
                page_start=int(unit["page"]),
                page_end=int(unit["page"]),
                evidence=unit["text"],
                score=1 / rank,
            )
        )
        if triple not in base_triples:
            neighbor = unit.get("previous_block_id") or unit.get("next_block_id")
            adjacent.append(
                {
                    "paper_id": unit["paper_id"],
                    "page": int(unit["page"]),
                    "block_id": unit["block_id"],
                    "original_neighboring_block": neighbor,
                    "inclusion_reason": "immediate_same_page_neighbor_of_top5_selected_evidence",
                }
            )
    trace = {
        "query_router_profile": "phase_b_routed_evidence_retrieval",
        "candidate_count": candidate[question_id]["metrics"]["candidate_count"],
        "selected_evidence_count": len(contexts),
        "adjacent_completion_blocks": adjacent,
        "selected_evidence_roles": [
            {"evidence_id": item.chunk_id, "roles": units[(item.paper_id, item.page_start, item.block_ids[0])]["evidence_roles"]}
            for item in contexts
        ],
        "claim_allocation": None,
        "unsupported_before_generation_claims": [],
        "context_tokens": candidate[question_id]["metrics"]["context_token_count"],
        "truncated_evidence": [],
        "allowed_citation_triples": [list(item) for item in selected_triples],
        "metadata_contamination": candidate[question_id]["metrics"]["metadata_contamination_rate"],
        "gold_used_for_selection": False,
        "oracle_used_for_selection": False,
        "human_pilot_used_for_selection": False,
    }
    return contexts, trace


def enriched_answer(answer: dict[str, Any], contexts: list[ContextItem]) -> dict[str, Any]:
    triple_to_evidence = {
        (item.paper_id, item.page_start, item.block_ids[0]): item.chunk_id for item in contexts
    }
    citations = []
    for claim in answer.get("claims", []):
        citation_ids = []
        assigned = []
        for citation in claim["citations"]:
            triple = (citation["paper_id"], int(citation["page"]), citation["block_id"])
            citation_id = "cit-" + uuid.uuid5(uuid.NAMESPACE_URL, repr(triple)).hex[:16]
            citation_ids.append(citation_id)
            assigned.append(triple_to_evidence[triple])
            citations.append({"citation_id": citation_id, **citation})
        claim["claim_text"] = claim.pop("text")
        claim["evidence_complete"] = bool(assigned)
        claim["assigned_evidence_ids"] = assigned
        claim["citation_ids"] = citation_ids
    return {**answer, "citations": citations}


def run_one(provider: SiliconFlowLLMProvider, settings: Settings, manifest: dict[str, Any], protocol: dict[str, Any], gold: dict[str, Any], question_id: str, preflight: dict[str, Any]) -> dict[str, Any]:
    run_id = f"live-{VARIANT_B}-{question_id}-{uuid.uuid4().hex[:12]}"
    run_dir = RUN_ROOT / VARIANT_B / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    contexts, context_trace = load_contexts(question_id)
    retrieval_trace = {
        "question_id": question_id,
        "retrieval_scope": protocol["retrieval_scope"],
        "retrieval_filter": protocol["retrieval_filter"],
        **context_trace,
    }
    (run_dir / "retrieval-trace.json").write_text(json.dumps(retrieval_trace, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "context-trace.json").write_text(json.dumps(context_trace, ensure_ascii=False, indent=2), encoding="utf-8")
    request_id = f"{run_id}:primary:1:{uuid.uuid4().hex[:12]}"
    ledger = run_dir / "request-ledger.jsonl"
    reservation = 20000
    prepared = {
        "event": "request_prepared",
        "request_id": request_id,
        "request_status": "prepared",
        "usage_status": "reserved",
        "reserved_tokens": reservation,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    append_jsonl(ledger, prepared)
    append_jsonl(ledger, {**prepared, "event": "request_started", "request_status": "started", "timestamp": datetime.now(UTC).isoformat()})
    started = time.perf_counter()
    base = {
        "run_id": run_id,
        "variant": VARIANT_B,
        "question_id": question_id,
        "category": gold["category"],
        "difficulty": gold["difficulty"],
        "retrieval_scope": protocol["retrieval_scope"],
        "retrieval_filter": protocol["retrieval_filter"],
        "prompt_version": "qa-production-v1",
        "retrieval_variant": "phase_b_adjacent_same_page_completion",
        "context_version": "phase-b-adjacent-same-page-v1",
        "gold": {
            "answerable": gold["answerable"],
            "gold_paper_ids": gold["gold_paper_ids"],
            "gold_pages": gold["gold_pages"],
            "gold_block_ids": gold["gold_block_ids"],
            "required_claims": gold["required_claims"],
        },
    }
    try:
        generated = provider.generate_claim_answer(protocol["retrieval_query"], contexts, "qa-production-v1")
        elapsed = time.perf_counter() - started
        if generated.usage.total_tokens <= 0:
            raise RuntimeError("provider completed without valid usage")
        if generated.usage.total_tokens > 20000 or elapsed > 180:
            raise RuntimeError("per-question budget exceeded")
        answer = enriched_answer(generated.model_dump(exclude={"usage", "first_token_latency_ms", "total_latency_ms", "raw_model", "api_request_count", "retry_count", "retry_reasons", "rate_limit_events", "diagnostic_attempts"}), contexts)
        allowed = {tuple(item) for item in context_trace["allowed_citation_triples"]}
        metrics = evaluate_answer(answer, gold, allowed)
        usage = generated.usage.model_dump()
        append_jsonl(
            ledger,
            {
                "event": "request_completed",
                "request_id": request_id,
                "request_status": "completed",
                "usage_status": "settled",
                "input_tokens": usage["input_tokens"],
                "output_tokens": usage["output_tokens"],
                "total_tokens": usage["total_tokens"],
                "active_reserved_tokens": 0,
                "usage_source": "provider_reported",
                "monetary_cost_usd": "0",
                "cost_basis": "explicit_free_provider",
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )
        row = {
            **base,
            "status": "completed",
            "answer": answer,
            "metrics": metrics,
            "request_attempt_count": 1,
            "provider_completed_request_count": 1,
            "usage_record_count": 1,
            "request_ids": [request_id],
            "usage": usage,
            "usage_source": "provider_reported",
            "active_reserved_tokens": 0,
            "citation_retry_count": 0,
            "elapsed_seconds": round(elapsed, 6),
            "monetary_cost_usd": "0",
            "cost_basis": "explicit_free_provider",
            "citation_validation": "passed",
            "reranker_called": False,
            "template_fallback": False,
        }
    except (LLMProviderError, RuntimeError) as exc:
        elapsed = time.perf_counter() - started
        append_jsonl(
            ledger,
            {
                "event": "request_failed",
                "request_id": request_id,
                "request_status": "failed_after_send_unknown",
                "usage_status": "reserved_conservative",
                "active_reserved_tokens": reservation,
                "failure_type": type(exc).__name__,
                "failure_message": str(exc)[:500],
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )
        row = {
            **base,
            "status": "provider_failed",
            "failure_reason": str(exc),
            "metrics": {},
            "request_attempt_count": 1,
            "provider_completed_request_count": 0,
            "usage_record_count": 0,
            "request_ids": [request_id],
            "usage": {},
            "usage_source": "unavailable_after_send_attempt",
            "active_reserved_tokens": reservation,
            "citation_retry_count": 0,
            "elapsed_seconds": round(elapsed, 6),
            "monetary_cost_usd": "0",
            "cost_basis": "explicit_free_provider",
            "citation_validation": "not_run",
            "reranker_called": False,
            "template_fallback": False,
        }
    metadata = {
        "schema_version": "evidence-qa-dev-run-v1",
        "run_id": run_id,
        "manifest_hash": manifest["manifest_hash"],
        "question_id": question_id,
        "variant": VARIANT_B,
        "preflight": preflight,
        "configuration": {
            "provider": "siliconflow",
            "model": "Qwen/Qwen3-8B",
            "temperature": 0,
            "llm_max_retries": 0,
            "billing_mode": "free",
            "rerank_enabled": False,
            "prompt_version": "qa-production-v1",
        },
        "source_hashes": manifest["artifacts"],
        "api_key_recorded": False,
        "headers_recorded": False,
        "deep_research_called": False,
    }
    (run_dir / "run-metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "result.json").write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    with (run_dir / "result.csv").open("w", encoding="utf-8", newline="") as stream:
        flat = {
            "run_id": run_id,
            "question_id": question_id,
            "variant": VARIANT_B,
            "status": row["status"],
            "request_attempt_count": row["request_attempt_count"],
            "provider_completed_request_count": row["provider_completed_request_count"],
            "total_tokens": row.get("usage", {}).get("total_tokens"),
            "active_reserved_tokens": row["active_reserved_tokens"],
            "elapsed_seconds": row["elapsed_seconds"],
            "monetary_cost_usd": row["monetary_cost_usd"],
        }
        writer = csv.DictWriter(stream, fieldnames=list(flat))
        writer.writeheader()
        writer.writerow(flat)
    return row


def main() -> int:
    args = parse_args()
    manifest = write_manifest()
    if args.prepare_only:
        print(json.dumps({"status": "prepared", "manifest_hash": manifest["manifest_hash"]}))
        return 0
    if not args.variant:
        raise SystemExit("--variant is required unless --prepare-only is used")
    if args.manifest.resolve() != MANIFEST.resolve():
        raise RuntimeError("only the frozen Stage 13.2 manifest is allowed")
    settings = Settings()
    preflight = safe_preflight(settings, args.variant)
    if args.mode != "live":
        print(json.dumps({"status": "dry_run_preflight_passed", "preflight": preflight}))
        return 0
    protocol = {row["question_id"]: row for row in read_jsonl(DATA / "retrieval-gold-v2.jsonl")}
    gold = {row["question_id"]: row for row in read_jsonl(DATA / "gold-set-v1.jsonl")}
    recovered = recover_orphaned_attempts(manifest, protocol, gold)
    attempted = {
        json.loads(path.read_text(encoding="utf-8"))["question_id"]
        for path in (RUN_ROOT / VARIANT_B).glob("*/result.json")
    }
    provider = SiliconFlowLLMProvider(
        settings.llm_base_url or "",
        settings.llm_api_key or "",
        settings.llm_model,
        temperature=0,
        timeout_seconds=min(settings.llm_timeout_seconds, 180),
        max_output_tokens=settings.llm_max_output_tokens,
        max_retries=0,
        input_cost_per_million=0,
        output_cost_per_million=0,
    )
    failures = 0
    consecutive = 0
    rows = []
    for question_id in DEV_IDS:
        if question_id in attempted:
            continue
        row = run_one(provider, settings, manifest, protocol[question_id], gold[question_id], question_id, preflight)
        rows.append(row)
        if row["status"] == "provider_failed":
            failures += 1
            consecutive += 1
        else:
            consecutive = 0
        if consecutive >= 2 or failures / len(DEV_IDS) > 0.20:
            print(json.dumps({"status": "DEV_QA_BLOCKED_BY_PROVIDER_CONNECTIVITY", "completed_rows": len(rows), "failures": failures}))
            return 2
        if sum(item.get("usage", {}).get("total_tokens", 0) + item.get("active_reserved_tokens", 0) for item in rows) > 300000:
            raise RuntimeError("global Dev token budget exceeded")
        if sum(item["elapsed_seconds"] for item in rows) > 1800:
            raise RuntimeError("global Dev elapsed budget exceeded")
    print(json.dumps({"status": "live_variant_completed", "variant": VARIANT_B, "new_runs": len(rows), "recovered_orphans": recovered, "skipped_previously_attempted": sorted(attempted), "failures": failures, "top_level_summary_modified": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
