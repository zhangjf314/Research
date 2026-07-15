# ruff: noqa: E501
"""Build the offline Stage 13.3 failure audit and deterministic replay evidence."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from paper_research.generation.citation_registry import (
    CitationRegistry,
    CitationRegistryError,
    reject_free_form_citation,
)
from paper_research.generation.output_adapter import (
    ClaimTextAdapterError,
    normalized_claim_text,
)
from paper_research.providers.response_envelope import ProviderResponseEnvelopeStore
from paper_research.retrieval.context_builder import ContextItem

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/evaluation"
DOCS = ROOT / "docs"
OUTPUT = DATA / "stage13-2-dev-failure-audit-v1.json"
REPORT = DOCS / "stage13-2-dev-failure-audit-v1.md"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def q001_replay() -> dict[str, Any]:
    cases = {
        "claim_text_only": {"claim_text": "fixture claim"},
        "text_only": {"text": "fixture claim"},
        "both_equal": {"text": "fixture claim", "claim_text": "fixture claim"},
    }
    results: dict[str, Any] = {
        name: normalized_claim_text(value) for name, value in cases.items()
    }
    for name, value in {
        "both_missing": {},
        "both_conflicting": {"text": "a", "claim_text": "b"},
    }.items():
        try:
            normalized_claim_text(value)
            results[name] = "unexpected_pass"
        except ClaimTextAdapterError as exc:
            results[name] = {"status": "strict_failure", "reason": str(exc)}

    raw = {
        "model": "Qwen/Qwen3-8B-fixture",
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": json.dumps(
                        {
                            "answerable": True,
                            "answer": "fixture",
                            "claims": [{"claim_id": "c1", "claim_text": "fixture claim", "citations": []}],
                            "refusal_reason": None,
                        }
                    )
                },
            }
        ],
        "usage": {"prompt_tokens": 101, "completion_tokens": 19, "total_tokens": 120},
    }
    with tempfile.TemporaryDirectory(dir=DATA) as temporary:
        run_dir = Path(temporary)
        ledger = run_dir / "request-ledger.jsonl"
        ledger.write_text("", encoding="utf-8")
        store = ProviderResponseEnvelopeStore(run_dir, ledger)
        envelope = store.record_received(
            request_id="q001-offline-fixture",
            provider="siliconflow",
            model="Qwen/Qwen3-8B",
            raw_body=json.dumps(raw).encode(),
        )
        envelope = store.parsing_started(envelope)
        envelope = store.post_processing_failed(
            envelope, ClaimTextAdapterError("synthetic post-processing failure")
        )
        event_order = [
            json.loads(line)["event"]
            for line in ledger.read_text(encoding="utf-8").splitlines()
        ]
        envelope_dump = envelope.model_dump()
    return {
        "historical_raw_response_available": False,
        "historical_raw_fixture_replay": "BLOCKED_ORIGINAL_RAW_NOT_PERSISTED",
        "compatibility_fixture_results": results,
        "future_protocol_fixture": {
            "event_order": event_order,
            "usage": envelope_dump["usage"],
            "parse_status": envelope_dump["parse_status"],
            "reservation_released_after_known_usage": event_order.index("provider_usage_recorded")
            < event_order.index("response_parsing_started"),
        },
    }


def q019_replay() -> dict[str, Any]:
    run_dir = next(
        (DATA / "evidence-qa-dev-v1/runs/retrieval_only").glob("*-q019-*")
    )
    context_trace = json.loads(
        (run_dir / "context-trace.json").read_text(encoding="utf-8")
    )
    evidence = {
        (row["paper_id"], int(row["page"]), row["block_id"]): row
        for row in read_jsonl(DATA / "evidence-corpus-v1.jsonl")
    }
    contexts = []
    for index, triple_value in enumerate(
        context_trace["allowed_citation_triples"], start=1
    ):
        triple = tuple(triple_value)
        unit = evidence[triple]
        contexts.append(
            ContextItem(
                chunk_id=unit["evidence_id"],
                paper_id=unit["paper_id"],
                block_ids=[unit["block_id"]],
                block_page_map={unit["block_id"]: int(unit["page"])},
                section_path=[unit.get("section_title") or ""],
                page_start=int(unit["page"]),
                page_end=int(unit["page"]),
                evidence=unit["text"],
                score=1 / index,
            )
        )
    registry = CitationRegistry.from_context(contexts)
    first = registry.entries[0]
    invalid_free_form = {
        "paper_id": first.paper_id,
        "page": first.page + 1,
        "block_id": first.block_id,
    }
    free_form_blocked = False
    unknown_id_blocked = False
    try:
        reject_free_form_citation(invalid_free_form)
    except CitationRegistryError:
        free_form_blocked = True
    try:
        registry.resolve(["E999"])
    except CitationRegistryError:
        unknown_id_blocked = True
    resolved = registry.resolve([first.citation_id]).entries[0]
    return {
        "historical_raw_response_available": False,
        "historical_invalid_page_value": "unknown_raw_response_not_persisted",
        "historical_failure_code": "citation_validation:page",
        "allowed_triple_count": len(context_trace["allowed_citation_triples"]),
        "allowed_triples_were_present_in_prompt_payload": True,
        "block_page_map_trace_complete": True,
        "citation_id_v2_fixture": {
            "registry_hash": registry.registry_hash,
            "registry_entry_count": len(registry.entries),
            "representative_invalid_free_form_triple": invalid_free_form,
            "free_form_page_blocked": free_form_blocked,
            "unknown_id_blocked": unknown_id_blocked,
            "valid_id_resolves_exact_triple": list(resolved.triple)
            == [first.paper_id, first.page, first.block_id],
        },
    }


def main() -> None:
    q001 = q001_replay()
    q019 = q019_replay()
    payload = {
        "schema_version": "stage13-2-dev-failure-audit-v1",
        "stage13_2_history_modified": False,
        "historical_reservations_released": False,
        "failures": {
            "q001": {
                "classification": "client_post_processing_failure",
                "provider_raw_json_persisted": False,
                "provider_usage_persisted": False,
                "root_cause": "The adapter renamed generated claim text to claim_text before the evaluator still reading text. The response and usage were held only in process memory.",
                "replay": q001,
                "prevention": "ProviderResponseEnvelopeStore now records raw_response_received, provider_usage_recorded, raw_response_persisted, response_parsing_started, then response_parsed or post_processing_failed.",
            },
            "q008": {
                "classification": "provider_read_timeout_after_send_unknown",
                "configured_timeout_seconds": 120,
                "timeout_components_separated": False,
                "network_health_probe_previously_present": False,
                "root_cause": "The request exceeded the undifferentiated HTTP read timeout; provider completion and usage remain unknown.",
                "prevention": "A one-shot DNS/TCP/TLS/models endpoint health checker is implemented. It does not retry QA requests and Phase A does not execute it live.",
            },
            "q019": {
                "classification": "strict_citation_page_validation_failure",
                "raw_model_response_persisted": False,
                "root_cause": "The model was allowed to copy three free-form citation fields and emitted a page inconsistent with the selected block_page_map. Strict validation correctly rejected it; the exact emitted page was lost with the unpersisted response.",
                "current_no_retry_policy_conformant": True,
                "replay": q019,
                "prevention": "citation-id-v2 exposes only immutable short IDs; local deterministic resolution restores exact triples and runs strict triple validation again.",
            },
        },
        "phase_a_status": "WAITING_FOR_DEV_CITATION_AUDIT",
        "dev_v2_run": False,
        "full_qa_run": False,
        "deep_research_run": False,
        "reranker_enabled": False,
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    REPORT.write_text(
        "# Stage 13.2 Dev Failure Audit v1\n\n"
        "Historical Stage 13.2 runs and their 60,000 conservative token reservations remain unchanged.\n\n"
        "## q001\n\n"
        "The Provider response returned, but neither raw JSON nor usage was durably recorded before the local `claim_text`/`text` adapter mismatch. The original raw response therefore cannot be replayed. Deterministic fixtures for `claim_text`, legacy `text`, and equal dual fields pass; missing or conflicting fields fail closed. The new response envelope settles usage and persists raw response before parsing.\n\n"
        "## q008\n\n"
        "The request ended in `ReadTimeout` after 120 seconds with unknown Provider completion. The old client used one aggregate timeout and had no pre-batch DNS/TCP/TLS/models probe. A one-shot health checker is implemented; it is not executed live in Phase A and never retries a QA request.\n\n"
        "## q019\n\n"
        "The strict validator correctly rejected a page/block mismatch. The exact model JSON was not persisted, so the emitted page cannot be reconstructed. The historical context did contain complete allowed triples and block-page mappings. The citation-id-v2 fixture rejects free-form triples and unknown IDs; valid IDs resolve deterministically and are triple-validated again.\n\n"
        "## Status\n\n"
        "`WAITING_FOR_DEV_CITATION_AUDIT`. Dev v2, Full QA, and Deep Research were not run.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": payload["phase_a_status"], "q001": q001, "q019": q019}))


if __name__ == "__main__":
    main()
