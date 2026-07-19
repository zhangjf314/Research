"""Stage 13.31 bounded Claim QA diagnostics.

Default modes are offline. ``provider-call`` requires ``--allow-live-call`` and
persists only sanitized provider response structure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from pydantic import ValidationError

from paper_research.config import Settings
from paper_research.providers.factory import build_llm_provider
from paper_research.providers.llm import (
    LLMProviderError,
    StructuredQA,
    classify_json_parse_failure,
    normalize_structured_qa_content,
)
from paper_research.retrieval.context_builder import ContextItem

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "evaluation"
ARTIFACTS = ROOT / "artifacts"
DOCS = ROOT / "docs"
RETRIEVAL_GOLD = DATA / "retrieval-gold-v2.jsonl"
GOLD = DATA / "gold-set-v1.jsonl"
REPRO_JSON = ARTIFACTS / "stage13-31-q002-reproduction-input.json"
PROVIDER_AUDIT_JSON = ARTIFACTS / "stage13-31-q002-provider-response-audit.json"
CONTEXT_DOC = DOCS / "stage13-31-q002-context-audit.md"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def find_record(path: Path, question_id: str) -> dict[str, Any]:
    for row in read_jsonl(path):
        if row["question_id"] == question_id:
            return row
    raise RuntimeError(f"question not found: {question_id}")


def find_paper_uuid(client: httpx.Client, api_base: str, paper_id: str) -> str:
    response = client.get(f"{api_base}/papers", params={"limit": 100})
    response.raise_for_status()
    papers = response.json()
    if isinstance(papers, dict):
        papers = papers.get("items") or papers.get("value") or []
    for paper in papers:
        if paper.get("arxiv_id") == paper_id or paper.get("title") == paper_id:
            return paper["id"]
    raise RuntimeError(f"paper UUID not found for {paper_id}")


def safe_context(context: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for rank, item in enumerate(context, start=1):
        text = item.get("evidence") or ""
        rows.append(
            {
                "rank": rank,
                "chunk_id": item.get("chunk_id"),
                "paper_id": item.get("paper_id"),
                "block_ids": item.get("block_ids") or [item.get("chunk_id")],
                "page_start": item.get("page_start"),
                "page_end": item.get("page_end"),
                "score": item.get("score"),
                "section_path": item.get("section_path"),
                "context_length": len(text),
                "context_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "snippet": text[:240],
            }
        )
    return rows


def build_context(api_base: str, sample_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    retrieval_record = find_record(RETRIEVAL_GOLD, sample_id)
    gold_record = find_record(GOLD, sample_id)
    with httpx.Client(timeout=180) as client:
        capabilities = client.get(f"{api_base}/capabilities")
        capabilities.raise_for_status()
        paper_uuid = find_paper_uuid(client, api_base, retrieval_record["gold_paper_ids"][0])
        retrieve_payload = {
            "query": retrieval_record["retrieval_query"],
            "filters": {"paper_ids": [paper_uuid]},
            "recall_k": 20,
            "top_k": 10,
        }
        response = client.post(f"{api_base}/retrieve", json=retrieve_payload)
        response.raise_for_status()
        retrieval = response.json()
    qa_payload = {
        "question": retrieval_record["retrieval_query"],
        "paper_ids": [paper_uuid],
        "top_k": 10,
    }
    reproduction = {
        "schema_version": "stage13-31-q002-reproduction-input-v1",
        "sample_id": sample_id,
        "question": retrieval_record["retrieval_query"],
        "original_question": retrieval_record.get("original_question"),
        "retrieval_scope": retrieval_record.get("retrieval_scope"),
        "retrieval_filter": retrieval_record.get("retrieval_filter"),
        "gold_paper_ids": retrieval_record.get("gold_paper_ids"),
        "gold_pages": retrieval_record.get("gold_pages"),
        "gold_block_ids": retrieval_record.get("gold_block_ids"),
        "answerable": gold_record.get("answerable"),
        "paper_uuid": paper_uuid,
        "retrieve_payload": retrieve_payload,
        "qa_payload": qa_payload,
        "retrieved_top_k": len(retrieval.get("context") or []),
        "retrieved_context": safe_context(retrieval.get("context") or []),
        "capabilities": {
            "app_profile": capabilities.json().get("profile"),
            "version": capabilities.json().get("version"),
            "stage13_30_budget": capabilities.json().get("stage13_30_budget"),
        },
        "created_at": datetime.now(UTC).isoformat(),
        "safety": {
            "api_key_persisted": False,
            "authorization_header_persisted": False,
            "full_paper_text_persisted": False,
        },
    }
    return reproduction, retrieval


def write_context_outputs(reproduction: dict[str, Any]) -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    REPRO_JSON.write_text(json.dumps(reproduction, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = reproduction["retrieved_context"]
    CONTEXT_DOC.write_text(
        "# Stage 13.31 q002 Context Audit\n\n"
        f"- Sample: `{reproduction['sample_id']}`\n"
        f"- Retrieval scope: `{reproduction.get('retrieval_scope')}`\n"
        f"- Gold paper IDs: `{reproduction.get('gold_paper_ids')}`\n"
        f"- Gold pages: `{reproduction.get('gold_pages')}`\n"
        f"- Gold block IDs: `{reproduction.get('gold_block_ids')}`\n"
        f"- Retrieved context count: `{len(rows)}`\n"
        "- Stored evidence: metadata, hashes, lengths, and short snippets only.\n\n"
        "| rank | page | score | block ids | text sha256 | length |\n"
        "|---:|---:|---:|---|---|---:|\n"
        + "\n".join(
            f"| {row['rank']} | {row['page_start']} | {row['score']} | "
            f"`{','.join(str(x) for x in row['block_ids'])}` | "
            f"`{row['context_sha256'][:16]}` | {row['context_length']} |"
            for row in rows
        )
        + "\n",
        encoding="utf-8",
    )


def context_items(raw_context: list[dict[str, Any]]) -> list[ContextItem]:
    return [ContextItem.model_validate(item) for item in raw_context]


def mode_provider_call(args: argparse.Namespace) -> int:
    if not args.allow_live_call:
        raise RuntimeError("provider-call requires --allow-live-call")
    reproduction, retrieval = build_context(args.api_base, args.sample_id)
    write_context_outputs(reproduction)
    settings = Settings()
    provider = build_llm_provider(settings)
    try:
        result = provider.generate_claim_answer(
            reproduction["question"],
            context_items(retrieval.get("context") or []),
            settings.prompt_version,
        )
        audit = {
            "schema_version": "stage13-31-provider-response-audit-v1",
            "sample_id": args.sample_id,
            "status": "PASSED",
            "provider": provider.provider_name,
            "model": provider.model_name,
            "prompt_version": settings.prompt_version,
            "api_request_count": result.api_request_count,
            "usage": result.usage.model_dump(),
            "diagnostic_attempts": result.diagnostic_attempts,
            "answerable": result.answerable,
            "claim_count": len(result.claims),
            "created_at": datetime.now(UTC).isoformat(),
            "safety": {"api_key_persisted": False, "authorization_header_persisted": False},
        }
    except LLMProviderError as exc:
        audit = {
            "schema_version": "stage13-31-provider-response-audit-v1",
            "sample_id": args.sample_id,
            "status": "FAILED",
            "provider": provider.provider_name,
            "model": provider.model_name,
            "prompt_version": settings.prompt_version,
            "error_code": exc.error_code,
            "stage": exc.stage,
            "message": str(exc),
            "api_request_count": exc.api_request_count,
            "retry_reasons": exc.retry_reasons,
            "diagnostic_attempts": exc.diagnostic_attempts,
            "created_at": datetime.now(UTC).isoformat(),
            "safety": {"api_key_persisted": False, "authorization_header_persisted": False},
        }
    PROVIDER_AUDIT_JSON.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"status": audit["status"], "sample_id": args.sample_id}))
    return 0 if audit["status"] == "PASSED" else 2


def mode_parse_only(args: argparse.Namespace) -> int:
    source = Path(args.response_audit_file or args.input)
    payload = json.loads(source.read_text(encoding="utf-8"))
    content: Any = None
    partial = False
    if isinstance(payload, dict) and payload.get("schema_version") == "qa-response-audit-v1":
        content = payload.get("full_payload_sanitized")
        partial = not bool(content)
        if partial:
            content = (
                str(payload.get("content_prefix_sanitized") or "")
                + str(payload.get("content_suffix_sanitized") or "")
            )
    elif isinstance(payload, dict):
        content = payload.get("content", payload)
    else:
        content = payload
    stages: dict[str, Any] = {
        "response_extract": {
            "status": "partial" if partial else "complete",
            "source": str(source),
        }
    }
    if partial:
        stages.update(
            {
                "json_normalization": {"status": "blocked"},
                "json_parse": {"status": "blocked"},
                "schema_validation": {"status": "blocked"},
                "reference_resolution": {"status": "blocked"},
                "citation_validation": {"status": "blocked"},
            }
        )
        print(
            json.dumps(
                {
                    "status": "PARSE_REPLAY_BLOCKED_BY_PARTIAL_PAYLOAD",
                    "sample_id": args.sample_id,
                    "stages": stages,
                },
                ensure_ascii=False,
            )
        )
        return 2
    try:
        parsed, events = normalize_structured_qa_content(str(content))
        stages["json_normalization"] = {"status": "passed", "events": events}
        stages["json_parse"] = {"status": "passed", "fields": sorted(parsed)}
    except json.JSONDecodeError as exc:
        stages["json_normalization"] = {"status": "failed"}
        stages["json_parse"] = {
            "status": "failed",
            **classify_json_parse_failure(str(content), exc),
        }
        stages["schema_validation"] = {"status": "blocked"}
        stages["reference_resolution"] = {"status": "blocked"}
        stages["citation_validation"] = {"status": "blocked"}
        print(json.dumps({"status": "failed", "sample_id": args.sample_id, "stages": stages}))
        return 2
    try:
        StructuredQA.model_validate(parsed)
    except ValidationError as exc:
        stages["schema_validation"] = {"status": "failed", "errors": exc.errors()}
        stages["reference_resolution"] = {"status": "blocked"}
        stages["citation_validation"] = {"status": "blocked"}
        print(json.dumps({"status": "failed", "sample_id": args.sample_id, "stages": stages}))
        return 2
    stages["schema_validation"] = {"status": "passed"}
    stages["reference_resolution"] = {"status": "not_run_without_context"}
    stages["citation_validation"] = {"status": "not_run_without_context"}
    print(json.dumps({"status": "parsed", "sample_id": args.sample_id, "stages": stages}))
    return 0


def mode_validate_only(args: argparse.Namespace) -> int:
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    try:
        StructuredQA.model_validate(payload)
    except ValidationError as exc:
        print(
            json.dumps(
                {"status": "failed", "stage": "CLAIM_SCHEMA_VALIDATE", "errors": exc.errors()}
            )
        )
        return 2
    print(json.dumps({"status": "passed", "stage": "CLAIM_SCHEMA_VALIDATE"}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("context-only", "provider-call", "parse-only", "validate-only"),
        required=True,
    )
    parser.add_argument("--sample-id", default="q002")
    parser.add_argument("--api-base", default="http://localhost/api/v1")
    parser.add_argument("--input")
    parser.add_argument("--response-audit-file")
    parser.add_argument("--allow-live-call", action="store_true")
    args = parser.parse_args()
    args.api_base = args.api_base.rstrip("/")
    if args.sample_id not in {"q002", "q017"}:
        raise RuntimeError("diagnostics are intentionally limited to q002/q017")
    if args.mode == "context-only":
        reproduction, _retrieval = build_context(args.api_base, args.sample_id)
        write_context_outputs(reproduction)
        print(json.dumps({"status": "context_built", "sample_id": args.sample_id}))
        return 0
    if args.mode == "provider-call":
        return mode_provider_call(args)
    if args.mode == "parse-only":
        if not args.input and not args.response_audit_file:
            raise RuntimeError("parse-only requires --input or --response-audit-file")
        return mode_parse_only(args)
    if args.mode == "validate-only":
        if not args.input:
            raise RuntimeError("validate-only requires --input")
        return mode_validate_only(args)
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
