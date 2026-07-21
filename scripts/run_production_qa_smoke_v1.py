"""Run a single Stage 13.30 Docker API QA smoke test.

The script calls the running API through http://localhost/api/v1. It does not
read or print API keys. It intentionally runs exactly one approved answerable
sample and persists the result for audit before any Full QA batch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "evaluation"
DOCS = ROOT / "docs"
ARTIFACTS = ROOT / "artifacts"

RETRIEVAL_GOLD = DATA / "retrieval-gold-v2.jsonl"
GOLD = DATA / "gold-set-v1.jsonl"
SMOKE_JSON = ARTIFACTS / "live-model-smoke-test-v1.json"
SMOKE_DOC = DOCS / "live-model-smoke-test-v1.md"
REPRO_JSON = ARTIFACTS / "stage13-31-q002-reproduction-input.json"
Q017_RETRY_JSON = ARTIFACTS / "q017-live-retry-result-v1.json"
Q019_RETRY_JSON = ARTIFACTS / "q019-live-retry-result-v1.json"
PRODUCTION_CORPUS = DATA / "production-corpus-v1.json"

API_BASE = "http://localhost/api/v1"
SMOKE_QUESTION_ID = "q002"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def find_record(path: Path, question_id: str) -> dict[str, Any]:
    for row in read_jsonl(path):
        if row["question_id"] == question_id:
            return row
    raise RuntimeError(f"question not found: {question_id}")


def require_budget_ready(client: httpx.Client) -> dict[str, Any]:
    capabilities = client.get(f"{API_BASE}/capabilities").raise_for_status()
    payload = capabilities.json()
    budget = payload.get("stage13_30_budget") or {}
    if not budget.get("full_qa_budget_ready"):
        raise RuntimeError(f"container budget is not ready: {budget.get('status')}")
    return payload


def find_paper_uuid(client: httpx.Client, paper_id: str) -> str:
    manifest = json.loads(PRODUCTION_CORPUS.read_text(encoding="utf-8"))
    for paper in manifest.get("papers", []):
        if paper.get("included_in_production") and paper.get("paper_id") == paper_id:
            return str(paper["database_id"])
    response = client.get(f"{API_BASE}/papers")
    if response.status_code != 422:
        response.raise_for_status()
        papers = response.json()
        if isinstance(papers, dict):
            papers = papers.get("items") or papers.get("value") or []
        for paper in papers:
            if paper.get("arxiv_id") == paper_id or paper.get("title") == paper_id:
                return paper["id"]
    raise RuntimeError(f"paper UUID not found for {paper_id}")


def validate_citations(answer: dict[str, Any], context: list[dict[str, Any]]) -> dict[str, Any]:
    allowed = set()
    for item in context:
        block_ids = item.get("block_ids") or [item["chunk_id"]]
        block_page_map = item.get("block_page_map") or {
            block_id: item["page_start"] for block_id in block_ids
        }
        allowed.update(
            (item["paper_id"], block_page_map[block_id], block_id) for block_id in block_ids
        )
    citations = [
        citation
        for claim in answer.get("claims", [])
        for citation in claim.get("citations", [])
    ]
    valid = [
        (citation["paper_id"], citation["page"], citation["block_id"]) in allowed
        for citation in citations
    ]
    return {
        "citation_count": len(citations),
        "citation_context_validity": sum(valid) / len(valid) if valid else 0,
        "all_citations_in_retrieved_context": bool(citations) and all(valid),
        "citation_pages": [citation["page"] for citation in citations],
    }


def safe_context(context: list[dict[str, Any]]) -> list[dict[str, Any]]:
    safe = []
    for rank, item in enumerate(context, start=1):
        text = item.get("evidence") or ""
        safe.append(
            {
                "rank": rank,
                "chunk_id": item.get("chunk_id"),
                "paper_id": item.get("paper_id"),
                "block_ids": item.get("block_ids") or [item.get("chunk_id")],
                "page_start": item.get("page_start"),
                "page_end": item.get("page_end"),
                "block_page_map": item.get("block_page_map") or {},
                "score": item.get("score"),
                "section_path": item.get("section_path"),
                "context_length": len(text),
                "context_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "snippet": text[:240],
            }
        )
    return safe


def write_reproduction_input(
    *,
    sample_id: str,
    retrieval_record: dict[str, Any],
    gold_record: dict[str, Any],
    paper_uuid: str,
    retrieve_payload: dict[str, Any],
    qa_payload: dict[str, Any],
    retrieval: dict[str, Any],
    capabilities: dict[str, Any],
    request_id: str | None = None,
) -> None:
    capability_items = capabilities.get("capabilities") or {}
    provider = capability_items.get("llm") or {}
    embedding = capability_items.get("embedding") or {}
    output = {
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
        "prompt_version": provider.get("prompt_version") or capabilities.get("prompt_version"),
        "schema_version_runtime": "StructuredQA",
        "llm_params": {
            "provider_detail": provider.get("detail"),
            "available": provider.get("available"),
            "status": provider.get("status"),
        },
        "api_profile": capabilities.get("profile"),
        "collection": embedding.get("detail"),
        "request_id": request_id,
        "created_at": datetime.now(UTC).isoformat(),
        "safety": {
            "api_key_persisted": False,
            "authorization_header_persisted": False,
            "full_paper_text_persisted": False,
        },
    }
    REPRO_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPRO_JSON.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")


def classify_failure(status_code: int, detail: Any) -> str:
    if isinstance(detail, dict):
        code = str(detail.get("code") or "")
        if code == "CLAIM_QA_CONFIGURATION_ERROR":
            return "BLOCKED_BY_RUNTIME_CONFIGURATION"
        if code == "CLAIM_QA_JSON_PARSE_ERROR":
            return "BLOCKED_BY_JSON_PARSE"
        if code == "CLAIM_QA_SCHEMA_VALIDATION_ERROR":
            return "BLOCKED_BY_SCHEMA_VALIDATION"
        if code == "CLAIM_QA_CITATION_VALIDATION_ERROR":
            return "BLOCKED_BY_CITATION_VALIDATION"
        if code == "CLAIM_QA_PROVIDER_RESPONSE_ERROR":
            return "BLOCKED_BY_PROVIDER_RESPONSE_FORMAT"
    if status_code == 503:
        return "BLOCKED_BY_INTERNAL_ERROR"
    return "BLOCKED_BY_INTERNAL_ERROR"


def extract_error_detail(error_body: Any) -> Any:
    if not isinstance(error_body, dict):
        return error_body
    if "detail" in error_body:
        return error_body.get("detail")
    error = error_body.get("error")
    if isinstance(error, dict) and "message" in error:
        return error.get("message")
    return error_body


def write_outputs(output: dict[str, Any]) -> None:
    usage = output.get("model_usage") or {}
    validation = output.get("citation_validation") or {}
    latency = output.get("latency") or {}
    cost = output.get("cost") or {}
    SMOKE_JSON.parent.mkdir(parents=True, exist_ok=True)
    SMOKE_JSON.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    if output.get("sample_id") == "q017":
        Q017_RETRY_JSON.write_text(
            json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    if output.get("sample_id") == "q019":
        Q019_RETRY_JSON.write_text(
            json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    SMOKE_DOC.write_text(
        "# Live Model Smoke Test v1\n\n"
        f"- Status: `{output['status']}`\n"
        f"- Sample ID: `{output.get('sample_id')}`\n"
        f"- Provider/model: `{output.get('provider')}`/`{output.get('model')}`\n"
        f"- Real model called: `{output.get('real_model_called')}`\n"
        f"- Prompt version: `{output.get('prompt_version')}`\n"
        f"- Error type: `{output.get('error_type')}`\n"
        f"- Error detail: `{output.get('error_detail')}`\n"
        f"- Tokens input/output/total: `{usage.get('input_tokens')}`/"
        f"`{usage.get('output_tokens')}`/`{usage.get('total_tokens')}`\n"
        f"- Cost status: `{cost.get('cost_status')}`\n"
        f"- Estimated cost USD: `{cost.get('estimated_cost_usd')}`\n"
        f"- Claim count: `{output.get('claim_count')}`\n"
        f"- Citation count: `{validation.get('citation_count')}`\n"
        "- Citation in retrieved context: "
        f"`{validation.get('all_citations_in_retrieved_context')}`\n"
        f"- Retrieval endpoint latency ms: `{latency.get('retrieve_endpoint_ms')}`\n"
        f"- QA endpoint latency ms: `{latency.get('qa_endpoint_ms')}`\n"
        "\nNo API key or authorization header is persisted.\n",
        encoding="utf-8",
    )


def main() -> int:
    global API_BASE
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-id", default=SMOKE_QUESTION_ID)
    parser.add_argument("--api-base", default=API_BASE)
    parser.add_argument("--single-attempt", action="store_true")
    parser.add_argument("--no-json-repair", action="store_true")
    parser.add_argument("--no-qa-retry", action="store_true")
    args = parser.parse_args()
    API_BASE = args.api_base.rstrip("/")
    sample_id = args.sample_id
    if sample_id not in {"q002", "q017", "q019", "q024"}:
        raise RuntimeError(
            "production QA smoke is intentionally limited to q002/q017/q019/q024"
        )
    if sample_id in {"q017", "q019", "q024"} and not (
        args.single_attempt and args.no_json_repair and args.no_qa_retry
    ):
        raise RuntimeError(
            f"{sample_id} retry requires --single-attempt --no-json-repair --no-qa-retry"
        )

    retrieval_record = find_record(RETRIEVAL_GOLD, sample_id)
    gold_record = find_record(GOLD, sample_id)
    if retrieval_record["review_status"] != "approved" or not gold_record["answerable"]:
        raise RuntimeError("smoke sample must be approved and answerable")
    if not gold_record["gold_block_ids"]:
        raise RuntimeError("smoke sample must have gold evidence")

    started = time.perf_counter()
    with httpx.Client(timeout=180) as client:
        capabilities = require_budget_ready(client)
        paper_uuid = find_paper_uuid(client, retrieval_record["gold_paper_ids"][0])
        retrieve_payload = {
            "query": retrieval_record["retrieval_query"],
            "filters": {"paper_ids": [paper_uuid]},
            "recall_k": 20,
            "top_k": 10,
        }
        retrieve_started = time.perf_counter()
        retrieve_response = client.post(f"{API_BASE}/retrieve", json=retrieve_payload)
        retrieve_latency_ms = round((time.perf_counter() - retrieve_started) * 1000, 3)
        retrieve_response.raise_for_status()
        retrieval = retrieve_response.json()

        qa_payload = {
            "question": retrieval_record["retrieval_query"],
            "paper_ids": [paper_uuid],
            "top_k": 10,
            "sample_id": sample_id,
            "run_id": f"live-retry-{sample_id}-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}",
        }
        write_reproduction_input(
            sample_id=sample_id,
            retrieval_record=retrieval_record,
            gold_record=gold_record,
            paper_uuid=paper_uuid,
            retrieve_payload=retrieve_payload,
            qa_payload=qa_payload,
            retrieval=retrieval,
            capabilities=capabilities,
            request_id=None,
        )
        qa_started = time.perf_counter()
        qa_response = client.post(f"{API_BASE}/qa", json=qa_payload)
        qa_latency_ms = round((time.perf_counter() - qa_started) * 1000, 3)
        if qa_response.status_code >= 400:
            raw_error_body = qa_response.text
            try:
                error_body: Any = qa_response.json()
            except json.JSONDecodeError:
                error_body = raw_error_body
            error_detail = extract_error_detail(error_body)
            output = {
                "schema_version": "live-model-smoke-test-v1",
                "status": "FAILED",
                "sample_id": sample_id,
                "question": retrieval_record["retrieval_query"],
                "paper_id": retrieval_record["gold_paper_ids"][0],
                "paper_uuid": paper_uuid,
                "real_model_called": None,
                "real_model_call_status": "unknown_after_api_503",
                "model": None,
                "provider": None,
                "prompt_version": None,
                "answerable": None,
                "claim_count": 0,
                "retrieved_top_k": len(retrieval["context"]),
                "retrieved_block_ids": [
                    block_id
                    for item in retrieval["context"]
                    for block_id in (item.get("block_ids") or [item["chunk_id"]])
                ],
                "retrieval_scores": [item.get("score") for item in retrieval["context"]],
                "context_blocks": safe_context(retrieval["context"]),
                "citation_validation": {
                    "citation_count": 0,
                    "citation_context_validity": 0,
                    "all_citations_in_retrieved_context": False,
                    "citation_pages": [],
                },
                "model_usage": {
                    "input_tokens": None,
                    "output_tokens": None,
                    "total_tokens": None,
                },
                "cost": {
                    "estimated_cost_usd": None,
                    "cost_type": "unknown",
                    "cost_status": "unknown",
                },
                "latency": {
                    "retrieve_endpoint_ms": retrieve_latency_ms,
                    "qa_endpoint_ms": qa_latency_ms,
                    "api_reported_total_ms": None,
                    "api_reported_first_token_ms": None,
                    "wall_total_ms": round((time.perf_counter() - started) * 1000, 3),
                },
                "capabilities_budget_status": capabilities.get("stage13_30_budget"),
                "error_type": f"HTTP_{qa_response.status_code}",
                "error_detail": error_detail,
                "error_body": error_body,
                "raw_error_body": raw_error_body,
                "failure_classification": classify_failure(qa_response.status_code, error_detail),
                "created_at": datetime.now(UTC).isoformat(),
            }
            write_outputs(output)
            print(
                json.dumps(
                    {
                        "status": output["status"],
                        "sample_id": output["sample_id"],
                        "error_type": output["error_type"],
                    }
                )
            )
            return 2
        qa_response.raise_for_status()
        answer = qa_response.json()

    usage = answer.get("model_usage") or {}
    validation = validate_citations(answer, retrieval["context"])
    estimated_cost = usage.get("estimated_cost_usd")
    output = {
        "schema_version": "live-model-smoke-test-v1",
        "status": "PASSED"
        if (
            answer.get("provider") != "template"
            and answer.get("answerable") is True
            and len(answer.get("claims", [])) >= 1
            and usage.get("total_tokens", 0) > 0
            and validation["all_citations_in_retrieved_context"]
        )
        else "FAILED",
        "sample_id": sample_id,
        "question": retrieval_record["retrieval_query"],
        "paper_id": retrieval_record["gold_paper_ids"][0],
        "paper_uuid": paper_uuid,
        "real_model_called": answer.get("provider") != "template",
        "model": answer.get("model"),
        "provider": answer.get("provider"),
        "prompt_version": answer.get("prompt_version"),
        "answerable": answer.get("answerable"),
        "claim_count": len(answer.get("claims", [])),
        "retrieved_top_k": len(retrieval["context"]),
        "retrieved_block_ids": [
            block_id
            for item in retrieval["context"]
            for block_id in (item.get("block_ids") or [item["chunk_id"]])
        ],
        "retrieval_scores": [item.get("score") for item in retrieval["context"]],
        "context_blocks": safe_context(retrieval["context"]),
        "citation_validation": validation,
        "model_usage": usage,
        "cost": {
            "estimated_cost_usd": estimated_cost,
            "cost_type": "estimated" if estimated_cost is not None else "unknown",
            "cost_status": "known" if estimated_cost is not None else "unknown",
        },
        "latency": {
            "retrieve_endpoint_ms": retrieve_latency_ms,
            "qa_endpoint_ms": qa_latency_ms,
            "api_reported_total_ms": (answer.get("latency") or {}).get("total_latency_ms"),
            "api_reported_first_token_ms": (answer.get("latency") or {}).get(
                "llm_first_token_latency_ms"
            ),
            "wall_total_ms": round((time.perf_counter() - started) * 1000, 3),
        },
        "capabilities_budget_status": capabilities.get("stage13_30_budget"),
        "failure_classification": None,
        "created_at": datetime.now(UTC).isoformat(),
    }
    if output["status"] == "FAILED":
        if answer.get("provider") == "template":
            output["failure_classification"] = "BLOCKED_BY_RUNTIME_CONFIGURATION"
        elif answer.get("answerable") is False and gold_record.get("answerable") is True:
            output["failure_classification"] = "BLOCKED_BY_CONTEXT_DATA"
        elif not validation["all_citations_in_retrieved_context"]:
            output["failure_classification"] = "BLOCKED_BY_CITATION_VALIDATION"
        else:
            output["failure_classification"] = "BLOCKED_BY_MODEL_OUTPUT_QUALITY"
    write_outputs(output)
    print(json.dumps({k: output[k] for k in ("status", "sample_id", "real_model_called")}))
    return 0 if output["status"] == "PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
