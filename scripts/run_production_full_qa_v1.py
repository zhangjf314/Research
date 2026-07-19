"""Run the Stage 13.31 Production Full QA batch through the Docker API.

This runner intentionally uses http://localhost/api/v1 so the already-validated
container runtime owns Embedding, Qdrant retrieval, and LLM provider behavior.
It does not read or persist API keys.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "evaluation"
DOCS = ROOT / "docs"
ARTIFACTS = ROOT / "artifacts"
GOLD = DATA / "gold-set-v1.jsonl"
RETRIEVAL_GOLD = DATA / "retrieval-gold-v2.jsonl"
PRODUCTION_CORPUS = DATA / "production-corpus-v1.json"
FULL_QA_JSON = DATA / "full-qa-production-v1.json"
FULL_QA_CSV = DATA / "full-qa-production-v1.csv"
FULL_QA_ITEMS = DATA / "full-qa-production-items-v1.jsonl"
FULL_QA_TRACE = ARTIFACTS / "full-qa-production-trace-v1.json"
FULL_QA_AUDIT_DOC = DOCS / "full-qa-production-audit-v1.md"
FULL_QA_SUMMARY_DOC = DOCS / "full-qa-production-summary-v1.md"
SMOKE_JSON = ARTIFACTS / "live-model-smoke-test-v1.json"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[max(0, math.ceil(len(ordered) * fraction) - 1)], 3)


def mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def load_existing_items(resume: bool) -> dict[str, dict[str, Any]]:
    if not resume or not FULL_QA_ITEMS.exists():
        return {}
    rows = read_jsonl(FULL_QA_ITEMS)
    return {row["question_id"]: row for row in rows if row.get("status") == "COMPLETED"}


def load_all_items() -> list[dict[str, Any]]:
    if not FULL_QA_ITEMS.exists():
        raise RuntimeError(f"Full QA item file does not exist: {FULL_QA_ITEMS}")
    return read_jsonl(FULL_QA_ITEMS)


def write_items(rows: list[dict[str, Any]]) -> None:
    FULL_QA_ITEMS.parent.mkdir(parents=True, exist_ok=True)
    FULL_QA_ITEMS.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def load_manifest_paper_maps() -> tuple[dict[str, str], dict[str, str]]:
    manifest = json.loads(PRODUCTION_CORPUS.read_text(encoding="utf-8"))
    public_to_uuid: dict[str, str] = {}
    uuid_to_public: dict[str, str] = {}
    for paper in manifest.get("papers", []):
        if not paper.get("included_in_production"):
            continue
        public_id = str(paper["paper_id"])
        database_id = str(paper["database_id"])
        public_to_uuid[public_id] = database_id
        uuid_to_public[database_id] = public_id
    return public_to_uuid, uuid_to_public


def find_paper_uuid_map(
    client: httpx.Client,
    api_base: str,
) -> tuple[dict[str, str], dict[str, str]]:
    public_to_uuid, uuid_to_public = load_manifest_paper_maps()
    response = client.get(f"{api_base}/papers")
    if response.status_code == 422:
        return public_to_uuid, uuid_to_public
    response.raise_for_status()
    papers = response.json()
    if isinstance(papers, dict):
        papers = papers.get("items") or papers.get("value") or []
    for paper in papers:
        if paper.get("arxiv_id"):
            public_to_uuid[str(paper["arxiv_id"])] = str(paper["id"])
            uuid_to_public[str(paper["id"])] = str(paper["arxiv_id"])
        if paper.get("title"):
            public_to_uuid.setdefault(str(paper["title"]), str(paper["id"]))
    return public_to_uuid, uuid_to_public


def terms(text: str) -> set[str]:
    return {
        token.lower().strip(".,:;!?()[]{}'\"")
        for token in text.split()
        if token.strip(".,:;!?()[]{}'\"")
    }


def overlap(expected: str, actual: str) -> float:
    expected_terms = terms(expected)
    return len(expected_terms & terms(actual)) / max(1, len(expected_terms))


def parse_error_metadata(response_text: str) -> dict[str, Any]:
    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError:
        return {"api_request_count": 0, "retry_reasons": [], "rate_limit_events": 0}
    error = payload.get("error") or {}
    message = error.get("message") or {}
    if not isinstance(message, dict):
        return {"api_request_count": 0, "retry_reasons": [], "rate_limit_events": 0}
    return {
        "api_request_count": int(message.get("api_request_count") or 0),
        "retry_count": len(message.get("retry_reasons") or []),
        "retry_reasons": message.get("retry_reasons") or [],
        "rate_limit_events": int(message.get("rate_limit_events") or 0),
        "provider_error_code": message.get("code"),
        "provider_error_stage": message.get("stage"),
    }


def public_paper_id(raw_paper_id: Any, uuid_to_public: dict[str, str]) -> str:
    raw = str(raw_paper_id)
    return uuid_to_public.get(raw, raw)


def evaluate(
    answer: dict[str, Any],
    gold: dict[str, Any],
    context: list[dict[str, Any]],
    uuid_to_public: dict[str, str],
) -> dict[str, Any]:
    context_citations = {
        (public_paper_id(item["paper_id"], uuid_to_public), page, block_id)
        for item in context
        for page in range(item["page_start"], item["page_end"] + 1)
        for block_id in (item.get("block_ids") or [item["chunk_id"]])
    }
    citations = [
        citation
        for claim in answer.get("claims", [])
        for citation in claim.get("citations", [])
    ]
    gold_blocks = set(gold.get("gold_block_ids") or [])
    gold_pages = set(gold.get("gold_pages") or [])
    gold_papers = set(gold.get("gold_paper_ids") or [])
    valid = [
        (
            public_paper_id(citation["paper_id"], uuid_to_public),
            citation["page"],
            citation["block_id"],
        )
        in context_citations
        for citation in citations
    ]
    precise = [
        public_paper_id(citation["paper_id"], uuid_to_public) in gold_papers
        and citation["block_id"] in gold_blocks
        and citation["page"] in gold_pages
        for citation in citations
    ]
    claim_scores = []
    for required in gold.get("required_claims") or []:
        scores = [overlap(required, claim.get("text", "")) for claim in answer.get("claims", [])]
        claim_scores.append({"required_claim": required, "best": max(scores, default=0)})
    cited_gold_blocks = {citation["block_id"] for citation in citations} & gold_blocks
    unsupported_claims = sum(
        not any(
            public_paper_id(citation["paper_id"], uuid_to_public) in gold_papers
            and citation["block_id"] in gold_blocks
            and citation["page"] in gold_pages
            for citation in claim.get("citations", [])
        )
        for claim in answer.get("claims", [])
    )
    return {
        "answerable_correct": answer.get("answerable") == gold.get("answerable"),
        "required_claim_coverage": (
            sum(item["best"] >= 0.35 for item in claim_scores) / len(claim_scores)
            if claim_scores
            else None
        ),
        "citation_id_validity": (
            sum(valid) / len(valid)
            if valid
            else (1.0 if not answer.get("answerable") else 0.0)
        ),
        "citation_precision": (
            sum(precise) / len(precise)
            if precise
            else (1.0 if not gold.get("answerable") else 0.0)
        ),
        "citation_recall": (
            len(cited_gold_blocks) / len(gold_blocks)
            if gold_blocks
            else (1.0 if not citations else 0.0)
        ),
        "claim_citation_binding_rate": (
            sum(bool(claim.get("citations")) for claim in answer.get("claims", []))
            / len(answer.get("claims", []))
            if answer.get("claims")
            else (1.0 if not answer.get("answerable") else 0.0)
        ),
        "unsupported_claim_count": unsupported_claims,
        "claim_scores": claim_scores,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [row for row in rows if row.get("status") == "COMPLETED"]
    failed = [row for row in rows if row.get("status") == "FAILED"]
    answerable = [row for row in completed if row["gold"]["answerable"]]
    unanswerable = [row for row in completed if not row["gold"]["answerable"]]
    metrics = [row["metrics"] for row in completed]
    answerable_metrics = [row["metrics"] for row in answerable]

    def avg(name: str, source: list[dict[str, Any]] = metrics) -> float | None:
        values = [row[name] for row in source if row.get(name) is not None]
        return round(sum(values) / len(values), 6) if values else None

    latencies = [
        row["answer"]["latency"]["total_latency_ms"]
        for row in completed
        if row.get("answer", {}).get("latency")
    ]
    usage = [row["answer"]["model_usage"] for row in completed if row.get("answer")]
    cost_values = [item.get("estimated_cost_usd") for item in usage]
    cost_configured = bool(cost_values) and all(value is not None for value in cost_values)
    request_count = 0
    retry_count = 0
    rate_limit_events = 0
    for row in rows:
        if row.get("status") == "FAILED" and not row.get("api_request_count"):
            metadata = parse_error_metadata(str(row.get("failure_reason") or ""))
            request_count += int(metadata.get("api_request_count") or 0)
            retry_count += int(metadata.get("retry_count") or 0)
            rate_limit_events += int(metadata.get("rate_limit_events") or 0)
        else:
            request_count += int(row.get("api_request_count") or 0)
            retry_count += int(row.get("retry_count") or 0)
            rate_limit_events += int(row.get("rate_limit_events") or 0)
    return {
        "attempted": len(rows),
        "completed": len(completed),
        "failed": len(failed),
        "answerable_items_completed": len(answerable),
        "unanswerable_items_completed": len(unanswerable),
        "answerable_accuracy": (
            round(sum(bool(row["answer"]["answerable"]) for row in answerable) / len(answerable), 6)
            if answerable
            else None
        ),
        "refusal_accuracy": (
            round(
                sum(not bool(row["answer"]["answerable"]) for row in unanswerable)
                / len(unanswerable),
                6,
            )
            if unanswerable
            else None
        ),
        "required_claim_coverage": avg("required_claim_coverage", answerable_metrics),
        "citation_id_validity": avg("citation_id_validity", answerable_metrics),
        "citation_precision": avg("citation_precision", answerable_metrics),
        "citation_recall": avg("citation_recall", answerable_metrics),
        "claim_citation_binding_rate": avg("claim_citation_binding_rate", answerable_metrics),
        "unsupported_claim_count": sum(row.get("unsupported_claim_count", 0) for row in metrics),
        "gold_block_retrieved_rate": (
            round(sum(bool(row["gold_block_present"]) for row in answerable) / len(answerable), 6)
            if answerable
            else None
        ),
        "latency_ms": {
            "mean": mean(latencies),
            "p50": percentile(latencies, 0.5),
            "p95": percentile(latencies, 0.95),
        },
        "input_tokens": sum(int(item.get("input_tokens") or 0) for item in usage),
        "output_tokens": sum(int(item.get("output_tokens") or 0) for item in usage),
        "total_tokens": sum(int(item.get("total_tokens") or 0) for item in usage),
        "estimated_cost_usd": (
            round(sum(float(value) for value in cost_values if value is not None), 8)
            if cost_configured
            else None
        ),
        "cost_is_configured": cost_configured,
        "api_requests": request_count,
        "retry_count": retry_count,
        "rate_limit_events": rate_limit_events,
    }


def write_summary(
    rows: list[dict[str, Any]],
    capabilities: dict[str, Any],
    started: float,
) -> dict[str, Any]:
    metrics = summarize(rows)
    payload = {
        "schema_version": "full-qa-production-v1",
        "status": "COMPLETED_WITH_FAILURES" if metrics["failed"] else "COMPLETED",
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset": "gold-dev-v1",
        "total": 50,
        "approved": 50,
        "answerable": 48,
        "unanswerable": 2,
        "completed_count": metrics["completed"],
        "failed_count": metrics["failed"],
        "production_full_qa_gate": "COMPLETED_WITH_FAILURES" if metrics["failed"] else "PASSED",
        "ready_for_production_deep_research": metrics["failed"] == 0,
        "strong_generalization_claim_allowed": False,
        "full_qa_executed": True,
        "deep_research_executed": False,
        "llm_provider": "siliconflow",
        "llm_model": "Qwen/Qwen3-8B",
        "prompt_version": "qa-production-v1",
        "rerank_enabled": False,
        "capabilities_budget_status": capabilities.get("stage13_30_budget"),
        "elapsed_wall_ms": round((time.perf_counter() - started) * 1000, 3),
        "metrics": metrics,
    }
    write_json(FULL_QA_JSON, payload)
    with FULL_QA_CSV.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "question_id",
                "status",
                "gold_answerable",
                "predicted_answerable",
                "gold_block_present",
                "claim_count",
                "citation_count",
                "citation_precision",
                "citation_recall",
                "total_tokens",
                "total_latency_ms",
                "failure_reason",
            ],
        )
        writer.writeheader()
        for row in rows:
            answer = row.get("answer") or {}
            writer.writerow(
                {
                    "question_id": row["question_id"],
                    "status": row["status"],
                    "gold_answerable": row["gold"]["answerable"],
                    "predicted_answerable": answer.get("answerable"),
                    "gold_block_present": row.get("gold_block_present"),
                    "claim_count": len(answer.get("claims") or []),
                    "citation_count": sum(
                        len(claim.get("citations") or [])
                        for claim in answer.get("claims") or []
                    ),
                    "citation_precision": row.get("metrics", {}).get("citation_precision"),
                    "citation_recall": row.get("metrics", {}).get("citation_recall"),
                    "total_tokens": (answer.get("model_usage") or {}).get("total_tokens"),
                    "total_latency_ms": (answer.get("latency") or {}).get("total_latency_ms"),
                    "failure_reason": row.get("failure_reason"),
                }
            )
    write_json(FULL_QA_TRACE, {"schema_version": "full-qa-production-trace-v1", "queries": rows})
    lines = [
        "# Full QA Production Summary v1",
        "",
        f"- Status: `{payload['status']}`",
        f"- Production Full QA gate: `{payload['production_full_qa_gate']}`",
        f"- Completed/failed: `{metrics['completed']}` / `{metrics['failed']}`",
        "- Model: `siliconflow` / `Qwen/Qwen3-8B`",
        "- Reranker: `disabled`",
        "- Deep Research executed: `false`",
        f"- Answerable accuracy: `{metrics['answerable_accuracy']}`",
        f"- Refusal accuracy: `{metrics['refusal_accuracy']}`",
        f"- Required claim coverage: `{metrics['required_claim_coverage']}`",
        (
            f"- Citation precision / recall: `{metrics['citation_precision']}` / "
            f"`{metrics['citation_recall']}`"
        ),
        f"- Citation ID validity: `{metrics['citation_id_validity']}`",
        f"- Gold block retrieved rate: `{metrics['gold_block_retrieved_rate']}`",
        (
            f"- Tokens input/output/total: `{metrics['input_tokens']}` / "
            f"`{metrics['output_tokens']}` / `{metrics['total_tokens']}`"
        ),
        (
            f"- Estimated cost USD: `{metrics['estimated_cost_usd']}` "
            f"configured=`{metrics['cost_is_configured']}`"
        ),
        f"- Latency mean/p50/p95 ms: `{metrics['latency_ms']}`",
        "",
        "This is a 50-item human-reviewed internal development evaluation, not a blind holdout.",
    ]
    FULL_QA_SUMMARY_DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    FULL_QA_AUDIT_DOC.write_text(
        "\n".join(
            [
                "# Full QA Production Audit v1",
                "",
                *lines[2:],
                "",
                "## Failed items",
                "",
                *(
                    [
                        f"- `{row['question_id']}`: {row.get('failure_reason')}"
                        for row in rows
                        if row.get("status") == "FAILED"
                    ]
                    or ["- None"]
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", default="http://localhost/api/v1")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--max-items", type=int, default=50)
    parser.add_argument("--max-requests", type=int, default=60)
    args = parser.parse_args()
    api_base = args.api_base.rstrip("/")
    if args.max_items != 50:
        raise RuntimeError("Production Full QA must run exactly 50 approved items")
    smoke = json.loads(SMOKE_JSON.read_text(encoding="utf-8"))
    if smoke.get("status") != "PASSED" or smoke.get("sample_id") not in {
        "q002",
        "q017",
        "q019",
        "q024",
    }:
        raise RuntimeError("a q002, q017, q019, or q024 live smoke must pass before Full QA")
    gold_by_id = {row["question_id"]: row for row in read_jsonl(GOLD)}
    retrieval_rows = [
        row
        for row in read_jsonl(RETRIEVAL_GOLD)
        if gold_by_id[row["question_id"]]["review_status"] == "approved"
    ]
    if len(retrieval_rows) != 50:
        raise RuntimeError(f"expected 50 approved records, got {len(retrieval_rows)}")
    started = time.perf_counter()
    if args.summarize_only:
        with httpx.Client(timeout=30) as client:
            capabilities = client.get(f"{api_base}/capabilities")
            capabilities.raise_for_status()
            capabilities_json = capabilities.json()
        rows = sorted(load_all_items(), key=lambda row: row["question_id"])
        payload = write_summary(rows, capabilities_json, started)
        print(
            json.dumps(
                {"status": payload["status"], "metrics": payload["metrics"]},
                ensure_ascii=False,
            )
        )
        return 0 if payload["completed_count"] == 50 else 2
    rows: list[dict[str, Any]] = list(load_existing_items(args.resume).values())
    existing = {row["question_id"] for row in rows}
    with httpx.Client(timeout=240) as client:
        capabilities = client.get(f"{api_base}/capabilities")
        capabilities.raise_for_status()
        capabilities_json = capabilities.json()
        budget = capabilities_json.get("stage13_30_budget") or {}
        if budget.get("status") != "FULL_QA_BUDGET_READY":
            raise RuntimeError(f"Full QA budget is not ready: {budget}")
        reranker_status = (
            (capabilities_json.get("capabilities") or {})
            .get("reranker", {})
            .get("status")
        )
        if reranker_status != "disabled":
            raise RuntimeError("Reranker must remain disabled")
        paper_map, uuid_to_public = find_paper_uuid_map(client, api_base)
        for record in retrieval_rows:
            if record["question_id"] in existing:
                continue
            if sum(int(row.get("api_request_count") or 0) for row in rows) >= args.max_requests:
                break
            gold = gold_by_id[record["question_id"]]
            filter_papers = (
                (record.get("retrieval_filter") or {}).get("paper_ids")
                or record.get("gold_paper_ids")
                or []
            )
            missing_papers = [paper_id for paper_id in filter_papers if paper_id not in paper_map]
            if missing_papers:
                raise RuntimeError(f"Missing production paper UUID mapping for: {missing_papers}")
            paper_uuids = (
                [paper_map[paper_id] for paper_id in filter_papers]
                if filter_papers
                else None
            )
            qa_payload = {
                "question": record["retrieval_query"],
                "paper_ids": paper_uuids,
                "top_k": 10,
                "sample_id": record["question_id"],
                "run_id": f"full-qa-rerun-{record['question_id']}-{int(time.time())}",
            }
            item_started = time.perf_counter()
            try:
                response = client.post(f"{api_base}/qa", json=qa_payload)
                wall_ms = round((time.perf_counter() - item_started) * 1000, 3)
                if response.status_code >= 400:
                    error_metadata = parse_error_metadata(response.text)
                    row = {
                        "question_id": record["question_id"],
                        "status": "FAILED",
                        "retrieval_query": record["retrieval_query"],
                        "retrieval_scope": record["retrieval_scope"],
                        "retrieval_filter": record["retrieval_filter"],
                        "gold": {
                            "answerable": gold["answerable"],
                            "gold_paper_ids": gold["gold_paper_ids"],
                            "gold_block_ids": gold["gold_block_ids"],
                            "gold_pages": gold["gold_pages"],
                            "required_claims": gold["required_claims"],
                        },
                        "failure_reason": response.text[:1000],
                        **error_metadata,
                        "wall_ms": wall_ms,
                    }
                else:
                    answer = response.json()
                    context = answer.get("citations") or []
                    context_block_ids = {
                        block
                        for citation in context
                        for block in citation.get("block_ids", [])
                    }
                    gold_block_present = bool(
                        context_block_ids & set(gold.get("gold_block_ids") or [])
                    )
                    metrics = evaluate(
                        answer,
                        gold,
                        [
                            {
                                "paper_id": citation["paper_id"],
                                "page_start": citation["page_start"],
                                "page_end": citation["page_end"],
                                "block_ids": citation.get("block_ids") or [],
                                "chunk_id": (
                                    citation.get("block_ids", [""])[0]
                                    if citation.get("block_ids")
                                    else ""
                                ),
                            }
                            for citation in context
                        ],
                        uuid_to_public,
                    )
                    row = {
                        "question_id": record["question_id"],
                        "status": "COMPLETED",
                        "retrieval_query": record["retrieval_query"],
                        "retrieval_scope": record["retrieval_scope"],
                        "retrieval_filter": record["retrieval_filter"],
                        "gold": {
                            "answerable": gold["answerable"],
                            "gold_paper_ids": gold["gold_paper_ids"],
                            "gold_block_ids": gold["gold_block_ids"],
                            "gold_pages": gold["gold_pages"],
                            "required_claims": gold["required_claims"],
                        },
                        "answer": answer,
                        "metrics": metrics,
                        "gold_block_present": gold_block_present,
                        "api_request_count": int(answer.get("api_request_count") or 0),
                        "retry_count": int(answer.get("retry_count") or 0),
                        "retry_reasons": answer.get("retry_reasons") or [],
                        "rate_limit_events": int(answer.get("rate_limit_events") or 0),
                        "wall_ms": wall_ms,
                    }
            except httpx.HTTPError as exc:
                row = {
                    "question_id": record["question_id"],
                    "status": "FAILED",
                    "retrieval_query": record["retrieval_query"],
                    "retrieval_scope": record["retrieval_scope"],
                    "retrieval_filter": record["retrieval_filter"],
                    "gold": {
                        "answerable": gold["answerable"],
                        "gold_paper_ids": gold["gold_paper_ids"],
                        "gold_block_ids": gold["gold_block_ids"],
                        "gold_pages": gold["gold_pages"],
                        "required_claims": gold["required_claims"],
                    },
                    "failure_reason": type(exc).__name__,
                    "api_request_count": 0,
                }
            rows.append(row)
            rows = sorted(rows, key=lambda row: row["question_id"])
            write_items(rows)
            write_summary(rows, capabilities_json, started)
    rows = sorted(rows, key=lambda row: row["question_id"])
    payload = write_summary(rows, capabilities_json, started)
    print(
        json.dumps(
            {"status": payload["status"], "metrics": payload["metrics"]},
            ensure_ascii=False,
        )
    )
    return 0 if payload["completed_count"] == 50 else 2


if __name__ == "__main__":
    raise SystemExit(main())
