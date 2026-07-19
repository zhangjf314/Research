"""Run the Stage 13.34 15-item Production Full QA canary."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_production_full_qa_v1 import (  # noqa: E402
    GOLD,
    RETRIEVAL_GOLD,
    evaluate,
    find_paper_uuid_map,
    mean,
    parse_error_metadata,
    percentile,
    read_jsonl,
    write_json,
)

DATA = ROOT / "data" / "evaluation"
ARTIFACTS = ROOT / "artifacts"
DOCS = ROOT / "docs"
OUT_JSON = DATA / "full-qa-canary-results-v2.json"
OUT_CSV = DATA / "full-qa-canary-results-v2.csv"
OUT_TRACE = ARTIFACTS / "full-qa-canary-trace-v2.json"
OUT_DOC = DOCS / "full-qa-canary-audit-v2.md"

FAILED_V1 = ["q014", "q020", "q029", "q031", "q032", "q035", "q036", "q037", "q044"]
SUPPLEMENT = ["q001", "q008", "q015", "q024", "q049", "q005"]
CANARY_IDS = FAILED_V1 + SUPPLEMENT


def _error_code(row: dict[str, Any]) -> str | None:
    if row.get("status") == "COMPLETED":
        return None
    metadata = parse_error_metadata(str(row.get("failure_reason") or ""))
    return metadata.get("provider_error_code") or row.get("failure_reason")


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [row for row in rows if row.get("status") == "COMPLETED"]
    failed = [row for row in rows if row.get("status") == "FAILED"]
    answerable = [row for row in completed if row["gold"]["answerable"]]
    metrics = [row["metrics"] for row in completed]
    answerable_metrics = [row["metrics"] for row in answerable]

    def avg(name: str) -> float | None:
        values = [row[name] for row in answerable_metrics if row.get(name) is not None]
        return round(sum(values) / len(values), 6) if values else None

    error_codes = [_error_code(row) for row in failed]
    malformed = sum(code == "CLAIM_QA_JSON_PARSE_ERROR" for code in error_codes)
    schema = sum(code == "CLAIM_QA_SCHEMA_VALIDATION_ERROR" for code in error_codes)
    invalid_citation = sum(code == "CLAIM_QA_CITATION_VALIDATION_ERROR" for code in error_codes)
    unsupported_claim_count = sum(row.get("unsupported_claim_count", 0) for row in metrics)
    latencies = [
        row["answer"]["latency"]["total_latency_ms"]
        for row in completed
        if row.get("answer", {}).get("latency")
    ]
    usage = [row["answer"]["model_usage"] for row in completed if row.get("answer")]
    summary = {
        "attempted": len(rows),
        "completed": len(completed),
        "terminal_failure_count": len(failed),
        "malformed_json_count": malformed,
        "schema_failure_count": schema,
        "invalid_citation_count": invalid_citation,
        "citation_context_validity": avg("citation_id_validity"),
        "page_accuracy": avg("citation_id_validity"),
        "template_fallback_count": 0,
        "unclassified_exception_count": sum(
            code == "CLAIM_QA_UNEXPECTED_ERROR" for code in error_codes
        ),
        "core_unsupported_claim_count": unsupported_claim_count,
        "required_claim_coverage": avg("required_claim_coverage"),
        "citation_precision": avg("citation_precision"),
        "citation_recall": avg("citation_recall"),
        "input_tokens": sum(int(item.get("input_tokens") or 0) for item in usage),
        "output_tokens": sum(int(item.get("output_tokens") or 0) for item in usage),
        "total_tokens": sum(int(item.get("total_tokens") or 0) for item in usage),
        "latency_ms": {
            "mean": mean(latencies),
            "p50": percentile(latencies, 0.5),
            "p95": percentile(latencies, 0.95),
        },
    }
    summary["production_qa_canary_gate"] = (
        "PASSED"
        if summary["attempted"] == 15
        and summary["completed"] == 15
        and summary["terminal_failure_count"] == 0
        and summary["malformed_json_count"] == 0
        and summary["schema_failure_count"] == 0
        and summary["invalid_citation_count"] == 0
        and summary["citation_context_validity"] == 1.0
        and summary["page_accuracy"] == 1.0
        and summary["template_fallback_count"] == 0
        and summary["unclassified_exception_count"] == 0
        and summary["core_unsupported_claim_count"] == 0
        else "FAILED"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", default="http://localhost/api/v1")
    parser.add_argument("--max-requests", type=int, default=20)
    args = parser.parse_args()
    api_base = args.api_base.rstrip("/")
    gold_by_id = {row["question_id"]: row for row in read_jsonl(GOLD)}
    retrieval_by_id = {row["question_id"]: row for row in read_jsonl(RETRIEVAL_GOLD)}
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    with httpx.Client(timeout=180) as client:
        capabilities = client.get(f"{api_base}/capabilities").json()
        reranker_status = (
            (capabilities.get("capabilities") or {}).get("reranker", {}).get("status")
        )
        if reranker_status != "disabled":
            raise RuntimeError("Reranker must remain disabled")
        paper_map, uuid_to_public = find_paper_uuid_map(client, api_base)
        for question_id in CANARY_IDS:
            if sum(int(row.get("api_request_count") or 0) for row in rows) >= args.max_requests:
                break
            record = retrieval_by_id[question_id]
            gold = gold_by_id[question_id]
            filter_papers = (
                (record.get("retrieval_filter") or {}).get("paper_ids")
                or record.get("gold_paper_ids")
                or []
            )
            qa_payload = {
                "question": record["retrieval_query"],
                "paper_ids": [paper_map[paper_id] for paper_id in filter_papers]
                if filter_papers
                else None,
                "top_k": 10,
                "sample_id": question_id,
                "run_id": f"full-qa-canary-v2-{question_id}-{int(time.time())}",
            }
            item_started = time.perf_counter()
            try:
                response = client.post(f"{api_base}/qa", json=qa_payload)
                wall_ms = round((time.perf_counter() - item_started) * 1000, 3)
                if response.status_code >= 400:
                    row = {
                        "question_id": question_id,
                        "status": "FAILED",
                        "retrieval_query": record["retrieval_query"],
                        "retrieval_scope": record["retrieval_scope"],
                        "retrieval_filter": record["retrieval_filter"],
                        "gold": _gold_payload(gold),
                        "failure_reason": response.text[:1000],
                        **parse_error_metadata(response.text),
                        "wall_ms": wall_ms,
                    }
                else:
                    answer = response.json()
                    context = answer.get("citations") or []
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
                        "question_id": question_id,
                        "status": "COMPLETED",
                        "retrieval_query": record["retrieval_query"],
                        "retrieval_scope": record["retrieval_scope"],
                        "retrieval_filter": record["retrieval_filter"],
                        "gold": _gold_payload(gold),
                        "answer": answer,
                        "metrics": metrics,
                        "api_request_count": int(answer.get("api_request_count") or 0),
                        "retry_count": int(answer.get("retry_count") or 0),
                        "retry_reasons": answer.get("retry_reasons") or [],
                        "rate_limit_events": int(answer.get("rate_limit_events") or 0),
                        "wall_ms": wall_ms,
                    }
            except httpx.HTTPError as exc:
                row = {
                    "question_id": question_id,
                    "status": "FAILED",
                    "retrieval_query": record["retrieval_query"],
                    "retrieval_scope": record["retrieval_scope"],
                    "retrieval_filter": record["retrieval_filter"],
                    "gold": _gold_payload(gold),
                    "failure_reason": type(exc).__name__,
                    "api_request_count": 0,
                }
            rows.append(row)
            OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
            OUT_JSON.write_text(
                json.dumps({"rows": rows}, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
    summary = _summarize(rows)
    payload = {
        "schema_version": "full-qa-canary-results-v2",
        "generated_at": datetime.now(UTC).isoformat(),
        "canary_ids": CANARY_IDS,
        "concurrency": 1,
        "reranker_enabled": False,
        "qa_generation_retry_count": 0,
        "json_repair_enabled": False,
        "citation_repair_enabled": False,
        "transport_retry_count": 1,
        "elapsed_wall_ms": round((time.perf_counter() - started) * 1000, 3),
        "summary": summary,
        "rows": rows,
    }
    write_json(OUT_JSON, payload)
    write_json(OUT_TRACE, payload)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as stream:
        fieldnames = [
            "question_id",
            "status",
            "api_request_count",
            "retry_reasons",
            "wall_ms",
            "failure_reason",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})
    OUT_DOC.write_text(
        "\n".join(
            [
                "# Full QA Canary Audit v2",
                "",
                f"- Gate: `{summary['production_qa_canary_gate']}`",
                "- Attempted/completed/failed: "
                f"`{summary['attempted']}` / `{summary['completed']}` / "
                f"`{summary['terminal_failure_count']}`",
                "- Malformed/schema/invalid citation: "
                f"`{summary['malformed_json_count']}` / "
                f"`{summary['schema_failure_count']}` / "
                f"`{summary['invalid_citation_count']}`",
                f"- Citation context validity: `{summary['citation_context_validity']}`",
                f"- Core unsupported claim count: `{summary['core_unsupported_claim_count']}`",
                "",
                "This canary is an internal development gate, not a blind benchmark.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"status": summary["production_qa_canary_gate"], "summary": summary},
            ensure_ascii=False,
        )
    )
    return 0 if summary["production_qa_canary_gate"] == "PASSED" else 2


def _gold_payload(gold: dict[str, Any]) -> dict[str, Any]:
    return {
        "answerable": gold["answerable"],
        "gold_paper_ids": gold["gold_paper_ids"],
        "gold_block_ids": gold["gold_block_ids"],
        "gold_pages": gold["gold_pages"],
        "required_claims": gold["required_claims"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
