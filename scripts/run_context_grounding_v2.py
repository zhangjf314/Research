"""Offline retrieval/context grounding audit for Stage 13.34.

This script calls only the local retrieval API. It does not call the LLM.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from scripts.run_production_full_qa_v1 import (
    GOLD,
    RETRIEVAL_GOLD,
    find_paper_uuid_map,
    mean,
    percentile,
    read_jsonl,
    write_json,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "evaluation"
DOCS = ROOT / "docs"
OUT_JSON = DATA / "context-grounding-v2.json"
OUT_CSV = DATA / "context-grounding-v2.csv"
OUT_DOC = DOCS / "context-grounding-audit-v2.md"


def _category(row: dict[str, Any]) -> str:
    value = str(row.get("category") or "unknown")
    return value


def _context_tokens(context: list[dict[str, Any]]) -> int:
    return sum(max(1, (len(str(item.get("evidence") or "")) + 3) // 4) for item in context)


def _row_metrics(
    row: dict[str, Any],
    gold: dict[str, Any],
    context: list[dict[str, Any]],
) -> dict[str, Any]:
    block_ids = {
        block_id
        for item in context
        for block_id in (item.get("block_ids") or [item.get("chunk_id")])
    }
    pages = {
        page
        for item in context
        for page in range(int(item.get("page_start") or 0), int(item.get("page_end") or 0) + 1)
    }
    gold_blocks = set(gold.get("gold_block_ids") or [])
    gold_pages = set(gold.get("gold_pages") or [])
    required_claims = gold.get("required_claims") or []
    context_text = "\n".join(str(item.get("evidence") or "") for item in context).lower()
    claim_hits = 0
    for claim in required_claims:
        terms = [term.strip(".,:;!?()[]{}'\"").lower() for term in str(claim).split()]
        terms = [term for term in terms if len(term) > 3]
        if terms and sum(term in context_text for term in terms) / len(terms) >= 0.35:
            claim_hits += 1
    return {
        "question_id": row["question_id"],
        "category": _category(gold),
        "difficulty": gold.get("difficulty"),
        "answerable": bool(gold.get("answerable")),
        "context_block_count": sum(
            len(item.get("block_ids") or [item.get("chunk_id")]) for item in context
        ),
        "context_chunk_count": len(context),
        "context_tokens": _context_tokens(context),
        "gold_block_context_hit": bool(gold_blocks & block_ids) if gold_blocks else None,
        "gold_page_context_hit": bool(gold_pages & pages) if gold_pages else None,
        "required_claim_evidence_coverage": (
            claim_hits / len(required_claims) if required_claims else None
        ),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    answerable = [row for row in rows if row["answerable"]]
    claim_values = [
        row["required_claim_evidence_coverage"]
        for row in answerable
        if row["required_claim_evidence_coverage"] is not None
    ]
    token_values = [row["context_tokens"] for row in rows]
    latency_values = [row["retrieval_latency_ms"] for row in rows]
    by_category: dict[str, dict[str, Any]] = {}
    for category, group in _groups(rows, "category").items():
        by_category[category] = _summary_for(group)
    summary = {
        "total": len(rows),
        "answerable": len(answerable),
        "gold_block_context_coverage": _rate(
            row["gold_block_context_hit"]
            for row in answerable
            if row["gold_block_context_hit"] is not None
        ),
        "gold_page_context_coverage": _rate(
            row["gold_page_context_hit"]
            for row in answerable
            if row["gold_page_context_hit"] is not None
        ),
        "gold_or_equivalent_context_coverage": _rate(
            row["gold_block_context_hit"]
            for row in answerable
            if row["gold_block_context_hit"] is not None
        ),
        "required_claim_evidence_coverage": (
            round(sum(claim_values) / len(claim_values), 6) if claim_values else None
        ),
        "context_block_count_mean": mean([row["context_block_count"] for row in rows]),
        "context_tokens_mean": mean(token_values),
        "context_tokens_p95": percentile(token_values, 0.95),
        "retrieval_latency_ms_mean": mean(latency_values),
        "retrieval_latency_ms_p95": percentile(latency_values, 0.95),
        "by_category": by_category,
    }
    summary["canary_status"] = (
        "READY"
        if (summary["gold_or_equivalent_context_coverage"] or 0) >= 0.75
        and (summary["required_claim_evidence_coverage"] or 0) >= 0.65
        else "BLOCKED_BY_CONTEXT_COVERAGE"
    )
    return summary


def _summary_for(rows: list[dict[str, Any]]) -> dict[str, Any]:
    claim_values = [
        row["required_claim_evidence_coverage"]
        for row in rows
        if row["required_claim_evidence_coverage"] is not None
    ]
    return {
        "count": len(rows),
        "gold_block_context_coverage": _rate(
            row["gold_block_context_hit"]
            for row in rows
            if row["gold_block_context_hit"] is not None
        ),
        "required_claim_evidence_coverage": (
            round(sum(claim_values) / len(claim_values), 6) if claim_values else None
        ),
    }


def _groups(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key) or "unknown")].append(row)
    return dict(grouped)


def _rate(values: Any) -> float | None:
    values = list(values)
    return round(sum(bool(value) for value in values) / len(values), 6) if values else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", default="http://localhost/api/v1")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    api_base = args.api_base.rstrip("/")
    gold_by_id = {row["question_id"]: row for row in read_jsonl(GOLD)}
    retrieval_rows = [
        row
        for row in read_jsonl(RETRIEVAL_GOLD)
        if gold_by_id[row["question_id"]]["review_status"] == "approved"
    ]
    rows: list[dict[str, Any]] = []
    with httpx.Client(timeout=120) as client:
        paper_map, _uuid_to_public = find_paper_uuid_map(client, api_base)
        for record in retrieval_rows:
            gold = gold_by_id[record["question_id"]]
            filter_papers = (
                (record.get("retrieval_filter") or {}).get("paper_ids")
                or record.get("gold_paper_ids")
                or []
            )
            payload = {
                "query": record["retrieval_query"],
                "filters": {
                    "paper_ids": [paper_map[paper_id] for paper_id in filter_papers]
                    if filter_papers
                    else None
                },
                "recall_k": 30,
                "top_k": args.top_k,
            }
            started = time.perf_counter()
            response = client.post(f"{api_base}/retrieve", json=payload)
            elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
            response.raise_for_status()
            body = response.json()
            row = _row_metrics(record, gold, body.get("context") or [])
            row["retrieval_latency_ms"] = elapsed_ms
            rows.append(row)
    summary = summarize(rows)
    write_json(
        OUT_JSON,
        {
            "schema_version": "context-grounding-v2",
            "generated_at": datetime.now(UTC).isoformat(),
            "llm_called": False,
            "reranker_enabled": False,
            "top_k": args.top_k,
            "summary": summary,
            "rows": rows,
        },
    )
    with OUT_CSV.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Context Grounding Audit v2",
        "",
        "- LLM called: `false`",
        "- Reranker: `disabled`",
        f"- Total: `{summary['total']}`",
        f"- Gold block context coverage: `{summary['gold_block_context_coverage']}`",
        f"- Gold page context coverage: `{summary['gold_page_context_coverage']}`",
        f"- Required claim evidence coverage: `{summary['required_claim_evidence_coverage']}`",
        "- Context tokens mean/P95: "
        f"`{summary['context_tokens_mean']}` / `{summary['context_tokens_p95']}`",
        "- Retrieval latency mean/P95 ms: "
        f"`{summary['retrieval_latency_ms_mean']}` / "
        f"`{summary['retrieval_latency_ms_p95']}`",
        f"- CANARY_STATUS: `{summary['canary_status']}`",
        "",
        "This is an internal development diagnostic, not a blind benchmark.",
    ]
    OUT_DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["canary_status"], "summary": summary}, ensure_ascii=False))
    return 0 if summary["canary_status"] == "READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
