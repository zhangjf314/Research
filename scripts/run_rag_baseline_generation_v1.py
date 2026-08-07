from __future__ import annotations

# ruff: noqa: E501
import argparse
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

from paper_research.evaluation.rag_benchmark import read_jsonl
from paper_research.evaluation.rag_official_baseline import (
    BASELINE_CONFIG_HASH,
    DATASET_VERSION,
    DEV_DATASET_HASH,
    FULL_DATASET_HASH,
    SYSTEM_UNDER_TEST_COMMIT,
    TEST_DATASET_HASH,
    aggregate_generation_rows,
    append_jsonl,
    evaluate_generation_answer,
    generation_bad_cases,
    grouped_generation,
    write_json_artifact,
)

ROOT = Path("data/evaluation/rag-benchmark")
DOCS = Path("docs/rag-benchmark")
ITEMS = ROOT / "generation-baseline-items-v1.jsonl"
MAX_LOGICAL_QUESTIONS = 146
MAX_TOTAL_TOKENS = 2_000_000
MAX_TOTAL_COST_USD = 0.30


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def load_existing(resume: bool) -> dict[str, dict]:
    if not resume or not ITEMS.exists():
        return {}
    return {row["question_id"]: row for row in read_jsonl(ITEMS) if row.get("status") == "COMPLETED"}


def public_paper_id(raw: str) -> str:
    return str(raw)


def parse_error_metadata(text: str) -> dict:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {"api_request_count": 0}
    detail = payload.get("detail") or payload.get("error") or {}
    if isinstance(detail, str):
        return {"api_request_count": 0, "provider_error": detail[:300]}
    return {
        "api_request_count": int(detail.get("api_request_count") or 0),
        "retry_count": len(detail.get("retry_reasons") or []),
        "retry_reasons": detail.get("retry_reasons") or [],
        "provider_error_code": detail.get("code"),
        "provider_error_stage": detail.get("stage"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", default="http://localhost/api/v1")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-requests", type=int, default=146)
    parser.add_argument("--max-total-tokens", type=int, default=MAX_TOTAL_TOKENS)
    parser.add_argument("--max-total-cost-usd", type=float, default=MAX_TOTAL_COST_USD)
    args = parser.parse_args()
    records = read_jsonl(ROOT / "gold-full-v1.jsonl")
    if len(records) != MAX_LOGICAL_QUESTIONS:
        raise RuntimeError(f"expected {MAX_LOGICAL_QUESTIONS} questions, got {len(records)}")
    retrieval_rows = {row["question_id"]: row for row in read_jsonl(ROOT / "retrieval-baseline-items-v1.jsonl")}
    existing = load_existing(args.resume)
    rows = list(existing.values())
    duplicate_execution_count = 0
    with httpx.Client(timeout=240) as client:
        for gold in records:
            if gold["question_id"] in existing:
                continue
            current_summary = aggregate_generation_rows(rows) if rows else {"total_tokens": 0, "cost": 0, "provider_request_count": 0}
            if current_summary.get("provider_request_count", 0) >= args.max_requests:
                break
            if current_summary.get("total_tokens", 0) >= args.max_total_tokens:
                break
            if (current_summary.get("cost") or 0) >= args.max_total_cost_usd:
                break
            started = time.perf_counter()
            try:
                response = client.post(
                    f"{args.api_base.rstrip('/')}/qa",
                    json={
                        "question": gold["question"],
                        "paper_ids": None,
                        "top_k": 5,
                        "sample_id": gold["question_id"],
                        "run_id": f"rag-baseline-v1-{gold['question_id']}",
                    },
                )
                wall_ms = round((time.perf_counter() - started) * 1000, 3)
                if response.status_code >= 400:
                    error_metadata = parse_error_metadata(response.text)
                    row = {
                        "question_id": gold["question_id"],
                        "split": gold.get("split"),
                        "category": gold.get("category"),
                        "difficulty": gold.get("difficulty"),
                        "status": "FAILED",
                        "gold": {
                            "answerable": gold.get("answerable"),
                            "gold_paper_ids": gold.get("gold_paper_ids", []),
                            "gold_block_ids": gold.get("gold_block_ids", []),
                            "gold_pages": gold.get("gold_pages", []),
                            "required_claims": gold.get("required_claims", []),
                        },
                        "failure_reason": response.text[:1000],
                        "generation_metrics": {},
                        "wall_ms": wall_ms,
                        **error_metadata,
                    }
                else:
                    answer = response.json()
                    metrics = evaluate_generation_answer(answer, gold, retrieval_rows.get(gold["question_id"]))
                    row = {
                        "question_id": gold["question_id"],
                        "split": gold.get("split"),
                        "category": gold.get("category"),
                        "difficulty": gold.get("difficulty"),
                        "status": "COMPLETED",
                        "gold": {
                            "answerable": gold.get("answerable"),
                            "gold_paper_ids": gold.get("gold_paper_ids", []),
                            "gold_block_ids": gold.get("gold_block_ids", []),
                            "gold_pages": gold.get("gold_pages", []),
                            "required_claims": gold.get("required_claims", []),
                        },
                        "answer": answer,
                        "generation_metrics": metrics,
                        "api_request_count": int(answer.get("api_request_count") or 0),
                        "retry_count": int(answer.get("retry_count") or 0),
                        "retry_reasons": answer.get("retry_reasons") or [],
                        "wall_ms": wall_ms,
                    }
            except httpx.HTTPError as exc:
                row = {
                    "question_id": gold["question_id"],
                    "split": gold.get("split"),
                    "category": gold.get("category"),
                    "difficulty": gold.get("difficulty"),
                    "status": "FAILED",
                    "gold": {
                        "answerable": gold.get("answerable"),
                        "gold_paper_ids": gold.get("gold_paper_ids", []),
                        "gold_block_ids": gold.get("gold_block_ids", []),
                        "gold_pages": gold.get("gold_pages", []),
                        "required_claims": gold.get("required_claims", []),
                    },
                    "failure_reason": f"{type(exc).__name__}: {exc}",
                    "generation_metrics": {},
                    "api_request_count": 0,
                    "wall_ms": round((time.perf_counter() - started) * 1000, 3),
                }
            rows.append(row)
            rows.sort(key=lambda item: item["question_id"])
            append_jsonl(ITEMS, row)
    rows = sorted(read_jsonl(ITEMS), key=lambda item: item["question_id"])

    split_rows = {
        "full": rows,
        "dev": [row for row in rows if row.get("split") == "dev"],
        "test": [row for row in rows if row.get("split") == "test"],
    }
    bad_cases = generation_bad_cases(rows)
    payload = {
        "schema_version": "generation-baseline-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "system_under_test_commit": SYSTEM_UNDER_TEST_COMMIT,
        "benchmark_harness_commit": git_head(),
        "dataset_version": DATASET_VERSION,
        "baseline_config_hash": BASELINE_CONFIG_HASH,
        "full_dataset_hash": FULL_DATASET_HASH,
        "dev_dataset_hash": DEV_DATASET_HASH,
        "test_dataset_hash": TEST_DATASET_HASH,
        "generation_questions": len(records),
        "generation_completed": sum(1 for row in rows if row.get("status") == "COMPLETED"),
        "generation_failed": sum(1 for row in rows if row.get("status") == "FAILED"),
        "provider_request_count": aggregate_generation_rows(rows)["provider_request_count"],
        "provider_failure_count": aggregate_generation_rows(rows)["provider_failure_count"],
        "duplicate_execution_count": duplicate_execution_count,
        "resume_count": 1 if args.resume else 0,
        "metrics": {split: aggregate_generation_rows(values) for split, values in split_rows.items()},
        "by_category": grouped_generation(rows, "category"),
        "by_difficulty": grouped_generation(rows, "difficulty"),
        "bad_cases": bad_cases,
        "semantic_claim_support_audit": "NOT_FORMALLY_VALIDATED",
        "budget": {
            "max_logical_questions": MAX_LOGICAL_QUESTIONS,
            "max_total_tokens": args.max_total_tokens,
            "max_total_cost_usd": args.max_total_cost_usd,
        },
    }
    write_json_artifact(ROOT / "generation-baseline-v1.json", payload)
    lines = [
        "# Generation baseline v1",
        "",
        f"- generation_questions: {payload['generation_questions']}",
        f"- generation_completed: {payload['generation_completed']}",
        f"- generation_failed: {payload['generation_failed']}",
        f"- required_claim_coverage: `{payload['metrics']['full']['required_claim_coverage']}`",
        f"- citation_precision: `{payload['metrics']['full']['citation_precision']}`",
        f"- citation_recall: `{payload['metrics']['full']['citation_recall']}`",
        f"- abstention_accuracy: `{payload['metrics']['full']['abstention_accuracy']}`",
        f"- total_tokens: `{payload['metrics']['full']['total_tokens']}`",
        f"- cost: `{payload['metrics']['full']['cost']}`",
        "",
        "Semantic claim support is not formally validated.",
    ]
    (DOCS / "generation-baseline-v1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "COMPLETED", "metrics": payload["metrics"]["full"]}, ensure_ascii=False))
    return 0 if payload["generation_completed"] + payload["generation_failed"] == len(records) else 2


if __name__ == "__main__":
    raise SystemExit(main())
