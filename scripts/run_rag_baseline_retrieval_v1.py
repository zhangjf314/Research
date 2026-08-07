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
    aggregate_retrieval_rows,
    append_jsonl,
    grouped_retrieval,
    retrieval_bad_cases,
    retrieval_row,
    write_json_artifact,
)

ROOT = Path("data/evaluation/rag-benchmark")
DOCS = Path("docs/rag-benchmark")
ITEMS = ROOT / "retrieval-baseline-items-v1.jsonl"


def paper_id_map() -> dict[str, str]:
    manifest = json.loads(Path("data/evaluation/production-corpus-v1.json").read_text(encoding="utf-8"))
    mapping = {}
    for paper in manifest.get("papers", []):
        if paper.get("included_in_production"):
            mapping[str(paper.get("database_id"))] = str(paper.get("paper_id"))
    return mapping


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def load_existing(resume: bool) -> dict[str, dict]:
    if not resume or not ITEMS.exists():
        return {}
    return {row["question_id"]: row for row in read_jsonl(ITEMS)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", default="http://localhost/api/v1")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    records = read_jsonl(ROOT / "gold-full-v1.jsonl")
    id_map = paper_id_map()
    existing = load_existing(args.resume)
    rows = list(existing.values())
    with httpx.Client(timeout=180) as client:
        for gold in records:
            if gold["question_id"] in existing:
                continue
            started = time.perf_counter()
            try:
                response = client.post(
                    f"{args.api_base.rstrip('/')}/retrieve",
                    json={
                        "query": gold["question"],
                        "filters": {},
                        "recall_k": 20,
                        "top_k": 20,
                    },
                )
                latency = round((time.perf_counter() - started) * 1000, 3)
                response.raise_for_status()
                payload = response.json()
                row = retrieval_row(
                    gold,
                    payload.get("context", []),
                    latency_ms=latency,
                    paper_id_map=id_map,
                )
            except Exception as exc:  # noqa: BLE001 - benchmark records failed question and continues
                latency = round((time.perf_counter() - started) * 1000, 3)
                row = retrieval_row(gold, [], latency_ms=latency, failure=f"{type(exc).__name__}: {exc}")
            rows.append(row)
            rows.sort(key=lambda item: item["question_id"])
            append_jsonl(ITEMS, row)
    rows = sorted(read_jsonl(ITEMS), key=lambda item: item["question_id"])

    split_rows = {
        "full": rows,
        "dev": [row for row in rows if row.get("split") == "dev"],
        "test": [row for row in rows if row.get("split") == "test"],
    }
    bad_cases = retrieval_bad_cases(rows)
    payload = {
        "schema_version": "retrieval-baseline-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "system_under_test_commit": SYSTEM_UNDER_TEST_COMMIT,
        "benchmark_harness_commit": git_head(),
        "dataset_version": DATASET_VERSION,
        "baseline_config_hash": BASELINE_CONFIG_HASH,
        "full_dataset_hash": FULL_DATASET_HASH,
        "dev_dataset_hash": DEV_DATASET_HASH,
        "test_dataset_hash": TEST_DATASET_HASH,
        "retrieval_questions": len(records),
        "retrieval_completed": sum(1 for row in rows if not row.get("retrieval_failure")),
        "retrieval_failed": sum(1 for row in rows if row.get("retrieval_failure")),
        "metrics": {split: aggregate_retrieval_rows(values) for split, values in split_rows.items()},
        "by_category": grouped_retrieval(rows, "category"),
        "by_difficulty": grouped_retrieval(rows, "difficulty"),
        "bad_cases": bad_cases,
    }
    write_json_artifact(ROOT / "retrieval-baseline-v1.json", payload)
    lines = [
        "# Retrieval baseline v1",
        "",
        f"- retrieval_questions: {payload['retrieval_questions']}",
        f"- retrieval_completed: {payload['retrieval_completed']}",
        f"- retrieval_failed: {payload['retrieval_failed']}",
        f"- full Recall@10: `{payload['metrics']['full']['recall_at_10']}`",
        f"- full MRR@10: `{payload['metrics']['full']['mrr_at_10']}`",
        f"- full nDCG@10: `{payload['metrics']['full']['ndcg_at_10']}`",
        f"- largest retrieval bad case: `{max(bad_cases['distribution'], key=bad_cases['distribution'].get) if bad_cases['distribution'] else None}`",
        "",
        "No retriever parameters were changed for this benchmark.",
    ]
    (DOCS / "retrieval-baseline-v1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "COMPLETED", "metrics": payload["metrics"]["full"]}, ensure_ascii=False))
    return 0 if payload["retrieval_completed"] == len(records) else 2


if __name__ == "__main__":
    raise SystemExit(main())
