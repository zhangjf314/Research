from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime

from paper_research.evaluation.rag_stage2a import (
    OPT_DOCS,
    OPT_ROOT,
    grouped_headroom,
    load_baseline_retrieval_dev_rows,
    retrieval_headroom,
    write_json,
)


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def main() -> int:
    rows = load_baseline_retrieval_dev_rows()
    payload = {
        "schema_version": "retrieval-headroom-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "benchmark_harness_commit": git_head(),
        "dataset_version": "rag-gold-v1",
        "split": "dev",
        "question_count": len(rows),
        "test_questions_evaluated": 0,
        "headroom": retrieval_headroom(rows),
        "by_category": grouped_headroom(rows, "category"),
        "by_difficulty": grouped_headroom(rows, "difficulty"),
    }
    write_json(OPT_ROOT / "retrieval-headroom-v1.json", payload)
    headroom = payload["headroom"]
    lines = [
        "# Retrieval headroom v1",
        "",
        "- split: `dev`",
        f"- question_count: `{payload['question_count']}`",
        "- test_questions_evaluated: `0`",
        f"- same_paper_wrong_block_rate: `{headroom['same_paper_wrong_block_rate']}`",
        f"- top20_to_top10_rerank_headroom: `{headroom['top20_to_top10_rerank_headroom']}`",
        f"- top10_to_top5_rerank_headroom: `{headroom['top10_to_top5_rerank_headroom']}`",
        f"- full_evidence_coverage_rate_at_10: `{headroom['evidence_full_coverage_rate_at_10']}`",
        f"- full_evidence_coverage_rate_at_20: `{headroom['evidence_full_coverage_rate_at_20']}`",
    ]
    OPT_DOCS.mkdir(parents=True, exist_ok=True)
    (OPT_DOCS / "retrieval-headroom-v1.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "COMPLETED", "headroom": headroom}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
