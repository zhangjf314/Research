from __future__ import annotations

# ruff: noqa: E501
import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from paper_research.evaluation.rag_benchmark import (
    aggregate_retrieval,
    dataset_hash,
    evaluate_retrieval_question,
    read_jsonl,
    stratify,
    write_json,
)

ROOT = Path("data/evaluation/rag-benchmark")
DOCS = Path("docs/rag-benchmark")
GOLD = Path("data/evaluation/retrieval-gold-v2.jsonl")
CONFIG = ROOT / "baseline-config-v1.json"


def load_ranked_results(path: Path | None) -> dict[str, list[dict[str, Any]]]:
    if path is None:
        return {}
    records = read_jsonl(path) if path.suffix == ".jsonl" else json.loads(path.read_text(encoding="utf-8"))
    if isinstance(records, dict):
        records = records.get("per_question", records.get("items", []))
    return {row["question_id"]: row.get("ranked_results", []) for row in records}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-file", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and emit ready metadata only.")
    parser.add_argument("--output-json", type=Path, default=ROOT / "retrieval-benchmark-v1.json")
    args = parser.parse_args()

    gold = read_jsonl(GOLD)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    ranked_by_question = load_ranked_results(args.results_file)
    dry_run = args.dry_run or not ranked_by_question
    per_question: list[dict[str, Any]] = []
    if not dry_run:
        for record in gold:
            per_question.append(
                evaluate_retrieval_question(
                    {
                        **record,
                        "answerable": record.get("retrieval_scope") != "unanswerable",
                    },
                    ranked_by_question.get(record["question_id"], []),
                )
            )
    aggregate = aggregate_retrieval(per_question) if per_question else {}
    gold_by_id = {row["question_id"]: row for row in gold}
    payload = {
        "schema_version": "rag-retrieval-benchmark-v1",
        "status": "READY_NOT_RUN" if dry_run else "COMPLETED",
        "started_at": datetime.now(UTC).isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
        "git_commit": config["git_commit"],
        "dataset_version": "retrieval-gold-v2",
        "dataset_hash": dataset_hash(gold),
        "baseline_config_hash": config["baseline_config_hash"],
        "provider": config["llm"]["provider"],
        "model": config["llm"]["model"],
        "embedding_model": config["embedding"]["model"],
        "retrieval_config": config["retrieval"],
        "tokens": 0,
        "cost": 0.0,
        "dry_run": dry_run,
        "per_question": per_question,
        "aggregate": aggregate,
        "by_category": stratify(per_question, gold_by_id, "category") if per_question else {},
        "by_difficulty": stratify(per_question, gold_by_id, "difficulty") if per_question else {},
        "required_metrics": [
            "Recall@5",
            "Recall@10",
            "Recall@20",
            "MRR@10",
            "nDCG@10",
            "Paper Recall@5",
            "Paper Recall@10",
            "Evidence Coverage@5",
            "Evidence Coverage@10",
            "Evidence Coverage@20",
            "irrelevant_retrieval_rate",
        ],
    }
    write_json(args.output_json, payload)
    DOCS.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Retrieval Benchmark v1",
        "",
        f"- status: `{payload['status']}`",
        f"- dataset_hash: `{payload['dataset_hash']}`",
        f"- baseline_config_hash: `{payload['baseline_config_hash']}`",
        f"- dry_run: `{dry_run}`",
        "",
    ]
    if dry_run:
        lines.append("Harness is ready. No real retrieval run was executed in Stage 1.")
    else:
        lines.append("## Aggregate")
        lines.append("")
        lines.extend(f"- {key}: {value}" for key, value in aggregate.items())
    (DOCS / "retrieval-benchmark-v1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
