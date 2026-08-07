from __future__ import annotations

# ruff: noqa: E501
import argparse
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from paper_research.evaluation.rag_benchmark import (
    aggregate_generation,
    dataset_hash,
    evaluate_generation_item,
    read_jsonl,
    write_json,
)

ROOT = Path("data/evaluation/rag-benchmark")
DOCS = Path("docs/rag-benchmark")
GOLD = Path("data/evaluation/gold-set-v1.jsonl")
CONFIG = ROOT / "baseline-config-v1.json"
DEFAULT_RESULTS = Path("data/evaluation/deepseek-full-qa-final-items-v1.jsonl")


def stratify_generation(
    rows: list[dict[str, Any]],
    gold_by_id: dict[str, dict[str, Any]],
    field: str,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(gold_by_id[row["question_id"]].get(field, "unknown"))].append(row)
    return {key: aggregate_generation(value) for key, value in sorted(grouped.items())}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-file", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-json", type=Path, default=ROOT / "generation-benchmark-v1.json")
    args = parser.parse_args()

    gold = read_jsonl(GOLD)
    gold_by_id = {row["question_id"]: row for row in gold}
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    use_results = args.results_file.exists() and not args.dry_run
    items = read_jsonl(args.results_file) if use_results else []
    per_question = [evaluate_generation_item(item) for item in items]
    aggregate = aggregate_generation(per_question) if per_question else {}
    payload = {
        "schema_version": "rag-generation-benchmark-v1",
        "status": "HISTORICAL_EXISTING_RUN_EVALUATED" if use_results else "READY_NOT_RUN",
        "started_at": datetime.now(UTC).isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
        "git_commit": config["git_commit"],
        "dataset_version": "gold-set-v1",
        "dataset_hash": dataset_hash(gold),
        "baseline_config_hash": config["baseline_config_hash"],
        "provider": config["llm"]["provider"],
        "model": config["llm"]["model"],
        "embedding_model": config["embedding"]["model"],
        "generation_parameters": config["generation"],
        "tokens": aggregate.get("total_tokens", 0),
        "cost": aggregate.get("cost", 0.0),
        "llm_judge": "NOT_USED",
        "result_source": str(args.results_file) if use_results else None,
        "per_question": per_question,
        "aggregate": aggregate,
        "by_category": stratify_generation(per_question, gold_by_id, "category") if per_question else {},
        "by_difficulty": stratify_generation(per_question, gold_by_id, "difficulty") if per_question else {},
        "required_metrics": [
            "required_claim_coverage",
            "supported_claim_ratio",
            "citation_precision",
            "citation_recall",
            "answer_completeness",
            "abstention_accuracy",
            "latency",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "cost",
        ],
        "failure_stage_semantics": [
            "retrieval failed",
            "retrieval succeeded but generation failed",
        ],
    }
    write_json(args.output_json, payload)
    DOCS.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Generation Benchmark v1",
        "",
        f"- status: `{payload['status']}`",
        f"- dataset_hash: `{payload['dataset_hash']}`",
        f"- baseline_config_hash: `{payload['baseline_config_hash']}`",
        f"- LLM judge: `{payload['llm_judge']}`",
        "",
    ]
    if not per_question:
        lines.append("Harness is ready. No generation result file was evaluated.")
    else:
        lines.append("## Aggregate")
        lines.append("")
        lines.extend(f"- {key}: {value}" for key, value in aggregate.items())
    (DOCS / "generation-benchmark-v1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
