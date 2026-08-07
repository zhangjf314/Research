from __future__ import annotations

# ruff: noqa: E501
import argparse
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from paper_research.config import get_settings
from paper_research.evaluation.rag_benchmark import (
    audit_gold,
    canonical_json_hash,
    classify_bad_case,
    dataset_hash,
    evaluate_generation_item,
    file_sha256,
    read_jsonl,
    write_json,
)
from paper_research.version import __version__

ROOT = Path("data/evaluation/rag-benchmark")
DOCS = Path("docs/rag-benchmark")
GOLD = Path("data/evaluation/gold-set-v1.jsonl")
RETRIEVAL_GOLD = Path("data/evaluation/retrieval-gold-v2.jsonl")
HISTORICAL_QA = Path("data/evaluation/deepseek-full-qa-final-items-v1.jsonl")


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def current_baseline_config() -> dict[str, Any]:
    settings = get_settings()
    return {
        "schema_version": "rag-baseline-config-v1",
        "git_commit": git_commit(),
        "runtime_version": __version__,
        "llm": {
            "provider": settings.llm_provider_name or settings.llm_provider,
            "model": settings.llm_model,
            "response_format": settings.llm_response_format,
            "temperature": settings.llm_temperature,
            "max_output_tokens": settings.llm_max_output_tokens,
            "stream": settings.llm_stream,
            "thinking_enabled": settings.llm_thinking_enabled,
        },
        "embedding": {
            "provider": settings.embedding_provider,
            "model": settings.embedding_model,
            "revision": settings.embedding_revision,
            "dimensions": settings.embedding_dimensions,
        },
        "retrieval": {
            "production_collection": settings.production_collection,
            "baseline_collection": settings.baseline_collection,
            "score_threshold": settings.retrieval_score_threshold,
            "recall_k": settings.retrieval_recall_k,
            "top_k": 5,
            "strategy": "current production route; frozen for benchmark",
        },
        "chunking": {
            "chunk_max_tokens": settings.chunk_max_tokens,
            "chunk_overlap_tokens": settings.chunk_overlap_tokens,
        },
        "reranker": {
            "enabled": settings.rerank_enabled,
            "provider": settings.rerank_provider,
            "model": settings.rerank_model,
        },
        "query_rewrite": {"enabled": False, "status": "not implemented in baseline"},
        "context_selection": {
            "strategy": "current production context builder; no Stage 1 optimization"
        },
        "generation": {
            "prompt_version": settings.prompt_version,
            "temperature": settings.llm_temperature,
            "max_output_tokens": settings.llm_max_output_tokens,
        },
        "stage1_constraints": {
            "retrieval_algorithm_changes_allowed": False,
            "prompt_optimization_allowed": False,
            "gold_modification_allowed": False,
            "real_llm_benchmark_run_allowed_without_next_authorization": False,
        },
    }


def write_baseline_docs(config: dict[str, Any], config_hash: str) -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    lines = [
        "# RAG Benchmark Baseline v1",
        "",
        f"- git_commit: `{config['git_commit']}`",
        f"- runtime_version: `{config['runtime_version']}`",
        f"- baseline_config_hash: `{config_hash}`",
        f"- LLM: `{config['llm']['provider']}/{config['llm']['model']}`",
        f"- Embedding: `{config['embedding']['provider']}/{config['embedding']['model']}`",
        f"- Retrieval collection: `{config['retrieval']['production_collection']}`",
        f"- top_k: `{config['retrieval']['top_k']}`",
        f"- chunking: max `{config['chunking']['chunk_max_tokens']}`, overlap `{config['chunking']['chunk_overlap_tokens']}`",
        f"- reranker_enabled: `{config['reranker']['enabled']}`",
        "- query_rewrite: `disabled`",
        "- context_selection: current production context builder; no Stage 1 changes",
        "",
        "Stage 1 freezes this configuration. Do not change these parameters to improve benchmark scores.",
    ]
    (DOCS / "baseline-v1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_gold_docs(audit: dict[str, Any], hash_value: str) -> None:
    lines = [
        "# RAG Benchmark Gold Audit v1",
        "",
        f"- dataset_hash: `{hash_value}`",
        f"- total: {audit['total']}",
        f"- approved: {audit['approved']}",
        f"- answerable: {audit['answerable']}",
        f"- unanswerable: {audit['unanswerable']}",
        "",
        "## Category distribution",
        "",
    ]
    lines.extend(f"- {key}: {value}" for key, value in audit["category_distribution"].items())
    lines.extend(["", "## Difficulty distribution", ""])
    lines.extend(f"- {key}: {value}" for key, value in audit["difficulty_distribution"].items())
    lines.extend(
        [
            "",
            "## Evidence completeness",
            "",
            f"- answerable questions with gold blocks: {audit['gold_evidence_coverage']['answerable_questions_with_gold_blocks']}/{audit['answerable']}",
            f"- total gold block refs: {audit['gold_evidence_coverage']['total_gold_block_refs']}",
            f"- complete_for_answerable: `{audit['gold_evidence_coverage']['complete_for_answerable']}`",
            "",
            "## Gap plan to 150 questions",
            "",
            f"- current approved: {audit['gap_plan']['current_approved_count']}",
            f"- recommended target: {audit['gap_plan']['recommended_target_gold_count']}",
            f"- questions to add: {audit['gap_plan']['questions_to_add']}",
            "- required types: single-hop factual, multi-evidence synthesis, cross-paper comparison, methods / experiments, limitations / research gaps, unanswerable",
            "- all new Gold must receive human review; do not bulk-generate 100 Gold questions with an LLM.",
        ]
    )
    (DOCS / "gold-audit-v1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_bad_cases(items: list[dict[str, Any]]) -> None:
    rows = []
    for item in items:
        failure_stage, bad_type = classify_bad_case(item)
        if failure_stage == "none":
            continue
        rows.append(
            {
                "question_id": item["question_id"],
                "category": item.get("category") or item.get("gold", {}).get("category"),
                "failure_stage": failure_stage,
                "bad_case_type": bad_type,
                "gold_evidence": item.get("gold", {}),
                "retrieved_evidence": {
                    "gold_block_present": item.get("gold_block_present"),
                    "retrieval_query": item.get("retrieval_query"),
                    "retrieval_scope": item.get("retrieval_scope"),
                },
                "generated_answer": item.get("answer", {}).get("answer"),
                "diagnosis": "Historical existing QA run diagnostic; no new model call was made.",
            }
        )
    write_json(ROOT / "bad-cases-v1.json", {"source": str(HISTORICAL_QA), "items": rows})
    lines = [
        "# RAG Benchmark Bad Cases v1",
        "",
        "Source: historical existing Full QA artifact. No new LLM call was made.",
        "",
        f"- bad_case_count: {len(rows)}",
        "",
    ]
    for row in rows[:30]:
        lines.append(
            f"- {row['question_id']}: {row['failure_stage']} / {row['bad_case_type']}"
        )
    (DOCS / "bad-cases-v1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(
    config_hash: str,
    gold_hash: str,
    gold_audit: dict[str, Any],
    generation_rows: list[dict[str, Any]],
) -> None:
    retrieval_problem = "Formal Stage 1 retrieval benchmark has not been run yet; current evidence shows historical gold-block-present gaps in Full QA artifacts."
    generation_problem = "Historical Full QA diagnostics show low required-claim and exact citation recall; semantic support is not formally validated."
    worst_category = max(gold_audit["category_distribution"], key=gold_audit["category_distribution"].get)
    best_category = "not yet measured by the Stage 1 harness"
    gen_retrieval_failures = sum(1 for row in generation_rows if row["failure_stage"] == "retrieval failed")
    gen_generation_failures = sum(
        1 for row in generation_rows if row["failure_stage"] != "retrieval failed"
    )
    payload = {
        "schema_version": "rag-baseline-report-v1",
        "status": "FRAMEWORK_READY_BASELINE_NOT_RERUN",
        "baseline_config_hash": config_hash,
        "dataset_hash": gold_hash,
        "gold_audit": gold_audit,
        "retrieval_benchmark_ready": True,
        "generation_benchmark_ready": True,
        "bad_case_taxonomy_ready": True,
        "answers": {
            "current_retrieval_biggest_problem": retrieval_problem,
            "current_generation_biggest_problem": generation_problem,
            "worst_question_type": worst_category,
            "best_question_type": best_category,
            "unanswerable_performance": "Existing Gold contains two unanswerable questions; formal Stage 1 benchmark run is pending authorization.",
            "failure_source": {
                "retrieval_failed_items_in_historical_generation_artifact": gen_retrieval_failures,
                "generation_or_citation_failed_items_in_historical_generation_artifact": gen_generation_failures,
            },
            "stage2_hypothesis": "Hypothesis only: first validate whether retrieval misses versus context/citation selection dominate exact-Gold failures before changing algorithms.",
        },
    }
    write_json(ROOT / "rag-baseline-v1.json", payload)
    lines = [
        "# RAG Baseline Report v1",
        "",
        "Status: `FRAMEWORK_READY_BASELINE_NOT_RERUN`",
        "",
        f"- baseline_config_hash: `{config_hash}`",
        f"- dataset_hash: `{gold_hash}`",
        f"- retrieval_benchmark_ready: `{payload['retrieval_benchmark_ready']}`",
        f"- generation_benchmark_ready: `{payload['generation_benchmark_ready']}`",
        f"- bad_case_taxonomy_ready: `{payload['bad_case_taxonomy_ready']}`",
        "",
        "## Required answers",
        "",
        f"- Current Retrieval biggest problem: {retrieval_problem}",
        f"- Current Generation biggest problem: {generation_problem}",
        f"- Worst question type: {worst_category}",
        f"- Best question type: {best_category}",
        f"- Unanswerable: {payload['answers']['unanswerable_performance']}",
        f"- Retrieval vs Generation failures: {payload['answers']['failure_source']}",
        f"- Stage 2 hypothesis: {payload['answers']['stage2_hypothesis']}",
        "",
        "No Stage 1 optimization was implemented.",
    ]
    (DOCS / "rag-baseline-report-v1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical-generation-results", type=Path, default=HISTORICAL_QA)
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)

    config = current_baseline_config()
    config_hash = canonical_json_hash(config)
    config["frozen_at"] = datetime.now(UTC).isoformat()
    config["baseline_config_hash"] = config_hash
    write_json(ROOT / "baseline-config-v1.json", config)
    write_baseline_docs(config, config_hash)

    gold_records = read_jsonl(GOLD)
    gold_hash = dataset_hash(gold_records)
    audit = audit_gold(gold_records)
    audit["dataset_hash"] = gold_hash
    audit["source_path"] = str(GOLD)
    audit["retrieval_gold_hash"] = file_sha256(RETRIEVAL_GOLD)
    write_json(ROOT / "gold-audit-v1.json", audit)
    write_gold_docs(audit, gold_hash)

    generation_items = (
        read_jsonl(args.historical_generation_results)
        if args.historical_generation_results.exists()
        else []
    )
    generation_rows = [evaluate_generation_item(item) for item in generation_items]
    write_bad_cases(generation_items)
    write_report(config_hash, gold_hash, audit, generation_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
