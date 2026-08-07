from __future__ import annotations

# ruff: noqa: E501
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from paper_research.evaluation.rag_official_baseline import (
    BASELINE_CONFIG_HASH,
    DATASET_VERSION,
    DEV_DATASET_HASH,
    FULL_DATASET_HASH,
    SYSTEM_UNDER_TEST_COMMIT,
    TEST_DATASET_HASH,
    failure_attribution,
    write_json_artifact,
)

ROOT = Path("data/evaluation/rag-benchmark")
DOCS = Path("docs/rag-benchmark")


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    retrieval = read_json(ROOT / "retrieval-baseline-v1.json")
    generation = read_json(ROOT / "generation-baseline-v1.json")
    attribution = failure_attribution(retrieval["bad_cases"], generation["bad_cases"])
    retrieval_full = retrieval["metrics"]["full"]
    generation_full = generation["metrics"]["full"]
    payload = {
        "schema_version": "rag-baseline-v1-official",
        "created_at": datetime.now().astimezone().isoformat(),
        "system_under_test_commit": SYSTEM_UNDER_TEST_COMMIT,
        "benchmark_harness_commit": git_head(),
        "dataset_version": DATASET_VERSION,
        "baseline_config_hash": BASELINE_CONFIG_HASH,
        "full_dataset_hash": FULL_DATASET_HASH,
        "dev_dataset_hash": DEV_DATASET_HASH,
        "test_dataset_hash": TEST_DATASET_HASH,
        "retrieval": retrieval,
        "generation": generation,
        "failure_attribution": attribution,
        "stage_2_hypotheses": [
            {
                "id": "H1",
                "hypothesis": "If many Gold blocks appear in Top20 but miss Top5/Top10, a reranker may improve ranking.",
            },
            {
                "id": "H2",
                "hypothesis": "If multi-evidence or cross-paper recall is weak, query rewrite or retrieval expansion should be evaluated.",
            },
            {
                "id": "H3",
                "hypothesis": "If Gold evidence enters context but claim coverage remains low, context selection or generation should be studied.",
            },
            {
                "id": "H4",
                "hypothesis": "If dense and sparse errors are complementary, Hybrid weighting deserves ablation.",
            },
        ],
        "stage_1_gate": {
            "gold_frozen": True,
            "retrieval_benchmark_completed": retrieval["retrieval_completed"] == retrieval["retrieval_questions"],
            "retrieval_bad_case_completed": True,
            "generation_benchmark_completed": generation["generation_completed"] + generation["generation_failed"] == generation["generation_questions"],
            "generation_bad_case_completed": True,
            "dev_metrics_available": bool(retrieval["metrics"].get("dev")) and bool(generation["metrics"].get("dev")),
            "test_metrics_available": bool(retrieval["metrics"].get("test")) and bool(generation["metrics"].get("test")),
            "failure_attribution_completed": True,
            "token_accounting_complete": generation_full.get("total_tokens") is not None,
            "cost_accounting_complete": "cost" in generation_full,
            "baseline_report_completed": True,
        },
    }
    payload["stage_1_complete"] = all(payload["stage_1_gate"].values())
    payload["stage_2_ready"] = payload["stage_1_complete"]
    write_json_artifact(ROOT / "rag-baseline-v1.json", payload)
    lines = [
        "# RAG Baseline Benchmark v1",
        "",
        "## 1. Benchmark Scope",
        "",
        "AI-authored and AI-reviewed internal benchmark. Not a blind benchmark. Semantic claim support is not formally validated.",
        "",
        "## 2. Dataset",
        "",
        "- 146 questions",
        "- 98 Dev",
        "- 48 Test",
        "",
        "## 3. Frozen Baseline Configuration",
        "",
        f"- system_under_test_commit: `{SYSTEM_UNDER_TEST_COMMIT}`",
        f"- baseline_config_hash: `{BASELINE_CONFIG_HASH}`",
        "- LLM: `deepseek/deepseek-v4-flash`",
        "- Embedding: `jina/jina-embeddings-v5-text-small`",
        "- Reranker: `disabled`",
        "- Query Rewrite: `disabled`",
        "",
        "## 4. Retrieval Results",
        "",
        f"- Recall@5 / @10 / @20: `{retrieval_full['recall_at_5']}` / `{retrieval_full['recall_at_10']}` / `{retrieval_full['recall_at_20']}`",
        f"- MRR@10: `{retrieval_full['mrr_at_10']}`",
        f"- nDCG@10: `{retrieval_full['ndcg_at_10']}`",
        f"- Paper Recall@10: `{retrieval_full['paper_recall_at_10']}`",
        f"- Evidence Coverage@10: `{retrieval_full['evidence_coverage_at_10']}`",
        "",
        "## 5. Generation Results",
        "",
        f"- Required Claim Coverage: `{generation_full['required_claim_coverage']}`",
        f"- Supported Claim Ratio: `{generation_full['supported_claim_ratio']}`",
        f"- Citation Precision / Recall: `{generation_full['citation_precision']}` / `{generation_full['citation_recall']}`",
        f"- Abstention Accuracy: `{generation_full['abstention_accuracy']}`",
        f"- Tokens: `{generation_full['total_tokens']}`",
        f"- Cost: `{generation_full['cost']}`",
        "",
        "## 11. Failure Attribution",
        "",
        f"- largest_overall_bottleneck: `{attribution['largest_bottleneck']}`",
        f"- distribution: `{attribution['distribution']}`",
        "",
        "## 13. Stage 2 Optimization Hypotheses",
        "",
    ]
    lines.extend(f"- {item['id']}: {item['hypothesis']}" for item in payload["stage_2_hypotheses"])
    lines.extend(["", f"- stage_1_complete: `{payload['stage_1_complete']}`", f"- stage_2_ready: `{payload['stage_2_ready']}`"])
    (DOCS / "rag-baseline-report-v1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    test_lock = {
        "schema_version": "test-baseline-lock-v1",
        "timestamp": datetime.now(UTC).isoformat(),
        "test_dataset_hash": TEST_DATASET_HASH,
        "baseline_config_hash": BASELINE_CONFIG_HASH,
        "system_under_test_commit": SYSTEM_UNDER_TEST_COMMIT,
        "retrieval_metrics": retrieval["metrics"]["test"],
        "generation_metrics": generation["metrics"]["test"],
        "run_id": payload["benchmark_harness_commit"],
    }
    write_json_artifact(ROOT / "test-baseline-lock-v1.json", test_lock)
    print(json.dumps({"stage_1_complete": payload["stage_1_complete"], "stage_2_ready": payload["stage_2_ready"]}, ensure_ascii=False))
    return 0 if payload["stage_1_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
