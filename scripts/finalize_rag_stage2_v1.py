from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from paper_research.evaluation.rag_stage2d import (
    behavioral_config_projection,
    canonical_json_hash,
    component_decisions,
    final_config_from_baseline,
    metric_row,
    read_json,
    validate_component_decisions,
    validate_stage2_artifacts,
    write_json,
)

OPT_ROOT = Path("data/evaluation/rag-optimization")
OPT_DOCS = Path("docs/rag-optimization")
RAG_ROOT = Path("data/evaluation/rag-benchmark")
STAGE3_ROOT = Path("data/evaluation/research-agent")


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def main() -> int:
    baseline_config = read_json(RAG_ROOT / "baseline-config-v1.json")
    baseline = read_json(RAG_ROOT / "rag-baseline-v1.json")
    stage2a = read_json(OPT_ROOT / "stage2a-retrieval-ablation-v1.json")
    stage2b = read_json(OPT_ROOT / "stage2b-query-rewrite-v1.json")
    stage2c = read_json(OPT_ROOT / "stage2c-context-selection-v1.json")
    funnel = read_json(OPT_ROOT / "stage2c-evidence-funnel-v1.json")
    validation = validate_stage2_artifacts(stage2a, stage2b, stage2c)
    if not validation["valid"]:
        raise RuntimeError(f"STAGE2_PROTOCOL_INVALID: {validation['issues']}")
    decisions = component_decisions()
    validate_component_decisions(decisions)
    final_config = final_config_from_baseline(baseline_config, decisions)
    final_hash = canonical_json_hash(behavioral_config_projection(final_config))
    final_config["stage2_final_config_hash"] = final_hash
    write_json(OPT_ROOT / "rag-stage2-final-config-v1.json", final_config)

    registry = build_registry(stage2a, stage2b, stage2c, decisions)
    write_json(OPT_ROOT / "stage2-experiment-registry-v1.json", registry)

    ablation = build_ablation_payload(
        baseline,
        baseline_config,
        stage2a,
        stage2b,
        stage2c,
        funnel,
        decisions,
        final_config,
        final_hash,
        validation,
    )
    write_json(OPT_ROOT / "rag-stage2-final-ablation-v1.json", ablation)
    write_report(OPT_DOCS / "rag-stage2-final-ablation-v1.md", ablation)
    lock = build_lock(baseline, baseline_config, decisions, final_hash)
    write_json(OPT_ROOT / "rag-stage2-final-lock-v1.json", lock)
    stage3_lock = build_stage3_lock(final_config, final_hash, baseline)
    write_json(STAGE3_ROOT / "stage3-rag-backend-lock-v1.json", stage3_lock)
    print(
        json.dumps(
            {
                "status": "COMPLETED",
                "stage2_complete": True,
                "stage2_final_config_hash": final_hash,
                "new_provider_requests": 0,
                "new_test_questions_evaluated": 0,
            },
            ensure_ascii=False,
        )
    )
    return 0


def build_registry(
    stage2a: dict[str, Any],
    stage2b: dict[str, Any],
    stage2c: dict[str, Any],
    decisions: dict[str, str],
) -> dict[str, Any]:
    return {
        "schema_version": "stage2-experiment-registry-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": git_head(),
        "experiments": [
            _registry_item(
                "dense_only",
                "2A",
                "retriever",
                "Dense Only",
                decisions["dense_only"],
                "R1_dense_only-items-v1.jsonl",
                "Hybrid complementarity gate favored Current Hybrid.",
            ),
            _registry_item(
                "sparse_only",
                "2A",
                "retriever",
                "Sparse Only",
                decisions["sparse_only"],
                "R2_sparse_only-items-v1.jsonl",
                "Hybrid complementarity gate favored Current Hybrid.",
            ),
            _registry_item(
                "current_hybrid",
                "2A",
                "retriever",
                "Current Hybrid",
                decisions["current_hybrid"],
                "stage2a-retrieval-ablation-v1.json",
                "Dense and sparse complementarity was validated.",
                gate="PASSED",
            ),
            _registry_item(
                "lexical_rerank",
                "2A",
                "reranker",
                "Hybrid + Lexical Rerank",
                decisions["lexical_rerank"],
                "RR1_hybrid_lexical_rerank-items-v1.jsonl",
                "Recall/evidence coverage gains did not meet preregistered threshold.",
            ),
            _registry_item(
                "single_rewrite",
                "2B",
                "query_rewrite",
                "Single Rewrite",
                decisions["single_rewrite"],
                "stage2b-query-rewrite-v1.json",
                "Recall@10 regressed against Q0 Current Hybrid.",
            ),
            _registry_item(
                "original_plus_rewrite",
                "2B",
                "query_rewrite",
                "Original + Rewrite",
                decisions["original_plus_rewrite"],
                "stage2b-query-rewrite-v1.json",
                "Fusion with rewritten query substantially reduced Recall@10.",
            ),
            _registry_item(
                "query_decomposition",
                "2B",
                "query_decomposition",
                "Original + Decomposition",
                decisions["query_decomposition"],
                "stage2b-query-rewrite-v1.json",
                "Decomposition reduced retrieval quality against Q0.",
            ),
            _registry_item(
                "baseline_context",
                "2C",
                "context_selection",
                "Baseline Context",
                decisions["baseline_context"],
                "stage2c-context-selection-v1.json",
                "No candidate selector passed offline gate.",
                gate="BASELINE",
            ),
            _registry_item(
                "score_budgeted_dedup_context",
                "2C",
                "context_selection",
                "Score-Budgeted Dedup Context",
                decisions["score_budgeted_dedup_context"],
                "stage2c-context-selection-v1.json",
                "Small coverage gain was below offline gate threshold.",
            ),
            _registry_item(
                "diversity_aware_context",
                "2C",
                "context_selection",
                "Diversity-Aware Context",
                decisions["diversity_aware_context"],
                "stage2c-context-selection-v1.json",
                "Coverage regressed.",
            ),
        ],
        "stage2_test_optimization_runs": (
            int(stage2a.get("test_questions_evaluated") or 0)
            + int(stage2b.get("test_questions_evaluated") or 0)
            + int(stage2c.get("test_questions_evaluated") or 0)
        ),
    }


def _registry_item(
    experiment_id: str,
    stage: str,
    component: str,
    label: str,
    decision: str,
    artifact: str,
    reason: str,
    *,
    gate: str = "FAILED",
) -> dict[str, Any]:
    return {
        "experiment_id": experiment_id,
        "stage": stage,
        "component": component,
        "label": label,
        "status": "COMPLETED",
        "selected": decision == "SELECTED",
        "dataset_split": "dev",
        "primary_metric": "stage_preregistered_gate",
        "gate": gate,
        "decision": decision,
        "artifact": str(OPT_ROOT / artifact),
        "reason": reason,
    }


def build_ablation_payload(
    baseline: dict[str, Any],
    baseline_config: dict[str, Any],
    stage2a: dict[str, Any],
    stage2b: dict[str, Any],
    stage2c: dict[str, Any],
    funnel: dict[str, Any],
    decisions: dict[str, str],
    final_config: dict[str, Any],
    final_hash: str,
    validation: dict[str, Any],
) -> dict[str, Any]:
    m2a = stage2a["metrics"]
    m2b = stage2b["metrics"]
    context_metrics = stage2c["offline_metrics"]
    retrieval_table = [
        metric_row("Dense Only", m2a["R1_dense_only"], decisions["dense_only"]),
        metric_row("Sparse Only", m2a["R2_sparse_only"], decisions["sparse_only"]),
        metric_row(
            "Current Hybrid",
            m2a["R3_current_hybrid"],
            decisions["current_hybrid"],
            required_claim_metric=m2b["Q0_CURRENT_HYBRID"].get(
                "required_claim_evidence_coverage_at_10"
            ),
        ),
        metric_row(
            "Hybrid + Lexical Rerank",
            m2a["RR1_hybrid_lexical_rerank"],
            decisions["lexical_rerank"],
        ),
        metric_row(
            "Single Rewrite",
            m2b["Q1_SINGLE_REWRITE_REPLACE"],
            decisions["single_rewrite"],
            required_claim_metric=m2b["Q1_SINGLE_REWRITE_REPLACE"].get(
                "required_claim_evidence_coverage_at_10"
            ),
        ),
        metric_row(
            "Original + Rewrite",
            m2b["Q2_ORIGINAL_PLUS_SINGLE_REWRITE"],
            decisions["original_plus_rewrite"],
            required_claim_metric=m2b["Q2_ORIGINAL_PLUS_SINGLE_REWRITE"].get(
                "required_claim_evidence_coverage_at_10"
            ),
        ),
        metric_row(
            "Original + Decomposition",
            m2b["Q3_ORIGINAL_PLUS_DECOMPOSITION"],
            decisions["query_decomposition"],
            required_claim_metric=m2b["Q3_ORIGINAL_PLUS_DECOMPOSITION"].get(
                "required_claim_evidence_coverage_at_10"
            ),
        ),
    ]
    context_table = [
        _context_row("Baseline Context", context_metrics["C0_BASELINE"], "BASELINE_RETAINED"),
        _context_row(
            "Score-Budgeted Dedup",
            context_metrics["C1_SCORE_BUDGETED_DEDUP"],
            "REJECTED",
        ),
        _context_row("Diversity-Aware", context_metrics["C2_DIVERSITY_AWARE"], "REJECTED"),
    ]
    baseline_generation = baseline["generation"]["metrics"]["full"]
    return {
        "schema_version": "rag-stage2-final-ablation-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": git_head(),
        "dataset_version": baseline["dataset_version"],
        "full_dataset_hash": baseline["full_dataset_hash"],
        "dev_dataset_hash": baseline["dev_dataset_hash"],
        "test_dataset_hash": baseline["test_dataset_hash"],
        "baseline_config_hash": baseline_config["baseline_config_hash"],
        "stage2_artifact_validation": validation,
        "stage2a_status": "COMPLETED",
        "stage2b_status": "COMPLETED",
        "stage2c_status": "COMPLETED",
        "retrieval_ablation_table": retrieval_table,
        "context_ablation_table": context_table,
        "component_decisions": decisions,
        "component_decision_matrix": build_decision_matrix(),
        "hybrid_selected": True,
        "reranker_selected": False,
        "query_rewrite_selected": False,
        "query_decomposition_selected": False,
        "context_selector_selected": False,
        "context_selection_bottleneck_confirmed": True,
        "final_config": final_config,
        "stage2_final_config_hash": final_hash,
        "stage2_final_behavior_change": False,
        "stage2_final_behaviorally_equivalent_to_stage1_baseline": True,
        "stage2_final_test_source": "REUSED_STAGE1_FROZEN_BASELINE",
        "new_test_questions_evaluated": 0,
        "test_protocol_violation": False,
        "new_provider_requests": {
            "retrieval_provider_new_calls": 0,
            "rewrite_provider_new_calls": 0,
            "generation_provider_new_calls": 0,
            "deep_research_calls": 0,
        },
        "new_tokens": 0,
        "new_cost": 0,
        "stage2b_cost_gap": {
            "status": stage2b["rewrite_usage"]["cost_accounting_status"],
            "effective_provider_requests": stage2b["rewrite_usage"][
                "effective_provider_requests_for_artifact"
            ],
            "estimated_tokens_from_cache_text": stage2b["rewrite_usage"]["total_tokens"],
            "exact_cost_available": False,
        },
        "stage2c_trace_limitations": {
            "context_trace_source": stage2c["context_trace_source"],
            "captured_at_original_generation_time": False,
        },
        "strict_success_definition": (
            "SUCCESS in the Stage 2C funnel means strict full-question success: all "
            "benchmark-required claims pass retrieval, context retention, generation, and citation."
        ),
        "stage1_generation_full_metrics": baseline_generation,
        "remaining_bottlenecks": {
            "retrieval": "UNRESOLVED",
            "context": "CONFIRMED_UNRESOLVED",
            "generation": "CONFIRMED_DIAGNOSTIC",
            "citation": "UNRESOLVED",
        },
        "stage3_ready": True,
        "stage2_complete": True,
    }


def _context_row(label: str, metrics: dict[str, Any], decision: str) -> dict[str, Any]:
    return {
        "configuration": label,
        "required_claim_context_coverage": metrics[
            "required_claim_evidence_coverage_in_final_context"
        ],
        "full_context_coverage": metrics[
            "full_required_claim_evidence_coverage_in_final_context"
        ],
        "p95_tokens": metrics["context_token_p95"],
        "context_gold_density": metrics["context_gold_density"],
        "context_redundancy": metrics["context_redundancy"],
        "decision": decision,
    }


def build_decision_matrix() -> list[dict[str, str]]:
    return [
        {
            "component": "Hybrid",
            "hypothesis": "Dense/Sparse complementary",
            "evidence": "Supported",
            "gate": "Passed",
            "decision": "SELECTED",
        },
        {
            "component": "Lexical Rerank",
            "hypothesis": "Deep candidates can be promoted",
            "evidence": "Partial",
            "gate": "Failed",
            "decision": "REJECTED",
        },
        {
            "component": "Single Rewrite",
            "hypothesis": "Better search formulation improves recall",
            "evidence": "Not supported",
            "gate": "Failed",
            "decision": "REJECTED",
        },
        {
            "component": "Multi-query Rewrite",
            "hypothesis": "Original plus rewrite improves retrieval",
            "evidence": "Contradicted",
            "gate": "Failed",
            "decision": "REJECTED",
        },
        {
            "component": "Query Decomposition",
            "hypothesis": "Complex questions benefit from decomposition",
            "evidence": "Contradicted",
            "gate": "Failed",
            "decision": "REJECTED",
        },
        {
            "component": "Context Selection Bottleneck",
            "hypothesis": "Evidence is lost between retrieval and final context",
            "evidence": "Confirmed",
            "gate": "Diagnostic",
            "decision": "SUPPORTED_BUT_UNRESOLVED",
        },
        {
            "component": "Score Context",
            "hypothesis": "Deduplication reduces evidence loss",
            "evidence": "Weak",
            "gate": "Failed",
            "decision": "REJECTED",
        },
        {
            "component": "Diversity Context",
            "hypothesis": "Diversity improves multi-paper evidence",
            "evidence": "Contradicted",
            "gate": "Failed",
            "decision": "REJECTED",
        },
    ]


def build_lock(
    baseline: dict[str, Any],
    baseline_config: dict[str, Any],
    decisions: dict[str, str],
    final_hash: str,
) -> dict[str, Any]:
    return {
        "schema_version": "rag-stage2-final-lock-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": git_head(),
        "dataset_version": baseline["dataset_version"],
        "dev_dataset_hash": baseline["dev_dataset_hash"],
        "test_dataset_hash": baseline["test_dataset_hash"],
        "baseline_config_hash": baseline_config["baseline_config_hash"],
        "stage2_final_config_hash": final_hash,
        "selected_components": {
            key: value for key, value in decisions.items() if value == "SELECTED"
        },
        "rejected_components": {
            key: value for key, value in decisions.items() if value == "REJECTED"
        },
        "stage2_final_behavior_change": False,
        "stage2_final_test_source": "REUSED_STAGE1_FROZEN_BASELINE",
        "new_test_questions_evaluated": 0,
    }


def build_stage3_lock(
    final_config: dict[str, Any], final_hash: str, baseline: dict[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": "stage3-rag-backend-lock-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": git_head(),
        "rag_backend": {
            "retrieval": "Current Hybrid",
            "reranker": "disabled",
            "query_rewrite": "disabled",
            "query_decomposition": "disabled",
            "context_selector": "baseline",
        },
        "embedding": final_config["embedding"],
        "retrieval_parameters": final_config["retrieval"],
        "corpus_index_identity": {
            "dataset_version": baseline["dataset_version"],
            "dev_dataset_hash": baseline["dev_dataset_hash"],
            "test_dataset_hash": baseline["test_dataset_hash"],
            "production_collection": final_config["retrieval"]["production_collection"],
        },
        "stage2_final_config_hash": final_hash,
        "agent_must_not_enable": [
            "lexical_reranker",
            "query_rewrite",
            "query_decomposition",
            "C1_context_selector",
            "C2_context_selector",
        ],
    }


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# RAG Stage 2 Final Ablation",
        "",
        "## Scope",
        "",
        "Stage 2 finalized predefined RAG optimization experiments. It did not add "
        "new retrievers, rerankers, rewrites, context selectors, prompts, embeddings, "
        "models, or agent behavior.",
        "",
        "## Experimental Protocol",
        "",
        f"- dataset: `{payload['dataset_version']}`",
        f"- dev hash: `{payload['dev_dataset_hash']}`",
        f"- test hash: `{payload['test_dataset_hash']}`",
        "- Stage 2 split: `dev` only",
        "- new TEST runs: `0`",
        "- new provider requests in Stage 2D: `0`",
        "",
        "## Retrieval Ablation Table",
        "",
        "| Configuration | R@10 | MRR@10 | nDCG@10 | EvidenceCov@10 | "
        "ReqClaimEvidenceCov@10 | Decision |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in payload["retrieval_ablation_table"]:
        lines.append(
            f"| {row['configuration']} | {_fmt(row['recall_at_10'])} | "
            f"{_fmt(row['mrr_at_10'])} | {_fmt(row['ndcg_at_10'])} | "
            f"{_fmt(row['evidence_coverage_at_10'])} | "
            f"{_fmt(row['required_claim_evidence_coverage_at_10'])} | "
            f"{row['decision']} |"
        )
    lines.extend(
        [
            "",
            "## Context Ablation Table",
            "",
            "| Configuration | ReqClaim Context Cov | Full Context Cov | P95 Tokens | Decision |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in payload["context_ablation_table"]:
        lines.append(
            f"| {row['configuration']} | "
            f"{_fmt(row['required_claim_context_coverage'])} | "
            f"{_fmt(row['full_context_coverage'])} | {_fmt(row['p95_tokens'])} | "
            f"{row['decision']} |"
        )
    lines.extend(
        [
            "",
            "Context Selection Bottleneck: `CONFIRMED`.",
            "",
            "Effective Selector: `NOT FOUND IN PREREGISTERED EXPERIMENTS`.",
            "",
            "## Component Decision Matrix",
            "",
            "| Component | Hypothesis | Evidence | Gate | Decision |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in payload["component_decision_matrix"]:
        lines.append(
            f"| {row['component']} | {row['hypothesis']} | {row['evidence']} | "
            f"{row['gate']} | {row['decision']} |"
        )
    lines.extend(
        [
            "",
            "## Final RAG Configuration",
            "",
            "- final retriever: `Current Hybrid`",
            "- final reranker: `Disabled`",
            "- final query rewrite: `Disabled`",
            "- final query decomposition: `Disabled`",
            "- final context selector: `Baseline`",
            f"- behavior change: `{payload['stage2_final_behavior_change']}`",
            f"- final config hash: `{payload['stage2_final_config_hash']}`",
            "",
            "## Held-out Test Status",
            "",
            f"- source: `{payload['stage2_final_test_source']}`",
            f"- new_test_questions_evaluated: `{payload['new_test_questions_evaluated']}`",
            "",
            "Because the final Stage 2 behavior is equivalent to the Stage 1 frozen "
            "baseline behavior, Stage 2 reuses the Stage 1 frozen TEST baseline instead "
            "of rerunning 48 TEST retrieval/generation cases.",
            "",
            "## Negative Findings",
            "",
            "### Lexical Rerank",
            "",
            "MRR improved, but retrieval coverage did not materially improve enough to "
            "meet the preregistered recall/evidence-coverage gate.",
            "",
            "### Query Rewrite",
            "",
            "All preregistered rewrite strategies failed to beat the untouched Hybrid "
            "baseline. Q2/Q3 substantially reduced Recall@10.",
            "",
            "### Context Selector",
            "",
            "The evidence-drop bottleneck is real, but simple deterministic "
            "dedup/diversity heuristics did not recover enough required-claim evidence.",
            "",
            "## Remaining Bottlenecks",
            "",
            "- Retrieval miss: `UNRESOLVED`",
            "- Context evidence drop: `CONFIRMED_UNRESOLVED`",
            "- Generation utilization: `CONFIRMED_DIAGNOSTIC`",
            "- Citation exactness: `UNRESOLVED`",
            "",
            "## Stage 2 Limitations",
            "",
            "- Stage 2B exact rewrite cost is unavailable because the first sanitized "
            "cache schema did not persist complete provider usage.",
            "- Stage 2C context traces are deterministic reconstructions, not original "
            "generation-time telemetry.",
            "- Stage 2C `SUCCESS=0` means strict full-question success; it does not mean "
            "all 88 answerable questions were entirely wrong.",
            "",
            "## Stage 3 Handoff",
            "",
            "Stage 3 must use the Stage 2 frozen Hybrid backend. It must not automatically "
            "enable rejected Stage 2 components.",
            "",
            "## Final Conclusion",
            "",
            "Stage 2 experimentally evaluated four predefined optimization families. "
            "Hybrid retrieval was validated and retained. Lexical reranking improved "
            "rank-sensitive metrics but failed the preregistered recall/evidence gate. "
            "Query rewriting and decomposition regressed retrieval quality. Context "
            "selection was confirmed as a major evidence-loss stage, but the two "
            "preregistered deterministic selectors did not pass the offline gate. No "
            "unsupported component was promoted into the final system.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
