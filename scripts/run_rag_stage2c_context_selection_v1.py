from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from paper_research.evaluation.rag_stage2a import OPT_DOCS, OPT_ROOT
from paper_research.evaluation.rag_stage2c import (
    aggregate_traces,
    build_trace,
    context_selection_hypothesis_supported,
    diversity_aware_context,
    load_chunk_texts,
    offline_selector_gate,
    read_jsonl,
    reconstruct_baseline_context,
    score_budgeted_deduplicated_context,
    write_json,
    write_markdown,
)

FUNNEL_JSON = OPT_ROOT / "stage2c-evidence-funnel-v1.json"
FUNNEL_MD = OPT_DOCS / "stage2c-evidence-funnel-v1.md"
CONTEXT_JSON = OPT_ROOT / "stage2c-context-selection-v1.json"
CONTEXT_MD = OPT_DOCS / "stage2c-context-selection-v1.md"
TRACE_JSONL = OPT_ROOT / "stage2c-generation-trace-v1.jsonl"
PLAN_JSON = OPT_ROOT / "stage2c-context-selection-plan-v1.json"
PLAN_MD = OPT_DOCS / "stage2c-context-selection-plan-v1.md"


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def write_jsonl(path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> int:
    gold_rows = read_jsonl(Path("data/evaluation/rag-benchmark/gold-dev-v1.jsonl"))
    retrieval_rows = read_jsonl(
        Path("data/evaluation/rag-optimization/R3_current_hybrid-items-v1.jsonl")
    )
    generation_rows = read_jsonl(
        Path("data/evaluation/rag-benchmark/generation-baseline-items-v1.jsonl")
    )
    if len(gold_rows) != 98 or any(row.get("split") != "dev" for row in gold_rows):
        raise RuntimeError("TEST_PROTOCOL_VIOLATION")
    retrieval_by_id = {row["question_id"]: row for row in retrieval_rows}
    generation_by_id = {row["question_id"]: row for row in generation_rows}
    chunk_texts = load_chunk_texts()
    traces = []
    selector_contexts: dict[str, dict[str, list[dict[str, Any]]]] = {
        "C0_BASELINE": {},
        "C1_SCORE_BUDGETED_DEDUP": {},
        "C2_DIVERSITY_AWARE": {},
    }
    for gold in gold_rows:
        qid = gold["question_id"]
        retrieval = retrieval_by_id[qid]
        baseline = reconstruct_baseline_context(
            retrieval.get("ranked_results", []), chunk_texts, top_k=5
        )
        token_budget = max(1, sum(int(item.get("estimated_tokens") or 0) for item in baseline))
        c1 = score_budgeted_deduplicated_context(
            retrieval.get("ranked_results", []), chunk_texts, token_budget=token_budget
        )
        c2 = diversity_aware_context(
            retrieval.get("ranked_results", []), chunk_texts, token_budget=token_budget
        )
        selector_contexts["C0_BASELINE"][qid] = baseline
        selector_contexts["C1_SCORE_BUDGETED_DEDUP"][qid] = c1
        selector_contexts["C2_DIVERSITY_AWARE"][qid] = c2
        traces.append(build_trace(gold, retrieval, generation_by_id.get(qid), baseline))
    write_jsonl(TRACE_JSONL, traces)
    c0_metrics = aggregate_traces(traces)
    pre_availability = c0_metrics["required_claim_evidence_available_at_20"]
    hypothesis = context_selection_hypothesis_supported(c0_metrics, pre_availability)
    funnel_payload = {
        "schema_version": "stage2c-evidence-funnel-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "commit": git_head(),
        "split": "dev",
        "dev_questions": len(gold_rows),
        "dev_answerable": sum(1 for row in gold_rows if row.get("answerable")),
        "test_questions_evaluated": 0,
        "test_protocol_violation": False,
        "context_trace_source": "RECONSTRUCTED_DETERMINISTIC",
        "retrieval_config": "Current Hybrid / Stage 2A selected Q0",
        "generation_requests": 0,
        "metrics": c0_metrics,
        "context_selection_hypothesis_supported": hypothesis,
        "hypothesis_gate": {
            "c0_context_selection_drop_min_answerable_rate": 0.15,
            "required_claim_context_retention_min": 0.90,
            "pre_to_final_required_claim_coverage_drop_min": 0.10,
        },
    }
    write_json(FUNNEL_JSON, funnel_payload)
    write_markdown(FUNNEL_MD, funnel_payload, "Stage 2C evidence funnel v1")
    if hypothesis:
        plan_payload = selector_plan_payload(funnel_payload)
        write_json(PLAN_JSON, plan_payload)
        write_markdown(PLAN_MD, plan_payload, "Stage 2C context selection plan v1")
    selector_payload = build_selector_payload(
        gold_rows,
        retrieval_by_id,
        generation_by_id,
        selector_contexts,
        c0_metrics,
        hypothesis,
    )
    write_json(CONTEXT_JSON, selector_payload)
    write_markdown(CONTEXT_MD, selector_payload, "Stage 2C context selection v1")
    print(
        json.dumps(
            {
                "status": "COMPLETED",
                "context_selection_hypothesis_supported": hypothesis,
                "offline_selected_candidate": selector_payload["offline_selected_candidate"],
                "generation_experiment_run": False,
                "test_questions_evaluated": 0,
            },
            ensure_ascii=False,
        )
    )
    return 0


def selector_plan_payload(funnel_payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "stage2c-context-selection-plan-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "commit": git_head(),
        "split": "dev",
        "test_questions_allowed": False,
        "test_questions_evaluated": 0,
        "hypothesis_source": "stage2c-evidence-funnel-v1",
        "context_selection_hypothesis_supported": funnel_payload[
            "context_selection_hypothesis_supported"
        ],
        "base_chain": "Current Hybrid; no reranker; no query rewrite; no generation LLM",
        "selectors": {
            "C0_BASELINE": "Frozen baseline context selection reconstructed deterministically.",
            "C1_SCORE_BUDGETED_DEDUP": (
                "Deterministic score-ordered selector that removes duplicate block coverage "
                "within the baseline token budget."
            ),
            "C2_DIVERSITY_AWARE": (
                "Deterministic selector that caps per-paper and per-section early selection, "
                "then refills by rank within the baseline token budget."
            ),
        },
        "allowed_features": [
            "retrieval rank",
            "retrieval score",
            "paper_id",
            "section_path",
            "chunk text token estimate",
            "block overlap",
        ],
        "forbidden_features": [
            "gold answers",
            "gold block ids",
            "required claims",
            "LLM selector",
            "prompt changes",
            "retrieval changes",
            "reranker",
            "query rewrite",
            "test split",
        ],
        "offline_gate": {
            "required_claim_context_coverage_gain_min": 0.05,
            "full_context_coverage_gain_min": 0.05,
            "token_p95_max_baseline_multiplier": 1.10,
            "single_hop_context_coverage_no_obvious_drop": True,
        },
    }


def build_selector_payload(
    gold_rows: list[dict[str, Any]],
    retrieval_by_id: dict[str, dict[str, Any]],
    generation_by_id: dict[str, dict[str, Any]],
    selector_contexts: dict[str, dict[str, list[dict[str, Any]]]],
    c0_metrics: dict[str, Any],
    hypothesis: bool,
) -> dict[str, Any]:
    metrics = {"C0_BASELINE": c0_metrics}
    for selector in ("C1_SCORE_BUDGETED_DEDUP", "C2_DIVERSITY_AWARE"):
        traces = [
            build_trace(
                gold,
                retrieval_by_id[gold["question_id"]],
                generation_by_id.get(gold["question_id"]),
                selector_contexts[selector][gold["question_id"]],
            )
            for gold in gold_rows
        ]
        metrics[selector] = aggregate_traces(traces)
    gate_results = {
        selector: offline_selector_gate(c0_metrics, metrics[selector])
        for selector in ("C1_SCORE_BUDGETED_DEDUP", "C2_DIVERSITY_AWARE")
    }
    selected = "C0_BASELINE"
    status = "NOT_RUN_HYPOTHESIS_REJECTED"
    if hypothesis:
        passing = [selector for selector, passed in gate_results.items() if passed]
        if passing:
            selected = max(
                passing,
                key=lambda selector: (
                    metrics[selector]["required_claim_evidence_coverage_in_final_context"],
                    metrics[selector]["full_required_claim_evidence_coverage_in_final_context"],
                ),
            )
            status = "GENERATION_NOT_RUN_AWAITING_AUTHORIZATION"
        else:
            status = "NOT_RUN_OFFLINE_GATE_FAILED"
    return {
        "schema_version": "stage2c-context-selection-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "commit": git_head(),
        "split": "dev",
        "dev_questions": len(gold_rows),
        "dev_answerable": sum(1 for row in gold_rows if row.get("answerable")),
        "test_questions_evaluated": 0,
        "test_protocol_violation": False,
        "context_trace_source": "RECONSTRUCTED_DETERMINISTIC",
        "context_selection_hypothesis_supported": hypothesis,
        "offline_metrics": metrics,
        "offline_gate": {
            "required_claim_context_coverage_gain_min": 0.05,
            "full_context_coverage_gain_min": 0.05,
            "token_p95_max_baseline_multiplier": 1.10,
            "results": gate_results,
        },
        "offline_selected_candidate": selected,
        "stage2c_optimization_status": status,
        "generation_experiment_run": False,
        "selected_stage2c_candidate": selected if status.startswith("GENERATION") else "BASELINE",
        "remaining_bottleneck": (
            "GENERATION_UTILIZATION_OR_CITATION"
            if not hypothesis
            else "CONTEXT_SELECTION_OFFLINE_UNRESOLVED"
        ),
        "production_defaults_changed": False,
        "reranker_enabled": False,
        "query_rewrite_enabled": False,
    }


if __name__ == "__main__":
    raise SystemExit(main())
