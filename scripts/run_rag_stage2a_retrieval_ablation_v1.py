from __future__ import annotations

import json
import subprocess
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from qdrant_client import QdrantClient

from paper_research.chunking.types import Chunk
from paper_research.config import get_settings
from paper_research.evaluation.rag_stage2a import (
    OPT_DOCS,
    OPT_ROOT,
    RetrievalExperimentConfig,
    RetrievalExperimentPlan,
    as_ranked_from_fused,
    bad_case_delta,
    bootstrap_delta_ci,
    category_metrics,
    complementarity,
    difficulty_metrics,
    evaluate_ranked_candidates,
    is_success_candidate,
    load_baseline_generation_rows,
    load_baseline_retrieval_dev_rows,
    load_dev_gold,
    metrics_summary,
    paired_comparison,
    summarize_context_selection_from_generation,
    write_json,
    write_jsonl,
)
from paper_research.indexing.registry import IndexRegistry
from paper_research.retrieval.fusion import FusedResult, reciprocal_rank_fusion
from paper_research.retrieval.reranker import LexicalReranker

PLAN = OPT_ROOT / "hybrid-experiment-plan-v1.json"


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def paper_id_map() -> dict[str, str]:
    manifest = json.loads(
        Path("data/evaluation/production-corpus-v1.json").read_text(encoding="utf-8")
    )
    mapping = {}
    for paper in manifest.get("papers", []):
        if paper.get("included_in_production"):
            mapping[str(paper.get("database_id"))] = str(paper.get("paper_id"))
    return mapping


def load_plan() -> RetrievalExperimentPlan:
    if not PLAN.exists():
        raise FileNotFoundError(f"missing preregistered experiment plan: {PLAN}")
    return RetrievalExperimentPlan.model_validate_json(PLAN.read_text(encoding="utf-8"))


def run_experiment(
    config: RetrievalExperimentConfig,
    *,
    gold_rows: list[dict[str, Any]],
    traces: dict[str, dict[str, Any]],
    chunks: dict[str, Chunk],
    id_map: dict[str, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    reranker = LexicalReranker() if config.reranker == "lexical" else None
    rerank_latencies = []
    for gold in gold_rows:
        trace = traces[gold["question_id"]]
        if trace.get("failure_reason"):
            row = evaluate_ranked_candidates(
                gold,
                [],
                latency_ms=float(trace.get("latency_ms") or 0.0),
                paper_id_map=id_map,
            )
            row["retrieval_failure"] = trace["failure_reason"]
            row["experiment_id"] = config.experiment_id
            row["retrieval_mode"] = config.mode
            row["reranker"] = config.reranker
            rows.append(row)
            continue
        trace_latency_ms = float(trace.get("latency_ms") or 0.0)
        dense_results = _trace_to_fused(trace.get("dense_results", []), chunks, "dense")
        sparse_results = _trace_to_fused(trace.get("sparse_results", []), chunks, "sparse")
        rerank_latency_ms = 0.0
        if config.mode == "dense":
            ranked = as_ranked_from_fused(dense_results)
        elif config.mode == "sparse":
            ranked = as_ranked_from_fused(sparse_results)
        else:
            fused = reciprocal_rank_fusion(
                dense_results,
                sparse_results,
                k=config.rrf_k,
                dense_weight=config.dense_weight,
                lexical_weight=config.sparse_weight,
            )
            if reranker:
                input_k = config.reranker_candidate_k or config.recall_k
                output_k = config.reranker_output_k or config.final_k
                rerank_started = time.perf_counter()
                fused = reranker.rerank(gold["question"], fused[:input_k], output_k)
                rerank_latency_ms = round((time.perf_counter() - rerank_started) * 1000, 3)
                rerank_latencies.append(rerank_latency_ms)
            ranked = as_ranked_from_fused(fused)
        latency_ms = round(trace_latency_ms + rerank_latency_ms, 3)
        row = evaluate_ranked_candidates(
            gold,
            ranked[: config.final_k],
            latency_ms=latency_ms,
            paper_id_map=id_map,
        )
        row["experiment_id"] = config.experiment_id
        row["retrieval_mode"] = config.mode
        row["reranker"] = config.reranker
        rows.append(row)
    metadata = {
        "reranker_latency_ms": {
            "p50": _percentile(rerank_latencies, 0.5),
            "p95": _percentile(rerank_latencies, 0.95),
        },
        "reranker_request_count": len(rerank_latencies),
        "latency_source": (
            "production /retrieve trace latency reused for DEV-only ablation; "
            "lexical rerank latency is added for RR1"
        ),
    }
    return rows, metadata


def _trace_to_fused(
    trace_items: list[dict[str, Any]], chunks: dict[str, Chunk], source: str
) -> list[FusedResult]:
    results = []
    for rank, item in enumerate(trace_items, start=1):
        chunk_id = str(item["chunk_id"])
        chunk = chunks[chunk_id]
        results.append(
            FusedResult(
                chunk=chunk,
                score=float(item.get("score") or 0.0),
                dense_rank=rank if source == "dense" else item.get("dense_rank"),
                sparse_rank=rank if source == "sparse" else item.get("sparse_rank"),
            )
        )
    return results


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return round(ordered[index], 3)


def main() -> int:
    plan = load_plan()
    gold_rows = load_dev_gold()
    if len(gold_rows) != 98 or any(row.get("split") != "dev" for row in gold_rows):
        raise RuntimeError("TEST_PROTOCOL_VIOLATION")
    settings = get_settings()
    qdrant = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    collection = IndexRegistry(settings.data_dir / "index_registry.json").resolve(
        settings.active_collection
    )
    traces = collect_retrieval_traces(gold_rows)
    chunks = fetch_trace_chunks(qdrant, collection, traces)
    id_map = paper_id_map()
    baseline_rows = load_baseline_retrieval_dev_rows()
    experiment_rows: dict[str, list[dict[str, Any]]] = {
        "R0_frozen_baseline": baseline_rows
    }
    experiment_meta: dict[str, dict[str, Any]] = {"R0_frozen_baseline": {}}
    for config in plan.experiments:
        rows, metadata = run_experiment(
            config, gold_rows=gold_rows, traces=traces, chunks=chunks, id_map=id_map
        )
        experiment_rows[config.experiment_id] = rows
        experiment_meta[config.experiment_id] = metadata
        write_jsonl(OPT_ROOT / f"{config.experiment_id}-items-v1.jsonl", rows)
    metrics = {
        experiment_id: metrics_summary(rows)
        for experiment_id, rows in experiment_rows.items()
    }
    for experiment_id, value in metrics.items():
        value.update(experiment_meta[experiment_id])
    dense_rows = experiment_rows["R1_dense_only"]
    sparse_rows = experiment_rows["R2_sparse_only"]
    hybrid_rows = experiment_rows["R3_current_hybrid"]
    selected_id = max(
        metrics,
        key=lambda exp_id: (
            metrics[exp_id].get("evidence_coverage_at_10") or 0,
            metrics[exp_id].get("recall_at_10") or 0,
            metrics[exp_id].get("mrr_at_10") or 0,
        ),
    )
    baseline_metric = metrics_summary(baseline_rows)
    selected_metric = metrics[selected_id]
    paired = paired_comparison(baseline_rows, experiment_rows[selected_id])
    bootstrap = {
        metric: bootstrap_delta_ci(baseline_rows, experiment_rows[selected_id], metric=metric)
        for metric in ("recall_at_10", "mrr_at_10", "evidence_coverage_at_10")
    }
    payload = {
        "schema_version": "stage2a-retrieval-ablation-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "benchmark_harness_commit": git_head(),
        "system_under_test_commit": "f97746e84b98d6b4e07984a3abbdab206f156839",
        "dataset_version": plan.dataset_version,
        "dataset_hash": plan.dataset_hash,
        "split": "dev",
        "dev_question_count": len(gold_rows),
        "test_questions_evaluated": 0,
        "test_protocol_violation": False,
        "baseline_actual_retrieval": {
            "mode": "hybrid dense+sparse",
            "fusion": "reciprocal_rank_fusion",
            "rrf_k": 60,
            "dense_k": 20,
            "sparse_k": 20,
            "final_k": 20,
            "reranker": "disabled",
            "query_routing": (
                "global queries unchanged; paper-scope routing exists in production code"
            ),
            "frozen_baseline_equals_current_hybrid": True,
        },
        "plan": plan.model_dump(),
        "metrics": metrics,
        "by_category": {
            experiment_id: category_metrics(rows)
            for experiment_id, rows in experiment_rows.items()
        },
        "by_difficulty": {
            experiment_id: difficulty_metrics(rows)
            for experiment_id, rows in experiment_rows.items()
        },
        "complementarity": complementarity(dense_rows, sparse_rows, hybrid_rows),
        "paired_comparison_selected_vs_baseline": paired,
        "bootstrap_ci_selected_vs_baseline": bootstrap,
        "bad_case_delta_selected_vs_baseline": bad_case_delta(
            baseline_rows, experiment_rows[selected_id]
        ),
        "selected_stage2a_candidate": selected_id,
        "selected_candidate_meets_success_threshold": is_success_candidate(
            baseline_metric, selected_metric
        ),
        "query_rewrite_hypothesis_supported": _query_rewrite_supported(
            experiment_rows[selected_id]
        ),
        "context_selection_analysis": summarize_context_selection_from_generation(
            load_baseline_generation_rows()
        ),
        "llm_requests": 0,
    }
    write_json(OPT_ROOT / "stage2a-retrieval-ablation-v1.json", payload)
    _write_markdown(payload)
    print(
        json.dumps(
            {
                "status": "COMPLETED",
                "selected_stage2a_candidate": selected_id,
                "test_questions_evaluated": 0,
                "metrics": metrics,
            },
            ensure_ascii=False,
        )
    )
    return 0


def collect_retrieval_traces(gold_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    traces = {}
    with httpx.Client(timeout=180) as client:
        for gold in gold_rows:
            started = time.perf_counter()
            try:
                response = client.post(
                    "http://localhost/api/v1/retrieve",
                    json={
                        "query": gold["question"],
                        "filters": {},
                        "recall_k": 20,
                        "top_k": 20,
                    },
                )
                latency_ms = round((time.perf_counter() - started) * 1000, 3)
                response.raise_for_status()
                trace = response.json()["trace"]
                trace["latency_ms"] = trace.get("latency_ms") or latency_ms
                traces[gold["question_id"]] = trace
            except Exception as exc:  # noqa: BLE001 - experiment records failed query
                latency_ms = round((time.perf_counter() - started) * 1000, 3)
                traces[gold["question_id"]] = {
                    "failure_reason": f"{type(exc).__name__}: {exc}",
                    "latency_ms": latency_ms,
                    "dense_results": [],
                    "sparse_results": [],
                    "fusion_results": [],
                }
    return traces


def fetch_trace_chunks(
    qdrant: QdrantClient, collection: str, traces: dict[str, dict[str, Any]]
) -> dict[str, Chunk]:
    trace_items = {
        str(item["chunk_id"]): item
        for trace in traces.values()
        for field in ("dense_results", "sparse_results", "fusion_results")
        for item in trace.get(field, [])
    }
    chunk_ids = sorted(trace_items)
    chunks: dict[str, Chunk] = {}
    for start in range(0, len(chunk_ids), 128):
        batch = chunk_ids[start : start + 128]
        points = qdrant.retrieve(
            collection_name=collection,
            ids=[str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id)) for chunk_id in batch],
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            chunk = Chunk.model_validate(point.payload)
            chunks[chunk.chunk_id] = chunk
    missing = sorted(set(chunk_ids) - set(chunks))
    for chunk_id in missing:
        item = trace_items[chunk_id]
        chunks[chunk_id] = Chunk(
            chunk_id=chunk_id,
            paper_id=str(item.get("paper_id") or "missing-payload"),
            block_ids=[],
            section_path=["missing_qdrant_payload"],
            block_type="missing_payload",
            page_start=0,
            page_end=0,
            chunk_text="",
            token_count=0,
        )
    return chunks


def _query_rewrite_supported(rows: list[dict[str, Any]]) -> bool:
    answerable = [row for row in rows if row.get("answerable")]
    misses = sum(1 for row in answerable if (row["metrics"].get("recall_at_20") or 0) == 0)
    partial = sum(
        1
        for row in answerable
        if 0 < (row["metrics"].get("evidence_coverage_at_20") or 0) < 1
    )
    return (misses + partial) / max(1, len(answerable)) >= 0.25


def _write_markdown(payload: dict[str, Any]) -> None:
    metrics = payload["metrics"]
    lines = [
        "# Stage 2A retrieval ablation v1",
        "",
        f"- split: `{payload['split']}`",
        f"- dev_question_count: `{payload['dev_question_count']}`",
        f"- test_questions_evaluated: `{payload['test_questions_evaluated']}`",
        f"- selected_stage2a_candidate: `{payload['selected_stage2a_candidate']}`",
        f"- llm_requests: `{payload['llm_requests']}`",
        "",
        "| Config | Recall@5 | Recall@10 | Recall@20 | MRR@10 | nDCG@10 | "
        "EvidenceCov@10 | P50 | P95 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    labels = {
        "R0_frozen_baseline": "Baseline",
        "R1_dense_only": "Dense",
        "R2_sparse_only": "Sparse",
        "R3_current_hybrid": "Hybrid",
        "RR1_hybrid_lexical_rerank": "Hybrid + Rerank",
    }
    for exp_id, label in labels.items():
        metric = metrics[exp_id]
        latency = metric.get("latency_ms") or {}
        lines.append(
            f"| {label} | {metric.get('recall_at_5')} | {metric.get('recall_at_10')} | "
            f"{metric.get('recall_at_20')} | {metric.get('mrr_at_10')} | "
            f"{metric.get('ndcg_at_10')} | {metric.get('evidence_coverage_at_10')} | "
            f"{latency.get('p50')} | {latency.get('p95')} |"
        )
    lines.extend(
        [
            "",
            "## Complementarity",
            "",
            json.dumps(payload["complementarity"]["summary"], ensure_ascii=False, indent=2),
            "",
            "## Selected vs baseline",
            "",
            json.dumps(
                {
                    "paired": {
                        key: payload["paired_comparison_selected_vs_baseline"][key]
                        for key in ("win_count", "tie_count", "loss_count")
                    },
                    "bootstrap": payload["bootstrap_ci_selected_vs_baseline"],
                    "bad_case_delta": payload["bad_case_delta_selected_vs_baseline"],
                },
                ensure_ascii=False,
                indent=2,
            ),
        ]
    )
    OPT_DOCS.mkdir(parents=True, exist_ok=True)
    (OPT_DOCS / "stage2a-retrieval-ablation-v1.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    raise SystemExit(main())
