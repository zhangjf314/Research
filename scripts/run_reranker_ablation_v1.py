# ruff: noqa: E501
"""Run Stage 11B reranker-only ablations over one shared Top-30 retrieval snapshot."""

import csv
import hashlib
import json
import math
import time
from datetime import UTC, datetime
from pathlib import Path

from qdrant_client import QdrantClient

from paper_research.config import Settings
from paper_research.indexing.vector_store import QdrantVectorStore
from paper_research.providers.factory import build_embedding_provider, build_reranker
from paper_research.retrieval.dense import DenseRetriever
from paper_research.retrieval.filters import RetrievalFilter
from paper_research.retrieval.fusion import FusedResult, reciprocal_rank_fusion
from paper_research.retrieval.reranker import (
    DisabledReranker,
    LexicalReranker,
    Reranker,
    RerankerProviderError,
    RerankOutcome,
)
from paper_research.retrieval.sparse import BM25Retriever

try:
    import scripts.run_retrieval_ablation_v2 as v2
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    import run_retrieval_ablation_v2 as v2  # type: ignore[no-redef]

PROTOCOL = Path("data/evaluation/retrieval-gold-v2.jsonl")
CORPUS = Path("data/evaluation/production-corpus-v1.json")
INDEX_MANIFEST = Path("data/evaluation/retrieval-index-v2.json")
JSON_OUTPUT = Path("data/evaluation/reranker-ablation-v1.json")
CSV_OUTPUT = Path("data/evaluation/reranker-ablation-v1.csv")
REPORT_OUTPUT = Path("docs/reranker-ablation-v1.md")
INPUT_K = 30
RERANK_OUTPUT_K = 30
EVALUATION_K = 10
SEED = 42
DEMO_P95_LIMIT_MS = 3000.0


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def latency_summary(values: list[float]) -> dict:
    return {
        "mean_ms": round(mean(values), 3),
        "p50_ms": round(percentile(values, 0.50), 3),
        "p95_ms": round(percentile(values, 0.95), 3),
    }


def initial_signature(results: list[FusedResult]) -> str:
    digest = hashlib.sha256()
    for rank, result in enumerate(results, start=1):
        digest.update(
            f"{rank}|{result.chunk.chunk_id}|{result.score:.12f}|{result.dense_rank}|{result.sparse_rank}\n".encode()
        )
    return digest.hexdigest()


def retrieve_initial_candidates(
    *,
    protocol: list[dict],
    dense: DenseRetriever,
    sparse: BM25Retriever,
    public_to_raw: dict[str, str],
    production_raw_ids: list[str],
) -> list[dict]:
    snapshots = []
    for record in protocol:
        public_filter = record["retrieval_filter"]["paper_ids"]
        raw_filter = (
            production_raw_ids
            if record["retrieval_scope"] == "global"
            else [public_to_raw[paper_id] for paper_id in public_filter]
        )
        retrieval_filter = RetrievalFilter(paper_ids=raw_filter)
        started = time.perf_counter()
        error = None
        try:
            dense_results = dense.retrieve(
                record["retrieval_query"],
                retrieval_filter=retrieval_filter,
                top_k=INPUT_K,
            )
            sparse_results = sparse.retrieve(
                record["retrieval_query"],
                retrieval_filter=retrieval_filter,
                top_k=INPUT_K,
            )
            fused = reciprocal_rank_fusion(dense_results, sparse_results)[:INPUT_K]
        except Exception as exc:
            dense_results = []
            sparse_results = []
            fused = []
            error = f"{type(exc).__name__}: {exc}"
        snapshots.append(
            {
                "record": record,
                "filter_database_ids": raw_filter,
                "dense_count": len(dense_results),
                "sparse_count": len(sparse_results),
                "candidates": fused,
                "candidate_signature": initial_signature(fused),
                "retrieval_latency_ms": round((time.perf_counter() - started) * 1000, 3),
                "failure_reason": error,
            }
        )
    return snapshots


def ranking_trace(
    pre: list[FusedResult], post: list[FusedResult], raw_to_public: dict[str, str]
) -> list[dict]:
    post_map = {
        result.chunk.chunk_id: (rank, result.score)
        for rank, result in enumerate(post, start=1)
    }
    return [
        {
            "chunk_id": result.chunk.chunk_id,
            "paper_id": raw_to_public[result.chunk.paper_id],
            "database_id": result.chunk.paper_id,
            "pre_rerank_rank": rank,
            "pre_rerank_score": round(float(result.score), 9),
            "rerank_score": (
                round(float(post_map[result.chunk.chunk_id][1]), 9)
                if result.chunk.chunk_id in post_map
                else None
            ),
            "post_rerank_rank": (
                post_map[result.chunk.chunk_id][0]
                if result.chunk.chunk_id in post_map
                else None
            ),
            "dense_rank": result.dense_rank,
            "sparse_rank": result.sparse_rank,
        }
        for rank, result in enumerate(pre, start=1)
    ]


def apply_variant(
    *,
    name: str,
    reranker: Reranker,
    snapshots: list[dict],
    raw_to_public: dict[str, str],
) -> dict:
    queries = []
    rerank_latencies = []
    total_latencies = []
    failure_count = 0
    fallback_count = 0
    api_request_count = 0
    for snapshot in snapshots:
        record = snapshot["record"]
        candidates: list[FusedResult] = snapshot["candidates"]
        error = snapshot["failure_reason"]
        if error:
            outcome = RerankOutcome(
                [], reranker.provider_name, reranker.model_name, 0, 0, 0.0,
                failure_reason=error,
            )
            failure_count += 1
        else:
            try:
                if reranker.provider_name == "disabled":
                    outcome = RerankOutcome(
                        candidates,
                        "disabled",
                        "none",
                        len(candidates),
                        len(candidates),
                        0.0,
                    )
                else:
                    outcome = reranker.rerank_with_trace(
                        record["retrieval_query"], candidates, len(candidates)
                    )
            except RerankerProviderError as exc:
                outcome = RerankOutcome(
                    [],
                    reranker.provider_name,
                    reranker.model_name,
                    len(candidates),
                    0,
                    0.0,
                    failure_reason=str(exc),
                    api_request_count=exc.api_request_count,
                )
                failure_count += 1
        fallback_count += int(outcome.fallback_occurred)
        api_request_count += outcome.api_request_count
        rerank_latencies.append(outcome.latency_ms)
        total_latency = snapshot["retrieval_latency_ms"] + outcome.latency_ms
        total_latencies.append(total_latency)
        post_full = outcome.results
        pre_rows = v2.ranked_rows(candidates[:EVALUATION_K], record, raw_to_public)
        post_rows = v2.ranked_rows(post_full[:EVALUATION_K], record, raw_to_public)
        queries.append(
            {
                **record,
                "applied_filter_database_ids": snapshot["filter_database_ids"],
                "initial_candidate_signature": snapshot["candidate_signature"],
                "pre_rerank_candidate_count": len(candidates),
                "pre_rerank_ranked_results": pre_rows,
                "ranked_results": post_rows,
                "ranking_changes": ranking_trace(candidates, post_full, raw_to_public),
                "rerank_provider": outcome.provider,
                "rerank_model": outcome.model,
                "rerank_output_count": outcome.output_count,
                "retrieval_latency_ms": snapshot["retrieval_latency_ms"],
                "rerank_latency_ms": outcome.latency_ms,
                "total_latency_ms": round(total_latency, 3),
                "fallback_occurred": outcome.fallback_occurred,
                "failure_reason": outcome.failure_reason,
                "api_request_count": outcome.api_request_count,
            }
        )
    post_by_scope = {
        scope: [query for query in queries if query["retrieval_scope"] == scope]
        for scope in ("global", "paper", "multi_paper", "unanswerable")
    }
    pre_queries = [
        {**query, "ranked_results": query["pre_rerank_ranked_results"]}
        for query in queries
    ]
    pre_by_scope = {
        scope: [query for query in pre_queries if query["retrieval_scope"] == scope]
        for scope in ("global", "paper", "multi_paper", "unanswerable")
    }
    retrieval_latencies = [snapshot["retrieval_latency_ms"] for snapshot in snapshots]
    return {
        "name": name,
        "configuration": {
            "embedding_provider": "jina",
            "embedding_model": "jina-embeddings-v5-text-small",
            "retriever": "structural_hybrid",
            "rerank_provider": reranker.provider_name,
            "rerank_model": reranker.model_name,
            "rerank_input_k": INPUT_K,
            "rerank_output_k": RERANK_OUTPUT_K,
            "evaluation_k": EVALUATION_K,
            "fallback_allowed": False,
            "llm_called": False,
            "deep_research_called": False,
        },
        "metrics": {
            "global": v2.summarize_global(post_by_scope["global"]),
            "paper": v2.block_scope_metrics(post_by_scope["paper"]),
            "multi_paper": v2.summarize_multi(post_by_scope["multi_paper"]),
            "unanswerable": v2.summarize_unanswerable(post_by_scope["unanswerable"]),
            "pre_rerank": {
                "paper": v2.block_scope_metrics(pre_by_scope["paper"]),
                "multi_paper": v2.summarize_multi(pre_by_scope["multi_paper"]),
            },
            "latency": {
                "retrieval": latency_summary(retrieval_latencies),
                "rerank": latency_summary(rerank_latencies),
                "total": latency_summary(total_latencies),
            },
            "pre_rerank_recall_at_10": v2.block_scope_metrics(pre_by_scope["paper"]).get(
                "block_recall_at_10"
            ),
            "post_rerank_recall_at_10": v2.block_scope_metrics(
                post_by_scope["paper"]
            ).get("block_recall_at_10"),
            "failure_count": failure_count,
            "fallback_count": fallback_count,
            "api_request_count": api_request_count,
        },
        "by_category": v2.grouped_evidence(queries, "category"),
        "by_difficulty": v2.grouped_evidence(queries, "difficulty"),
        "queries": queries,
    }


def first_gold_rank(query: dict) -> int:
    return next(
        (row["rank"] for row in query["ranked_results"] if row["gold_block_hit"]),
        EVALUATION_K + 1,
    )


def comparison_counts(baseline: dict, candidate: dict) -> dict:
    changes = []
    for before, after in zip(baseline["queries"], candidate["queries"], strict=True):
        if before["retrieval_scope"] == "unanswerable":
            continue
        old_rank = first_gold_rank(before)
        new_rank = first_gold_rank(after)
        changes.append(
            {
                "question_id": before["question_id"],
                "question": before["retrieval_query"],
                "before_rank": old_rank,
                "after_rank": new_rank,
                "delta": old_rank - new_rank,
            }
        )
    return {
        "improved_count": sum(row["delta"] > 0 for row in changes),
        "regressed_count": sum(row["delta"] < 0 for row in changes),
        "unchanged_count": sum(row["delta"] == 0 for row in changes),
        "improvements": sorted(changes, key=lambda row: row["delta"], reverse=True),
        "regressions": sorted(changes, key=lambda row: row["delta"]),
    }


def write_csv(variants: list[dict]) -> None:
    rows = []
    for variant in variants:
        for scope in ("paper", "multi_paper", "unanswerable"):
            rows.append(
                {
                    "variant": variant["name"],
                    "breakdown": "scope",
                    "group": scope,
                    **variant["metrics"][scope],
                }
            )
        for phase in ("retrieval", "rerank", "total"):
            rows.append(
                {
                    "variant": variant["name"],
                    "breakdown": "latency",
                    "group": phase,
                    **variant["metrics"]["latency"][phase],
                }
            )
        rows.append(
            {
                "variant": variant["name"],
                "breakdown": "operations",
                "group": "counts",
                "failure_count": variant["metrics"]["failure_count"],
                "fallback_count": variant["metrics"]["fallback_count"],
                "api_request_count": variant["metrics"]["api_request_count"],
                "pre_rerank_recall_at_10": variant["metrics"]["pre_rerank_recall_at_10"],
                "post_rerank_recall_at_10": variant["metrics"]["post_rerank_recall_at_10"],
            }
        )
        for breakdown in ("by_category", "by_difficulty"):
            for group, metrics in variant[breakdown].items():
                rows.append(
                    {
                        "variant": variant["name"],
                        "breakdown": breakdown.removeprefix("by_"),
                        "group": group,
                        **metrics,
                    }
                )
    fieldnames = ["variant", "breakdown", "group"]
    fieldnames.extend(sorted({key for row in rows for key in row} - set(fieldnames)))
    with CSV_OUTPUT.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: object) -> str:
    return "N/A" if value is None else f"{value:.3f}" if isinstance(value, float) else str(value)


def write_report(payload: dict) -> None:
    lines = [
        "# Reranker Ablation v1",
        "",
        "- Fixed retrieval: Jina Embedding + Structural Hybrid",
        "- Shared initial candidates: Top-30",
        "- Rerank output retained in Trace: Top-30",
        "- Evaluation cutoff: Top-10",
        "- Corpus: production-corpus-v1, 34 documents, 2062 points",
        "- Protocol: retrieval-gold-v2, pending review 0",
        "- Formal fallback: disabled",
        "- LLM calls: none",
        "- Deep Research: not run",
        "",
        "## Paper-scoped metrics",
        "",
        "| Variant | Hit@1 | Hit@5 | Recall@5 | Recall@10 | MRR | NDCG@10 | Pre Recall@10 | Post Recall@10 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in payload["variants"]:
        paper = variant["metrics"]["paper"]
        lines.append(
            f"| {variant['name']} | {fmt(paper.get('block_hit_at_1'))} | {fmt(paper.get('block_hit_at_5'))} | {fmt(paper.get('block_recall_at_5'))} | {fmt(paper.get('block_recall_at_10'))} | {fmt(paper.get('mrr'))} | {fmt(paper.get('ndcg_at_10'))} | {fmt(variant['metrics']['pre_rerank_recall_at_10'])} | {fmt(variant['metrics']['post_rerank_recall_at_10'])} |"
        )
    lines.extend(
        [
            "",
            "## Multi-paper metrics",
            "",
            "| Variant | Coverage@5 | All@5 | Evidence Recall@5 | Coverage@10 | All@10 | Evidence Recall@10 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for variant in payload["variants"]:
        multi = variant["metrics"]["multi_paper"]
        lines.append(
            f"| {variant['name']} | {fmt(multi.get('target_paper_coverage_at_5'))} | {fmt(multi.get('all_target_papers_at_5'))} | {fmt(multi.get('evidence_recall_at_5'))} | {fmt(multi.get('target_paper_coverage_at_10'))} | {fmt(multi.get('all_target_papers_at_10'))} | {fmt(multi.get('evidence_recall_at_10'))} |"
        )
    lines.extend(
        [
            "",
            "## Latency and reliability",
            "",
            "| Variant | Retrieval mean/p50/p95 ms | Rerank mean/p50/p95 ms | Total mean/p50/p95 ms | Failures | Fallbacks | API requests |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for variant in payload["variants"]:
        metrics = variant["metrics"]
        retrieval = metrics["latency"]["retrieval"]
        rerank = metrics["latency"]["rerank"]
        total = metrics["latency"]["total"]
        lines.append(
            f"| {variant['name']} | {retrieval['mean_ms']:.1f}/{retrieval['p50_ms']:.1f}/{retrieval['p95_ms']:.1f} | {rerank['mean_ms']:.1f}/{rerank['p50_ms']:.1f}/{rerank['p95_ms']:.1f} | {total['mean_ms']:.1f}/{total['p50_ms']:.1f}/{total['p95_ms']:.1f} | {metrics['failure_count']} | {metrics['fallback_count']} | {metrics['api_request_count']} |"
        )

    for breakdown, title in (("by_category", "Category"), ("by_difficulty", "Difficulty")):
        lines.extend(
            [
                "",
                f"## {title} breakdown",
                "",
                f"| Variant | {title} | Answerable | Hit@5 | Recall@10 | MRR |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for variant in payload["variants"]:
            for group, metrics in variant[breakdown].items():
                lines.append(
                    f"| {variant['name']} | {group} | {metrics['answerable_query_count']} | {fmt(metrics.get('block_hit_at_5'))} | {fmt(metrics.get('block_recall_at_10'))} | {fmt(metrics.get('mrr'))} |"
                )

    comparison = payload["jina_vs_no_rerank"]
    lines.extend(
        [
            "",
            "## Jina ranking changes",
            "",
            f"- Improved queries: {comparison['improved_count']}",
            f"- Regressed queries: {comparison['regressed_count']}",
            f"- Unchanged queries: {comparison['unchanged_count']}",
            "",
            "### Improvements",
            "",
        ]
    )
    for row in [item for item in comparison["improvements"] if item["delta"] > 0][:5]:
        before = ">10" if row["before_rank"] > EVALUATION_K else row["before_rank"]
        after = ">10" if row["after_rank"] > EVALUATION_K else row["after_rank"]
        lines.append(f"- `{row['question_id']}`: {before} → {after} — {row['question']}")
    lines.extend(["", "### Regressions", ""])
    for row in [item for item in comparison["regressions"] if item["delta"] < 0][:5]:
        before = ">10" if row["before_rank"] > EVALUATION_K else row["before_rank"]
        after = ">10" if row["after_rank"] > EVALUATION_K else row["after_rank"]
        lines.append(f"- `{row['question_id']}`: {before} → {after} — {row['question']}")
    lines.extend(
        [
            "",
            "Every query in the JSON artifact retains all initial Top-30 candidates with pre-rerank rank/score, rerank score, post-rerank rank, provider/model, latency, fallback, failure reason, and API request count.",
            "",
            "## Decision",
            "",
            payload["decision"]["summary"],
            "",
            "The decision compares Jina against no-rerank on paper-scoped Hit@1/MRR, NDCG@10, Recall@10, total P95, failures, fallbacks, and the number of improved queries. A disabled recommendation means `RERANK_ENABLED=false` remains the correct production setting; it is not an engineering failure.",
            "",
        ]
    )
    REPORT_OUTPUT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    settings = Settings()
    if settings.app_profile != "production":
        raise RuntimeError("APP_PROFILE=production is required")
    if settings.embedding_provider != "jina" or settings.embedding_dimensions != 1024:
        raise RuntimeError("the fixed Stage 11A.5 Jina embedding configuration is required")
    if not settings.rerank_enabled or settings.rerank_provider != "jina":
        raise RuntimeError("RERANK_ENABLED=true and RERANK_PROVIDER=jina are required")
    if settings.rerank_allow_fallback:
        raise RuntimeError("formal reranker ablation requires RERANK_ALLOW_FALLBACK=false")
    if settings.rerank_input_k != INPUT_K or settings.rerank_output_k != RERANK_OUTPUT_K:
        raise RuntimeError("formal reranker ablation requires input/output K = 30")
    if settings.llm_provider != "template":
        raise RuntimeError("LLM_PROVIDER=template is required")

    protocol = v2.load_jsonl(PROTOCOL)
    if any(row["query_revision_review_status"] == "pending_human_review" for row in protocol):
        raise RuntimeError("retrieval protocol still contains pending query reviews")
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    index_manifest = json.loads(INDEX_MANIFEST.read_text(encoding="utf-8"))
    v2.validate_inputs(protocol, corpus, index_manifest)
    collection = index_manifest["collections"]["jina"]
    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        check_compatibility=False,
    )
    chunks = v2.load_chunks(client, collection["name"])
    if v2.chunk_signature(chunks) != collection["chunk_signature"]:
        raise RuntimeError("live Jina evaluation collection chunk signature changed")
    included = [paper for paper in corpus["papers"] if paper["included_in_production"]]
    public_to_raw = {paper["paper_id"]: paper["database_id"] for paper in included}
    raw_to_public = {paper["database_id"]: paper["paper_id"] for paper in included}
    embedding = build_embedding_provider(settings)
    dense = DenseRetriever(
        embedding,
        QdrantVectorStore(client, collection["name"], embedding.dimensions),
    )
    snapshots = retrieve_initial_candidates(
        protocol=protocol,
        dense=dense,
        sparse=BM25Retriever(chunks),
        public_to_raw=public_to_raw,
        production_raw_ids=list(raw_to_public),
    )
    rerankers = {
        "no_rerank": DisabledReranker(),
        "lexical_rerank": LexicalReranker(),
        "jina_reranker_v3": build_reranker(settings),
    }
    variants = [
        apply_variant(
            name=name,
            reranker=reranker,
            snapshots=snapshots,
            raw_to_public=raw_to_public,
        )
        for name, reranker in rerankers.items()
    ]
    by_name = {variant["name"]: variant for variant in variants}
    comparison = comparison_counts(by_name["no_rerank"], by_name["jina_reranker_v3"])
    baseline = by_name["no_rerank"]["metrics"]["paper"]
    jina = by_name["jina_reranker_v3"]["metrics"]
    jina_paper = jina["paper"]
    conditions = {
        "hit1_or_mrr_improved": (
            jina_paper["block_hit_at_1"] > baseline["block_hit_at_1"]
            or jina_paper["mrr"] > baseline["mrr"]
        ),
        "ndcg_not_materially_lower": jina_paper["ndcg_at_10"] >= baseline["ndcg_at_10"] - 0.01,
        "recall10_not_unacceptably_lower": jina_paper["block_recall_at_10"] >= baseline["block_recall_at_10"] - 0.02,
        "p95_within_demo_limit": jina["latency"]["total"]["p95_ms"] <= DEMO_P95_LIMIT_MS,
        "failure_zero": jina["failure_count"] == 0,
        "fallback_zero": jina["fallback_count"] == 0,
        "improvement_not_tiny_subset": comparison["improved_count"] >= 5,
    }
    recommend = all(conditions.values())
    decision = {
        "recommend_rerank_enabled": recommend,
        "conditions": conditions,
        "summary": (
            "Recommend `RERANK_ENABLED=true` for the evaluated Jina v3 configuration."
            if recommend
            else "Keep `RERANK_ENABLED=false`; the evaluated Jina v3 configuration did not satisfy every production acceptance condition."
        ),
    }
    payload = {
        "status": "COMPLETED",
        "generated_at": datetime.now(UTC).isoformat(),
        "model": "jina-reranker-v3",
        "corpus_version": corpus["manifest_version"],
        "protocol_version": "retrieval-query-v2",
        "collection": collection,
        "initial_retrieval_executions": len(snapshots),
        "shared_initial_candidates": True,
        "random_seed": SEED,
        "llm_called": False,
        "deep_research_called": False,
        "variants": variants,
        "jina_vs_no_rerank": comparison,
        "decision": decision,
    }
    JSON_OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(variants)
    write_report(payload)
    print(
        json.dumps(
            {
                "status": "COMPLETED",
                "variants": list(rerankers),
                "recommend_rerank_enabled": recommend,
            }
        )
    )


if __name__ == "__main__":
    main()
