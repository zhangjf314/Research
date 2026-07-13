# ruff: noqa: E501
"""Run the six RC retrieval/answer ablations and emit reproducible artifacts."""

import csv
import json
import math
import platform
import random
import time
from datetime import UTC, datetime
from pathlib import Path

from qdrant_client import QdrantClient

from paper_research.chunking.fixed_chunker import FixedTokenChunker
from paper_research.chunking.structural_chunker import StructuralChunker
from paper_research.evaluation.answer_metrics import evaluate_answer
from paper_research.evaluation.dataset import EvaluationItem, RetrievalPrediction
from paper_research.evaluation.retrieval_metrics import evaluate_retrieval
from paper_research.indexing.embedding import HashEmbeddingProvider
from paper_research.indexing.vector_store import QdrantVectorStore
from paper_research.parsing.types import PaperBlock
from paper_research.retrieval.context_builder import ContextBuilder
from paper_research.retrieval.dense import DenseRetriever
from paper_research.retrieval.fusion import reciprocal_rank_fusion
from paper_research.retrieval.reranker import LexicalReranker
from paper_research.retrieval.sparse import BM25Retriever

SEED = 42
DATASET = Path("data/evaluation/gold-set-v1.jsonl")
JSON_OUTPUT = Path("data/evaluation/results-v1.json")
CSV_OUTPUT = Path("data/evaluation/results-v1.csv")
REPORT_OUTPUT = Path("docs/evaluation-report-v1.md")
VARIANTS = (
    "fixed_chunk_dense",
    "structural_chunk_dense",
    "structural_chunk_sparse",
    "structural_chunk_hybrid",
    "structural_chunk_hybrid_rerank",
    "hybrid_rerank_neighbor_context",
)


def load_corpus() -> tuple[list, list]:
    structural, fixed = [], []
    for path in sorted(Path("data/reports/parsing-audit").glob("*/paper_blocks.jsonl")):
        blocks = [
            PaperBlock.model_validate_json(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        structural.extend(StructuralChunker().chunk(path.parent.name, blocks))
        fixed.extend(FixedTokenChunker().chunk(path.parent.name, blocks))
    return structural, fixed


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)] if ordered else 0.0


def main() -> None:
    random.seed(SEED)
    raw_items = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines()]
    if len(raw_items) != 50:
        raise RuntimeError(f"expected 50 evaluation items, got {len(raw_items)}")
    answerable_raw = [item for item in raw_items if item["answerable"]]
    eval_items = [
        EvaluationItem(
            id=item["question_id"],
            question=item["question"],
            question_type=item["category"],
            relevant_paper_ids=item["gold_paper_ids"],
            relevant_block_ids=item["gold_block_ids"],
            relevant_pages=item["gold_pages"],
            expected_answer_points=item["required_claims"],
            annotation_status="silver",
            notes="Pending human review; RC metrics are provisional.",
        )
        for item in answerable_raw
    ]
    structural, fixed = load_corpus()
    embedding = HashEmbeddingProvider(384)
    structural_store = QdrantVectorStore(QdrantClient(":memory:"), "structural-v1", 384)
    fixed_store = QdrantVectorStore(QdrantClient(":memory:"), "fixed-v1", 384)
    structural_store.upsert(structural, embedding.embed([chunk.chunk_text for chunk in structural]))
    fixed_store.upsert(fixed, embedding.embed([chunk.chunk_text for chunk in fixed]))
    dense = DenseRetriever(embedding, structural_store)
    dense_fixed = DenseRetriever(embedding, fixed_store)
    sparse = BM25Retriever(structural)
    reranker = LexicalReranker()
    predictions = {name: [] for name in VARIANTS}
    answer_scores = {name: [] for name in VARIANTS}
    latencies = {name: [] for name in VARIANTS}
    no_answer_correct = {name: 0 for name in VARIANTS}

    for item in raw_items:
        query = item["question"]
        started = time.perf_counter()
        fixed_results = dense_fixed.retrieve(query, top_k=10)
        latencies[VARIANTS[0]].append((time.perf_counter() - started) * 1000)
        started = time.perf_counter()
        dense_results = dense.retrieve(query, top_k=20)
        dense_ms = (time.perf_counter() - started) * 1000
        latencies[VARIANTS[1]].append(dense_ms)
        started = time.perf_counter()
        sparse_results = sparse.retrieve(query, top_k=20)
        sparse_ms = (time.perf_counter() - started) * 1000
        latencies[VARIANTS[2]].append(sparse_ms)
        started = time.perf_counter()
        fused = reciprocal_rank_fusion(dense_results, sparse_results)[:10]
        hybrid_ms = dense_ms + sparse_ms + (time.perf_counter() - started) * 1000
        latencies[VARIANTS[3]].append(hybrid_ms)
        started = time.perf_counter()
        reranked = reranker.rerank(query, reciprocal_rank_fusion(dense_results, sparse_results), 10)
        rerank_ms = dense_ms + sparse_ms + (time.perf_counter() - started) * 1000
        latencies[VARIANTS[4]].append(rerank_ms)
        started = time.perf_counter()
        neighbor_context = ContextBuilder(include_neighbors=True).build(reranked[:5])
        latencies[VARIANTS[5]].append(
            rerank_ms + (time.perf_counter() - started) * 1000
        )
        result_sets = {
            VARIANTS[0]: fixed_results,
            VARIANTS[1]: dense_results[:10],
            VARIANTS[2]: sparse_results[:10],
            VARIANTS[3]: fused,
            VARIANTS[4]: reranked,
            VARIANTS[5]: reranked,
        }
        for name, results in result_sets.items():
            predicted_refusal = not results
            no_answer_correct[name] += int(predicted_refusal == (not item["answerable"]))
            if item["answerable"]:
                predictions[name].append(
                    RetrievalPrediction(
                        item_id=item["question_id"],
                        ranked_paper_ids=[result.chunk.paper_id for result in results],
                        ranked_block_ids=[
                            block_id for result in results for block_id in result.chunk.block_ids
                        ],
                    )
                )
                contexts = (
                    [context.evidence for context in neighbor_context]
                    if name == VARIANTS[5]
                    else [result.chunk.chunk_text for result in results[:5]]
                )
                answer = ""
                if results:
                    top = results[0].chunk
                    answer = f"{contexts[0]} [{top.paper_id}, p.{top.page_start}]"
                citations = [
                    {
                        "paper_id": result.chunk.paper_id,
                        "page": result.chunk.page_start,
                        "valid": result.chunk.paper_id in item["gold_paper_ids"]
                        and (
                            not item["gold_pages"]
                            or result.chunk.page_start in item["gold_pages"]
                        ),
                    }
                    for result in results[:5]
                ]
                answer_scores[name].append(
                    evaluate_answer(answer, contexts, citations, item["required_claims"])
                )

    results = {}
    for name in VARIANTS:
        retrieval = evaluate_retrieval(eval_items, predictions[name], k_values=(1, 5, 10))
        answers = answer_scores[name]
        answer = {
            key: round(sum(score[key] for score in answers) / len(answers), 6)
            for key in answers[0]
        }
        results[name] = {
            "hit_at_1": retrieval["hit_at_1"],
            "hit_at_5": retrieval["hit_at_5"],
            "recall_at_5": retrieval["recall_at_5"],
            "recall_at_10": retrieval["recall_at_10"],
            "mrr": retrieval["mrr"],
            "ndcg_at_10": retrieval["ndcg_at_10"],
            "faithfulness": answer["faithfulness"],
            "citation_coverage": answer["citation_coverage"],
            "citation_correctness": answer["citation_correctness"],
            "unsupported_claim_rate": answer["unsupported_claim_rate"],
            "unanswerable_accuracy": round(no_answer_correct[name] / len(raw_items), 6),
            "mean_latency_ms": round(sum(latencies[name]) / len(latencies[name]), 3),
            "p95_latency_ms": round(percentile(latencies[name], 0.95), 3),
        }
    payload = {
        "run": {
            "started_at": datetime.now(UTC).isoformat(),
            "dataset": str(DATASET),
            "dataset_version": "gold-set-v1-pending-review",
            "items": len(raw_items),
            "answerable_items": len(answerable_raw),
            "human_approved_items": sum(item["review_status"] == "approved" for item in raw_items),
            "random_seed": SEED,
            "embedding_model": "HashEmbeddingProvider(dimensions=384)",
            "reranker_model": "LexicalReranker",
            "answer_model": "extractive-top-context baseline",
            "python": platform.python_version(),
            "configuration": {"dense_recall_k": 20, "sparse_recall_k": 20, "top_k": 10},
            "quality_status": "provisional: all 50 annotations remain pending human review",
        },
        "results": results,
    }
    JSON_OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with CSV_OUTPUT.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["variant", *next(iter(results.values())).keys()])
        writer.writeheader()
        for variant, metrics in results.items():
            writer.writerow({"variant": variant, **metrics})
    write_report(payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def write_report(payload: dict) -> None:
    run = payload["run"]
    lines = [
        "# Evaluation Report v1",
        "",
        "> Provisional RC evidence only: 0/50 items have been approved by a human reviewer.",
        "",
        f"- Run time: `{run['started_at']}`",
        f"- Dataset: `{run['dataset_version']}` ({run['items']} items, {run['answerable_items']} answerable)",
        f"- Seed: `{run['random_seed']}`",
        f"- Embedding: `{run['embedding_model']}`",
        f"- Reranker: `{run['reranker_model']}`",
        f"- Answer provider: `{run['answer_model']}`",
        "",
        "| Variant | Hit@1 | Hit@5 | Recall@5 | Recall@10 | MRR | NDCG@10 | Faithfulness | Citation Coverage | Citation Correctness | Unsupported Claim Rate | No-answer Accuracy | Mean ms | P95 ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant, metrics in payload["results"].items():
        lines.append(
            f"| {variant} | {metrics['hit_at_1']:.3f} | {metrics['hit_at_5']:.3f} | "
            f"{metrics['recall_at_5']:.3f} | {metrics['recall_at_10']:.3f} | "
            f"{metrics['mrr']:.3f} | {metrics['ndcg_at_10']:.3f} | "
            f"{metrics['faithfulness']:.3f} | {metrics['citation_coverage']:.3f} | "
            f"{metrics['citation_correctness']:.3f} | {metrics['unsupported_claim_rate']:.3f} | "
            f"{metrics['unanswerable_accuracy']:.3f} | {metrics['mean_latency_ms']:.3f} | "
            f"{metrics['p95_latency_ms']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "These values are computed by the script from pending silver-derived annotations. They are reproducible engineering baselines, not formal human-gold quality claims. Latency covers in-process retrieval/answer assembly and excludes model/network latency because the configured providers are deterministic local baselines.",
            "",
        ]
    )
    REPORT_OUTPUT.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
