import json
import time
from pathlib import Path

from qdrant_client import QdrantClient

from paper_research.chunking.fixed_chunker import FixedTokenChunker
from paper_research.chunking.structural_chunker import StructuralChunker
from paper_research.chunking.tokenizer import count_tokens
from paper_research.evaluation.agent_metrics import evaluate_agent_run
from paper_research.evaluation.answer_metrics import evaluate_answer
from paper_research.evaluation.dataset import EvaluationItem, RetrievalPrediction
from paper_research.evaluation.observability import UsageEvent, UsageRecorder
from paper_research.evaluation.retrieval_metrics import evaluate_retrieval
from paper_research.indexing.embedding import HashEmbeddingProvider
from paper_research.indexing.vector_store import QdrantVectorStore
from paper_research.parsing.types import PaperBlock
from paper_research.retrieval.dense import DenseRetriever
from paper_research.retrieval.fusion import reciprocal_rank_fusion
from paper_research.retrieval.reranker import LexicalReranker
from paper_research.retrieval.sparse import BM25Retriever


def load_corpus() -> tuple[list, list]:
    structural, fixed = [], []
    for path in sorted(Path("data/reports/parsing-audit").glob("*/paper_blocks.jsonl")):
        blocks = [
            PaperBlock.model_validate(json.loads(line))
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        structural.extend(StructuralChunker().chunk(path.parent.name, blocks))
        fixed.extend(FixedTokenChunker().chunk(path.parent.name, blocks))
    return structural, fixed


def main() -> None:
    items = [
        EvaluationItem.model_validate(json.loads(line))
        for line in Path("data/evaluation/research_qa_50.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    structural, fixed = load_corpus()
    embedding = HashEmbeddingProvider(384)
    structural_store = QdrantVectorStore(QdrantClient(":memory:"), "structural", 384)
    fixed_store = QdrantVectorStore(QdrantClient(":memory:"), "fixed", 384)
    structural_store.upsert(
        structural, embedding.embed([chunk.chunk_text for chunk in structural])
    )
    fixed_store.upsert(fixed, embedding.embed([chunk.chunk_text for chunk in fixed]))
    dense = DenseRetriever(embedding, structural_store)
    dense_fixed = DenseRetriever(embedding, fixed_store)
    sparse = BM25Retriever(structural)
    reranker = LexicalReranker()
    variant_names = (
        "fixed_dense",
        "structural_dense",
        "sparse",
        "hybrid",
        "hybrid_rerank",
    )
    variants: dict[str, list[RetrievalPrediction]] = {name: [] for name in variant_names}
    answer_scores = []
    generated_answer_tokens = 0
    started = time.perf_counter()
    for item in items:
        fixed_results = dense_fixed.retrieve(item.question, top_k=10)
        dense_results = dense.retrieve(item.question, top_k=20)
        sparse_results = sparse.retrieve(item.question, top_k=20)
        fused = reciprocal_rank_fusion(dense_results, sparse_results)
        reranked = reranker.rerank(item.question, fused, 10)
        result_sets = {
            "fixed_dense": fixed_results,
            "structural_dense": dense_results[:10],
            "sparse": sparse_results[:10],
            "hybrid": fused[:10],
            "hybrid_rerank": reranked,
        }
        for name, results in result_sets.items():
            variants[name].append(
                RetrievalPrediction(
                    item_id=item.id,
                    ranked_paper_ids=[result.chunk.paper_id for result in results],
                    ranked_block_ids=[
                        block_id for result in results for block_id in result.chunk.block_ids
                    ],
                )
            )
        contexts = [result.chunk.chunk_text for result in reranked[:5]]
        answer = ""
        if contexts:
            citation = f"[{reranked[0].chunk.paper_id}, p.{reranked[0].chunk.page_start}]"
            answer = f"{contexts[0]} {citation}"
        generated_answer_tokens += count_tokens(answer)
        citations = [
            {
                "valid": result.chunk.paper_id in item.relevant_paper_ids,
                "paper_id": result.chunk.paper_id,
                "page": result.chunk.page_start,
            }
            for result in reranked[:5]
        ]
        answer_scores.append(
            evaluate_answer(answer, contexts, citations, item.expected_answer_points)
        )
    elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
    retrieval = {
        name: evaluate_retrieval(items, predictions) for name, predictions in variants.items()
    }
    answer_metrics = {
        key: round(sum(score[key] for score in answer_scores) / len(answer_scores), 6)
        for key in answer_scores[0]
    }
    agent_state = json.loads(
        Path("data/reports/deep-research-audit/research_state.json").read_text(encoding="utf-8")
    )
    result = {
        "dataset": {
            "items": len(items),
            "human_reviewed": sum(item.annotation_status == "human_reviewed" for item in items),
            "silver": sum(item.annotation_status == "silver" for item in items),
        },
        "retrieval": retrieval,
        "answer": answer_metrics,
        "agent": evaluate_agent_run(agent_state),
        "runtime": {
            "queries": len(items),
            "total_latency_ms": elapsed_ms,
            "mean_latency_ms": round(elapsed_ms / len(items), 3),
            "estimated_tokens": agent_state.get("estimated_tokens", 0),
            "estimated_cost_usd": 0.0,
            "cost_note": "All evaluated providers are local deterministic baselines.",
        },
    }
    write_reports(result)
    UsageRecorder(Path("data/reports/evaluation/usage_events.jsonl")).append(
        UsageEvent(
            operation="evaluation_suite",
            latency_ms=elapsed_ms,
            input_tokens=sum(count_tokens(item.question) for item in items),
            output_tokens=generated_answer_tokens,
            estimated_cost_usd=0.0,
            attributes={"queries": len(items), "variants": len(variants)},
        )
    )


def write_reports(result: dict) -> None:
    output = Path("data/reports/evaluation")
    output.mkdir(parents=True, exist_ok=True)
    (output / "evaluation_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = [
        "# Evaluation and Ablation Report",
        "",
        f"- Dataset items: {result['dataset']['items']}",
        f"- Human reviewed: {result['dataset']['human_reviewed']}",
        f"- Silver pending review: {result['dataset']['silver']}",
        f"- Mean query latency: {result['runtime']['mean_latency_ms']} ms",
        f"- Estimated cost: ${result['runtime']['estimated_cost_usd']:.4f}",
        "",
        "## Retrieval ablation",
        "",
        "| Variant | Hit@1 | Hit@5 | Recall@10 | MRR | NDCG@10 | Block Hit@10 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in result["retrieval"].items():
        report.append(
            f"| {name} | {metrics['hit_at_1']:.3f} | {metrics['hit_at_5']:.3f} | "
            f"{metrics['recall_at_10']:.3f} | {metrics['mrr']:.3f} | "
            f"{metrics['ndcg_at_10']:.3f} | {metrics['block_hit_at_10']:.3f} |"
        )
    report.extend(["", "## Answer and citation metrics", ""])
    report.extend(f"- {name}: {value:.3f}" for name, value in result["answer"].items())
    report.extend(["", "## Agent metrics", ""])
    report.extend(f"- {name}: {value}" for name, value in result["agent"].items())
    (output / "evaluation_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
