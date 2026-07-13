# ruff: noqa: E501
"""Run four approved-only dense/hybrid retrieval ablations without LLM or reranking."""

import csv
import json
import math
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import httpx
from qdrant_client import QdrantClient
from sqlalchemy import create_engine, text

from paper_research.chunking.types import Chunk
from paper_research.config import Settings
from paper_research.evaluation.dataset import EvaluationItem, RetrievalPrediction
from paper_research.evaluation.retrieval_metrics import evaluate_retrieval
from paper_research.indexing.embedding import EmbeddingProvider, HashEmbeddingProvider
from paper_research.indexing.vector_store import QdrantVectorStore
from paper_research.providers.factory import build_embedding_provider
from paper_research.retrieval.dense import DenseRetriever
from paper_research.retrieval.fusion import reciprocal_rank_fusion
from paper_research.retrieval.sparse import BM25Retriever

DATASET = Path("data/evaluation/gold-set-v1.jsonl")
JSON_OUTPUT = Path("data/evaluation/retrieval-ablation-v1.json")
CSV_OUTPUT = Path("data/evaluation/retrieval-ablation-v1.csv")
REPORT_OUTPUT = Path("docs/retrieval-ablation-v1.md")
TOP_K = 10
RECALL_K = 20
SEED = 42

# The first ten fixture papers predate arxiv_id persistence in PostgreSQL.  Keep
# this migration-era mapping explicit and validated instead of comparing UUIDs
# with gold arXiv identifiers (which would silently produce all-zero metrics).
LEGACY_TITLE_TO_ARXIV = {
    "1706.03762": "1706.03762",
    "BERT: Pre-training of Deep Bidirectional Transformers for": "1810.04805",
    "1910.10683": "1910.10683",
    "Scaling Laws for Neural Language Models": "2001.08361",
    "Language Models are Few-Shot Learners": "2005.14165",
    "The Power of Scale for Parameter-Efﬁcient Prompt Tuning": "2104.08691",
    "LORA: LOW-RANK ADAPTATION OF LARGE LAN-\nGUAGE MODELS": "2106.09685",
    "Training language models to follow instructions": "2203.02155",
    "Scaling Instruction-Finetuned Language Models": "2210.11416",
    "LLaMA: Open and Efﬁcient Foundation Language Models": "2302.13971",
}


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def load_items(path: Path = DATASET) -> tuple[list[dict], list[EvaluationItem]]:
    raw = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    approved = [item for item in raw if item.get("review_status") == "approved"]
    pending = [item for item in raw if item.get("review_status") != "approved"]
    if pending:
        raise RuntimeError(f"formal retrieval evaluation excludes {len(pending)} non-approved items")
    answerable = [item for item in approved if item.get("answerable")]
    items = [
        EvaluationItem(
            id=item["question_id"],
            question=item["question"],
            question_type=item["category"],
            relevant_paper_ids=item["gold_paper_ids"],
            relevant_block_ids=item["gold_block_ids"],
            relevant_pages=item["gold_pages"],
            expected_answer_points=item["required_claims"],
            annotation_status="human_reviewed",
            reviewer=item.get("reviewer"),
            notes=item.get("review_notes"),
        )
        for item in answerable
    ]
    return answerable, items


def load_registry(settings: Settings) -> dict:
    path = settings.data_dir / "index_registry.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    try:
        response = httpx.get("http://localhost/api/v1/indexes", timeout=5)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as exc:
        raise RuntimeError("index registry is unavailable from file and API") from exc


def resolve_collection(registry: dict, logical: str) -> tuple[str, dict]:
    physical = registry.get("defaults", {}).get(logical)
    if not physical:
        raise RuntimeError(f"logical collection is not activated: {logical}")
    metadata = registry.get("indexes", {}).get(physical, {})
    return str(physical), metadata


def load_chunks(client: QdrantClient, collection: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        chunks.extend(Chunk.model_validate(point.payload) for point in points if point.payload)
        if offset is None:
            break
    if not chunks:
        raise RuntimeError(f"collection contains no chunks: {collection}")
    return chunks


def paper_id_map(settings: Settings) -> dict[str, str]:
    engine = create_engine(settings.database_url)
    with engine.connect() as connection:
        rows = connection.execute(text("SELECT id, title, arxiv_id FROM papers")).all()
    return {
        str(row.id): str(
            row.arxiv_id
            or LEGACY_TITLE_TO_ARXIV.get(str(row.title).strip())
            or row.id
        )
        for row in rows
    }


def validate_gold_id_coverage(raw_items: list[dict], canonical_ids: dict[str, str]) -> None:
    gold_ids = {paper_id for item in raw_items for paper_id in item["gold_paper_ids"]}
    missing = sorted(gold_ids - set(canonical_ids.values()))
    if missing:
        raise RuntimeError(f"gold paper IDs are not mapped to indexed papers: {missing}")


def evaluate_variant(
    *,
    name: str,
    profile: str,
    provider: EmbeddingProvider,
    retriever_type: str,
    collection: str,
    collection_metadata: dict,
    chunks: list[Chunk],
    client: QdrantClient,
    raw_items: list[dict],
    eval_items: list[EvaluationItem],
    canonical_ids: dict[str, str],
    settings: Settings,
) -> dict:
    dense = DenseRetriever(
        provider,
        QdrantVectorStore(client, collection, provider.dimensions),
    )
    sparse = BM25Retriever(chunks) if retriever_type == "hybrid" else None
    predictions: list[RetrievalPrediction] = []
    per_query = []
    latencies = []
    failure_count = 0
    for raw, item in zip(raw_items, eval_items, strict=True):
        started = time.perf_counter()
        error = None
        try:
            dense_results = dense.retrieve(item.question, top_k=RECALL_K)
            if sparse is None:
                results = dense_results[:TOP_K]
            else:
                sparse_results = sparse.retrieve(item.question, top_k=RECALL_K)
                results = reciprocal_rank_fusion(dense_results, sparse_results)[:TOP_K]
        except Exception as exc:
            failure_count += 1
            error = f"{type(exc).__name__}: {exc}"
            results = []
        latency = (time.perf_counter() - started) * 1000
        latencies.append(latency)
        ranked = []
        for rank, result in enumerate(results, start=1):
            chunk = result.chunk
            canonical = canonical_ids.get(chunk.paper_id, chunk.paper_id)
            ranked.append(
                {
                    "rank": rank,
                    "paper_id": canonical,
                    "raw_paper_id": chunk.paper_id,
                    "chunk_id": chunk.chunk_id,
                    "block_ids": chunk.block_ids,
                    "pages": list(range(chunk.page_start, chunk.page_end + 1)),
                    "score": round(float(result.score), 9),
                    "gold_paper_hit": canonical in item.relevant_paper_ids,
                    "gold_block_hit": bool(set(chunk.block_ids) & set(item.relevant_block_ids)),
                    "gold_page_hit": bool(
                        set(range(chunk.page_start, chunk.page_end + 1))
                        & set(item.relevant_pages)
                    ),
                }
            )
        prediction = RetrievalPrediction(
            item_id=item.id,
            ranked_paper_ids=[row["paper_id"] for row in ranked],
            ranked_block_ids=[block for row in ranked for block in row["block_ids"]],
        )
        predictions.append(prediction)
        per_query.append(
            {
                "question_id": item.id,
                "question": item.question,
                "category": raw["category"],
                "difficulty": raw["difficulty"],
                "latency_ms": round(latency, 3),
                "gold_paper_ids": item.relevant_paper_ids,
                "gold_block_ids": item.relevant_block_ids,
                "gold_pages": item.relevant_pages,
                "ranked_results": ranked,
                "failure_reason": error
                or (None if any(row["gold_paper_hit"] for row in ranked) else "gold paper not in Top-10"),
            }
        )
    overall = evaluate_retrieval(eval_items, predictions, k_values=(1, 5, 10))
    grouped = {}
    for field in ("category", "difficulty"):
        buckets: dict[str, list[int]] = defaultdict(list)
        for index, raw in enumerate(raw_items):
            buckets[str(raw[field])].append(index)
        grouped[field] = {
            key: evaluate_retrieval(
                [eval_items[index] for index in indexes],
                [predictions[index] for index in indexes],
                k_values=(1, 5, 10),
            )
            for key, indexes in sorted(buckets.items())
        }
    point_count = int(client.count(collection, exact=True).count)
    return {
        "name": name,
        "configuration": {
            "profile": profile,
            "provider": provider.provider_name,
            "model": provider.model_name,
            "revision": provider.revision,
            "dimension": provider.dimensions,
            "collection": collection,
            "index_version": collection_metadata.get("index_version", settings.index_version),
            "dataset_version": settings.dataset_version,
            "retriever_type": retriever_type,
            "rerank_enabled": False,
            "top_k": TOP_K,
            "recall_k": RECALL_K,
        },
        "metrics": {
            **overall,
            "mean_latency_ms": round(sum(latencies) / len(latencies), 3),
            "p50_latency_ms": round(percentile(latencies, 0.50), 3),
            "p95_latency_ms": round(percentile(latencies, 0.95), 3),
            "query_count": len(eval_items),
            "failure_count": failure_count,
            "index_build_duration_seconds": collection_metadata.get("build_duration_seconds"),
            "paper_count": collection_metadata.get("paper_count"),
            "point_count": point_count,
        },
        "by_category": grouped["category"],
        "by_difficulty": grouped["difficulty"],
        "queries": per_query,
    }


def main() -> None:
    settings = Settings()
    if settings.app_profile != "production":
        raise RuntimeError("APP_PROFILE=production is required for the four-way ablation")
    if settings.embedding_provider != "jina":
        raise RuntimeError("EMBEDDING_PROVIDER=jina is required for the real embedding variants")
    if settings.rerank_enabled:
        raise RuntimeError("RERANK_ENABLED must remain false for Stage 11A")
    if settings.llm_provider != "template":
        raise RuntimeError("LLM_PROVIDER must remain template; this script never calls an LLM")
    raw_items, eval_items = load_items()
    registry = load_registry(settings)
    baseline_collection, baseline_metadata = resolve_collection(
        registry, settings.baseline_collection
    )
    production_collection, production_metadata = resolve_collection(
        registry, settings.production_collection
    )
    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        check_compatibility=False,
    )
    baseline_chunks = load_chunks(client, baseline_collection)
    production_chunks = load_chunks(client, production_collection)
    canonical_ids = paper_id_map(settings)
    validate_gold_id_coverage(raw_items, canonical_ids)
    providers = {
        "hash": HashEmbeddingProvider(384),
        "jina": build_embedding_provider(settings),
    }
    variants = []
    for provider_name, profile, collection, metadata, chunks in (
        ("hash", "baseline", baseline_collection, baseline_metadata, baseline_chunks),
        ("jina", "production", production_collection, production_metadata, production_chunks),
    ):
        for retriever_type in ("dense", "hybrid"):
            variants.append(
                evaluate_variant(
                    name=f"{provider_name}_structural_{retriever_type}",
                    profile=profile,
                    provider=providers[provider_name],
                    retriever_type=retriever_type,
                    collection=collection,
                    collection_metadata=metadata,
                    chunks=chunks,
                    client=client,
                    raw_items=raw_items,
                    eval_items=eval_items,
                    canonical_ids=canonical_ids,
                    settings=settings,
                )
            )
    dataset_versions = {item["dataset_version"] for item in raw_items}
    if len(dataset_versions) != 1:
        raise RuntimeError(f"evaluation records have mixed dataset versions: {dataset_versions}")
    dataset_version = dataset_versions.pop()
    for variant in variants:
        variant["configuration"]["dataset_version"] = dataset_version
    payload = {
        "status": "COMPLETED",
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset": str(DATASET),
        "dataset_version": dataset_version,
        "approved_records": 50,
        "evaluated_answerable_records": len(eval_items),
        "approved_unanswerable_excluded": 50 - len(eval_items),
        "random_seed": SEED,
        "llm_called": False,
        "reranker_called": False,
        "variants": variants,
    }
    JSON_OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(variants)
    write_report(payload)
    print(json.dumps({"status": "COMPLETED", "variants": [v["name"] for v in variants]}))


def write_csv(variants: list[dict]) -> None:
    metric_names = list(variants[0]["metrics"])
    with CSV_OUTPUT.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["variant", *metric_names])
        writer.writeheader()
        for variant in variants:
            writer.writerow({"variant": variant["name"], **variant["metrics"]})


def write_report(payload: dict) -> None:
    lines = [
        "# Retrieval Ablation v1",
        "",
        f"- Dataset: `{payload['dataset_version']}`",
        f"- Approved: {payload['approved_records']}/50",
        f"- Evaluated answerable queries: {payload['evaluated_answerable_records']}",
        f"- Approved unanswerable records excluded from retrieval metrics: {payload['approved_unanswerable_excluded']}",
        "- Reranker: disabled",
        "- LLM calls: none",
        "",
        "| Variant | Hit@1 | Hit@5 | Recall@5 | Recall@10 | MRR | NDCG@10 | Mean ms | P50 ms | P95 ms | Failures | Points |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in payload["variants"]:
        metrics = variant["metrics"]
        lines.append(
            f"| {variant['name']} | {metrics['hit_at_1']:.3f} | {metrics['hit_at_5']:.3f} | "
            f"{metrics['recall_at_5']:.3f} | {metrics['recall_at_10']:.3f} | "
            f"{metrics['mrr']:.3f} | {metrics['ndcg_at_10']:.3f} | "
            f"{metrics['mean_latency_ms']:.3f} | {metrics['p50_latency_ms']:.3f} | "
            f"{metrics['p95_latency_ms']:.3f} | {metrics['failure_count']} | "
            f"{metrics['point_count']} |"
        )
    variants = {variant["name"]: variant for variant in payload["variants"]}
    metric_names = ("hit_at_1", "hit_at_5", "recall_at_5", "recall_at_10", "mrr", "ndcg_at_10")
    lines.extend(["", "## Measured deltas", "", "| Retriever | Metric | Jina - Hash |", "|---|---|---:|"])
    for retriever in ("dense", "hybrid"):
        baseline = variants[f"hash_structural_{retriever}"]["metrics"]
        production = variants[f"jina_structural_{retriever}"]["metrics"]
        for metric in metric_names:
            lines.append(f"| {retriever} | {metric} | {production[metric] - baseline[metric]:+.3f} |")

    def first_gold_rank(query: dict) -> int:
        return next(
            (result["rank"] for result in query["ranked_results"] if result["gold_paper_hit"]),
            TOP_K + 1,
        )

    hash_queries = variants["hash_structural_hybrid"]["queries"]
    jina_queries = variants["jina_structural_hybrid"]["queries"]
    comparisons = []
    for hash_query, jina_query in zip(hash_queries, jina_queries, strict=True):
        hash_rank = first_gold_rank(hash_query)
        jina_rank = first_gold_rank(jina_query)
        comparisons.append(
            {
                "question_id": hash_query["question_id"],
                "question": hash_query["question"],
                "hash_rank": hash_rank,
                "jina_rank": jina_rank,
                "delta": hash_rank - jina_rank,
            }
        )

    lines.extend(["", "## Query examples", ""])
    for heading, examples in (
        ("Semantic improvements", sorted(comparisons, key=lambda row: row["delta"], reverse=True)),
        ("Regressions", sorted(comparisons, key=lambda row: row["delta"])),
    ):
        lines.extend([f"### {heading}", ""])
        selected = [row for row in examples if (row["delta"] > 0 if heading == "Semantic improvements" else row["delta"] < 0)][:3]
        if not selected:
            lines.append("No rank-changing examples in Top-10.")
        for row in selected:
            hash_rank = ">10" if row["hash_rank"] > TOP_K else str(row["hash_rank"])
            jina_rank = ">10" if row["jina_rank"] > TOP_K else str(row["jina_rank"])
            lines.append(f"- `{row['question_id']}`: Hash rank {hash_rank}, Jina rank {jina_rank} — {row['question']}")
        lines.append("")

    no_hit = [row for row in comparisons if row["hash_rank"] > TOP_K and row["jina_rank"] > TOP_K]
    lines.extend(["## Failure examples", ""])
    if no_hit:
        for row in no_hit[:5]:
            lines.append(f"- `{row['question_id']}`: neither hybrid variant retrieved a gold paper in Top-10 — {row['question']}")
    else:
        lines.append("Every answerable query had a gold-paper hit in at least one hybrid variant.")

    lines.extend(
        [
            "",
            "## Evaluation validity note",
            "",
            "Many gold questions deliberately use generic phrases such as `the target paper` and do not include a title, arXiv ID, or topic-bearing subject. In an unrestricted 36-paper corpus retrieval task, those queries do not identify which paper is intended. The scores are reproducible for this dataset, but they must not be interpreted as a general model leaderboard until the protocol either supplies a non-gold paper scope filter or rewrites questions to be independently identifiable.",
        ]
    )

    hash_hybrid = variants["hash_structural_hybrid"]["metrics"]
    jina_hybrid = variants["jina_structural_hybrid"]["metrics"]
    jina_default = (
        jina_hybrid["mrr"] > hash_hybrid["mrr"]
        and jina_hybrid["ndcg_at_10"] > hash_hybrid["ndcg_at_10"]
        and jina_hybrid["recall_at_5"] >= hash_hybrid["recall_at_5"]
        and jina_hybrid["recall_at_10"] >= hash_hybrid["recall_at_10"]
    )
    lines.extend(
        [
            "",
            "## Stage 11B recommendation",
            "",
            (
                "Use Structural + Jina Dense + Sparse Hybrid as the Stage 11B production retrieval default; retain Hash Hybrid as the offline baseline."
                if jina_default
                else "Do not promote Jina Hybrid as the Stage 11B default from this run; retain Hash Hybrid until the measured MRR, NDCG@10, and recall acceptance condition is met."
            ),
            "This decision is generated only from the measured hybrid metrics; no reranker or LLM result is involved.",
            "",
        ]
    )
    REPORT_OUTPUT.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
