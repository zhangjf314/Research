# ruff: noqa: E501
"""Run Stage 11A.5 scope-aware retrieval ablations on the fixed 34-document corpus."""

import csv
import hashlib
import json
import math
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from qdrant_client import QdrantClient

from paper_research.chunking.types import Chunk
from paper_research.config import Settings
from paper_research.indexing.embedding import EmbeddingProvider, HashEmbeddingProvider
from paper_research.indexing.vector_store import QdrantVectorStore
from paper_research.providers.factory import build_embedding_provider
from paper_research.retrieval.dense import DenseRetriever, RetrievalResult
from paper_research.retrieval.filters import RetrievalFilter
from paper_research.retrieval.fusion import reciprocal_rank_fusion
from paper_research.retrieval.sparse import BM25Retriever

PROTOCOL = Path("data/evaluation/retrieval-gold-v2.jsonl")
CORPUS = Path("data/evaluation/production-corpus-v1.json")
INDEX_MANIFEST = Path("data/evaluation/retrieval-index-v2.json")
V1_RESULTS = Path("data/evaluation/retrieval-ablation-v1.json")
JSON_OUTPUT = Path("data/evaluation/retrieval-ablation-v2.json")
CSV_OUTPUT = Path("data/evaluation/retrieval-ablation-v2.csv")
REPORT_OUTPUT = Path("docs/retrieval-ablation-v2.md")
TOP_K = 10
RECALL_K = 20
SEED = 42


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_chunks(client: QdrantClient, collection: str) -> list[Chunk]:
    chunks = []
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
            return chunks


def chunk_signature(chunks: list[Chunk]) -> str:
    digest = hashlib.sha256()
    for chunk in sorted(chunks, key=lambda item: item.chunk_id):
        digest.update(
            json.dumps(chunk.model_dump(), sort_keys=True, ensure_ascii=False).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def validate_inputs(protocol: list[dict], corpus: dict, index_manifest: dict) -> None:
    if len(protocol) != 50:
        raise RuntimeError(f"retrieval protocol must contain 50 records, got {len(protocol)}")
    included = [paper for paper in corpus["papers"] if paper["included_in_production"]]
    excluded = [paper for paper in corpus["papers"] if not paper["included_in_production"]]
    if len(included) != 34 or len(excluded) != 2:
        raise RuntimeError("production corpus boundary must be exactly 34 included and 2 excluded")
    if any(record["review_status"] != "approved" for record in protocol):
        raise RuntimeError("all source gold annotations must remain approved")
    if index_manifest["collections"]["hash"]["chunk_signature"] != index_manifest[
        "collections"
    ]["jina"]["chunk_signature"]:
        raise RuntimeError("Hash and Jina index manifests do not share a chunk signature")


def dcg_binary(relevance: list[bool], ideal_count: int) -> float:
    dcg = sum(
        (1.0 if relevant else 0.0) / math.log2(index + 2)
        for index, relevant in enumerate(relevance[:TOP_K])
    )
    ideal = sum(1.0 / math.log2(index + 2) for index in range(min(ideal_count, TOP_K)))
    return dcg / ideal if ideal else 0.0


def first_rank(values: list[bool]) -> int | None:
    return next((index for index, relevant in enumerate(values, start=1) if relevant), None)


def summarize_global(queries: list[dict]) -> dict:
    if not queries:
        return {"query_count": 0, "status": "NOT_APPLICABLE_NO_GLOBAL_ITEMS"}
    output: dict[str, float | int | str] = {"query_count": len(queries), "status": "COMPUTED"}
    for k in (1, 5):
        output[f"paper_hit_at_{k}"] = round(
            mean(
                [
                    float(any(row["gold_paper_hit"] for row in query["ranked_results"][:k]))
                    for query in queries
                ]
            ),
            6,
        )
    for k in (5, 10):
        output[f"block_recall_at_{k}"] = round(
            mean(
                [
                    len(
                        set(query["gold_block_ids"])
                        & {
                            block
                            for row in query["ranked_results"][:k]
                            for block in row["block_ids"]
                        }
                    )
                    / len(set(query["gold_block_ids"]))
                    for query in queries
                ]
            ),
            6,
        )
    ranks = [first_rank([row["gold_paper_hit"] for row in query["ranked_results"]]) for query in queries]
    output["mrr"] = round(mean([1 / rank if rank else 0.0 for rank in ranks]), 6)
    output["ndcg_at_10"] = round(
        mean(
            [
                dcg_binary(
                    [row["gold_paper_hit"] for row in query["ranked_results"]],
                    len(set(query["gold_paper_ids"])),
                )
                for query in queries
            ]
        ),
        6,
    )
    return output


def block_scope_metrics(queries: list[dict]) -> dict:
    if not queries:
        return {"query_count": 0, "status": "NOT_APPLICABLE"}
    output: dict[str, float | int | str] = {"query_count": len(queries), "status": "COMPUTED"}
    for k in (1, 5):
        output[f"block_hit_at_{k}"] = round(
            mean(
                [
                    float(any(row["gold_block_hit"] for row in query["ranked_results"][:k]))
                    for query in queries
                ]
            ),
            6,
        )
    for k in (5, 10):
        recalls = []
        for query in queries:
            gold = set(query["gold_block_ids"])
            retrieved = {
                block
                for row in query["ranked_results"][:k]
                for block in row["block_ids"]
            }
            recalls.append(len(gold & retrieved) / len(gold))
        output[f"block_recall_at_{k}"] = round(mean(recalls), 6)
    ranks = [first_rank([row["gold_block_hit"] for row in query["ranked_results"]]) for query in queries]
    output["mrr"] = round(mean([1 / rank if rank else 0.0 for rank in ranks]), 6)
    output["ndcg_at_10"] = round(
        mean(
            [
                dcg_binary(
                    [row["gold_block_hit"] for row in query["ranked_results"]],
                    min(len(set(query["gold_block_ids"])), TOP_K),
                )
                for query in queries
            ]
        ),
        6,
    )
    return output


def summarize_multi(queries: list[dict]) -> dict:
    if not queries:
        return {"query_count": 0, "status": "NOT_APPLICABLE"}
    output: dict[str, float | int | str] = {"query_count": len(queries), "status": "COMPUTED"}
    for k in (5, 10):
        coverages = []
        all_covered = []
        evidence = []
        for query in queries:
            target = set(query["gold_paper_ids"])
            papers = {row["paper_id"] for row in query["ranked_results"][:k]}
            blocks = {
                block
                for row in query["ranked_results"][:k]
                for block in row["block_ids"]
            }
            gold_blocks = set(query["gold_block_ids"])
            coverages.append(len(target & papers) / len(target))
            all_covered.append(float(target <= papers))
            evidence.append(len(gold_blocks & blocks) / len(gold_blocks))
        output[f"target_paper_coverage_at_{k}"] = round(mean(coverages), 6)
        output[f"all_target_papers_at_{k}"] = round(mean(all_covered), 6)
        output[f"evidence_recall_at_{k}"] = round(mean(evidence), 6)
    ranks = [first_rank([row["gold_block_hit"] for row in query["ranked_results"]]) for query in queries]
    output["evidence_mrr"] = round(mean([1 / rank if rank else 0.0 for rank in ranks]), 6)
    return output


def summarize_unanswerable(queries: list[dict]) -> dict:
    scores = [
        query["ranked_results"][0]["score"]
        for query in queries
        if query["ranked_results"]
    ]
    return {
        "query_count": len(queries),
        "returned_results_count": sum(bool(query["ranked_results"]) for query in queries),
        "nonempty_rate": round(mean([float(bool(query["ranked_results"])) for query in queries]), 6),
        "mean_top_score": round(mean(scores), 6),
        "max_top_score": round(max(scores), 6) if scores else None,
        "note": "Descriptive only; retrieval cannot determine answerability without a calibrated QA threshold.",
    }


def grouped_evidence(queries: list[dict], field: str) -> dict:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for query in queries:
        buckets[str(query[field])].append(query)
    output = {}
    for key, rows in sorted(buckets.items()):
        answerable = [row for row in rows if row["retrieval_scope"] != "unanswerable"]
        output[key] = {
            "query_count": len(rows),
            "answerable_query_count": len(answerable),
            **block_scope_metrics(answerable),
        }
    return output


def ranked_rows(
    results: list[RetrievalResult], record: dict, raw_to_public: dict[str, str]
) -> list[dict]:
    gold_papers = set(record["gold_paper_ids"])
    gold_blocks = set(record["gold_block_ids"])
    gold_pages = set(record["gold_pages"])
    rows = []
    for rank, result in enumerate(results, start=1):
        chunk = result.chunk
        public_id = raw_to_public[chunk.paper_id]
        pages = list(range(chunk.page_start, chunk.page_end + 1))
        rows.append(
            {
                "rank": rank,
                "paper_id": public_id,
                "database_id": chunk.paper_id,
                "chunk_id": chunk.chunk_id,
                "block_ids": chunk.block_ids,
                "pages": pages,
                "score": round(float(result.score), 9),
                "gold_paper_hit": public_id in gold_papers,
                "gold_block_hit": bool(set(chunk.block_ids) & gold_blocks),
                "gold_page_hit": bool(set(pages) & gold_pages),
            }
        )
    return rows


def evaluate_variant(
    *,
    name: str,
    provider: EmbeddingProvider,
    retriever_type: str,
    collection: str,
    chunks: list[Chunk],
    protocol: list[dict],
    public_to_raw: dict[str, str],
    raw_to_public: dict[str, str],
    production_raw_ids: list[str],
    client: QdrantClient,
) -> dict:
    dense = DenseRetriever(provider, QdrantVectorStore(client, collection, provider.dimensions))
    sparse = BM25Retriever(chunks) if retriever_type == "hybrid" else None
    queries = []
    latencies = []
    failure_count = 0
    for record in protocol:
        filter_public = record["retrieval_filter"]["paper_ids"]
        filter_raw = (
            production_raw_ids
            if record["retrieval_scope"] == "global"
            else [public_to_raw[paper_id] for paper_id in filter_public]
        )
        retrieval_filter = RetrievalFilter(paper_ids=filter_raw)
        started = time.perf_counter()
        error = None
        try:
            dense_results = dense.retrieve(
                record["retrieval_query"],
                retrieval_filter=retrieval_filter,
                top_k=RECALL_K,
            )
            if sparse is None:
                results = dense_results[:TOP_K]
            else:
                sparse_results = sparse.retrieve(
                    record["retrieval_query"],
                    retrieval_filter=retrieval_filter,
                    top_k=RECALL_K,
                )
                results = reciprocal_rank_fusion(dense_results, sparse_results)[:TOP_K]
        except Exception as exc:
            failure_count += 1
            error = f"{type(exc).__name__}: {exc}"
            results = []
        latency = (time.perf_counter() - started) * 1000
        latencies.append(latency)
        ranked = ranked_rows(results, record, raw_to_public)
        queries.append(
            {
                **record,
                "applied_filter_database_ids": filter_raw,
                "latency_ms": round(latency, 3),
                "ranked_results": ranked,
                "failure_reason": error,
            }
        )
    by_scope = {
        scope: [query for query in queries if query["retrieval_scope"] == scope]
        for scope in ("global", "paper", "multi_paper", "unanswerable")
    }
    return {
        "name": name,
        "configuration": {
            "profile": "baseline" if provider.provider_name == "hash" else "production",
            "provider": provider.provider_name,
            "model": provider.model_name,
            "revision": provider.revision,
            "dimension": provider.dimensions,
            "collection": collection,
            "corpus_manifest": str(CORPUS),
            "protocol": str(PROTOCOL),
            "retriever_type": retriever_type,
            "rerank_enabled": False,
            "llm_called": False,
            "top_k": TOP_K,
            "recall_k": RECALL_K,
        },
        "metrics": {
            "global": summarize_global(by_scope["global"]),
            "paper": block_scope_metrics(by_scope["paper"]),
            "multi_paper": summarize_multi(by_scope["multi_paper"]),
            "unanswerable": summarize_unanswerable(by_scope["unanswerable"]),
            "latency": {
                "query_count": len(queries),
                "failure_count": failure_count,
                "mean_ms": round(mean(latencies), 3),
                "p50_ms": round(percentile(latencies, 0.50), 3),
                "p95_ms": round(percentile(latencies, 0.95), 3),
            },
        },
        "by_category": grouped_evidence(queries, "category"),
        "by_difficulty": grouped_evidence(queries, "difficulty"),
        "queries": queries,
    }


def write_csv(variants: list[dict]) -> None:
    rows = []
    for variant in variants:
        for scope in ("global", "paper", "multi_paper", "unanswerable", "latency"):
            rows.append(
                {
                    "variant": variant["name"],
                    "breakdown": "scope",
                    "group": scope,
                    **variant["metrics"][scope],
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


def metric(value: object) -> str:
    return "N/A" if value is None else f"{value:.3f}" if isinstance(value, float) else str(value)


def write_report(payload: dict) -> None:
    variants = {variant["name"]: variant for variant in payload["variants"]}
    lines = [
        "# Retrieval Ablation v2 — Scope-aware Protocol",
        "",
        "- Corpus: 34 included documents (33 research papers + 1 text-native release fixture)",
        "- Excluded but retained: 2 OCR fixtures",
        "- Protocol: 50 approved source records; query revisions are independently tracked",
        f"- Pending query revision reviews: {payload['pending_query_review_count']}",
        "- Scope distribution: global 0, paper 46, multi_paper 2, unanswerable 2",
        "- Reranker: disabled",
        "- LLM calls: none",
        "- Hash and Jina evaluation collections share the same chunk signature",
        "",
        "## Global retrieval",
        "",
        "No source question is a genuine paper-discovery task. Global metrics are intentionally `N/A` rather than manufacturing title-bearing queries from known answers.",
        "",
        "## Paper-scoped block retrieval",
        "",
        "| Variant | Block Hit@1 | Block Hit@5 | Recall@5 | Recall@10 | MRR | NDCG@10 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in payload["variants"]:
        m = variant["metrics"]["paper"]
        lines.append(
            f"| {variant['name']} | {metric(m.get('block_hit_at_1'))} | {metric(m.get('block_hit_at_5'))} | {metric(m.get('block_recall_at_5'))} | {metric(m.get('block_recall_at_10'))} | {metric(m.get('mrr'))} | {metric(m.get('ndcg_at_10'))} |"
        )
    lines.extend(
        [
            "",
            "## Multi-paper retrieval",
            "",
            "| Variant | Paper coverage@5 | All papers@5 | Evidence recall@5 | Paper coverage@10 | All papers@10 | Evidence recall@10 | Evidence MRR |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for variant in payload["variants"]:
        m = variant["metrics"]["multi_paper"]
        lines.append(
            f"| {variant['name']} | {metric(m.get('target_paper_coverage_at_5'))} | {metric(m.get('all_target_papers_at_5'))} | {metric(m.get('evidence_recall_at_5'))} | {metric(m.get('target_paper_coverage_at_10'))} | {metric(m.get('all_target_papers_at_10'))} | {metric(m.get('evidence_recall_at_10'))} | {metric(m.get('evidence_mrr'))} |"
        )
    lines.extend(
        [
            "",
            "## Unanswerable retrieval behavior",
            "",
            "| Variant | Queries | Non-empty rate | Mean top score | Max top score |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for variant in payload["variants"]:
        m = variant["metrics"]["unanswerable"]
        lines.append(
            f"| {variant['name']} | {m['query_count']} | {metric(m['nonempty_rate'])} | {metric(m['mean_top_score'])} | {metric(m['max_top_score'])} |"
        )
    lines.extend(
        [
            "",
            "These scores are descriptive only. Retrieval returning a passage is not evidence that an answer exists; refusal requires a later calibrated QA protocol.",
            "",
            "## Latency",
            "",
            "| Variant | Mean ms | P50 ms | P95 ms | Failures |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for variant in payload["variants"]:
        m = variant["metrics"]["latency"]
        lines.append(
            f"| {variant['name']} | {m['mean_ms']:.3f} | {m['p50_ms']:.3f} | {m['p95_ms']:.3f} | {m['failure_count']} |"
        )

    for breakdown, title in (("by_category", "Category"), ("by_difficulty", "Difficulty")):
        lines.extend(
            [
                "",
                f"## {title} breakdown",
                "",
                f"| Variant | {title} | Answerable | Block Hit@5 | Block Recall@10 | MRR |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for variant in payload["variants"]:
            for group, m in variant[breakdown].items():
                lines.append(
                    f"| {variant['name']} | {group} | {m['answerable_query_count']} | {metric(m.get('block_hit_at_5'))} | {metric(m.get('block_recall_at_10'))} | {metric(m.get('mrr'))} |"
                )

    def relevant_rank(query: dict) -> int:
        return next(
            (row["rank"] for row in query["ranked_results"] if row["gold_block_hit"]),
            TOP_K + 1,
        )

    hash_queries = variants["hash_structural_hybrid"]["queries"]
    jina_queries = variants["jina_structural_hybrid"]["queries"]
    comparisons = []
    for hash_query, jina_query in zip(hash_queries, jina_queries, strict=True):
        if hash_query["retrieval_scope"] == "unanswerable":
            continue
        hash_rank = relevant_rank(hash_query)
        jina_rank = relevant_rank(jina_query)
        comparisons.append(
            {
                "id": hash_query["question_id"],
                "question": hash_query["retrieval_query"],
                "hash": hash_rank,
                "jina": jina_rank,
                "delta": hash_rank - jina_rank,
            }
        )
    lines.extend(["", "## Hash/Jina query changes", ""])
    for heading, ordered, positive in (
        ("Improvements", sorted(comparisons, key=lambda row: row["delta"], reverse=True), True),
        ("Regressions", sorted(comparisons, key=lambda row: row["delta"]), False),
    ):
        lines.extend([f"### {heading}", ""])
        selected = [row for row in ordered if (row["delta"] > 0 if positive else row["delta"] < 0)][:5]
        if not selected:
            lines.append("No Top-10 rank changes in this direction.")
        for row in selected:
            hash_rank = ">10" if row["hash"] > TOP_K else row["hash"]
            jina_rank = ">10" if row["jina"] > TOP_K else row["jina"]
            lines.append(
                f"- `{row['id']}`: Hash rank {hash_rank}, Jina rank {jina_rank} — {row['question']}"
            )
        lines.append("")

    v1 = payload.get("v1_reference", {})
    lines.extend(
        [
            "## v1 versus v2 protocol",
            "",
            "v1 ran all answerable items as unrestricted paper discovery and scored paper IDs. v2 treats 46 known-paper questions as filtered within-paper block retrieval and two comparisons as filtered multi-paper evidence retrieval. Therefore the numeric MRR values below describe different tasks and must not be interpreted as a direct improvement percentage.",
            "",
            "| Run | Hash Hybrid MRR | Jina Hybrid MRR | Meaning |",
            "|---|---:|---:|---|",
            f"| v1 | {metric(v1.get('hash_hybrid_mrr'))} | {metric(v1.get('jina_hybrid_mrr'))} | Unrestricted paper-ID retrieval |",
            f"| v2 | {metric(variants['hash_structural_hybrid']['metrics']['paper'].get('mrr'))} | {metric(variants['jina_structural_hybrid']['metrics']['paper'].get('mrr'))} | Paper-filtered block retrieval |",
        ]
    )
    hash_m = variants["hash_structural_hybrid"]["metrics"]["paper"]
    jina_m = variants["jina_structural_hybrid"]["metrics"]["paper"]
    promote = (
        jina_m["mrr"] > hash_m["mrr"]
        and jina_m["ndcg_at_10"] > hash_m["ndcg_at_10"]
        and jina_m["block_recall_at_5"] >= hash_m["block_recall_at_5"]
        and jina_m["block_recall_at_10"] >= hash_m["block_recall_at_10"]
    )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            (
                "The measured paper-scoped acceptance condition supports Jina as the retrieval embedding candidate."
                if promote
                else "The measured paper-scoped acceptance condition does not support making Jina the Production default."
            ),
            (
                "All query revisions are human-approved. Stage 11B may begin with Reranker still disabled by default; no Reranker result is inferred from this run."
                if payload["pending_query_review_count"] == 0
                else "Stage 11B may begin only after pending query revisions receive human review; no Reranker result is inferred from this run."
            ),
            "",
        ]
    )
    REPORT_OUTPUT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    settings = Settings()
    if settings.app_profile != "production" or settings.embedding_provider != "jina":
        raise RuntimeError("Stage 11A.5 requires the configured real Jina Production profile")
    if settings.rerank_enabled:
        raise RuntimeError("RERANK_ENABLED must remain false")
    if settings.llm_provider != "template":
        raise RuntimeError("LLM_PROVIDER must remain template")
    protocol = load_jsonl(PROTOCOL)
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    index_manifest = json.loads(INDEX_MANIFEST.read_text(encoding="utf-8"))
    validate_inputs(protocol, corpus, index_manifest)
    included = [paper for paper in corpus["papers"] if paper["included_in_production"]]
    public_to_raw = {paper["paper_id"]: paper["database_id"] for paper in included}
    raw_to_public = {paper["database_id"]: paper["paper_id"] for paper in included}
    production_raw_ids = list(raw_to_public)
    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        check_compatibility=False,
    )
    collections = index_manifest["collections"]
    chunks = {
        provider: load_chunks(client, metadata["name"])
        for provider, metadata in collections.items()
    }
    signatures = {provider: chunk_signature(items) for provider, items in chunks.items()}
    if len(set(signatures.values())) != 1 or signatures["hash"] != collections["hash"][
        "chunk_signature"
    ]:
        raise RuntimeError(f"live evaluation collections do not share canonical chunks: {signatures}")
    providers = {"hash": HashEmbeddingProvider(384), "jina": build_embedding_provider(settings)}
    variants = []
    for provider_name in ("hash", "jina"):
        for retriever_type in ("dense", "hybrid"):
            variants.append(
                evaluate_variant(
                    name=f"{provider_name}_structural_{retriever_type}",
                    provider=providers[provider_name],
                    retriever_type=retriever_type,
                    collection=collections[provider_name]["name"],
                    chunks=chunks[provider_name],
                    protocol=protocol,
                    public_to_raw=public_to_raw,
                    raw_to_public=raw_to_public,
                    production_raw_ids=production_raw_ids,
                    client=client,
                )
            )
    v1_reference = {}
    if V1_RESULTS.exists():
        v1 = json.loads(V1_RESULTS.read_text(encoding="utf-8"))
        v1_variants = {variant["name"]: variant for variant in v1["variants"]}
        v1_reference = {
            "hash_hybrid_mrr": v1_variants["hash_structural_hybrid"]["metrics"]["mrr"],
            "jina_hybrid_mrr": v1_variants["jina_structural_hybrid"]["metrics"]["mrr"],
        }
    payload = {
        "status": "COMPLETED",
        "generated_at": datetime.now(UTC).isoformat(),
        "protocol_version": "retrieval-query-v2",
        "corpus_version": corpus["manifest_version"],
        "index_manifest_version": index_manifest["index_manifest_version"],
        "random_seed": SEED,
        "scope_distribution": {
            scope: sum(record["retrieval_scope"] == scope for record in protocol)
            for scope in ("global", "paper", "multi_paper", "unanswerable")
        },
        "rewritten_query_count": sum(
            record["retrieval_query"] != record["original_question"] for record in protocol
        ),
        "filter_query_count": sum(bool(record["retrieval_filter"]["paper_ids"]) for record in protocol),
        "pending_query_review_count": sum(
            record["query_revision_review_status"] == "pending_human_review"
            for record in protocol
        ),
        "llm_called": False,
        "reranker_called": False,
        "v1_reference": v1_reference,
        "variants": variants,
    }
    JSON_OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(variants)
    write_report(payload)
    print(
        json.dumps(
            {
                "status": "COMPLETED",
                "scope_distribution": payload["scope_distribution"],
                "variants": [variant["name"] for variant in variants],
            }
        )
    )


if __name__ == "__main__":
    main()
