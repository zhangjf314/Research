import json
from pathlib import Path

from qdrant_client import QdrantClient

from paper_research.chunking.structural_chunker import StructuralChunker
from paper_research.indexing.embedding import HashEmbeddingProvider
from paper_research.indexing.vector_store import QdrantVectorStore
from paper_research.parsing.types import PaperBlock
from paper_research.retrieval.dense import DenseRetriever
from paper_research.retrieval.fusion import reciprocal_rank_fusion
from paper_research.retrieval.reranker import LexicalReranker
from paper_research.retrieval.sparse import BM25Retriever


def main() -> None:
    parsed_root = Path("data/reports/parsing-audit")
    chunks = []
    for blocks_path in sorted(parsed_root.glob("*/paper_blocks.jsonl")):
        blocks = [
            PaperBlock.model_validate(json.loads(line))
            for line in blocks_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        chunks.extend(StructuralChunker().chunk(blocks_path.parent.name, blocks))
    embedding = HashEmbeddingProvider(384)
    store = QdrantVectorStore(QdrantClient(":memory:"), "audit", 384)
    store.upsert(chunks, embedding.embed([chunk.chunk_text for chunk in chunks]))
    dense = DenseRetriever(embedding, store)
    sparse = BM25Retriever(chunks)
    reranker = LexicalReranker()
    queries = json.loads(
        Path("data/evaluation/retrieval_smoke_queries.json").read_text(encoding="utf-8")
    )
    hits = {"dense": 0, "sparse": 0, "hybrid": 0, "hybrid_rerank": 0}
    details = []
    for item in queries:
        dense_results = dense.retrieve(item["query"], top_k=20)
        sparse_results = sparse.retrieve(item["query"], top_k=20)
        fused = reciprocal_rank_fusion(dense_results, sparse_results)
        reranked = reranker.rerank(item["query"], fused, 5)
        variants = {
            "dense": [result.chunk.paper_id for result in dense_results[:5]],
            "sparse": [result.chunk.paper_id for result in sparse_results[:5]],
            "hybrid": [result.chunk.paper_id for result in fused[:5]],
            "hybrid_rerank": [result.chunk.paper_id for result in reranked],
        }
        for name, paper_ids in variants.items():
            hits[name] += item["paper_id"] in paper_ids
        details.append({**item, **{name: ids for name, ids in variants.items()}})
    report = [
        "# Retrieval Baseline Smoke Audit",
        "",
        f"- Queries: {len(queries)}",
        f"- Chunks: {len(chunks)}",
        "",
        "| Variant | Hit@5 |",
        "|---|---:|",
    ]
    report.extend(
        f"| {name} | {count / len(queries):.1%} |" for name, count in hits.items()
    )
    output = Path("data/reports/retrieval-baseline-audit.md")
    output.write_text("\n".join(report) + "\n", encoding="utf-8")
    Path("data/reports/retrieval-baseline-details.json").write_text(
        json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
