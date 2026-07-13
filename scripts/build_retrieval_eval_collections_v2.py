"""Create immutable 34-document Hash/Jina evaluation collections.

Jina vectors are copied from the completed real Production index.  Hash vectors
are recomputed over those exact canonical chunk payloads.  No source collection
is deleted, overwritten, or switched in the application registry.
"""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
from qdrant_client import QdrantClient, models

from paper_research.chunking.types import Chunk
from paper_research.config import Settings
from paper_research.indexing.embedding import HashEmbeddingProvider

CORPUS = Path("data/evaluation/production-corpus-v1.json")
OUTPUT = Path("data/evaluation/retrieval-index-v2.json")
BATCH_SIZE = 128


def load_registry(settings: Settings) -> dict:
    path = settings.data_dir / "index_registry.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    response = httpx.get("http://127.0.0.1:8000/api/v1/indexes", timeout=10)
    response.raise_for_status()
    return response.json()


def active_collection(registry: dict, logical: str) -> str:
    physical = registry.get("defaults", {}).get(logical)
    if not physical:
        raise RuntimeError(f"logical collection is not activated: {logical}")
    return str(physical)


def scroll_points(
    client: QdrantClient, collection: str, paper_ids: list[str]
) -> list[models.Record]:
    points = []
    offset = None
    query_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="paper_id", match=models.MatchAny(any=paper_ids)
            )
        ]
    )
    while True:
        batch, offset = client.scroll(
            collection_name=collection,
            scroll_filter=query_filter,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        points.extend(batch)
        if offset is None:
            return points


def chunk_signature(chunks: list[Chunk]) -> str:
    digest = hashlib.sha256()
    for chunk in sorted(chunks, key=lambda item: item.chunk_id):
        digest.update(
            json.dumps(chunk.model_dump(), sort_keys=True, ensure_ascii=False).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def create_collection(client: QdrantClient, name: str, dimensions: int) -> None:
    if client.collection_exists(name):
        raise RuntimeError(f"refusing to overwrite existing evaluation collection: {name}")
    client.create_collection(
        collection_name=name,
        vectors_config=models.VectorParams(size=dimensions, distance=models.Distance.COSINE),
    )


def upsert_batches(
    client: QdrantClient, collection: str, points: list[models.PointStruct]
) -> None:
    for start in range(0, len(points), BATCH_SIZE):
        client.upsert(
            collection_name=collection,
            points=points[start : start + BATCH_SIZE],
            wait=True,
        )


def main() -> None:
    settings = Settings()
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    included = [paper for paper in corpus["papers"] if paper["included_in_production"]]
    paper_ids = [paper["database_id"] for paper in included]
    if len(paper_ids) != 34 or len(set(paper_ids)) != 34:
        raise RuntimeError("production corpus manifest must contain 34 unique database IDs")

    registry = load_registry(settings)
    source_jina = active_collection(registry, settings.production_collection)
    source_hash = active_collection(registry, settings.baseline_collection)
    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        check_compatibility=False,
    )
    source_points = scroll_points(client, source_jina, paper_ids)
    chunks = [Chunk.model_validate(point.payload) for point in source_points]
    represented = {chunk.paper_id for chunk in chunks}
    if represented != set(paper_ids):
        missing = sorted(set(paper_ids) - represented)
        raise RuntimeError(f"Jina Production index is missing manifest papers: {missing}")
    if any(point.vector is None or not isinstance(point.vector, list) for point in source_points):
        raise RuntimeError("source Jina points must contain single unnamed vectors")

    timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    hash_collection = f"papers_hash_eval34_v2__{timestamp}"
    jina_collection = f"papers_jina_eval34_v2__{timestamp}"
    create_collection(client, hash_collection, 384)
    create_collection(client, jina_collection, 1024)

    jina_points = [
        models.PointStruct(id=point.id, vector=point.vector, payload=point.payload)
        for point in source_points
    ]
    upsert_batches(client, jina_collection, jina_points)

    hash_provider = HashEmbeddingProvider(384)
    hash_vectors = hash_provider.embed_documents([chunk.chunk_text for chunk in chunks])
    hash_points = [
        models.PointStruct(id=point.id, vector=vector, payload=point.payload)
        for point, vector in zip(source_points, hash_vectors, strict=True)
    ]
    upsert_batches(client, hash_collection, hash_points)

    hash_count = int(client.count(hash_collection, exact=True).count)
    jina_count = int(client.count(jina_collection, exact=True).count)
    if hash_count != jina_count or hash_count != len(chunks):
        raise RuntimeError(
            f"evaluation collection count mismatch: hash={hash_count}, jina={jina_count}, "
            f"chunks={len(chunks)}"
        )
    signature = chunk_signature(chunks)
    output = {
        "index_manifest_version": "retrieval-index-v2",
        "created_at": datetime.now(UTC).isoformat(),
        "construction_method": (
            "Filtered 34-document snapshot of the completed real Jina Production index; "
            "Jina vectors copied unchanged and Hash vectors recomputed over identical payloads"
        ),
        "source_collections": {"hash": source_hash, "jina": source_jina},
        "collections": {
            "hash": {
                "name": hash_collection,
                "provider": "hash",
                "model": "hash-v1",
                "dimension": 384,
                "paper_count": len(represented),
                "point_count": hash_count,
                "chunk_signature": signature,
            },
            "jina": {
                "name": jina_collection,
                "provider": "jina",
                "model": "jina-embeddings-v5-text-small",
                "revision": settings.embedding_revision,
                "dimension": 1024,
                "paper_count": len(represented),
                "point_count": jina_count,
                "chunk_signature": signature,
                "document_task": "retrieval.passage",
                "query_task": "retrieval.query",
            },
        },
        "registry_switched": False,
        "source_collections_modified": False,
    }
    OUTPUT.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
