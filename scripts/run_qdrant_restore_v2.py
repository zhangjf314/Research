from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient, models

from paper_research.config import Settings

INDEX_MANIFEST = Path("data/evaluation/retrieval-index-v2.json")
OUTPUT_JSON = Path("data/evaluation/qdrant-backup-restore-v2.json")
OUTPUT_MD = Path("docs/qdrant-backup-restore-audit-v2.md")
TOP_K = 10
QUERY_VECTOR_COUNT = 5


def vector_size(vectors_config: Any) -> int | None:
    if hasattr(vectors_config, "size"):
        return int(vectors_config.size)
    if isinstance(vectors_config, dict):
        first = next(iter(vectors_config.values()), None)
        return vector_size(first)
    return None


def distance_metric(vectors_config: Any) -> str | None:
    if hasattr(vectors_config, "distance"):
        return str(vectors_config.distance.value)
    if isinstance(vectors_config, dict):
        first = next(iter(vectors_config.values()), None)
        return distance_metric(first)
    return None


def payload_key_schema(payloads: Iterable[dict[str, Any]]) -> dict[str, str]:
    schema: dict[str, str] = {}
    for payload in payloads:
        for key, value in payload.items():
            schema.setdefault(key, type(value).__name__)
    return dict(sorted(schema.items()))


def search_ids_and_scores(
    client: QdrantClient,
    collection: str,
    query: list[float],
) -> list[dict[str, Any]]:
    response = client.query_points(
        collection_name=collection,
        query=query,
        limit=TOP_K,
        with_payload=False,
        with_vectors=False,
    )
    return [
        {"id": str(point.id), "score": float(point.score)}
        for point in response.points
    ]


def finite_score_close(left: float, right: float, tolerance: float = 1e-6) -> bool:
    return abs(left - right) <= tolerance


def compare_topk(
    source_results: list[dict[str, Any]],
    restored_results: list[dict[str, Any]],
) -> dict[str, Any]:
    source_ids = [item["id"] for item in source_results]
    restored_ids = [item["id"] for item in restored_results]
    same_order = source_ids == restored_ids
    same_set = set(source_ids) == set(restored_ids)
    score_close = all(
        finite_score_close(left["score"], right["score"])
        for left, right in zip(source_results, restored_results, strict=False)
    )
    return {
        "same_order": same_order,
        "same_set": same_set,
        "score_close": score_close,
        "source_ids": source_ids,
        "restored_ids": restored_ids,
    }


def main() -> None:
    started = datetime.now(UTC)
    stamp = started.strftime("%Y%m%d%H%M%S")
    settings = Settings()
    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        check_compatibility=False,
    )
    manifest = json.loads(INDEX_MANIFEST.read_text(encoding="utf-8"))
    source_collection = manifest["collections"]["jina"]["name"]
    restore_collection = f"{source_collection}__restore_v2_{stamp}"

    source_info_before = client.get_collection(source_collection)
    source_count_before = client.count(source_collection, exact=True).count
    source_vector_config = source_info_before.config.params.vectors
    source_dimension = vector_size(source_vector_config)
    source_distance = distance_metric(source_vector_config)

    records, _ = client.scroll(
        collection_name=source_collection,
        limit=QUERY_VECTOR_COUNT,
        with_payload=True,
        with_vectors=True,
    )
    query_vectors: list[list[float]] = []
    source_payloads: list[dict[str, Any]] = []
    for record in records:
        if isinstance(record.vector, dict):
            vector = next(iter(record.vector.values()))
        else:
            vector = record.vector
        query_vectors.append([float(value) for value in vector])
        source_payloads.append(record.payload or {})

    snapshot = client.create_snapshot(source_collection, wait=True)
    if snapshot is None:
        raise RuntimeError("Qdrant did not return a snapshot description.")
    snapshot_name = snapshot.name
    snapshot_location = (
        f"{settings.qdrant_url.rstrip('/')}/collections/"
        f"{source_collection}/snapshots/{snapshot_name}"
    )
    client.recover_snapshot(
        collection_name=restore_collection,
        location=snapshot_location,
        priority=models.SnapshotPriority.SNAPSHOT,
        wait=True,
    )

    restored_info = client.get_collection(restore_collection)
    restored_count = client.count(restore_collection, exact=True).count
    restored_vector_config = restored_info.config.params.vectors
    restored_dimension = vector_size(restored_vector_config)
    restored_distance = distance_metric(restored_vector_config)
    restored_records, _ = client.scroll(
        collection_name=restore_collection,
        limit=QUERY_VECTOR_COUNT,
        with_payload=True,
        with_vectors=False,
    )
    restored_payloads = [record.payload or {} for record in restored_records]

    topk_comparisons = []
    for query in query_vectors:
        topk_comparisons.append(
            compare_topk(
                search_ids_and_scores(client, source_collection, query),
                search_ids_and_scores(client, restore_collection, query),
            )
        )

    source_count_after = client.count(source_collection, exact=True).count
    point_count_match = source_count_before == restored_count
    vector_dimension_match = source_dimension == restored_dimension
    distance_metric_match = source_distance == restored_distance
    payload_schema_match = payload_key_schema(source_payloads) == payload_key_schema(
        restored_payloads
    )
    retrieval_topk_equivalence = all(
        item["same_set"] and item["score_close"] for item in topk_comparisons
    )
    production_collection_unchanged = source_count_before == source_count_after
    gate = (
        "PASSED"
        if point_count_match
        and vector_dimension_match
        and distance_metric_match
        and payload_schema_match
        and retrieval_topk_equivalence
        and production_collection_unchanged
        else "FAILED"
    )
    payload = {
        "schema_version": "qdrant-backup-restore-v2",
        "started_at": started.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
        "gate": gate,
        "source_collection": source_collection,
        "restore_collection": restore_collection,
        "snapshot_created": True,
        "snapshot_name": snapshot_name,
        "snapshot_location": snapshot_location,
        "restore_collection_created": True,
        "source_point_count_before": source_count_before,
        "source_point_count_after": source_count_after,
        "restored_point_count": restored_count,
        "point_count_match": point_count_match,
        "source_dimension": source_dimension,
        "restored_dimension": restored_dimension,
        "vector_dimension_match": vector_dimension_match,
        "source_distance": source_distance,
        "restored_distance": restored_distance,
        "distance_metric_match": distance_metric_match,
        "source_payload_schema": payload_key_schema(source_payloads),
        "restored_payload_schema": payload_key_schema(restored_payloads),
        "payload_schema_match": payload_schema_match,
        "top_k": TOP_K,
        "query_vector_count": len(query_vectors),
        "topk_comparisons": topk_comparisons,
        "retrieval_topk_equivalence": "passed"
        if retrieval_topk_equivalence
        else "failed",
        "production_collection_unchanged": production_collection_unchanged,
        "restore_collection_retained_for_audit": True,
    }
    OUTPUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Qdrant Backup Restore Audit v2",
        "",
        f"- Gate: `{gate}`",
        f"- Source collection: `{source_collection}`",
        f"- Restore collection: `{restore_collection}`",
        f"- Snapshot: `{snapshot_name}`",
        f"- Point count match: `{point_count_match}`",
        f"- Vector dimension match: `{vector_dimension_match}`",
        f"- Distance metric match: `{distance_metric_match}`",
        f"- Payload schema match: `{payload_schema_match}`",
        f"- Retrieval Top-{TOP_K} equivalence: `{retrieval_topk_equivalence}`",
        f"- Production collection unchanged: `{production_collection_unchanged}`",
        "- Restore collection retained for audit; production collection was not overwritten.",
    ]
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "gate": gate,
                "source_collection": source_collection,
                "restore_collection": restore_collection,
                "topk_equivalence": retrieval_topk_equivalence,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
