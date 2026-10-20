import uuid

from qdrant_client import QdrantClient, models

from paper_research.chunking.types import Chunk
from paper_research.indexing.registry import IndexRegistry, CorpusLease, CorpusTrustError


class QdrantVectorStore:
    def __init__(
        self,
        client: QdrantClient,
        collection: str,
        dimensions: int,
        *,
        chunk_model: type[Chunk] = Chunk,
        registry: IndexRegistry | None = None,
        capability: CorpusLease | None = None,
    ) -> None:
        self.client = client
        self.collection = collection
        self.dimensions = dimensions
        self.chunk_model = chunk_model
        self.registry = registry
        self.capability = capability

    def _require_write(self, states: set[str]) -> None:
        if self.registry is None:
            raise CorpusTrustError("CORPUS_WRITE_NOT_AUTHORIZED")
        self.registry.require_mutation(self.collection, self.capability, states)

    def create_exclusive(self) -> None:
        self._require_write({"PREPARED"})
        if self.client.collection_exists(self.collection):
            raise CorpusTrustError("CORPUS_AUTHORITY_NOT_PROVABLE")
        try:
            acknowledged = self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(size=self.dimensions, distance=models.Distance.COSINE),
            )
        except Exception as exc:
            raise CorpusTrustError("CORPUS_AUTHORITY_NOT_PROVABLE", ["unknown create ACK"]) from exc
        if acknowledged is not True:
            raise CorpusTrustError("CORPUS_AUTHORITY_NOT_PROVABLE", ["unknown create ACK"])
        if self.registry.physical_points(self.client, self.collection):
            raise CorpusTrustError("CORPUS_PHYSICAL_MEMBERSHIP_MISMATCH", ["fresh collection not empty"])

    def ensure_collection(self) -> None:
        self._require_write({"BUILDING"})
        if self.client.collection_exists(self.collection):
            info = self.client.get_collection(self.collection)
            vectors = info.config.params.vectors
            actual = vectors.size if isinstance(vectors, models.VectorParams) else None
            if actual is not None and actual != self.dimensions:
                raise ValueError(
                    f"collection {self.collection} has dimension {actual}; "
                    f"provider requires {self.dimensions}"
                )
        else:
            # A vanished generation must never be silently recreated under the same name.
            raise CorpusTrustError("CORPUS_PHYSICAL_MEMBERSHIP_MISMATCH", ["collection missing"])

    def count(self) -> int:
        if not self.client.collection_exists(self.collection):
            return 0
        return int(self.client.count(self.collection, exact=True).count)

    def count_by_paper_id(self, paper_id: str) -> int:
        if not self.client.collection_exists(self.collection):
            return 0
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="paper_id",
                    match=models.MatchValue(value=paper_id),
                )
            ]
        )
        return int(
            self.client.count(
                collection_name=self.collection,
                count_filter=query_filter,
                exact=True,
            ).count
        )

    def delete_by_paper_id(self, paper_id: str) -> int:
        if self.registry is not None:
            self.registry.refuse_direct_mutation()
        raise CorpusTrustError("CORPUS_WRITE_NOT_AUTHORIZED")

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        self._require_write({"BUILDING"})
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        self.ensure_collection()
        inventory = self.registry.inventory(self.capability.generation_id)
        bindings = {(r["paper_id"], r["chunk_id"]): r for r in inventory["expected_chunks"]}
        from paper_research.indexing.identity import trust_digest, point_identity
        point_ids = [point_identity(c.chunk_id) for c in chunks]
        if len(point_ids) != len(set(point_ids)) or any(p in self.capability.written_points for p in point_ids):
            raise CorpusTrustError("CANONICAL_SOURCE_IDENTITY_CONFLICT")
        for chunk in chunks:
            row = bindings.get((chunk.paper_id, chunk.chunk_id))
            if row is None or row["canonical_payload_sha256"] != trust_digest(chunk.model_dump()):
                raise CorpusTrustError("CORPUS_MEMBERSHIP_MISMATCH", ["upsert not expected"])
        self.capability.written_points.update(point_ids)
        points = [
            models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.chunk_id)),
                vector=vector,
                payload=chunk.model_dump(),
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        if points:
            self.client.upsert(self.collection, points=points, wait=True)

    def search(
        self,
        vector: list[float],
        *,
        paper_ids: list[str] | None = None,
        limit: int = 5,
        score_threshold: float | None = None,
    ) -> list[tuple[Chunk, float]]:
        query_filter = None
        if paper_ids:
            query_filter = models.Filter(
                must=[models.FieldCondition(key="paper_id", match=models.MatchAny(any=paper_ids))]
            )
        response = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            query_filter=query_filter,
            limit=limit,
            score_threshold=score_threshold,
            with_payload=True,
        )
        return [
            (self.chunk_model.model_validate(point.payload), point.score)
            for point in response.points
        ]
