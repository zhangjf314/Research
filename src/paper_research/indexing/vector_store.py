import uuid

from qdrant_client import QdrantClient, models

from paper_research.chunking.types import Chunk


class QdrantVectorStore:
    def __init__(self, client: QdrantClient, collection: str, dimensions: int) -> None:
        self.client = client
        self.collection = collection
        self.dimensions = dimensions

    def ensure_collection(self) -> None:
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
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(
                    size=self.dimensions, distance=models.Distance.COSINE
                ),
            )

    def count(self) -> int:
        if not self.client.collection_exists(self.collection):
            return 0
        return int(self.client.count(self.collection, exact=True).count)

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        self.ensure_collection()
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
                must=[
                    models.FieldCondition(
                        key="paper_id", match=models.MatchAny(any=paper_ids)
                    )
                ]
            )
        response = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            query_filter=query_filter,
            limit=limit,
            score_threshold=score_threshold,
            with_payload=True,
        )
        return [(Chunk.model_validate(point.payload), point.score) for point in response.points]
