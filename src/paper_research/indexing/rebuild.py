import json
import uuid
from datetime import UTC, datetime

from qdrant_client import QdrantClient

from paper_research.chunking.structural_chunker import StructuralChunker
from paper_research.config import Settings
from paper_research.indexing.registry import IndexRegistry
from paper_research.indexing.service import IndexingService
from paper_research.indexing.vector_store import QdrantVectorStore
from paper_research.providers.factory import build_provider_bundle


class IndexRebuildService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
        self.registry = IndexRegistry(settings.data_dir / "index_registry.json")
        self.status_dir = settings.data_dir / "index-rebuilds"

    def rebuild(self) -> dict:
        rebuild_id = str(uuid.uuid4())
        logical = self.settings.active_collection
        physical = f"{logical}__{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"
        paths = sorted(self.settings.parsed_papers_dir.glob("*/paper_blocks.jsonl"))
        status = {
            "rebuild_id": rebuild_id,
            "status": "RUNNING",
            "logical_collection": logical,
            "staging_collection": physical,
            "total_papers": len(paths),
            "completed_papers": 0,
            "failed_papers": [],
            "started_at": datetime.now(UTC).isoformat(),
            **self.settings.provider_metadata,
        }
        self._write_status(rebuild_id, status)
        previous = self.registry.resolve(logical)
        try:
            providers = build_provider_bundle(self.settings)
            store = QdrantVectorStore(self.client, physical, self.settings.embedding_dimensions)
            store.ensure_collection()
            for blocks_path in paths:
                paper_id = blocks_path.parent.name
                chunks_path = (
                    blocks_path.parent
                    / f"paper_chunks.{self.settings.index_version}.jsonl"
                )
                try:
                    IndexingService(
                        StructuralChunker(
                            self.settings.chunk_max_tokens,
                            self.settings.chunk_overlap_tokens,
                        ),
                        providers.embedding,
                        store,
                    ).index(
                        paper_id,
                        blocks_path,
                        chunks_path,
                        metadata=self.settings.provider_metadata,
                    )
                    status["completed_papers"] += 1
                except Exception as exc:
                    status["failed_papers"].append(
                        {"paper_id": paper_id, "error": type(exc).__name__}
                    )
                    raise
                finally:
                    self._write_status(rebuild_id, status)
            self.registry.switch(
                logical,
                physical,
                {**self.settings.provider_metadata, "point_count": store.count()},
            )
            status.update(
                {
                    "status": "COMPLETED",
                    "active_collection": physical,
                    "previous_collection": previous,
                    "point_count": store.count(),
                    "completed_at": datetime.now(UTC).isoformat(),
                }
            )
        except Exception as exc:
            if self.client.collection_exists(physical):
                self.client.delete_collection(physical)
            status.update(
                {
                    "status": "ROLLED_BACK",
                    "active_collection": previous,
                    "error": f"{type(exc).__name__}: {exc}",
                    "completed_at": datetime.now(UTC).isoformat(),
                }
            )
        self._write_status(rebuild_id, status)
        return status

    def get_status(self, rebuild_id: str) -> dict | None:
        path = self.status_dir / f"{rebuild_id}.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def switch(self, collection: str) -> dict:
        if not self.client.collection_exists(collection):
            raise ValueError("collection does not exist")
        store = QdrantVectorStore(self.client, collection, self.settings.embedding_dimensions)
        store.ensure_collection()
        self.registry.switch(
            self.settings.active_collection,
            collection,
            {
                **self.settings.provider_metadata,
                "manual_switch": True,
                "point_count": store.count(),
            },
        )
        return {
            "logical_collection": self.settings.active_collection,
            "active_collection": collection,
        }

    def _write_status(self, rebuild_id: str, status: dict) -> None:
        self.status_dir.mkdir(parents=True, exist_ok=True)
        (self.status_dir / f"{rebuild_id}.json").write_text(
            json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8"
        )
