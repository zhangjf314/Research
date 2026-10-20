import json
import logging
import re
import time
import uuid
from datetime import UTC, datetime

from qdrant_client import QdrantClient

from paper_research.chunking.structural_chunker import StructuralChunker
from paper_research.config import Settings
from paper_research.indexing.registry import (IndexRegistry, runtime_index_identity, IndexIdentityError, CorpusTrustError)
from paper_research.indexing.identity import trust_digest
from paper_research.indexing.service import IndexingService
from paper_research.indexing.vector_store import QdrantVectorStore
from paper_research.providers.factory import build_embedding_provider


logger = logging.getLogger(__name__)


class IndexRebuildService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
        self.registry = IndexRegistry(settings.data_dir / "index_registry.json")
        self.status_dir = settings.data_dir / "index-rebuilds"

    def rebuild(self, admission_id: str) -> dict:
        started = time.monotonic()
        intent, auth = self.registry.intent(self.registry.uuid_text(admission_id))
        generation, physical = auth["generation_id"], auth["collection_name"]
        with self.registry.lease("build", generation) as capability:
            if self.registry.journal(generation)["state"] != "ADMITTING":
                raise CorpusTrustError("CORPUS_GENERATION_IMMUTABLE")
            if self.registry.read().get("defaults", {}).get(auth["logical_collection"]) is not None:
                raise CorpusTrustError("CORPUS_GENERATION_IMMUTABLE")
            try:
                embedding = build_embedding_provider(self.settings)
                expected = runtime_index_identity(self.settings.provider_metadata, physical)
                actual = runtime_index_identity({**self.settings.provider_metadata,
                    "embedding_provider": embedding.provider_name, "embedding_model": embedding.model_name,
                    "embedding_revision": embedding.revision, "embedding_dimension": embedding.dimensions}, physical)
                if actual != expected:
                    raise IndexIdentityError("INDEX_IDENTITY_CONFLICT", status_code=409)
                if (expected != auth["index_identity"] or self.settings.active_collection != auth["logical_collection"]
                        or self.settings.provider_metadata != auth["runtime_metadata"]):
                    raise CorpusTrustError("CORPUS_INVENTORY_BINDING_CONFLICT")
                store = QdrantVectorStore(self.client, physical, embedding.dimensions,
                                          registry=self.registry, capability=capability)
                service = IndexingService(StructuralChunker(self.settings.chunk_max_tokens,
                    self.settings.chunk_overlap_tokens), embedding, store)
                inventory = service.prepare(self.registry, capability)
                store.create_exclusive()
                receipt = self.registry.save_record(self.registry.generation_path(generation, "create.json"), {
                    "generation_id": generation, "collection_name": physical, "existed_before": False,
                    "creation_acknowledged": True, "observed_initial_point_count": 0,
                    "initial_full_scroll_empty": True, "exclusive_owner_id": str(capability.pid),
                    "created_at": datetime.now(UTC).isoformat(), "receipt_digest": "0" * 64},
                    "create_receipt", "receipt_digest", immutable=True)
                self.registry.bind_identity(physical, actual, self.client, capability=capability)
                self.registry.transition(capability, "BUILDING", receipts={"create_receipt_digest": receipt["receipt_digest"]})
                service.build_prepared(self.registry, capability, {**self.settings.provider_metadata, "collection": physical})
                self.registry.transition(capability, "VERIFYING")
                bundle = self.registry.authority_bundle(generation)
                _, observed_canonical, points = self.registry.verify_complete(bundle, self.client, sealed=False)
                # Source auxiliaries are required at seal; query membership does not confuse them with inventory.
                import hashlib
                for doc in bundle["commitment"]["documents"]:
                    folder = self.settings.parsed_papers_dir / doc["paper_id"]
                    for name, key in [("paper_blocks.jsonl", "parse_blocks_sha256"),
                                      ("parse_manifest.json", "parse_manifest_sha256")]:
                        if hashlib.sha256((folder / name).read_bytes()).hexdigest() != doc[key]:
                            raise CorpusTrustError("CORPUS_AUTHORITY_NOT_PROVABLE", ["seal source receipt"])
                    manifest = folder / f"paper_chunks.{self.settings.index_version}_manifest.json"
                    fields = json.loads(manifest.read_text("utf-8"))
                    expected_doc = next(d for d in inventory["document_to_chunk_membership"] if d["paper_id"] == doc["paper_id"])
                    if (fields["paper_id"] != doc["paper_id"] or fields["collection"] != physical
                            or fields["chunk_count"] != len(expected_doc["chunk_ids"])
                            or fields["index_version"] != self.settings.index_version
                            or fields["chunks_path"] != str(self.registry.protected_path(expected_doc["chunks_relative_path"]))):
                        raise CorpusTrustError("CORPUS_INVENTORY_BINDING_CONFLICT", ["D1 manifest"])
                from paper_research.db import SessionLocal
                from paper_research.models.paper import Paper, PaperStatus
                with SessionLocal() as session:
                    for row in session.query(Paper).filter(Paper.id.in_([uuid.UUID(k) for k in inventory["expected_document_ids"]])).all():
                        row.parse_status = PaperStatus.ready
                        row.index_status = "READY"
                    session.commit()
                vectors = self.registry.read_record(self.registry.generation_path(generation, "vectors.json"),
                    "build_vector_receipt", "receipt_digest")
                now = datetime.now(UTC).isoformat()
                seal = self.registry.save_record(self.registry.generation_path(generation, "seal.json"), {
                    "schema_version": "verified-index-generation-seal-v1", "generation_id": generation,
                    "inventory_id": inventory["inventory_id"], "inventory_digest": inventory["inventory_digest"],
                    "index_identity": inventory["index_identity"], "create_receipt_digest": receipt["receipt_digest"],
                    "build_vector_receipt_digest": vectors["receipt_digest"], "observed_canonical_digest": observed_canonical,
                    "observed_physical_points": points, "observed_physical_digest": trust_digest(points),
                    "verification_result": "EXACT_MEMBERSHIP_PAYLOAD_AND_VECTOR_BINDING_VERIFIED",
                    "verified_at": now, "sealed_at": now, "seal_digest": "0" * 64}, "seal", "seal_digest", immutable=True)
                self.registry.transition(capability, "SEALED", receipts={"seal_digest": seal["seal_digest"]})
                binding = self.registry.publish(generation, self.client, capability)
                status = {"rebuild_id": generation, "generation_id": generation, "status": "ACTIVE",
                    "inventory_digest": inventory["inventory_digest"], "seal_digest": seal["seal_digest"],
                    "registry_binding": binding, "build_duration_seconds": time.monotonic() - started}
                self._write_status(generation, status)
                return status
            except Exception as exc:
                logger.exception("Controlled generation build failed")
                code = exc.code if isinstance(exc, IndexIdentityError) else "CORPUS_INTEGRITY_UNAVAILABLE"
                try:
                    current = self.registry.journal(generation)["state"]
                    if current not in {"SEALED", "FAILED", "ABORTED"}:
                        self.registry.transition(capability, "ABORTED" if code == "CORPUS_AUTHORITY_NOT_PROVABLE" else "FAILED", error_code=code)
                    self._write_status(generation, {"rebuild_id": generation, "generation_id": generation,
                        "status": current if current == "SEALED" else "FAILED",
                        "error_code": code, "error_type": type(exc).__name__, "build_duration_seconds": time.monotonic() - started})
                except Exception:
                    logger.exception("Generation failure receipt could not be persisted")
                if isinstance(exc, IndexIdentityError):
                    raise
                raise CorpusTrustError(code) from exc

    def _physical_collection_name(self) -> str:
        provider = _slug(self.settings.embedding_provider)
        model = _slug(self.settings.embedding_model).removeprefix(f"{provider}_embeddings_")
        index_version = _slug(self.settings.index_version)
        timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
        return (
            f"papers_{provider}_{model}_{self.settings.embedding_dimensions}"
            f"__{index_version}__{timestamp}"
        )

    def get_status(self, rebuild_id: str) -> dict | None:
        path = self.status_dir / f"{rebuild_id}.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def switch(self, collection: str) -> dict:
        auth = self.registry.authorization()
        if collection != auth["collection_name"]:
            raise CorpusTrustError("CORPUS_WRITE_NOT_AUTHORIZED")
        with self.registry.lease("publish", auth["generation_id"]) as capability:
            return self.registry.publish(auth["generation_id"], self.client, capability)

    def _write_status(self, rebuild_id: str, status: dict) -> None:
        self.status_dir.mkdir(parents=True, exist_ok=True)
        (self.status_dir / f"{rebuild_id}.json").write_text(
            json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
