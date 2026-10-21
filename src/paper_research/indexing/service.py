import json
import hashlib
import uuid
from datetime import UTC, datetime
from pathlib import Path

from paper_research.chunking.structural_chunker import StructuralChunker
from paper_research.chunking.types import Chunk
from paper_research.indexing.embedding import EmbeddingProvider
from paper_research.indexing.registry import (
    IndexIdentityError,
    IndexRegistry,
    runtime_index_identity,
    CorpusLease,
    CorpusTrustError,
)
from paper_research.indexing.identity import (
    trust_digest, chunk_file_bytes, point_identity, vector_f32, normalized_f32,
)
from paper_research.indexing.vector_store import QdrantVectorStore
from paper_research.parsing.types import PaperBlock
from paper_research.retrieval.contextual import ContextualChunk, contextualize


class IndexingService:
    def __init__(
        self,
        chunker: StructuralChunker,
        embedding: EmbeddingProvider,
        vector_store: QdrantVectorStore,
    ) -> None:
        self.chunker = chunker
        self.embedding = embedding
        self.vector_store = vector_store

    def prepare(self, registry: IndexRegistry, capability: CorpusLease) -> dict:
        auth = registry.require_mutation(self.vector_store.collection, capability, {"ADMITTING"})
        intent, _ = registry.intent(auth["admission_id"])
        commitment = registry.read_record(registry.admission_path(auth["admission_id"], "commitment.json"),
                                          "admission_commitment", "admission_digest")
        if (commitment["intent_digest"] != intent["intent_digest"]
                or [r["source_ref"] for r in commitment["documents"]]
                    != [r["source_ref"] for r in intent["sources"]]
                or [r["pdf_sha256"] for r in commitment["documents"]]
                    != [r["pdf_sha256"] for r in intent["sources"]]):
            raise CorpusTrustError("CORPUS_INVENTORY_INVALID", ["admission closure"])
        registry.verify_documents({"commitment": commitment})
        documents = sorted(commitment["documents"], key=lambda r: r["paper_id"])
        chunks, membership, digests = [], [], []
        for document in documents:
            paper = document["paper_id"]
            blocks_path = registry.protected_path(f"parsed/{paper}/paper_blocks.jsonl")
            manifest_path = registry.protected_path(f"parsed/{paper}/parse_manifest.json")
            raw = registry.file_bytes(blocks_path, "CORPUS_AUTHORITY_NOT_PROVABLE")
            manifest = registry.file_bytes(manifest_path, "CORPUS_AUTHORITY_NOT_PROVABLE")
            if (hashlib.sha256(raw).hexdigest() != document["parse_blocks_sha256"]
                    or hashlib.sha256(manifest).hexdigest() != document["parse_manifest_sha256"]):
                raise CorpusTrustError("CORPUS_AUTHORITY_NOT_PROVABLE", ["parser receipt"])
            blocks = [PaperBlock.model_validate(json.loads(line)) for line in raw.splitlines() if line]
            prepared = self.chunker.chunk(paper, blocks)
            if not prepared:
                raise CorpusTrustError("CORPUS_AUTHORITY_NOT_PROVABLE", ["zero chunks"])
            raw_chunks = chunk_file_bytes(prepared)
            registry.durable_bytes(registry.generation_path(auth["generation_id"],
                f"prepared/{paper}/chunks.jsonl"), raw_chunks, immutable=True)
            digest = hashlib.sha256(raw_chunks).hexdigest()
            membership.append({"paper_id": paper, "chunk_ids": sorted(c.chunk_id for c in prepared),
                "chunks_relative_path": f"parsed/{paper}/paper_chunks.{auth['index_identity']['index_version']}.jsonl",
                "chunks_file_sha256": digest})
            digests.append([paper, digest])
            chunks.extend(prepared)
        # This check precedes strict inventory serialization, including same-document repeats.
        raw_ids = [c.chunk_id for c in chunks]
        if len(raw_ids) != len(set(raw_ids)):
            raise CorpusTrustError("CANONICAL_SOURCE_IDENTITY_CONFLICT")
        inventory = {"schema_version": "authoritative-corpus-inventory-v1",
            "inventory_id": str(uuid.uuid4()), "generation_id": auth["generation_id"],
            "index_identity": auth["index_identity"], "source_admission_id": auth["admission_id"],
            "source_admission_digest": commitment["admission_digest"],
            "expected_document_ids": [d["paper_id"] for d in documents],
            "expected_document_count": len(documents), "expected_chunk_count": len(chunks),
            "expected_chunks": [{"paper_id": c.paper_id, "chunk_id": c.chunk_id,
                "canonical_payload_sha256": trust_digest(c.model_dump()), "point_id": point_identity(c.chunk_id)}
                for c in sorted(chunks, key=lambda c: (c.paper_id, c.chunk_id))],
            "document_to_chunk_membership": membership,
            "raw_chunk_id_collision_policy": "REJECT_ANY_DUPLICATE_BARE_CHUNK_ID_BEFORE_EMBED_OR_UPSERT",
            "prepared_records_digest": trust_digest(digests),
            "parser_receipt_digest": trust_digest(commitment["documents"]),
            "chunker_config": {"max_tokens": self.chunker.max_tokens, "overlap_tokens": self.chunker.overlap_tokens},
            "created_at": datetime.now(UTC).isoformat(), "inventory_digest": "0" * 64}
        registry.validate_membership(inventory, commitment)
        inventory = registry.save_record(registry.generation_path(auth["generation_id"], "inventory.json"),
                                          inventory, "inventory", "inventory_digest", immutable=True)
        registry.verify_prepared(inventory)
        registry.transition(capability, "PREPARED", receipts={"inventory_digest": inventory["inventory_digest"]})
        return inventory

    def build_prepared(self, registry: IndexRegistry, capability: CorpusLease, metadata: dict) -> None:
        auth = registry.require_mutation(self.vector_store.collection, capability, {"BUILDING"})
        inventory = registry.inventory(auth["generation_id"])
        bundle = registry.authority_bundle(auth["generation_id"], inventory=inventory)
        registry.validate_membership(inventory, bundle["commitment"])
        registry.verify_prepared(inventory)
        chunks = []
        for doc in inventory["document_to_chunk_membership"]:
            raw = registry.file_bytes(registry.generation_path(auth["generation_id"],
                f"prepared/{doc['paper_id']}/chunks.jsonl"), "CORPUS_AUTHORITY_NOT_PROVABLE")
            chunks.extend(Chunk.model_validate(r) for r in registry.chunk_records(raw))
        vectors = self.embedding.embed_documents([c.chunk_text for c in chunks])
        if len(vectors) != len(chunks):
            raise CorpusTrustError("CORPUS_PHYSICAL_CONTENT_MISMATCH", ["document embedding length"])
        dimension = inventory["index_identity"]["embedding_dimension"]
        rows = [{"point_id": point_identity(c.chunk_id), "input_f32": vector_f32(v, dimension),
                 "normalized_f32": normalized_f32(v, dimension)}
                for c, v in zip(chunks, vectors, strict=True)]
        rows.sort(key=lambda r: r["point_id"])
        raw = "".join(json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in rows).encode("utf-8")
        relative = f"corpus-trust/generations/{auth['generation_id']}/vectors.jsonl"
        registry.durable_bytes(registry.protected_path(relative), raw, immutable=True)
        receipt = registry.save_record(registry.generation_path(auth["generation_id"], "vectors.json"), {
            "schema_version": "corpus-build-vector-receipt-v1", "generation_id": auth["generation_id"],
            "inventory_digest": inventory["inventory_digest"], "vectors_relative_path": relative,
            "vectors_file_sha256": hashlib.sha256(raw).hexdigest(), "point_count": len(rows),
            "receipt_digest": "0" * 64}, "build_vector_receipt", "receipt_digest", immutable=True)
        journal = registry.journal(auth["generation_id"])
        journal.update(build_vector_receipt_digest=receipt["receipt_digest"], updated_at=datetime.now(UTC).isoformat())
        registry.journal(auth["generation_id"], value=journal)
        registry.save_record(registry.generation_path(auth["generation_id"], "build.json"), journal, "build_journal")
        self.vector_store.upsert(chunks, vectors)
        for doc in inventory["document_to_chunk_membership"]:
            path = registry.protected_path(doc["chunks_relative_path"])
            prepared = registry.file_bytes(registry.generation_path(auth["generation_id"],
                f"prepared/{doc['paper_id']}/chunks.jsonl"), "CORPUS_AUTHORITY_NOT_PROVABLE")
            registry.durable_bytes(path, prepared, immutable=True)
            registry.durable_bytes(path.with_name(f"{path.stem}_manifest.json"), json.dumps({
                **metadata, "collection": self.vector_store.collection, "paper_id": doc["paper_id"],
                "chunk_count": len(doc["chunk_ids"]), "chunks_path": str(path)},
                ensure_ascii=False, indent=2).encode("utf-8"), immutable=True)

    def index(
        self,
        paper_id: str,
        blocks_path: Path,
        chunks_path: Path,
        *,
        metadata: dict[str, object] | None = None,
        registry: IndexRegistry | None = None,
    ) -> list[Chunk]:
        if registry is not None:
            registry.refuse_direct_mutation()
        raise CorpusTrustError("CORPUS_WRITE_NOT_AUTHORIZED")
