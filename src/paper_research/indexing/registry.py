import json
import logging
import threading
import os
import hashlib
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from qdrant_client import QdrantClient, models

from paper_research.indexing.identity import (
    canonical_digest, trust_digest, trust_json_loads, validate_trust_record,
    point_identity, vector_f32_digest, normalized_f32,
)

IDENTITY_SCHEMA = "runtime-index-identity-v1"
IDENTITY_FIELDS = (
    "embedding_provider", "embedding_model", "embedding_revision",
    "embedding_dimension", "index_version", "collection_name", "distance_metric",
)
logger = logging.getLogger(__name__)


def _io_path(path: Path) -> Path:
    """Native extended-path spelling; logical paths and identity bytes stay unchanged."""
    if os.name == "nt":
        absolute = str(path.resolve())
        if not absolute.startswith("\\\\?\\"):
            return Path("\\\\?\\" + absolute)
    return path


class IndexIdentityError(RuntimeError):
    def __init__(self, code: str, fields: list[str] | None = None, *, status_code: int = 503):
        self.code = code
        self.fields = fields or []
        self.status_code = status_code
        super().__init__(code + (": " + ", ".join(self.fields) if self.fields else ""))
        logger.warning("Index identity rejected: %s", self)


class CorpusTrustError(IndexIdentityError):
    """Propagates through the existing Agent identity refusal boundary."""


class CorpusLease:
    def __init__(self, registry, role: str, generation_id: str | None) -> None:
        self.registry = registry
        self.role = role
        self.generation_id = generation_id
        self.pid = os.getpid()
        self.handle = None
        self.written_points: set[str] = set()

    def require(self, registry, roles) -> None:
        if (self.handle is None or self.handle.closed or self.pid != os.getpid()
                or self.registry.path.resolve() != registry.path.resolve()
                or self.role not in roles
                or registry._active_leases.get(str(registry.path.resolve())) is not self):
            raise CorpusTrustError("CORPUS_WRITE_NOT_AUTHORIZED")


def runtime_index_identity(metadata: dict, collection: str) -> dict:
    identity = {key: metadata.get(key) for key in IDENTITY_FIELDS[:5]}
    return {**identity, "collection_name": collection, "distance_metric": "Cosine"}


class IndexRegistry:
    _lock = threading.Lock()
    _namespace_locks: dict[str, threading.Lock] = {}
    _active_leases: dict[str, CorpusLease] = {}

    def __init__(self, path: Path) -> None:
        self.path = path

    @property
    def trust_root(self) -> Path:
        return self.path.parent / "corpus-trust"

    @contextmanager
    def lease(self, role: str, generation_id: str | None = None):
        """Hold a nonblocking OS exclusive lock for the full trust operation."""
        if os.name == "nt":
            import msvcrt
        elif os.name == "posix":
            import fcntl
        else:
            raise CorpusTrustError("CORPUS_TRUST_SCOPE_UNSUPPORTED")

        key = str(self.path.resolve())
        with self._lock:
            lock = self._namespace_locks.setdefault(key, threading.Lock())
        if not lock.acquire(blocking=False):
            raise CorpusTrustError("CORPUS_INTEGRITY_BUSY")
        capability = CorpusLease(self, role, generation_id)
        acquired = False
        try:
            try:
                self.trust_root.mkdir(parents=True, exist_ok=True)
                capability.handle = (self.trust_root / "namespace.lock").open("a+b")
                if capability.handle.seek(0, os.SEEK_END) == 0:
                    capability.handle.write(b"\0")
                    capability.handle.flush()
                    os.fsync(capability.handle.fileno())
                capability.handle.seek(0)
                try:
                    if os.name == "nt":
                        msvcrt.locking(capability.handle.fileno(), msvcrt.LK_NBLCK, 1)
                    else:
                        fcntl.flock(capability.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError as exc:
                    raise CorpusTrustError("CORPUS_INTEGRITY_BUSY") from exc
            except CorpusTrustError:
                raise
            except OSError as exc:
                raise CorpusTrustError("CORPUS_INTEGRITY_UNAVAILABLE", ["lease I/O"]) from exc
            acquired = True
            self._active_leases[key] = capability
            yield capability
        finally:
            if self._active_leases.get(key) is capability:
                del self._active_leases[key]
            if capability.handle is not None:
                try:
                    if acquired:
                        capability.handle.seek(0)
                        if os.name == "nt":
                            msvcrt.locking(capability.handle.fileno(), msvcrt.LK_UNLCK, 1)
                        else:
                            fcntl.flock(capability.handle.fileno(), fcntl.LOCK_UN)
                finally:
                    capability.handle.close()
            lock.release()

    def protected_path(self, relative: str) -> Path:
        if (not isinstance(relative, str) or not relative
                or Path(relative).is_absolute() or ".." in Path(relative).parts
                or ":" in relative or "\\" in relative):
            raise CorpusTrustError("CORPUS_INVENTORY_INVALID", ["path"])
        root = self.path.parent.resolve()
        target = (root / relative).resolve()
        if not target.is_relative_to(root):
            raise CorpusTrustError("CORPUS_INVENTORY_INVALID", ["path"])
        return target

    @staticmethod
    def uuid_text(value: str) -> str:
        try:
            if str(uuid.UUID(value)) != value:
                raise ValueError("noncanonical UUID")
            return value
        except (ValueError, TypeError, AttributeError) as exc:
            raise CorpusTrustError("CORPUS_INVENTORY_INVALID", ["UUID"]) from exc

    def generation_path(self, generation: str, name: str) -> Path:
        return self.protected_path(
            f"corpus-trust/generations/{self.uuid_text(generation)}/{name}"
        )

    def admission_path(self, admission: str, name: str) -> Path:
        return self.protected_path(
            f"corpus-trust/admissions/{self.uuid_text(admission)}/{name}"
        )

    @staticmethod
    def read_record(path: Path, schema: str, digest_field: str | None,
                    missing: str = "CORPUS_AUTHORITY_NOT_PROVABLE") -> dict:
        try:
            raw = _io_path(path).read_bytes()
        except FileNotFoundError as exc:
            raise CorpusTrustError(missing) from exc
        except OSError as exc:
            raise CorpusTrustError("CORPUS_INTEGRITY_UNAVAILABLE") from exc
        try:
            value = trust_json_loads(raw)
            validate_trust_record(schema, value)
            if digest_field and trust_digest(value, digest_field) != value[digest_field]:
                raise ValueError("digest mismatch")
            return value
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            raise CorpusTrustError("CORPUS_INVENTORY_INVALID", [schema]) from exc

    @staticmethod
    def durable_bytes(path: Path, raw: bytes, *, immutable: bool = False) -> None:
        path = _io_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if immutable and path.exists():
            raise CorpusTrustError("CORPUS_GENERATION_IMMUTABLE")
        temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
        # Retain an interrupted temporary file as failure evidence.
        with temporary.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        if immutable and path.exists():
            raise CorpusTrustError("CORPUS_GENERATION_IMMUTABLE")
        os.replace(temporary, path)

    def save_record(self, path: Path, value: dict, schema: str,
                    digest_field: str | None = None, *, immutable: bool = False) -> dict:
        if digest_field:
            value = {**value, digest_field: trust_digest(value, digest_field)}
        validate_trust_record(schema, value)
        self.durable_bytes(path, (json.dumps(value, ensure_ascii=True, sort_keys=True,
                           separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8"),
                           immutable=immutable)
        return value

    def authorization(self) -> dict:
        """Protected executor record, never accepted from an HTTP request body."""
        try:
            value = trust_json_loads((self.trust_root / "authorization.json").read_bytes())
            required = {"schema_version", "authority_id", "authorization_ref", "admission_id",
                        "generation_id", "logical_collection", "collection_name", "index_identity",
                        "runtime_metadata"}
            if (type(value) is not dict or set(value) != required
                    or value["schema_version"] != "controlled-corpus-authorization-v1"):
                raise ValueError("invalid authorization")
            self.uuid_text(value["admission_id"])
            self.uuid_text(value["generation_id"])
            validate_trust_record("index_identity", value["index_identity"])
            if (type(value["runtime_metadata"]) is not dict or
                    runtime_index_identity(value["runtime_metadata"], value["collection_name"]) != value["index_identity"]):
                raise ValueError("runtime metadata binding")
            trust_digest(value["runtime_metadata"])
            if (not all(type(value[k]) is str and value[k] for k in
                        ("authority_id", "authorization_ref", "logical_collection", "collection_name"))
                    or value["index_identity"]["collection_name"] != value["collection_name"]):
                raise ValueError("invalid authority binding")
            return value
        except FileNotFoundError as exc:
            raise CorpusTrustError("CORPUS_AUTHORITY_NOT_PROVABLE") from exc
        except OSError as exc:
            raise CorpusTrustError("CORPUS_INTEGRITY_UNAVAILABLE") from exc
        except (ValueError, TypeError, KeyError) as exc:
            raise CorpusTrustError("CORPUS_INVENTORY_INVALID", ["authorization"]) from exc

    def intent(self, admission_id: str) -> tuple[dict, dict]:
        auth = self.authorization()
        if auth["admission_id"] != admission_id:
            raise CorpusTrustError("CORPUS_WRITE_NOT_AUTHORIZED")
        intent = self.read_record(self.admission_path(admission_id, "intent.json"),
                                  "admission_intent", "intent_digest")
        if (intent["authority_id"] != auth["authority_id"]
                or intent["authorization_ref"] != auth["authorization_ref"]):
            raise CorpusTrustError("CORPUS_AUTHORITY_NOT_PROVABLE")
        refs = [v["source_ref"] for v in intent["sources"]]
        hashes = [v["pdf_sha256"] for v in intent["sources"]]
        if len(set(refs)) != len(refs) or len(set(hashes)) != len(hashes):
            raise CorpusTrustError("CORPUS_INVENTORY_INVALID", ["duplicate input"])
        for source in intent["sources"]:
            self.protected_path(source["input_relative_path"])
        return intent, auth

    def journal(self, generation: str, *, value: dict | None = None) -> dict:
        value = value if value is not None else self.read_record(
            self.generation_path(generation, "build.json"),
            "build_journal", None, "CORPUS_GENERATION_NOT_ACTIVE")
        try:
            validate_trust_record("build_journal", value)
        except (ValueError, TypeError, KeyError) as exc:
            raise CorpusTrustError("CORPUS_INVENTORY_INVALID", ["journal schema"]) from exc
        if value["generation_id"] != generation:
            raise CorpusTrustError("CORPUS_INVENTORY_BINDING_CONFLICT")
        state = "NEW"
        previous_at = None
        allowed = {("NEW", "ADMITTING"), ("ADMITTING", "PREPARED"),
                   ("PREPARED", "BUILDING"), ("BUILDING", "VERIFYING"),
                   ("VERIFYING", "SEALED")}
        for row in value["transition_history"]:
            at = datetime.fromisoformat(row["at"])
            failed = row["to_state"] in {"FAILED", "ABORTED"}
            if (row["from_state"] != state or (previous_at is not None and at < previous_at)
                    or (not failed and (state, row["to_state"]) not in allowed)
                    or state in {"SEALED", "FAILED", "ABORTED"}
                    or (failed != (row["error_code"] is not None))):
                raise CorpusTrustError("CORPUS_INVENTORY_INVALID", ["journal transition"])
            state = row["to_state"]
            previous_at = at
        if state != value["state"]:
            raise CorpusTrustError("CORPUS_INVENTORY_INVALID", ["journal state"])
        if state in {"NEW", "ADMITTING"} and any(value[k] is not None for k in
                ("inventory_digest", "create_receipt_digest", "build_vector_receipt_digest", "seal_digest")):
            raise CorpusTrustError("CORPUS_INVENTORY_INVALID", ["premature receipt"])
        if state in {"PREPARED", "BUILDING", "VERIFYING", "SEALED"} and value["inventory_digest"] is None:
            raise CorpusTrustError("CORPUS_INVENTORY_INVALID", ["inventory receipt"])
        if state in {"BUILDING", "VERIFYING", "SEALED"} and value["create_receipt_digest"] is None:
            raise CorpusTrustError("CORPUS_INVENTORY_INVALID", ["create receipt"])
        if state == "SEALED" and (value["seal_digest"] is None or value["build_vector_receipt_digest"] is None):
            raise CorpusTrustError("CORPUS_INVENTORY_INVALID", ["seal receipt"])
        if (state in {"FAILED", "ABORTED"}) != (value["failure"] is not None):
            raise CorpusTrustError("CORPUS_INVENTORY_INVALID", ["failure state"])
        return value

    def transition(self, capability: CorpusLease, state: str, *,
                   error_code: str | None = None, receipts: dict | None = None) -> dict:
        capability.require(self, {"intake", "build", "publish"})
        journal = self.journal(capability.generation_id)
        if journal["state"] in {"SEALED", "FAILED", "ABORTED"}:
            raise CorpusTrustError("CORPUS_GENERATION_IMMUTABLE")
        now = datetime.now(UTC).isoformat()
        journal["transition_history"].append({"from_state": journal["state"],
            "to_state": state, "at": now, "error_code": error_code})
        journal.update(state=state, updated_at=now,
                       failure={"code": error_code, "stage": journal["state"]} if error_code else None)
        journal.update(receipts or {})
        self.journal(capability.generation_id, value=journal)
        self.save_record(self.generation_path(capability.generation_id, "build.json"),
                         journal, "build_journal")
        return self.journal(capability.generation_id)

    def initialize_admission(self, intent: dict, authorization: dict) -> None:
        """Trusted executor only, before API intake; no HTTP authority creation route."""
        admission = self.uuid_text(intent["admission_id"])
        generation = self.uuid_text(authorization["generation_id"])
        with self.lease("intake", generation) as capability:
            if (self.trust_root / "authorization.json").exists():
                raise CorpusTrustError("CORPUS_GENERATION_IMMUTABLE")
            self.durable_bytes(self.trust_root / "authorization.json",
                               json.dumps(authorization, allow_nan=False).encode("utf-8"), immutable=True)
            self.save_record(self.admission_path(admission, "intent.json"), intent,
                             "admission_intent", "intent_digest", immutable=True)
            self.intent(admission)
            self.save_record(self.generation_path(generation, "build.json"), {
                "schema_version": "corpus-generation-journal-v1", "generation_id": generation,
                "admission_id": admission, "state": "NEW", "inventory_digest": None,
                "create_receipt_digest": None, "build_vector_receipt_digest": None,
                "seal_digest": None, "transition_history": [], "failure": None,
                "updated_at": datetime.now(UTC).isoformat(),
            }, "build_journal", immutable=True)
            self.transition(capability, "ADMITTING")

    def require_mutation(self, collection: str, capability: CorpusLease | None,
                         states: set[str]) -> dict:
        # A known protected generation is immutable even for a caller lacking a token.
        if (self.trust_root / "authorization.json").exists():
            auth = self.authorization()
            if collection != auth["collection_name"]:
                raise CorpusTrustError("CORPUS_WRITE_NOT_AUTHORIZED")
            journal = self.journal(auth["generation_id"])
            if journal["state"] in {"SEALED", "FAILED", "ABORTED"}:
                raise CorpusTrustError("CORPUS_GENERATION_IMMUTABLE")
        else:
            raise CorpusTrustError("CORPUS_WRITE_NOT_AUTHORIZED")
        if capability is None:
            raise CorpusTrustError("CORPUS_WRITE_NOT_AUTHORIZED")
        capability.require(self, {"build", "publish"})
        if capability.generation_id != auth["generation_id"] or journal["state"] not in states:
            raise CorpusTrustError("CORPUS_WRITE_NOT_AUTHORIZED")
        return auth

    def intake_source(self, admission: str, source_ref: str, upload, db, settings):
        """Bind genuine ingest identities to a closed, pre-upload executor intent."""
        from paper_research.ingestion.upload_service import UploadService
        from paper_research.repositories.paper import PaperRepository

        intent, auth = self.intent(self.uuid_text(admission))
        with self.lease("intake", auth["generation_id"]) as capability:
            if self.journal(auth["generation_id"])["state"] != "ADMITTING":
                raise CorpusTrustError("CORPUS_GENERATION_IMMUTABLE")
            if self.admission_path(admission, "commitment.json").exists():
                raise CorpusTrustError("CORPUS_GENERATION_IMMUTABLE")
            source = next((s for s in intent["sources"] if s["source_ref"] == source_ref), None)
            if source is None:
                raise CorpusTrustError("CORPUS_WRITE_NOT_AUTHORIZED")
            position = intent["sources"].index(source)
            target = self.admission_path(admission, f"document-{position}.json")
            if target.exists():
                raise CorpusTrustError("CORPUS_GENERATION_IMMUTABLE")
            digest, size = hashlib.sha256(), 0
            upload.file.seek(0)
            while part := upload.file.read(1024 * 1024):
                size += len(part)
                digest.update(part)
            upload.file.seek(0)
            if (size != source["byte_count"] or digest.hexdigest() != source["pdf_sha256"]
                    or hashlib.sha256(self.file_bytes(self.protected_path(source["input_relative_path"]),
                                                      "CORPUS_AUTHORITY_NOT_PROVABLE")).hexdigest()
                    != source["pdf_sha256"]):
                raise CorpusTrustError("CORPUS_WRITE_NOT_AUTHORIZED")
            if PaperRepository(db).get_by_hash(source["pdf_sha256"]) is not None:
                raise CorpusTrustError("CORPUS_AUTHORITY_NOT_PROVABLE", ["preexisting document"])
            try:
                source_type = ("upload" if intent["closed_input_scope"] ==
                               "EXPLICIT_OPERATOR_AUTHORIZED_PDFS" else "audit_fixture")
                result = UploadService(db, settings).ingest(upload, source_type=source_type)
                if result.duplicate or result.paper.file_hash != source["pdf_sha256"]:
                    raise CorpusTrustError("CORPUS_AUTHORITY_NOT_PROVABLE")
                folder = settings.parsed_papers_dir / str(result.paper.id)
                self.save_record(target, {"source_ref": source_ref, "paper_id": str(result.paper.id),
                    "pdf_sha256": result.paper.file_hash,
                    "parse_blocks_sha256": hashlib.sha256((folder / "paper_blocks.jsonl").read_bytes()).hexdigest(),
                    "parse_manifest_sha256": hashlib.sha256((folder / "parse_manifest.json").read_bytes()).hexdigest()},
                    "document_binding", immutable=True)
                paths = [self.admission_path(admission, f"document-{i}.json")
                         for i in range(len(intent["sources"]))]
                if all(p.exists() for p in paths):
                    documents = [self.read_record(p, "document_binding", None) for p in paths]
                    self.save_record(self.admission_path(admission, "commitment.json"), {
                        "schema_version": "corpus-admission-commitment-v1", "admission_id": admission,
                        "intent_digest": intent["intent_digest"], "documents": documents, "closed": True,
                        "committed_at": datetime.now(UTC).isoformat(), "admission_digest": "0" * 64},
                        "admission_commitment", "admission_digest", immutable=True)
                return result
            except Exception as exc:
                self.transition(capability, "FAILED", error_code=exc.code if isinstance(exc, IndexIdentityError)
                                else "CORPUS_INTEGRITY_UNAVAILABLE")
                raise

    def refuse_direct_mutation(self) -> None:
        if (self.trust_root / "authorization.json").exists():
            auth = self.authorization()
            if self.journal(auth["generation_id"])["state"] in {"SEALED", "FAILED", "ABORTED"}:
                raise CorpusTrustError("CORPUS_GENERATION_IMMUTABLE")
        raise CorpusTrustError("CORPUS_WRITE_NOT_AUTHORIZED")

    def inventory(self, generation: str) -> dict:
        value = self.read_record(self.generation_path(generation, "inventory.json"),
                                 "inventory", "inventory_digest", "CORPUS_INVENTORY_MISSING")
        for document in value["document_to_chunk_membership"]:
            self.protected_path(document["chunks_relative_path"])
        return value

    def validate_membership(self, inventory: dict, commitment: dict) -> None:
        rows = inventory["expected_chunks"]
        ids = [r["chunk_id"] for r in rows]
        if len(ids) != len(set(ids)):
            raise CorpusTrustError("CANONICAL_SOURCE_IDENTITY_CONFLICT")
        points = [r["point_id"] for r in rows]
        if (len(points) != len(set(points)) or
                any(r["point_id"] != point_identity(r["chunk_id"]) for r in rows)):
            raise CorpusTrustError("CORPUS_INVENTORY_INVALID", ["point binding"])
        docs = inventory["expected_document_ids"]
        admitted = [r["paper_id"] for r in commitment["documents"]]
        memberships = inventory["document_to_chunk_membership"]
        if (docs != sorted(admitted) or len(docs) != inventory["expected_document_count"]
                or len(rows) != inventory["expected_chunk_count"]
                or rows != sorted(rows, key=lambda r: (r["paper_id"], r["chunk_id"]))
                or [r["paper_id"] for r in memberships] != docs
                or len(admitted) != len(set(admitted))):
            raise CorpusTrustError("CORPUS_INVENTORY_INVALID", ["document partition"])
        partition = []
        for doc in memberships:
            paper = doc["paper_id"]
            if doc["chunk_ids"] != sorted(doc["chunk_ids"]):
                raise CorpusTrustError("CORPUS_INVENTORY_INVALID", ["member ordering"])
            expected_path = f"parsed/{paper}/paper_chunks.{inventory['index_identity']['index_version']}.jsonl"
            if doc["chunks_relative_path"] != expected_path:
                raise CorpusTrustError("CORPUS_INVENTORY_INVALID", ["canonical path"])
            self.protected_path(expected_path)
            partition.extend((paper, cid) for cid in doc["chunk_ids"])
        if partition != [(r["paper_id"], r["chunk_id"]) for r in rows]:
            raise CorpusTrustError("CORPUS_INVENTORY_INVALID", ["chunk partition"])

    def authority_bundle(self, generation: str, *, inventory: dict | None = None) -> dict:
        inventory = inventory or self.inventory(generation)
        auth = self.authorization()
        intent, _ = self.intent(inventory["source_admission_id"])
        commitment = self.read_record(self.admission_path(intent["admission_id"], "commitment.json"),
                                      "admission_commitment", "admission_digest")
        expected_refs = [r["source_ref"] for r in intent["sources"]]
        if (commitment["admission_id"] != intent["admission_id"]
                or commitment["intent_digest"] != intent["intent_digest"]
                or [r["source_ref"] for r in commitment["documents"]] != expected_refs
                or [r["pdf_sha256"] for r in commitment["documents"]]
                    != [r["pdf_sha256"] for r in intent["sources"]]):
            raise CorpusTrustError("CORPUS_INVENTORY_INVALID", ["source closure"])
        if (auth["generation_id"] != generation or inventory["generation_id"] != generation
                or inventory["source_admission_digest"] != commitment["admission_digest"]
                or inventory["index_identity"] != auth["index_identity"]):
            raise CorpusTrustError("CORPUS_INVENTORY_BINDING_CONFLICT")
        if (datetime.fromisoformat(intent["created_at"]) > datetime.fromisoformat(commitment["committed_at"])
                or datetime.fromisoformat(commitment["committed_at"]) > datetime.fromisoformat(inventory["created_at"])):
            raise CorpusTrustError("CORPUS_INVENTORY_INVALID", ["commit order"])
        if inventory["parser_receipt_digest"] != trust_digest(commitment["documents"]):
            raise CorpusTrustError("CORPUS_INVENTORY_INVALID", ["parser receipt"])
        return {"inventory": inventory, "authorization": auth, "intent": intent,
                "commitment": commitment}

    def verify_prepared(self, inventory: dict) -> None:
        digests = []
        bindings = {(r["paper_id"], r["chunk_id"]): r for r in inventory["expected_chunks"]}
        for doc in inventory["document_to_chunk_membership"]:
            path = self.generation_path(inventory["generation_id"],
                                        f"prepared/{doc['paper_id']}/chunks.jsonl")
            raw = self.file_bytes(path, "CORPUS_AUTHORITY_NOT_PROVABLE")
            digest = hashlib.sha256(raw).hexdigest()
            if digest != doc["chunks_file_sha256"]:
                raise CorpusTrustError("CORPUS_INVENTORY_INVALID", ["prepared bytes"])
            records = self.chunk_records(raw)
            keys = [(r["paper_id"], r["chunk_id"]) for r in records]
            if sorted(keys) != [(doc["paper_id"], cid) for cid in doc["chunk_ids"]]:
                raise CorpusTrustError("CORPUS_INVENTORY_INVALID", ["prepared members"])
            if any(trust_digest(r) != bindings[key]["canonical_payload_sha256"]
                   for r, key in zip(records, keys, strict=True)):
                raise CorpusTrustError("CORPUS_INVENTORY_INVALID", ["prepared content"])
            digests.append([doc["paper_id"], digest])
        if inventory["prepared_records_digest"] != trust_digest(digests):
            raise CorpusTrustError("CORPUS_INVENTORY_INVALID", ["prepared aggregate"])

    @staticmethod
    def file_bytes(path: Path, missing: str) -> bytes:
        try:
            return _io_path(path).read_bytes()
        except FileNotFoundError as exc:
            raise CorpusTrustError(missing) from exc
        except OSError as exc:
            raise CorpusTrustError("CORPUS_INTEGRITY_UNAVAILABLE") from exc

    @staticmethod
    def chunk_records(raw: bytes) -> list[dict]:
        from paper_research.chunking.types import Chunk

        try:
            records = [trust_json_loads(line) for line in raw.decode("utf-8").splitlines() if line]
            for row in records:
                if type(row) is not dict or set(row) != set(Chunk.model_fields):
                    raise ValueError("invalid full Chunk fields")
                normalized = Chunk.model_validate(row).model_dump()
                if trust_digest(normalized) != trust_digest(row):
                    raise ValueError("coerced Chunk")
            return records
        except (ValueError, TypeError, KeyError) as exc:
            raise CorpusTrustError("CORPUS_CANONICAL_CONTENT_MISMATCH") from exc

    def verify_documents(self, bundle: dict) -> None:
        from paper_research.db import SessionLocal
        from paper_research.models.paper import Paper

        expected = {r["paper_id"]: r["pdf_sha256"] for r in bundle["commitment"]["documents"]}
        try:
            with SessionLocal() as session:
                rows = session.query(Paper).filter(Paper.id.in_([uuid.UUID(k) for k in expected])).all()
                actual = {str(row.id): row.file_hash for row in rows}
        except Exception as exc:
            raise CorpusTrustError("CORPUS_INTEGRITY_UNAVAILABLE", ["PostgreSQL"]) from exc
        if actual != expected:
            raise CorpusTrustError("CORPUS_MEMBERSHIP_MISMATCH", ["admitted documents"])

    def verify_canonical(self, inventory: dict):
        from paper_research.chunking.types import Chunk

        index = inventory["index_identity"]["index_version"]
        if "/" in index or "\\" in index or ":" in index or ".." in index:
            raise CorpusTrustError("CORPUS_INVENTORY_INVALID", ["index path"])
        root = self.path.parent / "parsed"
        expected_paths = {self.protected_path(d["chunks_relative_path"])
                          for d in inventory["document_to_chunk_membership"]}
        try:
            actual_paths = {self.protected_path(p.relative_to(self.path.parent).as_posix())
                            for p in root.glob(f"*/paper_chunks.{index}.jsonl")}
        except OSError as exc:
            raise CorpusTrustError("CORPUS_INTEGRITY_UNAVAILABLE") from exc
        if actual_paths != expected_paths:
            raise CorpusTrustError("CORPUS_MEMBERSHIP_MISMATCH", ["canonical files"])
        bindings = {(r["paper_id"], r["chunk_id"]): r for r in inventory["expected_chunks"]}
        chunks, observed = [], []
        for doc in inventory["document_to_chunk_membership"]:
            raw = self.file_bytes(self.protected_path(doc["chunks_relative_path"]),
                                  "CORPUS_MEMBERSHIP_MISMATCH")
            records = self.chunk_records(raw)
            keys = [(r["paper_id"], r["chunk_id"]) for r in records]
            if sorted(keys) != [(doc["paper_id"], cid) for cid in doc["chunk_ids"]]:
                raise CorpusTrustError("CORPUS_MEMBERSHIP_MISMATCH", ["canonical owners/IDs"])
            digest = hashlib.sha256(raw).hexdigest()
            if (digest != doc["chunks_file_sha256"] or
                    any(trust_digest(r) != bindings[key]["canonical_payload_sha256"]
                        for r, key in zip(records, keys, strict=True))):
                raise CorpusTrustError("CORPUS_CANONICAL_CONTENT_MISMATCH")
            chunks.extend(Chunk.model_validate(r) for r in records)
            observed.append([doc["paper_id"], digest])
        return chunks, trust_digest(observed)

    @staticmethod
    def physical_points(client: QdrantClient, collection: str) -> list:
        points, offsets, ids = [], set(), set()
        offset = None
        while True:
            try:
                page, following = client.scroll(collection_name=collection, limit=128,
                    offset=offset, with_payload=True, with_vectors=True)
            except Exception as exc:
                raise CorpusTrustError("CORPUS_INTEGRITY_UNAVAILABLE", ["physical scroll"]) from exc
            for point in page:
                key = str(point.id)
                if key in ids:
                    raise CorpusTrustError("CORPUS_PHYSICAL_MEMBERSHIP_MISMATCH", ["duplicate point"])
                ids.add(key)
                points.append(point)
            if following is None:
                return points
            key = str(following)
            if key in offsets or not page:
                raise CorpusTrustError("CORPUS_INTEGRITY_UNAVAILABLE", ["unstable pagination"])
            offsets.add(key)
            offset = following

    def verify_physical(self, inventory: dict, client: QdrantClient,
                        seal: dict | None = None) -> list[dict]:
        bindings = {r["point_id"]: r for r in inventory["expected_chunks"]}
        points = self.physical_points(client, inventory["index_identity"]["collection_name"])
        if {str(p.id) for p in points} != set(bindings):
            raise CorpusTrustError("CORPUS_PHYSICAL_MEMBERSHIP_MISMATCH", ["point IDs"])
        # Check all owners before contents, so an out-of-scope bad owner has priority.
        for p in points:
            row = bindings[str(p.id)]
            if (type(p.payload) is not dict or p.payload.get("paper_id") != row["paper_id"]
                    or p.payload.get("chunk_id") != row["chunk_id"]):
                raise CorpusTrustError("CORPUS_PHYSICAL_MEMBERSHIP_MISMATCH", ["point owner"])
        sealed = {r["point_id"]: r for r in seal["observed_physical_points"]} if seal else None
        vectors = None
        receipt = self.read_record(self.generation_path(inventory["generation_id"], "vectors.json"),
                                   "build_vector_receipt", "receipt_digest")
        if receipt["generation_id"] != inventory["generation_id"] or receipt["inventory_digest"] != inventory["inventory_digest"]:
            raise CorpusTrustError("CORPUS_INVENTORY_BINDING_CONFLICT")
        if receipt["vectors_relative_path"] != f"corpus-trust/generations/{inventory['generation_id']}/vectors.jsonl":
            raise CorpusTrustError("CORPUS_INVENTORY_BINDING_CONFLICT", ["vector receipt path"])
        raw = self.file_bytes(self.protected_path(receipt["vectors_relative_path"]),
                              "CORPUS_AUTHORITY_NOT_PROVABLE")
        if hashlib.sha256(raw).hexdigest() != receipt["vectors_file_sha256"]:
            raise CorpusTrustError("CORPUS_INVENTORY_INVALID", ["embedding receipt bytes"])
        rows = [trust_json_loads(line) for line in raw.splitlines() if line]
        vectors = {r["point_id"]: r for r in rows}
        if len(rows) != len(vectors) or set(vectors) != set(bindings) or receipt["point_count"] != len(rows):
            raise CorpusTrustError("CORPUS_INVENTORY_INVALID", ["embedding receipt members"])
        if sealed is not None and (set(sealed) != set(bindings) or len(sealed) != len(seal["observed_physical_points"])):
            raise CorpusTrustError("CORPUS_INVENTORY_INVALID", ["seal members"])
        observed = []
        dimension = inventory["index_identity"]["embedding_dimension"]
        for p in points:
            key = str(p.id)
            row = bindings[key]
            payload_digest = trust_digest(p.payload)
            if payload_digest != row["canonical_payload_sha256"]:
                raise CorpusTrustError("CORPUS_PHYSICAL_CONTENT_MISMATCH", ["payload"])
            try:
                vector_digest = vector_f32_digest(p.vector, dimension)
                if sealed is None:
                    vector = vectors[key]
                    if set(vector) != {"point_id", "input_f32", "normalized_f32"}:
                        raise ValueError("embedding row fields")
                    normalized = normalized_f32(vector["input_f32"], dimension)
                    if normalized != vector["normalized_f32"]:
                        raise ValueError("embedding normalization receipt")
                    if any(abs(actual - expected) > 1e-6 + 1e-5 * abs(expected)
                           for actual, expected in zip(p.vector, normalized, strict=True)):
                        raise ValueError("stored vector differs")
                elif (sealed[key]["payload_sha256"] != payload_digest
                      or sealed[key]["vector_f32_sha256"] != vector_digest):
                    raise ValueError("sealed content differs")
            except (ValueError, TypeError, KeyError, OverflowError) as exc:
                raise CorpusTrustError("CORPUS_PHYSICAL_CONTENT_MISMATCH", ["vector"]) from exc
            observed.append({"point_id": key, "paper_id": row["paper_id"],
                "chunk_id": row["chunk_id"], "payload_sha256": payload_digest,
                "vector_f32_sha256": vector_digest})
        return sorted(observed, key=lambda r: r["point_id"])

    def sealed_bundle(self, generation: str) -> dict:
        inventory = self.inventory(generation)
        # Parse all present protected records before semantic authority/binding checks.
        # Absent authority beats absent seal, while malformed present data is INVALID.
        records, missing = {}, []
        for key, path, schema, digest, absent in [
            ("intent", self.admission_path(inventory["source_admission_id"], "intent.json"),
             "admission_intent", "intent_digest", "CORPUS_AUTHORITY_NOT_PROVABLE"),
            ("commitment", self.admission_path(inventory["source_admission_id"], "commitment.json"),
             "admission_commitment", "admission_digest", "CORPUS_AUTHORITY_NOT_PROVABLE"),
            ("create", self.generation_path(generation, "create.json"),
             "create_receipt", "receipt_digest", "CORPUS_AUTHORITY_NOT_PROVABLE"),
            ("vectors", self.generation_path(generation, "vectors.json"),
             "build_vector_receipt", "receipt_digest", "CORPUS_AUTHORITY_NOT_PROVABLE"),
            ("seal", self.generation_path(generation, "seal.json"),
             "seal", "seal_digest", "CORPUS_GENERATION_NOT_ACTIVE"),
            ("journal", self.generation_path(generation, "build.json"),
             "build_journal", None, "CORPUS_GENERATION_NOT_ACTIVE"),
        ]:
            try:
                records[key] = self.read_record(path, schema, digest, absent)
            except CorpusTrustError as exc:
                if exc.code != absent:
                    raise
                missing.append(absent)
        bundle = self.authority_bundle(generation, inventory=inventory)
        if "CORPUS_AUTHORITY_NOT_PROVABLE" in missing:
            raise CorpusTrustError("CORPUS_AUTHORITY_NOT_PROVABLE")
        if missing:
            raise CorpusTrustError("CORPUS_GENERATION_NOT_ACTIVE")
        bundle.update(records)
        auth, seal, journal = bundle["authorization"], records["seal"], self.journal(generation)
        if (seal["generation_id"] != generation or seal["inventory_id"] != inventory["inventory_id"]
                or seal["inventory_digest"] != inventory["inventory_digest"]
                or seal["index_identity"] != inventory["index_identity"]
                or records["create"]["generation_id"] != generation
                or records["create"]["collection_name"] != auth["collection_name"]
                or records["vectors"]["generation_id"] != generation
                or records["vectors"]["inventory_digest"] != inventory["inventory_digest"]
                or seal["create_receipt_digest"] != records["create"]["receipt_digest"]
                or seal["build_vector_receipt_digest"] != records["vectors"]["receipt_digest"]
                or journal["admission_id"] != inventory["source_admission_id"]
                or journal["inventory_digest"] != inventory["inventory_digest"]
                or journal["create_receipt_digest"] != records["create"]["receipt_digest"]
                or journal["build_vector_receipt_digest"] != records["vectors"]["receipt_digest"]):
            raise CorpusTrustError("CORPUS_INVENTORY_BINDING_CONFLICT")
        if journal["state"] != "SEALED" or journal["seal_digest"] != seal["seal_digest"]:
            raise CorpusTrustError("CORPUS_GENERATION_NOT_ACTIVE")
        times = [inventory["created_at"], records["create"]["created_at"],
                 seal["verified_at"], seal["sealed_at"]]
        if [datetime.fromisoformat(t) for t in times] != sorted(datetime.fromisoformat(t) for t in times):
            raise CorpusTrustError("CORPUS_INVENTORY_INVALID", ["seal order"])
        return bundle

    def verify_complete(self, bundle: dict, client: QdrantClient, *, sealed: bool):
        inventory = bundle["inventory"]
        self.validate_membership(inventory, bundle["commitment"])
        self.verify_prepared(inventory)
        self.verify_documents(bundle)
        chunks, canonical_digest_value = self.verify_canonical(inventory)
        # Bind planned D1 helper bytes at seal without requiring live auxiliary
        # availability during a later query. The expected metadata is pre-input.
        helpers = []
        for doc in inventory["document_to_chunk_membership"]:
            path = self.protected_path(doc["chunks_relative_path"])
            raw = json.dumps({**bundle["authorization"]["runtime_metadata"],
                "collection": inventory["index_identity"]["collection_name"],
                "paper_id": doc["paper_id"], "chunk_count": len(doc["chunk_ids"]),
                "chunks_path": str(path)}, ensure_ascii=False, indent=2).encode("utf-8")
            digest = hashlib.sha256(raw).hexdigest()
            helpers.append([doc["paper_id"], digest])
            if not sealed:
                actual = self.file_bytes(path.with_name(f"{path.stem}_manifest.json"),
                                         "CORPUS_AUTHORITY_NOT_PROVABLE")
                if hashlib.sha256(actual).hexdigest() != digest:
                    raise CorpusTrustError("CORPUS_INVENTORY_BINDING_CONFLICT", ["D1 helper bytes"])
        canonical_digest_value = trust_digest({"critical_chunks": canonical_digest_value,
                                               "planned_D1_helpers": helpers})
        seal = bundle.get("seal") if sealed else None
        observed = self.verify_physical(inventory, client, seal)
        if sealed and (seal["observed_canonical_digest"] != canonical_digest_value
                       or seal["observed_physical_digest"] != trust_digest(observed)
                       or seal["observed_physical_points"] != observed):
            raise CorpusTrustError("CORPUS_INVENTORY_BINDING_CONFLICT", ["seal observation"])
        return chunks, canonical_digest_value, observed

    def verify_active(self, logical: str, collection: str, expected: dict,
                      client: QdrantClient):
        data = self.read()
        binding = data.get("indexes", {}).get(collection, {}).get("trust_binding")
        if binding is None:
            raise CorpusTrustError("CORPUS_INVENTORY_MISSING")
        try:
            validate_trust_record("active_binding", binding)
        except (ValueError, TypeError, KeyError) as exc:
            raise CorpusTrustError("CORPUS_INVENTORY_INVALID", ["ACTIVE binding"]) from exc
        bundle = self.sealed_bundle(binding["generation_id"])
        inventory, seal = bundle["inventory"], bundle["seal"]
        if (data.get("defaults", {}).get(logical) != collection
                or binding["logical_collection"] != logical
                or binding["collection_name"] != collection
                or bundle["authorization"]["logical_collection"] != logical
                or binding["inventory_id"] != inventory["inventory_id"]
                or binding["inventory_digest"] != inventory["inventory_digest"]
                or binding["source_admission_digest"] != inventory["source_admission_digest"]
                or binding["seal_digest"] != seal["seal_digest"]
                or binding["index_identity"] != expected
                or inventory["index_identity"] != expected):
            raise CorpusTrustError("CORPUS_INVENTORY_BINDING_CONFLICT")
        if datetime.fromisoformat(binding["published_at"]) < datetime.fromisoformat(seal["sealed_at"]):
            raise CorpusTrustError("CORPUS_INVENTORY_INVALID", ["publication order"])
        return self.verify_complete(bundle, client, sealed=True)[0]

    @contextmanager
    def query_admission(self, logical: str, collection: str, expected: dict,
                        client: QdrantClient, *, object_check=None, supported: bool = True):
        # Preserve every existing P0 refusal before the new corpus requirements.
        self.verify_identity(collection, expected, client)
        if object_check is not None:
            object_check()
        with self.lease("query") as capability:
            self.verify_identity(collection, expected, client)
            if object_check is not None:
                object_check()
            if not supported:
                raise CorpusTrustError("CORPUS_TRUST_SCOPE_UNSUPPORTED")
            try:
                chunks = self.verify_active(logical, collection, expected, client)
            except IndexIdentityError:
                raise
            except (ValueError, TypeError, KeyError, AttributeError) as exc:
                raise CorpusTrustError("CORPUS_INVENTORY_INVALID") from exc
            except Exception as exc:
                raise CorpusTrustError("CORPUS_INTEGRITY_UNAVAILABLE") from exc
            yield chunks, capability

    def publish(self, generation: str, client: QdrantClient,
                capability: CorpusLease) -> dict:
        capability.require(self, {"build", "publish"})
        if capability.generation_id != generation:
            raise CorpusTrustError("CORPUS_WRITE_NOT_AUTHORIZED")
        bundle = self.sealed_bundle(generation)
        auth, inventory, seal = bundle["authorization"], bundle["inventory"], bundle["seal"]
        collection = auth["collection_name"]
        self.verify_identity(collection, inventory["index_identity"], client)
        self.verify_complete(bundle, client, sealed=True)
        with self._lock:
            data = self.read()
            if data.get("defaults", {}).get(auth["logical_collection"]) is not None:
                raise CorpusTrustError("CORPUS_GENERATION_IMMUTABLE")
            binding = {"schema_version": "active-corpus-generation-v1",
                "logical_collection": auth["logical_collection"], "collection_name": collection,
                "generation_id": generation, "inventory_id": inventory["inventory_id"],
                "inventory_digest": inventory["inventory_digest"], "seal_digest": seal["seal_digest"],
                "source_admission_digest": inventory["source_admission_digest"],
                "index_identity": inventory["index_identity"], "publication_state": "ACTIVE",
                "published_at": datetime.now(UTC).isoformat()}
            validate_trust_record("active_binding", binding)
            data.setdefault("defaults", {})[auth["logical_collection"]] = collection
            data.setdefault("indexes", {}).setdefault(collection, {})["trust_binding"] = binding
            self._write(data)
        return binding

    def resolve(self, logical_collection: str) -> str:
        try:
            data = self.read()
            return str(data.get("defaults", {}).get(logical_collection, logical_collection))
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            raise IndexIdentityError("INDEX_IDENTITY_INVALID") from exc

    def switch(
        self,
        logical_collection: str,
        physical_collection: str,
        metadata: dict[str, object],
    ) -> None:
        self.refuse_direct_mutation()

    def bind_identity(self, collection: str, identity: dict, client: QdrantClient,
                      *, capability: CorpusLease | None = None) -> None:
        """Called only by a builder after it creates a new physical collection."""
        self._validate_shape(identity)
        self._validate_physical(collection, identity, client)
        self.require_mutation(collection, capability, {"PREPARED", "BUILDING"})
        with self._lock:
            data = self.read()
            entry = data.setdefault("indexes", {}).setdefault(collection, {})
            if "embedding_identity" in entry:
                if self.load_identity(collection) != identity:
                    raise IndexIdentityError("INDEX_IDENTITY_CONFLICT", status_code=409)
                return
            entry["embedding_identity"] = {
                "schema_version": IDENTITY_SCHEMA,
                "identity": identity,
                "sha256": canonical_digest(identity),
            }
            self._write(data)

    def load_identity(self, collection: str) -> dict:
        try:
            record = self.read().get("indexes", {}).get(collection, {}).get("embedding_identity")
            if record is None:
                raise IndexIdentityError("INDEX_IDENTITY_MISSING")
            identity = record["identity"]
            if record.get("schema_version") != IDENTITY_SCHEMA:
                raise IndexIdentityError("INDEX_IDENTITY_INVALID", ["schema_version"])
            self._validate_shape(identity)
            if record.get("sha256") != canonical_digest(identity):
                raise IndexIdentityError("INDEX_IDENTITY_INVALID", ["sha256"])
            if identity["collection_name"] != collection:
                raise IndexIdentityError("INDEX_IDENTITY_INVALID", ["collection_name"])
            return identity
        except IndexIdentityError:
            raise
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            raise IndexIdentityError("INDEX_IDENTITY_INVALID") from exc

    def verify_identity(self, collection: str, expected: dict, client: QdrantClient) -> dict:
        identity = self.load_identity(collection)
        self._validate_physical(collection, identity, client)
        mismatches = [key for key in IDENTITY_FIELDS if identity[key] != expected.get(key)]
        if mismatches:
            raise IndexIdentityError("INDEX_IDENTITY_CONFLICT", mismatches, status_code=409)
        return identity

    @staticmethod
    def _validate_shape(identity: dict) -> None:
        if not isinstance(identity, dict) or set(identity) != set(IDENTITY_FIELDS):
            raise IndexIdentityError("INDEX_IDENTITY_INVALID", ["fields"])
        if type(identity["embedding_dimension"]) is not int or identity["embedding_dimension"] <= 0:
            raise IndexIdentityError("INDEX_IDENTITY_INVALID", ["embedding_dimension"])
        if any(not isinstance(identity[key], str) or not identity[key].strip()
               for key in IDENTITY_FIELDS if key != "embedding_dimension"):
            raise IndexIdentityError("INDEX_IDENTITY_INVALID", ["fields"])

    @staticmethod
    def _validate_physical(collection: str, identity: dict, client: QdrantClient) -> None:
        try:
            vectors = client.get_collection(collection).config.params.vectors
        except Exception as exc:
            raise IndexIdentityError("INDEX_IDENTITY_UNAVAILABLE") from exc
        if not isinstance(vectors, models.VectorParams):
            raise IndexIdentityError("INDEX_IDENTITY_PHYSICAL_CONFLICT", ["vectors"])
        fields = []
        if vectors.size != identity["embedding_dimension"]:
            fields.append("embedding_dimension")
        if vectors.distance.value != identity["distance_metric"]:
            fields.append("distance_metric")
        if fields:
            raise IndexIdentityError("INDEX_IDENTITY_PHYSICAL_CONFLICT", fields)

    def read(self) -> dict:
        if not self.path.exists():
            return {"defaults": {}, "indexes": {}}
        return trust_json_loads(self.path.read_bytes())

    def _write(self, data: dict) -> None:
        self.durable_bytes(self.path, json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8"))
