"""Explicit Phase 2 scientific configuration, isolated from production defaults."""

import hashlib
import json
from pathlib import Path

from qdrant_client import QdrantClient

from paper_research.chunking.types import Chunk
from paper_research.config import Settings
from paper_research.indexing.identity import assert_runtime_compatible, corpus_identity
from paper_research.indexing.vector_store import QdrantVectorStore
from paper_research.providers.factory import build_embedding_provider, build_reranker
from paper_research.retrieval.context_builder import ContextBuilder
from paper_research.retrieval.dense import DenseRetriever
from paper_research.retrieval.filters import RetrievalFilter
from paper_research.retrieval.pipeline import RetrievalPipeline
from paper_research.retrieval.sparse import BM25Retriever

BACKEND_ID = "R2F_RAW_QWEN_ALWAYS_RERANK_V1"
MANIFEST_SHA256 = "40fc4a292a25f55b25dbb15170316e00f07f096338787c0cf55afd72d53f5d74"
ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = (
    ROOT / "artifacts/paperresearch-v2/phase1q-r2f/freeze/phase2_retrieval_backend_manifest.json"
)


class FrozenBackendIdentityError(ValueError):
    def __init__(self, detail):
        super().__init__("FROZEN_BACKEND_IDENTITY_MISMATCH: " + detail)


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_manifest(backend_id=BACKEND_ID, manifest_path=MANIFEST_PATH):
    """Pin the whole manifest, not just model names or vector dimensions."""
    try:
        if backend_id != BACKEND_ID or file_sha(manifest_path) != MANIFEST_SHA256:
            raise FrozenBackendIdentityError("backend_id or manifest digest")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["backend_id"] != backend_id or manifest["status"] != "FROZEN":
            raise FrozenBackendIdentityError("unfrozen backend")
        for relative, expected in manifest["code_sha256"].items():
            if file_sha(ROOT / relative) != expected:
                raise FrozenBackendIdentityError("source digest: " + relative)
        for relative, expected in manifest["evidence_sha256"].items():
            if file_sha(ROOT / relative) != expected:
                raise FrozenBackendIdentityError("evidence digest: " + relative)
        corpus = manifest["corpus"]
        for field, hash_field in [
            ("path", "sha256"),
            ("metadata_path", "metadata_sha256"),
            ("index_manifest_path", "index_manifest_sha256"),
        ]:
            if file_sha(ROOT / corpus[field]) != corpus[hash_field]:
                raise FrozenBackendIdentityError(hash_field)
        return manifest
    except FrozenBackendIdentityError:
        raise
    except (KeyError, OSError, ValueError) as exc:
        raise FrozenBackendIdentityError(type(exc).__name__) from exc


class FrozenRetrievalBackend:
    """The same dense/raw BM25/RRF/always model rerank/ContextBuilder pipeline."""

    backend_id = BACKEND_ID

    def __init__(self, manifest, pipeline, chunks, *, execution_mode="LIVE_FROZEN_BACKEND"):
        if manifest != read_manifest():
            raise FrozenBackendIdentityError("constructor manifest")
        self.manifest = manifest
        self.pipeline = pipeline
        self.chunks = {c.chunk_id: c for c in chunks}
        self.execution_mode = execution_mode
        self.calls = []

    def guard(self):
        p, b = self.pipeline, self.manifest
        dense = b["dense"]
        observed = (
            p.dense.embedding.provider_name,
            p.dense.embedding.model_name,
            p.dense.embedding.revision,
            p.dense.embedding.dimensions,
            p.dense.vector_store.collection,
            p.sparse.k1,
            p.sparse.b,
            p.sparse.contextual,
            p.reranker.provider_name,
            p.reranker.model_name,
            p.rerank_mode,
            p.candidate_top_k,
            p.final_top_k,
            p.context_builder.include_neighbors,
            p.context_builder.max_tokens,
            p.context_builder.max_characters,
        )
        expected = (
            dense["provider"],
            dense["model"],
            dense["revision"],
            dense["dimension"],
            dense["collection"],
            1.5,
            0.75,
            False,
            "siliconflow",
            "Qwen/Qwen3-Reranker-0.6B",
            "always",
            30,
            5,
            False,
            12000,
            48000,
        )
        if observed != expected:
            raise FrozenBackendIdentityError("runtime scientific configuration")
        if (
            p.rank_stages
            or p.contextual_required
            or p.selective_policy is not None
            or p.reranker.base_url + "/rerank" != b["reranker"]["endpoint"]
            or p.reranker.timeout_seconds != b["reranker"]["timeout_seconds"]
            or p.reranker.max_retries != b["reranker"]["max_retries"]
            or [c.chunk_id for c in p.sparse.chunks] != list(self.chunks)
        ):
            raise FrozenBackendIdentityError("pipeline or reranker contract")

    def search(self, query, paper_ids=None, limit=5):
        self.guard()
        if limit != self.manifest["final_top_k"]:
            raise FrozenBackendIdentityError("agent attempted to change final_top_k")
        result = self.pipeline.retrieve(
            query,
            RetrievalFilter(paper_ids=paper_ids or None),
            recall_k=20,
            top_k=5,
            retrieval_scope="paper",
        )
        self.calls.append(
            {
                "backend_id": self.backend_id,
                "query": query[:800],
                "execution_mode": self.execution_mode,
                "collection": self.manifest["dense"]["collection"],
                "rerank_mode": "always",
                "candidate_top_k": 30,
                "final_top_k": 5,
                "rerank_triggered": result.trace.rerank_triggered,
                "evidence_ids": [c.chunk_id for c in result.context],
            }
        )
        items = []
        for c in result.context:
            raw = self.chunks[c.chunk_id]
            if (c.paper_id, c.block_ids, c.block_page_map, c.page_start, c.page_end) != (
                raw.paper_id,
                raw.block_ids,
                raw.block_page_map,
                raw.page_start,
                raw.page_end,
            ) or c.evidence != raw.chunk_text[: len(c.evidence)]:
                raise FrozenBackendIdentityError("returned provenance or original text")
            items.append(
                {
                    "evidence_id": c.chunk_id,
                    "chunk_id": c.chunk_id,
                    "paper_id": c.paper_id,
                    "block_ids": c.block_ids,
                    "block_page_map": c.block_page_map,
                    "page_start": c.page_start,
                    "page_end": c.page_end,
                    "section_path": c.section_path,
                    "text": c.evidence,
                    "retrieval_score": c.score,
                    "backend_id": self.backend_id,
                }
            )
        return items

    def close(self):
        self.pipeline.dense.embedding.client.close()
        self.pipeline.reranker.client.close()
        self.pipeline.dense.vector_store.client.close()


def load_frozen_retrieval_backend(
    backend_id=BACKEND_ID,
    *,
    manifest_path=MANIFEST_PATH,
    credentials=None,
    client=None,
    embedding_client=None,
    reranker_client=None,
    verify_live=True,
):
    b = read_manifest(backend_id, manifest_path)
    # Only credentials/infrastructure may come from the environment; every scientific
    # setting below is explicitly supplied by the pinned manifest.
    secrets = credentials or Settings()
    d, r = b["dense"], b["reranker"]
    settings = secrets.model_copy(
        update={
            "embedding_provider": d["provider"],
            "embedding_model": d["model"],
            "embedding_revision": d["revision"],
            "embedding_dimensions": d["dimension"],
            "embedding_base_url": d["base_url"],
            "embedding_timeout_seconds": d["timeout_seconds"],
            "embedding_max_retries": d["max_retries"],
            "rerank_enabled": True,
            "rerank_provider": r["provider"],
            "rerank_model": r["model"],
            "rerank_base_url": r["endpoint"].removesuffix("/rerank"),
            "rerank_timeout_seconds": r["timeout_seconds"],
            "rerank_max_retries": r["max_retries"],
            "rerank_allow_fallback": False,
        }
    )
    chunks = [
        Chunk.model_validate_json(line)
        for line in (ROOT / b["corpus"]["path"]).read_text(encoding="utf-8").splitlines()
    ]
    metadata = json.loads((ROOT / b["corpus"]["metadata_path"]).read_text(encoding="utf-8"))
    index = json.loads((ROOT / b["corpus"]["index_manifest_path"]).read_text(encoding="utf-8"))
    identity = corpus_identity(chunks, metadata)
    if (
        any(index[k] != v for k, v in identity.items())
        or identity["sorted_chunk_id_digest"] != b["corpus"]["chunk_digest"]
    ):
        raise FrozenBackendIdentityError("canonical corpus")
    embedding, reranker = build_embedding_provider(settings), build_reranker(settings)
    for provider, transport in [(embedding, embedding_client), (reranker, reranker_client)]:
        if transport is not None:
            provider.client.close()
            provider.client = transport
    qdrant = client or QdrantClient(url=d["qdrant_url"], api_key=secrets.qdrant_api_key)
    assert_runtime_compatible(index, embedding, collection=d["collection"])
    if verify_live:
        config = qdrant.get_collection(d["collection"]).config.params.vectors
        if config.size != d["dimension"] or config.distance.value != d["distance"]:
            raise FrozenBackendIdentityError("live vector config")
        payloads, offset = [], None
        while True:
            batch, offset = qdrant.scroll(
                d["collection"], limit=256, offset=offset, with_payload=True, with_vectors=False
            )
            payloads.extend(Chunk.model_validate(p.payload) for p in batch)
            if offset is None:
                break
        if corpus_identity(payloads, metadata) != identity:
            raise FrozenBackendIdentityError("live payload corpus")
    store = QdrantVectorStore(qdrant, d["collection"], d["dimension"], chunk_model=Chunk)
    pipeline = RetrievalPipeline(
        DenseRetriever(embedding, store),
        BM25Retriever(chunks, **b["sparse"]["parameters"]),
        reranker,
        ContextBuilder(**b["context_builder"]),
        provider_metadata={"backend_id": backend_id, "collection": d["collection"]},
        candidate_top_k=b["candidate_top_k"],
        final_top_k=b["final_top_k"],
        rerank_mode=r["mode"],
        index_manifest=index,
    )
    mode = "LIVE_FROZEN_BACKEND" if verify_live else "SYNTHETIC_MECHANISM_TEST"
    backend = FrozenRetrievalBackend(b, pipeline, chunks, execution_mode=mode)
    backend.guard()
    return backend
