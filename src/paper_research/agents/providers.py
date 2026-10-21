import json
import uuid
import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import replace
from pathlib import Path

from qdrant_client import QdrantClient

from paper_research.agents.research_models import ResearchEvidence
from paper_research.chunking.structural_chunker import StructuralChunker
from paper_research.chunking.types import Chunk
from paper_research.config import Settings
from paper_research.indexing.registry import (
    CorpusTrustError, IndexIdentityError, IndexRegistry, runtime_index_identity,
)
from paper_research.indexing.vector_store import QdrantVectorStore
from paper_research.parsing.types import PaperBlock
from paper_research.retrieval.context_builder import ContextBuilder
from paper_research.retrieval.contextual import ContextualChunk
from paper_research.retrieval.dense import DenseRetriever
from paper_research.retrieval.filters import RetrievalFilter
from paper_research.retrieval.hybrid import HybridRetriever
from paper_research.retrieval.pipeline import build_v2_retriever, retrieval_collection
from paper_research.retrieval.sparse import BM25Retriever
from paper_research.retrieval.trace import JsonlTraceRepository
from paper_research.search.models import SearchRequest
from paper_research.search.service import PaperSearchService


class LocalResearchProvider(ABC):
    @abstractmethod
    def search(self, query: str, paper_ids: list[str] | None, limit: int = 5) -> list[dict]:
        """Return evidence dictionaries for a research sub-question."""


class ArtifactLocalResearchProvider(LocalResearchProvider):
    """Legacy artifact-only local provider.

    This provider remains available for offline tests and explicit fallback
    diagnostics. Production API routing should prefer HybridLocalResearchProvider
    so a successful Deep Research response is not backed by a per-request BM25
    evidence dump.
    """

    def __init__(self, root: Path) -> None:
        self.root = root

    def search(self, query: str, paper_ids: list[str] | None, limit: int = 5) -> list[dict]:
        chunks = self._load_chunks()
        results = BM25Retriever(chunks).retrieve(
            query,
            retrieval_filter=RetrievalFilter(paper_ids=paper_ids or None),
            top_k=limit,
        )
        return [
            {
                "evidence_id": result.chunk.chunk_id,
                "paper_id": result.chunk.paper_id,
                "section_path": result.chunk.section_path,
                "page_start": result.chunk.page_start,
                "page_end": result.chunk.page_end,
                "quote": result.chunk.chunk_text[:1200],
                "score": result.score,
                "source": "local",
            }
            for result in results
        ]

    def _load_chunks(self) -> list[Chunk]:
        chunks: list[Chunk] = []
        chunk_paths = list(self.root.glob("*/paper_chunks.jsonl"))
        indexed_dirs = {path.parent for path in chunk_paths}
        for path in chunk_paths:
            chunks.extend(
                Chunk.model_validate(json.loads(line))
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        for path in self.root.glob("*/paper_blocks.jsonl"):
            if path.parent in indexed_dirs:
                continue
            blocks = [
                PaperBlock.model_validate(json.loads(line))
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            chunks.extend(StructuralChunker().chunk(path.parent.name, blocks))
        return chunks


class HybridLocalResearchProvider(LocalResearchProvider):
    """Production local provider backed by the existing hybrid retrieval stack."""

    def __init__(
        self, settings: Settings, *, require_identity: bool = False, graph_options=None,
    ) -> None:
        from paper_research.providers.factory import build_embedding_provider, build_reranker

        self.settings = settings
        self.require_identity = require_identity
        self.graph_options = graph_options
        if graph_options is not None:
            from paper_research.evidence_graph.runtime import GraphRuntimeError
            if (not require_identity or not settings.graph_rag_enabled
                    or settings.retrieval_v2_enabled):
                raise GraphRuntimeError("GRAPH_RAG_CONFIGURATION_UNSUPPORTED", 422)
        self._registry = IndexRegistry(settings.data_dir / "index_registry.json")
        self._client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
        self._collection = self._resolve_collection()
        self._registry.verify_identity(
            self._collection, runtime_index_identity(settings.provider_metadata, self._collection), self._client,
        )
        self._embedding = build_embedding_provider(settings)
        self._guard_identity()
        self._reranker = build_reranker(settings)
        self._store = QdrantVectorStore(self._client, self._collection, settings.embedding_dimensions,
            chunk_model=ContextualChunk if settings.retrieval_v2_enabled else Chunk)
        with self._admission() as (chunks, _):
            self._set_retriever(chunks)

    def _admission(self):
        return self._registry.query_admission(
            self.settings.active_collection, self._collection,
            runtime_index_identity(self.settings.provider_metadata, self._collection), self._client,
            object_check=self._guard_identity,
            supported=not self.settings.retrieval_v2_enabled,
        )

    def _set_retriever(self, chunks):
        self._chunks = chunks
        self.retriever = HybridRetriever(
            DenseRetriever(self._embedding, self._store), BM25Retriever(chunks), self._reranker,
            ContextBuilder(include_neighbors=False,
                max_characters=self.settings.qa_context_token_budget * 4,
                max_tokens=self.settings.qa_context_token_budget),
            JsonlTraceRepository(self.settings.retrieval_trace_path),
            provider_metadata=self.settings.provider_metadata,
            rerank_input_k=self.settings.rerank_input_k, rerank_output_k=self.settings.rerank_output_k,
        )

    def search(self, query: str, paper_ids: list[str] | None, limit: int = 5) -> list[dict]:
        with self._admission() as (chunks, _):
            self._set_retriever(chunks)
            result = self.retriever.retrieve(
                query,
                RetrievalFilter(paper_ids=paper_ids or None),
                recall_k=max(self.settings.retrieval_recall_k, limit),
                top_k=limit,
                retrieval_scope="paper" if paper_ids else "global",
            )
            return [
                ResearchEvidence(
                    evidence_id=item.chunk_id,
                    paper_id=item.paper_id,
                    section_path=item.section_path,
                    page_start=item.page_start,
                    page_end=item.page_end,
                    text=item.evidence,
                    retrieval_score=item.score,
                    retrieval_sources=next(
                        (
                            list(candidate["retrieval_sources"])
                            for candidate in result.trace.candidate_scores
                            if candidate["chunk_id"] == item.chunk_id
                        ),
                        ["dense", "sparse", "rrf"],
                    ),
                ).model_dump()
                for item in result.context
            ]

    def _resolve_collection(self) -> str:
        try:
            return retrieval_collection(self.settings, self._registry)
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            if self.require_identity:
                raise IndexIdentityError("INDEX_IDENTITY_INVALID") from exc
            raise

    def _guard_identity(self) -> None:
        collection = self._resolve_collection()
        if collection != self._collection or (
            hasattr(self, "_store") and self._store.collection != self._collection
        ):
            raise IndexIdentityError("INDEX_IDENTITY_CONFLICT", ["collection_name"], status_code=409)
        expected = runtime_index_identity(self.settings.provider_metadata, collection)
        self._registry.verify_identity(collection, expected, self._client)
        embedding = getattr(getattr(self, "retriever", None), "dense", None)
        embedding = embedding.embedding if embedding is not None else self._embedding
        store = getattr(getattr(self, "retriever", None), "dense", None)
        if store is not None and store.vector_store is not self._store:
            raise IndexIdentityError("INDEX_IDENTITY_CONFLICT", ["collection_name"], status_code=409)
        actual = runtime_index_identity(
            {**self.settings.provider_metadata,
             "embedding_provider": embedding.provider_name,
             "embedding_model": embedding.model_name,
             "embedding_revision": embedding.revision,
             "embedding_dimension": embedding.dimensions}, collection,
        )
        differences = [key for key in actual if actual[key] != expected[key]]
        if differences:
            raise IndexIdentityError("INDEX_IDENTITY_CONFLICT", differences, status_code=409)

    def retrieve_readonly_observation(
        self, query: str, paper_ids: list[str] | None, limit: int = 5,
    ):
        # Local imports reuse the D1 API boundary without introducing an import cycle.
        from paper_research.api.routes.rag import (
            HybridRetrievalEvidenceResponse, _load_block_pages, _with_block_page_map,
        )
        from paper_research.generation.evidence_contract_adapter import (
            build_readonly_evidence_contract, evidence_contract_failure,
        )

        if not self.require_identity:
            raise IndexIdentityError("INDEX_IDENTITY_MISSING")
        if self.settings.retrieval_v2_enabled:
            raise ValueError("READONLY_RETRIEVAL_MODE_UNSUPPORTED")
        with self._admission() as (chunks, _):
            self._set_retriever(chunks)
            retrieval_filter = RetrievalFilter(paper_ids=paper_ids or None)
            trunk = self.retriever.common_trunk(
                query, retrieval_filter,
                recall_k=max(self.settings.retrieval_recall_k, limit),
                retrieval_scope="paper" if paper_ids else "global",
            )
            root = self.settings.parsed_papers_dir.resolve()
            pages_by_paper = {}
            hydrated = {}
            for candidate in trunk.candidates:
                chunk = candidate.chunk
                if chunk.paper_id not in pages_by_paper:
                    pages = {}
                    try:
                        if str(uuid.UUID(chunk.paper_id)) == chunk.paper_id:
                            folder = (root / chunk.paper_id).resolve()
                            if folder.is_relative_to(root):
                                pages = _load_block_pages(folder)
                    except (OSError, ValueError, TypeError, AttributeError):
                        pass  # D1 validates missing/corrupt canonical sources after finalization.
                    pages_by_paper[chunk.paper_id] = pages
                hydrated[id(candidate)] = replace(
                    candidate, chunk=_with_block_page_map(chunk, pages_by_paper[chunk.paper_id]),
                )
            trunk = replace(
                trunk,
                candidates=[hydrated[id(item)] for item in trunk.candidates],
                fused_results=[hydrated.get(id(item), item) for item in trunk.fused_results],
            )
            result = self.retriever.finalize(trunk, retrieval_filter=retrieval_filter, top_k=limit)
            try:
                sidecar = build_readonly_evidence_contract(
                    query, result.context, result.trace, parsed_root=root,
                )
            except Exception:
                sidecar = evidence_contract_failure(query, result.context, result.trace)
            return HybridRetrievalEvidenceResponse(
                context=result.context, trace=result.trace, evidence_contract=sidecar,
            )

    def _active_binding(self, paper_ids: list[str] | None) -> tuple[dict, str]:
        active = self._registry.read()["indexes"][self._collection]["trust_binding"]
        scope = list(dict.fromkeys(paper_ids or []))
        binding = {
            "schema_version": "d6-c2-generation-binding-v1",
            "generation_id": active["generation_id"],
            "inventory_digest": active["inventory_digest"],
            "seal_digest": active["seal_digest"],
            "index_identity": dict(active["index_identity"]),
            "collection_identity": {
                "logical_collection": self.settings.active_collection,
                "collection_name": self._collection,
            },
            "source_scope": {"kind": "paper" if scope else "global", "paper_ids": scope},
        }
        raw = json.dumps(binding, sort_keys=True, ensure_ascii=True,
                         separators=(",", ":"), allow_nan=False).encode("utf8")
        return binding, hashlib.sha256(raw).hexdigest()

    @staticmethod
    def _active_chunk_digest(chunk: Chunk) -> str:
        fields = ("paper_id", "chunk_id", "block_ids", "section_path", "block_type",
                  "page_start", "page_end", "block_page_map", "chunk_text", "token_count")
        value = {field: getattr(chunk, field) for field in fields}
        raw = json.dumps(value, sort_keys=True, ensure_ascii=True,
                         separators=(",", ":"), allow_nan=False).encode("utf8")
        return hashlib.sha256(raw).hexdigest()

    def _active_membership(self, chunks: list[Chunk]) -> dict[tuple[str, str], Chunk]:
        pairs: dict[tuple[str, str], Chunk] = {}
        owners: dict[str, str] = {}
        for chunk in chunks:
            if chunk.chunk_id in owners and owners[chunk.chunk_id] != chunk.paper_id:
                raise CorpusTrustError("CANONICAL_SOURCE_IDENTITY_CONFLICT")
            owners[chunk.chunk_id] = chunk.paper_id
            key = (chunk.paper_id, chunk.chunk_id)
            if key in pairs:
                raise CorpusTrustError("CANONICAL_SOURCE_IDENTITY_CONFLICT")
            pairs[key] = chunk
        return pairs

    def active_binding(self, paper_ids: list[str] | None) -> tuple[dict, str]:
        """Freeze the D6 generation before an Agent decision or tool call."""
        with self._admission() as (chunks, _):
            self._active_membership(chunks)
            return self._active_binding(paper_ids)

    def retrieve_active_observation(
        self, query: str, paper_ids: list[str] | None, limit: int, tool_call_id: str,
        state=None, *, retrieval_paper_ids: list[str] | None = None,
    ):
        from paper_research.agents.research_agent.models import (
            CanonicalEvidenceItem, CanonicalRetrievalBatch, CanonicalRetrievalOccurrence,
            CanonicalTextProof,
        )
        from paper_research.api.routes.rag import _load_block_pages, _with_block_page_map
        from paper_research.generation.evidence_contract_adapter import (
            _read_paper, _result, EvidenceAdapterError, build_readonly_evidence_contract, evidence_contract_failure, RetrievalIdentity,
        )

        if not self.require_identity or self.settings.retrieval_v2_enabled:
            raise CorpusTrustError("CORPUS_TRUST_SCOPE_UNSUPPORTED")
        if state is None or state.last_action is None:
            raise CorpusTrustError("CANONICAL_PROVENANCE_INVALID")
        if retrieval_paper_ids is not None and (
                not isinstance(retrieval_paper_ids, list) or not retrieval_paper_ids
                or any(not isinstance(p, str) for p in retrieval_paper_ids)
                or len(set(retrieval_paper_ids)) != len(retrieval_paper_ids)
                or not set(retrieval_paper_ids) <= set(paper_ids or [])):
            raise CorpusTrustError("CANONICAL_PROVENANCE_INVALID")
        with self._admission() as (chunks, _):
            membership = self._active_membership(chunks)
            binding, binding_digest = self._active_binding(paper_ids)
            if state is not None:
                self.validate_active_snapshot(state, _pinned=(chunks, binding, binding_digest))
            self._set_retriever(chunks)
            retrieval_filter = RetrievalFilter(paper_ids=retrieval_paper_ids or paper_ids or None)
            trunk = self.retriever.common_trunk(
                query, retrieval_filter,
                recall_k=max(self.settings.retrieval_recall_k, limit),
                retrieval_scope="paper" if paper_ids else "global",
            )
            rejected = {}
            for candidate in (*trunk.dense_results, *trunk.sparse_results, *trunk.candidates):
                observed = candidate.chunk
                pair = observed.paper_id, observed.chunk_id
                canonical = membership.get(pair)
                fields = ["paper_id", "chunk_id"] if canonical is None else [
                    field for field in ("chunk_text", "block_ids", "section_path", "page_start", "page_end")
                    if getattr(observed, field) != getattr(canonical, field)]
                if canonical is not None and observed.block_page_map and observed.block_page_map != canonical.block_page_map:
                    fields.append("block_page_map")
                if fields:
                    rejected.setdefault(pair, set()).update(fields)
            if rejected:
                # A valid physical corpus remains admitted. Reject every fused
                # occurrence of the offending pair; sparse cannot hide bad dense core.
                trunk = replace(trunk,
                    candidates=[c for c in trunk.candidates if (c.chunk.paper_id, c.chunk.chunk_id) not in rejected],
                    fused_results=[c for c in trunk.fused_results if (c.chunk.paper_id, c.chunk.chunk_id) not in rejected])
            root = self.settings.parsed_papers_dir.resolve()
            pages_by_paper: dict[str, dict] = {}
            hydrated = {}
            for candidate in trunk.candidates:
                chunk = candidate.chunk
                if chunk.paper_id not in pages_by_paper:
                    folder = (root / chunk.paper_id).resolve()
                    if not folder.is_relative_to(root):
                        raise CorpusTrustError("CANONICAL_SOURCE_CORPUS_INVALID")
                    pages_by_paper[chunk.paper_id] = _load_block_pages(folder)
                hydrated[id(candidate)] = replace(
                    candidate, chunk=_with_block_page_map(chunk, pages_by_paper[chunk.paper_id]),
                )
            trunk = replace(trunk,
                candidates=[hydrated[id(item)] for item in trunk.candidates],
                fused_results=[hydrated.get(id(item), item) for item in trunk.fused_results],
            )
            result = self.retriever.finalize(trunk, retrieval_filter=retrieval_filter, top_k=limit)
            if getattr(self, "graph_options", None) is not None:
                from paper_research.retrieval.graph import augment
                result = augment(result, query, retrieval_paper_ids or paper_ids, limit,
                                 self.graph_options, settings=self.settings,
                                 registry=self._registry,
                                 chunks=chunks, collection=self._collection)
                JsonlTraceRepository(self.settings.retrieval_trace_path).append(result.trace)
            try:
                sidecar = build_readonly_evidence_contract(
                    query, result.context, result.trace, parsed_root=root,
                )
            except IndexIdentityError:
                raise
            except Exception:
                sidecar = evidence_contract_failure(query, result.context, result.trace)
            if rejected:
                errors = [EvidenceAdapterError(code="PROVENANCE_INVALID", trace_id=result.trace.trace_id,
                    item_index=None, paper_id=pair[0], chunk_id=pair[1], fields=sorted(fields))
                    for pair, fields in rejected.items()]
                sidecar = _result(query, result.trace, sidecar.evidence, [*sidecar.errors, *errors],
                    sidecar.coverage[0].invalid_evidence_count + len(rejected),
                    failed=any(error.code == "EVIDENCE_ADAPTER_ERROR" for error in sidecar.errors))
            identity = RetrievalIdentity(
                **{key: getattr(result.trace, key) for key in RetrievalIdentity.model_fields}
            )
            selected = {(item.paper_id, item.chunk_id): (rank, item)
                        for rank, item in enumerate(result.context, 1)}
            canonical_items = []
            for structured in sidecar.evidence:
                item, provenance = structured.item, structured.provenance
                rank, context = selected[(item.paper_id, item.chunk_id)]
                paper_chunks, _ = _read_paper(root, item.paper_id, identity)
                canonical = paper_chunks[item.chunk_id]
                folder = (root / item.paper_id).resolve()
                chunk_path = folder / f"paper_chunks.{identity.index_version}.jsonl"
                source_files = {
                    "versioned_chunks": chunk_path,
                    "manifest": chunk_path.with_name(f"{chunk_path.stem}_manifest.json"),
                    "paper_blocks": folder / "paper_blocks.jsonl",
                }
                file_hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest()
                               for name, path in source_files.items()}
                canonical_text = canonical.chunk_text
                display = context.evidence[:800]
                digest = lambda value: hashlib.sha256(value.encode("utf8")).hexdigest()
                source_hash = hashlib.sha256(json.dumps(
                    provenance.model_dump(mode="json"), sort_keys=True,
                    ensure_ascii=True, separators=(",", ":"), allow_nan=False,
                ).encode("utf8")).hexdigest()
                proof = CanonicalTextProof(
                    canonical_chunk_sha256=self._active_chunk_digest(canonical),
                    canonical_text_sha256=digest(canonical_text),
                    canonical_text_length=len(canonical_text),
                    context_text_sha256=digest(context.evidence),
                    context_text_length=len(context.evidence),
                    display_text_sha256=digest(display),
                    display_text_length=len(display),
                    canonical_source_sha256=source_hash,
                    source_artifact_sha256=file_hashes,
                )
                occurrence = CanonicalRetrievalOccurrence(
                    task_id=state.task_id,
                    tool_call_id=tool_call_id, trace_id=result.trace.trace_id,
                    original_context_rank=rank, step=state.step_count, query=query,
                    paper_scope=list(binding["source_scope"]["paper_ids"]),
                    target_subquestion_ids=list(state.last_action.target_subquestion_ids),
                    retrieval_score=context.score,
                    retrieval_sources=[name for name, entries in (
                        ("dense", result.trace.dense_results), ("sparse", result.trace.sparse_results))
                        if any((entry.paper_id, entry.chunk_id) == (item.paper_id, item.chunk_id)
                               for entry in entries)],
                    observed_text=display, display_text_sha256=proof.display_text_sha256,
                    context_text_sha256=proof.context_text_sha256,
                    context_text_length=proof.context_text_length,
                    generation_binding_digest=binding_digest,
                )
                canonical_items.append(CanonicalEvidenceItem(
                    **{**item.model_dump(), "text_or_reference": display,
                       "discovered_by_tool": tool_call_id,
                       "discovered_at_step": state.step_count,
                       "generation_binding_digest": binding_digest,
                       "provenance": provenance.model_dump(mode="json"),
                       "retrieval_identity": identity.model_dump(mode="json"),
                       "text_proof": proof, "occurrences": [occurrence]},
                ))
            if getattr(self, "graph_options", None) is not None:
                if sidecar.errors:
                    from paper_research.evidence_graph.runtime import GraphRuntimeError
                    raise GraphRuntimeError("GRAPH_CANONICAL_CITATION_INVALID")
                if state.graph_rag is None or state.graph_rag.options != self.graph_options:
                    from paper_research.evidence_graph.runtime import GraphRuntimeError
                    raise GraphRuntimeError("GRAPH_CHECKPOINT_MODE_MISMATCH")
                state.graph_rag.requests.append({"tool_call_id": tool_call_id, **result.graph_rag})
            return CanonicalRetrievalBatch(
                items=canonical_items,
                errors=[error.model_dump(mode="json") for error in sidecar.errors],
                gaps=[gap.model_dump(mode="json") for gap in sidecar.gaps],
                coverage=[coverage.model_dump(mode="json") for coverage in sidecar.coverage],
                trace_id=result.trace.trace_id,
                generation_binding=binding,
                generation_binding_digest=binding_digest,
            )

    def validate_active_snapshot(self, state, *, _pinned=None) -> None:
        """Recheck mode, D6 and stored canonical prefixes without retrieval."""
        if ((state.graph_rag is not None) != (getattr(self, "graph_options", None) is not None)
                or (state.graph_rag is not None and state.graph_rag.options != self.graph_options)):
            from paper_research.evidence_graph.runtime import GraphRuntimeError
            raise GraphRuntimeError("GRAPH_CHECKPOINT_MODE_MISMATCH")
        from contextlib import nullcontext
        from paper_research.agents.research_agent.models import CanonicalEvidenceItem
        from paper_research.generation.evidence_contract_adapter import (
            _read_paper, _SourceError, EvidenceProvenance, RetrievalIdentity,
        )

        receipt = state.evidence_provenance.get("p1_active_canonical_provenance_v2")
        if not isinstance(receipt, dict):
            raise CorpusTrustError("CANONICAL_PROVENANCE_INVALID")
        with (self._admission() if _pinned is None else nullcontext((_pinned[0], None))) as (chunks, _):
            membership = self._active_membership(chunks)
            binding, digest = self._active_binding(state.paper_ids) if _pinned is None else _pinned[1:]
            if (receipt.get("generation_binding") != binding
                    or receipt.get("generation_binding_digest") != digest):
                raise CorpusTrustError("CANONICAL_GENERATION_STALE", status_code=409)
            root = self.settings.parsed_papers_dir.resolve()
            papers = {}
            for key, item in state.evidence_state.items.items():
                if not isinstance(item, CanonicalEvidenceItem) or key != item.stable_key:
                    raise CorpusTrustError("CANONICAL_PROVENANCE_INVALID")
                pair = item.paper_id, item.chunk_id
                if pair not in membership or item.generation_binding_digest != digest:
                    raise CorpusTrustError("CANONICAL_PROVENANCE_INVALID")
                identity = RetrievalIdentity.model_validate(item.retrieval_identity)
                expected_identity = {**binding["index_identity"], "collection": self._collection}
                if any(getattr(identity, field) != expected_identity[field]
                       for field in RetrievalIdentity.model_fields):
                    raise CorpusTrustError("CANONICAL_PROVENANCE_INVALID")
                if item.paper_id not in papers:
                    try:
                        papers[item.paper_id] = _read_paper(root, item.paper_id, identity)
                    except _SourceError as exc:
                        raise CorpusTrustError("CANONICAL_PROVENANCE_UNAVAILABLE") from exc
                folder = (root / item.paper_id).resolve()
                chunk_path = folder / f"paper_chunks.{identity.index_version}.jsonl"
                files = {"versioned_chunks": chunk_path,
                         "manifest": chunk_path.with_name(f"{chunk_path.stem}_manifest.json"),
                         "paper_blocks": folder / "paper_blocks.jsonl"}
                try:
                    changed = any(hashlib.sha256(path.read_bytes()).hexdigest()
                                  != item.text_proof.source_artifact_sha256.get(name)
                                  for name, path in files.items())
                except OSError as exc:
                    raise CorpusTrustError("CANONICAL_PROVENANCE_UNAVAILABLE") from exc
                if changed:
                    raise CorpusTrustError("CANONICAL_PROVENANCE_STALE", status_code=409)
                canonical = papers[item.paper_id][0].get(item.chunk_id)
                if canonical is None or self._active_chunk_digest(canonical) != item.text_proof.canonical_chunk_sha256:
                    raise CorpusTrustError("CANONICAL_PROVENANCE_INVALID")
                text = canonical.chunk_text
                proof = item.text_proof
                provenance = EvidenceProvenance.model_validate(item.provenance)
                encoded = json.dumps(item.provenance, sort_keys=True, ensure_ascii=True,
                                     separators=(",", ":"), allow_nan=False).encode("utf8")
                blocks = papers[item.paper_id][1]
                sources = [(block_id, blocks[block_id].page_start)
                           for block_id in canonical.block_ids]
                expected_map = dict(sources)
                if (hashlib.sha256(encoded).hexdigest() != proof.canonical_source_sha256
                        or provenance.paper_id != canonical.paper_id
                        or provenance.chunk_id != canonical.chunk_id
                        or provenance.block_ids != canonical.block_ids
                        or provenance.block_page_map != expected_map
                        or (canonical.block_page_map is not None
                            and canonical.block_page_map != expected_map)
                        or provenance.page_start != canonical.page_start
                        or provenance.page_end != canonical.page_end
                        or provenance.section_path != canonical.section_path
                        or [(source.block_id, source.page) for source in provenance.sources] != sources
                        or any(source.page_end != blocks[source.block_id].page_end
                               for source in provenance.sources)):
                    raise CorpusTrustError("CANONICAL_PROVENANCE_INVALID")
                prefix = text[:proof.context_text_length]
                display = prefix[:800]
                if (hashlib.sha256(text.encode("utf8")).hexdigest() != proof.canonical_text_sha256
                        or len(text) != proof.canonical_text_length
                        or hashlib.sha256(prefix.encode("utf8")).hexdigest() != proof.context_text_sha256
                        or hashlib.sha256(display.encode("utf8")).hexdigest() != proof.display_text_sha256
                        or item.text_or_reference != display):
                    raise CorpusTrustError("CANONICAL_PROVENANCE_INVALID")
                for occurrence in item.occurrences:
                    prefix = text[:occurrence.context_text_length]
                    if (occurrence.task_id != state.task_id
                            or occurrence.paper_scope != receipt["paper_scope"]
                            or len(prefix) != occurrence.context_text_length
                            or occurrence.observed_text != prefix[:800]
                            or hashlib.sha256(prefix.encode("utf8")).hexdigest()
                                != occurrence.context_text_sha256):
                        raise CorpusTrustError("CANONICAL_PROVENANCE_INVALID")

    def draft_evidence(self, state) -> list[dict]:
        """Read full chunks under the existing D6 lease; never mutate C2 display evidence."""
        if not self.require_identity or self.settings.retrieval_v2_enabled:
            raise CorpusTrustError("CORPUS_TRUST_SCOPE_UNSUPPORTED")
        with self._admission() as (chunks, _):
            binding, digest = self._active_binding(state.paper_ids)
            self.validate_active_snapshot(state, _pinned=(chunks, binding, digest))
            membership = self._active_membership(chunks)
            evidence = []
            for item in state.evidence_state.items.values():
                chunk = membership[(item.paper_id, item.chunk_id)]
                if (self._active_chunk_digest(chunk) != item.text_proof.canonical_chunk_sha256
                        or hashlib.sha256(chunk.chunk_text.encode("utf8")).hexdigest()
                        != item.text_proof.canonical_text_sha256):
                    raise CorpusTrustError("CANONICAL_PROVENANCE_INVALID")
                evidence.append({**item.model_dump(mode="json"), "model_text": chunk.chunk_text})
            return evidence

    def _load_chunks(self, root: Path) -> list[Chunk]:
        filename = f"paper_chunks.{self.settings.index_version}.jsonl"
        paths = list(root.glob(f"*/{filename}"))
        if not paths:
            paths = list(root.glob("*/paper_chunks.jsonl"))
        chunks: list[Chunk] = []
        for path in paths:
            if path.exists():
                chunks.extend(
                    Chunk.model_validate(json.loads(line))
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                )
        return chunks


class ExternalResearchProvider(ABC):
    @abstractmethod
    def search(self, query: str, limit: int = 10) -> list[dict]:
        """Return normalized external paper candidates."""


class SearchServiceExternalProvider(ExternalResearchProvider):
    def __init__(self, service: PaperSearchService) -> None:
        self.service = service

    def search(self, query: str, limit: int = 10) -> list[dict]:
        response = self.service.search(
            SearchRequest(query=query, limit=limit, open_access_only=True)
        )
        return [candidate.model_dump(mode="json") for candidate in response.candidates]
