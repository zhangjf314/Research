import json
from abc import ABC, abstractmethod
from pathlib import Path

from qdrant_client import QdrantClient

from paper_research.agents.research_models import ResearchEvidence
from paper_research.chunking.structural_chunker import StructuralChunker
from paper_research.chunking.types import Chunk
from paper_research.config import Settings
from paper_research.indexing.registry import IndexRegistry
from paper_research.indexing.vector_store import QdrantVectorStore
from paper_research.parsing.types import PaperBlock
from paper_research.retrieval.context_builder import ContextBuilder
from paper_research.retrieval.dense import DenseRetriever
from paper_research.retrieval.filters import RetrievalFilter
from paper_research.retrieval.hybrid import HybridRetriever
from paper_research.retrieval.sparse import BM25Retriever
from paper_research.retrieval.trace import JsonlTraceRepository
from paper_research.search.models import SearchRequest
from paper_research.search.service import PaperSearchService


class LocalResearchProvider(ABC):
    @abstractmethod
    def search(
        self, query: str, paper_ids: list[str] | None, limit: int = 5
    ) -> list[dict]:
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

    def search(
        self, query: str, paper_ids: list[str] | None, limit: int = 5
    ) -> list[dict]:
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

    def __init__(self, settings: Settings) -> None:
        from paper_research.providers.factory import build_embedding_provider, build_reranker

        self.settings = settings
        self._chunks = self._load_chunks(settings.parsed_papers_dir)
        if not self._chunks:
            raise RuntimeError("no indexed chunk files found for hybrid research")
        embedding = build_embedding_provider(settings)
        reranker = build_reranker(settings)
        store = QdrantVectorStore(
            QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key),
            IndexRegistry(settings.data_dir / "index_registry.json").resolve(
                settings.active_collection
            ),
            settings.embedding_dimensions,
        )
        self.retriever = HybridRetriever(
            DenseRetriever(embedding, store),
            BM25Retriever(self._chunks),
            reranker,
            ContextBuilder(
                include_neighbors=False,
                max_characters=settings.qa_context_token_budget * 4,
                max_tokens=settings.qa_context_token_budget,
            ),
            JsonlTraceRepository(settings.retrieval_trace_path),
            provider_metadata=settings.provider_metadata,
            rerank_input_k=settings.rerank_input_k,
            rerank_output_k=settings.rerank_output_k,
        )

    def search(
        self, query: str, paper_ids: list[str] | None, limit: int = 5
    ) -> list[dict]:
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
                retrieval_sources=["dense", "sparse", "rrf"],
            ).model_dump()
            for item in result.context
        ]

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
