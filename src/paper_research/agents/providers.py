import json
from abc import ABC, abstractmethod
from pathlib import Path

from paper_research.chunking.structural_chunker import StructuralChunker
from paper_research.chunking.types import Chunk
from paper_research.parsing.types import PaperBlock
from paper_research.retrieval.filters import RetrievalFilter
from paper_research.retrieval.sparse import BM25Retriever
from paper_research.search.models import SearchRequest
from paper_research.search.service import PaperSearchService


class LocalResearchProvider(ABC):
    @abstractmethod
    def search(
        self, query: str, paper_ids: list[str] | None, limit: int = 5
    ) -> list[dict]:
        """Return evidence dictionaries for a research sub-question."""


class ArtifactLocalResearchProvider(LocalResearchProvider):
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
