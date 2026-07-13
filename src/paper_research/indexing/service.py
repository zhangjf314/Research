import json
from pathlib import Path

from paper_research.chunking.structural_chunker import StructuralChunker
from paper_research.chunking.types import Chunk
from paper_research.indexing.embedding import EmbeddingProvider
from paper_research.indexing.vector_store import QdrantVectorStore
from paper_research.parsing.types import PaperBlock


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

    def index(
        self,
        paper_id: str,
        blocks_path: Path,
        chunks_path: Path,
        *,
        metadata: dict[str, object] | None = None,
    ) -> list[Chunk]:
        blocks = [
            PaperBlock.model_validate(json.loads(line))
            for line in blocks_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        chunks = self.chunker.chunk(paper_id, blocks)
        vectors = self.embedding.embed([chunk.chunk_text for chunk in chunks])
        self.vector_store.upsert(chunks, vectors)
        chunks_path.parent.mkdir(parents=True, exist_ok=True)
        with chunks_path.open("w", encoding="utf-8", newline="\n") as stream:
            for chunk in chunks:
                stream.write(json.dumps(chunk.model_dump(), ensure_ascii=False) + "\n")
        if metadata:
            manifest_path = chunks_path.with_name(f"{chunks_path.stem}_manifest.json")
            manifest_path.write_text(
                json.dumps(
                    {
                        **metadata,
                        "paper_id": paper_id,
                        "chunk_count": len(chunks),
                        "chunks_path": str(chunks_path),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        return chunks
