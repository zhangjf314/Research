from qdrant_client import QdrantClient

from paper_research.chunking.fixed_chunker import FixedTokenChunker
from paper_research.chunking.structural_chunker import StructuralChunker
from paper_research.generation.qa_service import QAService
from paper_research.indexing.embedding import HashEmbeddingProvider
from paper_research.indexing.vector_store import QdrantVectorStore
from paper_research.parsing.types import BoundingBox, PaperBlock
from paper_research.retrieval.dense import DenseRetriever


def block(block_id: str, text: str, page: int, section: str) -> PaperBlock:
    return PaperBlock(
        block_id=block_id,
        block_type="paragraph",
        section_path=[section],
        page_start=page,
        page_end=page,
        block_index=int(block_id[1:]),
        text=text,
        bbox=BoundingBox(x0=1, y0=1, x1=20, y1=20),
    )


def test_structural_chunking_preserves_evidence_metadata() -> None:
    blocks = [
        block("b1", "transformers use self attention for sequence modeling", 2, "Method"),
        block("b2", "experiments evaluate translation quality with BLEU", 5, "Experiments"),
    ]

    chunks = StructuralChunker(max_tokens=5, overlap_tokens=1).chunk("paper-1", blocks)

    assert chunks[0].paper_id == "paper-1"
    assert chunks[0].section_path == ["Method"]
    assert chunks[0].page_start == 2
    assert chunks[0].block_ids == ["b1"]
    assert chunks[0].next_context is not None
    assert all(chunk.token_count <= 5 for chunk in chunks)


def test_fixed_chunker_is_available_as_baseline() -> None:
    chunks = FixedTokenChunker(max_tokens=4, overlap_tokens=1).chunk(
        "paper-1", [block("b1", "one two three four five six", 1, "Body")]
    )
    assert len(chunks) == 2
    assert all(chunk.block_type == "mixed" for chunk in chunks)


def test_dense_retrieval_and_qa_include_page_citation() -> None:
    embedding = HashEmbeddingProvider(dimensions=64)
    store = QdrantVectorStore(QdrantClient(":memory:"), "chunks", 64)
    chunks = StructuralChunker().chunk(
        "paper-1",
        [block("b1", "self attention computes relationships between all tokens", 3, "Method")],
    )
    store.upsert(chunks, embedding.embed([item.chunk_text for item in chunks]))
    service = QAService(DenseRetriever(embedding, store), score_threshold=0.0)

    answer = service.answer("How does self attention relate tokens?", ["paper-1"])

    assert answer.refused is False
    assert answer.citations[0].page_start == 3
    assert answer.citations[0].section == "Method"
    assert "#page=3" in answer.citations[0].pdf_url


def test_qa_refuses_when_no_evidence_passes_threshold() -> None:
    embedding = HashEmbeddingProvider(dimensions=32)
    store = QdrantVectorStore(QdrantClient(":memory:"), "empty", 32)
    store.ensure_collection()

    answer = QAService(DenseRetriever(embedding, store), score_threshold=0.9).answer(
        "unsupported claim", None
    )

    assert answer.refused is True
    assert answer.citations == []
