from paper_research.chunking.tokenizer import count_tokens, token_windows
from paper_research.chunking.types import Chunk
from paper_research.parsing.types import PaperBlock


class StructuralChunker:
    def __init__(self, max_tokens: int = 400, overlap_tokens: int = 60) -> None:
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def chunk(self, paper_id: str, blocks: list[PaperBlock]) -> list[Chunk]:
        chunks: list[Chunk] = []
        buffer: list[PaperBlock] = []

        def flush() -> None:
            if not buffer:
                return
            text = "\n".join(item.text for item in buffer)
            chunks.append(self._make_chunk(paper_id, buffer, text))
            buffer.clear()

        for block in blocks:
            if block.block_type == "heading":
                flush()
                continue
            block_tokens = count_tokens(block.text)
            if block_tokens > self.max_tokens:
                flush()
                for window in token_windows(
                    block.text, self.max_tokens, self.overlap_tokens
                ):
                    chunks.append(self._make_chunk(paper_id, [block], window))
                continue
            same_structure = not buffer or (
                buffer[-1].section_path == block.section_path
                and buffer[-1].block_type == block.block_type
            )
            combined_tokens = count_tokens("\n".join(item.text for item in [*buffer, block]))
            if buffer and (not same_structure or combined_tokens > self.max_tokens):
                flush()
            buffer.append(block)
        flush()
        self._enrich_neighbors(chunks)
        return chunks

    @staticmethod
    def _make_chunk(paper_id: str, blocks: list[PaperBlock], text: str) -> Chunk:
        first = blocks[0]
        return Chunk(
            paper_id=paper_id,
            block_ids=[block.block_id for block in blocks],
            section_path=first.section_path,
            block_type=first.block_type,
            page_start=min(block.page_start for block in blocks),
            page_end=max(block.page_end for block in blocks),
            chunk_text=text,
            parent_context=" > ".join(first.section_path) or None,
            token_count=count_tokens(text),
        )

    @staticmethod
    def _enrich_neighbors(chunks: list[Chunk]) -> None:
        for index, chunk in enumerate(chunks):
            chunk.previous_context = chunks[index - 1].chunk_text if index else None
            chunk.next_context = chunks[index + 1].chunk_text if index + 1 < len(chunks) else None
