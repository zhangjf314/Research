from paper_research.chunking.tokenizer import count_tokens, token_windows
from paper_research.chunking.types import Chunk
from paper_research.parsing.types import PaperBlock


class FixedTokenChunker:
    def __init__(self, max_tokens: int = 400, overlap_tokens: int = 60) -> None:
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def chunk(self, paper_id: str, blocks: list[PaperBlock]) -> list[Chunk]:
        text = "\n".join(block.text for block in blocks)
        page_start = min((block.page_start for block in blocks), default=1)
        page_end = max((block.page_end for block in blocks), default=1)
        return [
            Chunk(
                paper_id=paper_id,
                block_ids=[],
                block_type="mixed",
                page_start=page_start,
                page_end=page_end,
                chunk_text=window,
                token_count=count_tokens(window),
            )
            for window in token_windows(text, self.max_tokens, self.overlap_tokens)
        ]
