"""Deterministic index text; raw chunk_text remains the citation source."""

import json
from pathlib import Path

from pydantic import Field

from paper_research.chunking.types import Chunk

CONTEXTUAL_VERSION = "deterministic-context-v1"


class ContextualChunk(Chunk):
    contextual_text: str = ""
    contextual_metadata: dict[str, object] = Field(default_factory=dict)


def contextualize(chunk: Chunk, metadata: dict[str, object] | None = None) -> ContextualChunk:
    metadata = metadata or {}
    lines = [f"Paper: {chunk.paper_id}"]
    for key in ("title", "year", "venue", "doi", "arxiv_id"):
        if metadata.get(key):
            lines.append(f"{key.title()}: {metadata[key]}")
    if chunk.section_path:
        lines.append(f"Section: {' > '.join(chunk.section_path)}")
    lines.append(f"Pages: {chunk.page_start}-{chunk.page_end}")
    return ContextualChunk(
        **chunk.model_dump(exclude={"contextual_text", "contextual_metadata"}),
        contextual_text="\n".join(lines) + "\n\nOriginal text:\n" + chunk.chunk_text,
        contextual_metadata=dict(metadata),
    )


def contextualize_corpus(chunks: list[Chunk], parsed_root: Path) -> list[ContextualChunk]:
    by_paper: dict[str, dict[str, object]] = {}
    for paper_id in {chunk.paper_id for chunk in chunks}:
        path = parsed_root / paper_id / "paper_metadata.json"
        by_paper[paper_id] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    return [contextualize(chunk, by_paper[chunk.paper_id]) for chunk in chunks]
