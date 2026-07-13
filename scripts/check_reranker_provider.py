"""Safely verify the configured reranker without printing credentials or document text."""

import json

from paper_research.chunking.types import Chunk
from paper_research.config import Settings
from paper_research.providers.factory import build_reranker
from paper_research.retrieval.fusion import FusedResult


def candidate(identifier: str, text: str, score: float) -> FusedResult:
    return FusedResult(
        chunk=Chunk(
            chunk_id=identifier,
            paper_id="connectivity-check",
            block_ids=[f"block-{identifier}"],
            section_path=["Check"],
            block_type="paragraph",
            page_start=1,
            page_end=1,
            chunk_text=text,
            token_count=len(text.split()),
        ),
        score=score,
    )


def main() -> None:
    settings = Settings()
    if not settings.rerank_enabled or settings.rerank_provider != "jina":
        raise RuntimeError("RERANK_ENABLED=true and RERANK_PROVIDER=jina are required")
    if settings.rerank_allow_fallback:
        raise RuntimeError("RERANK_ALLOW_FALLBACK=false is required for the real check")
    reranker = build_reranker(settings)
    candidates = [
        candidate("method", "Low-rank matrices adapt a frozen language model.", 0.03),
        candidate("unrelated", "Marine weather observations from coastal stations.", 0.04),
        candidate("background", "Parameter-efficient adaptation reduces trainable weights.", 0.02),
    ]
    outcome = reranker.rerank_with_trace(
        "How does low-rank adaptation reduce trainable parameters?",
        candidates,
        len(candidates),
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "provider": outcome.provider,
                "model": outcome.model,
                "input_count": outcome.input_count,
                "output_count": outcome.output_count,
                "ranked_identifiers": [item.chunk.chunk_id for item in outcome.results],
                "latency_ms": outcome.latency_ms,
                "api_request_count": outcome.api_request_count,
                "fallback_occurred": outcome.fallback_occurred,
            }
        )
    )


if __name__ == "__main__":
    main()
