"""Minimal embedding connectivity check without printing vectors or credentials."""

import json
import time

from paper_research.config import Settings
from paper_research.providers.factory import build_embedding_provider


def main() -> None:
    settings = Settings()
    provider = build_embedding_provider(settings)
    started = time.perf_counter()
    query = provider.embed_query("How does dense retrieval find relevant papers?")
    documents = provider.embed_documents(
        [
            "Dense retrieval maps queries and documents into a shared vector space.",
            "Structural chunking preserves section and source-page metadata.",
        ]
    )
    elapsed_ms = (time.perf_counter() - started) * 1000
    if len(query) != provider.dimensions:
        raise RuntimeError("query embedding dimension check failed")
    if len(documents) != 2 or any(len(vector) != provider.dimensions for vector in documents):
        raise RuntimeError("document embedding dimension check failed")
    print(
        json.dumps(
            {
                "status": "PASS",
                "provider": provider.provider_name,
                "model": provider.model_name,
                "revision": provider.revision,
                "dimension": provider.dimensions,
                "query_vector_count": 1,
                "document_vector_count": len(documents),
                "elapsed_ms": round(elapsed_ms, 3),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
