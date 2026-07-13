"""Minimal credential-safe SiliconFlow structured QA connectivity check."""

import json

from paper_research.config import Settings
from paper_research.generation.qa_service import QAService
from paper_research.providers.factory import build_llm_provider
from paper_research.retrieval.context_builder import ContextItem


def main() -> None:
    settings = Settings()
    if settings.llm_provider != "siliconflow" or settings.llm_model != "Qwen/Qwen3-8B":
        raise RuntimeError("expected SiliconFlow Qwen/Qwen3-8B configuration")
    context = [
        ContextItem(
            chunk_id="connectivity-chunk",
            paper_id="connectivity-paper",
            block_ids=["connectivity-block"],
            section_path=["Test"],
            page_start=1,
            page_end=1,
            evidence="The test method uses attention.",
            score=1.0,
        )
    ]
    answer = QAService(
        llm=build_llm_provider(settings), prompt_version="qa-production-v1"
    ).answer_from_context("What method is used?", context)
    print(
        json.dumps(
            {
                "status": "PASS",
                "provider": answer.provider,
                "model": answer.model,
                "answerable": answer.answerable,
                "claim_count": len(answer.claims),
                "citation_count": sum(len(claim.citations) for claim in answer.claims),
                "input_tokens": answer.model_usage.input_tokens,
                "output_tokens": answer.model_usage.output_tokens,
                "total_latency_ms": answer.latency.total_latency_ms,
                "first_token_latency_ms": answer.latency.llm_first_token_latency_ms,
                "api_request_count": answer.api_request_count,
                "retry_count": answer.retry_count,
            }
        )
    )


if __name__ == "__main__":
    main()
