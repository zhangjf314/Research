from dataclasses import dataclass

from paper_research.config import Settings
from paper_research.indexing.embedding import (
    EmbeddingProvider,
    HashEmbeddingProvider,
    JinaEmbeddingProvider,
    OpenAICompatibleEmbeddingProvider,
)
from paper_research.providers.llm import (
    LLMProvider,
    OpenAICompatibleLLMProvider,
    SiliconFlowLLMProvider,
    TemplateLLMProvider,
)
from paper_research.retrieval.reranker import (
    CrossEncoderReranker,
    DisabledReranker,
    JinaReranker,
    LexicalReranker,
    Reranker,
)


class ProviderConfigurationError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProviderBundle:
    embedding: EmbeddingProvider
    reranker: Reranker
    llm: LLMProvider
    metadata: dict[str, str | int | bool]


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    if settings.embedding_configuration_issues:
        raise ProviderConfigurationError(
            "production embedding is not configured: "
            + ", ".join(settings.embedding_configuration_issues)
        )
    if settings.embedding_provider == "hash":
        return HashEmbeddingProvider(settings.embedding_dimensions)
    if settings.embedding_provider == "jina":
        assert settings.embedding_base_url is not None
        assert settings.embedding_api_key is not None
        return JinaEmbeddingProvider(
            base_url=settings.embedding_base_url,
            api_key=settings.embedding_api_key,
            model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
            revision=settings.embedding_revision,
            batch_size=settings.embedding_batch_size,
            timeout_seconds=settings.embedding_timeout_seconds,
            max_retries=settings.embedding_max_retries,
        )
    elif settings.embedding_provider == "openai_compatible":
        if not settings.embedding_base_url:
            raise ProviderConfigurationError("EMBEDDING_BASE_URL is required")
        return OpenAICompatibleEmbeddingProvider(
            base_url=settings.embedding_base_url,
            api_key=settings.embedding_api_key,
            model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
            revision=settings.embedding_revision,
            timeout=settings.embedding_timeout_seconds,
        )
    raise ProviderConfigurationError(
        f"unsupported embedding provider: {settings.embedding_provider}"
    )


def build_provider_bundle(settings: Settings) -> ProviderBundle:
    if settings.production_configuration_issues:
        raise ProviderConfigurationError(
            "production providers are not configured: "
            + ", ".join(settings.production_configuration_issues)
        )
    embedding = build_embedding_provider(settings)

    reranker = build_reranker(settings)

    llm = build_llm_provider(settings)
    return ProviderBundle(embedding, reranker, llm, settings.provider_metadata)


def build_llm_provider(settings: Settings) -> LLMProvider:
    if settings.llm_configuration_issues:
        raise ProviderConfigurationError(
            "production LLM is not configured: "
            + ", ".join(settings.llm_configuration_issues)
        )
    if settings.llm_provider == "template":
        return TemplateLLMProvider()
    if not settings.llm_base_url or not settings.llm_api_key:
        raise ProviderConfigurationError("LLM_BASE_URL and LLM_API_KEY are required")
    input_cost_per_million = settings.llm_input_cost_per_million
    output_cost_per_million = settings.llm_output_cost_per_million
    if input_cost_per_million is None and settings.llm_input_price_per_million_tokens is not None:
        input_cost_per_million = float(settings.llm_input_price_per_million_tokens)
    if (
        output_cost_per_million is None
        and settings.llm_output_price_per_million_tokens is not None
    ):
        output_cost_per_million = float(settings.llm_output_price_per_million_tokens)
    if settings.llm_provider == "siliconflow":
        return SiliconFlowLLMProvider(
            settings.llm_base_url,
            settings.llm_api_key,
            settings.llm_model,
            temperature=settings.llm_temperature,
            timeout_seconds=settings.llm_timeout_seconds,
            max_output_tokens=settings.llm_max_output_tokens,
            max_retries=settings.llm_max_retries,
            input_cost_per_million=input_cost_per_million,
            output_cost_per_million=output_cost_per_million,
            response_audit_enabled=settings.qa_response_audit_enabled,
            response_audit_dir=settings.qa_response_audit_dir,
            response_audit_max_prefix_chars=settings.qa_response_audit_max_prefix_chars,
            response_audit_max_suffix_chars=settings.qa_response_audit_max_suffix_chars,
            response_audit_max_error_window_chars=(
                settings.qa_response_audit_max_error_window_chars
            ),
            response_audit_store_full_payload=settings.qa_response_audit_store_full_payload,
        )
    if settings.llm_provider == "openai_compatible":
        return OpenAICompatibleLLMProvider(
            settings.llm_base_url,
            settings.llm_api_key,
            settings.llm_model,
            temperature=settings.llm_temperature,
            timeout=settings.llm_timeout_seconds,
            max_output_tokens=settings.llm_max_output_tokens,
            max_retries=settings.llm_max_retries,
            input_cost_per_million=input_cost_per_million,
            output_cost_per_million=output_cost_per_million,
            provider_name=settings.llm_provider_name,
            response_format=settings.llm_response_format,
            thinking_enabled=settings.llm_thinking_enabled,
            stream=settings.llm_stream,
            response_audit_enabled=settings.qa_response_audit_enabled,
            response_audit_dir=settings.qa_response_audit_dir,
            response_audit_max_prefix_chars=settings.qa_response_audit_max_prefix_chars,
            response_audit_max_suffix_chars=settings.qa_response_audit_max_suffix_chars,
            response_audit_max_error_window_chars=(
                settings.qa_response_audit_max_error_window_chars
            ),
            response_audit_store_full_payload=settings.qa_response_audit_store_full_payload,
        )
    raise ProviderConfigurationError(f"unsupported LLM provider: {settings.llm_provider}")


def build_reranker(settings: Settings) -> Reranker:
    if not settings.rerank_enabled:
        return DisabledReranker()
    if settings.rerank_configuration_issues:
        raise ProviderConfigurationError(
            "production reranker is not configured: "
            + ", ".join(settings.rerank_configuration_issues)
        )
    elif settings.rerank_provider == "lexical":
        return LexicalReranker()
    elif settings.rerank_provider == "jina":
        assert settings.rerank_base_url is not None
        assert settings.rerank_api_key is not None
        return JinaReranker(
            base_url=settings.rerank_base_url,
            api_key=settings.rerank_api_key,
            model=settings.rerank_model,
            timeout_seconds=settings.rerank_timeout_seconds,
            max_retries=settings.rerank_max_retries,
            allow_fallback=settings.rerank_allow_fallback,
            fallback=LexicalReranker() if settings.rerank_allow_fallback else None,
        )
    elif settings.rerank_provider == "cross_encoder":
        assert settings.rerank_base_url is not None
        return CrossEncoderReranker(
            settings.rerank_base_url,
            settings.rerank_api_key,
            settings.rerank_model,
            settings.rerank_timeout_seconds,
        )
    raise ProviderConfigurationError(f"unsupported rerank provider: {settings.rerank_provider}")
