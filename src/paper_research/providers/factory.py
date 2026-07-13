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
    TemplateLLMProvider,
)
from paper_research.retrieval.reranker import (
    CrossEncoderReranker,
    DisabledReranker,
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

    if settings.llm_provider == "template":
        llm: LLMProvider = TemplateLLMProvider()
    elif settings.llm_provider == "openai_compatible":
        if not settings.llm_base_url or not settings.llm_api_key:
            raise ProviderConfigurationError("LLM_BASE_URL and LLM_API_KEY are required")
        llm = OpenAICompatibleLLMProvider(
            settings.llm_base_url,
            settings.llm_api_key,
            settings.llm_model,
            settings.llm_temperature,
        )
    else:
        raise ProviderConfigurationError(f"unsupported LLM provider: {settings.llm_provider}")
    return ProviderBundle(embedding, reranker, llm, settings.provider_metadata)


def build_reranker(settings: Settings) -> Reranker:

    if not settings.rerank_enabled:
        return DisabledReranker()
    elif settings.rerank_provider == "lexical":
        return LexicalReranker()
    elif settings.rerank_provider == "cross_encoder":
        if not settings.rerank_base_url:
            raise ProviderConfigurationError("RERANK_BASE_URL is required")
        return CrossEncoderReranker(
            settings.rerank_base_url,
            settings.rerank_api_key,
            settings.rerank_model,
        )
    raise ProviderConfigurationError(f"unsupported rerank provider: {settings.rerank_provider}")
