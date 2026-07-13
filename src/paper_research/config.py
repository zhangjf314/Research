from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "PaperResearch Agent"
    app_env: str = "development"
    app_profile: str = "baseline"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    api_prefix: str = "/api/v1"
    database_url: str = "postgresql+psycopg://paper:paper@localhost:5432/paper_research"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = Field(default=None)
    data_dir: Path = Path("data")
    max_upload_bytes: int = 50 * 1024 * 1024
    parser_backend: str = "auto"
    grobid_url: str | None = None
    ocr_language: str = "eng"
    page_asset_dpi: int = 144
    qdrant_collection: str | None = None
    baseline_collection: str = "papers_hash_v1"
    production_collection: str = "papers_production_v1"
    embedding_provider: str = "hash"
    embedding_model: str = "hash-v1"
    embedding_revision: str = "v1"
    embedding_base_url: str | None = "https://api.jina.ai/v1"
    embedding_api_key: str | None = None
    embedding_dimensions: int = 384
    embedding_batch_size: int = Field(default=32, ge=1, le=2048)
    embedding_timeout_seconds: float = Field(default=60.0, gt=0, le=600)
    embedding_max_retries: int = Field(default=2, ge=0, le=10)
    rerank_provider: str = "lexical"
    rerank_model: str = "lexical-v1"
    rerank_enabled: bool = False
    rerank_base_url: str | None = None
    rerank_api_key: str | None = None
    llm_provider: str = "template"
    llm_model: str = "template-v1"
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_temperature: float = 0.0
    prompt_version: str = "claim-qa-v1"
    index_version: str = "hash-v1"
    dataset_version: str = "gold-set-v1-pending-review"
    chunk_max_tokens: int = 400
    chunk_overlap_tokens: int = 60
    retrieval_score_threshold: float = 0.12
    retrieval_trace_path: Path = Path("data/reports/retrieval_traces.jsonl")
    retrieval_recall_k: int = 20
    semantic_scholar_api_key: str | None = None
    search_cache_dir: Path = Path("data/search_cache")
    search_cache_ttl_seconds: int = 3600
    external_request_retries: int = 3
    redis_url: str | None = "redis://localhost:6379/0"
    redis_cache_ttl_seconds: int = 3600
    redis_max_cache_keys: int = 10000
    api_rate_limit_per_minute: int = 120
    checkpoint_provider: str = "memory"
    checkpoint_database_url: str | None = None

    @model_validator(mode="after")
    def validate_profile(self) -> "Settings":
        if self.app_profile not in {"baseline", "production"}:
            raise ValueError("APP_PROFILE must be baseline or production")
        return self

    @property
    def production_configuration_issues(self) -> list[str]:
        if self.app_profile != "production":
            return []
        missing = list(self.embedding_configuration_issues)
        if self.rerank_enabled and self.rerank_provider == "lexical":
            missing.append("RERANK_PROVIDER")
        if self.rerank_enabled and not self.rerank_base_url:
            missing.append("RERANK_BASE_URL")
        if self.llm_provider == "template":
            missing.append("LLM_PROVIDER")
        if not self.llm_base_url:
            missing.append("LLM_BASE_URL")
        if not self.llm_api_key:
            missing.append("LLM_API_KEY")
        return sorted(set(missing))

    @property
    def embedding_configuration_issues(self) -> list[str]:
        if self.app_profile != "production":
            return []
        provider = self.embedding_provider.strip().lower()
        missing: list[str] = []
        if provider == "hash":
            return ["EMBEDDING_PROVIDER"]
        if provider not in {"jina", "openai_compatible"}:
            return ["EMBEDDING_PROVIDER"]
        if (
            "embedding_model" not in self.model_fields_set
            or not self.embedding_model.strip()
            or self.embedding_model == "hash-v1"
        ):
            missing.append("EMBEDDING_MODEL")
        if (
            "embedding_dimensions" not in self.model_fields_set
            or self.embedding_dimensions <= 0
        ):
            missing.append("EMBEDDING_DIMENSIONS")
        if not self.embedding_api_key:
            missing.append("EMBEDDING_API_KEY")
        if provider == "openai_compatible" and (
            "embedding_base_url" not in self.model_fields_set or not self.embedding_base_url
        ):
            missing.append("EMBEDDING_BASE_URL")
        return sorted(set(missing))

    @property
    def active_collection(self) -> str:
        if self.qdrant_collection:
            return self.qdrant_collection
        return (
            self.production_collection
            if self.app_profile == "production"
            else self.baseline_collection
        )

    @property
    def provider_metadata(self) -> dict[str, str | int | bool]:
        return {
            "app_profile": self.app_profile,
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "embedding_revision": self.embedding_revision,
            "embedding_dimension": self.embedding_dimensions,
            "embedding_batch_size": self.embedding_batch_size,
            "embedding_timeout_seconds": self.embedding_timeout_seconds,
            "embedding_max_retries": self.embedding_max_retries,
            "rerank_provider": self.rerank_provider,
            "rerank_model": self.rerank_model,
            "rerank_enabled": self.rerank_enabled,
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "prompt_version": self.prompt_version,
            "index_version": self.index_version,
            "dataset_version": self.dataset_version,
            "collection": self.active_collection,
        }

    @property
    def raw_papers_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def parsed_papers_dir(self) -> Path:
        return self.data_dir / "parsed"


@lru_cache
def get_settings() -> Settings:
    return Settings()
