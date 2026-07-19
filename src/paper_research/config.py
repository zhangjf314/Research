from decimal import Decimal
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator, model_validator
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
    rerank_base_url: str | None = "https://api.jina.ai/v1"
    rerank_api_key: str | None = None
    rerank_input_k: int = Field(default=30, ge=1, le=100)
    rerank_output_k: int = Field(default=30, ge=1, le=100)
    rerank_timeout_seconds: float = Field(default=60.0, gt=0, le=600)
    rerank_max_retries: int = Field(default=2, ge=0, le=10)
    rerank_allow_fallback: bool = False
    llm_provider: str = "template"
    llm_provider_name: str | None = None
    llm_model: str = "template-v1"
    llm_base_url: str | None = "https://api.siliconflow.cn/v1"
    llm_api_key: str | None = None
    llm_temperature: float = 0.0
    llm_timeout_seconds: float = Field(default=120.0, gt=0, le=600)
    llm_max_output_tokens: int = Field(default=2048, ge=128, le=65536)
    llm_max_retries: int = Field(default=2, ge=0, le=2)
    llm_response_format: str = "json_object"
    llm_thinking_enabled: bool = False
    llm_stream: bool = False
    llm_input_cost_per_million: float | None = Field(default=None, ge=0)
    llm_output_cost_per_million: float | None = Field(default=None, ge=0)
    llm_billing_mode: str | None = None
    llm_input_price_per_million_tokens: Decimal | None = Field(default=None, ge=0)
    llm_output_price_per_million_tokens: Decimal | None = Field(default=None, ge=0)
    qa_response_audit_enabled: bool = False
    qa_response_audit_max_prefix_chars: int = Field(default=500, ge=0, le=5000)
    qa_response_audit_max_suffix_chars: int = Field(default=500, ge=0, le=5000)
    qa_response_audit_max_error_window_chars: int = Field(default=500, ge=0, le=5000)
    qa_response_audit_store_full_payload: bool = False
    qa_response_audit_dir: Path = Path("artifacts/private/qa-response-audits")
    qa_context_token_budget: int = Field(default=12000, ge=512, le=100000)
    prompt_version: str = "qa-production-v1"
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
    deep_research_mode: str = "disabled"
    deep_research_max_queries: int = Field(default=3, ge=1, le=3)
    deep_research_max_iterations_per_query: int = Field(default=2, ge=1, le=2)
    deep_research_max_llm_requests_per_query: int = Field(default=4, ge=1, le=4)
    deep_research_max_llm_requests_total: int = Field(default=12, ge=1, le=12)
    deep_research_max_tokens_per_query: int = Field(default=40000, ge=1, le=40000)
    deep_research_max_tokens_total: int = Field(default=120000, ge=1, le=120000)
    deep_research_max_cost_usd: Decimal | None = Field(default=None, ge=0)
    deep_research_max_elapsed_seconds_per_query: int = Field(default=300, ge=1, le=300)
    deep_research_max_elapsed_seconds_total: int = Field(default=900, ge=1, le=900)
    deep_research_require_usage: bool = True
    deep_research_checkpoint_path: Path = Path("data/checkpoints/deep-research-smoke-v1.sqlite3")
    deepseek_canary_max_input_tokens: int | None = Field(default=None, ge=1)
    deepseek_canary_max_output_tokens: int | None = Field(default=None, ge=1)
    deepseek_canary_max_total_tokens: int | None = Field(default=None, ge=1)
    deepseek_canary_max_cost_usd: Decimal | None = Field(default=None, ge=0)
    deepseek_canary_max_total_seconds: int | None = Field(default=None, ge=1)

    @field_validator(
        "llm_input_cost_per_million",
        "llm_output_cost_per_million",
        "llm_input_price_per_million_tokens",
        "llm_output_price_per_million_tokens",
        "deep_research_max_cost_usd",
        "deepseek_canary_max_cost_usd",
        "deepseek_canary_max_input_tokens",
        "deepseek_canary_max_output_tokens",
        "deepseek_canary_max_total_tokens",
        "deepseek_canary_max_total_seconds",
        mode="before",
    )
    @classmethod
    def empty_string_optional_number(cls, value: object) -> object:
        if value == "":
            return None
        return value

    @model_validator(mode="after")
    def validate_profile(self) -> "Settings":
        if self.app_profile not in {"baseline", "production"}:
            raise ValueError("APP_PROFILE must be baseline or production")
        if self.rerank_output_k > self.rerank_input_k:
            raise ValueError("RERANK_OUTPUT_K must not exceed RERANK_INPUT_K")
        return self

    @property
    def production_configuration_issues(self) -> list[str]:
        if self.app_profile != "production":
            return []
        missing = list(self.embedding_configuration_issues)
        missing.extend(self.rerank_configuration_issues)
        missing.extend(self.llm_configuration_issues)
        return sorted(set(missing))

    @property
    def llm_configuration_issues(self) -> list[str]:
        if self.app_profile != "production":
            return []
        provider = self.llm_provider.strip().lower()
        if provider == "template" or provider not in {"siliconflow", "openai_compatible"}:
            return ["LLM_PROVIDER"]
        missing: list[str] = []
        provider_name = (self.llm_provider_name or self.llm_provider).strip().lower()
        if provider_name == "deepseek":
            if self.llm_base_url and "api.deepseek.com" not in self.llm_base_url:
                missing.append("LLM_BASE_URL")
            if self.llm_model in {"deepseek-chat", "deepseek-reasoner"}:
                missing.append("LLM_MODEL")
            if self.llm_response_format != "json_object":
                missing.append("LLM_RESPONSE_FORMAT")
            if self.llm_thinking_enabled:
                missing.append("LLM_THINKING_ENABLED")
            if self.llm_stream:
                missing.append("LLM_STREAM")
        if (
            "llm_model" not in self.model_fields_set
            or not self.llm_model.strip()
            or self.llm_model == "template-v1"
        ):
            missing.append("LLM_MODEL")
        if not self.llm_base_url:
            missing.append("LLM_BASE_URL")
        if not self.llm_api_key:
            missing.append("LLM_API_KEY")
        if self.prompt_version == "qa-production-v1" and self.llm_temperature != 0:
            missing.append("LLM_TEMPERATURE")
        return sorted(set(missing))

    @property
    def rerank_configuration_issues(self) -> list[str]:
        if self.app_profile != "production" or not self.rerank_enabled:
            return []
        provider = self.rerank_provider.strip().lower()
        if provider == "lexical":
            return ["RERANK_PROVIDER"]
        if provider not in {"jina", "cross_encoder"}:
            return ["RERANK_PROVIDER"]
        missing: list[str] = []
        if (
            "rerank_model" not in self.model_fields_set
            or not self.rerank_model.strip()
            or self.rerank_model == "lexical-v1"
        ):
            missing.append("RERANK_MODEL")
        if not self.rerank_api_key:
            missing.append("RERANK_API_KEY")
        if provider == "cross_encoder" and (
            "rerank_base_url" not in self.model_fields_set or not self.rerank_base_url
        ):
            missing.append("RERANK_BASE_URL")
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
            "rerank_input_k": self.rerank_input_k,
            "rerank_output_k": self.rerank_output_k,
            "rerank_timeout_seconds": self.rerank_timeout_seconds,
            "rerank_max_retries": self.rerank_max_retries,
            "rerank_allow_fallback": self.rerank_allow_fallback,
            "llm_provider": self.llm_provider,
            "llm_provider_name": self.llm_provider_name or self.llm_provider,
            "llm_model": self.llm_model,
            "llm_timeout_seconds": self.llm_timeout_seconds,
            "llm_max_output_tokens": self.llm_max_output_tokens,
            "llm_max_retries": self.llm_max_retries,
            "llm_response_format": self.llm_response_format,
            "llm_thinking_enabled": self.llm_thinking_enabled,
            "llm_stream": self.llm_stream,
            "qa_context_token_budget": self.qa_context_token_budget,
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
