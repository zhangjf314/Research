import json
from pathlib import Path

import pytest

from paper_research.config import Settings
from paper_research.evaluation.review_store import GoldReviewStore
from paper_research.providers.factory import (
    ProviderConfigurationError,
    build_provider_bundle,
)
from paper_research.retrieval.reranker import DisabledReranker


def test_baseline_keeps_deterministic_providers_and_rerank_disabled() -> None:
    settings = Settings(_env_file=None)
    providers = build_provider_bundle(settings)
    assert settings.app_profile == "baseline"
    assert settings.active_collection == "papers_hash_v1"
    assert providers.embedding.provider_name == "hash"
    assert isinstance(providers.reranker, DisabledReranker)
    assert providers.llm.provider_name == "template"


def test_incomplete_production_profile_fails_without_silent_fallback() -> None:
    settings = Settings(app_profile="production", _env_file=None)
    assert settings.active_collection == "papers_production_v1"
    with pytest.raises(ProviderConfigurationError, match="not configured"):
        build_provider_bundle(settings)


def test_optional_numeric_env_empty_strings_parse_as_none() -> None:
    settings = Settings(
        app_profile="production",
        llm_provider="siliconflow",
        llm_model="Qwen/Qwen3-8B",
        llm_base_url="https://api.siliconflow.cn/v1",
        llm_api_key="test-key",
        embedding_provider="jina",
        embedding_model="jina-embeddings-v5-text-small",
        embedding_api_key="test-key",
        embedding_dimensions=1024,
        llm_input_cost_per_million="",
        llm_output_cost_per_million="",
        llm_input_price_per_million_tokens="",
        llm_output_price_per_million_tokens="",
        deep_research_max_cost_usd="",
        _env_file=None,
    )
    assert settings.production_configuration_issues == []
    assert settings.llm_input_cost_per_million is None
    assert settings.llm_output_cost_per_million is None
    assert settings.llm_input_price_per_million_tokens is None
    assert settings.llm_output_price_per_million_tokens is None
    assert settings.deep_research_max_cost_usd is None


def test_defer_cannot_approve_and_review_metadata_is_written(tmp_path: Path) -> None:
    dataset = tmp_path / "gold.jsonl"
    item = {
        "question_id": "q001",
        "category": "method",
        "review_status": "pending",
    }
    dataset.write_text(json.dumps(item) + "\n", encoding="utf-8")
    store = GoldReviewStore(dataset, tmp_path)
    reviewed = store.review(
        "q001",
        action="defer",
        reviewer="human-reviewer",
        review_notes="needs source verification",
    )
    assert reviewed["review_status"] == "pending"
    assert reviewed["reviewer"] == "human-reviewer"
    assert reviewed["reviewed_at"]
    assert reviewed["dataset_version"] == "gold-set-v1-human-review"
