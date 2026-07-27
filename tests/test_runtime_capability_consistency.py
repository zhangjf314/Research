from __future__ import annotations

from fastapi.testclient import TestClient

from paper_research.config import get_settings
from paper_research.infrastructure.redis_service import get_redis_service
from paper_research.main import app


def _reset_settings() -> None:
    get_settings.cache_clear()
    get_redis_service.cache_clear()


def test_deep_research_capability_reports_real_provider_without_template_fallback(
    monkeypatch,
) -> None:
    monkeypatch.setenv("APP_PROFILE", "production")
    monkeypatch.setenv("DEEP_RESEARCH_ENABLED", "true")
    monkeypatch.setenv("LIVE_MODEL_CALLS_ENABLED", "true")
    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("LLM_PROVIDER_NAME", "deepseek")
    monkeypatch.setenv("LLM_MODEL", "deepseek-v4-flash")
    monkeypatch.setenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key-not-real")
    monkeypatch.setenv("LLM_RESPONSE_FORMAT", "json_object")
    monkeypatch.setenv("LLM_THINKING_ENABLED", "false")
    monkeypatch.setenv("LLM_STREAM", "false")
    _reset_settings()

    response = TestClient(app).get("/api/v1/capabilities")
    payload = response.json()
    capability = payload["capabilities"]["deep_research"]

    assert response.status_code == 200
    assert capability["status"] == "available"
    assert capability["provider"] == "deepseek"
    assert capability["model"] == "deepseek-v4-flash"
    assert capability["thinking"] == "disabled"
    assert capability["response_format"] == "json_object"
    assert capability["template_fallback"] is False


def test_deep_research_capability_reports_configuration_disabled(monkeypatch) -> None:
    monkeypatch.setenv("APP_PROFILE", "production")
    monkeypatch.setenv("DEEP_RESEARCH_ENABLED", "false")
    monkeypatch.setenv("LIVE_MODEL_CALLS_ENABLED", "true")
    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("LLM_PROVIDER_NAME", "deepseek")
    monkeypatch.setenv("LLM_MODEL", "deepseek-v4-flash")
    monkeypatch.setenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key-not-real")
    _reset_settings()

    response = TestClient(app).get("/api/v1/capabilities")
    capability = response.json()["capabilities"]["deep_research"]

    assert capability["status"] == "degraded"
    assert "DEEP_RESEARCH_ENABLED=false" in capability["detail"]
