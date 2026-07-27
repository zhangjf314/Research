from __future__ import annotations

from fastapi.testclient import TestClient
from starlette.requests import Request

import paper_research.api.rate_limit as rate_limit
from paper_research.api.rate_limit import limit_for_bucket, rate_limit_bucket_for_request
from paper_research.config import Settings
from paper_research.main import app


def _request(method: str, path: str) -> Request:
    return Request(
        {
            "type": "http",
            "method": method,
            "path": path,
            "headers": [],
            "client": ("127.0.0.1", 12345),
            "server": ("testserver", 80),
            "scheme": "http",
        }
    )


def test_ui_and_runtime_get_pages_are_exempt_from_business_quota() -> None:
    for path in [
        "/api/v1/ui",
        "/api/v1/ui/library",
        "/api/v1/ui/search",
        "/api/v1/ui/research",
        "/api/v1/health",
        "/api/v1/capabilities",
        "/docs",
        "/openapi.json",
    ]:
        assert rate_limit_bucket_for_request(_request("GET", path)) is None


def test_business_routes_use_separate_buckets() -> None:
    assert rate_limit_bucket_for_request(_request("GET", "/api/v1/papers")) == "read_api"
    assert rate_limit_bucket_for_request(_request("POST", "/api/v1/search/papers")) == "search"
    assert rate_limit_bucket_for_request(_request("POST", "/api/v1/papers/upload")) == "upload"
    assert (
        rate_limit_bucket_for_request(
            _request("POST", "/api/v1/papers/abc/enrich-metadata")
        )
        == "metadata_enrichment"
    )
    assert rate_limit_bucket_for_request(_request("POST", "/api/v1/research/deep")) == (
        "deep_research"
    )


def test_bucket_limits_are_configurable() -> None:
    settings = Settings(
        api_rate_limit_read_api_per_minute=301,
        api_rate_limit_search_per_minute=61,
        api_rate_limit_upload_per_minute=21,
        api_rate_limit_metadata_enrichment_per_minute=22,
        api_rate_limit_deep_research_per_minute=6,
    )

    assert limit_for_bucket(settings, "read_api") == 301
    assert limit_for_bucket(settings, "search") == 61
    assert limit_for_bucket(settings, "upload") == 21
    assert limit_for_bucket(settings, "metadata_enrichment") == 22
    assert limit_for_bucket(settings, "deep_research") == 6


def test_rate_limited_response_includes_retry_after_and_request_id(monkeypatch) -> None:
    class FakeRedisService:
        def allow_request(self, identity: str, *, bucket_name: str, limit_per_minute: int):
            return False, 17

    monkeypatch.setattr(rate_limit, "get_redis_service", lambda: FakeRedisService())

    response = TestClient(app).post(
        "/api/v1/search/papers",
        json={"query": "transformer", "limit": 1},
        headers={"x-request-id": "rate-test-1"},
    )

    assert response.status_code == 429
    assert response.headers["Retry-After"] == "17"
    payload = response.json()
    assert payload["error_code"] == "RATE_LIMITED"
    assert payload["request_id"] == "rate-test-1"
