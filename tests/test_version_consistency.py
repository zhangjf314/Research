from __future__ import annotations

import tomllib
from pathlib import Path

from fastapi.testclient import TestClient

import paper_research
from paper_research.api.routes import capabilities as capabilities_route
from paper_research.api.routes import health as health_route
from paper_research.main import create_app
from paper_research.version import display_version, package_version


class _FakeDb:
    def execute(self, _query: object) -> None:
        return None


class _FakeRedis:
    last_error = None

    def ping(self) -> bool:
        return True

    def stats(self) -> dict[str, object]:
        return {
            "available": True,
            "used": True,
            "cache_hits": 1,
            "cache_misses": 1,
            "cache_hit_rate": 0.5,
            "writes": 1,
            "key_count": 1,
            "ttl_seconds": 3600,
        }


class _FakeResponse:
    is_success = True

    def raise_for_status(self) -> None:
        return None


def _pyproject_version() -> str:
    data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    return str(data["project"]["version"])


def test_runtime_version_matches_pyproject() -> None:
    assert package_version() == _pyproject_version()
    assert paper_research.__version__ == _pyproject_version()
    assert paper_research.__display_version__ == display_version(_pyproject_version())


def test_openapi_root_health_and_capabilities_versions_match(
    monkeypatch,
) -> None:
    monkeypatch.setattr(health_route.httpx, "get", lambda *args, **kwargs: _FakeResponse())
    monkeypatch.setattr(health_route, "get_redis_service", lambda: _FakeRedis())
    monkeypatch.setattr(capabilities_route, "get_redis_service", lambda: _FakeRedis())

    app = create_app()
    app.dependency_overrides[health_route.get_db] = lambda: _FakeDb()
    client = TestClient(app)

    expected = _pyproject_version()
    expected_display = display_version(expected)

    assert client.get("/").json()["version"] == expected
    assert client.get("/openapi.json").json()["info"]["version"] == expected

    health = client.get("/api/v1/health").json()
    assert health["version"] == expected
    assert health["display_version"] == expected_display

    capabilities = client.get("/api/v1/capabilities").json()
    assert capabilities["version"] == expected
    assert capabilities["display_version"] == expected_display
