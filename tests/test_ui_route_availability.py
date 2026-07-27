from __future__ import annotations

from fastapi.testclient import TestClient

from paper_research.main import app


def test_ui_routes_are_available() -> None:
    client = TestClient(app)

    for route in [
        "/api/v1/ui",
        "/api/v1/ui/library",
        "/api/v1/ui/search",
        "/api/v1/ui/research",
        "/api/v1/ui/evaluation",
        "/api/v1/ui/gold-review",
    ]:
        response = client.get(route)
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]


def test_dashboard_links_match_registered_ui_routes() -> None:
    client = TestClient(app)
    response = client.get("/api/v1/ui")

    assert response.status_code == 200
    for href in [
        "/api/v1/ui/library",
        "/api/v1/ui/search",
        "/api/v1/ui/research",
        "/api/v1/ui/evaluation",
    ]:
        assert href in response.text
