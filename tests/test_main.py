from fastapi.testclient import TestClient

from paper_research.main import create_app


def test_root() -> None:
    response = TestClient(create_app()).get("/")
    assert response.status_code == 200
    assert response.json()["name"] == "PaperResearch Agent"


def test_evaluation_page() -> None:
    response = TestClient(create_app()).get("/api/v1/ui/evaluation")
    assert response.status_code == 200
    assert "基础评测中心" in response.text
    assert "检索冒烟评测" in response.text


def test_dashboard_and_request_id() -> None:
    response = TestClient(create_app()).get(
        "/api/v1/ui", headers={"x-request-id": "test-request-1"}
    )
    assert response.status_code == 200
    assert "PaperResearch Agent" in response.text
    assert response.headers["x-request-id"] == "test-request-1"


def test_validation_errors_use_structured_envelope() -> None:
    response = TestClient(create_app()).post("/api/v1/qa", json={})
    assert response.status_code == 422
    payload = response.json()["error"]
    assert payload["code"] == "VALIDATION_ERROR"
    assert payload["request_id"] == response.headers["x-request-id"]
