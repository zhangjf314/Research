from fastapi.testclient import TestClient

from paper_research.main import create_app


def test_gold_review_ui_has_error_empty_and_structured_states() -> None:
    response = TestClient(create_app()).get("/api/v1/ui/gold-review")
    assert response.status_code == 200
    text = response.text
    assert "Loading" in text
    assert "Failed:" in text
    assert "No review items match the current filter." in text
    assert "Question ID" in text
    assert "Evidence blocks" in text
    assert "textContent" in text

