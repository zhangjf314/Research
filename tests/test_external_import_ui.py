from fastapi.testclient import TestClient

from paper_research.main import create_app


def test_search_ui_defaults_empty_and_exposes_provider_import_only() -> None:
    response = TestClient(create_app()).get("/api/v1/ui/search")
    assert response.status_code == 200
    text = response.text
    assert "placeholder='Search arXiv and Semantic Scholar'" in text
    assert "Import PDF" in text
    assert "Import and Index" in text
    assert "No downloadable PDF" in text
    assert "pdf_url" in text
    assert "type='url'" not in text

