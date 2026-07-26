from fastapi.testclient import TestClient

from paper_research.main import create_app


def test_library_ui_has_upload_metadata_and_no_arbitrary_url_input() -> None:
    response = TestClient(create_app()).get("/api/v1/ui/library")
    assert response.status_code == 200
    text = response.text
    assert 'id="paper-file"' in text or "id='paper-file'" in text
    assert "accept='application/pdf'" in text
    assert "auto-index" in text
    assert "Edit Metadata" in text
    assert "Missing year" in text
    assert "http://" not in text.replace("http://localhost", "")

