from __future__ import annotations

from pathlib import Path

from paper_research.api.routes.research import _response
from paper_research.api.routes.ui import research_page


def test_failed_response_contract_does_not_write_blank_markdown(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    response = _response(
        {
            "task_id": "failed-task",
            "status": "FAILED_PROVIDER_SCHEMA",
            "stop_reason": "research synthesis schema validation failed",
            "draft_report": "",
            "model_usage": {"total_tokens": 123, "estimated_cost_usd": 0.001},
            "request_attempt_count": 2,
            "provider_completed_request_count": 2,
            "usage_record_count": 2,
            "active_reserved_tokens": 0,
        }
    )
    assert response.succeeded is False
    assert response.terminal is True
    assert response.error_code == "FAILED_PROVIDER_SCHEMA"
    assert response.report_available is False
    assert response.report_path == ""
    assert not Path("data/reports/research/failed-task.md").exists()
    assert Path("data/reports/research/failed-task.json").exists()


def test_completed_without_report_is_contract_bug(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    response = _response({"task_id": "bad-completed", "status": "COMPLETED", "draft_report": ""})
    assert response.succeeded is False
    assert response.report_available is False
    assert response.error_code is None


def test_completed_with_report_available(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    response = _response(
        {
            "task_id": "ok",
            "status": "COMPLETED",
            "stop_reason": "research_complete",
            "draft_report": "# Report",
        }
    )
    assert response.succeeded is True
    assert response.report_available is True
    assert Path(response.report_path).exists()


def test_research_ui_uses_status_contract_not_ambiguous_message() -> None:
    html = research_page().body.decode()
    assert "Research response contract error: completed task has no report." in html
    assert "Research completed without a report." not in html
    assert "data.succeeded" in html
    assert "data.report_available" in html
    assert "Status:" in html
    assert "Estimated cost:" in html
