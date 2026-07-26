from fastapi.testclient import TestClient

from paper_research.evaluation.report_catalog import REPORTS, report_by_id
from paper_research.main import create_app


def test_report_catalog_is_explicit_and_core_reports_open() -> None:
    assert report_by_id("../known-limitations") is None
    client = TestClient(create_app())
    catalog = client.get("/api/v1/ui/evaluation")
    assert catalog.status_code == 200
    assert "Evaluation Report Catalog" in catalog.text
    for report_id in (
        "portfolio-release-audit-v1",
        "deepseek-full-qa-final-summary-v1",
        "deep-research-report-hotfix-smoke-v1",
        "known-limitations",
    ):
        assert report_by_id(report_id) is not None
        detail = client.get(f"/api/v1/ui/evaluation/{report_id}")
        assert detail.status_code == 200
        assert "<script" not in detail.text.lower()


def test_registered_report_paths_are_public_safe() -> None:
    for report in REPORTS:
        assert not report.markdown_path.is_absolute()
        assert ".." not in report.markdown_path.parts

