from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EvaluationReport:
    report_id: str
    title: str
    category: str
    markdown_path: Path
    summary_json_path: Path | None = None
    description: str = ""


REPORTS: tuple[EvaluationReport, ...] = (
    EvaluationReport(
        "portfolio-release-audit-v1",
        "Portfolio Release Audit",
        "Release",
        Path("docs/portfolio-release-audit-v1.md"),
        Path("data/evaluation/portfolio-evidence-manifest-v1.json"),
        "Portfolio v1 release gate evidence and boundaries.",
    ),
    EvaluationReport(
        "deepseek-full-qa-final-summary-v1",
        "DeepSeek Full QA Summary",
        "QA",
        Path("docs/deepseek-full-qa-final-summary-v1.md"),
        Path("data/evaluation/deepseek-full-qa-final-summary-v1.json"),
        "Full QA engineering and citation validation summary.",
    ),
    EvaluationReport(
        "deep-research-report-hotfix-smoke-v1",
        "Deep Research Hotfix Smoke",
        "Deep Research",
        Path("docs/deep-research-report-hotfix-smoke-v1.md"),
        Path("data/evaluation/deep-research-report-hotfix-smoke-v1.json"),
        "Structured synthesis, citation allowlist, replay, and report quality smoke.",
    ),
    EvaluationReport(
        "docker-ocr-production-audit-v2",
        "Docker OCR Audit",
        "OCR",
        Path("docs/docker-ocr-production-audit-v2.md"),
        Path("data/evaluation/docker-ocr-production-v2.json"),
        "Docker OCR text, mixed, and scanned PDF roundtrip audit.",
    ),
    EvaluationReport(
        "langgraph-production-recovery-audit-v2",
        "LangGraph Recovery Audit",
        "Recovery",
        Path("docs/langgraph-production-recovery-audit-v2.md"),
        Path("data/evaluation/langgraph-production-recovery-audit-v2.json"),
        "PostgreSQL checkpoint recovery and resume behavior.",
    ),
    EvaluationReport(
        "postgresql-backup-restore-audit-v2",
        "PostgreSQL Backup / Restore",
        "Recovery",
        Path("docs/postgresql-backup-restore-audit-v2.md"),
        Path("data/evaluation/postgresql-backup-restore-v2.json"),
        "PostgreSQL backup and restore audit.",
    ),
    EvaluationReport(
        "qdrant-backup-restore-audit-v2",
        "Qdrant Snapshot / Restore",
        "Recovery",
        Path("docs/qdrant-backup-restore-audit-v2.md"),
        Path("data/evaluation/qdrant-backup-restore-v2.json"),
        "Qdrant snapshot restore and Top-K comparison audit.",
    ),
    EvaluationReport(
        "redis-production-audit-v2",
        "Redis Production Audit",
        "Recovery",
        Path("docs/redis-production-audit-v2.md"),
        Path("data/evaluation/redis-production-audit-v2.json"),
        "Redis cache, usage, and graceful degradation audit.",
    ),
    EvaluationReport(
        "soak-test-portfolio-v1",
        "Portfolio Stability Test",
        "Stability",
        Path("docs/soak-test-portfolio-v1.md"),
        Path("artifacts/soak-test-portfolio-v1.json"),
        "Portfolio 30-minute mixed-load stability window.",
    ),
    EvaluationReport(
        "security-audit-v1",
        "Security Audit",
        "Security",
        Path("docs/git-history-secret-review-v1.md"),
        Path("data/evaluation/security-audit-v1.json"),
        "Secret and publication safety review.",
    ),
    EvaluationReport(
        "content-claims-audit-v1",
        "Content Claims Audit",
        "Security",
        Path("docs/content-claims-audit-v1.md"),
        Path("data/evaluation/content-claims-audit-v1.json"),
        "Portfolio wording and prohibited claims audit.",
    ),
    EvaluationReport(
        "known-limitations",
        "Known Limitations",
        "Limitations",
        Path("docs/known-limitations.md"),
        None,
        "Current limitations and boundary statements.",
    ),
    EvaluationReport(
        "research-synthesis-schema-replay-v1",
        "Research Synthesis Schema Replay",
        "Deep Research",
        Path("docs/research-synthesis-schema-replay-v1.md"),
        Path("data/evaluation/research-synthesis-schema-replay-v1.json"),
        "Raw-response replay against the structured synthesis contract.",
    ),
    EvaluationReport(
        "version-consistency-audit-v1",
        "Version Consistency Audit",
        "Release",
        Path("docs/version-consistency-audit-v1.md"),
        Path("data/evaluation/version-consistency-audit-v1.json"),
        "Package, runtime, OpenAPI, health, and capabilities version consistency.",
    ),
)


def report_by_id(report_id: str) -> EvaluationReport | None:
    return next((report for report in REPORTS if report.report_id == report_id), None)
