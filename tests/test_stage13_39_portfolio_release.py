from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read_json(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def _pyproject_version() -> str:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return str(data["project"]["version"])


def test_portfolio_manifest_preserves_real_model_evidence_and_release_blockers() -> None:
    manifest = _read_json("data/evaluation/portfolio-evidence-manifest-v1.json")

    assert manifest["package_version"] == _pyproject_version()
    assert manifest["release_decision"] == "BLOCKED_BY_OPERATIONS_GATES"
    assert manifest["strong_generalization_claim_allowed"] is False
    assert manifest["semantic_claim_support_audit"] == "NOT_FORMALLY_VALIDATED"
    assert manifest["llm"] == {
        "provider": "deepseek",
        "model": "deepseek-v4-flash",
        "template_fallback": False,
    }
    assert manifest["full_qa"]["attempted"] == 50
    assert manifest["full_qa"]["completed"] == 50
    assert manifest["full_qa"]["failed"] == 0
    assert manifest["deep_research"]["run_id"] == "live-q003-ed900ef2e202"
    assert manifest["deep_research"]["citation_validation"] == "passed"

    blockers = manifest["operations_gates"]
    assert blockers["postgresql_checkpoint_recovery_v2"] == "NOT_EXECUTED"
    assert blockers["postgresql_backup_restore_v2"] == "NOT_EXECUTED"
    assert blockers["qdrant_snapshot_restore_v2"] == "NOT_EXECUTED"
    assert blockers["portfolio_30_minute_stability_test"] == "BLOCKED"


def test_security_policy_keeps_sensitive_artifacts_local_only() -> None:
    security = _read_json("data/evaluation/security-audit-v1.json")
    assert security["tracked_file_scan"]["actual_secret_findings"] == 0
    policy = security["publication_policy"]
    assert policy["commit_env_files"] is False
    assert policy["commit_review_zip_files"] is False
    assert policy["commit_raw_provider_responses"] is False
    assert policy["commit_long_trace_files"] is False

    artifact_audit = (ROOT / "docs/artifact-publication-audit-v1.md").read_text(
        encoding="utf-8"
    )
    assert "artifacts/stage13-9-human-citation-review-results.zip" in artifact_audit
    assert "artifacts/stage13-10-human-claim-gold-review-results.zip" in artifact_audit
    assert "Keep local-only" in artifact_audit


def test_dockerfile_oci_label_matches_pyproject_version() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    match = re.search(r"ARG APP_VERSION=([^\s]+)", dockerfile)
    assert match
    assert match.group(1) == _pyproject_version()
    assert "org.opencontainers.image.version=$APP_VERSION" in dockerfile


def test_content_claims_forbid_strong_generalization_and_v1_release() -> None:
    release_audit = (ROOT / "docs/portfolio-release-audit-v1.md").read_text(
        encoding="utf-8"
    )
    claims_audit = (ROOT / "docs/content-claims-audit-v1.md").read_text(encoding="utf-8")

    assert (
        "B. Core QA/Deep Research passed, but safety/restore/stability blockers remain"
        in release_audit
    )
    assert "STRONG_GENERALIZATION_CLAIM_ALLOWED=false" in claims_audit
    assert "production-grade generalization" in claims_audit
    assert "v1.0.0-portfolio" in claims_audit


def test_portfolio_stability_gate_is_thirty_minutes_not_extended_soak() -> None:
    soak = _read_json("artifacts/soak-test-portfolio-v1.json")

    assert soak["gate_name"] == "Portfolio 30-minute stability test"
    assert soak["required_duration_seconds"] == 1800
    assert soak["configuration"]["SOAK_DURATION_SECONDS"] == 1800
    assert soak["configuration"]["SOAK_MAX_LLM_REQUESTS"] == 8
    assert soak["configuration"]["SOAK_MAX_TOTAL_TOKENS"] == 80000
    assert soak["configuration"]["SOAK_MAX_COST_USD"] == 0.05
    assert soak["configuration"]["SOAK_LLM_SAMPLE_INTERVAL_SECONDS"] == 300
    assert soak["pass_gate"]["qa_success_rate_gte"] == 0.95
    assert soak["pass_gate"]["deep_research_success_count_gte"] == 1
    assert soak["pass_gate"]["ocr_roundtrip"] == "passed"

    policy_files = [
        "README.md",
        "docs/soak-test-portfolio-v1.md",
        "docs/release-checklist-v1.0.0-portfolio.md",
        "docs/portfolio-release-audit-v1.md",
        "docs/known-limitations.md",
        "docs/content-claims-audit-v1.md",
    ]
    combined = "\n".join((ROOT / path).read_text(encoding="utf-8") for path in policy_files)
    assert "Portfolio 30-minute stability test" in combined
    assert "".join(["72", "00"]) not in combined
    assert "-".join(["two", "hour"]) not in combined.lower()
    assert "-".join(["2", "hour"]) not in combined.lower()
    assert " ".join(["long-term", "soak"]) not in combined.lower()
    assert " ".join(["production-grade", "endurance", "test"]) not in combined.lower()
