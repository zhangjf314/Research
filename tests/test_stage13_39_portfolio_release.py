from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

from paper_research.version import display_version

ROOT = Path(__file__).resolve().parents[1]


def _read_json(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def _pyproject_version() -> str:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return str(data["project"]["version"])


def test_portfolio_manifest_preserves_real_model_evidence_and_release_gates() -> None:
    manifest = _read_json("data/evaluation/portfolio-evidence-manifest-v1.json")

    assert manifest["package_version"] == _pyproject_version()
    assert (
        manifest["release_decision"]
        == "LOCAL_RELEASE_PREPARED_AWAITING_USER_MERGE_TAG_PUSH_AUTHORIZATION"
    )
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

    gates = manifest["operations_gates"]
    assert gates["postgresql_checkpoint_recovery_v2"] == "PASSED"
    assert gates["postgresql_backup_restore_v2"] == "PASSED"
    assert gates["qdrant_snapshot_restore_v2"] == "PASSED"
    assert gates["docker_ocr_production_v2"] == "PASSED"
    assert gates["portfolio_30_minute_stability_test"] == "PASSED"
    assert manifest["stage13_40_gates"]["git_history_secret_review"]["confirmed_real_secret"] == 0


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


def test_git_history_secret_review_has_no_real_or_unresolved_secret() -> None:
    review = _read_json("data/evaluation/git-history-secret-review-v1.json")

    assert review["gate"] == "PASSED"
    assert review["public_release_allowed"] is True
    assert review["confirmed_real_secret"] == 0
    assert review["unresolved"] == 0
    assert review["classification_counts"].get("UNRESOLVED", 0) == 0
    assert review["classification_counts"].get("CONFIRMED_REAL_SECRET", 0) == 0
    allowed = {
        "PLACEHOLDER",
        "EMPTY_VALUE",
        "DOCUMENTATION_EXAMPLE",
        "HASH_OR_FINGERPRINT",
        "FALSE_POSITIVE",
    }
    assert {record["classification"] for record in review["records"]} <= allowed


def test_dockerfile_oci_label_matches_pyproject_version() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    match = re.search(r"ARG APP_VERSION=([^\s]+)", dockerfile)
    assert match
    assert match.group(1) == display_version(_pyproject_version())
    assert "org.opencontainers.image.version=$APP_VERSION" in dockerfile


def test_content_claims_forbid_strong_generalization_and_remote_release() -> None:
    release_audit = (ROOT / "docs/portfolio-release-audit-v1.md").read_text(
        encoding="utf-8"
    )
    claims_audit = (ROOT / "docs/content-claims-audit-v1.md").read_text(encoding="utf-8")

    assert "All v1.0.0-portfolio hard gates passed" in release_audit
    assert "awaiting explicit user authorization for merge/tag/push" in release_audit
    assert "STRONG_GENERALIZATION_CLAIM_ALLOWED=false" in claims_audit
    assert "production-grade generalization" in claims_audit
    assert "production-ready commercial v1.0" in claims_audit


def test_portfolio_stability_gate_is_thirty_minutes_not_extended_soak() -> None:
    soak = _read_json("artifacts/soak-test-portfolio-v1.json")

    assert soak["gate_name"] == "Portfolio 30-minute stability test"
    assert soak["status"] == "PASSED"
    assert soak["actual_duration_seconds"] >= 1800
    assert soak["configuration"]["SOAK_DURATION_SECONDS"] == 1800
    assert soak["configuration"]["SOAK_MAX_LLM_REQUESTS"] == 8
    assert soak["configuration"]["SOAK_MAX_TOTAL_TOKENS"] == 80000
    assert soak["configuration"]["SOAK_MAX_COST_USD"] == 0.05
    assert soak["configuration"]["SOAK_LLM_SAMPLE_INTERVAL_SECONDS"] == 300
    assert soak["qa_success_rate"] >= 0.95
    assert soak["deep_research_success_count"] >= 1
    assert soak["ocr_roundtrip"] == "passed"
    assert soak["budget_violations"] == []

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
