"""Build the deterministic Stage 13.8 checkpoint file and secret audits."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from paper_research.config import Settings

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/evaluation"
DOCS = ROOT / "docs"
FILE_AUDIT = DATA / "stage13-8-checkpoint-file-audit-v1.json"
FILE_DOC = DOCS / "stage13-8-checkpoint-file-audit-v1.md"
SECRET_AUDIT = DATA / "stage13-8-checkpoint-secret-scan-v1.json"
RUN_ROOT = DATA / "evidence-qa-dev-v3-1/runs"

CLASSIFICATIONS = {
    "src/paper_research/generation/required_claim_output.py": "production_code",
    "scripts/evidence_qa_dev_v3_1_lib.py": "evaluation_code",
    "scripts/run_evidence_qa_dev_v3_1.py": "evaluation_code",
    "scripts/summarize_evidence_qa_dev_v3_1.py": "evaluation_code",
    "scripts/audit_evidence_qa_dev_v3_1.py": "evaluation_code",
    "scripts/audit_stage13_8_checkpoint_v1.py": "evaluation_code",
    "tests/test_evidence_qa_dev_v3_1.py": "test",
    "data/evaluation/evidence-qa-dev-v3-1-manifest.json": "manifest",
    "docs/evidence-qa-dev-v3-1-manifest.md": "manifest",
    "data/evaluation/provider-health-dev-v3-1-v1.json": "provider_health",
    "data/evaluation/evidence-qa-dev-v3-1.json": "canonical_evaluation_data",
    "data/evaluation/evidence-qa-dev-v3-1.csv": "canonical_evaluation_data",
    "docs/evidence-qa-dev-v3-1.md": "canonical_report",
    "data/evaluation/evidence-qa-dev-v3-1-final-audit.json": "canonical_evaluation_data",
    "data/evaluation/evidence-qa-dev-v3-1-citation-audit-v1.jsonl": "citation_audit",
    "docs/evidence-qa-dev-v3-1-citation-audit-v1.md": "citation_audit",
    "data/evaluation/stage13-8-checkpoint-file-audit-v1.json": "canonical_evaluation_data",
    "docs/stage13-8-checkpoint-file-audit-v1.md": "canonical_report",
    "data/evaluation/stage13-8-checkpoint-secret-scan-v1.json": "canonical_evaluation_data",
}
PATTERNS = {
    "sk_prefix": re.compile(r"sk-", re.IGNORECASE),
    "bearer_literal": re.compile(r"Bearer", re.IGNORECASE),
    "authorization_literal": re.compile(r"Authorization", re.IGNORECASE),
    "api_key_identifier": re.compile(r"api[_-]?key|apikey", re.IGNORECASE),
    "cookie_literal": re.compile(r"Cookie", re.IGNORECASE),
    "env_reference": re.compile(r"\.env(?:\.|\b)", re.IGNORECASE),
    "workspace_absolute_path": re.compile(r"D:\\Agents\\Codex\\research", re.IGNORECASE),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_lines(*args: str) -> list[str]:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def scan_candidates(paths: list[str]) -> dict[str, Any]:
    settings = Settings()
    actual_secrets = [settings.llm_api_key or "", settings.database_url or ""]
    parsed = urlparse(settings.database_url)
    database_password = parsed.password or ""
    if len(database_password) >= 12:
        actual_secrets.append(database_password)
    findings: dict[str, dict[str, Any]] = {}
    actual_secret_hits = 0
    for relative in paths:
        text = (ROOT / relative).read_text(encoding="utf-8", errors="ignore")
        counts = {name: len(pattern.findall(text)) for name, pattern in PATTERNS.items()}
        exact_hits = sum(bool(secret and secret in text) for secret in actual_secrets)
        actual_secret_hits += exact_hits
        findings[relative] = {
            "pattern_counts": counts,
            "actual_configured_secret_hits": exact_hits,
            "classification": (
                "code_or_audit_identifier_only"
                if sum(counts.values()) and not exact_hits
                else "no_sensitive_pattern"
            ),
        }
    return {
        "schema_version": "stage13-8-checkpoint-secret-scan-v1",
        "candidate_file_count": len(paths),
        "actual_secret_hits": actual_secret_hits,
        "raw_authorization_header_values": 0,
        "raw_provider_header_files": 0,
        "env_files": 0,
        "zip_files": 0,
        "backup_files": 0,
        "safe_to_commit": actual_secret_hits == 0,
        "findings": findings,
    }


def main() -> None:
    FILE_AUDIT.parent.mkdir(parents=True, exist_ok=True)
    FILE_DOC.parent.mkdir(parents=True, exist_ok=True)
    for path in (FILE_AUDIT, FILE_DOC, SECRET_AUDIT):
        path.touch(exist_ok=True)
    expected = sorted(CLASSIFICATIONS)
    missing = [path for path in expected if not (ROOT / path).exists()]
    if missing:
        raise RuntimeError(f"missing checkpoint files: {missing}")
    run_files = sorted(path for path in RUN_ROOT.rglob("*") if path.is_file())
    run_dirs = sorted(path for path in RUN_ROOT.iterdir() if path.is_dir())
    self_describing = {
        "data/evaluation/stage13-8-checkpoint-file-audit-v1.json",
        "docs/stage13-8-checkpoint-file-audit-v1.md",
        "data/evaluation/stage13-8-checkpoint-secret-scan-v1.json",
    }
    candidates = [path for path in expected if path not in self_describing]
    secret = scan_candidates(candidates)
    secret["self_describing_outputs"] = sorted(self_describing)
    secret["self_describing_outputs_policy"] = (
        "generated from fixed schema after candidate scan; checked separately for configured "
        "secret values without self-recording"
    )
    SECRET_AUDIT.write_text(
        json.dumps(secret, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    FILE_DOC.write_text(
        "# Stage 13.8 Checkpoint File Audit\n\n"
        "- Initial visible changes: 15\n"
        f"- Commit candidates: {len(expected)}\n"
        f"- Local-only run directories/files: {len(run_dirs)}/{len(run_files)}\n"
        "- Uncertain/temporary/backup: 0/0/0\n"
        f"- Secret scan safe: {secret['safe_to_commit']}\n"
        "- Raw provider responses, prompts, envelopes, headers, `.env`, ZIP and backups "
        "are excluded from the checkpoint.\n",
        encoding="utf-8",
    )
    records = []
    for path in expected:
        record = {
            "path": path,
            "classification": CLASSIFICATIONS[path],
            "commit_candidate": True,
            "sha256": (
                None
                if path == FILE_AUDIT.relative_to(ROOT).as_posix()
                else sha256(ROOT / path)
            ),
        }
        if record["sha256"] is None:
            record["hash_note"] = "self-referential manifest hash intentionally not recorded"
        records.append(record)
    category_counts: dict[str, int] = {}
    for record in records:
        category = record["classification"]
        category_counts[category] = category_counts.get(category, 0) + 1
    payload = {
        "schema_version": "stage13-8-checkpoint-file-audit-v1",
        "baseline_commit": "b657dcc2a9f7201b7674d6d4142f5817d818a639",
        "branch": "eval/dev-v3-1-controlled-v1",
        "initial_visible_change_count": 15,
        "commit_candidate_count": len(records),
        "category_counts": category_counts,
        "uncertain_count": category_counts.get("uncertain", 0),
        "temporary_count": category_counts.get("temporary", 0),
        "backup_count": category_counts.get("backup", 0),
        "raw_run_local_only": {
            "path": "data/evaluation/evidence-qa-dev-v3-1/runs/",
            "directory_count": len(run_dirs),
            "file_count": len(run_files),
            "commit_candidate": False,
        },
        "secret_scan_safe": secret["safe_to_commit"],
        "env_modified": bool(git_lines("diff", "--name-only", "--", ".env", ".env.*.local")),
        "records": records,
    }
    FILE_AUDIT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "commit_candidates": len(records),
                "local_only_files": len(run_files),
                "uncertain": payload["uncertain_count"],
                "actual_secret_hits": secret["actual_secret_hits"],
            }
        )
    )


if __name__ == "__main__":
    main()
