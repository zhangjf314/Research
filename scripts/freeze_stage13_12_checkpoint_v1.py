"""Audit and freeze the failed Stage 13.12 Dev v3.2 checkpoint."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/evaluation"
DOCS = ROOT / "docs"
RUN_ROOT = DATA / "evidence-qa-dev-v3-2/runs"
AUDIT_JSON = DATA / "stage13-12-checkpoint-file-audit-v1.json"
AUDIT_MD = DOCS / "stage13-12-checkpoint-file-audit-v1.md"
FREEZE_JSON = DATA / "stage13-12-dev-v3-2-failure-freeze-v1.json"
FREEZE_MD = DOCS / "stage13-12-dev-v3-2-failure-freeze-v1.md"

FORMAL_FILES = {
    "src/paper_research/generation/required_claim_output.py": "production_code",
    "scripts/audit_evidence_qa_dev_v3_2.py": "evaluation_code",
    "scripts/evidence_qa_dev_v3_2_lib.py": "evaluation_code",
    "scripts/finalize_evidence_qa_dev_v3_2_runs.py": "evaluation_code",
    "scripts/run_evidence_qa_dev_v3_2.py": "evaluation_code",
    "scripts/summarize_evidence_qa_dev_v3_2.py": "evaluation_code",
    "scripts/freeze_stage13_12_checkpoint_v1.py": "evaluation_code",
    "tests/test_stage13_12_dev_v3_2_live.py": "test",
    "data/evaluation/evidence-qa-dev-v3-2-manifest.json": "manifest",
    "data/evaluation/evidence-qa-dev-v3-2.json": "canonical_evaluation_data",
    "data/evaluation/evidence-qa-dev-v3-2.csv": "canonical_evaluation_data",
    "docs/evidence-qa-dev-v3-2-manifest.md": "canonical_report",
    "docs/evidence-qa-dev-v3-2.md": "canonical_report",
    "data/evaluation/provider-health-dev-v3-2-v1.json": "provider_health",
    "data/evaluation/evidence-qa-dev-v3-2-citation-audit-v1.jsonl": "citation_audit",
    "docs/evidence-qa-dev-v3-2-citation-audit-v1.md": "citation_audit",
    "data/evaluation/evidence-qa-dev-v3-2-final-audit.json": "failure_evidence",
    "data/evaluation/stage13-12-checkpoint-file-audit-v1.json": "failure_evidence",
    "docs/stage13-12-checkpoint-file-audit-v1.md": "failure_evidence",
    "data/evaluation/stage13-12-dev-v3-2-failure-freeze-v1.json": "failure_evidence",
    "docs/stage13-12-dev-v3-2-failure-freeze-v1.md": "failure_evidence",
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256_bytes(encoded)


def git_lines(*args: str) -> list[str]:
    output = subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, encoding="utf-8"
    )
    return [line for line in output.splitlines() if line]


def build_file_audit() -> dict[str, Any]:
    changed = set(git_lines("diff", "--name-only"))
    changed.update(git_lines("ls-files", "--others", "--exclude-standard"))
    entries: list[dict[str, Any]] = []
    for relative in sorted(changed):
        normalized = relative.replace("\\", "/")
        if normalized.startswith("data/evaluation/evidence-qa-dev-v3-2/runs/"):
            category = "raw_run_local_only"
        elif normalized.startswith("artifacts/imports/"):
            category = "temporary"
        elif normalized.endswith(".bak"):
            category = "backup"
        elif normalized in {
            "artifacts/stage13-10-human-claim-gold-review-results.zip",
            "artifacts/stage13-9-human-citation-review-results.zip",
        }:
            category = "raw_run_local_only"
        else:
            category = FORMAL_FILES.get(normalized, "uncertain")
        entries.append(
            {
                "path": normalized,
                "category": category,
                "commit": category
                not in {"raw_run_local_only", "temporary", "backup", "uncertain"},
            }
        )
    counts = {
        category: sum(row["category"] == category for row in entries)
        for category in (
            "production_code",
            "evaluation_code",
            "test",
            "manifest",
            "canonical_evaluation_data",
            "canonical_report",
            "provider_health",
            "citation_audit",
            "failure_evidence",
            "raw_run_local_only",
            "temporary",
            "backup",
            "uncertain",
        )
    }
    run_dirs = sorted(path.name for path in RUN_ROOT.glob("live-dev-v3-2-*"))
    return {
        "schema_version": "stage13-12-checkpoint-file-audit-v1",
        "branch": git_lines("branch", "--show-current")[0],
        "head": git_lines("rev-parse", "HEAD")[0],
        "entries": entries,
        "counts": counts,
        "raw_run_directories": run_dirs,
        "raw_run_directory_count": len(run_dirs),
        "historical_review_zips_local_only": [
            "artifacts/stage13-10-human-claim-gold-review-results.zip",
            "artifacts/stage13-9-human-citation-review-results.zip",
        ],
        "uncertain_zero": counts["uncertain"] == 0,
        "secrets_in_commit": False,
        "env_in_commit": False,
        "backup_in_commit": False,
    }


def write_file_audit(body: dict[str, Any]) -> None:
    AUDIT_JSON.write_text(
        json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# Stage 13.12 Checkpoint File Audit",
        "",
        f"- Branch / HEAD: `{body['branch']}` / `{body['head']}`",
        f"- Raw run directories kept local-only: {body['raw_run_directory_count']}",
        f"- Uncertain files: {body['counts']['uncertain']}",
        "- Historical review ZIPs, raw runs, imports, backups, `.env`, secrets, and "
        "provider headers are excluded from the checkpoint.",
        "",
        "| Category | Count |",
        "|---|---:|",
    ]
    lines.extend(f"| {key} | {value} |" for key, value in body["counts"].items())
    lines.extend(["", "| Path | Category | Commit |", "|---|---|---|"])
    lines.extend(
        f"| `{row['path']}` | {row['category']} | {str(row['commit']).lower()} |"
        for row in body["entries"]
    )
    AUDIT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_freeze() -> dict[str, Any]:
    summary_path = DATA / "evidence-qa-dev-v3-2.json"
    final_audit_path = DATA / "evidence-qa-dev-v3-2-final-audit.json"
    citation_audit_path = DATA / "evidence-qa-dev-v3-2-citation-audit-v1.jsonl"
    manifest = json.loads(
        (DATA / "evidence-qa-dev-v3-2-manifest.json").read_text(encoding="utf-8")
    )
    runs = []
    for run_dir in sorted(RUN_ROOT.glob("live-dev-v3-2-*")):
        result_path = run_dir / "final-result.json"
        raw_path = run_dir / "raw-provider-response.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        content = raw.get("choices", [{}])[0].get("message", {}).get("content", "")
        try:
            parsed = json.loads(content)
            raw_json_valid = True
        except json.JSONDecodeError:
            parsed = None
            raw_json_valid = False
        runs.append(
            {
                "run_id": result["run_id"],
                "question_id": result["question_id"],
                "terminal_status": result["status"],
                "raw_response_file_sha256": sha256_file(raw_path),
                "raw_content_sha256": sha256_bytes(content.encode("utf-8")),
                "result_file_sha256": sha256_file(result_path),
                "raw_json_valid": raw_json_valid,
                "raw_prompt_version": parsed.get("prompt_version")
                if isinstance(parsed, dict)
                else None,
                "raw_schema_success": bool(result.get("raw_answer")),
                "final_schema_success": result["status"] == "completed",
                "validation_failure": result.get("failure_type"),
                "usage": result.get("usage"),
                "reservation": {
                    "reserved_tokens": 24000,
                    "active_reserved_tokens": result["active_reserved_tokens"],
                    "historical_terminal_accounting_event_present": (
                        result["active_reserved_tokens"] == 0
                    ),
                },
            }
        )
    body = {
        "schema_version": "stage13-12-dev-v3-2-failure-freeze-v1",
        "evaluation_version": "evidence-qa-dev-v3.2",
        "manifest_hash": manifest["manifest_hash"],
        "prompt_hash": manifest["configuration"]["prompt_hash"],
        "schema_hash": manifest["configuration"]["schema_hash"],
        "policy_hash": manifest["configuration"]["combined_policy_hash"],
        "runs": runs,
        "formal_summary_sha256": sha256_file(summary_path),
        "final_audit_sha256": sha256_file(final_audit_path),
        "citation_audit_sha256": sha256_file(citation_audit_path),
        "historical_gate_status": {
            "engineering": "FAILED",
            "automated_quality": "FAILED",
            "human_support": "PENDING",
            "quality_candidate": "FAILED",
        },
        "historical_active_reservations": 4,
        "historical_active_reserved_tokens": 96000,
        "immutable": True,
    }
    body["freeze_signature"] = canonical_hash(body)
    return body


def write_freeze(body: dict[str, Any]) -> dict[str, Any]:
    if FREEZE_JSON.exists():
        existing = json.loads(FREEZE_JSON.read_text(encoding="utf-8"))
        comparable = {key: value for key, value in existing.items() if key != "frozen_at"}
        if comparable != body:
            raise RuntimeError("STAGE13_12_FAILURE_FREEZE_CHANGED")
        return existing
    body = {**body, "frozen_at": datetime.now(UTC).isoformat()}
    FREEZE_JSON.write_text(
        json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# Stage 13.12 Dev v3.2 Failure Freeze",
        "",
        f"- Freeze signature: `{body['freeze_signature']}`",
        f"- Runs frozen: {len(body['runs'])}",
        "- Historical gates: engineering FAILED; automated quality FAILED; "
        "human support PENDING; quality candidate FAILED.",
        "- Historical unclosed reservations: 4 / 96,000 tokens.",
        "- The formal summary, final audit, citation audit, raw responses, and run "
        "outcomes are immutable. This freeze does not normalize or repair failures.",
        "",
        "| Question | Run | Status | Failure | Active reserved |",
        "|---|---|---|---|---:|",
    ]
    lines.extend(
        f"| {row['question_id']} | `{row['run_id']}` | {row['terminal_status']} | "
        f"{row['validation_failure'] or '-'} | "
        f"{row['reservation']['active_reserved_tokens']} |"
        for row in body["runs"]
    )
    FREEZE_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return body


def main() -> None:
    audit = build_file_audit()
    if not audit["uncertain_zero"]:
        raise RuntimeError("STAGE13_12_CHECKPOINT_FILE_AUDIT_UNCERTAIN")
    write_file_audit(audit)
    freeze = write_freeze(build_freeze())
    print(
        json.dumps(
            {
                "file_audit_uncertain": audit["counts"]["uncertain"],
                "raw_run_directories": audit["raw_run_directory_count"],
                "freeze_signature": freeze["freeze_signature"],
                "historical_gate": "FAILED_AND_PRESERVED",
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
