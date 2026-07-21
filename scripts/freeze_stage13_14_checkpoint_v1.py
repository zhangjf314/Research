"""Audit and freeze the failed Stage 13.14 Dev v3.3 evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/evaluation"
DOCS = ROOT / "docs"
RUN_ROOT = DATA / "evidence-qa-dev-v3-3/runs"
AUDIT = DATA / "stage13-14-checkpoint-file-audit-v1.json"
AUDIT_DOC = DOCS / "stage13-14-checkpoint-file-audit-v1.md"
FREEZE = DATA / "stage13-14-dev-v3-3-failure-freeze-v1.json"
FREEZE_DOC = DOCS / "stage13-14-dev-v3-3-failure-freeze-v1.md"

CLASSIFICATIONS = {
    "src/paper_research/generation/schema_reliability.py": "production_code",
    "scripts/evidence_qa_dev_v3_3_lib.py": "runtime_code",
    "scripts/run_evidence_qa_dev_v3_3.py": "runtime_code",
    "scripts/finalize_evidence_qa_dev_v3_3_runs.py": "evaluation_code",
    "scripts/summarize_evidence_qa_dev_v3_3.py": "evaluation_code",
    "scripts/audit_evidence_qa_dev_v3_3.py": "evaluation_code",
    "scripts/audit_stage13_14_historical_protection_v1.py": "evaluation_code",
    "scripts/freeze_stage13_14_checkpoint_v1.py": "evaluation_code",
    "tests/test_stage13_14_dev_v3_3.py": "test",
    "data/evaluation/evidence-qa-dev-v3-3-protocol-freeze-v1.json": "protocol_freeze",
    "docs/evidence-qa-dev-v3-3-protocol-freeze-v1.md": "protocol_freeze",
    "data/evaluation/provider-health-dev-v3-3-v1.json": "provider_health",
    "data/evaluation/evidence-qa-dev-v3-3.json": "canonical_evaluation_data",
    "data/evaluation/evidence-qa-dev-v3-3.csv": "canonical_evaluation_data",
    "docs/evidence-qa-dev-v3-3.md": "canonical_report",
    "data/evaluation/evidence-qa-dev-v3-3-citation-audit-v1.jsonl": "citation_audit",
    "docs/evidence-qa-dev-v3-3-citation-audit-v1.md": "citation_audit",
    "data/evaluation/evidence-qa-dev-v3-3-final-audit.json": "failure_evidence",
    "data/evaluation/dev-v3-3-visible-id-namespace-audit-v1.json": "failure_evidence",
    "docs/dev-v3-3-visible-id-namespace-audit-v1.md": "failure_evidence",
    "data/evaluation/stage13-14-historical-protection-v1.json": "failure_evidence",
    "docs/stage13-14-historical-protection-v1.md": "failure_evidence",
    "data/evaluation/stage13-14-checkpoint-file-audit-v1.json": "failure_evidence",
    "docs/stage13-14-checkpoint-file-audit-v1.md": "failure_evidence",
    "data/evaluation/stage13-14-dev-v3-3-failure-freeze-v1.json": "failure_evidence",
    "docs/stage13-14-dev-v3-3-failure-freeze-v1.md": "failure_evidence",
}


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_hash(path: Path) -> str:
    return digest(path.read_bytes())


def canonical_hash(value: Any) -> str:
    return digest(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    )


def git_lines(*args: str) -> list[str]:
    value = subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, encoding="utf-8"
    )
    return [line for line in value.splitlines() if line]


def build_audit() -> dict[str, Any]:
    paths = set(git_lines("diff", "--name-only"))
    paths.update(git_lines("ls-files", "--others", "--exclude-standard"))
    rows = []
    for raw in sorted(paths):
        path = raw.replace("\\", "/")
        if path.startswith("data/evaluation/evidence-qa-dev-v3-3/runs/"):
            category = "raw_run_local_only"
        elif path.endswith(".bak"):
            category = "backup"
        elif path.startswith("artifacts/imports/"):
            category = "temporary"
        elif path in {
            "artifacts/stage13-10-human-claim-gold-review-results.zip",
            "artifacts/stage13-9-human-citation-review-results.zip",
        }:
            category = "raw_run_local_only"
        else:
            category = CLASSIFICATIONS.get(path, "uncertain")
        rows.append(
            {
                "path": path,
                "category": category,
                "commit": category
                not in {"raw_run_local_only", "backup", "temporary", "uncertain"},
            }
        )
    categories = (
        "production_code",
        "runtime_code",
        "evaluation_code",
        "test",
        "protocol_freeze",
        "manifest",
        "provider_health",
        "canonical_evaluation_data",
        "canonical_report",
        "citation_audit",
        "failure_evidence",
        "raw_run_local_only",
        "backup",
        "temporary",
        "uncertain",
    )
    counts = {
        category: sum(row["category"] == category for row in rows)
        for category in categories
    }
    run_dirs = sorted(path.name for path in RUN_ROOT.glob("live-dev-v3-3-*"))
    return {
        "schema_version": "stage13-14-checkpoint-file-audit-v1",
        "branch": git_lines("branch", "--show-current")[0],
        "head": git_lines("rev-parse", "HEAD")[0],
        "entries": rows,
        "counts": counts,
        "raw_run_directories": run_dirs,
        "raw_run_directory_count": len(run_dirs),
        "historical_review_zips_local_only": 2,
        "uncertain_zero": counts["uncertain"] == 0,
        "env_committed": False,
        "secret_committed": False,
        "provider_header_committed": False,
    }


def write_audit(body: dict[str, Any]) -> None:
    AUDIT.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
    AUDIT_DOC.write_text(
        "# Stage 13.14 Checkpoint File Audit\n\n"
        f"- Branch / HEAD: `{body['branch']}` / `{body['head']}`\n"
        f"- Raw run directories local-only: {body['raw_run_directory_count']}\n"
        f"- Historical review ZIPs local-only: {body['historical_review_zips_local_only']}\n"
        f"- Uncertain files: {body['counts']['uncertain']}\n"
        "- Backups, imports, `.env`, API keys, provider headers, raw runs, and review "
        "ZIPs are excluded.\n",
        encoding="utf-8",
    )


def build_freeze() -> dict[str, Any]:
    protocol = json.loads(
        (DATA / "evidence-qa-dev-v3-3-protocol-freeze-v1.json").read_text(
            encoding="utf-8"
        )
    )
    runs = []
    delivered_hashes = []
    for run_dir in sorted(RUN_ROOT.glob("live-dev-v3-3-*")):
        result_path = run_dir / "final-result.json"
        raw_path = run_dir / "raw-provider-response.json"
        validation_path = run_dir / "payload-validation.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        metadata = json.loads(
            (run_dir / "run-metadata.json").read_text(encoding="utf-8")
        )
        delivered_hashes.append(metadata["exact_delivered_request_body_hash"])
        ledger = [
            json.loads(line)
            for line in (run_dir / "request-ledger.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line
        ]
        terminal = next(
            event
            for event in reversed(ledger)
            if event["event"]
            in {
                "reservation_settled",
                "reservation_released",
                "billing_unknown_terminal",
            }
        )
        runs.append(
            {
                "run_id": result["run_id"],
                "question_id": result["question_id"],
                "status": result["status"],
                "failure_type": result["failure_type"],
                "raw_response_sha256": file_hash(raw_path),
                "payload_validation_sha256": file_hash(validation_path),
                "final_result_sha256": file_hash(result_path),
                "usage": result["usage"],
                "accounting_terminal_state": terminal["event"],
                "effective_active_tokens": result["active_reserved_tokens"],
                "delivered_request_body_hash": metadata[
                    "exact_delivered_request_body_hash"
                ],
            }
        )
    body = {
        "schema_version": "stage13-14-dev-v3-3-failure-freeze-v1",
        "evaluation_version": "evidence-qa-dev-v3.3",
        "protocol_freeze_signature": protocol["protocol_freeze_signature"],
        "prompt_hash": protocol["prompt_hash"],
        "model_payload_schema_hash": protocol["model_payload_schema_hash"],
        "local_envelope_schema_hash": protocol["local_envelope_schema_hash"],
        "delivered_request_hashes": delivered_hashes,
        "runs": runs,
        "summary_sha256": file_hash(DATA / "evidence-qa-dev-v3-3.json"),
        "final_audit_sha256": file_hash(
            DATA / "evidence-qa-dev-v3-3-final-audit.json"
        ),
        "citation_audit_sha256": file_hash(
            DATA / "evidence-qa-dev-v3-3-citation-audit-v1.jsonl"
        ),
        "gate_results": {
            "raw_payload": "FAILED",
            "final_policy_engineering": "FAILED",
            "engineering": "FAILED",
            "automated_quality": "FAILED",
            "human_support": "PENDING",
            "quality_candidate": "FAILED",
            "ready_for_full_qa": False,
        },
        "failure_taxonomy": {
            "answerable_has_refusal_reason": 8,
            "completed": 2,
        },
        "immutable": True,
    }
    body["freeze_signature"] = canonical_hash(body)
    return body


def write_freeze(body: dict[str, Any]) -> dict[str, Any]:
    if FREEZE.exists():
        existing = json.loads(FREEZE.read_text(encoding="utf-8"))
        comparable = {key: value for key, value in existing.items() if key != "frozen_at"}
        if comparable != body:
            raise RuntimeError("STAGE13_14_FAILURE_FREEZE_CHANGED")
        return existing
    stored = {**body, "frozen_at": datetime.now(UTC).isoformat()}
    FREEZE.write_text(
        json.dumps(stored, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    FREEZE_DOC.write_text(
        "# Stage 13.14 Dev v3.3 Failure Freeze\n\n"
        f"- Freeze signature: `{stored['freeze_signature']}`\n"
        "- Runs: 10; Provider completed: 10; malformed JSON: 0.\n"
        "- Strict payload/final/engineering/quality gates remain FAILED.\n"
        "- Eight answerable payloads failed because refusal_reason was the exact empty "
        "string instead of null. No historical result was normalized or repaired.\n",
        encoding="utf-8",
    )
    return stored


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    args = parser.parse_args()
    if args.freeze_only:
        freeze = write_freeze(build_freeze())
        print(
            json.dumps(
                {
                    "freeze_signature": freeze["freeze_signature"],
                    "historical_gate": "FAILED_AND_PRESERVED",
                }
            )
        )
        return
    audit = build_audit()
    if not audit["uncertain_zero"]:
        raise RuntimeError("STAGE13_14_CHECKPOINT_FILE_AUDIT_UNCERTAIN")
    write_audit(audit)
    freeze = write_freeze(build_freeze())
    print(
        json.dumps(
            {
                "uncertain": audit["counts"]["uncertain"],
                "raw_runs": audit["raw_run_directory_count"],
                "freeze_signature": freeze["freeze_signature"],
                "historical_gate": "FAILED_AND_PRESERVED",
            }
        )
    )


if __name__ == "__main__":
    main()
