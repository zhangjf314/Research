"""Audit and freeze the failed Stage 13.16 Dev v3.4 evaluation."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/evaluation"
DOCS = ROOT / "docs"
RUN_ROOT = DATA / "evidence-qa-dev-v3-4/runs"
AUDIT = DATA / "stage13-16-checkpoint-file-audit-v1.json"
AUDIT_DOC = DOCS / "stage13-16-checkpoint-file-audit-v1.md"
FREEZE = DATA / "stage13-16-dev-v3-4-failure-freeze-v1.json"
FREEZE_DOC = DOCS / "stage13-16-dev-v3-4-failure-freeze-v1.md"

CLASSIFICATIONS = {
    "src/paper_research/generation/schema_reliability.py": "production_code",
    "scripts/evidence_qa_dev_v3_4_lib.py": "runtime_code",
    "scripts/run_evidence_qa_dev_v3_4.py": "runtime_code",
    "scripts/summarize_evidence_qa_dev_v3_4.py": "evaluation_code",
    "scripts/audit_evidence_qa_dev_v3_4.py": "evaluation_code",
    "scripts/audit_stage13_16_historical_protection_v1.py": "evaluation_code",
    "scripts/freeze_stage13_16_checkpoint_v1.py": "evaluation_code",
    "tests/test_stage13_16_dev_v3_4.py": "test",
    "data/evaluation/evidence-qa-dev-v3-4-protocol-freeze-v1.json": "protocol_freeze",
    "docs/evidence-qa-dev-v3-4-protocol-freeze-v1.md": "protocol_freeze",
    "data/evaluation/provider-health-dev-v3-4-v1.json": "provider_health",
    "data/evaluation/evidence-qa-dev-v3-4.json": "canonical_evaluation_data",
    "data/evaluation/evidence-qa-dev-v3-4.csv": "canonical_evaluation_data",
    "docs/evidence-qa-dev-v3-4.md": "canonical_report",
    "data/evaluation/stage13-16-historical-protection-v1.json": "historical_protection",
    "docs/stage13-16-historical-protection-v1.md": "historical_protection",
    "data/evaluation/evidence-qa-dev-v3-4-citation-audit-v1.jsonl": "citation_audit",
    "docs/evidence-qa-dev-v3-4-citation-audit-v1.md": "citation_audit",
    "data/evaluation/dev-v3-4-visible-id-namespace-audit-v1.json": "failure_evidence",
    "docs/dev-v3-4-visible-id-namespace-audit-v1.md": "failure_evidence",
    "data/evaluation/evidence-qa-dev-v3-4-final-audit.json": "failure_evidence",
    "data/evaluation/stage13-16-checkpoint-file-audit-v1.json": "failure_evidence",
    "docs/stage13-16-checkpoint-file-audit-v1.md": "failure_evidence",
    "data/evaluation/stage13-16-dev-v3-4-failure-freeze-v1.json": "failure_evidence",
    "docs/stage13-16-dev-v3-4-failure-freeze-v1.md": "failure_evidence",
    "artifacts/stage13-10-human-claim-gold-review-results.zip": "review_zip_local_only",
    "artifacts/stage13-9-human-citation-review-results.zip": "review_zip_local_only",
}

COMMITTABLE = {
    "production_code",
    "runtime_code",
    "evaluation_code",
    "test",
    "protocol_freeze",
    "provider_health",
    "canonical_evaluation_data",
    "canonical_report",
    "historical_protection",
    "citation_audit",
    "failure_evidence",
}


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_hash(path: Path) -> str:
    return digest_bytes(path.read_bytes())


def canonical_hash(value: Any) -> str:
    return digest_bytes(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    )


def git_paths() -> list[tuple[str, str]]:
    output = subprocess.check_output(
        ["git", "status", "--short"], cwd=ROOT, text=True, encoding="utf-8"
    )
    rows = []
    for line in output.splitlines():
        if not line:
            continue
        rows.append((line[:2].strip() or "M", line[3:].replace("\\", "/")))
    return rows


def build_file_audit() -> dict[str, Any]:
    rows = []
    for status, path in git_paths():
        category = CLASSIFICATIONS.get(path, "uncertain")
        rows.append(
            {
                "path": path,
                "git_status": status,
                "classification": category,
                "commit_candidate": category in COMMITTABLE,
                "local_only": category not in COMMITTABLE,
            }
        )
    raw_runs = []
    for run_dir in sorted(RUN_ROOT.glob("live-dev-v3-4-*")):
        ignored = (
            subprocess.run(
                ["git", "check-ignore", "-q", str(run_dir.relative_to(ROOT))],
                cwd=ROOT,
                check=False,
            ).returncode
            == 0
        )
        raw_runs.append(
            {
                "path": str(run_dir.relative_to(ROOT)).replace("\\", "/"),
                "classification": "raw_run_local_only",
                "file_count": sum(path.is_file() for path in run_dir.iterdir()),
                "git_ignored": ignored,
                "local_only": True,
            }
        )
    secret_patterns = (
        re.compile(r"authorization\s*:\s*bearer\s+[a-z0-9_-]{16,}", re.IGNORECASE),
        re.compile(
            r"(?:api[_-]?key|llm_api_key)\s*[:=]\s*[\"'][a-z0-9_-]{16,}[\"']",
            re.IGNORECASE,
        ),
    )
    secret_hits = []
    for row in rows:
        if not row["commit_candidate"]:
            continue
        path = ROOT / row["path"]
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        for pattern in secret_patterns:
            if pattern.search(text):
                secret_hits.append({"path": row["path"], "pattern": pattern.pattern})
    uncertain = [row for row in rows if row["classification"] == "uncertain"]
    body = {
        "schema_version": "stage13-16-checkpoint-file-audit-v1",
        "files": rows,
        "raw_runs": raw_runs,
        "classification_counts": dict(
            Counter(row["classification"] for row in rows)
            + Counter(row["classification"] for row in raw_runs)
        ),
        "uncertain_count": len(uncertain),
        "raw_run_count": len(raw_runs),
        "raw_runs_local_only": len(raw_runs) == 10
        and all(row["git_ignored"] for row in raw_runs),
        "review_zip_local_only_count": sum(
            row["classification"] == "review_zip_local_only" for row in rows
        ),
        "secret_hits": secret_hits,
        "env_files_in_commit": [
            row["path"]
            for row in rows
            if row["commit_candidate"]
            and Path(row["path"]).name.startswith(".env")
        ],
        "backup_or_imports_in_commit": [
            row["path"]
            for row in rows
            if row["commit_candidate"]
            and (
                ".bak" in row["path"]
                or row["path"].startswith("artifacts/imports/")
            )
        ],
        "gate": "PASSED"
        if not uncertain
        and len(raw_runs) == 10
        and all(row["git_ignored"] for row in raw_runs)
        and not secret_hits
        else "FAILED",
    }
    return body


def write_file_audit() -> dict[str, Any]:
    body = build_file_audit()
    AUDIT.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
    AUDIT_DOC.write_text(
        "# Stage 13.16 Checkpoint File Audit\n\n"
        f"- Classified workspace files: {len(body['files'])}\n"
        f"- Uncertain: {body['uncertain_count']}\n"
        f"- Raw runs local-only: {body['raw_run_count']}/10\n"
        f"- Review ZIPs local-only: {body['review_zip_local_only_count']}/2\n"
        f"- Secret hits: {len(body['secret_hits'])}\n"
        f"- Gate: `{body['gate']}`\n",
        encoding="utf-8",
    )
    return body


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_failure_freeze() -> dict[str, Any]:
    protocol = load_json(DATA / "evidence-qa-dev-v3-4-protocol-freeze-v1.json")
    summary = load_json(DATA / "evidence-qa-dev-v3-4.json")
    final_audit = load_json(DATA / "evidence-qa-dev-v3-4-final-audit.json")
    health = load_json(DATA / "provider-health-dev-v3-4-v1.json")
    runs = []
    statuses: Counter[str] = Counter()
    missing_refusal = 0
    for run_dir in sorted(RUN_ROOT.glob("live-dev-v3-4-*")):
        final = load_json(run_dir / "final-result.json")
        raw = load_json(run_dir / "raw-model-payload.json")
        structural = load_json(run_dir / "structural-validation.json")
        envelope = load_json(run_dir / "provider-response-envelope.json")
        for slot in raw.get("required_claim_results", []):
            statuses[str(slot.get("status", "<missing>"))] += 1
        missing_refusal += "refusal_reason" not in raw
        runs.append(
            {
                "question_id": final["question_id"],
                "run_id": final["run_id"],
                "status": final["status"],
                "delivered_messages_hash": final["delivered_messages_hash"],
                "raw_response_sha256": file_hash(run_dir / "raw-provider-response.json"),
                "raw_model_payload_sha256": file_hash(run_dir / "raw-model-payload.json"),
                "structural_validation_sha256": file_hash(
                    run_dir / "structural-validation.json"
                ),
                "structural_validation": structural,
                "slot_validation": {
                    "required_claim_count": final["required_claim_count"],
                    "raw_slot_count": final["raw_slot_count"],
                    "strict_pass": final["status"] == "completed",
                },
                "final_result_sha256": file_hash(run_dir / "final-result.json"),
                "usage": envelope.get("usage"),
                "accounting_terminal": {
                    "settled": final["settled_reservation_count"],
                    "released": final["released_reservation_count"],
                    "billing_unknown": final["billing_unknown_reservation_count"],
                    "active_reserved_tokens": final["active_reserved_tokens"],
                },
            }
        )
    body = {
        "schema_version": "stage13-16-dev-v3-4-failure-freeze-v1",
        "evaluation_version": "evidence-qa-dev-v3.4",
        "protocol_freeze_signature": protocol["protocol_freeze_signature"],
        "prompt_version": protocol["prompt_version"],
        "prompt_hash": protocol["prompt_template_hash"],
        "payload_schema_version": protocol["model_payload_schema_version"],
        "payload_schema_hash": protocol["model_payload_schema_hash"],
        "envelope_schema_version": protocol["local_envelope_schema_version"],
        "envelope_schema_hash": protocol["local_envelope_schema_hash"],
        "canonicalization_version": protocol["policy_versions"]["canonicalization"],
        "canonicalization_hash": protocol["policy_hashes"]["canonicalization"],
        "runs": runs,
        "run_count": len(runs),
        "summary_sha256": file_hash(DATA / "evidence-qa-dev-v3-4.json"),
        "final_audit_sha256": file_hash(
            DATA / "evidence-qa-dev-v3-4-final-audit.json"
        ),
        "citation_audit_sha256": file_hash(
            DATA / "evidence-qa-dev-v3-4-citation-audit-v1.jsonl"
        ),
        "historical_protection_sha256": file_hash(
            DATA / "stage13-16-historical-protection-v1.json"
        ),
        "failure_taxonomy": {
            "raw_status_distribution": dict(sorted(statuses.items())),
            "missing_top_level_refusal_reason": missing_refusal,
            "validation_failures": summary["all_manifest_conservative"][
                "validation_failures"
            ],
            "completed": sum(run["status"] == "completed" for run in runs),
            "strict_structural_pass": summary["raw_payload_layer"][
                "structural_payload_success"
            ],
            "final_slots": summary["final_policy_layer"]["final_slot_success"],
        },
        "gate_results": {
            "provider_health": final_audit["dev_v3_4_provider_health"],
            "raw_payload": final_audit["dev_v3_4_raw_payload_gate"],
            "final_policy_engineering": final_audit[
                "dev_v3_4_final_policy_engineering_gate"
            ],
            "engineering": final_audit["dev_v3_4_engineering_gate"],
            "automated_quality": final_audit["dev_v3_4_automated_quality_gate"],
            "human_support": final_audit["dev_v3_4_human_support_gate"],
            "quality_candidate": final_audit["dev_v3_4_quality_candidate_gate"],
            "ready_for_full_qa": final_audit["ready_for_full_qa"],
        },
        "immutable": True,
        "frozen_at": health["checked_at"],
    }
    body["failure_freeze_signature"] = canonical_hash(body)
    return body


def write_failure_freeze() -> dict[str, Any]:
    body = build_failure_freeze()
    if FREEZE.exists():
        existing = load_json(FREEZE)
        if existing != body:
            raise RuntimeError("STAGE13_16_FAILURE_FREEZE_DRIFT")
    else:
        FREEZE.write_text(
            json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        FREEZE_DOC.write_text(
            "# Stage 13.16 Dev v3.4 Failure Freeze\n\n"
            f"- Signature: `{body['failure_freeze_signature']}`\n"
            f"- Runs: {body['run_count']}/10\n"
            f"- Raw status distribution: `{body['failure_taxonomy']['raw_status_distribution']}`\n"
            "- Historical strict structural pass: 1/10; final slots: 0/27.\n"
            "- Engineering and quality gates remain failed and immutable.\n",
            encoding="utf-8",
        )
    return body


def main() -> None:
    audit = write_file_audit()
    if audit["gate"] != "PASSED":
        raise SystemExit("STAGE13_16_CHECKPOINT_FILE_AUDIT_FAILED")
    freeze = write_failure_freeze()
    print(
        json.dumps(
            {
                "file_audit_gate": audit["gate"],
                "uncertain": audit["uncertain_count"],
                "raw_runs_local_only": audit["raw_runs_local_only"],
                "failure_freeze_signature": freeze["failure_freeze_signature"],
                "historical_gate": "FAILED_AND_PRESERVED",
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
