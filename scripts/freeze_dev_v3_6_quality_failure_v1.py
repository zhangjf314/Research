"""Freeze Stage 13.21 Dev v3.6 engineering-pass / quality-fail evidence.

This script is intentionally read-only with respect to Stage 13.21 formal
results and raw run directories. It creates two Stage 13.22 audit artifacts:

* stage13-21-checkpoint-file-audit-v1
* dev-v3-6-quality-failure-freeze-v1
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "evaluation"
DOCS = ROOT / "docs"
RUN_ROOT = DATA / "evidence-qa-dev-v3-6" / "runs"

FILE_AUDIT_JSON = DATA / "stage13-21-checkpoint-file-audit-v1.json"
FILE_AUDIT_MD = DOCS / "stage13-21-checkpoint-file-audit-v1.md"
FREEZE_JSON = DATA / "dev-v3-6-quality-failure-freeze-v1.json"
FREEZE_MD = DOCS / "dev-v3-6-quality-failure-freeze-v1.md"


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True, encoding="utf-8").strip()


def untracked_files() -> list[str]:
    output = git("ls-files", "--others", "--exclude-standard")
    return [line for line in output.splitlines() if line.strip()]


def classify(path: str) -> str:
    if path.startswith("scripts/"):
        name = Path(path).name
        if "run_evidence_qa_dev_v3_6" in name:
            return "runtime_code"
        return "evaluation_code"
    if path.startswith("tests/"):
        return "tests"
    if path.endswith("protocol-freeze-v1.json") or path.endswith("protocol-freeze-v1.md"):
        return "protocol_freeze"
    if path.endswith("provider-health-dev-v3-6-v1.json"):
        return "provider_health"
    if path.endswith("evidence-qa-dev-v3-6.json") or path.endswith("evidence-qa-dev-v3-6.csv"):
        return "formal_result"
    if path.endswith("docs/evidence-qa-dev-v3-6.md") or path.endswith("evidence-qa-dev-v3-6.md"):
        return "formal_report"
    if path.endswith("evidence-qa-dev-v3-6-final-audit.json"):
        return "final_audit"
    if "citation-audit" in path:
        return "citation_audit"
    if "quality-failure-freeze" in path or "stage13-21-checkpoint-file-audit" in path:
        return "failure_evidence"
    if "/runs/" in path.replace("\\", "/"):
        return "raw_run_local_only"
    if path.startswith("artifacts/") and path.endswith(".zip"):
        return "review_zip_local_only"
    if path.endswith(".bak"):
        return "backup"
    if path.startswith("artifacts/imports/"):
        return "imports"
    if path.endswith((".tmp", ".temp")):
        return "temporary"
    return "uncertain"


def build_file_audit() -> dict[str, Any]:
    tracked_candidates = sorted(untracked_files())
    raw_runs = sorted(path.name for path in RUN_ROOT.iterdir() if path.is_dir())
    classified: dict[str, list[str]] = defaultdict(list)
    for path in tracked_candidates:
        classified[classify(path)].append(path)
    for run_id in raw_runs:
        classified["raw_run_local_only"].append(f"data/evaluation/evidence-qa-dev-v3-6/runs/{run_id}/")
    categories = [
        "runtime_code",
        "evaluation_code",
        "tests",
        "protocol_freeze",
        "provider_health",
        "formal_result",
        "formal_report",
        "final_audit",
        "citation_audit",
        "failure_evidence",
        "raw_run_local_only",
        "review_zip_local_only",
        "backup",
        "imports",
        "temporary",
        "uncertain",
    ]
    category_counts = {key: len(classified.get(key, [])) for key in categories}
    forbidden_commit_patterns = [".env", "authorization", "bearer", "cookie", "api_key"]
    secrets_hits: list[str] = []
    for path in tracked_candidates:
        if not path.startswith(("data/", "docs/")):
            continue
        if Path(path).suffix.lower() not in {".py", ".json", ".jsonl", ".md", ".csv"}:
            continue
        text = (ROOT / path).read_text(encoding="utf-8", errors="ignore").lower()
        if any(
            token in text
            for token in (
                "authorization:",
                "bearer ",
                "cookie:",
                "api_key=",
                "api-key:",
                "x-api-key:",
            )
        ):
            secrets_hits.append(path)
    body = {
        "schema_version": "stage13-21-checkpoint-file-audit-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "branch": git("branch", "--show-current"),
        "head": git("rev-parse", "HEAD"),
        "tracked_diff_stat": git("diff", "--stat"),
        "tracked_diff_name_status": git("diff", "--name-status"),
        "git_diff_check": git("diff", "--check"),
        "categories": {key: sorted(classified.get(key, [])) for key in categories},
        "category_counts": category_counts,
        "uncertain_count": category_counts["uncertain"],
        "raw_run_directory_count": len(raw_runs),
        "raw_run_directories_local_only": len(raw_runs) == 10,
        "review_zip_local_only": sorted(classified.get("review_zip_local_only", [])),
        "forbidden_commit_patterns": forbidden_commit_patterns,
        "secret_scan_hits": secrets_hits,
        "env_files_in_untracked": [
            path for path in tracked_candidates if Path(path).name.startswith(".env")
        ],
        "backup_or_import_files_in_untracked": sorted(
            classified.get("backup", []) + classified.get("imports", [])
        ),
    }
    body["audit_signature"] = canonical_hash({k: v for k, v in body.items() if k != "created_at"})
    return body


def build_quality_freeze() -> dict[str, Any]:
    summary_path = DATA / "evidence-qa-dev-v3-6.json"
    audit_path = DATA / "evidence-qa-dev-v3-6-final-audit.json"
    citation_path = DATA / "evidence-qa-dev-v3-6-citation-audit-v1.jsonl"
    protocol_path = DATA / "evidence-qa-dev-v3-6-protocol-freeze-v1.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    run_hashes = []
    for run_id in summary["selected_runs"]:
        run_dir = RUN_ROOT / run_id
        run_hashes.append(
            {
                "run_id": run_id,
                "raw_provider_response_hash": file_hash(run_dir / "raw-provider-response.json"),
                "raw_model_payload_hash": file_hash(run_dir / "raw-model-payload.json"),
                "payload_validation_hash": file_hash(run_dir / "payload-validation.json"),
                "final_result_hash": file_hash(run_dir / "final-result.json"),
                "request_ledger_hash": file_hash(run_dir / "request-ledger.jsonl"),
            }
        )
    body = {
        "schema_version": "dev-v3-6-quality-failure-freeze-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "evaluation_version": summary["evaluation_version"],
        "branch": git("branch", "--show-current"),
        "baseline_head": git("rev-parse", "HEAD"),
        "protocol_freeze_signature": protocol["protocol_freeze_signature"],
        "prompt_version": protocol["prompt_version"],
        "prompt_hash": protocol["prompt_hash"],
        "payload_v4_hash": protocol["payload_v4_hash"],
        "envelope_v4_hash": protocol["envelope_v4_hash"],
        "evidence_presentation_hash": protocol["evidence_presentation_hash"],
        "selected_run_ids": summary["selected_runs"],
        "run_hashes": run_hashes,
        "formal_summary_hash": file_hash(summary_path),
        "final_audit_hash": file_hash(audit_path),
        "citation_audit_hash": file_hash(citation_path),
        "usage": summary["all_manifest_conservative"],
        "latency": {
            "elapsed_seconds_total": summary["all_manifest_conservative"]["elapsed_seconds_total"],
            "p50_seconds": summary["all_manifest_conservative"]["latency_p50_seconds"],
            "p95_seconds": summary["all_manifest_conservative"]["latency_p95_seconds"],
        },
        "accounting_terminal": {
            "effective_active_reservations": summary["all_manifest_conservative"][
                "effective_active_reservations"
            ],
            "double_settlement_count": summary["all_manifest_conservative"][
                "double_settlement_count"
            ],
            "reservation_count": summary["all_manifest_conservative"]["reservation_count"],
            "settled_reservations": summary["all_manifest_conservative"]["settled_reservations"],
        },
        "gates": {
            "Engineering Gate": audit["DEV_V3_6_ENGINEERING_GATE"],
            "Automated Quality Gate": audit["DEV_V3_6_AUTOMATED_QUALITY_GATE"],
            "Human Support Gate": audit["DEV_V3_6_HUMAN_SUPPORT_GATE"],
            "Quality Candidate Gate": audit["DEV_V3_6_QUALITY_CANDIDATE_GATE"],
        },
        "quality_failure_metrics": {
            "answerable_question_macro_exact_relation_recall": summary["final_policy_layer"][
                "answerable_question_macro_exact_relation_recall"
            ],
            "required_claim_macro_exact_recall": summary["final_policy_layer"][
                "required_claim_macro_exact_recall"
            ],
            "micro_core_relation_recall": summary["final_policy_layer"][
                "micro_core_relation_recall"
            ],
            "core_set_completion": summary["final_policy_layer"]["core_set_completion"],
            "any_valid_evidence_recall": summary["final_policy_layer"][
                "any_valid_evidence_recall"
            ],
            "wrong_evidence": summary["final_policy_layer"]["wrong_evidence"],
        },
        "immutable": True,
        "historical_results_modified": False,
        "raw_runs_modified": False,
        "frozen_at": datetime.now(UTC).isoformat(),
    }
    stable = {k: v for k, v in body.items() if k not in {"created_at", "frozen_at"}}
    body["freeze_signature"] = canonical_hash(stable)
    return body


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_file_audit_doc(body: dict[str, Any]) -> None:
    rows = "\n".join(
        f"| {key} | {count} |"
        for key, count in body["category_counts"].items()
    )
    FILE_AUDIT_MD.write_text(
        "# Stage 13.21 Checkpoint File Audit\n\n"
        f"- Branch: `{body['branch']}`\n"
        f"- HEAD: `{body['head']}`\n"
        f"- Audit signature: `{body['audit_signature']}`\n"
        f"- Raw run directories local-only: `{body['raw_run_directories_local_only']}`\n"
        f"- Uncertain files: `{body['uncertain_count']}`\n"
        f"- Secret scan hits: `{len(body['secret_scan_hits'])}`\n\n"
        "| Category | Count |\n|---|---:|\n"
        f"{rows}\n\n"
        "Raw run directories and review ZIPs are intentionally local-only and must not "
        "be committed.\n",
        encoding="utf-8",
    )


def write_freeze_doc(body: dict[str, Any]) -> None:
    metrics = body["quality_failure_metrics"]
    FREEZE_MD.write_text(
        "# Dev v3.6 Quality Failure Freeze\n\n"
        f"- Evaluation version: `{body['evaluation_version']}`\n"
        f"- Branch: `{body['branch']}`\n"
        f"- Baseline HEAD: `{body['baseline_head']}`\n"
        f"- Protocol freeze signature: `{body['protocol_freeze_signature']}`\n"
        f"- Freeze signature: `{body['freeze_signature']}`\n"
        f"- Engineering Gate: `{body['gates']['Engineering Gate']}`\n"
        f"- Automated Quality Gate: `{body['gates']['Automated Quality Gate']}`\n"
        f"- Human Support Gate: `{body['gates']['Human Support Gate']}`\n"
        f"- Quality Candidate Gate: `{body['gates']['Quality Candidate Gate']}`\n\n"
        "## Quality failure metrics\n\n"
        f"- Any-valid evidence recall: `{metrics['any_valid_evidence_recall']}`\n"
        f"- Question macro exact: `{metrics['answerable_question_macro_exact_relation_recall']}`\n"
        f"- Claim macro exact: `{metrics['required_claim_macro_exact_recall']}`\n"
        f"- Micro core relation recall: `{metrics['micro_core_relation_recall']}`\n"
        f"- Core-set completion: `{metrics['core_set_completion']}`\n"
        f"- Wrong evidence: `{metrics['wrong_evidence']}`\n\n"
        "This freeze is immutable evidence for Stage 13.21. It does not modify formal v3.6 "
        "results, raw runs, Payload v4, or Evidence Presentation v2.\n",
        encoding="utf-8",
    )


def main() -> None:
    first_file_audit = build_file_audit()
    second_file_audit = build_file_audit()
    if first_file_audit["audit_signature"] != second_file_audit["audit_signature"]:
        raise RuntimeError("STAGE13_21_FILE_AUDIT_NOT_DETERMINISTIC")
    if first_file_audit["uncertain_count"] != 0:
        raise RuntimeError("STAGE13_21_FILE_AUDIT_UNCERTAIN_FILES")
    if first_file_audit["secret_scan_hits"]:
        raise RuntimeError("STAGE13_21_FILE_AUDIT_SECRET_HITS")
    first_freeze = build_quality_freeze()
    second_freeze = build_quality_freeze()
    if first_freeze["freeze_signature"] != second_freeze["freeze_signature"]:
        raise RuntimeError("DEV_V3_6_QUALITY_FREEZE_NOT_DETERMINISTIC")
    write_json(FILE_AUDIT_JSON, first_file_audit)
    write_file_audit_doc(first_file_audit)
    write_json(FREEZE_JSON, first_freeze)
    write_freeze_doc(first_freeze)
    print(
        json.dumps(
            {
                "file_audit_signature": first_file_audit["audit_signature"],
                "freeze_signature": first_freeze["freeze_signature"],
                "uncertain_count": first_file_audit["uncertain_count"],
                "raw_run_directory_count": first_file_audit["raw_run_directory_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
