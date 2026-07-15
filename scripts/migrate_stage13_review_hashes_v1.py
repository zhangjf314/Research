"""Migrate Stage 13 human-review source hashes to canonical-hash-v1."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from paper_research.evaluation.canonical_hash import (
    CANONICAL_HASH_VERSION,
    SOURCE_HASH_SCHEMA_VERSION,
    canonicalize_json_value,
    hash_with_metadata,
    legacy_text_hash_variants,
    sha256_canonical_json_file,
    verify_legacy_raw_hash,
)

try:
    from scripts.evidence_qa_dev_lib_v1 import canonical_hash, read_jsonl
    from scripts.review_evidence_qa_dev_citations_v1 import (
        AUDIT as DEV_V1_AUDIT,
    )
    from scripts.review_evidence_qa_dev_citations_v1 import (
        SOURCE_SPECS,
    )
    from scripts.review_evidence_qa_dev_citations_v1 import (
        validate as validate_dev_v1,
    )
    from scripts.review_evidence_qa_dev_citations_v1 import (
        write_jsonl as write_dev_v1,
    )
    from scripts.review_evidence_qa_dev_v2_citations_v1 import (
        AUDIT as DEV_V2_AUDIT,
    )
    from scripts.review_evidence_qa_dev_v2_citations_v1 import (
        EVIDENCE,
    )
    from scripts.review_evidence_qa_dev_v2_citations_v1 import (
        HUMAN_FIELDS as DEV_V2_HUMAN_FIELDS,
    )
    from scripts.review_evidence_qa_dev_v2_citations_v1 import (
        validate as validate_dev_v2,
    )
    from scripts.review_evidence_qa_dev_v2_citations_v1 import (
        write as write_dev_v2,
    )
except ModuleNotFoundError:
    from evidence_qa_dev_lib_v1 import canonical_hash, read_jsonl  # type: ignore[no-redef]
    from review_evidence_qa_dev_citations_v1 import (  # type: ignore[no-redef]
        AUDIT as DEV_V1_AUDIT,
    )
    from review_evidence_qa_dev_citations_v1 import (
        SOURCE_SPECS,
    )
    from review_evidence_qa_dev_citations_v1 import (
        validate as validate_dev_v1,
    )
    from review_evidence_qa_dev_citations_v1 import (
        write_jsonl as write_dev_v1,
    )
    from review_evidence_qa_dev_v2_citations_v1 import (  # type: ignore[no-redef]
        AUDIT as DEV_V2_AUDIT,
    )
    from review_evidence_qa_dev_v2_citations_v1 import (
        EVIDENCE,
    )
    from review_evidence_qa_dev_v2_citations_v1 import (
        HUMAN_FIELDS as DEV_V2_HUMAN_FIELDS,
    )
    from review_evidence_qa_dev_v2_citations_v1 import (
        validate as validate_dev_v2,
    )
    from review_evidence_qa_dev_v2_citations_v1 import (
        write as write_dev_v2,
    )

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/evaluation"
DOCS = ROOT / "docs"
ROOT_CAUSE_JSON = DATA / "stage13-7-hash-root-cause-v1.json"
ROOT_CAUSE_MD = DOCS / "stage13-7-hash-root-cause-v1.md"
MIGRATION_JSON = DATA / "stage13-review-hash-migration-v1.json"
MIGRATION_MD = DOCS / "stage13-review-hash-migration-v1.md"
INTEGRITY_JSON = DATA / "stage13-7-review-integrity-audit-v1.json"
INTEGRITY_MD = DOCS / "stage13-7-review-integrity-audit-v1.md"
TARGET = DATA / "evidence-qa-dev-v1.json"
LEGACY_DEV_SUMMARY_HASH = "23126a87deb978216fe56bc8518e25d99fb8c9b2e461640ebe727c80ea73d170"
V2_HASH_FIELDS = {
    "source_hash",
    "source_canonical_sha256",
    "source_hash_mode",
    "source_hash_schema_version",
    "source_raw_sha256_at_review",
    "source_legacy_raw_hash_verified_via_newline_normalization",
    "immutable_record_hash",
}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git(*args: str) -> str | None:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_root_cause() -> dict[str, Any]:
    data = TARGET.read_bytes()
    text = data.decode("utf-8-sig")
    lf_text = text.replace("\r\n", "\n").replace("\r", "\n")
    variants = legacy_text_hash_variants(TARGET)
    operation = verify_legacy_raw_hash(TARGET, LEGACY_DEV_SUMMARY_HASH)
    parsed = json.loads(text)
    payload = {
        "schema_version": "stage13-7-hash-root-cause-v1",
        "target_file": str(TARGET.relative_to(ROOT)).replace("\\", "/"),
        "legacy_recorded_raw_sha256": LEGACY_DEV_SUMMARY_HASH,
        "current_raw_sha256": variants["raw"],
        "lf_sha256": variants["lf"],
        "crlf_sha256": variants["crlf"],
        "bom_removed_sha256": variants["bom_removed"],
        "single_terminal_lf_sha256": variants["single_terminal_lf"],
        "canonical_json_v1_sha256": sha256_canonical_json_file(TARGET),
        "canonical_json_serialization_sha256": _sha(canonicalize_json_value(parsed)),
        "legacy_hash_reproduced": operation == "crlf",
        "legacy_match_operation": operation,
        "semantic_json_equal_after_newline_conversion": json.loads(lf_text)
        == json.loads(lf_text.replace("\n", "\r\n")),
        "current_encoding": "utf-8-no-bom" if not data.startswith(b"\xef\xbb\xbf") else "utf-8-bom",
        "current_crlf_count": data.count(b"\r\n"),
        "current_lf_count": data.count(b"\n"),
        "hash_generator": "scripts/review_evidence_qa_dev_citations_v1.py",
        "hash_validator": "scripts/review_evidence_qa_dev_citations_v1.py",
        "legacy_hash_mode": "binary read_bytes raw SHA-256",
        "legacy_default_encoding_dependency": False,
        "legacy_newline_dependency": True,
        "legacy_json_canonical_serialization": False,
        "git": {
            "core_autocrlf": _git("config", "--get", "core.autocrlf") or "unset",
            "core_eol": _git("config", "--get", "core.eol") or "unset",
            "attributes": _git("check-attr", "text", "eol", "--", str(TARGET.relative_to(ROOT))),
            "ls_files_eol": _git("ls-files", "--eol", str(TARGET.relative_to(ROOT))),
        },
        "root_cause": (
            "The historical review captured CRLF raw bytes. The checkout uses LF bytes. "
            "JSON values are identical, but raw byte SHA-256 is platform-sensitive."
        ),
    }
    if not payload["legacy_hash_reproduced"]:
        raise RuntimeError("HASH_MISMATCH_NOT_EXPLAINED_BY_TEXT_NORMALIZATION")
    return payload


def _v1_non_hash_snapshot(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        row["sample_id"]: {
            key: value for key, value in row.items() if key != "source_hashes"
        }
        for row in rows
    }


def _v2_non_hash_snapshot(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        row["sample_id"]: {
            key: value for key, value in row.items() if key not in V2_HASH_FIELDS
        }
        for row in rows
    }


def _human_snapshot(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    fields = {"human_review_status", "human_label", "reviewer", "reviewed_at", "review_notes"}
    return {row["sample_id"]: {key: row.get(key) for key in fields} for row in rows}


def migrate() -> tuple[dict[str, Any], dict[str, Any]]:
    v1_rows = read_jsonl(DEV_V1_AUDIT)
    v2_rows = read_jsonl(DEV_V2_AUDIT)
    if len(v1_rows) != 24 or len(v2_rows) != 57:
        raise RuntimeError("expected 24 Dev v1 and 57 Dev v2 citation review rows")
    before_human = {"dev_v1": _human_snapshot(v1_rows), "dev_v2": _human_snapshot(v2_rows)}
    before_non_hash = {
        "dev_v1": _v1_non_hash_snapshot(v1_rows),
        "dev_v2": _v2_non_hash_snapshot(v2_rows),
    }
    v1_operations: dict[str, str] = {}
    for name, (path, mode) in SOURCE_SPECS.items():
        legacy_key = f"{name}_sha256"
        legacy_values = {row["source_hashes"][legacy_key] for row in v1_rows}
        if len(legacy_values) != 1:
            raise RuntimeError(f"inconsistent legacy source hash: {name}")
        legacy = next(iter(legacy_values))
        operation = verify_legacy_raw_hash(path, legacy)
        v1_operations[name] = operation
        metadata = hash_with_metadata(path, mode)
        for row in v1_rows:
            hashes = row["source_hashes"]
            hashes.update(
                {
                    f"{name}_canonical_sha256": metadata["value"],
                    f"{name}_hash_mode": metadata["mode"],
                    f"{name}_hash_schema_version": metadata["schema_version"],
                    f"{name}_raw_sha256_at_review": legacy,
                    f"{name}_legacy_raw_hash_verified_via_newline_normalization": operation
                    != "raw",
                }
            )

    v2_legacy_values = {row["source_hash"] for row in v2_rows}
    if len(v2_legacy_values) != 1:
        raise RuntimeError("inconsistent Dev v2 legacy source hash")
    v2_legacy = next(iter(v2_legacy_values))
    v2_operation = verify_legacy_raw_hash(EVIDENCE, v2_legacy)
    v2_metadata = hash_with_metadata(EVIDENCE, "canonical_jsonl_v1")
    for row in v2_rows:
        row.update(
            {
                "source_canonical_sha256": v2_metadata["value"],
                "source_hash_mode": v2_metadata["mode"],
                "source_hash_schema_version": v2_metadata["schema_version"],
                "source_raw_sha256_at_review": row["source_hash"],
                "source_legacy_raw_hash_verified_via_newline_normalization": v2_operation
                != "raw",
            }
        )
        immutable = {
            key: value
            for key, value in row.items()
            if key not in DEV_V2_HUMAN_FIELDS | {"immutable_record_hash"}
        }
        row["immutable_record_hash"] = canonical_hash(immutable)

    after_human = {"dev_v1": _human_snapshot(v1_rows), "dev_v2": _human_snapshot(v2_rows)}
    after_non_hash = {
        "dev_v1": _v1_non_hash_snapshot(v1_rows),
        "dev_v2": _v2_non_hash_snapshot(v2_rows),
    }
    label_changes = sum(
        before_human[layer][sample]["human_label"]
        != after_human[layer][sample]["human_label"]
        for layer in before_human
        for sample in before_human[layer]
    )
    reviewer_changes = sum(
        any(
            before_human[layer][sample][field] != after_human[layer][sample][field]
            for field in ("reviewer", "reviewed_at", "review_notes")
        )
        for layer in before_human
        for sample in before_human[layer]
    )
    non_hash_changes = sum(
        before_non_hash[layer][sample] != after_non_hash[layer][sample]
        for layer in before_non_hash
        for sample in before_non_hash[layer]
    )
    if label_changes or reviewer_changes or non_hash_changes:
        raise RuntimeError("migration changed protected human or non-hash fields")

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    backups = []
    for path in (DEV_V1_AUDIT, DEV_V2_AUDIT):
        backup = path.with_name(f"{path.name}.pre-canonical-hash-migration.{stamp}.bak")
        shutil.copy2(path, backup)
        backups.append(str(backup.relative_to(ROOT)).replace("\\", "/"))
    write_dev_v1(DEV_V1_AUDIT, v1_rows)
    write_dev_v2(v2_rows)
    validate_dev_v1(read_jsonl(DEV_V1_AUDIT))
    validate_dev_v2(read_jsonl(DEV_V2_AUDIT))

    migration = {
        "schema_version": "stage13-review-hash-migration-v1",
        "canonicalization_version": CANONICAL_HASH_VERSION,
        "source_hash_schema_version": SOURCE_HASH_SCHEMA_VERSION,
        "affected_records": 81,
        "layers": {"dev_v1_citation_audit": 24, "dev_v2_citation_audit": 57},
        "legacy_raw_hashes_preserved": True,
        "v1_source_operations": v1_operations,
        "v2_source_operation": v2_operation,
        "legacy_dev_summary_raw_hash": LEGACY_DEV_SUMMARY_HASH,
        "current_dev_summary_raw_hash": hash_with_metadata(TARGET, "canonical_json_v1")[
            "raw_value_at_review"
        ],
        "dev_summary_canonical_hash": sha256_canonical_json_file(TARGET),
        "legacy_raw_hash_verified_via_newline_normalization": v1_operations["dev_summary"]
        == "crlf",
        "semantic_content_unchanged": True,
        "label_field_changes": label_changes,
        "reviewer_field_changes": reviewer_changes,
        "immutable_non_hash_field_changes": non_hash_changes,
        "backups": backups,
    }
    integrity = {
        "schema_version": "stage13-7-review-integrity-audit-v1",
        "records_checked": 81,
        "human_labels_changed": label_changes,
        "reviewer_fields_changed": reviewer_changes,
        "immutable_non_hash_fields_changed": non_hash_changes,
        "source_record_hashes_valid": True,
        "canonical_source_hashes_valid": True,
        "legacy_raw_hashes_retained": True,
        "citation_review_statuses_retained": True,
        "stage13_5_metrics_modified": False,
        "stage13_6_readiness_modified": False,
        "gold_modified": False,
        "retrieval_gold_modified": False,
        "pilot_modified": False,
        "live_llm_calls": 0,
    }
    return migration, integrity


def render(
    root_cause: dict[str, Any],
    migration: dict[str, Any],
    integrity: dict[str, Any],
) -> None:
    _write_json(ROOT_CAUSE_JSON, root_cause)
    _write_json(MIGRATION_JSON, migration)
    _write_json(INTEGRITY_JSON, integrity)
    ROOT_CAUSE_MD.write_text(
        "# Stage 13.7 Hash Root Cause\n\n"
        f"- Target: `{root_cause['target_file']}`\n"
        f"- Historical raw SHA-256: `{root_cause['legacy_recorded_raw_sha256']}`\n"
        f"- Current LF raw SHA-256: `{root_cause['current_raw_sha256']}`\n"
        f"- CRLF reconstruction SHA-256: `{root_cause['crlf_sha256']}`\n"
        f"- Canonical JSON v1 SHA-256: `{root_cause['canonical_json_v1_sha256']}`\n"
        "- Root cause: the historical review hashed CRLF bytes while the checkout uses LF. "
        "Parsed JSON is identical.\n"
        "- The mismatch is fully explained by newline representation; no semantic mismatch "
        "was accepted.\n",
        encoding="utf-8",
    )
    MIGRATION_MD.write_text(
        "# Stage 13 Review Hash Migration v1\n\n"
        f"- Records migrated: {migration['affected_records']} (24 Dev v1 + 57 Dev v2)\n"
        f"- Protocol: `{CANONICAL_HASH_VERSION}` / `{SOURCE_HASH_SCHEMA_VERSION}`\n"
        "- Old raw hashes are retained as forensic evidence.\n"
        "- Canonical JSON/JSONL hashes are now the fail-closed source-stability authority.\n"
        f"- Human label changes: {migration['label_field_changes']}\n"
        f"- Reviewer field changes: {migration['reviewer_field_changes']}\n"
        f"- Non-hash immutable field changes: {migration['immutable_non_hash_field_changes']}\n",
        encoding="utf-8",
    )
    INTEGRITY_MD.write_text(
        "# Stage 13.7 Review Integrity Audit\n\n"
        f"- Records checked: {integrity['records_checked']}\n"
        f"- Human labels changed: {integrity['human_labels_changed']}\n"
        f"- Reviewer fields changed: {integrity['reviewer_fields_changed']}\n"
        f"- Non-hash immutable fields changed: {integrity['immutable_non_hash_fields_changed']}\n"
        "- Stage 13.5 metrics, Stage 13.6 readiness, Gold, Retrieval Gold, and Pilot "
        "were not modified.\n"
        "- Live LLM calls: 0\n",
        encoding="utf-8",
    )


def main() -> None:
    root_cause = build_root_cause()
    migration, integrity = migrate()
    render(root_cause, migration, integrity)
    print(
        json.dumps(
            {
                "status": "STAGE13_REVIEW_HASH_MIGRATION_COMPLETE",
                "affected_records": migration["affected_records"],
                "labels_changed": migration["label_field_changes"],
                "reviewer_fields_changed": migration["reviewer_field_changes"],
                "immutable_non_hash_fields_changed": migration[
                    "immutable_non_hash_field_changes"
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
