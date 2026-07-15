# ruff: noqa: E501
"""Human-only review utility for the frozen Dev v2 citation audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from paper_research.evaluation.canonical_hash import (
    SOURCE_HASH_SCHEMA_VERSION,
    hash_with_metadata,
    verify_legacy_raw_hash,
)
from paper_research.generation.citation_registry import CitationRegistry

try:
    from scripts.evidence_qa_dev_lib_v1 import DATA, canonical_hash, read_jsonl
except ModuleNotFoundError:
    from evidence_qa_dev_lib_v1 import DATA, canonical_hash, read_jsonl  # type: ignore[no-redef]

AUDIT = DATA / "evidence-qa-dev-v2-citation-audit-v1.jsonl"
EVIDENCE = DATA / "evidence-corpus-v1.jsonl"
RUN_ROOT = DATA / "evidence-qa-dev-v2/runs"
LABELS = {"fully_supported", "partially_supported", "related_but_insufficient", "unsupported", "gold_annotation_too_narrow", "ambiguous_claim", "malformed_evidence"}
HUMAN_FIELDS = {"human_review_status", "human_label", "reviewer", "reviewed_at", "review_notes"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list-pending", action="store_true")
    parser.add_argument("--sample-id")
    parser.add_argument("--label", choices=sorted(LABELS))
    parser.add_argument("--reviewer")
    parser.add_argument("--notes")
    parser.add_argument("--export", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(rows: list[dict[str, Any]]) -> None:
    if len(rows) != 57 or len({row["sample_id"] for row in rows}) != 57:
        raise RuntimeError("expected 57 unique audit samples")
    evidence_rows = read_jsonl(EVIDENCE)
    evidence = {(row["paper_id"], int(row["page"]), row["block_id"]): row for row in evidence_rows}
    source_metadata = hash_with_metadata(EVIDENCE, "canonical_jsonl_v1")
    registries: dict[str, CitationRegistry] = {}
    for row in rows:
        if row["human_review_status"] not in {"pending", "approved"}:
            raise RuntimeError(f"invalid human status: {row['sample_id']}")
        if row["human_review_status"] == "approved" and (row["human_label"] not in LABELS or not row["reviewer"] or not row["reviewed_at"] or not row["review_notes"]):
            raise RuntimeError(f"incomplete approved review: {row['sample_id']}")
        triple = (row["citation_triple"]["paper_id"], int(row["citation_triple"]["page"]), row["citation_triple"]["block_id"])
        unit = evidence.get(triple)
        if row.get("source_hash_schema_version") != SOURCE_HASH_SCHEMA_VERSION:
            raise RuntimeError(f"unsupported source hash schema: {row['sample_id']}")
        if row.get("source_hash_mode") != "canonical_jsonl_v1":
            raise RuntimeError(f"unsupported source hash mode: {row['sample_id']}")
        if row.get("source_canonical_sha256") != source_metadata["value"]:
            raise RuntimeError(f"canonical source changed: {row['sample_id']}")
        if row.get("source_raw_sha256_at_review") != row.get("source_hash"):
            raise RuntimeError(f"legacy source hash metadata changed: {row['sample_id']}")
        operation = verify_legacy_raw_hash(EVIDENCE, row["source_hash"])
        migrated = row.get("source_legacy_raw_hash_verified_via_newline_normalization")
        if bool(migrated) != (operation != "raw"):
            raise RuntimeError(f"legacy source migration evidence invalid: {row['sample_id']}")
        if unit is None or row["source_record_hash"] != canonical_hash(unit):
            raise RuntimeError(f"source changed: {row['sample_id']}")
        immutable = {key: value for key, value in row.items() if key not in HUMAN_FIELDS | {"immutable_record_hash"}}
        if row["immutable_record_hash"] != canonical_hash(immutable):
            raise RuntimeError(f"immutable fields changed: {row['sample_id']}")
        registry = registries.setdefault(row["run_id"], CitationRegistry.model_validate_json((RUN_ROOT / row["run_id"] / "citation-registry.json").read_text(encoding="utf-8")))
        entry = next((item for item in registry.entries if item.citation_id == row["citation_id"]), None)
        if entry is None or entry.triple != triple or registry.registry_hash != row["registry_hash"]:
            raise RuntimeError(f"registry mismatch: {row['sample_id']}")


def write(rows: list[dict[str, Any]]) -> None:
    AUDIT.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = read_jsonl(AUDIT)
    validate(rows)
    if args.list_pending:
        for row in rows:
            if row["human_review_status"] == "pending":
                print(f"{row['sample_id']}\t{row['question_id']}\t{row['claim_text']}")
    if args.export:
        if args.export.exists() and not args.overwrite:
            raise RuntimeError("export exists; pass --overwrite to replace")
        args.export.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(AUDIT, args.export)
    if args.sample_id:
        if not args.label or not args.reviewer or not args.notes:
            raise RuntimeError("--label, --reviewer, and --notes are mandatory")
        match = next((row for row in rows if row["sample_id"] == args.sample_id), None)
        if match is None:
            raise RuntimeError("unknown sample_id")
        if match["human_review_status"] == "approved" and not args.overwrite:
            raise RuntimeError("sample is already approved; pass --overwrite")
        backup = AUDIT.with_name(f"{AUDIT.name}.pre-review.{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}.bak")
        shutil.copy2(AUDIT, backup)
        match.update({"human_review_status": "approved", "human_label": args.label, "reviewer": args.reviewer.strip(), "reviewed_at": datetime.now(UTC).isoformat(), "review_notes": args.notes.strip()})
        if not match["reviewer"] or not match["review_notes"]:
            raise RuntimeError("reviewer and notes cannot be blank")
        validate(rows)
        write(rows)
        print(json.dumps({"updated": args.sample_id, "backup": str(backup)}))


if __name__ == "__main__":
    main()
