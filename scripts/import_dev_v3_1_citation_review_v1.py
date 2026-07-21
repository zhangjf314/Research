# ruff: noqa: E501
"""Atomically validate and import the Stage 13.9 reviewed citation audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from paper_research.evaluation.canonical_hash import hash_with_metadata

try:
    from scripts.evidence_qa_dev_lib_v1 import DATA, canonical_hash, read_jsonl
    from scripts.evidence_qa_dev_v3_1_lib import RUN_ROOT
except ModuleNotFoundError:
    from evidence_qa_dev_lib_v1 import DATA, canonical_hash, read_jsonl  # type: ignore[no-redef]
    from evidence_qa_dev_v3_1_lib import RUN_ROOT  # type: ignore[no-redef]

ROOT = DATA.parents[1]
AUDIT = DATA / "evidence-qa-dev-v3-1-citation-audit-v1.jsonl"
IMPORT_DIR = ROOT / "artifacts/imports/stage13-9-human-citation-review-results"
EXPECTED_HASH = "e2040c35ec7b09a175130ea1e157da3f9a8c53f1c219816361264de68d8eab82"
REVIEWED_NAME = "evidence-qa-dev-v3-1-citation-audit-v1-reviewed.jsonl"
EXPECTED_MEMBERS = {
    REVIEWED_NAME,
    "stage13-9-human-citation-review-summary.json",
    "stage13-9-human-citation-review-summary.md",
}
LABELS = {
    "fully_supported",
    "partially_supported",
    "related_but_insufficient",
    "unsupported",
    "gold_annotation_too_narrow",
    "ambiguous_claim",
    "malformed_evidence",
}
HUMAN_FIELDS = {
    "human_review_status",
    "human_label",
    "reviewer",
    "reviewed_at",
    "review_notes",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_rows(
    pending: list[dict[str, Any]], reviewed: list[dict[str, Any]]
) -> dict[str, Any]:
    if len(pending) != 33 or len(reviewed) != 33:
        raise RuntimeError("expected exactly 33 pending and reviewed records")
    if len({row["sample_id"] for row in reviewed}) != 33:
        raise RuntimeError("duplicate reviewed sample_id")
    pending_by_id = {row["sample_id"]: row for row in pending}
    if set(pending_by_id) != {row["sample_id"] for row in reviewed}:
        raise RuntimeError("partial or foreign reviewed sample set")
    evidence_path = DATA / "evidence-corpus-v1.jsonl"
    evidence = {
        (row["paper_id"], int(row["page"]), row["block_id"]): row
        for row in read_jsonl(evidence_path)
    }
    source = hash_with_metadata(evidence_path, "canonical_jsonl_v1")
    registries: dict[str, dict[str, Any]] = {}
    immutable_changes = 0
    for row in reviewed:
        sample_id = row["sample_id"]
        original = pending_by_id[sample_id]
        if row.get("human_review_status") != "approved":
            raise RuntimeError(f"review not approved: {sample_id}")
        if row.get("human_label") not in LABELS:
            raise RuntimeError(f"invalid human label: {sample_id}")
        if not row.get("reviewer") or not row.get("reviewed_at") or not row.get(
            "review_notes"
        ):
            raise RuntimeError(f"incomplete human metadata: {sample_id}")
        original_immutable = {
            key: value for key, value in original.items() if key not in HUMAN_FIELDS
        }
        reviewed_immutable = {
            key: value for key, value in row.items() if key not in HUMAN_FIELDS
        }
        if reviewed_immutable != original_immutable:
            immutable_changes += 1
            raise RuntimeError(f"immutable fields changed: {sample_id}")
        immutable_for_hash = {
            key: value
            for key, value in row.items()
            if key not in HUMAN_FIELDS | {"immutable_record_hash"}
        }
        if row["immutable_record_hash"] != canonical_hash(immutable_for_hash):
            raise RuntimeError(f"immutable record hash invalid: {sample_id}")
        triple = row["citation_triple"]
        triple_key = (
            triple["paper_id"],
            int(triple["page"]),
            triple["block_id"],
        )
        unit = evidence.get(triple_key)
        if unit is None or row["source_record_hash"] != canonical_hash(unit):
            raise RuntimeError(f"source record/triple invalid: {sample_id}")
        if row["source_canonical_sha256"] != source["value"]:
            raise RuntimeError(f"canonical source hash invalid: {sample_id}")
        registry = registries.setdefault(
            row["run_id"],
            json.loads(
                (RUN_ROOT / row["run_id"] / "citation-registry.json").read_text(
                    encoding="utf-8"
                )
            ),
        )
        entry = next(
            (
                item
                for item in registry["entries"]
                if item["citation_id"] == row["citation_id"]
            ),
            None,
        )
        if (
            entry is None
            or registry["registry_hash"] != row["registry_hash"]
            or any(
                entry[field] != triple[field]
                for field in ("paper_id", "page", "block_id")
            )
        ):
            raise RuntimeError(f"registry mismatch: {sample_id}")
    return {
        "records": 33,
        "approved": 33,
        "immutable_changes": immutable_changes,
        "labels": dict(sorted(Counter(row["human_label"] for row in reviewed).items())),
        "source_hash_valid": True,
        "source_record_hash_valid": True,
        "registry_hash_valid": True,
        "citation_triples_valid": True,
    }


def main() -> None:
    args = parse_args()
    package = args.package.resolve()
    if sha256(package) != EXPECTED_HASH:
        print("DEV_V3_1_CITATION_REVIEW_PACKAGE_HASH_MISMATCH")
        raise SystemExit(2)
    with zipfile.ZipFile(package) as archive:
        names = archive.namelist()
        if set(names) != EXPECTED_MEMBERS or len(names) != len(set(names)):
            raise RuntimeError("review package member set invalid")
        if any(
            ".." in Path(name).parts
            or name.lower().endswith((".env", ".sqlite", ".db"))
            for name in names
        ):
            raise RuntimeError("unsafe review package member")
        payloads = {name: archive.read(name) for name in names}
    IMPORT_DIR.mkdir(parents=True, exist_ok=True)
    for name, body in payloads.items():
        (IMPORT_DIR / name).write_bytes(body)
    reviewed = [
        json.loads(line)
        for line in payloads[REVIEWED_NAME].decode("utf-8").splitlines()
        if line.strip()
    ]
    pending = read_jsonl(AUDIT)
    if all(row.get("human_review_status") == "approved" for row in pending):
        validation = validate_rows(pending, reviewed)
        if pending != reviewed:
            raise RuntimeError("different approved review already imported")
        print(json.dumps({"status": "already_imported", **validation}))
        return
    validation = validate_rows(pending, reviewed)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    backup = AUDIT.with_name(f"{AUDIT.name}.pre-human-import.{stamp}.bak")
    shutil.copy2(AUDIT, backup)
    temporary = AUDIT.with_suffix(".jsonl.tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in reviewed),
        encoding="utf-8",
    )
    temporary.replace(AUDIT)
    post = read_jsonl(AUDIT)
    post_validation = validate_rows(post, reviewed)
    print(
        json.dumps(
            {
                "status": "imported",
                "package_hash": EXPECTED_HASH,
                "backup": backup.name,
                **post_validation,
            }
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"DEV_V3_1_CITATION_REVIEW_IMPORT_FAILED: {exc}")
        raise
