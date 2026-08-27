"""Mechanical integrity records for the immutable RAGQ3 execution contract."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

FREEZE_SCHEMA_VERSION = "ragq3-pre-result-freeze-v4"
SEMANTIC_PATHS = (
    "artifacts/rag-quality-v3/a0/baseline/p0-runtime-contract-v1.json",
    "artifacts/rag-quality-v3/a0/preregistration/metric-and-gate-v1.json",
    "artifacts/rag-quality-v3/a1r2/preregistration/gate-applicability-v1.json",
    "artifacts/rag-quality-v3/a1r2/preregistration/parent-child-contract-v1.json",
    "artifacts/rag-quality-v3/a1r2/preregistration/sentence-boundary-contract-v1.json",
    "artifacts/rag-quality-v3/a1r4/attribution/gold-attribution-contract-v1.json",
    "artifacts/rag-quality-v3/a1r4/attribution/metric-matching-contract-v1.json",
    "artifacts/rag-quality-v3/a1r4/identity/identity-contract-v2.json",
    "artifacts/rag-quality-v3/a1r4/preregistration/q3x-candidate-matrix-v1.json",
    "artifacts/rag-quality-v3/a1r4/spec/execution-semantic-graph-v1.json",
    "src/paper_research/evaluation/ragq3.py",
    "src/paper_research/evaluation/ragq3_attribution.py",
    "src/paper_research/evaluation/ragq3_execution.py",
    "src/paper_research/evaluation/ragq3_identity.py",
)


def _git(*args: str, root: Path) -> bytes:
    return subprocess.check_output(["git", *args], cwd=root)


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _payload_digest(manifest: dict[str, Any]) -> str:
    payload = {key: value for key, value in manifest.items() if key != "manifest_payload_sha256"}
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256(encoded)


def build_freeze_manifest(*, root: Path, source_commit: str) -> dict[str, Any]:
    """Construct a v4 record from bytes, rather than hand-copied digests."""
    files = []
    for relative in SEMANTIC_PATHS:
        path = root / relative
        current = path.read_bytes()
        blob = _git("rev-parse", f"{source_commit}:{relative}", root=root).decode("ascii").strip()
        blob_content = _git("show", f"{source_commit}:{relative}", root=root)
        if current != blob_content:
            raise ValueError(f"SEMANTIC_FILE_DRIFT_DETECTED {relative}")
        files.append(
            {
                "path": relative,
                "source_commit": source_commit,
                "git_blob": blob,
                "sha256": _sha256(current),
            }
        )
    manifest: dict[str, Any] = {
        "schema_version": FREEZE_SCHEMA_VERSION,
        "freeze_type": "FINAL_VALID_PRE_RESULT_FREEZE",
        "supersedes": {
            "path": "artifacts/rag-quality-v3/a1r4/preregistration/pre-result-freeze-v3.json",
            "reason": "FREEZE_MANIFEST_BOOKKEEPING_DEFECT_ONLY",
            "semantic_change": "none",
        },
        "candidate_namespace": "Q3X",
        "files": files,
        "runtime_invariants": {
            "provider_calls_before_freeze": 0,
            "real_index_builds_before_freeze": 0,
            "real_results_before_freeze": 0,
            "new_blind_papers_before_freeze": 0,
            "new_blind_questions_before_freeze": 0,
            "full_qa": "NOT_RUN",
            "production_default_change": "no",
        },
    }
    manifest["manifest_payload_sha256"] = _payload_digest(manifest)
    return manifest


def verify_ragq3_freeze_manifest(manifest: dict[str, Any], *, root: Path) -> list[str]:
    """Return no errors only when each frozen path and the manifest payload self-check."""
    errors: list[str] = []
    if manifest.get("schema_version") != FREEZE_SCHEMA_VERSION:
        errors.append("schema_version")
    if manifest.get("manifest_payload_sha256") != _payload_digest(manifest):
        errors.append("manifest_payload_sha256")
    records = manifest.get("files")
    if not isinstance(records, list):
        return [*errors, "files"]
    paths = [record.get("path") for record in records if isinstance(record, dict)]
    if tuple(paths) != SEMANTIC_PATHS:
        errors.append("file_set_or_order")
    for record in records:
        if not isinstance(record, dict):
            errors.append("invalid_file_record")
            continue
        relative = record.get("path")
        source_commit = record.get("source_commit")
        if not isinstance(relative, str) or not isinstance(source_commit, str):
            errors.append("record_fields")
            continue
        path = root / relative
        if not path.is_file():
            errors.append(f"missing:{relative}")
            continue
        current = path.read_bytes()
        if record.get("sha256") != _sha256(current):
            errors.append(f"sha256:{relative}")
        try:
            blob = _git("rev-parse", f"{source_commit}:{relative}", root=root)
            blob = blob.decode("ascii").strip()
            blob_content = _git("show", f"{source_commit}:{relative}", root=root)
        except subprocess.CalledProcessError:
            errors.append(f"git:{relative}")
            continue
        if record.get("git_blob") != blob:
            errors.append(f"blob:{relative}")
        if current != blob_content:
            errors.append(f"content:{relative}")
    return errors
