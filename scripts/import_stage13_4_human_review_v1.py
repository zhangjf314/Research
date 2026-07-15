# ruff: noqa: E501
"""Fail-closed import of externally reviewed Stage 13.4 citation and matcher records."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from scripts.evidence_qa_dev_lib_v1 import DATA, read_jsonl
    from scripts.review_evidence_qa_dev_v2_citations_v1 import AUDIT, HUMAN_FIELDS, LABELS, validate
except ModuleNotFoundError:
    from evidence_qa_dev_lib_v1 import DATA, read_jsonl  # type: ignore[no-redef]
    from review_evidence_qa_dev_v2_citations_v1 import (  # type: ignore[no-redef]
        AUDIT,
        HUMAN_FIELDS,
        LABELS,
        validate,
    )

COVERAGE = DATA / "dev-v2-claim-coverage-audit-v1.jsonl"
IMPORT_ROOT = Path("artifacts/imports/stage13-4-human-review-results")
CANDIDATES = {
    "cl-q001-c41ea3191cab92907d83": ("merged_claim_match", 1, None),
    "cl-q002-3cab1a55474dcd47a64a": ("valid_match", 1, None),
    "cl-q004-ba317f113c67f72f7260": ("partial_match", 0, 0.5),
    "cl-q015-281a83daa0567f6ae7ac": ("false_positive", 0, None),
}
MATCHER_HUMAN_FIELDS = {"matcher_human_decision", "coverage_credit", "diagnostic_partial_credit", "human_review_status", "reviewer", "reviewed_at", "review_notes", "coverage_failure_stage_before_review", "coverage_failure_stage"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def load_source(source: Path) -> tuple[Path, tempfile.TemporaryDirectory | None]:
    if source.is_dir():
        return source, None
    if not source.is_file() or not zipfile.is_zipfile(source):
        raise RuntimeError("input must be a review directory or ZIP")
    temporary = tempfile.TemporaryDirectory(prefix="stage13-4-review-")
    with zipfile.ZipFile(source) as archive:
        for name in archive.namelist():
            target = Path(name)
            if target.is_absolute() or ".." in target.parts:
                raise RuntimeError("unsafe ZIP member")
        archive.extractall(temporary.name)
    return Path(temporary.name), temporary


def immutable_equal(base: dict[str, Any], reviewed: dict[str, Any], ignored: set[str]) -> bool:
    return {key: value for key, value in base.items() if key not in ignored} == {key: value for key, value in reviewed.items() if key not in ignored}


def validate_citations(base: list[dict[str, Any]], reviewed: list[dict[str, Any]]) -> None:
    if len(reviewed) != 57 or len({row["sample_id"] for row in reviewed}) != 57:
        raise RuntimeError("citation review must contain 57 unique sample IDs")
    by_id = {row["sample_id"]: row for row in base}
    if set(by_id) != {row["sample_id"] for row in reviewed}:
        raise RuntimeError("citation review sample set changed")
    for row in reviewed:
        original = by_id[row["sample_id"]]
        if not immutable_equal(original, row, HUMAN_FIELDS):
            raise RuntimeError(f"citation immutable fields changed: {row['sample_id']}")
        if row.get("human_review_status") != "approved" or row.get("human_label") not in LABELS or not row.get("reviewer") or not row.get("reviewed_at") or not row.get("review_notes"):
            raise RuntimeError(f"invalid citation human review: {row['sample_id']}")
    validate(reviewed)


def validate_matchers(base: list[dict[str, Any]], reviewed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(reviewed) != len(base) or len({row["required_claim_id"] for row in reviewed}) != len(base):
        raise RuntimeError("matcher review claim set is not unique and complete")
    by_id = {row["required_claim_id"]: row for row in base}
    reviewed_by_id = {row["required_claim_id"]: row for row in reviewed}
    if set(by_id) != set(reviewed_by_id):
        raise RuntimeError("matcher review changed required claim set")
    imported = json.loads(json.dumps(base))
    output = {row["required_claim_id"]: row for row in imported}
    for claim_id, original in by_id.items():
        candidate = reviewed_by_id[claim_id]
        if claim_id not in CANDIDATES:
            if candidate != original:
                raise RuntimeError(f"non-candidate matcher record changed: {claim_id}")
            continue
        if not immutable_equal(original, candidate, MATCHER_HUMAN_FIELDS):
            raise RuntimeError(f"matcher immutable fields changed: {claim_id}")
        decision, expected_credit, expected_partial = CANDIDATES[claim_id]
        if candidate.get("matcher_human_decision") != decision or candidate.get("coverage_credit") != expected_credit or candidate.get("diagnostic_partial_credit") != expected_partial:
            raise RuntimeError(f"unexpected matcher decision or credit: {claim_id}")
        if candidate.get("human_review_status") != "approved" or not candidate.get("reviewer") or not candidate.get("reviewed_at") or not candidate.get("review_notes"):
            raise RuntimeError(f"incomplete matcher review: {claim_id}")
        target = output[claim_id]
        target.update({"matcher_human_decision": decision, "historical_formal_coverage_credit": original["coverage_credit"], "formal_coverage_credit": expected_credit, "diagnostic_partial_credit": expected_partial or 0.0, "human_review_status": "approved", "reviewer": candidate["reviewer"], "reviewed_at": candidate["reviewed_at"], "review_notes": candidate["review_notes"], "coverage_failure_stage_before_review": original.get("coverage_failure_stage"), "coverage_failure_stage_after_review": "covered_after_human_matcher_adjudication" if expected_credit else decision})
    return imported


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main() -> int:
    args = parse_args()
    try:
        source, temporary = load_source(args.input)
        citation_source = next(source.rglob("evidence-qa-dev-v2-citation-audit-v1-reviewed.jsonl"))
        matcher_source = next(source.rglob("dev-v2-claim-coverage-audit-v1-reviewed.jsonl"))
        citation_base, matcher_base = read_jsonl(AUDIT), read_jsonl(COVERAGE)
        citation_reviewed, matcher_reviewed = read_jsonl(citation_source), read_jsonl(matcher_source)
        validate_citations(citation_base, citation_reviewed)
        matcher_imported = validate_matchers(matcher_base, matcher_reviewed)
        if args.validate_only:
            print(json.dumps({"status": "HUMAN_REVIEW_VALID", "citations": 57, "matcher_candidates": 4}))
            if temporary:
                temporary.cleanup()
            return 0
        if any(row["human_review_status"] == "approved" for row in citation_base) or any(row.get("matcher_human_decision") for row in matcher_base):
            raise RuntimeError("formal review files already contain imported decisions")
        IMPORT_ROOT.mkdir(parents=True, exist_ok=True)
        for path in source.iterdir():
            if path.is_file():
                shutil.copy2(path, IMPORT_ROOT / path.name)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        citation_backup = AUDIT.with_name(f"{AUDIT.name}.pre-human-import.{stamp}.bak")
        matcher_backup = COVERAGE.with_name(f"{COVERAGE.name}.pre-human-import.{stamp}.bak")
        shutil.copy2(AUDIT, citation_backup)
        shutil.copy2(COVERAGE, matcher_backup)
        write_jsonl(AUDIT, citation_reviewed)
        write_jsonl(COVERAGE, matcher_imported)
        validate(read_jsonl(AUDIT))
        final_matchers = {row["required_claim_id"]: row for row in read_jsonl(COVERAGE)}
        for claim_id, (decision, credit, partial) in CANDIDATES.items():
            final = final_matchers[claim_id]
            if final.get("matcher_human_decision") != decision or final.get("formal_coverage_credit") != credit or final.get("diagnostic_partial_credit") != (partial or 0.0):
                raise RuntimeError(f"post-import matcher validation failed: {claim_id}")
        if temporary:
            temporary.cleanup()
        print(json.dumps({"status": "HUMAN_REVIEW_IMPORT_COMPLETE", "citations": 57, "matcher_candidates": 4, "citation_backup": str(citation_backup), "matcher_backup": str(matcher_backup), "immutable_changes": 0}))
        return 0
    except (RuntimeError, StopIteration, KeyError, ValueError) as exc:
        print(json.dumps({"status": "HUMAN_REVIEW_IMPORT_FAILED", "error_type": type(exc).__name__, "error": str(exc)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
