from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from scripts.import_evidence_qa_dev_citation_audit_v1 import (
    build_summary,
    validate_external,
)
from scripts.review_evidence_qa_dev_citations_v1 import AUDIT, read_jsonl

ROOT = Path(__file__).resolve().parents[1]
REVIEWED = AUDIT


def rows():
    return read_jsonl(AUDIT), read_jsonl(REVIEWED)


def test_external_review_is_complete_and_source_stable() -> None:
    current, reviewed = rows()
    validation = validate_external(current, reviewed)
    assert validation == {
        "records": 24,
        "unique_sample_ids": 24,
        "approved": 24,
        "source_hashes_valid": True,
        "source_record_hashes_valid": True,
        "citation_triples_valid": True,
        "immutable_changes": 0,
    }


@pytest.mark.parametrize("field", ["reviewer", "reviewed_at", "review_notes"])
def test_missing_manual_metadata_fails(field: str) -> None:
    current, reviewed = rows()
    changed = deepcopy(reviewed)
    changed[0][field] = None
    with pytest.raises(RuntimeError, match=field):
        validate_external(current, changed)


def test_duplicate_sample_and_changed_hash_fail() -> None:
    current, reviewed = rows()
    duplicate = deepcopy(reviewed)
    duplicate[1]["sample_id"] = duplicate[0]["sample_id"]
    with pytest.raises(RuntimeError, match="unique"):
        validate_external(current, duplicate)
    changed = deepcopy(reviewed)
    changed[0]["source_record_hash"] = "0" * 64
    with pytest.raises(RuntimeError, match="source_record_hash"):
        validate_external(current, changed)


def test_summary_rates_and_no_full_extrapolation() -> None:
    current, reviewed = rows()
    validation = validate_external(current, reviewed)
    summary = build_summary(reviewed, validation, REVIEWED)
    assert summary["overall"]["total_reviewed"] == 24
    assert summary["overall"]["strict_support_rate"] == 0.625
    assert summary["overall"]["lenient_support_rate"] == 0.708333
    assert summary["human_citation_audit_complete"] is True
    assert summary["dev_v2_run"] is False
    assert "must not be extrapolated" in summary["representativeness_warning"]
    assert sum(item["total_reviewed"] for item in summary["strata"].values()) == 24
    assert summary["strata"]["semantic_support"]["total_reviewed"] == 0


def test_reviewed_labels_match_external_file_without_adjustment() -> None:
    _, reviewed = rows()
    counts = {}
    for row in reviewed:
        counts[row["human_label"]] = counts.get(row["human_label"], 0) + 1
    assert counts == {
        "fully_supported": 15,
        "partially_supported": 2,
        "unsupported": 7,
    }
