# ruff: noqa: E501
from __future__ import annotations

import json
import zipfile

import pytest

from scripts.evidence_qa_dev_lib_v1 import DATA, read_jsonl
from scripts.prepare_stage13_9_citation_recall_audit_v1 import (
    LABELS,
    PACK,
    RECALL_JSONL,
    SUMMARY,
    aggregate,
    validate_citation_audit,
)


def test_stage13_9_citation_audit_is_reviewed_and_valid() -> None:
    rows = read_jsonl(DATA / "evidence-qa-dev-v3-1-citation-audit-v1.jsonl")
    validation = validate_citation_audit(rows)
    assert validation == {
        "records": 33,
        "unique_sample_ids": 33,
        "pending": 0,
        "approved": 33,
        "source_hash_valid": True,
        "source_record_hash_valid": True,
        "immutable_hash_valid": True,
        "registry_hash_valid": True,
        "citation_triples_valid": True,
    }
    assert len(LABELS) == 7


def test_recall_protocols_reconstruct_but_are_incomparable() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    rows = read_jsonl(RECALL_JSONL)
    metrics = aggregate(summary, rows)
    assert len(rows) == 27
    assert metrics["dev_v3_1_formal"]["value"] == pytest.approx(0.295)
    assert metrics["dev_v2_formal"]["value"] == pytest.approx(0.29583333333333334)
    assert metrics["macro_question_recall_answerable_only_diagnostic"]["value"] == pytest.approx(0.21666666666666667)
    assert metrics["macro_required_claim_recall_diagnostic"]["value"] == pytest.approx(0.13518518518518519)
    assert metrics["micro_question_gold_block_recall_diagnostic"]["value"] == pytest.approx(5 / 33)
    assert metrics["decision"] == "CITATION_RECALL_METRIC_INCOMPARABLE"
    assert metrics["metric_protocols_comparable"] is False
    assert metrics["formal_gate_modified"] is False
    assert metrics["calculation_bug_found"] is False


def test_review_pack_has_exact_safe_member_set() -> None:
    expected = {
        "evidence-qa-dev-v3-1-citation-audit-v1.jsonl",
        "evidence-corpus-v1.jsonl",
        "claim-units-v1.jsonl",
        "gold-set-v1.jsonl",
        "retrieval-gold-v2.jsonl",
        "evidence-qa-dev-v3-1.json",
        "dev-v3-1-citation-recall-audit-v1.jsonl",
        "evidence-qa-dev-v3-1-citation-review-guide-v1.md",
    }
    with zipfile.ZipFile(PACK) as archive:
        assert set(archive.namelist()) == expected
        assert not any(
            token in name.lower()
            for name in archive.namelist()
            for token in (".env", ".sqlite", "raw-provider", "runs/")
        )


def test_stage13_8_formal_gate_remains_frozen() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    metrics = summary["metrics"]["all_manifest_conservative"]
    assert metrics["citation_recall"] == 0.295
    assert summary["dev_v3_1_engineering_gate"] is True
    assert summary["dev_v3_1_quality_candidate_gate"] is False
    assert summary["ready_for_full_qa"] is False
