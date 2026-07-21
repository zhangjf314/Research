# ruff: noqa: E501
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.evidence_qa_dev_lib_v1 import DATA, read_jsonl
from scripts.import_dev_v3_1_citation_review_v1 import (
    EXPECTED_HASH,
    validate_rows,
)


def test_review_package_and_import_are_complete() -> None:
    package = Path("artifacts/stage13-9-human-citation-review-results.zip")
    import hashlib

    assert hashlib.sha256(package.read_bytes()).hexdigest() == EXPECTED_HASH
    rows = read_jsonl(DATA / "evidence-qa-dev-v3-1-citation-audit-v1.jsonl")
    result = validate_rows(rows, rows)
    assert result["approved"] == 33
    assert result["immutable_changes"] == 0
    assert result["labels"] == {
        "fully_supported": 16,
        "partially_supported": 10,
        "related_but_insufficient": 5,
        "unsupported": 2,
    }
    assert list(DATA.glob("evidence-qa-dev-v3-1-citation-audit-v1.jsonl.pre-human-import.*.bak"))


def test_human_support_summary_matches_reviewed_records() -> None:
    summary = json.loads(
        (DATA / "evidence-qa-dev-v3-1-citation-audit-summary-v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["strict_support_rate"] == pytest.approx(16 / 33)
    assert summary["lenient_support_rate"] == pytest.approx(26 / 33)
    assert summary["strata"]["automated_signal"]["exact_gold"]["n"] == 9
    assert summary["strata"]["automated_signal"]["same_page_non_exact"]["n"] == 11
    assert summary["strata"]["automated_signal"]["semantic_support"]["n"] == 13
    assert summary["strata"]["evidence_source"]["original_selected"]["n"] == 27
    assert summary["strata"]["evidence_source"]["adjacent_completion"]["n"] == 6
    assert summary["exact_miss_but_supported"]["count"] == 18
    assert summary["exact_miss_but_supported"]["external_narrative_discrepancy"] is True
    assert summary["exact_hit_but_not_fully_supported"]["count"] == 3


def test_metric_v2_is_fixed_and_historical_gate_is_unchanged() -> None:
    protocol = json.loads(
        (DATA / "citation-recall-metric-v2.json").read_text(encoding="utf-8")
    )
    assert protocol["selected_primary_metric"] == "answerable_question_macro_exact_recall_v2"
    assert protocol["fixed_denominator"] == 9
    assert protocol["failure_handling"] == "zero"
    assert protocol["unanswerable_handling"] == "excluded"
    assert protocol["frozen_before_next_live_run"] is True
    assert protocol["backward_gate_effect"] is False
    historical = json.loads(
        (DATA / "evidence-qa-dev-v3-1.json").read_text(encoding="utf-8")
    )
    assert historical["metrics"]["all_manifest_conservative"]["citation_recall"] == 0.295
    assert historical["dev_v3_1_quality_candidate_gate"] is False


def test_gold_relation_ambiguity_blocks_dev_v3_2() -> None:
    relations = read_jsonl(DATA / "citation-recall-gold-relation-audit-v1.jsonl")
    assert len(relations) == 99
    assert sum(row["ambiguity"] for row in relations) == 99
    comparison = json.loads(
        (DATA / "citation-recall-v2-comparison.json").read_text(encoding="utf-8")
    )
    assert comparison["status"] == "CITATION_RECALL_V2_BLOCKED_BY_GOLD_RELATION_AMBIGUITY"
    experiments = {
        row["evaluation_version"]: row for row in comparison["experiments"]
    }
    assert experiments["stage11c_a"]["answerable_question_macro_exact_recall_v2"] == pytest.approx(0.05925925925925926)
    assert experiments["stage13_2_b"]["answerable_question_macro_exact_recall_v2"] == pytest.approx(0.05555555555555555)
    assert experiments["stage13_3_dev_v2"]["answerable_question_macro_exact_recall_v2"] == pytest.approx(0.26296296296296295)
    assert experiments["stage13_8_dev_v3_1"]["answerable_question_macro_exact_recall_v2"] == pytest.approx(0.21666666666666667)
    readiness = json.loads(
        (DATA / "stage13-9-phase-b-readiness-v1.json").read_text(encoding="utf-8")
    )
    assert readiness["ready_for_dev_v3_2"] is False
    assert readiness["dev_v3_2_authorized"] is False


def test_exact_and_human_support_remain_separate() -> None:
    matrix = json.loads(
        (DATA / "dev-v3-1-exact-vs-human-support-v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert matrix["exact_miss_but_lenient_supported_rate"] == pytest.approx(18 / 24)
    assert matrix["exact_hit_but_not_fully_supported_rate"] == pytest.approx(3 / 9)
    assert matrix["human_support_changes_exact_recall"] is False
    assert matrix["exact_recall_changes_human_support"] is False
