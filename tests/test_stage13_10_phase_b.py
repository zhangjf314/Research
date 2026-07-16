from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.evidence_qa_dev_lib_v1 import DATA, read_jsonl
from scripts.import_stage13_10_claim_gold_review_v1 import (
    EXPECTED_PACKAGE_HASH,
    validate_external_summary,
    validate_review,
)

ROOT = DATA.parents[1]
PACKAGE = ROOT / "artifacts/stage13-10-human-claim-gold-review-results.zip"
IMPORT_ROOT = ROOT / "artifacts/imports/stage13-10-human-claim-gold-review-results"


def pending_backup() -> Path:
    backups = sorted(
        DATA.glob("claim-evidence-gold-dev-v1.jsonl.pre-human-import.*.bak")
    )
    assert len(backups) == 1
    return backups[0]


def test_package_hash_and_review_import_are_complete() -> None:
    assert hashlib.sha256(PACKAGE.read_bytes()).hexdigest() == EXPECTED_PACKAGE_HASH
    pending = read_jsonl(pending_backup())
    reviewed = read_jsonl(DATA / "claim-evidence-gold-dev-v1.jsonl")
    result = validate_review(pending, reviewed)
    assert result["required_claims"] == 27
    assert result["approved_claims"] == 27
    assert result["candidate_relations"] == 313
    assert result["immutable_changes"] == 0
    assert result["relation_triples_valid"] is True
    assert result["source_hashes_valid"] is True


def test_review_counts_match_external_summary() -> None:
    pending = read_jsonl(pending_backup())
    reviewed = read_jsonl(DATA / "claim-evidence-gold-dev-v1.jsonl")
    result = validate_review(pending, reviewed)
    assert result["core_gold"] == 25
    assert result["supporting_gold"] == 0
    assert result["equivalent_valid_evidence"] == 8
    assert result["partially_relevant"] == 115
    assert result["insufficient"] == 165
    assert result["multi_relation_minimum_sets"] == 5
    assert result["no_valid_gold_evidence"] == 0
    external = json.loads(
        (IMPORT_ROOT / "stage13-10-human-claim-gold-review-summary.json").read_text(
            encoding="utf-8"
        )
    )
    validate_external_summary(external, result)
    changed = copy.deepcopy(external)
    changed["approved_core_relations"] = 24
    with pytest.raises(RuntimeError, match="SUMMARY_MISMATCH"):
        validate_external_summary(changed, result)


def test_partial_duplicate_and_immutable_imports_fail() -> None:
    pending = read_jsonl(pending_backup())
    reviewed = read_jsonl(DATA / "claim-evidence-gold-dev-v1.jsonl")
    with pytest.raises(RuntimeError, match="expected 27"):
        validate_review(pending, reviewed[:-1])
    duplicate = copy.deepcopy(reviewed)
    duplicate[-1]["required_claim_id"] = duplicate[0]["required_claim_id"]
    with pytest.raises(RuntimeError, match="duplicate required claim"):
        validate_review(pending, duplicate)
    changed = copy.deepcopy(reviewed)
    changed[0]["required_claim_text"] += " mutated"
    with pytest.raises(RuntimeError, match="immutable hash invalid"):
        validate_review(pending, changed)


def test_relation_conflicts_and_unknown_relations_fail() -> None:
    pending = read_jsonl(pending_backup())
    reviewed = read_jsonl(DATA / "claim-evidence-gold-dev-v1.jsonl")
    changed = copy.deepcopy(reviewed)
    row = next(item for item in changed if item["approved_core_relations"])
    row["rejected_relations"].append(row["approved_core_relations"][0])
    with pytest.raises(RuntimeError, match="multiple outcomes"):
        validate_review(pending, changed)
    changed = copy.deepcopy(reviewed)
    changed[0]["equivalent_non_gold_relations"].append("unknown-relation")
    with pytest.raises(RuntimeError, match="unknown adjudicated relation"):
        validate_review(pending, changed)
    changed = copy.deepcopy(reviewed)
    changed[0]["no_valid_gold_evidence"] = True
    with pytest.raises(RuntimeError, match="no-valid-evidence conflict"):
        validate_review(pending, changed)


def test_freeze_is_complete_and_stable() -> None:
    freeze = json.loads(
        (DATA / "claim-evidence-gold-dev-v1-freeze.json").read_text(encoding="utf-8")
    )
    assert freeze["gold_version"] == "claim-evidence-gold-dev-v1"
    assert freeze["claim_gold_schema_version"] == "claim-evidence-gold-dev-schema-v1"
    assert freeze["reviewed_record_count"] == 27
    assert freeze["relation_count"] == 313
    assert freeze["core_relation_count"] == 25
    assert freeze["equivalent_relation_count"] == 8
    assert freeze["rejected_relation_count"] == 280
    assert freeze["frozen_before_next_live_run"] is True
    assert freeze["automatic_overwrite_allowed"] is False
    assert freeze["reviewed_file_hash"]["value"] == (
        "3cee289380c4b2ba861079d5f8470719a0d880f98812a5b55f28fb65693d37a6"
    )


def test_claim_gold_recalculation_metrics_are_fixed() -> None:
    comparison = json.loads(
        (DATA / "claim-gold-citation-comparison-v1.json").read_text(encoding="utf-8")
    )
    experiments = {
        row["evaluation_version"]: row for row in comparison["experiments"]
    }
    dev2 = experiments["stage13_3_dev_v2"]
    dev31 = experiments["stage13_8_dev_v3_1"]
    assert dev2["answerable_question_macro_exact_relation_recall"] == pytest.approx(
        0.12777777777777777
    )
    assert dev31["answerable_question_macro_exact_relation_recall"] == pytest.approx(
        0.14629629629629629
    )
    assert dev2["required_claim_macro_exact_relation_recall"] == pytest.approx(
        0.12962962962962962
    )
    assert dev31["required_claim_macro_exact_relation_recall"] == pytest.approx(
        0.18518518518518517
    )
    assert dev2["micro_core_relation_recall"] == pytest.approx(0.16)
    assert dev31["micro_core_relation_recall"] == pytest.approx(0.20)
    assert dev2["claim_core_set_completion"] == pytest.approx(3 / 27)
    assert dev31["claim_core_set_completion"] == pytest.approx(5 / 27)
    assert dev2["claim_any_valid_evidence_recall"] == pytest.approx(4 / 27)
    assert dev31["claim_any_valid_evidence_recall"] == pytest.approx(8 / 27)
    assert dev2["failed_questions"] == ["q050"]
    assert dev31["failed_questions"] == []


def test_core_equivalent_and_failure_protocols() -> None:
    comparison = json.loads(
        (DATA / "claim-gold-citation-comparison-v1.json").read_text(encoding="utf-8")
    )
    experiments = {
        row["evaluation_version"]: row for row in comparison["experiments"]
    }
    dev31 = experiments["stage13_8_dev_v3_1"]
    assert dev31["equivalent_valid_evidence_hit_rate"] == pytest.approx(3 / 8)
    assert dev31["supporting_only_hit_rate"] == 0
    assert dev31["answerable_denominator"] == 9
    assert dev31["required_claim_denominator"] == 27
    assert dev31["core_relation_denominator"] == 25
    assert comparison["dev_v2_vs_dev_v3_1"]["outcomes"] == {
        "regressed": 1,
        "improved": 1,
        "unchanged": 7,
    }
    assert comparison["dev_v2_vs_dev_v3_1"]["single_question_driven"] is False


def test_taxonomy_covers_generic_failure_stages() -> None:
    summary = json.loads(
        (DATA / "dev-v3-1-citation-failure-taxonomy-v2.json").read_text(
            encoding="utf-8"
        )
    )
    counts = summary["failure_type_counts"]
    assert counts["core_gold_not_retrieved"] > 0
    assert counts["core_gold_selected_not_cited"] > 0
    assert counts["equivalent_valid_evidence_cited"] > 0
    assert counts["numeric_evidence_missing"] > 0
    assert counts["comparison_side_missing"] > 0
    assert counts["claim_too_broad"] > 0
    assert summary["blocking_gold_ambiguity"] is False
    assert summary["historical_gate_modified"] is False


def test_historical_results_and_gold_are_unchanged() -> None:
    expected = {
        "stage13_5": "7b6eee5c69edbba9428aeec71c7e9a827341d10da44f69686aa88ecb4e35a1cd",
        "stage13_6": "6c0ffcde74ff8db78f7adb8cf60658f07d151b0b08d0165cb6b705d32a928ef6",
        "stage13_7": "4565f174e8b9090c7e37fc6cf7e2c3a62a3591d447495b5d65e3d571bab7e910",
        "stage13_8": "2effd6cb6045789352e1a2592eb20a49bed45350eafb5b8086d02f7db6baec96",
        "stage13_9_metric_v2": "64eca96d55b58c26650701120dd19044ef8fe493e3051d834a91afb49d3ed2dd",
        "gold_set_v1": "24b21d7ce5264d4f22230cfb6bc9ec788ef6b76dc0ad629a20ae682c5184599e",
        "retrieval_gold_v2": "a196fc0c40823dd66b3972cf1d455d647325a20872cfe1f81685b967ec4e2e8d",
    }
    comparison = json.loads(
        (DATA / "claim-gold-citation-comparison-v1.json").read_text(encoding="utf-8")
    )
    assert comparison["historical_protection_sha256"] == expected
    historical = json.loads(
        (DATA / "evidence-qa-dev-v3-1.json").read_text(encoding="utf-8")
    )
    assert historical["metrics"]["all_manifest_conservative"]["citation_recall"] == 0.295
    assert historical["dev_v3_1_quality_candidate_gate"] is False


def test_readiness_is_not_authorization() -> None:
    readiness = json.loads(
        (DATA / "stage13-10-phase-b-readiness-v1.json").read_text(encoding="utf-8")
    )
    assert readiness["ready_for_dev_v3_2"] is True
    assert readiness["dev_v3_2_authorized"] is False
    assert readiness["dev_v3_2_executed"] is False
    assert readiness["gold_online_dependency"] is False
    assert readiness["human_label_online_dependency"] is False
    assert readiness["question_or_block_special_case"] is False
