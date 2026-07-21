from __future__ import annotations

import json

from scripts.evidence_qa_dev_lib_v1 import DATA


def load(name: str) -> dict:
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def test_replay_is_fixed_offline_and_schema_preserving() -> None:
    replay = load("dev-v3-2-offline-replay-v1.json")
    assert replay["question_count"] == 10
    assert replay["slot_count"] == 27
    assert replay["q005_refusal_unchanged"] is True
    assert replay["provider_calls"] == 0
    assert replay["embedding_calls"] == 0
    assert replay["reranker_called"] is False
    assert set(replay["modes"]) == {
        "baseline_v3_1",
        "primary_only",
        "primary_plus_cap",
        "obligation_coverage",
        "numeric_validator",
        "comparison_validator",
        "full_v3_2_candidate",
    }
    for mode in replay["modes"].values():
        assert mode["slots_evaluated"] == 27
        assert all(len(row["citation_ids"]) <= 3 for row in mode["per_slot"])


def test_full_candidate_passes_frozen_quality_preflight() -> None:
    replay = load("dev-v3-2-offline-replay-v1.json")
    audit = load("dev-v3-2-offline-replay-v1-final-audit.json")
    baseline = replay["modes"]["baseline_v3_1"]
    candidate = replay["modes"]["full_v3_2_candidate"]
    assert audit["engineering_gate"] == "PASSED"
    assert audit["quality_preflight"] == "PASSED"
    assert audit["live_ready"] is True
    assert audit["dev_v3_2_authorized"] is False
    assert audit["deterministic_replay_hash"] == (
        "db91b24c1452ac11311da60ecf44fad5ed0d4c85e16abde1aaf3328eb8a31c57"
    )
    assert candidate["wrong_evidence_selected"] < baseline["wrong_evidence_selected"]
    assert candidate["citation_dilution_rate"] <= baseline["citation_dilution_rate"]
    assert candidate["numeric_complete_rate"] >= baseline["numeric_complete_rate"]
    assert candidate["comparison_complete_rate"] >= baseline["comparison_complete_rate"]
    assert candidate["obligation_complete_rate"] >= baseline["obligation_complete_rate"]
    assert candidate["any_valid_evidence_recall_diagnostic"] >= (
        baseline["any_valid_evidence_recall_diagnostic"] - 0.02
    )


def test_focus_questions_expose_or_resolve_known_failure_types() -> None:
    replay = load("dev-v3-2-offline-replay-v1.json")
    rows = replay["modes"]["full_v3_2_candidate"]["per_slot"]
    by_question: dict[str, list[dict]] = {}
    for row in rows:
        by_question.setdefault(row["question_id"], []).append(row)
    assert any(
        row["fallback_action"] in {"answered_narrowed", "unsupported"}
        for row in by_question["q001"]
    )
    assert any(row["status"] == "unsupported" for row in by_question["q004"])
    assert any(
        row["citation_ids"] != row["baseline_citation_ids"]
        or row["status"] == "unsupported"
        for row in by_question["q015"]
    )
    assert any(row["status"] == "unsupported" for row in by_question["q019"])
    q050_comparison = next(
        row
        for row in by_question["q050"]
        if "rather than" in row["original_claim_text"]
    )
    assert q050_comparison["fallback_action"] == "answered_narrowed"


def test_feature_leakage_audit_passes() -> None:
    audit = load("dev-v3-2-feature-leakage-audit-v1.json")
    assert audit["gate"] == "PASSED"
    assert audit["gold_leakage"] is False
    assert audit["human_label_leakage"] is False
    assert audit["fixed_id_special_cases"] is False
    assert audit["production_path_may_read_gold"] is False


def test_protocol_candidate_is_frozen_but_not_authorized() -> None:
    protocol = load("dev-v3-2-protocol-candidate-v1.json")
    assert protocol["prompt_version"] == "qa-required-claims-citation-id-v3.2-candidate"
    assert protocol["output_schema"] == "required-claim-slots-v1.1"
    assert protocol["policy_versions"]["citation_budget"] == {
        "version": "citation-budget-v1",
        "max_primary": 1,
        "max_supporting": 2,
        "max_total": 3,
        "allow_zero_for_unsupported": True,
    }
    assert protocol["frozen_candidate"] is True
    assert protocol["live_authorized"] is False
    assert protocol["json_schema_sent"] is False
    assert protocol["tools_or_functions_sent"] is False
    assert protocol["correction_retry"] is False


def test_historical_inputs_remain_immutable() -> None:
    inputs = load("dev-v3-2-offline-preflight-inputs-v1.json")
    hashes = {
        row["source_path"].split("/")[-1]: row["hash"]["value"]
        for row in inputs["sources"]
    }
    assert hashes["claim-evidence-gold-dev-v1.jsonl"] == (
        "3cee289380c4b2ba861079d5f8470719a0d880f98812a5b55f28fb65693d37a6"
    )
    historical = load("evidence-qa-dev-v3-1.json")
    assert historical["metrics"]["all_manifest_conservative"]["citation_recall"] == 0.295
    assert historical["dev_v3_1_quality_candidate_gate"] is False
