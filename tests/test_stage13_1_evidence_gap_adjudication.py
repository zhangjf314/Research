from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from paper_research.evidence.schema import EvidenceUnit
from scripts import build_evidence_gap_adjudication_v1 as build
from scripts import review_claim_evidence_pilot_v1 as review
from scripts import run_stage13_1_phase_b_v1 as phase_b

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/evaluation"


def rows(name: str) -> list[dict]:
    return [
        json.loads(line) for line in (DATA / name).read_text(encoding="utf-8").splitlines() if line
    ]


def digest(name: str) -> str:
    return hashlib.sha256((DATA / name).read_bytes()).hexdigest()


def test_baseline_freeze_is_exact_stable_and_selection_is_gold_free() -> None:
    freeze = json.loads((DATA / "stage13-baseline-freeze-v1.json").read_text(encoding="utf-8"))
    assert (
        freeze["corpus_signature"]
        == "fb826cb8b1fe8e55d96b7aa1446cfb2e8d347efd74643a627706ad00e3792239"
    )
    assert freeze["exact_block_hits"] == 31
    assert freeze["answerable_questions"] == 48
    assert freeze["exact_block_availability"] == pytest.approx(31 / 48)
    assert freeze["gold_page_hits"] == 42
    assert freeze["gold_page_availability"] == pytest.approx(42 / 48)
    assert freeze["gold_block_recall"] == pytest.approx(0.319444)
    assert freeze["metadata_contamination"] == 0
    assert freeze["multi_paper_coverage"] == 1
    assert freeze["selection_integrity"] == {
        "gold_used_for_selection": False,
        "gold_required_claims_used_for_selection": False,
        "oracle_used_for_selection": False,
        "proof": freeze["selection_integrity"]["proof"],
    }
    assert freeze["configuration_fingerprint"] == build.stable_hash(freeze["configuration"])
    assert freeze["input_artifact_hashes"]["gold"] == digest("gold-set-v1.jsonl")
    assert freeze["input_artifact_hashes"]["protocol"] == digest("retrieval-gold-v2.jsonl")


def test_exact_gap_cases_have_complete_reviewed_diagnostics() -> None:
    gaps = rows("evidence-gap-cases-v1.jsonl")
    assert len(gaps) == 17
    assert {row["question_id"] for row in gaps} == {
        "q002",
        "q007",
        "q009",
        "q012",
        "q013",
        "q015",
        "q016",
        "q019",
        "q021",
        "q026",
        "q029",
        "q031",
        "q032",
        "q033",
        "q034",
        "q039",
        "q050",
    }
    assert all(row["initial_failure_category"] in build.TAXONOMY for row in gaps)
    assert all(row["human_review_status"] == "reviewed" for row in gaps)
    assert Counter(row["human_failure_category"] for row in gaps) == {
        "page_hit_block_miss": 6,
        "block_type_filter_error": 6,
        "fusion_rank_failure": 4,
        "parsing_boundary_error": 1,
    }
    assert all(row["reviewer"] and row["reviewed_at"] for row in gaps)
    assert all(len(row["candidate_pool_top_30"]) <= 30 for row in gaps)
    assert all(
        item["rank"] == rank
        for row in gaps
        for rank, item in enumerate(row["candidate_pool_top_30"], 1)
    )
    page_hit_misses = [row for row in gaps if row["gold_page_hit"]]
    assert page_hit_misses
    assert all(not set(row["selected_block_ids"]) & set(row["gold_block_ids"]) for row in gaps)


def test_taxonomy_has_required_contract() -> None:
    required = {
        "definition",
        "criteria",
        "observable_evidence",
        "retrieval_implementation_issue",
        "automatic_fix_allowed",
        "human_review_required",
        "affects_gold",
        "affects_production_metric",
    }
    assert "metric_exact_match_limitation" in build.TAXONOMY
    assert "unknown" in build.TAXONOMY
    assert all(required <= set(value) for value in build.TAXONOMY.values())


def test_pilot_is_unique_representative_reviewed_and_does_not_modify_claims() -> None:
    pilot = rows("claim-evidence-gold-pilot-v1.jsonl")
    claims = {row["claim_id"]: row for row in rows("claim-units-v1.jsonl")}
    assert len(pilot) == 40
    assert len({row["pilot_sample_id"] for row in pilot}) == 40
    strata = Counter(row["sampling_stratum"] for row in pilot)
    assert strata == {
        "exact_miss_core_claim": 17,
        "exact_hit_incomplete_recall": 8,
        "multi_paper": 5,
        "unknown_claim_role": 5,
        "control": 5,
    }
    assert all(row["annotation_status"] == "reviewed" for row in pilot)
    assert all(row["decision"] == "approved" for row in pilot)
    assert all(row["reviewer"] and row["review_notes"] for row in pilot)
    assert all(row["claim_text"] == claims[row["claim_id"]]["claim_text"] for row in pilot)
    assert sum(row["claim_role"] == "unknown" for row in pilot) >= 5
    assert sum(bool(row["multi_block_required"]) for row in pilot) == 12


def test_review_parser_and_evidence_validation_enforce_triples_and_target_paper() -> None:
    parsed = review.parse_set("1706.03762:2:b000022,1706.03762:2:b000025")
    assert [row["block_id"] for row in parsed[0]] == ["b000022", "b000025"]
    evidence_rows = [EvidenceUnit.model_validate(row) for row in rows("evidence-corpus-v1.jsonl")]
    lookup = {(row.paper_id, row.page, row.block_id): row for row in evidence_rows}
    review.validate_sets(parsed, {"1706.03762"}, lookup)
    with pytest.raises(ValueError, match="outside target"):
        review.validate_sets(parsed, {"2104.08691"}, lookup)
    with pytest.raises(ValueError, match="does not exist"):
        review.validate_sets(
            [[{"paper_id": "1706.03762", "page": 99, "block_id": "missing"}]],
            {"1706.03762"},
            lookup,
        )


def test_source_hash_change_invalidates_old_review(monkeypatch: pytest.MonkeyPatch) -> None:
    sample = rows("claim-evidence-gold-pilot-v1.jsonl")[0]
    sample.update(
        {
            "annotation_status": "approved",
            "decision": "approved",
            "reviewer": "human",
            "reviewed_at": "2026-07-14",
            "review_notes": "checked",
            "approved_evidence_sets": [[{"paper_id": "p", "page": 1, "block_id": "b"}]],
            "source_hashes": {"old": "hash"},
        }
    )
    monkeypatch.setattr(review, "source_hashes", lambda: {"new": "hash"})
    assert review.invalidate_stale([sample])
    assert sample["annotation_status"] == "pending"
    assert sample["reviewer"] is None
    assert sample["approved_evidence_sets"] == []


def test_phase_a_security_and_claim_metrics_remain_pending() -> None:
    freeze = json.loads((DATA / "stage13-baseline-freeze-v1.json").read_text(encoding="utf-8"))
    audit = json.loads((DATA / "claim-first-failure-audit-v1.json").read_text(encoding="utf-8"))
    assert freeze["model_calls"] == {"llm": 0, "embedding": 0, "reranker": 0, "deep_research": 0}
    assert audit["gold_used_for_selection"] is False
    assert audit["approved_pilot_count"] == 0
    assert audit["claim_level_metrics_status"] == "unavailable_until_pilot_review"
    source = (ROOT / "scripts/build_evidence_gap_adjudication_v1.py").read_text(encoding="utf-8")
    assert "if question_id ==" not in source
    assert "if block_id ==" not in source


def test_phase_b_review_merge_validation_and_ablation_gates() -> None:
    validation = phase_b.validate_reviewed()
    assert validation["gap_schema_valid"]
    assert validation["pilot_schema_valid"]
    assert validation["source_hashes_valid"]
    assert validation["source_record_hash_valid"]
    assert validation["triples_valid"]
    assert validation["triples_checked"] == 806
    ablation = json.loads(
        (DATA / "evidence-retrieval-phase-b-ablation-v1.json").read_text(encoding="utf-8")
    )
    assert ablation["dev_qa_run"] is False
    assert ablation["llm_called"] is False
    assert ablation["reranker_enabled"] is False
    assert ablation["deep_research_called"] is False
    assert ablation["hit_loss_questions"] == []
    metrics = {
        item["name"]: item["metrics"]
        for item in ablation["variants"]
    }
    assert metrics["stage13_routed_baseline"]["exact_gold_block_availability"] == pytest.approx(
        31 / 48
    )
    assert metrics["phase_b_adjacent_same_page_completion"][
        "exact_gold_block_availability"
    ] == pytest.approx(35 / 48)
    assert all(ablation["offline_candidate_gates"].values())


def test_phase_b_rule_is_generic_and_does_not_use_gold_or_ids() -> None:
    source = (ROOT / "src/paper_research/retrieval/context_completion.py").read_text(
        encoding="utf-8"
    )
    assert "question_id" not in source
    assert "gold" not in source.casefold()
    assert "block_id ==" not in source
