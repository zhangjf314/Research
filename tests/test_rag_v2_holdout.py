"""RQ10 §20 tests: freeze provenance, holdout freezing/sealing, arms, defaults."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FREEZE_V2 = (
    ROOT / "artifacts" / "rag-quality-v2" / "candidate-freeze"
    / "candidate-freeze-C-star-e1-a3a-lexical-gated-v2.json"
)
QUESTIONS = (
    ROOT / "artifacts" / "rag-quality-v2" / "shadow-holdout" / "frozen"
    / "shadow-holdout-questions-v1.json"
)


def test_candidate_freeze_provenance_fields() -> None:
    manifest = json.loads(FREEZE_V2.read_text(encoding="utf-8"))
    for field in (
        "rq9_baseline_commit",
        "candidate_implementation_commit",
        "candidate_freeze_commit",
        "candidate_config_hash",
        "embedding_representation_hash",
        "dataset_hash",
        "fusion_policy_hash",
        "implementation_files",
    ):
        assert field in manifest, field
    assert manifest["rq9_baseline_commit"] != manifest["candidate_implementation_commit"]


def test_candidate_implementation_commit_reconstructs() -> None:
    """The recorded commit must actually contain the candidate implementation."""

    manifest = json.loads(FREEZE_V2.read_text(encoding="utf-8"))
    commit = manifest["candidate_implementation_commit"]
    for rel in manifest["implementation_files"]:
        code = subprocess.run(
            ["git", "cat-file", "-e", f"{commit}:{rel}"],
            cwd=ROOT,
        ).returncode
        assert code == 0, f"{rel} missing in {commit}"
    # config hash round-trips from the recorded config block
    import hashlib

    spec = {
        "embedding_representation": manifest["embedding_representation"],
        "embedding_model": manifest["embedding_model"],
        "embedding_dimensions": manifest["embedding_dimensions"],
        "distance": manifest["distance"],
        "retrieval_k": manifest["retrieval_k"],
        "fusion_config": manifest["fusion_config"],
        "context_config": manifest["context_config"],
        "dataset_hash": manifest["dataset_hash"],
    }
    recomputed = hashlib.sha256(
        json.dumps(spec, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    assert recomputed == manifest["candidate_config_hash"]


def test_question_set_is_frozen_and_hash_stable() -> None:
    import hashlib

    body = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    assert body["status"] == "FROZEN_BEFORE_RETRIEVAL"
    payload = json.dumps(
        {"questions": body["questions"]}, ensure_ascii=False, sort_keys=True
    ).encode("utf-8")
    assert hashlib.sha256(payload).hexdigest() == body["holdout_question_set_hash"]
    assert body["question_count"] == len(body["questions"]) == 50
    assert body["paper_count"] == 10
    # anti-lexical-bias audit registered
    audit = body["phrasing_audit"]
    assert audit["paraphrase"] >= 20
    assert audit["unanswerable"] >= 5


def test_annotation_contract_validation() -> None:
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    from seal_rag_v2_holdout_v1 import validate_records

    good = [
        {
            "question_id": "q1",
            "paper_id": "p",
            "question": "?",
            "question_type": "method",
            "answerability": True,
            "required_claims": ["c"],
            "primary_gold_block_ids": ["b1"],
            "acceptable_alternative_block_ids": ["b2"],
            "gold_pages": [1],
            "gold_sections": ["Methods"],
            "annotation_notes": "",
            "annotation_status": "finalized",
        },
        {
            "question_id": "q2",
            "paper_id": "p",
            "question": "?",
            "question_type": "unanswerable",
            "answerability": False,
            "required_claims": [],
            "primary_gold_block_ids": [],
            "acceptable_alternative_block_ids": [],
            "gold_pages": [],
            "gold_sections": [],
            "annotation_notes": "not in paper",
            "annotation_status": "finalized",
        },
    ]
    assert validate_records(good) == []
    bad_unanswerable = [good[0] | {"answerability": False}]
    assert any("empty gold" in p for p in validate_records(bad_unanswerable))
    bad_no_gold = [good[0] | {"primary_gold_block_ids": []}]
    assert any("requires primary gold" in p for p in validate_records(bad_no_gold))
    bad_status = [good[0] | {"annotation_status": "pending"}]
    assert any("finalized" in p for p in validate_records(bad_status))


def test_primary_vs_acceptable_gold_metrics() -> None:
    from paper_research.evaluation.rag_v2_diagnosis import recall_at

    ranked = ["b1", "x", "b2"]
    primary = {"b1"}
    acceptable = {"b1", "b2"}
    assert recall_at(ranked, primary, 5) == 1.0  # exact-match stays authoritative
    assert recall_at(ranked, acceptable, 5) == 1.0
    assert recall_at(ranked, primary, 1) == 1.0
    assert recall_at(ranked, acceptable, 1) == 0.5  # denominator = 2 gold blocks
    assert recall_at(ranked[:1], acceptable, 5) == 0.5  # alternative distinguishes


def test_paired_bootstrap_deterministic() -> None:
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    from run_rag_v2_holdout_eval_v1 import paired_bootstrap

    a = [1.0, 0.5, 0.0, 0.25, 0.75]
    b = [0.5, 0.5, 0.5, 0.25, 0.5]
    first = paired_bootstrap(a, b)
    second = paired_bootstrap(a, b)
    assert first == second
    assert first["ci95_low"] <= first["point_estimate"] <= first["ci95_high"]


def test_evaluation_arm_identity() -> None:
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    source = (ROOT / "scripts" / "run_rag_v2_holdout_eval_v1.py").read_text(
        encoding="utf-8"
    )
    for arm in ("H0_production_equal_rrf", "H1_bm25_only", "H2_cstar_e1_a3a"):
        assert arm in source


def test_production_default_backend_protection() -> None:
    from paper_research.config import Settings

    defaults = Settings()
    assert defaults.retrieval_backend == "production_hybrid_rrf"
    with pytest.raises(ValueError):
        Settings(retrieval_backend="bm25_only_from_holdout")
    with pytest.raises(ValueError):
        Settings(retrieval_backend="random")


def test_holdout_eval_refuses_without_seal(tmp_path: None = None) -> None:
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    from run_rag_v2_holdout_eval_v1 import load_sealed

    marker = (
        ROOT / "artifacts" / "rag-quality-v2" / "shadow-holdout" / "frozen"
        / "shadow-holdout-sealed-v1.json"
    )
    if marker.exists():
        pytest.skip("holdout already sealed in this tree")
    with pytest.raises(SystemExit) as exc:
        load_sealed()
    assert "SHADOW_HOLDOUT_PENDING_HUMAN_REVIEW" in str(exc.value)
