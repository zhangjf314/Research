"""RQ14 §29 tests: registry, freezes, quotas, schema, seal, harness guards."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "artifacts" / "rag-quality-v2" / "shadow-holdout-v2"


def test_c3_freeze_unchanged_and_pool_depth_12() -> None:
    manifest = json.loads(
        (
            ROOT
            / "artifacts"
            / "rag-quality-v2"
            / "rq13"
            / "candidate-freeze"
            / "candidate-freeze-C3-set-aware-rerank.json"
        ).read_text(encoding="utf-8")
    )
    assert manifest["candidate_config_hash"] == "bc9ee4c3465bb62f"
    assert manifest["policy_hash"] == "1c8dd0fbbdaddc61"
    assert manifest["candidate_generation_contract"]["pool_depth"] == 12


def test_contamination_registry_matching_methods() -> None:
    registry = json.loads(
        (V2 / "contamination" / "contamination-registry-v1.json").read_text(encoding="utf-8")
    )
    assert set(registry["matching_methods"]) == {"arxiv_id", "doi", "normalized_title"}
    ids = {p["arxiv_id"] for p in registry["papers"] if p["arxiv_id"]}
    titles = {p["normalized_title"] for p in registry["papers"] if p["normalized_title"]}
    assert "1706.03762" in ids and "2201.11903" in ids
    assert all(p["contaminated_for_future_holdout"] for p in registry["papers"])
    assert len(ids | titles) >= 14


def test_holdout_v2_papers_absent_from_registry() -> None:
    paper_set = json.loads(
        (V2 / "papers" / "shadow-holdout-v2-paper-set-v1.json").read_text(encoding="utf-8")
    )
    registry = json.loads(
        (V2 / "contamination" / "contamination-registry-v1.json").read_text(encoding="utf-8")
    )
    reg_ids = {p["arxiv_id"] for p in registry["papers"] if p["arxiv_id"]}
    reg_titles = {p["normalized_title"] for p in registry["papers"] if p["normalized_title"]}
    import re

    def norm(t):
        return re.sub(r"[^a-z0-9]+", "", (t or "").lower())

    for paper in paper_set["papers"]:
        assert paper["paper_id"] not in reg_ids
        assert norm(paper["title"]) not in reg_titles
    assert paper_set["paper_count"] == 12


def test_question_set_frozen_with_exact_quotas() -> None:
    body = json.loads(
        (V2 / "frozen" / "shadow-holdout-v2-questions-v1.json").read_text(encoding="utf-8")
    )
    assert body["question_count"] == 60
    assert body["category_distribution"] == {
        "comparison": 6,
        "definition": 5,
        "formula": 5,
        "limitation": 3,
        "method": 7,
        "multi_evidence": 10,
        "parameter_value": 3,
        "quantitative_result": 7,
        "semantic_paraphrase": 9,
        "unanswerable": 5,
    }
    per_paper = {}
    for q in body["questions"]:
        per_paper[q["paper_id"]] = per_paper.get(q["paper_id"], 0) + 1
    assert len(per_paper) == 12 and set(per_paper.values()) == {5}
    strong = sum(
        1
        for q in body["questions"]
        if q["question_type"] == "semantic_paraphrase"
        and q["phrasing_type"] in ("strong_paraphrase", "moderate_paraphrase")
    )
    assert strong >= 5
    assert all(q["retrieval_viewed_before_question_freeze"] is False for q in body["questions"])


def test_question_hash_stable() -> None:
    import hashlib

    body = json.loads(
        (V2 / "frozen" / "shadow-holdout-v2-questions-v1.json").read_text(encoding="utf-8")
    )
    payload = json.dumps(
        {"questions": body["questions"]}, ensure_ascii=False, sort_keys=True
    ).encode("utf-8")
    assert hashlib.sha256(payload).hexdigest() == body["question_set_hash"]


def test_workbook_blind_and_second_pass() -> None:
    lines = (
        (V2 / "annotation" / "annotation-workbook-v2.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    records = [json.loads(line) for line in lines if line.strip()]
    assert len(records) == 60
    assert sum(r["second_pass_review"] for r in records) >= 15
    for r in records:
        assert r["annotation_status"] == "pending"
        assert r["primary_gold_block_ids"] == []
        assert r["required_claims"] == []
        # no ranking/score fields in workbook
        for key in r:
            assert "score" not in key and "rank" not in key and "rrf" not in key.lower()


def test_seal_v2_validator_rejects_inconsistent_gold(tmp_path: Path) -> None:
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    from seal_rag_v2_holdout_v2_v1 import validate_records

    questions = json.loads(
        (V2 / "frozen" / "shadow-holdout-v2-questions-v1.json").read_text(encoding="utf-8")
    )["questions"]
    qmap = {q["question_id"]: q for q in questions}
    base = {
        "question_id": questions[0]["question_id"],
        "question": questions[0]["question"],
        "annotation_status": "finalized",
        "answerability": True,
        "required_claims": [
            {
                "claim_id": "c1",
                "text": "claim",
                "primary_gold_block_ids": ["b000001"],
                "acceptable_alternative_block_ids": [],
            }
        ],
        "primary_gold_block_ids": ["b000001"],
        "acceptable_alternative_block_ids": [],
        "block_inventory": [{"block_id": "b000001"}],
    }
    assert validate_records([base], qmap) == []
    bad_flat = base | {"primary_gold_block_ids": ["b000002"]}
    assert any("inconsistent" in p for p in validate_records([bad_flat], qmap))
    bad_unanswerable = base | {"answerability": False}
    assert any("unanswerable" in p for p in validate_records([bad_unanswerable], qmap))
    bad_missing = base | {"primary_gold_block_ids": ["bZZZZZ"]}
    bad_missing["required_claims"][0] = {
        **bad_missing["required_claims"][0],
        "primary_gold_block_ids": ["bZZZZZ"],
    }
    assert any("not in own inventory" in p for p in validate_records([bad_missing], qmap))


def test_gate_document_committed_before_evaluation() -> None:
    gate = ROOT / "docs" / "rag-quality-v2-rq14-holdout-v2-gate.md"
    assert gate.exists()
    text = gate.read_text(encoding="utf-8")
    for threshold in ("0.02", "0.05", "0.01", "10 000 resamples", "20260823", "12"):
        assert threshold in text
    code = subprocess.run(
        ["git", "log", "--oneline", "--", str(gate)], cwd=ROOT, capture_output=True, text=True
    ).stdout
    # gate must be tracked in git (committed) before RQ15 execution
    assert code.strip(), "gate document not committed"


def test_rq15_harness_loads_sealed_dataset_and_rejects_tampering(tmp_path: Path) -> None:
    # after sealing (RQ15 preflight), load_sealed must succeed and match
    import shutil
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    from run_rag_v2_holdout_v2_eval_v1 import load_sealed

    marker = json.loads(
        (V2 / "frozen" / "shadow-holdout-v2-sealed-v1.json").read_text(encoding="utf-8")
    )
    sealed = load_sealed()
    assert sealed["sealed"]["holdout_dataset_hash"] == marker["holdout_dataset_hash"]
    assert len(sealed["records"]) == 60
    # A byte-edited sealed copy must fail the evaluator's actual verifier.
    copied = tmp_path / "shadow-holdout-v2"
    shutil.copytree(V2, copied)
    annotated = copied / "frozen" / "shadow-holdout-v2-annotated-v1.jsonl"
    rows = [json.loads(line) for line in annotated.read_text(encoding="utf-8").splitlines()]
    rows[0]["primary_gold_block_ids"] = ["bTAMPER"]
    annotated.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    with pytest.raises(SystemExit, match="HOLDOUT_V2_SEAL_INVALID"):
        load_sealed(copied)


def test_v2_non_promotable() -> None:
    source = (ROOT / "scripts" / "run_rag_v2_holdout_v2_eval_v1.py").read_text(encoding="utf-8")
    assert "V2_bm25_only" in source
    gate = (ROOT / "docs" / "rag-quality-v2-rq14-holdout-v2-gate.md").read_text(encoding="utf-8")
    assert "non-promotable" in gate
