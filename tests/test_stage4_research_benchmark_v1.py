from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path("data/evaluation/research-agent")
BENCH = ROOT / "benchmark"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_stage4_benchmark_schema_distribution_and_review_status() -> None:
    tasks = _jsonl(BENCH / "research-tasks-v1.jsonl")
    rubrics = _jsonl(BENCH / "research-task-rubrics-v1.jsonl")
    manifest = _json(BENCH / "research-benchmark-manifest-v1.json")

    assert len(tasks) == 60
    assert len(rubrics) == 60
    assert manifest["task_count"] == 60
    assert manifest["task_distribution"] == {
        "cross_paper_comparison": 15,
        "evidence_insufficiency_or_noncomparability": 5,
        "limitations_and_research_gaps": 10,
        "methods_and_experiments": 10,
        "multi_paper_synthesis": 15,
        "observation_dependent_research": 5,
    }
    assert manifest["difficulty_distribution"] == {
        "easy": 10,
        "hard": 20,
        "medium": 30,
    }
    assert all(task["review_status"] == "approved" for task in tasks)
    assert all(task["ai_review_decision"] == "APPROVE" for task in tasks)


def test_research_task_contracts_are_evidence_first_and_self_contained() -> None:
    tasks = _jsonl(BENCH / "research-tasks-v1.jsonl")
    rubrics = {row["task_id"]: row for row in _jsonl(BENCH / "research-task-rubrics-v1.jsonl")}
    banned = ["target paper", "the two papers", "the studies above"]

    for task in tasks:
        question = task["research_question"].lower()
        assert all(phrase not in question for phrase in banned)
        assert len(task["target_paper_ids"]) >= 2
        assert 2 <= len(task["required_dimensions"]) <= 5
        rubric = rubrics[task["task_id"]]
        assert len(rubric["required_claims"]) >= 3
        evidence_set_ids = {
            evidence_set["evidence_set_id"]
            for evidence_set in rubric["required_evidence_sets"]
        }
        for claim in rubric["required_claims"]:
            assert claim["supporting_evidence_ids"]
            assert set(claim["supporting_evidence_ids"]) <= evidence_set_ids
        for evidence_set in rubric["required_evidence_sets"]:
            assert evidence_set["paper_ids"]
            assert evidence_set["block_ids"]
            assert evidence_set["evidence_ids"]
            assert evidence_set["pages"]
            assert all(
                len(item["claim_summary"]) <= 183
                for item in evidence_set["evidence_audit"]
            )


def test_validation_gate_and_stage3_exclusions_pass() -> None:
    validation = _json(BENCH / "research-benchmark-validation-v1.json")

    assert validation["stage4a_complete"] is True
    assert validation["stage4b_ready"] is True
    assert validation["benchmark_valid_tasks"] == 60
    assert validation["tasks_approved"] == 60
    assert validation["papers_covered"] == 33
    assert validation["required_dimension_completeness"] == 1.0
    assert validation["required_claim_evidence_completeness"] == 1.0
    assert validation["unsupported_gold_claim_count"] == 0
    assert validation["exact_duplicates"] == []
    assert validation["near_duplicate_clusters"] == []
    assert validation["unresolved_near_duplicates"] == 0
    assert validation["stage3_exclusion_count"] == 6
    assert validation["stage3_exclusion_violations"] == []


def test_freeze_hashes_match_manifest_and_lock_semantics() -> None:
    manifest = _json(BENCH / "research-benchmark-manifest-v1.json")
    agent_lock = _json(ROOT / "stage3-agent-lock-v1.json")
    workflow_lock = _json(ROOT / "stage4-workflow-control-lock-v1.json")

    assert manifest["dataset_hash"] == (
        "45e1369b2810630b0dfe94ab94b784d8984df791ea87500fea882752159288b5"
    )
    assert manifest["stage4_research_tasks_hash"] == (
        "f72418172c0ce1405c2884c190ff35577d1fcbc8b0afb332e63ee049036a6359"
    )
    assert manifest["stage4_research_rubric_hash"] == (
        "feb370b5521a8395200b4422392e67b33c44ed813cdc920073f28e8b4cf545fc"
    )
    assert manifest["stage4_execution_order_hash"] == (
        "166ea1f41583ee8db52fec5ec21561cc10979cf4f238af9850ea31b68e18beb7"
    )
    assert manifest["stage4_evaluation_protocol_hash"] == (
        "a5f6ac812173e2dcec23507954b383383a053fba5845cd524d45a4766d1a44a2"
    )
    assert manifest["workflow_lock_hash"] == _sha(
        ROOT / "stage4-workflow-control-lock-v1.json"
    )
    assert manifest["agent_lock_hash"] == _sha(ROOT / "stage3-agent-lock-v1.json")
    assert manifest["rag_backend_hash"] == (
        "995a144385180b2931ec2c6366f7f7306301a42d77ad7c85f4be9e6d9e5091d9"
    )
    assert manifest["agent_behavior_hash"] == (
        "bce71a51171b2e1187d579a2278cc34f1202ed7b84e9482cbffe42d00b92ff15"
    )
    assert agent_lock["stage2_rag_backend_hash"] == manifest["rag_backend_hash"]
    assert agent_lock["stage3_agent_behavior_hash"] == manifest["agent_behavior_hash"]
    assert workflow_lock["workflow_behavior_changed"] is False


def test_execution_order_is_deterministic_paired_and_not_executed() -> None:
    order = _json(BENCH / "stage4-execution-order-v1.json")
    units = order["units"]

    assert order["execution_seed"] == 40721
    assert order["execution_order_distribution"] == {"AW": 30, "WA": 30}
    assert order["workflow_execution_units"] == 60
    assert order["agent_execution_units"] == 60
    assert order["total_execution_units"] == 120
    assert len({unit["execution_unit_id"] for unit in units}) == 120
    assert all(unit["status"] == "PENDING" for unit in units)
    assert order["provider_requests"] == 0
    assert order["official_workflow_runs"] == 0
    assert order["official_agent_runs"] == 0


def test_fairness_and_replan_metrics_are_frozen() -> None:
    fairness = _json(BENCH / "stage4-fairness-audit-v1.json")
    protocol = _json(BENCH / "stage4-evaluation-protocol-v1.json")
    comparability = _json(ROOT / "stage4-comparability-lock-v1.json")

    assert fairness["corpus"] is True
    assert fairness["index"] is True
    assert fairness["embedding"] is True
    assert fairness["retriever"] is True
    assert fairness["reranker"] is True
    assert fairness["query_rewrite"] is True
    assert fairness["query_decomposition"] is True
    assert protocol["bootstrap"]["resamples"] == 1000
    assert protocol["bootstrap"]["seed"] == 41007
    assert protocol["agent_behavioral_metrics"][:8] == comparability[
        "replan_metrics_preregistered"
    ]
    assert "new real Tool Action" in protocol["effective_replan_definition"]


def test_validator_and_runner_dry_run_do_not_call_provider() -> None:
    validate = subprocess.run(
        [".\\.venv\\Scripts\\python.exe", "scripts\\validate_research_benchmark_v1.py"],
        text=True,
        capture_output=True,
        check=False,
    )
    dry_run = subprocess.run(
        [
            ".\\.venv\\Scripts\\python.exe",
            "scripts\\run_stage4_workflow_agent_benchmark_v1.py",
            "--dry-run",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert validate.returncode == 0, validate.stderr
    assert dry_run.returncode == 0, dry_run.stderr
    validate_payload = json.loads(validate.stdout)
    dry_run_payload = json.loads(dry_run.stdout)
    assert validate_payload["provider_requests"] == 0
    assert dry_run_payload["provider_requests"] == 0
    assert dry_run_payload["official_workflow_runs"] == 0
    assert dry_run_payload["official_agent_runs"] == 0
    assert dry_run_payload["passed"] is True
