"""Validate frozen Stage 4 research benchmark artifacts offline."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("data/evaluation/research-agent")
BENCH = ROOT / "benchmark"


def main() -> int:
    manifest = read_json(BENCH / "research-benchmark-manifest-v1.json")
    validation = read_json(BENCH / "research-benchmark-validation-v1.json")
    tasks = read_jsonl(BENCH / "research-tasks-v1.jsonl")
    rubrics = read_jsonl(BENCH / "research-task-rubrics-v1.jsonl")
    order = read_json(BENCH / "stage4-execution-order-v1.json")
    agent = read_json(ROOT / "stage3-agent-lock-v1.json")
    workflow = read_json(ROOT / "stage4-workflow-control-lock-v1.json")
    comparability = read_json(ROOT / "stage4-comparability-lock-v1.json")

    failures = []
    if len(tasks) != manifest["task_count"]:
        failures.append("TASK_COUNT_MISMATCH")
    if len(tasks) < 55:
        failures.append("BENCHMARK_VALID_TASKS_LT_55")
    if len(rubrics) != len(tasks):
        failures.append("RUBRIC_COUNT_MISMATCH")
    if any(task["review_status"] != "approved" for task in tasks):
        failures.append("NON_APPROVED_TASK")
    if any(len(task["required_dimensions"]) < 2 for task in tasks):
        failures.append("DIMENSION_CONTRACT_FAILED")
    if any(len(rubric["required_claims"]) < 3 for rubric in rubrics):
        failures.append("CLAIM_CONTRACT_FAILED")
    if validation["unsupported_gold_claim_count"] != 0:
        failures.append("UNSUPPORTED_GOLD_CLAIM")
    if validation["exact_duplicates"]:
        failures.append("EXACT_DUPLICATES")
    if validation["unresolved_near_duplicates"] != 0:
        failures.append("NEAR_DUPLICATES")
    if validation["stage3_exclusion_violations"]:
        failures.append("STAGE3_EXCLUSION_VIOLATION")
    if validation["papers_covered"] < 30:
        failures.append("PAPER_COVERAGE_FAILED")
    if order["workflow_execution_units"] != len(tasks):
        failures.append("WORKFLOW_UNIT_COUNT_MISMATCH")
    if order["agent_execution_units"] != len(tasks):
        failures.append("AGENT_UNIT_COUNT_MISMATCH")
    if order["total_execution_units"] != len(tasks) * 2:
        failures.append("TOTAL_UNIT_COUNT_MISMATCH")
    if agent["stage2_rag_backend_hash"] != manifest["rag_backend_hash"]:
        failures.append("AGENT_RAG_HASH_MISMATCH")
    if workflow["workflow_behavior_changed"] is not False:
        failures.append("WORKFLOW_BEHAVIOR_DRIFT")
    if comparability["same_retrieval_backend"] is not True:
        failures.append("COMPARABILITY_RETRIEVAL_FAILED")

    result = {
        "schema_version": "validate-research-benchmark-v1",
        "benchmark_version": manifest["benchmark_version"],
        "task_count": len(tasks),
        "total_execution_units": order["total_execution_units"],
        "provider_requests": 0,
        "official_workflow_runs": 0,
        "official_agent_runs": 0,
        "failures": failures,
        "passed": not failures,
    }
    print(json.dumps(result, ensure_ascii=False))
    return 0 if not failures else 2


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


if __name__ == "__main__":
    raise SystemExit(main())
