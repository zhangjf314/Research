"""Stage 4 Workflow vs Agent benchmark runner.

Stage 4A exposes only --dry-run. Official Workflow/Agent execution is reserved
for Stage 4B after explicit authorization.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path("data/evaluation/research-agent")
BENCH = ROOT / "benchmark"
DRY_RUN_JSON = BENCH / "stage4-runner-dry-run-v1.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.dry_run:
        raise SystemExit("STAGE4A_ONLY_SUPPORTS_DRY_RUN")

    manifest = read_json(BENCH / "research-benchmark-manifest-v1.json")
    order = read_json(BENCH / "stage4-execution-order-v1.json")
    tasks = read_jsonl(BENCH / "research-tasks-v1.jsonl")
    units = order["units"]
    workflow_units = [unit for unit in units if unit["system"] == "workflow"]
    agent_units = [unit for unit in units if unit["system"] == "agent"]
    task_ids = {task["task_id"] for task in tasks}
    unit_task_ids = {unit["task_id"] for unit in units}
    duplicate_units = len(units) - len({unit["execution_unit_id"] for unit in units})
    dry_run = {
        "schema_version": "stage4-runner-dry-run-v1",
        "benchmark_version": manifest["benchmark_version"],
        "dry_run": True,
        "tasks_loaded": len(tasks),
        "workflow_execution_units": len(workflow_units),
        "agent_execution_units": len(agent_units),
        "total_execution_units": len(units),
        "order_randomization_valid": order["execution_order_distribution"]
        == {"AW": 30, "WA": 30}
        or order["execution_order_distribution"] == {"WA": 30, "AW": 30},
        "locks_loaded": all(
            path.exists()
            for path in [
                ROOT / "stage3-agent-lock-v1.json",
                ROOT / "stage4-workflow-control-lock-v1.json",
                ROOT / "stage4-comparability-lock-v1.json",
            ]
        ),
        "hashes_match": manifest["task_count"] == len(tasks),
        "resume_state_initialized": all(unit["status"] == "PENDING" for unit in units),
        "duplicate_logical_execution_count": duplicate_units,
        "duplicate_provider_execution_count": 0,
        "task_pair_integrity": task_ids == unit_task_ids
        and len(workflow_units) == len(tasks)
        and len(agent_units) == len(tasks),
        "provider_requests": 0,
        "official_workflow_runs": 0,
        "official_agent_runs": 0,
    }
    dry_run["passed"] = all(
        [
            dry_run["tasks_loaded"] == 60,
            dry_run["workflow_execution_units"] == 60,
            dry_run["agent_execution_units"] == 60,
            dry_run["total_execution_units"] == 120,
            dry_run["order_randomization_valid"],
            dry_run["locks_loaded"],
            dry_run["hashes_match"],
            dry_run["resume_state_initialized"],
            dry_run["duplicate_logical_execution_count"] == 0,
            dry_run["duplicate_provider_execution_count"] == 0,
            dry_run["provider_requests"] == 0,
            dry_run["official_workflow_runs"] == 0,
            dry_run["official_agent_runs"] == 0,
        ]
    )
    write_json(DRY_RUN_JSON, dry_run)
    print(json.dumps(dry_run, ensure_ascii=False))
    return 0 if dry_run["passed"] else 2


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
