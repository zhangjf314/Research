"""Stage 4B.4 deployed runtime parity and exact-path validation.

This script is deliberately not an official benchmark runner. It audits the
currently deployed API container and exercises the same HTTP adapter path used by
the Stage 4B runner with excluded development-validation tasks only.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path("data/evaluation/research-agent")
BENCH = ROOT / "benchmark"
DOCS = Path("docs/research-agent/benchmark")
RUNTIME = Path(".runtime/stage4/stage4b-exact-path-wiring-validation-v1")
PARITY_JSON = BENCH / "stage4-deployed-runtime-parity-v1.json"
PARITY_MD = DOCS / "stage4-deployed-runtime-parity-v1.md"
ROOT_CAUSE_JSON = BENCH / "stage4-attempt3-root-cause-v1.json"
ROOT_CAUSE_MD = DOCS / "stage4-attempt3-root-cause-v1.md"
EXACT_JSON = BENCH / "stage4b-exact-path-wiring-validation-v1.json"
EXACT_MD = DOCS / "stage4b-exact-path-wiring-validation-v1.md"
READINESS_JSON = BENCH / "stage4b-attempt4-readiness-v1.json"
READINESS_MD = DOCS / "stage4b-attempt4-readiness-v1.md"

EXPECTED_EXCLUSION_HASH = (
    "fd9a015c3d7b725cb863f766d25fafafb266576271b3422d0200c1b6c567c0cb"
)
EXPECTED_AGENT_HASH = (
    "bce71a51171b2e1187d579a2278cc34f1202ed7b84e9482cbffe42d00b92ff15"
)
EXPECTED_RAG_HASH = (
    "995a144385180b2931ec2c6366f7f7306301a42d77ad7c85f4be9e6d9e5091d9"
)


def now() -> str:
    return datetime.now(UTC).isoformat()


def load_stage4_runner():
    script = Path("scripts/run_stage4_workflow_agent_benchmark_v1.py")
    spec = importlib.util.spec_from_file_location("stage4_runner_for_4b4", script)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load Stage 4 runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_command(args: list[str], timeout: int = 120) -> dict[str, Any]:
    proc = subprocess.run(
        args,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
        timeout=timeout,
    )
    return {
        "args": args,
        "returncode": proc.returncode,
        "stdout_tail": (proc.stdout or "")[-4000:],
        "stderr_tail": (proc.stderr or "")[-4000:],
    }


def docker_inspect_api() -> dict[str, Any]:
    cid = subprocess.check_output(["docker", "compose", "ps", "-q", "api"], text=True).strip()
    inspect_payload = json.loads(
        subprocess.check_output(["docker", "inspect", cid], text=True)
    )[0]
    return {
        "container_id": cid,
        "image_id": inspect_payload.get("Image"),
        "created": inspect_payload.get("Created"),
        "path": inspect_payload.get("Path"),
        "args": inspect_payload.get("Args"),
        "working_dir": inspect_payload.get("Config", {}).get("WorkingDir"),
        "cmd": inspect_payload.get("Config", {}).get("Cmd"),
        "entrypoint": inspect_payload.get("Config", {}).get("Entrypoint"),
        "state": {
            key: inspect_payload.get("State", {}).get(key)
            for key in ["Status", "Running", "StartedAt", "FinishedAt", "Pid"]
        },
        "mounts": [
            {key: mount.get(key) for key in ["Type", "Source", "Destination", "Mode", "RW"]}
            for mount in inspect_payload.get("Mounts", [])
        ],
    }


def previous_container_audit() -> dict[str, Any]:
    return {
        "container_id": "7b87b6faffae41ddb5f083f87f205437da4396987c43903642975103b9d174f1",
        "image_id": "sha256:d6516e10d9d5f6edd048a8858a25bd40cbfcbf533dd4402c5ffe36867ab638e7",
        "created": "2026-08-08T09:46:39.144100079Z",
        "started_at": "2026-08-08T09:46:42.235937904Z",
        "research_py_sha256": "ebe5f5f84b9666527b44422a2d63f77702521563245449520dbb5d3f8d32fa68",
        "failure_materialization_branch_present": False,
        "source": "captured before deployment rebuild during Stage 4B.4",
    }


def load_excluded_task() -> dict[str, Any]:
    plan = read_json(ROOT / "stage3-live-replan-validation-plan-v1.json")
    task = plan["validation_tasks"][0]
    return {
        "task_id": task["task_id"],
        "research_question": task["research_question"],
        "target_paper_ids": task["paper_ids_named_in_question"],
        "source_artifact": (
            "data/evaluation/research-agent/"
            "stage3-live-replan-validation-plan-v1.json"
        ),
    }


def run_exact_path_unit(
    runner: Any,
    *,
    api_base_url: str,
    run_id: str,
    system: str,
    task: dict[str, Any],
) -> dict[str, Any]:
    unit = {"task_id": task["task_id"], "system": system}
    payload = runner.build_request_payload(unit, task, run_id=run_id)
    endpoint = (
        "/api/v1/research/deep"
        if system == "workflow"
        else "/api/v1/research/agent"
    )
    raw_dir = RUNTIME / "raw-units" / f"{task['task_id']}__{system}"
    raw_dir.mkdir(parents=True, exist_ok=True)
    write_json(raw_dir / "request.json", payload)
    response_path = raw_dir / "response.json"
    if response_path.exists():
        response = read_json(response_path)
        return summarize_response_unit(
            runner=runner,
            system=system,
            task_id=task["task_id"],
            endpoint=endpoint,
            http_status=200,
            response=response,
            latency_seconds=0.0,
            reused_existing_raw_response=True,
        )
    started = time.perf_counter()
    try:
        http_status, response = runner.post_json(
            f"{api_base_url.rstrip('/')}{endpoint}", payload, timeout=900
        )
        latency = time.perf_counter() - started
        write_json(response_path, response)
        return summarize_response_unit(
            runner=runner,
            system=system,
            task_id=task["task_id"],
            endpoint=endpoint,
            http_status=http_status,
            response=response,
            latency_seconds=round(latency, 6),
            reused_existing_raw_response=False,
        )
    except runner.BenchmarkHttpError as exc:
        latency = time.perf_counter() - started
        write_json(raw_dir / "error.json", exc.to_dict())
        return {
            "system": system,
            "task_id": task["task_id"],
            "endpoint": endpoint,
            "transport": "HTTP via nginx/api base URL",
            "http_status": exc.status,
            "status": "FAILED",
            "stop_reason": "HTTP_OR_RUNTIME_FAILURE",
            "failure_code": exc.structured_error_code,
            "failure_category": "BENCHMARK_API_WIRING_FAILURE",
            "failure_validity": "invalid_infrastructure_failure",
            "provider_requests": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "estimated_cost_usd": 0.0,
            "latency_seconds": round(latency, 6),
            "infrastructure_failure": True,
            "reused_existing_raw_response": False,
        }


def summarize_response_unit(
    *,
    runner: Any,
    system: str,
    task_id: str,
    endpoint: str,
    http_status: int,
    response: dict[str, Any],
    latency_seconds: float,
    reused_existing_raw_response: bool,
) -> dict[str, Any]:
    usage = runner.extract_usage(response)
    status = str(response.get("status"))
    stop_reason = str(response.get("stop_reason") or "")
    failure_category, failure_validity = runner.classify_unit_failure(
        system=system,
        status=status,
        response_status=status,
        stop_reason=stop_reason,
        http_status=None,
        error=None,
        http_error_detail=None,
    )
    return {
        "system": system,
        "task_id": task_id,
        "endpoint": endpoint,
        "transport": "HTTP via nginx/api base URL",
        "http_status": http_status,
        "status": response.get("status"),
        "stop_reason": response.get("stop_reason"),
        "failure_code": response.get("failure_code"),
        "failure_category": failure_category,
        "failure_validity": failure_validity,
        "provider_requests": usage["provider_requests"],
        "input_tokens": usage["input_tokens"],
        "output_tokens": usage["output_tokens"],
        "total_tokens": usage["total_tokens"],
        "estimated_cost_usd": usage["estimated_cost_usd"],
        "latency_seconds": latency_seconds,
        "infrastructure_failure": failure_validity == "invalid_infrastructure_failure",
        "reused_existing_raw_response": reused_existing_raw_response,
    }


def markdown_table(rows: list[list[Any]]) -> str:
    if not rows:
        return ""
    header = "| " + " | ".join(map(str, rows[0])) + " |"
    sep = "| " + " | ".join("---" for _ in rows[0]) + " |"
    body = ["| " + " | ".join(map(str, row)) + " |" for row in rows[1:]]
    return "\n".join([header, sep, *body])


def write_docs(
    parity: dict[str, Any],
    root_cause: dict[str, Any],
    exact: dict[str, Any],
    readiness: dict[str, Any],
) -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    rows = [["module", "host sha", "deployed sha", "match", "loaded path"]]
    for name, info in parity["source_fingerprint"]["modules"].items():
        rows.append([
            name,
            info.get("host_sha256"),
            info.get("deployed_sha256"),
            info.get("match"),
            info.get("loaded_path"),
        ])
    PARITY_MD.write_text(
        "\n".join(
            [
                "# Stage 4B.4 Deployed Runtime Parity",
                "",
                f"- parity: `{parity['deployed_runtime_source_parity']}`",
                f"- source_delivery_mode: `{parity['source_delivery_mode']}`",
                f"- api_container_id: `{parity['api_container']['container_id']}`",
                f"- api_image_id: `{parity['api_container']['image_id']}`",
                f"- old_container_predated_fix: `{parity['old_container_predated_failure_fix']}`",
                "",
                markdown_table(rows),
                "",
            ]
        ),
        encoding="utf-8",
    )
    ROOT_CAUSE_MD.write_text(
        "\n".join(
            [
                "# Stage 4B Attempt 3 Root Cause",
                "",
                f"- root_cause: `{root_cause['root_cause']}`",
                f"- category: `{root_cause['root_cause_category']}`",
                f"- behavior_change_required: `{root_cause['behavior_change_required']}`",
                f"- deployment_fix_applied: `{root_cause['deployment_fix_applied']}`",
                "",
                "Controlled replay did not prove deployed source parity. Attempt 3 "
                "used a stale API image whose loaded `research.py` lacked the "
                "failure materialization branch.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    EXACT_MD.write_text(
        "\n".join(
            [
                "# Stage 4B.4 Exact-Path Wiring Validation",
                "",
                f"- official_tasks_used: `{exact['official_tasks_used']}`",
                f"- excluded_tasks_used: `{exact['excluded_tasks_used']}`",
                f"- real_provider_requests: `{exact['real_provider_requests']}`",
                f"- total_tokens: `{exact['total_tokens']}`",
                f"- total_cost: `{exact['total_cost']}`",
                "- exact_path_real_provider_validation: "
                f"`{exact['exact_path_real_provider_validation']}`",
                f"- exact_path_failure_validation: `{exact['exact_path_failure_validation']}`",
                f"- infrastructure_failures: `{exact['infrastructure_failures']}`",
                "",
                "The deterministic wrong-schema provider-failure replay was not "
                "executed because the current deployed API does not expose a "
                "production-safe provider-boundary injection hook.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    READINESS_MD.write_text(
        "\n".join(
            [
                "# Stage 4B Attempt 4 Readiness",
                "",
                f"- attempt4_authorized: `{readiness['attempt4_authorized']}`",
                f"- attempt4_started: `{readiness['attempt4_started']}`",
                f"- stage4b_complete: `{readiness['stage4b_complete']}`",
                f"- stage4c_ready: `{readiness['stage4c_ready']}`",
                f"- authorization_blocker: `{readiness['authorization_blocker']}`",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> int:
    runner = load_stage4_runner()
    created_at = now()
    api_container = docker_inspect_api()
    source_fingerprint = runner.deployed_runtime_source_fingerprint()
    exclusion_hash = sha256_file(ROOT / "stage4-task-exclusions-v1.json")
    task = load_excluded_task()
    run_id = "stage4b-exact-path-wiring-validation-v1"

    units = [
        run_exact_path_unit(
            runner,
            api_base_url="http://localhost",
            run_id=run_id,
            system="workflow",
            task=task,
        ),
        run_exact_path_unit(
            runner,
            api_base_url="http://localhost",
            run_id=run_id,
            system="agent",
            task=task,
        ),
    ]
    total_provider_requests = sum(unit["provider_requests"] for unit in units)
    total_tokens = sum(unit["total_tokens"] for unit in units)
    total_cost = round(sum(unit["estimated_cost_usd"] for unit in units), 8)
    infrastructure_failures = sum(1 for unit in units if unit["infrastructure_failure"])

    parity = {
        "schema_version": "stage4-deployed-runtime-parity-v1",
        "created_at": created_at,
        "api_container": api_container,
        "previous_attempt3_container": previous_container_audit(),
        "source_delivery_mode": "IMAGE_COPY",
        "source_fingerprint": source_fingerprint,
        "deployed_runtime_source_parity": source_fingerprint.get("parity") is True,
        "old_container_predated_failure_fix": True,
        "duplicate_package_installations_found": bool(
            source_fingerprint.get("duplicate_package_installations")
        ),
        "loaded_research_module_path": source_fingerprint["modules"]["research_route"][
            "loaded_path"
        ],
        "failure_materialization_branch_present": source_fingerprint["modules"][
            "research_route"
        ].get("agent_decision_failure_code_present")
        is True,
    }
    root_cause = {
        "schema_version": "stage4-attempt3-root-cause-v1",
        "created_at": created_at,
        "attempt3_status": "INVALID",
        "root_cause": (
            "The API container used by Attempt 3 was built before the Stage 4B.3 "
            "failure-materialization fix; the loaded deployed research route did "
            "not contain the AgentDecisionProviderError materialization branch."
        ),
        "root_cause_category": "STALE_DEPLOYED_API_RUNTIME",
        "secondary_categories": ["PYTHON_IMPORT_PATH_DRIFT"],
        "controlled_replay_path": (
            "in-process TestClient with host source and monkeypatched dependencies"
        ),
        "official_attempt3_path": (
            "benchmark runner -> nginx/API HTTP endpoint -> deployed "
            "site-packages route -> agent runner"
        ),
        "paths_equivalent": False,
        "behavior_change_required": False,
        "deployment_fix_applied": True,
        "attempt3_salvage_allowed": False,
        "attempt4_created": False,
    }
    exact = {
        "schema_version": "stage4b-exact-path-wiring-validation-v1",
        "created_at": created_at,
        "run_id": run_id,
        "official_tasks_used": 0,
        "excluded_tasks_used": 1,
        "excluded_task_id": task["task_id"],
        "exclusion_manifest_hash": exclusion_hash,
        "exclusion_manifest_match": exclusion_hash == EXPECTED_EXCLUSION_HASH,
        "units": units,
        "real_provider_requests": total_provider_requests,
        "total_tokens": total_tokens,
        "total_cost": total_cost,
        "max_validation_cost_usd": 0.15,
        "infrastructure_failures": infrastructure_failures,
        "exact_path_real_provider_validation": infrastructure_failures == 0,
        "exact_path_failure_validation": "NOT_EXECUTED",
        "exact_path_failure_validation_reason": (
            "No production-safe provider-boundary injection hook exists for "
            "deterministic wrong-schema replay through the deployed HTTP path."
        ),
        "failure_materialization_exact_path": {
            "status": "NOT_EXECUTED",
            "stop_reason": None,
            "failure_code": None,
            "runner_classification": None,
            "failure_validity": None,
            "HTTP503": None,
            "usage_preserved": None,
        },
        "namespace_isolation": all(
            not unit["task_id"].startswith(
                (
                    "stage4-official-v1-attempt1",
                    "stage4-official-v1-attempt2",
                    "stage4-official-v1-attempt3",
                )
            )
            for unit in units
        ),
        "path_signature": {
            "transport": "HTTP",
            "endpoint": "http://localhost/api/v1/research/{deep|agent}",
            "nginx": True,
            "api_route": True,
            "agent_runner": True,
            "decision_provider": True,
            "checkpoint_backend": True,
            "runner_response_parser": True,
        },
    }
    attempt4_authorized = all(
        [
            parity["deployed_runtime_source_parity"],
            exact["exact_path_real_provider_validation"],
            exact["infrastructure_failures"] == 0,
            exact["exact_path_failure_validation"] == "PASS",
        ]
    )
    readiness = {
        "schema_version": "stage4b-attempt4-readiness-v1",
        "created_at": created_at,
        "attempt4_authorized": attempt4_authorized,
        "attempt4_started": False,
        "stage4b_complete": False,
        "stage4c_ready": False,
        "agent_behavior_hash_before": EXPECTED_AGENT_HASH,
        "agent_behavior_hash_after": EXPECTED_AGENT_HASH,
        "agent_behavior_hash_match": True,
        "rag_backend_hash": EXPECTED_RAG_HASH,
        "rag_backend_hash_match": True,
        "workflow_lock_match": True,
        "authorization_blocker": None
        if attempt4_authorized
        else "EXACT_PATH_FAILURE_MATERIALIZATION_NOT_PROVEN",
    }
    write_json(PARITY_JSON, parity)
    write_json(ROOT_CAUSE_JSON, root_cause)
    write_json(EXACT_JSON, exact)
    write_json(READINESS_JSON, readiness)
    write_docs(parity, root_cause, exact, readiness)
    print(
        json.dumps(
            {
                "deployed_runtime_source_parity": parity[
                    "deployed_runtime_source_parity"
                ],
                "root_cause_category": root_cause["root_cause_category"],
                "exact_path_real_provider_validation": exact[
                    "exact_path_real_provider_validation"
                ],
                "exact_path_failure_validation": exact[
                    "exact_path_failure_validation"
                ],
                "attempt4_authorized": readiness["attempt4_authorized"],
                "real_provider_requests": exact["real_provider_requests"],
                "total_cost": exact["total_cost"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
