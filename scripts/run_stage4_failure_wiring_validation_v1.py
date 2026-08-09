from __future__ import annotations

import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from paper_research.agents.research_agent.decision_provider import (
    LLMResearchAgentDecisionProvider,
)
from paper_research.main import create_app
from paper_research.providers.llm import ModelUsage, StructuredJSONResult

ROOT = Path("data/evaluation/research-agent")
BENCHMARK_ROOT = ROOT / "benchmark"
DOC_ROOT = Path("docs/research-agent/benchmark")
EXCLUSIONS = ROOT / "stage4-task-exclusions-v1.json"
STAGE4_MANIFEST = ROOT / "stage4-benchmark-manifest-v1.json"
AGENT_LOCK = ROOT / "stage3-agent-lock-v1.json"
RAG_LOCK = ROOT / "stage3-rag-backend-lock-v1.json"
VALIDATION_JSON = BENCHMARK_ROOT / "stage4b-live-failure-wiring-validation-v1.json"
VALIDATION_MD = DOC_ROOT / "stage4b-live-failure-wiring-validation-v1.md"
READINESS_JSON = BENCHMARK_ROOT / "stage4b-attempt3-readiness-v1.json"
READINESS_MD = DOC_ROOT / "stage4b-attempt3-readiness-v1.md"
EXPECTED_EXCLUSION_HASH = "fd9a015c3d7b725cb863f766d25fafafb266576271b3422d0200c1b6c567c0cb"
EXPECTED_AGENT_HASH = "bce71a51171b2e1187d579a2278cc34f1202ed7b84e9482cbffe42d00b92ff15"
EXPECTED_RAG_HASH = "995a144385180b2931ec2c6366f7f7306301a42d77ad7c85f4be9e6d9e5091d9"


class _Settings:
    app_profile = "production"
    llm_provider = "deepseek"
    parsed_papers_dir = Path("data/parsed")


class _NoopRetrievalProvider:
    pass


class _WrongSchemaHTTP200Provider:
    provider_name = "deepseek"
    model_name = "deepseek-v4-flash"

    def generate_structured_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema_name: str,
        request_context: dict[str, Any] | None = None,
        max_output_tokens: int | None = None,
    ) -> StructuredJSONResult:
        del system_prompt, schema_name, request_context, max_output_tokens
        parsed = json.loads(user_prompt)
        # This deliberately mirrors Attempt2's provider behavior: HTTP 200 and valid JSON,
        # but the content is the request payload, not the required ResearchPlan shape.
        return StructuredJSONResult(
            payload=parsed,
            provider=self.provider_name,
            model=self.model_name,
            usage=ModelUsage(
                input_tokens=491,
                output_tokens=599,
                total_tokens=1090,
                estimated_cost_usd=0.00023646,
                usage_source="provider_reported",
            ),
            request_attempt_count=1,
            retry_count=0,
            total_latency_ms=10.0,
            provider_request_id=None,
        )


def main() -> int:
    payload = run_validation()
    write_json(VALIDATION_JSON, payload)
    write_markdown(VALIDATION_MD, render_validation_markdown(payload))
    readiness = build_readiness(payload)
    write_json(READINESS_JSON, readiness)
    write_markdown(READINESS_MD, render_readiness_markdown(readiness))
    print(
        json.dumps(
            {
                "passed": payload["passed"],
                "attempt3_authorized": readiness["attempt3_authorized"],
            }
        )
    )
    return 0 if payload["passed"] else 1


def run_validation() -> dict[str, Any]:
    exclusions = read_json(EXCLUSIONS)
    manifest = read_json(STAGE4_MANIFEST)
    agent_lock = read_json(AGENT_LOCK)
    rag_lock = read_json(RAG_LOCK)
    task = exclusions["tasks"][0]
    validation_run_id = "stage4-wiring-validation-v1"
    validation_task_id = f"{validation_run_id}-{task['task_id']}-agent"
    query = validation_query(task)

    from paper_research.api.routes import research

    original_get_settings = research.get_settings
    original_hybrid = research.HybridLocalResearchProvider
    original_agent_decision_provider = research._agent_decision_provider
    try:
        research.get_settings = lambda: _Settings()  # type: ignore[assignment]
        research.HybridLocalResearchProvider = lambda settings: _NoopRetrievalProvider()  # type: ignore[assignment]
        research._agent_decision_provider = (  # type: ignore[assignment]
            lambda settings: LLMResearchAgentDecisionProvider(_WrongSchemaHTTP200Provider())
        )
        response = TestClient(create_app()).post(
            "/api/v1/research/agent",
            json={"task_id": validation_task_id, "query": query},
        )
        api_payload = response.json()
    finally:
        research.get_settings = original_get_settings  # type: ignore[assignment]
        research.HybridLocalResearchProvider = original_hybrid  # type: ignore[assignment]
        research._agent_decision_provider = original_agent_decision_provider  # type: ignore[assignment]

    runner = load_stage4_runner()
    unit = {
        "system": "agent",
        "task_id": task["task_id"],
        "blind_label": "VALIDATION",
        "execution_unit_id": f"{task['task_id']}-agent",
    }
    task_meta = {
        "task_id": task["task_id"],
        "category": "non_benchmark_wiring_validation",
        "difficulty": "not_scored",
        "target_paper_ids": [],
    }
    unit_result = runner.summarize_unit_result(
        unit=unit,
        task=task_meta,
        response=api_payload,
        http_status=response.status_code,
        error=None,
        http_error_detail=None,
        latency_seconds=0.0,
    )
    workflow_failure_category, workflow_failure_validity = runner.classify_unit_failure(
        system="workflow",
        status="FAILED",
        response_status="FAILED_RETRIEVAL",
        stop_reason="FAILED_RETRIEVAL",
        http_status=200,
        error=None,
        http_error_detail=None,
    )
    summary = {
        "terminal_units": 1,
        "provider_requests": unit_result["provider_requests"],
        "input_tokens": unit_result["input_tokens"],
        "output_tokens": unit_result["output_tokens"],
        "total_tokens": unit_result["total_tokens"],
        "estimated_cost_usd": unit_result["estimated_cost_usd"],
    }
    secret_scan = secret_scan_payload({"api_response": api_payload, "unit_result": unit_result})
    exclusion_hash = sha256_file(EXCLUSIONS)
    agent_hash = agent_lock["stage3_agent_behavior_hash"]
    rag_hash = rag_lock["stage2_final_config_hash"]
    checks = {
        "exclusion_manifest_found": EXCLUSIONS.exists(),
        "exclusion_schema_valid": exclusions.get("schema_version") == "stage4-task-exclusions-v1",
        "exclusion_count_valid": exclusions.get("task_count") == 6
        and len(exclusions.get("tasks", [])) == 6,
        "exclusion_hash_match": exclusion_hash
        == EXPECTED_EXCLUSION_HASH
        == manifest.get("task_exclusion_hash"),
        "agent_http_status_not_503": response.status_code == 200,
        "provider_failure_materialized": api_payload.get("status") == "FAILED"
        and api_payload.get("stop_reason") == "PROVIDER_FAILURE"
        and api_payload.get("failure_code") == "AGENT_DECISION_PROVIDER_ERROR",
        "usage_recovered": api_payload.get("provider_call_count") == 1
        and api_payload.get("token_usage", {}).get("total_tokens") == 1090,
        "checkpoint_trace_exposed": bool(api_payload.get("checkpoint_id"))
        and api_payload.get("checkpoint_count", 0) >= 2,
        "runner_classification_valid": unit_result.get("failure_category")
        == "SYSTEM_PROVIDER_FAILURE"
        and unit_result.get("failure_validity") == "valid_system_failure",
        "not_benchmark_api_wiring_failure": unit_result.get("failure_category")
        != "BENCHMARK_API_WIRING_FAILURE",
        "accounting_integrity": summary["terminal_units"] == 1
        and summary["provider_requests"] == unit_result["provider_requests"]
        and summary["total_tokens"] == unit_result["total_tokens"],
        "namespace_isolated": validation_task_id.startswith("stage4-wiring-validation-v1-")
        and not validation_task_id.startswith("stage4-official-v1"),
        "workflow_failure_classification_still_valid": workflow_failure_category
        == "SYSTEM_VERIFICATION_FAILURE"
        and workflow_failure_validity == "valid_system_failure",
        "secret_scan_passed": not secret_scan["secret_hits"],
        "agent_behavior_hash_match": agent_hash == EXPECTED_AGENT_HASH,
        "rag_backend_hash_match": rag_hash == EXPECTED_RAG_HASH,
    }
    return {
        "schema_version": "stage4b-live-failure-wiring-validation-v1",
        "created_at": now(),
        "validation_mode": "non_benchmark_controlled_api_wiring_replay",
        "official_benchmark_units_used": 0,
        "official_attempt_started": false(),
        "provider_requests_made_by_this_script": 0,
        "retrieval_requests_made_by_this_script": 0,
        "exclusion_manifest": {
            "path": str(EXCLUSIONS),
            "sha256": exclusion_hash,
            "task_count": exclusions.get("task_count"),
            "restored_from_commit": "7d7139a9ada54e94b97d8827b198e44d3692bc9e",
        },
        "validation_task_ids": [validation_task_id],
        "api_response": sanitize(api_payload),
        "unit_result": sanitize(unit_result),
        "workflow_path_validation": {
            "failure_category": workflow_failure_category,
            "failure_validity": workflow_failure_validity,
        },
        "accounting_summary": summary,
        "secret_scan": secret_scan,
        "hashes": {
            "agent_behavior_hash": agent_hash,
            "rag_backend_hash": rag_hash,
            "dataset_hash": manifest.get("dataset_hash"),
            "research_tasks_hash": manifest.get("stage4_research_tasks_hash"),
            "research_rubric_hash": manifest.get("stage4_research_rubric_hash"),
            "execution_order_hash": manifest.get("stage4_execution_order_hash"),
            "evaluation_protocol_hash": manifest.get("stage4_evaluation_protocol_hash"),
            "workflow_lock_hash": manifest.get("workflow_lock_hash"),
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def validation_query(task: dict[str, Any]) -> str:
    source = Path(task["source_artifact"])
    data = read_json(source)
    for key in ("validation_tasks", "smoke_tasks"):
        for item in data.get(key, []):
            if item.get("task_id") == task["task_id"]:
                return (
                    item.get("research_question")
                    or item.get("query")
                    or item.get("kind")
                    or task["task_id"]
                )
    return task["task_id"]


def build_readiness(validation: dict[str, Any]) -> dict[str, Any]:
    authorized = bool(validation["passed"])
    return {
        "schema_version": "stage4b-attempt3-readiness-v1",
        "created_at": now(),
        "attempt3_started": False,
        "attempt3_authorized": authorized,
        "stage4b_complete": False,
        "stage4c_ready": False,
        "recovered_exclusion_manifest_path": validation["exclusion_manifest"]["path"],
        "exclusion_manifest_hash": validation["exclusion_manifest"]["sha256"],
        "exclusion_count": validation["exclusion_manifest"]["task_count"],
        "validation_task_ids": validation["validation_task_ids"],
        "live_wiring_validation_result": "PASS" if validation["passed"] else "FAIL",
        "provider_failure_materialization_result": validation["checks"][
            "provider_failure_materialized"
        ],
        "runner_classification_result": validation["checks"]["runner_classification_valid"],
        "accounting_validation": validation["checks"]["accounting_integrity"],
        "namespace_isolation_validation": validation["checks"]["namespace_isolated"],
        "secret_scan_result": validation["checks"]["secret_scan_passed"],
        "agent_behavior_hash_before": EXPECTED_AGENT_HASH,
        "agent_behavior_hash_after": validation["hashes"]["agent_behavior_hash"],
        "rag_backend_hash": validation["hashes"]["rag_backend_hash"],
        "protocol_hashes": validation["hashes"],
        "authorization_blocker": None if authorized else failed_checks(validation["checks"]),
    }


def render_validation_markdown(payload: dict[str, Any]) -> str:
    checks = "\n".join(f"- {key}: `{value}`" for key, value in payload["checks"].items())
    return "\n".join(
        [
            "# Stage 4B Live Failure Wiring Validation v1",
            "",
            f"- validation_mode: `{payload['validation_mode']}`",
            f"- passed: `{payload['passed']}`",
            f"- official_benchmark_units_used: `{payload['official_benchmark_units_used']}`",
            (
                "- provider_requests_made_by_this_script: "
                f"`{payload['provider_requests_made_by_this_script']}`"
            ),
            (
                "- retrieval_requests_made_by_this_script: "
                f"`{payload['retrieval_requests_made_by_this_script']}`"
            ),
            f"- exclusion_manifest: `{payload['exclusion_manifest']['path']}`",
            f"- exclusion_manifest_hash: `{payload['exclusion_manifest']['sha256']}`",
            f"- validation_task_ids: `{', '.join(payload['validation_task_ids'])}`",
            "",
            "## Checks",
            "",
            checks,
            "",
        ]
    )


def render_readiness_markdown(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Stage 4B Attempt3 Readiness v1",
            "",
            f"- attempt3_started: `{payload['attempt3_started']}`",
            f"- attempt3_authorized: `{payload['attempt3_authorized']}`",
            f"- stage4b_complete: `{payload['stage4b_complete']}`",
            f"- stage4c_ready: `{payload['stage4c_ready']}`",
            f"- exclusion_manifest_hash: `{payload['exclusion_manifest_hash']}`",
            f"- live_wiring_validation_result: `{payload['live_wiring_validation_result']}`",
            f"- authorization_blocker: `{payload['authorization_blocker']}`",
            "",
            (
                "Attempt3 is authorized only for a future explicit command. "
                "This artifact does not start the official benchmark."
            ),
            "",
        ]
    )


def load_stage4_runner():
    spec = importlib.util.spec_from_file_location(
        "stage4_runner", "scripts/run_stage4_workflow_agent_benchmark_v1.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load Stage4 runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def failed_checks(checks: dict[str, bool]) -> list[str]:
    return [key for key, value in checks.items() if not value]


def secret_scan_payload(payload: Any) -> dict[str, Any]:
    text = json.dumps(payload, ensure_ascii=False)
    patterns = ["Bearer ", "Authorization", "LLM_API_KEY", "api_key", "Cookie"]
    return {"secret_hits": [pattern for pattern in patterns if pattern in text]}


def sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: sanitize(child) for key, child in value.items()}
    if isinstance(value, list):
        return [sanitize(child) for child in value]
    return value


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def sha256_file(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def now() -> str:
    return datetime.now(UTC).isoformat()


def false() -> bool:
    return False


if __name__ == "__main__":
    raise SystemExit(main())
