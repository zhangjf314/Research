"""Stage 13.30 production Full QA gates and immutable preflight artifacts.

This script is intentionally fail-closed. It may prepare manifests and blocked
reports without live calls. A live smoke/full run must be separately authorized
with explicit budget environment variables.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from paper_research.config import Settings
from paper_research.version import __display_version__, __version__

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "evaluation"
DOCS = ROOT / "docs"
ARTIFACTS = ROOT / "artifacts"
GOLD = DATA / "gold-set-v1.jsonl"
RETRIEVAL_GENERALIZATION = DATA / "retrieval-generalization-v1.json"
PROVIDER_HEALTH = DATA / "provider-health-v1.json"

GOLD_DEV_MANIFEST = DATA / "gold-dev-v1-run-manifest.json"
FULL_QA_CONFIG = DATA / "full-qa-production-config-v1.json"
FULL_QA_CONFIG_DOC = DOCS / "full-qa-production-config-v1.md"
RUNTIME_AUDIT_DOC = DOCS / "production-full-qa-runtime-audit.md"
SMOKE_DOC = DOCS / "live-model-smoke-test-v1.md"
SMOKE_JSON = ARTIFACTS / "live-model-smoke-test-v1.json"
FULL_QA_AUDIT_DOC = DOCS / "full-qa-production-audit-v1.md"
FULL_QA_SUMMARY_DOC = DOCS / "full-qa-production-summary-v1.md"
FULL_QA_JSON = DATA / "full-qa-production-v1.json"
FULL_QA_CSV = DATA / "full-qa-production-v1.csv"
FULL_QA_ITEMS = DATA / "full-qa-production-items-v1.jsonl"
FULL_QA_TRACE = ARTIFACTS / "full-qa-production-trace-v1.json"

FULL_QA_BUDGET_VARS = [
    "LIVE_MODEL_CALLS_ENABLED",
    "FULL_QA_MAX_ITEMS",
    "FULL_QA_MAX_INPUT_TOKENS",
    "FULL_QA_MAX_OUTPUT_TOKENS",
    "FULL_QA_MAX_COST_USD",
    "FULL_QA_MAX_TOTAL_SECONDS",
]

DEEP_RESEARCH_BUDGET_VARS = [
    "DEEP_RESEARCH_ENABLED",
    "DEEP_RESEARCH_MAX_INPUT_TOKENS",
    "DEEP_RESEARCH_MAX_OUTPUT_TOKENS",
    "DEEP_RESEARCH_MAX_COST_USD",
    "DEEP_RESEARCH_MAX_TOTAL_SECONDS",
    "DEEP_RESEARCH_MAX_ITERATIONS",
    "DEEP_RESEARCH_MAX_PAPERS",
]

CONTAINER_BUDGET_VARS = FULL_QA_BUDGET_VARS + DEEP_RESEARCH_BUDGET_VARS


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_dataset_hash(rows: list[dict[str, Any]]) -> str:
    body = "\n".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for row in rows
    )
    return _sha256_bytes((body + "\n").encode("utf-8"))


def _git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _git_branch() -> str:
    return subprocess.check_output(
        ["git", "branch", "--show-current"],
        cwd=ROOT,
        text=True,
    ).strip()


def _git_status_short() -> list[str]:
    output = subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True)
    return [line for line in output.splitlines() if line.strip()]


def _env_value(name: str) -> str | None:
    value = os.getenv(name)
    if value not in {None, ""}:
        return value
    env_file = ROOT / ".env"
    if not env_file.exists():
        return None
    prefix = f"{name}="
    for line in env_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith(prefix):
            return stripped.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def budget_status() -> dict[str, Any]:
    values = {name: _env_value(name) for name in FULL_QA_BUDGET_VARS}
    missing = [name for name, value in values.items() if value in {None, ""}]
    live_enabled = (values["LIVE_MODEL_CALLS_ENABLED"] or "").lower() == "true"
    full_ready = live_enabled and not missing
    smoke_allowed = live_enabled
    if not live_enabled:
        status = "BLOCKED_BY_LIVE_MODEL_CALLS_DISABLED"
    elif not full_ready:
        status = "SMOKE_ONLY_BUDGET_INCOMPLETE"
    else:
        status = "FULL_QA_BUDGET_READY"
    return {
        "status": status,
        "live_model_calls_enabled": live_enabled,
        "full_qa_budget_ready": full_ready,
        "smoke_allowed": smoke_allowed,
        "missing_full_qa_budget_vars": missing,
        "full_qa_budget_vars_present": {
            name: values[name] not in {None, ""} for name in FULL_QA_BUDGET_VARS
        },
        "deep_research_budget_vars_present": {
            name: _env_value(name) not in {None, ""} for name in DEEP_RESEARCH_BUDGET_VARS
        },
    }


def container_budget_status() -> dict[str, Any]:
    script = (
        "import json, os; "
        f"names={CONTAINER_BUDGET_VARS!r}; "
        "print(json.dumps({n: (os.getenv(n) not in (None, '')) for n in names}))"
    )
    try:
        output = subprocess.check_output(
            ["docker", "compose", "exec", "-T", "api", "python", "-c", script],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        present = json.loads(output)
    except (subprocess.SubprocessError, json.JSONDecodeError):
        return {
            "status": "UNKNOWN",
            "available": False,
            "live_model_calls_enabled": False,
            "full_qa_budget_ready": False,
            "missing_full_qa_budget_vars": FULL_QA_BUDGET_VARS,
            "vars_present": {},
        }
    missing = [name for name in FULL_QA_BUDGET_VARS if not present.get(name, False)]
    live_enabled = present.get("LIVE_MODEL_CALLS_ENABLED", False)
    full_ready = live_enabled and not missing
    if full_ready:
        status = "FULL_QA_BUDGET_READY"
    elif live_enabled:
        status = "SMOKE_ONLY_BUDGET_INCOMPLETE"
    else:
        status = "BLOCKED_BY_CONTAINER_LIVE_MODEL_CALLS_DISABLED"
    return {
        "status": status,
        "available": True,
        "live_model_calls_enabled": live_enabled,
        "full_qa_budget_ready": full_ready,
        "missing_full_qa_budget_vars": missing,
        "vars_present": present,
    }


def build_gold_manifest() -> dict[str, Any]:
    rows = _read_jsonl(GOLD)
    statuses = Counter(str(row.get("review_status")) for row in rows)
    answerable_count = sum(bool(row.get("answerable")) for row in rows)
    unanswerable_count = len(rows) - answerable_count
    return {
        "dataset_id": "gold-dev-v1",
        "dataset_version": "gold-set-v1-human-reviewed-2026-07-13",
        "role": "人工审核的内部开发评测集",
        "blind": False,
        "total_count": len(rows),
        "approved_count": statuses.get("approved", 0),
        "answerable_count": answerable_count,
        "unanswerable_count": unanswerable_count,
        "review_status_counts": dict(sorted(statuses.items())),
        "dataset_sha256": _canonical_dataset_hash(rows),
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "git_branch": _git_branch(),
        "application_version": __version__,
        "display_version": __display_version__,
        "strong_generalization_claim_allowed": False,
    }


def _safe_secret(value: str | None) -> dict[str, Any]:
    if not value:
        return {"present": False, "length": 0, "sha256_prefix": None}
    return {
        "present": True,
        "length": len(value),
        "sha256_prefix": _sha256_bytes(value.encode("utf-8"))[:8],
    }


def _http_json(url: str) -> dict[str, Any] | None:
    try:
        response = httpx.get(url, timeout=5)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError:
        return None


def runtime_audit(settings: Settings, budget: dict[str, Any]) -> dict[str, Any]:
    health = _http_json("http://localhost/api/v1/health")
    capabilities = _http_json("http://localhost/api/v1/capabilities")
    provider_health = (
        json.loads(PROVIDER_HEALTH.read_text(encoding="utf-8"))
        if PROVIDER_HEALTH.exists()
        else None
    )
    base_host = urlparse(settings.llm_base_url or "").hostname
    container_budget = (
        capabilities.get("stage13_30_budget")
        if capabilities and capabilities.get("stage13_30_budget")
        else container_budget_status()
    )
    if not budget["live_model_calls_enabled"]:
        full_qa_status = "BLOCKED_BY_LIVE_MODEL_CALLS_DISABLED"
    elif not container_budget["full_qa_budget_ready"]:
        full_qa_status = "BLOCKED_BY_CONTAINER_BUDGET_NOT_INJECTED"
    else:
        full_qa_status = "READY_FOR_SMOKE_OR_FULL_QA"
    return {
        "schema_version": "production-full-qa-runtime-audit-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "git_branch": _git_branch(),
        "git_commit": _git_commit(),
        "host_profile": settings.app_profile,
        "container_profile": capabilities.get("profile") if capabilities else None,
        "health_status": health.get("status") if health else "unavailable",
        "capabilities_overall": capabilities.get("overall") if capabilities else "unavailable",
        "embedding_provider": settings.embedding_provider,
        "embedding_model": settings.embedding_model,
        "embedding_dimension": settings.embedding_dimensions,
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
        "llm_base_url_host": base_host,
        "llm_api_key": _safe_secret(settings.llm_api_key),
        "rerank_provider": settings.rerank_provider,
        "rerank_model": settings.rerank_model,
        "rerank_enabled": settings.rerank_enabled,
        "qdrant_logical_collection": settings.qdrant_collection,
        "production_collection": settings.production_collection,
        "baseline_collection": settings.baseline_collection,
        "provider_preflight": provider_health,
        "budget": budget,
        "container_budget": container_budget,
        "silent_fallback_detected": settings.llm_provider == "template"
        or settings.embedding_provider == "hash",
        "full_qa_status": full_qa_status,
    }


def full_qa_config(
    settings: Settings,
    manifest: dict[str, Any],
    budget: dict[str, Any],
) -> dict[str, Any]:
    retrieval = json.loads(RETRIEVAL_GENERALIZATION.read_text(encoding="utf-8"))
    body = {
        "schema_version": "full-qa-production-config-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "git_branch": _git_branch(),
        "git_commit": _git_commit(),
        "working_tree_status": _git_status_short(),
        "package_version": __version__,
        "runtime_version": __version__,
        "openapi_version": __version__,
        "dataset_version": manifest["dataset_version"],
        "dataset_hash": manifest["dataset_sha256"],
        "retrieval_config_hash": _sha256_bytes(
            json.dumps(
                retrieval.get("fusion_parameters", {}),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ),
        "embedding_provider": settings.embedding_provider,
        "embedding_model": settings.embedding_model,
        "embedding_dimension": settings.embedding_dimensions,
        "qdrant_logical_collection": settings.qdrant_collection,
        "qdrant_physical_collection": settings.production_collection,
        "chunk_strategy": "structural",
        "dense_top_k": 12,
        "sparse_top_k": 12,
        "fusion_parameters": retrieval.get("fusion_parameters", {}),
        "rerank_provider": settings.rerank_provider,
        "rerank_model": settings.rerank_model,
        "rerank_enabled": settings.rerank_enabled,
        "rerank_candidate_count": settings.rerank_input_k,
        "final_top_k": 10,
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
        "temperature": settings.llm_temperature,
        "max_tokens": settings.llm_max_output_tokens,
        "prompt_version": settings.prompt_version,
        "claim_validator_version": "claim-evidence-validator-v1",
        "random_seed": 42,
        "budget": budget,
        "full_qa_config_sha256": None,
    }
    signature_body = {key: value for key, value in body.items() if key != "full_qa_config_sha256"}
    body["full_qa_config_sha256"] = _sha256_bytes(
        json.dumps(signature_body, ensure_ascii=False, sort_keys=True).encode("utf-8")
    )
    return body


def write_blocked_outputs(
    manifest: dict[str, Any],
    runtime: dict[str, Any],
    config: dict[str, Any],
) -> None:
    budget = runtime["budget"]
    container_budget = runtime["container_budget"]
    if runtime["full_qa_status"].startswith("BLOCKED"):
        status = runtime["full_qa_status"]
    elif not budget["full_qa_budget_ready"]:
        status = "SMOKE_ONLY_BUDGET_INCOMPLETE"
    elif not container_budget["full_qa_budget_ready"]:
        status = "BLOCKED_BY_CONTAINER_BUDGET_NOT_INJECTED"
    else:
        status = "READY_NOT_EXECUTED"
    smoke = {
        "schema_version": "live-model-smoke-test-v1",
        "status": status,
        "sample_id": None,
        "real_model_called": False,
        "reason": (
            "live calls are not enabled"
            if not budget["live_model_calls_enabled"]
            else None
        ),
        "budget": budget,
        "container_budget": container_budget,
    }
    full = {
        "schema_version": "full-qa-production-v1",
        "status": status if status != "READY_NOT_EXECUTED" else "NOT_RUN",
        "dataset": "gold-dev-v1",
        "total": manifest["total_count"],
        "approved": manifest["approved_count"],
        "answerable": manifest["answerable_count"],
        "unanswerable": manifest["unanswerable_count"],
        "completed_count": 0,
        "failed_count": 0,
        "production_full_qa_gate": "BLOCKED_BY_BUDGET"
        if status != "READY_NOT_EXECUTED"
        else "NOT_RUN",
        "ready_for_production_deep_research": False,
        "strong_generalization_claim_allowed": False,
        "budget": budget,
        "container_budget": container_budget,
        "config_sha256": config["full_qa_config_sha256"],
    }
    _write_json(SMOKE_JSON, smoke)
    _write_json(FULL_QA_JSON, full)
    _write_json(FULL_QA_TRACE, {"events": [], "status": full["status"]})
    FULL_QA_ITEMS.write_text("", encoding="utf-8")
    FULL_QA_CSV.write_text(
        "run_id,sample_id,status,input_tokens,output_tokens,total_tokens,cost_usd\n",
        encoding="utf-8",
    )
    _write_docs(manifest, runtime, config, smoke, full)


def _write_docs(
    manifest: dict[str, Any],
    runtime: dict[str, Any],
    config: dict[str, Any],
    smoke: dict[str, Any],
    full: dict[str, Any],
) -> None:
    RUNTIME_AUDIT_DOC.write_text(
        "# Production Full QA Runtime Audit\n\n"
        f"- Host profile: `{runtime['host_profile']}`\n"
        f"- Container profile: `{runtime['container_profile']}`\n"
        f"- Health: `{runtime['health_status']}`\n"
        f"- Capabilities: `{runtime['capabilities_overall']}`\n"
        f"- Embedding: `{runtime['embedding_provider']}/{runtime['embedding_model']}`\n"
        f"- LLM: `{runtime['llm_provider']}/{runtime['llm_model']}`\n"
        f"- LLM base host: `{runtime['llm_base_url_host']}`\n"
        f"- LLM API key present: `{runtime['llm_api_key']['present']}`\n"
        f"- LLM API key length: `{runtime['llm_api_key']['length']}`\n"
        f"- LLM API key SHA-256 prefix: `{runtime['llm_api_key']['sha256_prefix']}`\n"
        f"- Reranker enabled: `{runtime['rerank_enabled']}`\n"
        f"- Silent fallback detected: `{runtime['silent_fallback_detected']}`\n"
        f"- Full QA status: `{runtime['full_qa_status']}`\n"
        f"- Container budget status: `{runtime['container_budget']['status']}`\n",
        encoding="utf-8",
    )
    FULL_QA_CONFIG_DOC.write_text(
        "# Full QA Production Config v1\n\n"
        f"- Dataset: `{manifest['dataset_id']}` / `{manifest['dataset_version']}`\n"
        f"- Dataset hash: `{manifest['dataset_sha256']}`\n"
        f"- Config hash: `{config['full_qa_config_sha256']}`\n"
        f"- LLM: `{config['llm_provider']}/{config['llm_model']}`\n"
        f"- Embedding: `{config['embedding_provider']}/{config['embedding_model']}`\n"
        f"- Reranker enabled: `{config['rerank_enabled']}`\n"
        f"- Strong generalization claim allowed: `False`\n",
        encoding="utf-8",
    )
    SMOKE_DOC.write_text(
        "# Live Model Smoke Test v1\n\n"
        f"- Status: `{smoke['status']}`\n"
        f"- Real model called: `{smoke['real_model_called']}`\n"
        "- No API key is printed or persisted.\n",
        encoding="utf-8",
    )
    FULL_QA_AUDIT_DOC.write_text(
        "# Full QA Production Audit v1\n\n"
        f"- Status: `{full['status']}`\n"
        f"- Production Full QA gate: `{full['production_full_qa_gate']}`\n"
        f"- Dataset: `gold-dev-v1`, approved `{manifest['approved_count']}`/"
        f"`{manifest['total_count']}`\n"
        "- This is an internal development evaluation set, not a blind benchmark.\n",
        encoding="utf-8",
    )
    FULL_QA_SUMMARY_DOC.write_text(
        "# Full QA Production Summary v1\n\n"
        f"- Completed count: `{full['completed_count']}`\n"
        f"- Failed count: `{full['failed_count']}`\n"
        f"- Ready for Production Deep Research: "
        f"`{full['ready_for_production_deep_research']}`\n"
        "- LLM Judge metrics: `NOT_RUN`\n"
        "- Unanswerable sample count: `2`; refusal metrics are internal diagnostics only.\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    del args
    settings = Settings()
    budget = budget_status()
    manifest = build_gold_manifest()
    runtime = runtime_audit(settings, budget)
    config = full_qa_config(settings, manifest, budget)
    _write_json(GOLD_DEV_MANIFEST, manifest)
    _write_json(FULL_QA_CONFIG, config)
    write_blocked_outputs(manifest, runtime, config)
    print(
        json.dumps(
            {
                "gold_dev_manifest": str(GOLD_DEV_MANIFEST.relative_to(ROOT)),
                "runtime_audit": str(RUNTIME_AUDIT_DOC.relative_to(ROOT)),
                "full_qa_status": runtime["full_qa_status"],
                "budget_status": budget["status"],
                "ready_for_full_qa_run": budget["full_qa_budget_ready"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
