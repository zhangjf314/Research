from __future__ import annotations

# ruff: noqa: E501
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from paper_research.evaluation.rag_benchmark import read_jsonl
from paper_research.evaluation.rag_official_baseline import (
    BASELINE_CONFIG_HASH,
    DATASET_VERSION,
    DEV_DATASET_HASH,
    FULL_DATASET_HASH,
    SYSTEM_UNDER_TEST_COMMIT,
    TEST_DATASET_HASH,
    sha256_file,
    write_json_artifact,
)

ROOT = Path("data/evaluation/rag-benchmark")
DOCS = Path("docs/rag-benchmark")


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def get_json(url: str) -> tuple[bool, dict[str, Any] | str]:
    try:
        with httpx.Client(timeout=30) as client:
            response = client.get(url)
            response.raise_for_status()
            return True, response.json()
    except Exception as exc:  # noqa: BLE001 - preflight records actual environment failures
        return False, f"{type(exc).__name__}: {exc}"


def redact_public_runtime_payload(value: Any) -> Any:
    """Remove stable secret fingerprints before writing public benchmark artifacts."""
    if isinstance(value, dict):
        return {
            key: ("redacted_public_artifact" if key == "api_key_fingerprint" else redact_public_runtime_payload(item))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_public_runtime_payload(item) for item in value]
    return value


def main() -> int:
    full_path = ROOT / "gold-full-v1.jsonl"
    dev_path = ROOT / "gold-dev-v1.jsonl"
    test_path = ROOT / "gold-test-v1.jsonl"
    config_path = ROOT / "baseline-config-v1.json"
    manifest_path = ROOT / "gold-manifest-v1.json"
    full = read_jsonl(full_path)
    dev = read_jsonl(dev_path)
    test = read_jsonl(test_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dev_ids = {row["question_id"] for row in dev}
    test_ids = {row["question_id"] for row in test}
    health_ok, health = get_json("http://localhost/api/v1/health")
    capabilities_ok, capabilities = get_json("http://localhost/api/v1/capabilities")
    checks = {
        "gold_full_readable": len(full) == 146,
        "gold_dev_readable": len(dev) == 98,
        "gold_test_readable": len(test) == 48,
        "dev_test_disjoint": not (dev_ids & test_ids),
        "dev_test_complete": len(dev_ids | test_ids) == len(full),
        "full_dataset_hash_correct": manifest.get("full_hash") == FULL_DATASET_HASH,
        "dev_dataset_hash_correct": manifest.get("dev_hash") == DEV_DATASET_HASH,
        "test_dataset_hash_correct": manifest.get("test_hash") == TEST_DATASET_HASH,
        "baseline_config_hash_correct": config.get("baseline_config_hash") == BASELINE_CONFIG_HASH,
        "reranker_disabled": not bool(config.get("reranker", {}).get("enabled")),
        "query_rewrite_disabled": not bool(config.get("query_rewrite", {}).get("enabled")),
        "api_health_available": health_ok,
        "api_capabilities_available": capabilities_ok,
    }
    if capabilities_ok and isinstance(capabilities, dict):
        caps = capabilities.get("capabilities") or {}
        checks["llm_provider_configured"] = bool(caps.get("llm", {}).get("configured"))
        checks["embedding_provider_configured"] = bool(caps.get("embedding", {}).get("configured"))
        checks["redis_available"] = (caps.get("redis", {}) or {}).get("status") in {"available", "degraded"}
    if health_ok and isinstance(health, dict):
        components = health.get("components") or {}
        checks["qdrant_available"] = (components.get("qdrant", {}) or {}).get("status") == "up"
        checks["postgresql_available"] = (components.get("postgres", {}) or {}).get("status") == "up"
    payload = {
        "schema_version": "rag-baseline-preflight-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "system_under_test_commit": SYSTEM_UNDER_TEST_COMMIT,
        "benchmark_harness_commit": git_head(),
        "dataset_version": DATASET_VERSION,
        "baseline_config_hash": BASELINE_CONFIG_HASH,
        "full_dataset_hash": FULL_DATASET_HASH,
        "dev_dataset_hash": DEV_DATASET_HASH,
        "test_dataset_hash": TEST_DATASET_HASH,
        "actual_full_sha256": sha256_file(full_path),
        "actual_dev_sha256": sha256_file(dev_path),
        "actual_test_sha256": sha256_file(test_path),
        "checks": checks,
        "config": {
            "top_k": config.get("retrieval", {}).get("top_k"),
            "recall_k": config.get("retrieval", {}).get("recall_k"),
            "rrf_parameters": "current production defaults; frozen",
            "context_size": config.get("context_selection", {}),
            "generation_max_tokens": config.get("generation", {}).get("max_output_tokens"),
            "temperature": config.get("generation", {}).get("temperature"),
            "retry_policy": "current production provider retry policy; not changed for Stage 1C",
        },
        "health": redact_public_runtime_payload(health),
        "capabilities": redact_public_runtime_payload(capabilities) if capabilities_ok else capabilities,
        "status": "PASSED" if all(checks.values()) else "FAILED",
    }
    write_json_artifact(ROOT / "baseline-run-preflight-v1.json", payload)
    lines = [
        "# RAG baseline run preflight v1",
        "",
        f"- status: `{payload['status']}`",
        f"- system_under_test_commit: `{SYSTEM_UNDER_TEST_COMMIT}`",
        f"- benchmark_harness_commit: `{payload['benchmark_harness_commit']}`",
        "",
        "| check | passed |",
        "| --- | --- |",
    ]
    lines.extend(f"| {key} | `{value}` |" for key, value in sorted(checks.items()))
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "baseline-run-preflight-v1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "checks": checks}, ensure_ascii=False))
    return 0 if payload["status"] == "PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
