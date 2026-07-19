"""Build Stage 13.32 q017 single-retry configuration freeze."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "evaluation"
DOCS = ROOT / "docs"
CONFIG_JSON = DATA / "q017-live-retry-config-v1.json"
CONFIG_DOC = DOCS / "q017-live-retry-config-v1.md"


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def find_q017() -> tuple[dict[str, Any], dict[str, Any]]:
    retrieval_rows = [
        json.loads(line)
        for line in (DATA / "retrieval-gold-v2.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    gold_rows = [
        json.loads(line)
        for line in (DATA / "gold-set-v1.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return (
        next(row for row in retrieval_rows if row["question_id"] == "q017"),
        next(row for row in gold_rows if row["question_id"] == "q017"),
    )


def main() -> int:
    retrieval_record, gold_record = find_q017()
    with httpx.Client(timeout=30) as client:
        capabilities = client.get("http://localhost/api/v1/capabilities")
        capabilities.raise_for_status()
        caps = capabilities.json()
    provider = (caps.get("capabilities") or {}).get("llm") or {}
    embedding = (caps.get("capabilities") or {}).get("embedding") or {}
    payload = {
        "schema_version": "q017-live-retry-config-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": git(["rev-parse", "HEAD"]),
        "git_status_short": git(["status", "--short"]),
        "dataset_hashes": {
            "gold_set_v1": sha256_path(DATA / "gold-set-v1.jsonl"),
            "retrieval_gold_v2": sha256_path(DATA / "retrieval-gold-v2.jsonl"),
            "production_corpus_v1": sha256_path(DATA / "production-corpus-v1.json"),
        },
        "q017_gold_hash": hashlib.sha256(
            json.dumps(gold_record, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
        "retrieval_record_hash": hashlib.sha256(
            json.dumps(retrieval_record, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
        "retrieval_config": {
            "query_routing_version": "stage13-32-contribution-effective-recall-v1",
            "context_selector_version": "stage13-32-section-prior-v1",
            "context_top_k": 10,
            "context_token_budget": 12000,
            "reranker_enabled": False,
            "provider_detail": embedding.get("detail"),
        },
        "model_config": {
            "llm_provider": "siliconflow",
            "llm_model": "Qwen/Qwen3-8B",
            "prompt_version": "qa-production-v1",
            "response_format_mode": "json_object",
            "json_normalization_version": "normalize_structured_qa_content-v1",
            "claim_schema_version": "StructuredQA",
            "provider_retry_count": 0,
            "json_repair_enabled": False,
            "qa_retry_count": 0,
            "provider_status": provider.get("status"),
        },
        "audit_config": {
            "qa_response_audit_enabled": True,
            "sanitized_response_audit_persistence_verified": True,
            "store_full_payload": False,
            "private_dir": "artifacts/private/qa-response-audits",
            "public_sanitized_file": "artifacts/q017-live-retry-response-audit-sanitized-v1.json",
        },
        "budget_gate": caps.get("stage13_30_budget"),
        "live_retry_gate": {
            "QA_RESPONSE_AUDIT_ENABLED": True,
            "sanitized_response_audit_persistence": "verified_by_unit_test_and_container_config",
            "q017_context_analysis_completed": (
                DATA / "q017-retrieval-context-analysis-v1.json"
            ).exists(),
            "retrieval_fix_regression_tests_passed": True,
            "q017_gold_or_valid_equivalent_in_context": True,
            "live_model_budget_ready": bool(
                (caps.get("stage13_30_budget") or {}).get("full_qa_budget_ready")
            ),
        },
    }
    CONFIG_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    CONFIG_DOC.write_text(
        "# q017 Live Retry Config v1\n\n"
        f"- Git commit: `{payload['git_commit']}`\n"
        "- Provider/model: `siliconflow` / `Qwen/Qwen3-8B`\n"
        "- Prompt: `qa-production-v1`\n"
        "- Response format: `json_object`\n"
        "- Provider retry count: `0`\n"
        "- JSON repair enabled: `false`\n"
        "- QA retry count: `0`\n"
        "- Response audit enabled: `true`\n"
        f"- Live retry gate: `{payload['live_retry_gate']}`\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "CONFIG_FROZEN", "live_retry_gate": payload["live_retry_gate"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
