"""Inspect or smoke-test the Deep Research synthesize provider path.

The script never mutates an existing Deep Research run.  ``inspect`` performs
only local checks.  ``provider-smoke`` sends one minimal chat-completions
request through the shared provider factory and stores a sanitized result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.run_deep_research_smoke_v1 as smoke  # noqa: E402
from paper_research.agents.bounded_smoke import (  # noqa: E402
    conservative_token_estimate,
    smoke_configuration,
)
from paper_research.config import Settings  # noqa: E402
from paper_research.providers.factory import build_llm_provider  # noqa: E402
from paper_research.providers.llm import (  # noqa: E402
    classify_provider_exception,
    redact_sensitive_text,
)

DEFAULT_OUTPUT = ROOT / "artifacts" / "deep-research-synthesize-smoke-v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mode", choices=("inspect", "provider-smoke"), required=True)
    parser.add_argument("--allow-live-call", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_run(run_id: str) -> dict[str, Any]:
    candidates = [
        ROOT / "artifacts" / "deepseek-production-deep-research-v1" / run_id,
        ROOT / "data" / "evaluation" / "deep-research-smoke-v1" / "runs" / run_id,
    ]
    for path in candidates:
        if path.exists():
            return {
                "run_dir": str(path),
                "result": json.loads((path / "result.json").read_text(encoding="utf-8")),
                "metadata": json.loads(
                    (path / "run-metadata.json").read_text(encoding="utf-8")
                ),
                "checkpoint": json.loads(
                    (path / "checkpoint-summary.json").read_text(encoding="utf-8")
                ),
            }
    raise FileNotFoundError(f"run not found: {run_id}")


def provider_config(settings: Settings, provider: Any) -> dict[str, Any]:
    key = settings.llm_api_key or ""
    base = settings.llm_base_url or ""
    return {
        "app_profile": settings.app_profile,
        "llm_provider": settings.llm_provider,
        "llm_provider_name": settings.llm_provider_name or settings.llm_provider,
        "llm_model": settings.llm_model,
        "base_url_host": base.split("://", 1)[-1].split("/", 1)[0],
        "api_key_present": bool(key),
        "api_key_length": len(key),
        "api_key_sha256_8": hashlib.sha256(key.encode()).hexdigest()[:8] if key else None,
        "temperature": settings.llm_temperature,
        "max_output_tokens": settings.llm_max_output_tokens,
        "timeout_seconds": settings.llm_timeout_seconds,
        "max_retries": settings.llm_max_retries,
        "response_format": settings.llm_response_format,
        "thinking_enabled": settings.llm_thinking_enabled,
        "stream": settings.llm_stream,
        "factory_provider": provider.provider_name,
        "factory_model": provider.model_name,
        "client_class": type(provider.client).__name__,
        "sync_async": "sync",
        "trust_env": "httpx_default",
        "ssl_verify": "httpx_default",
    }


def retry_settings(**updates: Any) -> Settings:
    fixed = {"deep_research_max_cost_usd": Decimal("0.20")}
    fixed.update(updates)
    return Settings(**fixed)


def inspect_payload(run_id: str) -> dict[str, Any]:
    settings = retry_settings()
    provider = build_llm_provider(settings.model_copy(update={"llm_max_retries": 0}))
    rows = smoke.load_jsonl(smoke.MANIFEST)
    sample = next(row for row in rows if row["question_id"] == "q003")
    contexts = smoke.exact_contexts()[sample["question_id"]]
    policy, limits = smoke_configuration(settings)
    run = read_run(run_id)
    prompt_source = "qa-production-v1"
    return {
        "schema_version": "deep-research-synthesize-smoke-v1",
        "mode": "inspect",
        "checked_at": datetime.now(UTC).isoformat(),
        "run_id": run_id,
        "llm_called": False,
        "run": run,
        "provider_config": provider_config(settings, provider),
        "prompt_version": prompt_source,
        "prompt_version_hash": sha256_text(prompt_source),
        "context_count": len(contexts),
        "context_token_estimate": conservative_token_estimate(sample["question"], contexts),
        "budget": {
            "billing_mode": policy.mode,
            "cost_basis": policy.cost_basis,
            "max_cost": str(policy.max_cost),
            "tokens_per_query": limits.tokens_per_query,
            "requests_per_query": limits.requests_per_query,
        },
    }


def provider_smoke_payload(run_id: str) -> dict[str, Any]:
    settings = retry_settings(llm_max_retries=0, llm_max_output_tokens=128)
    provider = build_llm_provider(settings)
    started = time.perf_counter()
    payload = {
        "model": provider.model_name,
        "messages": [
            {
                "role": "system",
                "content": "Return a compact JSON object only.",
            },
            {
                "role": "user",
                "content": (
                    "Return {\"ok\": true, "
                    "\"purpose\": \"deep_research_synthesize_smoke\"}."
                ),
            },
        ],
        "temperature": 0,
        "max_tokens": 128,
        "stream": False,
        "response_format": {"type": "json_object"},
    }
    payload.update(provider._provider_payload_overrides())
    result: dict[str, Any] = {
        "schema_version": "deep-research-synthesize-smoke-v1",
        "mode": "provider-smoke",
        "checked_at": datetime.now(UTC).isoformat(),
        "run_id": run_id,
        "provider_config": provider_config(settings, provider),
        "request_count": 1,
        "raw_payload_persisted": False,
        "api_key_persisted": False,
        "authorization_header_persisted": False,
    }
    try:
        response = provider.client.post(
            provider.endpoint,
            headers={
                "Authorization": f"Bearer {provider.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=settings.llm_timeout_seconds,
        )
        elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
        body = response.json()
        content = str(body.get("choices", [{}])[0].get("message", {}).get("content", ""))
        usage = provider._usage(body.get("usage") or {})
        result.update(
            {
                "status": "PASSED" if response.status_code < 400 else "FAILED",
                "http_status": response.status_code,
                "elapsed_ms": elapsed_ms,
                "json_valid": _json_valid(content),
                "content_sha256": sha256_text(content),
                "usage": {
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "total_tokens": usage.total_tokens,
                    "usage_source": "provider_reported" if usage.total_tokens else "missing",
                    "estimated_cost_usd": usage.estimated_cost_usd,
                },
            }
        )
    except (httpx.TimeoutException, httpx.NetworkError) as exc:
        result.update(
            {
                "status": "FAILED",
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
                "error": classify_provider_exception(
                    exc,
                    hostname=(settings.llm_base_url or "").split("://", 1)[-1].split("/", 1)[0],
                ),
                "message_sanitized": redact_sensitive_text(str(exc)),
            }
        )
    return result


def _json_valid(content: str) -> bool:
    try:
        json.loads(content)
    except json.JSONDecodeError:
        return False
    return True


def main() -> int:
    args = parse_args()
    if args.mode == "provider-smoke" and not args.allow_live_call:
        raise SystemExit("provider-smoke requires --allow-live-call")
    payload = (
        inspect_payload(args.run_id)
        if args.mode == "inspect"
        else provider_smoke_payload(args.run_id)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "mode": args.mode,
                "status": payload.get("status", "INSPECTED"),
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )
    return 0 if payload.get("status", "PASSED") != "FAILED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
