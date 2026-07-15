# ruff: noqa: E501
"""One-shot, fail-closed SiliconFlow health check without secret persistence."""

from __future__ import annotations

import argparse
import json
import socket
import ssl
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

import httpx

from paper_research.config import Settings

OUTPUT = Path("data/evaluation/provider-health-v1.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline-validate", action="store_true")
    parser.add_argument("--allow-minimal-completion", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


def _base_payload(base_url: str) -> dict:
    parsed = urlparse(base_url)
    return {
        "checked_at": datetime.now(UTC).isoformat(),
        "base_url_host": parsed.hostname,
        "base_url_valid": parsed.scheme == "https" and bool(parsed.hostname),
        "dns_status": "not_run",
        "tcp_status": "not_run",
        "tls_status": "not_run",
        "models_endpoint_status": "not_run",
        "minimal_completion_status": "not_run",
        "latency": {},
        "error_type": None,
        "safe_to_start_batch": False,
        "api_key_recorded": False,
        "authorization_header_recorded": False,
    }


def check_health(
    settings: Settings,
    *,
    allow_minimal_completion: bool = False,
) -> dict:
    base_url = settings.llm_base_url or ""
    result = _base_payload(base_url)
    if not result["base_url_valid"]:
        result["error_type"] = "invalid_base_url"
        return result
    host = result["base_url_host"]
    assert isinstance(host, str)
    try:
        started = time.perf_counter()
        addresses = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
        result["latency"]["dns_ms"] = round((time.perf_counter() - started) * 1000, 3)
        result["dns_status"] = "passed" if addresses else "failed"
    except OSError as exc:
        result["dns_status"] = "failed"
        result["error_type"] = type(exc).__name__
        return result
    try:
        started = time.perf_counter()
        with socket.create_connection((host, 443), timeout=10) as raw:
            result["tcp_status"] = "passed"
            result["latency"]["tcp_ms"] = round((time.perf_counter() - started) * 1000, 3)
            context = ssl.create_default_context()
            tls_started = time.perf_counter()
            with context.wrap_socket(raw, server_hostname=host):
                result["tls_status"] = "passed"
                result["latency"]["tls_ms"] = round((time.perf_counter() - tls_started) * 1000, 3)
    except (OSError, ssl.SSLError) as exc:
        if result["tcp_status"] != "passed":
            result["tcp_status"] = "failed"
        else:
            result["tls_status"] = "failed"
        result["error_type"] = type(exc).__name__
        return result
    try:
        started = time.perf_counter()
        response = httpx.get(
            f"{base_url.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {settings.llm_api_key}"},
            timeout=15,
        )
        result["latency"]["models_ms"] = round((time.perf_counter() - started) * 1000, 3)
        result["models_endpoint_status"] = "passed" if response.status_code < 400 else f"http_{response.status_code}"
    except httpx.HTTPError as exc:
        result["models_endpoint_status"] = "failed"
        result["error_type"] = type(exc).__name__
    if result["models_endpoint_status"] == "passed":
        result["safe_to_start_batch"] = True
        return result
    if not allow_minimal_completion:
        return result
    try:
        started = time.perf_counter()
        response = httpx.post(
            f"{base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {settings.llm_api_key}"},
            json={
                "model": settings.llm_model,
                "messages": [{"role": "user", "content": "Return JSON: {\"ok\":true}"}],
                "temperature": 0,
                "max_tokens": 8,
                "stream": False,
            },
            timeout=15,
        )
        result["latency"]["minimal_completion_ms"] = round((time.perf_counter() - started) * 1000, 3)
        result["minimal_completion_status"] = "passed" if response.status_code < 400 else f"http_{response.status_code}"
        result["safe_to_start_batch"] = result["minimal_completion_status"] == "passed"
    except httpx.HTTPError as exc:
        result["minimal_completion_status"] = "failed"
        result["error_type"] = type(exc).__name__
    return result


def main() -> int:
    args = parse_args()
    settings = Settings()
    if args.offline_validate:
        result = _base_payload(settings.llm_base_url or "")
        result.update(
            {
                "status": "NOT_RUN_PHASE_A",
                "error_type": "live_health_check_not_authorized_in_phase_a",
                "safe_to_start_batch": False,
            }
        )
    else:
        result = check_health(
            settings,
            allow_minimal_completion=args.allow_minimal_completion,
        )
        result["status"] = "PASSED" if result["safe_to_start_batch"] else "DEV_V2_BLOCKED_BY_PROVIDER_HEALTH"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result))
    return 0 if args.offline_validate or result["safe_to_start_batch"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
