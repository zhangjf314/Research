# ruff: noqa: E501
"""One-shot, fail-closed SiliconFlow health check without secret persistence."""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import ssl
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

import httpx

from paper_research.config import Settings
from paper_research.providers.factory import build_llm_provider

OUTPUT = Path("data/evaluation/provider-health-v1.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline-validate", action="store_true")
    parser.add_argument("--allow-minimal-completion", action="store_true")
    parser.add_argument("--require-minimal-completion", action="store_true")
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
        "minimal_completion_json_valid": None,
        "minimal_completion_model": None,
        "minimal_completion_usage": None,
        "minimal_completion_finish_reason": None,
        "minimal_completion_reasoning_content_present": None,
        "factory_provider": None,
        "factory_model": None,
        "provider_name": None,
        "model": None,
        "thinking_mode": None,
        "response_format": None,
        "stream": None,
        "temperature": None,
        "max_tokens": None,
        "api_key_present": False,
        "api_key_fingerprint": None,
        "request_contract": {},
        "template_fallback": None,
        "latency": {},
        "error_type": None,
        "safe_to_start_batch": False,
        "api_key_recorded": False,
        "authorization_header_recorded": False,
    }


def _fingerprint(value: str | None) -> str | None:
    if not value:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def _minimal_completion_payload(settings: Settings) -> dict:
    payload = {
        "model": settings.llm_model,
        "messages": [{"role": "user", "content": "Return exactly this JSON: {\"status\":\"ok\"}"}],
        "temperature": settings.llm_temperature,
        "max_tokens": min(settings.llm_max_output_tokens, 32),
        "stream": settings.llm_stream,
        "response_format": {"type": settings.llm_response_format},
    }
    if (settings.llm_provider_name or settings.llm_provider).lower() == "deepseek":
        payload["thinking"] = {
            "type": "enabled" if settings.llm_thinking_enabled else "disabled",
        }
    else:
        payload["enable_thinking"] = settings.llm_thinking_enabled
    return payload


def check_health(
    settings: Settings,
    *,
    allow_minimal_completion: bool = False,
    require_minimal_completion: bool = False,
) -> dict:
    base_url = settings.llm_base_url or ""
    result = _base_payload(base_url)
    provider_name = settings.llm_provider_name or settings.llm_provider
    result.update(
        {
            "provider_name": provider_name,
            "model": settings.llm_model,
            "thinking_mode": "enabled" if settings.llm_thinking_enabled else "disabled",
            "response_format": settings.llm_response_format,
            "stream": settings.llm_stream,
            "temperature": settings.llm_temperature,
            "max_tokens": settings.llm_max_output_tokens,
            "api_key_present": bool(settings.llm_api_key),
            "api_key_fingerprint": _fingerprint(settings.llm_api_key),
            "request_contract": {
                "model": settings.llm_model,
                "response_format.type": settings.llm_response_format,
                "thinking.type": (
                    "enabled" if settings.llm_thinking_enabled else "disabled"
                ),
                "stream": settings.llm_stream,
            },
        }
    )
    if not result["base_url_valid"]:
        result["error_type"] = "invalid_base_url"
        return result
    if settings.llm_configuration_issues:
        result["factory_provider"] = settings.llm_provider
        result["factory_model"] = settings.llm_model
        result["template_fallback"] = settings.llm_provider == "template"
        result["error_type"] = "configuration:" + ",".join(settings.llm_configuration_issues)
    else:
        try:
            provider = build_llm_provider(settings)
            result["factory_provider"] = provider.provider_name
            result["factory_model"] = provider.model_name
            result["template_fallback"] = provider.provider_name == "template"
        except Exception as exc:
            result["error_type"] = f"factory:{type(exc).__name__}"
            result["template_fallback"] = None
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
    llm_configured = not settings.llm_configuration_issues and not result["template_fallback"]
    if result["models_endpoint_status"] == "passed" and not require_minimal_completion:
        result["safe_to_start_batch"] = llm_configured
        return result
    if not allow_minimal_completion and not require_minimal_completion:
        return result
    try:
        started = time.perf_counter()
        response = httpx.post(
            f"{base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {settings.llm_api_key}"},
            json=_minimal_completion_payload(settings),
            timeout=15,
        )
        result["latency"]["minimal_completion_ms"] = round((time.perf_counter() - started) * 1000, 3)
        result["minimal_completion_status"] = "passed" if response.status_code < 400 else f"http_{response.status_code}"
        if response.status_code < 400:
            payload = response.json()
            result["minimal_completion_model"] = payload.get("model")
            result["minimal_completion_usage"] = payload.get("usage")
            choice = (payload.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            result["minimal_completion_finish_reason"] = choice.get("finish_reason")
            result["minimal_completion_reasoning_content_present"] = "reasoning_content" in message
            content = (
                message.get("content", "")
            )
            try:
                json.loads(content)
                result["minimal_completion_json_valid"] = True
            except json.JSONDecodeError:
                result["minimal_completion_json_valid"] = False
        result["safe_to_start_batch"] = (
            result["minimal_completion_status"] == "passed"
            and result["minimal_completion_json_valid"] is True
            and result["minimal_completion_finish_reason"] == "stop"
            and not result["minimal_completion_reasoning_content_present"]
            and llm_configured
        )
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
            require_minimal_completion=args.require_minimal_completion,
        )
        result["status"] = "PASSED" if result["safe_to_start_batch"] else "DEV_V2_BLOCKED_BY_PROVIDER_HEALTH"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result))
    return 0 if args.offline_validate or result["safe_to_start_batch"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
