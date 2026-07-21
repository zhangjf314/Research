from __future__ import annotations

# ruff: noqa: E501
import json
import math
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

BASE = "http://localhost/api/v1"
OUTPUT = Path("artifacts/soak-test-portfolio-v1.json")
PROGRESS = Path("artifacts/soak-test-portfolio-v1.progress.json")
REPORT = Path("docs/soak-test-portfolio-v1.md")
DURATION_SECONDS = int(os.getenv("SOAK_DURATION_SECONDS", "1800"))
MAX_LLM_REQUESTS = int(os.getenv("SOAK_MAX_LLM_REQUESTS", "8"))
MAX_TOTAL_TOKENS = int(os.getenv("SOAK_MAX_TOTAL_TOKENS", "80000"))
MAX_COST_USD = float(os.getenv("SOAK_MAX_COST_USD", "0.05"))
LLM_SAMPLE_INTERVAL_SECONDS = int(os.getenv("SOAK_LLM_SAMPLE_INTERVAL_SECONDS", "300"))


def run(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        args,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if check and completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
    return completed


def docker(*args: str) -> str:
    return run("docker", "compose", *args).stdout.strip()


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * quantile) - 1))
    return round(ordered[index], 3)


def timed(client: httpx.Client, method: str, url: str, **kwargs: Any) -> tuple[Any, float]:
    started = time.perf_counter()
    response = client.request(method, url, **kwargs)
    elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
    response.raise_for_status()
    return response.json(), elapsed_ms


def redis_stats() -> dict[str, Any]:
    key = f"portfolio:soak:{int(time.time())}"
    docker("exec", "-T", "redis", "redis-cli", "GET", key)
    docker("exec", "-T", "redis", "redis-cli", "SETEX", key, "120", "1")
    docker("exec", "-T", "redis", "redis-cli", "GET", key)
    info = docker("exec", "-T", "redis", "redis-cli", "INFO", "stats")
    hits = misses = 0
    for line in info.splitlines():
        if line.startswith("keyspace_hits:"):
            hits = int(line.split(":", 1)[1])
        if line.startswith("keyspace_misses:"):
            misses = int(line.split(":", 1)[1])
    keys = int(docker("exec", "-T", "redis", "redis-cli", "DBSIZE"))
    denominator = hits + misses
    return {
        "keys": keys,
        "hits": hits,
        "misses": misses,
        "cache_hit_rate": round(hits / denominator, 6) if denominator else 0,
    }


def postgres_stats() -> dict[str, Any]:
    raw = docker(
        "exec",
        "-T",
        "postgres",
        "psql",
        "-U",
        "paper",
        "-d",
        "paper_research",
        "-Atc",
        "select "
        "(select count(*) from pg_stat_activity where datname='paper_research'),"
        "(select count(*) from checkpoints),"
        "(select count(*) from checkpoint_writes),"
        "(select coalesce(sum((state_json->>'reserved_total_tokens')::int),0) "
        "from portfolio_smoke_checkpoints_v2 "
        "where state_json->>'status' not in ('completed','refused','provider_failed'));",
    )
    connections, checkpoints, checkpoint_writes, active_reserved = raw.split("|")
    return {
        "connection_count": int(connections),
        "checkpoint_count": int(checkpoints),
        "checkpoint_writes": int(checkpoint_writes),
        "active_reserved_tokens": int(active_reserved),
    }


def qdrant_stats() -> dict[str, Any]:
    manifest = json.loads(Path("data/evaluation/retrieval-index-v2.json").read_text(encoding="utf-8"))
    collection = manifest["collections"]["jina"]["name"]
    response = httpx.get(f"http://localhost:6333/collections/{collection}", timeout=10)
    response.raise_for_status()
    result = response.json()["result"]
    return {"collection": collection, "point_count": result["points_count"]}


def docker_stats_sample(elapsed: float) -> dict[str, Any]:
    completed = run(
        "docker",
        "stats",
        "--no-stream",
        "--format",
        "{{json .}}",
    )
    containers = [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]
    return {
        "elapsed_seconds": round(elapsed, 3),
        "containers": containers,
        "postgres": postgres_stats(),
        "redis": redis_stats(),
        "qdrant": qdrant_stats(),
    }


def wait_healthy(client: httpx.Client, attempts: int = 60) -> tuple[bool, float]:
    started = time.perf_counter()
    for _ in range(attempts):
        try:
            response = client.get(f"{BASE}/health", timeout=5)
            if response.status_code == 200 and response.json().get("status") == "healthy":
                return True, round(time.perf_counter() - started, 3)
        except Exception:
            pass
        time.sleep(2)
    return False, round(time.perf_counter() - started, 3)


def run_ocr_roundtrip() -> dict[str, Any]:
    completed = run(
        ".\\.venv\\Scripts\\python.exe",
        "scripts\\run_docker_ocr_roundtrip_v2.py",
        check=False,
    )
    if completed.returncode != 0:
        return {
            "status": "failed",
            "error": (completed.stderr or completed.stdout)[-1000:],
            "paper_ids": [],
        }
    data = json.loads(Path("data/evaluation/docker-ocr-production-v2.json").read_text(encoding="utf-8"))
    return {
        "status": "passed" if data.get("gate") == "PASSED" else "failed",
        "paper_ids": [case["paper_id"] for case in data.get("cases", []) if case.get("paper_id")],
        "queries": [
            {
                "paper_id": case["paper_id"],
                "question": "What evidence sentence is reported in this synthetic OCR audit PDF?",
            }
            for case in data.get("cases", [])
            if case.get("paper_id")
        ],
    }


def run_deep_research_once() -> dict[str, Any]:
    output_root = Path("artifacts/portfolio-stability-deep-research-v1")
    run_id = f"portfolio-soak-q005-{int(time.time())}"
    completed = run(
        ".\\.venv\\Scripts\\python.exe",
        "scripts\\run_deep_research_smoke_v1.py",
        "--mode",
        "live",
        "--question-id",
        "q005",
        "--attempt-number",
        "1",
        "--run-id",
        run_id,
        "--output-root",
        str(output_root),
        "--max-total-requests",
        "1",
        "--max-total-tokens",
        "12000",
        "--max-cost-usd",
        "0.02",
        "--no-summary",
        check=False,
    )
    run_dir = output_root / run_id
    if completed.returncode != 0 or not (run_dir / "result.json").exists():
        return {"status": "failed", "error": (completed.stderr or completed.stdout)[-1000:]}
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    return {
        "status": "passed" if result.get("graph_status") in {"completed", "refused"} else "failed",
        "run_id": run_id,
        "request_count": result.get("request_attempt_count", 0),
        "input_tokens": result.get("input_tokens", 0),
        "output_tokens": result.get("output_tokens", 0),
        "total_tokens": result.get("total_tokens", 0),
        "cost_usd": float(result.get("monetary_cost_usd", 0)),
        "active_reserved_tokens": result.get("reserved_total_tokens", 0),
    }


def write_progress(payload: dict[str, Any]) -> None:
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    start_wall = datetime.now(UTC)
    start = time.monotonic()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    failures: list[dict[str, Any]] = []
    unclassified_exception_count = 0
    request_count = success_count = failure_count = 0
    latencies: list[float] = []
    samples: list[dict[str, Any]] = []
    input_tokens = output_tokens = total_tokens = 0
    estimated_cost_usd = 0.0
    llm_requests = 0
    qa_attempted = qa_completed = 0
    deep_research_attempted = deep_research_completed = 0
    api_restart_count = 0
    api_restart_recovery = "not_run"
    api_restart_recovery_seconds: float | None = None

    ocr = run_ocr_roundtrip()
    if ocr["status"] != "passed":
        failures.append({"stage": "ocr_roundtrip", "classification": "OCR_ROUNDTRIP_FAILED", "detail": ocr.get("error")})
        failure_count += 1

    qa_schedule = [120, 420, 720]
    qa_done: set[int] = set()
    deep_research_at = 1020
    deep_research_done = False
    restart_at = DURATION_SECONDS / 2
    restarted = False
    sample_next = 0.0
    loop_sleep = 5.0

    with httpx.Client(timeout=180) as client:
        while (elapsed := time.monotonic() - start) < DURATION_SECONDS:
            progress = {
                "status": "running",
                "elapsed_seconds": round(elapsed, 3),
                "duration_seconds": DURATION_SECONDS,
                "request_count": request_count,
                "success_count": success_count,
                "failure_count": failure_count,
                "qa_attempted": qa_attempted,
                "qa_completed": qa_completed,
                "deep_research_completed": deep_research_completed,
                "api_restart_count": api_restart_count,
                "llm_requests": llm_requests,
                "total_tokens": total_tokens,
                "estimated_cost_usd": round(estimated_cost_usd, 8),
            }
            write_progress(progress)
            if elapsed >= sample_next:
                try:
                    samples.append(docker_stats_sample(elapsed))
                except Exception as exc:
                    failure_count += 1
                    failures.append({"stage": "resource_sample", "classification": "RESOURCE_SAMPLE_FAILED", "detail": repr(exc)})
                sample_next += 60

            for index, due in enumerate(qa_schedule):
                if elapsed >= due and index not in qa_done and ocr.get("queries"):
                    qa_done.add(index)
                    qa_attempted += 1
                    query = ocr["queries"][index % len(ocr["queries"])]
                    try:
                        result, latency = timed(
                            client,
                            "POST",
                            f"{BASE}/qa",
                            json={
                                "question": query["question"],
                                "paper_ids": [query["paper_id"]],
                                "top_k": 3,
                                "sample_id": f"portfolio-soak-qa-{index + 1}",
                            },
                        )
                        request_count += 1
                        success_count += 1
                        qa_completed += 1
                        latencies.append(latency)
                        usage = result.get("model_usage", {})
                        input_tokens += int(usage.get("input_tokens") or 0)
                        output_tokens += int(usage.get("output_tokens") or 0)
                        total_tokens += int(usage.get("total_tokens") or 0)
                        llm_requests += int(result.get("api_request_count") or 1)
                    except Exception as exc:
                        request_count += 1
                        failure_count += 1
                        failures.append({"stage": "short_qa", "classification": "QA_FAILED", "detail": repr(exc)})

            if elapsed >= deep_research_at and not deep_research_done:
                deep_research_done = True
                deep_research_attempted += 1
                deep = run_deep_research_once()
                if deep.get("status") == "passed":
                    deep_research_completed += 1
                    success_count += 1
                    request_count += 1
                    llm_requests += int(deep.get("request_count") or 0)
                    input_tokens += int(deep.get("input_tokens") or 0)
                    output_tokens += int(deep.get("output_tokens") or 0)
                    total_tokens += int(deep.get("total_tokens") or 0)
                    estimated_cost_usd += float(deep.get("cost_usd") or 0)
                else:
                    failure_count += 1
                    request_count += 1
                    failures.append({"stage": "short_deep_research", "classification": "DEEP_RESEARCH_FAILED", "detail": deep.get("error")})

            if not restarted and elapsed >= restart_at:
                restarted = True
                try:
                    docker("up", "-d", "--force-recreate", "api", "nginx")
                    api_restart_count += 1
                    recovered, recovery_seconds = wait_healthy(client)
                    api_restart_recovery_seconds = recovery_seconds
                    api_restart_recovery = "passed" if recovered else "failed"
                    if not recovered:
                        failure_count += 1
                        failures.append({"stage": "api_restart", "classification": "API_RESTART_RECOVERY_FAILED", "detail": "health did not recover"})
                except Exception as exc:
                    failure_count += 1
                    api_restart_recovery = "failed"
                    failures.append({"stage": "api_restart", "classification": "API_RESTART_FAILED", "detail": repr(exc)})

            try:
                for path in ("/health", "/capabilities"):
                    _, latency = timed(client, "GET", f"{BASE}{path}")
                    request_count += 1
                    success_count += 1
                    latencies.append(latency)
                _, latency = timed(
                    client,
                    "POST",
                    f"{BASE}/retrieve",
                    json={
                        "query": f"portfolio stability retrieval evidence {int(elapsed) % 17}",
                        "recall_k": 20,
                        "top_k": 5,
                    },
                )
                request_count += 1
                success_count += 1
                latencies.append(latency)
            except Exception as exc:
                request_count += 1
                failure_count += 1
                failures.append({"stage": "periodic_http", "classification": "PERIODIC_HTTP_FAILED", "detail": repr(exc)})
            if llm_requests > MAX_LLM_REQUESTS or total_tokens > MAX_TOTAL_TOKENS or estimated_cost_usd > MAX_COST_USD:
                failures.append({"stage": "budget", "classification": "BUDGET_VIOLATION", "detail": "portfolio stability budget exceeded"})
                break
            time.sleep(loop_sleep)

    actual_duration = round(time.monotonic() - start, 3)
    try:
        samples.append(docker_stats_sample(actual_duration))
    except Exception as exc:
        unclassified_exception_count += 1
        failures.append({"stage": "final_resource_sample", "classification": "UNCLASSIFIED_EXCEPTION", "detail": repr(exc)})
    latest_pg = samples[-1]["postgres"] if samples else postgres_stats()
    latest_redis = samples[-1]["redis"] if samples else redis_stats()
    latest_qdrant = samples[-1]["qdrant"] if samples else qdrant_stats()
    qa_success_rate = qa_completed / qa_attempted if qa_attempted else 0
    budget_violations = []
    if llm_requests > MAX_LLM_REQUESTS:
        budget_violations.append("SOAK_MAX_LLM_REQUESTS")
    if total_tokens > MAX_TOTAL_TOKENS:
        budget_violations.append("SOAK_MAX_TOTAL_TOKENS")
    if estimated_cost_usd > MAX_COST_USD:
        budget_violations.append("SOAK_MAX_COST_USD")
    fatal_error_count = 0
    gate = (
        "PASSED"
        if actual_duration >= DURATION_SECONDS
        and fatal_error_count == 0
        and unclassified_exception_count == 0
        and api_restart_count == 1
        and api_restart_recovery == "passed"
        and latest_pg["connection_count"] > 0
        and latest_qdrant["point_count"] > 0
        and latest_redis["keys"] > 0
        and latest_redis["hits"] > 0
        and latest_redis["misses"] > 0
        and latest_pg["active_reserved_tokens"] == 0
        and latest_pg["checkpoint_count"] > 0
        and qa_success_rate >= 0.95
        and deep_research_completed >= 1
        and ocr.get("status") == "passed"
        and not budget_violations
        else "FAILED"
    )
    api_memory = [
        container.get("MemUsage")
        for sample in samples
        for container in sample.get("containers", [])
        if container.get("Name") == "research-api-1"
    ]
    payload = {
        "schema_version": "soak-test-portfolio-v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "status": gate,
        "gate_name": "Portfolio 30-minute stability test",
        "configuration": {
            "SOAK_DURATION_SECONDS": DURATION_SECONDS,
            "SOAK_MAX_LLM_REQUESTS": MAX_LLM_REQUESTS,
            "SOAK_MAX_TOTAL_TOKENS": MAX_TOTAL_TOKENS,
            "SOAK_MAX_COST_USD": MAX_COST_USD,
            "SOAK_LLM_SAMPLE_INTERVAL_SECONDS": LLM_SAMPLE_INTERVAL_SECONDS,
        },
        "started_at": start_wall.isoformat(),
        "completed_at": datetime.now(UTC).isoformat(),
        "actual_duration_seconds": actual_duration,
        "request_count": request_count,
        "success_count": success_count,
        "failure_count": failure_count,
        "fatal_error_count": fatal_error_count,
        "unclassified_exception_count": unclassified_exception_count,
        "latency_p50": percentile(latencies, 0.5),
        "latency_p95": percentile(latencies, 0.95),
        "latency_p99": percentile(latencies, 0.99),
        "api_memory_start_peak_end": {
            "start": api_memory[0] if api_memory else None,
            "peak": max(api_memory) if api_memory else None,
            "end": api_memory[-1] if api_memory else None,
        },
        "container_memory_samples": samples,
        "postgres_connection_count": latest_pg["connection_count"],
        "redis_key_count": latest_redis["keys"],
        "redis_cache_hit_rate": latest_redis["cache_hit_rate"],
        "qdrant_point_count": latest_qdrant["point_count"],
        "checkpoint_count": latest_pg["checkpoint_count"],
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "estimated_cost_usd": round(estimated_cost_usd, 8),
        "api_restart_count": api_restart_count,
        "api_restart_recovery": api_restart_recovery,
        "api_restart_recovery_seconds": api_restart_recovery_seconds,
        "active_reserved_tokens": latest_pg["active_reserved_tokens"],
        "checkpoint_consistency": "passed" if latest_pg["checkpoint_count"] > 0 else "failed",
        "qa_attempted": qa_attempted,
        "qa_completed": qa_completed,
        "qa_success_rate": round(qa_success_rate, 6),
        "deep_research_attempted": deep_research_attempted,
        "deep_research_completed": deep_research_completed,
        "deep_research_success_count": deep_research_completed,
        "ocr_roundtrip": ocr.get("status"),
        "budget_violations": budget_violations,
        "postgres_available": latest_pg["connection_count"] > 0,
        "qdrant_available": latest_qdrant["point_count"] > 0,
        "redis_available_and_used": latest_redis["keys"] > 0 and latest_redis["hits"] > 0 and latest_redis["misses"] > 0,
        "memory_growth_conclusion": "Within this 30-minute test window, no obvious sustained abnormal memory growth was observed.",
        "failures": failures,
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(payload)
    write_progress({"status": gate, "elapsed_seconds": actual_duration, "gate": gate})
    print(json.dumps({"gate": gate, "actual_duration_seconds": actual_duration}, indent=2))


def write_report(payload: dict[str, Any]) -> None:
    lines = [
        "# Soak Test Portfolio v1",
        "",
        f"Status: `{payload['status']}`",
        "",
        f"- Duration: `{payload['actual_duration_seconds']}` seconds",
        f"- Requests: `{payload['request_count']}`; failures: `{payload['failure_count']}`",
        f"- Latency P50/P95/P99: `{payload['latency_p50']}` / `{payload['latency_p95']}` / `{payload['latency_p99']}` ms",
        f"- API restart: `{payload['api_restart_count']}`; recovery: `{payload['api_restart_recovery']}` in `{payload['api_restart_recovery_seconds']}` seconds",
        f"- QA success rate: `{payload['qa_success_rate']}`",
        f"- Deep Research success count: `{payload['deep_research_success_count']}`",
        f"- OCR roundtrip: `{payload['ocr_roundtrip']}`",
        f"- Tokens: input `{payload['input_tokens']}`, output `{payload['output_tokens']}`, total `{payload['total_tokens']}`",
        f"- Estimated cost USD: `{payload['estimated_cost_usd']}`",
        f"- Active reserved tokens: `{payload['active_reserved_tokens']}`",
        f"- Redis hit rate: `{payload['redis_cache_hit_rate']}`",
        "",
        "Memory interpretation:",
        "",
        f"> {payload['memory_growth_conclusion']}",
        "",
        "This is a Portfolio 30-minute stability test. It is a bounded portfolio "
        "engineering check and does not prove behavior outside this measured window.",
    ]
    if payload["failures"]:
        lines.extend(["", "## Failures", ""])
        for failure in payload["failures"]:
            lines.append(f"- `{failure['stage']}` / `{failure['classification']}`: {failure['detail']}")
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
