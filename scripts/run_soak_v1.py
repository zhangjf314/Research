# ruff: noqa: E501
"""Time-bounded Docker soak with queries, imports, research, restart, and resource samples."""

import json
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

BASE = "http://localhost/api/v1"
DURATION_SECONDS = int(os.getenv("SOAK_SECONDS", "1800"))
SAMPLE_SECONDS = 15
OUTPUT = Path("artifacts/soak-test-v1.json")
REPORT = Path("docs/soak-test-v1.md")


def docker(*args: str) -> str:
    return subprocess.run(
        ["docker", "compose", *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def resource_sample(elapsed: float) -> dict:
    stats = subprocess.run(
        ["docker", "stats", "--no-stream", "--format", "{{json .}}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    db = docker(
        "exec",
        "-T",
        "postgres",
        "psql",
        "-U",
        "paper",
        "-d",
        "paper_research",
        "-Atc",
        "SELECT (SELECT count(*) FROM pg_stat_activity WHERE datname='paper_research'),"
        "(SELECT count(*) FROM checkpoints),"
        "(SELECT count(*) FROM checkpoint_writes);",
    )
    qdrant = httpx.get(
        "http://localhost:6333/collections/papers_hash_v1__20260713104355", timeout=5
    ).json()["result"]
    redis_keys = int(docker("exec", "-T", "redis", "redis-cli", "DBSIZE"))
    return {
        "elapsed_seconds": round(elapsed, 3),
        "containers": [json.loads(line) for line in stats],
        "postgres": db,
        "qdrant_points": qdrant["points_count"],
        "redis_keys": redis_keys,
    }


def wait_healthy(client: httpx.Client, attempts: int = 45) -> bool:
    for _ in range(attempts):
        try:
            if client.get(f"{BASE}/health").status_code == 200:
                return True
        except httpx.HTTPError:
            pass
        time.sleep(1)
    return False


def main() -> None:
    start = time.monotonic()
    next_sample = next_import = next_research = 0.0
    restart_at = DURATION_SECONDS / 2
    restarted = False
    queries = imports = research_runs = failures = 0
    latencies: list[float] = []
    samples = []
    failure_details = []
    fixture = Path("data/ocr-audit-v1/text-native.pdf")
    with httpx.Client(timeout=90) as client:
        while (elapsed := time.monotonic() - start) < DURATION_SECONDS:
            if elapsed >= next_sample:
                samples.append(resource_sample(elapsed))
                next_sample += SAMPLE_SECONDS
            if elapsed >= next_import:
                try:
                    with fixture.open("rb") as stream:
                        response = client.post(
                            f"{BASE}/papers/upload",
                            files={"file": (fixture.name, stream, "application/pdf")},
                        )
                    response.raise_for_status()
                    imports += 1
                except Exception as exc:
                    failures += 1
                    failure_details.append(f"import: {type(exc).__name__}")
                next_import += 45
            if elapsed >= next_research:
                try:
                    response = client.post(
                        f"{BASE}/research/deep",
                        json={
                            "query": "Compare retrieval methods, evidence, and limitations",
                            "allow_external_search": False,
                            "budget": {"max_iterations": 2, "max_papers": 10},
                        },
                    )
                    response.raise_for_status()
                    research_runs += 1
                except Exception as exc:
                    failures += 1
                    failure_details.append(f"research: {type(exc).__name__}")
                next_research += 60
            if not restarted and elapsed >= restart_at:
                docker("restart", "api")
                restarted = True
                if not wait_healthy(client):
                    failures += 1
                    failure_details.append("API did not recover after restart")
            before = time.perf_counter()
            try:
                response = client.post(
                    f"{BASE}/retrieve",
                    json={
                        "query": f"retrieval evidence limitations soak {queries % 7}",
                        "recall_k": 20,
                        "top_k": 5,
                    },
                )
                response.raise_for_status()
                queries += 1
                latencies.append((time.perf_counter() - before) * 1000)
            except Exception as exc:
                failures += 1
                failure_details.append(f"query: {type(exc).__name__}")
            time.sleep(0.25)
    samples.append(resource_sample(time.monotonic() - start))
    ordered = sorted(latencies)
    p95 = ordered[max(0, int(len(ordered) * 0.95) - 1)] if ordered else None
    payload = {
        "started_at": datetime.now(UTC).isoformat(),
        "duration_seconds": round(time.monotonic() - start, 3),
        "profile": "baseline",
        "queries": queries,
        "periodic_imports": imports,
        "deep_research_runs": research_runs,
        "api_restart_performed": restarted,
        "failures": failures,
        "failure_details": failure_details,
        "mean_query_ms": round(sum(latencies) / len(latencies), 3) if latencies else None,
        "p95_query_ms": round(p95, 3) if p95 is not None else None,
        "input_tokens": 0,
        "output_tokens": 0,
        "cost_usd": 0.0,
        "samples": samples,
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Soak Test v1",
        "",
        f"- Duration: {payload['duration_seconds']} seconds",
        f"- Queries: {queries}; mean {payload['mean_query_ms']} ms; P95 {payload['p95_query_ms']} ms",
        f"- Periodic imports: {imports}",
        f"- Deep Research runs: {research_runs}",
        f"- API restart performed: {restarted}",
        f"- Failures: {failures}",
        "- Token / cost: 0 / $0.00 (baseline template provider; not production-model evidence)",
        "",
        "## Resource samples",
        "",
        "| Seconds | API memory | PostgreSQL state | Qdrant points | Redis keys |",
        "|---:|---:|---|---:|---:|",
    ]
    for sample in samples:
        api = next(
            (row.get("MemUsage", "unknown") for row in sample["containers"] if row.get("Name") == "research-api-1"),
            "unknown",
        )
        lines.append(
            f"| {sample['elapsed_seconds']} | {api} | {sample['postgres']} | "
            f"{sample['qdrant_points']} | {sample['redis_keys']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "This is a time-bounded local soak. It can expose immediate restart, connection, and memory-growth defects, but cannot prove the absence of slow leaks over production-scale durations.",
            "",
        ]
    )
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "samples"}, indent=2))


if __name__ == "__main__":
    main()
