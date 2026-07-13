# ruff: noqa: E501
"""Run the 30-paper/100-query RC stability workload against Docker Compose."""

import json
import math
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

BASE = "http://localhost/api/v1"
TARGET_PAPERS = 30
REPORT = Path("docs/stability-report-v1.md")
ARTIFACT = Path("artifacts/stability-results-v1.json")


def p95(values: list[float]) -> float:
    values = sorted(values)
    return values[max(0, math.ceil(len(values) * 0.95) - 1)] if values else 0.0


def memory_snapshot(label: str) -> dict:
    command = ["docker", "stats", "--no-stream", "--format", "{{json .}}"]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return {"label": label, "containers": [json.loads(line) for line in completed.stdout.splitlines()]}


def timed(client: httpx.Client, method: str, path: str, **kwargs) -> tuple[dict, float]:
    started = time.perf_counter()
    response = client.request(method, f"{BASE}{path}", **kwargs)
    elapsed = (time.perf_counter() - started) * 1000
    response.raise_for_status()
    return response.json(), elapsed


def main() -> None:
    started_at = datetime.now(UTC)
    parse_records, index_records, failures = [], [], []
    retry_attempts = retry_successes = 0
    paper_ids: list[str] = []
    memory = [memory_snapshot("start")]
    with httpx.Client(timeout=240) as client:
        for path in sorted(Path("data/raw/audit").glob("*.pdf")):
            with path.open("rb") as stream:
                try:
                    uploaded, latency = timed(
                        client,
                        "POST",
                        "/papers/upload",
                        files={"file": (path.name, stream, "application/pdf")},
                    )
                    paper_id = uploaded["paper"]["id"]
                    paper_ids.append(paper_id)
                    parse_records.append(
                        {
                            "paper_id": paper_id,
                            "source": str(path),
                            "success": True,
                            "duplicate": uploaded["duplicate"],
                            "latency_ms": round(latency, 3),
                        }
                    )
                    indexed, index_ms = timed(client, "POST", f"/papers/{paper_id}/index")
                    index_records.append(
                        {
                            "paper_id": paper_id,
                            "success": indexed["status"] == "READY",
                            "chunk_count": indexed["chunk_count"],
                            "latency_ms": round(index_ms, 3),
                        }
                    )
                except Exception as exc:
                    failures.append({"stage": "local_upload", "source": str(path), "error": repr(exc)})
        search, _ = timed(
            client,
            "POST",
            "/search/papers",
            json={
                "query": "retrieval augmented generation language models evaluation",
                "limit": 50,
                "open_access_only": True,
            },
        )
        for candidate in search["candidates"]:
            if len(set(paper_ids)) >= TARGET_PAPERS or not candidate.get("pdf_url"):
                break
            imported = None
            for attempt in range(2):
                if attempt:
                    retry_attempts += 1
                try:
                    imported, latency = timed(client, "POST", "/search/import", json=candidate)
                    if attempt:
                        retry_successes += 1
                    break
                except Exception as exc:
                    if attempt == 1:
                        failures.append(
                            {
                                "stage": "external_import",
                                "source": candidate["source_id"],
                                "error": repr(exc),
                            }
                        )
                    time.sleep(1)
            if imported is None:
                continue
            paper_id = imported["id"]
            if paper_id in paper_ids:
                continue
            paper_ids.append(paper_id)
            parse_records.append(
                {
                    "paper_id": paper_id,
                    "source": candidate["source_id"],
                    "success": imported["parse_status"] in {"PARSED", "READY"},
                    "duplicate": False,
                    "latency_ms": round(latency, 3),
                }
            )
            try:
                indexed, index_ms = timed(client, "POST", f"/papers/{paper_id}/index")
                index_records.append(
                    {
                        "paper_id": paper_id,
                        "success": indexed["status"] == "READY",
                        "chunk_count": indexed["chunk_count"],
                        "latency_ms": round(index_ms, 3),
                    }
                )
            except Exception as exc:
                failures.append({"stage": "index", "source": paper_id, "error": repr(exc)})
        paper_ids = list(dict.fromkeys(paper_ids))
        memory.append(memory_snapshot("after_ingestion"))
        retrieval_latencies, qa_latencies = [], []
        queries = [
            "main method and architecture",
            "experimental setup datasets and metrics",
            "reported results and comparison with baselines",
            "limitations and future work",
            "retrieval augmented generation evidence",
        ]
        for index in range(100):
            query = f"{queries[index % len(queries)]} run {index % 7}"
            _, latency = timed(
                client,
                "POST",
                "/retrieve",
                json={
                    "query": query,
                    "filters": {"paper_ids": paper_ids},
                    "recall_k": 20,
                    "top_k": 5,
                },
            )
            retrieval_latencies.append(latency)
            _, qa_latency = timed(
                client,
                "POST",
                "/qa",
                json={"question": query, "paper_ids": paper_ids, "top_k": 5},
            )
            qa_latencies.append(qa_latency)
        research_latencies = []
        for query in (
            "Compare the main RAG methods, evidence, and limitations.",
            "What evaluation practices are used and where are the evidence gaps?",
            "Synthesize architectures, results, disagreements, and open problems.",
        ):
            _, latency = timed(
                client,
                "POST",
                "/research/deep",
                json={
                    "query": query,
                    "paper_ids": paper_ids,
                    "allow_external_search": False,
                    "budget": {"max_iterations": 3, "max_papers": TARGET_PAPERS},
                },
            )
            research_latencies.append(latency)
        memory.append(memory_snapshot("after_workload"))
    subprocess.run(
        ["docker", "compose", "restart", "postgres", "qdrant", "redis", "api", "nginx"],
        check=True,
    )
    recovered = False
    with httpx.Client(timeout=5) as client:
        for _ in range(30):
            try:
                health = client.get(f"{BASE}/health")
                recovered = health.status_code == 200 and health.json()["status"] == "healthy"
                if recovered:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(2)
    memory.append(memory_snapshot("after_restart"))
    completed_at = datetime.now(UTC)
    actual_count = len(paper_ids)
    payload = {
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "target_papers": TARGET_PAPERS,
        "actual_unique_papers": actual_count,
        "target_met": 30 <= actual_count <= 50,
        "parse_success_rate": round(
            sum(record["success"] for record in parse_records) / max(1, len(parse_records)), 6
        ),
        "mean_parse_ms": round(
            sum(record["latency_ms"] for record in parse_records) / max(1, len(parse_records)), 3
        ),
        "mean_index_ms": round(
            sum(record["latency_ms"] for record in index_records) / max(1, len(index_records)), 3
        ),
        "retrieval_queries": len(retrieval_latencies),
        "mean_retrieval_ms": round(sum(retrieval_latencies) / len(retrieval_latencies), 3),
        "p95_retrieval_ms": round(p95(retrieval_latencies), 3),
        "mean_qa_ms": round(sum(qa_latencies) / len(qa_latencies), 3),
        "p95_qa_ms": round(p95(qa_latencies), 3),
        "deep_research_runs": len(research_latencies),
        "deep_research_ms": [round(value, 3) for value in research_latencies],
        "failure_count": len(failures),
        "failure_rate": round(len(failures) / max(1, len(parse_records) + len(failures)), 6),
        "retry_attempts": retry_attempts,
        "retry_successes": retry_successes,
        "retry_success_rate": round(retry_successes / max(1, retry_attempts), 6),
        "llm_tokens": 0,
        "estimated_cost_usd": 0.0,
        "service_restart_recovered": recovered,
        "memory_snapshots": memory,
        "memory_growth_conclusion": "inconclusive from discrete docker stats snapshots; no soak test",
        "elapsed_seconds": round((completed_at - started_at).total_seconds(), 3),
        "failures": failures,
        "parse_records": parse_records,
        "index_records": index_records,
    }
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(payload)
    print(json.dumps({key: value for key, value in payload.items() if key not in {"failures", "parse_records", "index_records", "memory_snapshots"}}, ensure_ascii=False, indent=2))


def write_report(result: dict) -> None:
    lines = [
        "# Stability Report v1",
        "",
        f"- Unique papers: {result['actual_unique_papers']} / target {result['target_papers']} (met: {result['target_met']})",
        f"- Parse success rate: {result['parse_success_rate']:.1%}",
        f"- Mean ingest/parse latency: {result['mean_parse_ms']:.3f} ms",
        f"- Mean index latency: {result['mean_index_ms']:.3f} ms",
        f"- Retrievals: {result['retrieval_queries']}; mean {result['mean_retrieval_ms']:.3f} ms; P95 {result['p95_retrieval_ms']:.3f} ms",
        f"- QA: mean {result['mean_qa_ms']:.3f} ms; P95 {result['p95_qa_ms']:.3f} ms",
        f"- Deep Research runs: {result['deep_research_runs']}; latencies {result['deep_research_ms']}",
        f"- Failures: {result['failure_count']} ({result['failure_rate']:.1%})",
        f"- Retry success: {result['retry_successes']}/{result['retry_attempts']} ({result['retry_success_rate']:.1%})",
        f"- Service restart recovered: {result['service_restart_recovered']}",
        f"- Tokens / cost: {result['llm_tokens']} / ${result['estimated_cost_usd']:.2f} (no real LLM)",
        f"- Total elapsed: {result['elapsed_seconds']:.3f} s",
        "",
        "## Memory",
        "",
        f"- Conclusion: {result['memory_growth_conclusion']}",
    ]
    for snapshot in result["memory_snapshots"]:
        lines.append(f"- `{snapshot['label']}`: " + "; ".join(f"{item['Name']}={item['MemUsage']}" for item in snapshot["containers"]))
    lines.extend(["", "## Failures", ""])
    lines.extend(
        [f"- {item['stage']} / {item['source']}: `{item['error']}`" for item in result["failures"]]
        or ["- None"]
    )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "This is a bounded RC workload, not a long-duration soak test. Discrete memory snapshots can identify a large jump but cannot prove the absence of a slow leak.",
            "",
        ]
    )
    REPORT.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
