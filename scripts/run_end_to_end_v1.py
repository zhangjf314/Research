# ruff: noqa: E501
"""Execute the fixed-topic RC demo against the deployed HTTP API."""

import json
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

BASE_URL = "http://localhost/api/v1"
TOPIC = "retrieval augmented generation for scientific literature review: methods evaluation and limitations"
REPORT = Path("artifacts/demo-research-report-v1.md")
TRACE = Path("artifacts/demo-trace-v1.json")
DOC = Path("docs/end-to-end-run-v1.md")


def request(client: httpx.Client, trace: dict, method: str, path: str, **kwargs) -> dict:
    started = time.perf_counter()
    response = client.request(method, f"{BASE_URL}{path}", **kwargs)
    elapsed = round((time.perf_counter() - started) * 1000, 3)
    trace["tool_calls"].append(
        {"method": method, "path": path, "status_code": response.status_code, "latency_ms": elapsed}
    )
    response.raise_for_status()
    return response.json()


def main() -> None:
    started_at = datetime.now(UTC)
    trace = {
        "run_id": f"e2e-v1-{started_at.strftime('%Y%m%dT%H%M%SZ')}",
        "topic": TOPIC,
        "started_at": started_at.isoformat(),
        "providers": {
            "query_rewriter": "deterministic QueryRewriter",
            "external_search": ["arXiv Atom API", "Semantic Scholar Graph API"],
            "pdf_download": "httpx via CachedRetryClient",
            "parser": "ParserRouter (PyMuPDF baseline; OCR fallback when routed)",
            "embedding": "HashEmbeddingProvider(dimensions=384)",
            "vector_store": "Qdrant HTTP",
            "sparse": "BM25Retriever",
            "fusion": "RRF(k=60)",
            "reranker": "LexicalReranker",
            "report": "deterministic evidence-template generator (no LLM)",
            "workflow": "LangGraph with InMemorySaver",
        },
        "tool_calls": [],
        "search": {},
        "imports": [],
        "retrievals": [],
        "deep_research": {},
    }
    with httpx.Client(timeout=180) as client:
        search = request(
            client,
            trace,
            "POST",
            "/search/papers",
            json={"query": TOPIC, "limit": 8, "open_access_only": True},
        )
        trace["search"] = {
            "rewritten_queries": search["rewritten_queries"],
            "candidate_count": len(search["candidates"]),
            "source_errors": search.get("source_errors", {}),
        }
        selected = [candidate for candidate in search["candidates"] if candidate.get("pdf_url")][:3]
        paper_ids = []
        for candidate in selected:
            imported = request(client, trace, "POST", "/search/import", json=candidate)
            paper_id = imported["id"]
            paper_ids.append(paper_id)
            indexed = request(client, trace, "POST", f"/papers/{paper_id}/index")
            trace["imports"].append(
                {
                    "paper_id": paper_id,
                    "source": candidate["source"],
                    "source_id": candidate["source_id"],
                    "title": candidate["title"],
                    "parse_status": imported["parse_status"],
                    "index_status": indexed["status"],
                    "chunk_count": indexed["chunk_count"],
                }
            )
        retrieval_queries = [
            f"{TOPIC} main methods",
            f"{TOPIC} evaluation results",
            f"{TOPIC} limitations",
        ]
        for query in retrieval_queries:
            result = request(
                client,
                trace,
                "POST",
                "/retrieve",
                json={
                    "query": query,
                    "filters": {"paper_ids": paper_ids},
                    "recall_k": 20,
                    "top_k": 5,
                },
            )
            trace["retrievals"].append(
                {
                    "query": query,
                    "trace_id": result["trace"]["trace_id"],
                    "context_count": len(result["context"]),
                    "latency_ms": result["trace"]["latency_ms"],
                    "pages": sorted({item["page_start"] for item in result["context"]}),
                }
            )
        deep = request(
            client,
            trace,
            "POST",
            "/research/deep",
            json={
                "query": TOPIC,
                "paper_ids": paper_ids,
                "allow_external_search": False,
                "allow_external_import": False,
                "budget": {
                    "max_iterations": 3,
                    "max_external_searches": 1,
                    "max_papers": 3,
                    "max_evidence_items": 40,
                    "max_estimated_tokens": 30000,
                    "max_no_new_evidence_rounds": 2,
                },
            },
        )
    completed_at = datetime.now(UTC)
    citations = deep.get("citation_results", [])
    trace["deep_research"] = {
        "task_id": deep["task_id"],
        "status": deep["status"],
        "stop_reason": deep["stop_reason"],
        "subquestion_count": len(deep["sub_questions"]),
        "evidence_gap_count": len(deep["evidence_gaps"]),
        "iteration_count": deep["node_history"].count("local_search"),
        "node_history": deep["node_history"],
        "citation_count": len(citations),
        "valid_citation_count": sum(bool(item.get("valid")) for item in citations),
    }
    trace["metrics"] = {
        "paper_count": len(trace["imports"]),
        "parse_success_rate": round(
            sum(item["parse_status"] in {"PARSED", "READY"} for item in trace["imports"])
            / max(1, len(trace["imports"])),
            6,
        ),
        "indexed_chunk_count": sum(item["chunk_count"] for item in trace["imports"]),
        "retrieval_rounds": len(trace["retrievals"]),
        "workflow_iterations": trace["deep_research"]["iteration_count"],
        "tool_call_count": len(trace["tool_calls"]),
        "llm_input_tokens": 0,
        "llm_output_tokens": 0,
        "estimated_cost_usd": 0.0,
        "elapsed_seconds": round((completed_at - started_at).total_seconds(), 3),
        "citation_check_pass_rate": round(
            trace["deep_research"]["valid_citation_count"] / max(1, len(citations)), 6
        ),
    }
    trace["completed_at"] = completed_at.isoformat()
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(deep["report"], encoding="utf-8")
    TRACE.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    write_doc(trace)
    print(json.dumps(trace["metrics"], ensure_ascii=False, indent=2))


def write_doc(trace: dict) -> None:
    metrics = trace["metrics"]
    errors = trace["search"]["source_errors"] or {"none": "none"}
    lines = [
        "# End-to-End Run v1",
        "",
        f"- Run ID: `{trace['run_id']}`",
        f"- Fixed topic: {trace['topic']}",
        f"- Started: `{trace['started_at']}`",
        f"- Completed: `{trace['completed_at']}`",
        f"- Papers imported/indexed: {metrics['paper_count']}",
        f"- Parse success rate: {metrics['parse_success_rate']:.1%}",
        f"- Indexed chunks: {metrics['indexed_chunk_count']}",
        f"- Retrieval rounds: {metrics['retrieval_rounds']}",
        f"- LangGraph local-search iterations: {metrics['workflow_iterations']}",
        f"- Tool calls: {metrics['tool_call_count']}",
        "- LLM tokens / cost: 0 / $0.00 (no real LLM is configured)",
        f"- Elapsed: {metrics['elapsed_seconds']} s",
        f"- Citation validation pass rate: {metrics['citation_check_pass_rate']:.1%}",
        "",
        "## Provider truth table",
        "",
    ]
    lines.extend(f"- {name}: {value}" for name, value in trace["providers"].items())
    lines.extend(["", "## External source errors", ""])
    lines.extend(f"- {name}: `{message}`" for name, message in errors.items())
    lines.extend(
        [
            "",
            "## Acceptance boundary",
            "",
            "The download, PDF parsing, indexing, Qdrant storage, hybrid retrieval, reranking, LangGraph execution, report file, and citation-marker validation are real. Query rewriting, embedding, reranking, and report generation are deterministic baselines. This run therefore validates system wiring, not production-model answer quality.",
            "",
        ]
    )
    DOC.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
