"""Actual PDF -> D6 -> HTTP Qdrant -> graph/wiki -> HTTP API -> Agent acceptance.

Requires authorized real PDFs and an existing local Qdrant server. No mock data,
benchmark inputs, paid model, new service stack or production collection mutation.
All generated evidence stays in a fresh project-local output directory.
"""

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
import unicodedata
import uuid
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, action="append", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qdrant-url", default="http://127.0.0.1:6333")
    args = parser.parse_args()
    database_url = os.environ.get("GRAPH_ACCEPTANCE_DATABASE_URL", "")
    from urllib.parse import urlsplit

    database = urlsplit(database_url)
    if (
        database.scheme != "postgresql+psycopg"
        or database.hostname not in {"127.0.0.1", "localhost"}
        or not database.path.startswith("/graph_acceptance_")
    ):
        parser.error(
            "Set GRAPH_ACCEPTANCE_DATABASE_URL to a new local "
            "PostgreSQL graph_acceptance_* database"
        )
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if not output.is_relative_to(root) or output.exists():
        parser.error("Use a NEW directory inside this checkout; prior receipts are immutable")
    pdfs = [p.resolve() for p in args.pdf]
    if any(not p.is_file() or p.suffix.lower() != ".pdf" for p in pdfs):
        parser.error("Actual authorized PDF files required")
    if not args.qdrant_url.startswith(("http://127.0.0.1:", "http://localhost:")):
        parser.error("Acceptance is limited to a local Qdrant endpoint")
    output.mkdir(parents=True)
    # Runner's existing default lock is repository-relative. Preserve its exact
    # bytes in the isolated working directory; never fabricate a replacement.
    lock_relative = Path("data/evaluation/research-agent/stage3-rag-backend-lock-v1.json")
    lock_target = output / lock_relative
    lock_target.parent.mkdir(parents=True)
    lock_target.write_bytes((root / lock_relative).read_bytes())
    os.chdir(output)
    data = output / "data"
    data.mkdir(exist_ok=True)
    collection = "graph_wiki_v21_acceptance_" + uuid.uuid4().hex[:12]
    os.environ.update(
        {
            "DATABASE_URL": database_url,
            "DATA_DIR": str(data),
            "QDRANT_URL": args.qdrant_url,
            "QDRANT_COLLECTION": collection,
            "BASELINE_COLLECTION": collection,
            "APP_PROFILE": "baseline",
            "PARSER_BACKEND": "pymupdf",
            "EMBEDDING_PROVIDER": "hash",
            "EMBEDDING_MODEL": "hash-v1",
            "EMBEDDING_REVISION": "v1",
            "EMBEDDING_DIMENSIONS": "384",
            "RERANK_ENABLED": "false",
            "LLM_PROVIDER": "template",
            "LIVE_MODEL_CALLS_ENABLED": "false",
            "RESEARCH_DRAFT_ENABLED": "false",
            "GRAPH_RAG_ENABLED": "true",
            "GRAPH_RAG_ALLOW_HYBRID_FALLBACK": "true",
            "INDEX_VERSION": "graph-v21",
            "DATASET_VERSION": "authorized-runtime-pdfs",
            "RETRIEVAL_TRACE_PATH": str(output / "retrieval.jsonl"),
            "PAGE_ASSET_DPI": "36",
            "CHUNK_MAX_TOKENS": "300",
            "CHUNK_OVERLAP_TOKENS": "40",
            "PYTHONPATH": os.pathsep.join([str(root / "src"), str(root)]),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONIOENCODING": "utf-8",
            "REDIS_URL": "redis://127.0.0.1:1/0",
        }
    )
    sys.path[:0] = [str(root / "src"), str(root)]
    # Audit forbidden benchmark reads, including in the actual API subprocess.
    guard = """
import os, sys
def deny_gold(event,args):
    if event == 'open' and args and isinstance(args[0], (str,bytes,os.PathLike)):
        name=os.fsdecode(args[0]).replace(chr(92),'/').lower()
        forbidden = ('/phase2b/benchmark/', '/phase1q-r2/corpus/',
                     'gold_claims.jsonl', 'gold_evidence.jsonl')
        if any(s in name for s in forbidden):
            raise RuntimeError('BENCHMARK_GOLD_IO_FORBIDDEN')
sys.addaudithook(deny_gold)
"""
    exec(guard)
    import httpx
    import pymupdf as fitz

    import paper_research.models  # noqa: F401
    from paper_research import db
    from paper_research.admission import admit
    from paper_research.agents.providers import HybridLocalResearchProvider
    from paper_research.config import get_settings
    from paper_research.evidence_graph.runtime import build_active, graph_path

    started = time.perf_counter()
    records = {
        "kind": "REAL_PDF_RUNTIME_INTEGRATION_NOT_SCIENTIFIC_GOLD",
        "query": args.query,
        "collection": collection,
        "pdfs": [{"sha256": sha(p), "bytes": p.stat().st_size} for p in pdfs],
        "model_requests": 0,
        "neural_model_forwards": 0,
        "paid_cost_usd": 0,
        "benchmark_read_guard": True,
        "tests": [],
    }

    def put(name, value):
        (output / name).write_text(
            json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf8"
        )

    process = None
    try:
        from sqlalchemy import inspect

        if inspect(db.engine).get_table_names():
            raise RuntimeError("Acceptance requires an empty dedicated PostgreSQL database")
        db.Base.metadata.create_all(db.engine)
        settings = get_settings()
        records["admission"] = admit(
            settings, pdfs, "user-authorized-graph-wiki-runtime-local-integration"
        )
        assert records["admission"]["status"] == "ACTIVE"
        provider = HybridLocalResearchProvider(settings, require_identity=True)
        try:
            records["build"] = build_active(provider)
        finally:
            provider._client.close()
        put("build_receipt.json", records["build"])
        graph_file = graph_path(settings)
        graph_bytes = graph_file.read_bytes()
        graph_record = json.loads(graph_bytes)
        assert all(
            n["type"] in {"Paper", "Chunk"}
            and n["provenance"]["source_phase"] == "D6_ACTIVE_CANONICAL"
            for n in graph_record["nodes"]
        )
        assert all(e["kind"] == "structural" for e in graph_record["edges"])
        docs = {
            d["paper_id"]: d
            for d in provider._registry.authority_bundle(records["admission"]["generation_id"])[
                "commitment"
            ]["documents"]
        }
        paper_ids = list(docs)
        # Bind an ephemeral local HTTP port; no production API configuration changes.
        with socket.socket() as port_socket:
            port_socket.bind(("127.0.0.1", 0))
            port = port_socket.getsockname()[1]
        boot = (
            guard
            + "\nimport uvicorn\nuvicorn.run('paper_research.main:app', host='127.0.0.1', port="
            + str(port)
            + ", log_level='warning')"
        )
        log = (output / "api.log").open("wb")
        process = subprocess.Popen(
            [sys.executable, "-B", "-c", boot],
            cwd=output,
            env=os.environ.copy(),
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        records["api_pid"] = process.pid
        client = httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=120)
        for _ in range(60):
            if process.poll() is not None:
                raise RuntimeError("API_PROCESS_EXITED: inspect api.log")
            try:
                if client.get("/").status_code == 200:
                    break
            except httpx.TransportError:
                pass
            time.sleep(0.25)
        else:
            raise RuntimeError("API_START_TIMEOUT")
        request = {"query": args.query, "filters": {"paper_ids": paper_ids}, "top_k": 5}

        def call(path, payload, name):
            t = time.perf_counter()
            response = client.post(path, json=payload)
            receipt = {
                "path": path,
                "request": payload,
                "http_status": response.status_code,
                "wall_ms": (time.perf_counter() - t) * 1000,
                "body": response.json(),
            }
            put(name + ".json", receipt)
            return receipt

        baseline = call("/api/v1/retrieve", request, "hybrid")
        enhanced = call("/api/v1/retrieve", {**request, "graph_rag": {}}, "graphrag")
        assert baseline["http_status"] == enhanced["http_status"] == 200, (baseline, enhanced)
        assert "graph_rag" not in baseline["body"]
        assert enhanced["body"]["graph_rag"]["status"] == "USED"
        assert not enhanced["body"]["evidence_contract"]["errors"]
        seed = enhanced["body"]["context"][0]
        linked = call(
            "/api/v1/graph/query",
            {"paper_ids": [seed["paper_id"]], "seed_chunk_ids": [seed["chunk_id"]]},
            "graph-query",
        )
        wiki = call(
            "/api/v1/wiki/query",
            {"paper_ids": [seed["paper_id"]], "paper_id": seed["paper_id"]},
            "wiki-query",
        )
        assert linked["http_status"] == wiki["http_status"] == 200, (linked, wiki)
        assert len(linked["body"]["traversal"]["node_ids"]) > 2
        assert wiki["body"]["bundle"]["page"]["evidence_status"] == "STRUCTURAL_SOURCE"
        assert wiki["body"]["bundle"]["evidence"]
        agent = call(
            "/api/v1/research/agent",
            {
                "query": args.query,
                "paper_ids": paper_ids,
                "graph_rag": {},
                "generate_draft": False,
                "pause_after_phase": "STATE_UPDATED",
                "budget": {"max_steps": 8, "max_tool_calls": 6, "max_provider_requests": 0},
            },
            "agent",
        )
        assert agent["http_status"] == 200, agent
        assert agent["body"]["status"] == "PAUSED"
        agent = call(
            f"/api/v1/research/agent/{agent['body']['task_id']}/resume", {}, "agent-resumed"
        )
        assert agent["http_status"] == 200, agent
        assert agent["body"]["terminal"]
        records["agent_resume"] = {
            "status": agent["body"]["status"],
            "graph_options_preserved": True,
        }
        active = agent["body"]["active_provenance"]
        assert active["graph_rag"]["requests"] and active["evidence"]
        assert any(r["status"] == "USED" for r in active["graph_rag"]["requests"])
        # Every returned citation is checked against real parser blocks and PDF pages.
        by_sha = {sha(p): p for p in pdfs}

        def normal(s):
            return "".join(unicodedata.normalize("NFKC", s).split())

        verified = []
        contexts = list(enhanced["body"]["context"]) + linked["body"]["evidence"]
        contexts += [
            {
                "paper_id": i["paper_id"],
                "chunk_id": i["chunk_id"],
                "block_ids": i["block_ids"],
                "evidence": i["text_or_reference"],
            }
            for i in active["evidence"]
        ]
        contexts += [
            {
                "paper_id": i["paper_id"],
                "chunk_id": i["chunk_id"],
                "block_ids": i["block_ids"],
                "evidence": i["text"],
            }
            for i in wiki["body"]["bundle"]["evidence"]
        ]
        for item in contexts:
            pid = item["paper_id"]
            folder = settings.parsed_papers_dir / pid
            chunks = {
                r["chunk_id"]: r
                for r in map(
                    json.loads,
                    (folder / "paper_chunks.graph-v21.jsonl").read_text("utf8").splitlines(),
                )
            }
            chunk = chunks[item["chunk_id"]]
            assert chunk["chunk_text"].startswith(item["evidence"])
            blocks = {
                r["block_id"]: r
                for r in map(
                    json.loads, (folder / "paper_blocks.jsonl").read_text("utf8").splitlines()
                )
            }
            with fitz.open(by_sha[docs[pid]["pdf_sha256"]]) as pdf:
                for block_id in item["block_ids"]:
                    block = blocks[block_id]
                    pages = range(block["page_start"] - 1, block["page_end"])
                    page_text = "".join(pdf[p].get_text() for p in pages)
                    assert normal(block["text"]) in normal(page_text), (
                        block_id,
                        block["page_start"],
                    )
            verified.append(
                {
                    "paper_id": pid,
                    "chunk_id": item["chunk_id"],
                    "pdf_sha256": docs[pid]["pdf_sha256"],
                    "pages": [chunk["page_start"], chunk["page_end"]],
                    "block_ids": item["block_ids"],
                    "quote_sha256": hashlib.sha256(item["evidence"].encode()).hexdigest(),
                }
            )
        put("citation_checks.json", verified)
        records["citation_occurrences_verified"] = len(verified)
        records["agent"] = {
            "status": agent["body"]["status"],
            "tool_calls": agent["body"]["tool_call_count"],
            "graph_requests": len(active["graph_rag"]["requests"]),
            "evidence_count": len(active["evidence"]),
            "paid_provider_requests": agent["body"]["provider_call_count"],
            "report_status": agent["body"]["report_status"],
        }

        def negative(name, path, payload, *, mutate=None, expect=None):
            try:
                if mutate:
                    edited = json.loads(graph_bytes)
                    mutate(edited)
                    graph_file.write_text(json.dumps(edited), encoding="utf8")
                row = call(path, payload, "negative-" + name)
                assert row["http_status"] != 200 and (
                    expect is None or row["http_status"] == expect
                ), row
                records["tests"].append(
                    {
                        "name": name,
                        "status": "PASS_REJECTED",
                        "http_status": row["http_status"],
                        "reason": row["body"],
                    }
                )
            finally:
                graph_file.write_bytes(graph_bytes)

        negative(
            "unauthorized-paper",
            "/api/v1/retrieve",
            {**request, "filters": {"paper_ids": [*paper_ids, str(uuid.uuid4())]}, "graph_rag": {}},
            expect=403,
        )
        negative(
            "stale-generation",
            "/api/v1/retrieve",
            {**request, "graph_rag": {}},
            mutate=lambda g: g["binding"].update(generation_id=str(uuid.uuid4())),
        )
        negative(
            "wrong-edge",
            "/api/v1/retrieve",
            {**request, "graph_rag": {}},
            mutate=lambda g: g["edges"][0].update(target="chunk:invalid"),
        )
        negative(
            "orphan-node",
            "/api/v1/retrieve",
            {**request, "graph_rag": {}},
            mutate=lambda g: g["edges"].pop(),
        )
        negative(
            "sourceless-wiki",
            "/api/v1/wiki/query",
            {"paper_ids": paper_ids, "paper_id": seed["paper_id"]},
            mutate=lambda g: next(iter(g["wiki_pages"].values())).update(evidence=[]),
        )
        negative(
            "gold-schema",
            "/api/v1/retrieve",
            {**request, "graph_rag": {}},
            mutate=lambda g: g.update(schema_version="PHASE2B_FROZEN_60_V1"),
        )
        negative(
            "extra-gold-node",
            "/api/v1/retrieve",
            {**request, "graph_rag": {}},
            mutate=lambda g: g["nodes"].append({"type": "Claim", "text": "not a runtime source"}),
        )
        negative(
            "node-budget",
            "/api/v1/graph/query",
            {
                "paper_ids": [seed["paper_id"]],
                "seed_chunk_ids": [seed["chunk_id"]],
                "options": {"max_nodes": 1},
            },
            expect=422,
        )
        negative(
            "hop-budget",
            "/api/v1/graph/query",
            {
                "paper_ids": [seed["paper_id"]],
                "seed_chunk_ids": [seed["chunk_id"]],
                "options": {"max_hops": 3},
            },
            expect=422,
        )
        # Identity tampering is a D6 rejection, not merely an index checksum failure.
        canonical_file = (
            settings.parsed_papers_dir / seed["paper_id"] / "paper_chunks.graph-v21.jsonl"
        )
        saved = canonical_file.read_bytes()
        try:
            rows = [json.loads(line) for line in saved.splitlines()]
            rows[0]["chunk_text"] += " modified identity"
            canonical_file.write_text(
                "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf8"
            )
            negative("canonical-tampering", "/api/v1/retrieve", {**request, "graph_rag": {}})
        finally:
            canonical_file.write_bytes(saved)
        # Scoped queries must not return the other approved paper either.
        assert all(e["paper_id"] == seed["paper_id"] for e in linked["body"]["evidence"])
        graph_file.write_bytes(b"not-json")
        negative("corrupt-runtime", "/api/v1/retrieve", {**request, "graph_rag": {}})
        missing = graph_file.with_suffix(".held")
        graph_file.rename(missing)
        try:
            row = call("/api/v1/retrieve", {**request, "graph_rag": {}}, "negative-missing-index")
            assert row["http_status"] == 503
            fallback = call(
                "/api/v1/retrieve",
                {**request, "graph_rag": {"fallback_to_hybrid": True}},
                "explicit-fallback",
            )
            assert (
                fallback["http_status"] == 200
                and fallback["body"]["graph_rag"]["status"] == "FALLBACK_HYBRID"
            )
            records["tests"].append(
                {
                    "name": "missing-index-and-explicit-fallback",
                    "status": "PASS",
                    "strict_status": 503,
                }
            )
        finally:
            missing.rename(graph_file)

        def ids(row):
            return [r["chunk_id"] for r in row["body"]["context"]]

        records["comparison"] = {
            "scope": "ONE_QUESTION_ENGINEERING_OBSERVATION_NOT_QUALITY_BENCHMARK",
            "hybrid_wall_ms": baseline["wall_ms"],
            "graph_wall_ms": enhanced["wall_ms"],
            "hybrid_chunk_ids": ids(baseline),
            "graph_chunk_ids": ids(enhanced),
            "graph_added": sorted(set(ids(enhanced)) - set(ids(baseline))),
            "hybrid_lost": sorted(set(ids(baseline)) - set(ids(enhanced))),
            "graph_receipt": enhanced["body"]["graph_rag"],
            "graph_query_evidence_units": len(linked["body"]["evidence"]),
            "wiki_evidence_units": len(wiki["body"]["bundle"]["evidence"]),
        }
        records["status"] = "PAPERRESEARCH_GRAPHRAG_RUNTIME_INTEGRATED"
        records["runtime_index_sha256"] = sha(graph_file)
        records["runtime_index_bytes"] = graph_file.stat().st_size
        records["scientific_multi_hop_accuracy_qualified"] = False
    except Exception as exc:
        records["status"] = "INCOMPLETE"
        records["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            records["api_stopped"] = process.poll() is not None
            log.close()
            client.close()
        records["wall_seconds"] = time.perf_counter() - started
        put("acceptance.json", records)
    print(
        json.dumps(
            {
                "status": records["status"],
                "output": str(output),
                "comparison": records.get("comparison"),
                "negative_checks": len(records["tests"]),
                "model_requests": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
