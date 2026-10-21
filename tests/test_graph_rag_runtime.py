"""Small synthetic contract controls; real PDF/HTTP acceptance is a separate script."""

import copy
import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from paper_research.agents.research_agent.checkpoint import checkpoint_kind
from paper_research.agents.research_agent.verifier import ResearchSafetyError
from paper_research.api.routes.rag import HybridRetrievalRequest
from paper_research.api.routes.research import ResearchAgentRequest
from paper_research.chunking.types import Chunk
from paper_research.evidence_graph.runtime import (
    GraphRuntimeError,
    RuntimeGraph,
    RuntimeSources,
    graph_path,
)
from paper_research.evidence_graph.schema import digest, node_id
from paper_research.retrieval.context_builder import ContextBuilder
from paper_research.retrieval.fusion import FusedResult
from paper_research.retrieval.graph import GraphOptions, augment, neighborhood, scoped_graph
from paper_research.retrieval.hybrid import HybridRetrievalResult
from paper_research.retrieval.trace import RetrievalTrace


@pytest.fixture
def runtime():
    metadata = {p: {"title": p, "pdf_sha256": "a" * 64} for p in ("paper-a", "paper-b")}
    chunks = {}
    for paper in metadata:
        for i in range(3):
            cid = f"{paper}-{i}"
            chunks[cid] = dict(
                chunk_id=cid,
                paper_id=paper,
                block_ids=[f"b{i}"],
                block_page_map={f"b{i}": i + 1},
                section_path=["Methods"],
                page_start=i + 1,
                page_end=i + 1,
                chunk_text=f"Exact canonical text {cid}",
                block_type="paragraph",
                token_count=6,
            )
    sources = RuntimeSources(
        metadata,
        chunks,
        {},
        {},
        {},
        {},
        {
            "runtime:metadata": digest(metadata),
            "runtime:chunks": digest(chunks),
        },
    )
    return RuntimeGraph(sources, {"schema_version": "D6_STRUCTURAL_GRAPH_WIKI_V1"})


def test_real_legacy_queries_reused_without_gold(runtime):
    assert {n.type for n in runtime.graph.nodes.values()} == {"Paper", "Chunk"}
    assert {e.type for e in runtime.graph.edges.values()} == {"CONTAINS"}
    assert all(e.kind == "structural" for e in runtime.graph.edges.values())
    assert runtime.wiki.search("paper-a")
    for page in runtime.wiki.pages.values():
        assert page.evidence_status == "STRUCTURAL_SOURCE"
        assert page.evidence and page.provenance_paths
        for item in page.evidence:
            assert runtime.sources.chunks[item["chunk_id"]]["chunk_text"] == item["text"]
    with pytest.raises(GraphRuntimeError, match="GOLD_FORBIDDEN"):
        runtime.sources.provenance("claims", "forbidden", semantic=True)


def test_scope_and_two_hop_source_expansion(runtime):
    seed = node_id("Chunk", "paper-a", "paper-a-0")
    result = neighborhood(runtime, [seed], ["paper-a"], GraphOptions())
    assert len(result["node_ids"]) == 4
    assert result["paths"][node_id("Chunk", "paper-a", "paper-a-2")] == [
        seed,
        node_id("Paper", "paper-a"),
        node_id("Chunk", "paper-a", "paper-a-2"),
    ]
    assert all(
        runtime.graph.nodes[n].properties["paper_id"] == "paper-a" for n in result["node_ids"]
    )
    with pytest.raises(GraphRuntimeError, match="OUTSIDE_SCOPE"):
        neighborhood(runtime, [seed], ["paper-b"], GraphOptions())
    for invalid in (["unknown"], ["paper-a", "paper-a"]):
        with pytest.raises(GraphRuntimeError, match="NOT_AUTHORIZED"):
            scoped_graph(runtime, invalid)


def test_traversal_budget_and_no_path(runtime):
    seed = node_id("Chunk", "paper-a", "paper-a-0")
    with pytest.raises(GraphRuntimeError, match="BUDGET_EXCEEDED"):
        neighborhood(runtime, [seed], ["paper-a"], GraphOptions(max_nodes=1))
    with pytest.raises(GraphRuntimeError, match="BUDGET_EXCEEDED"):
        neighborhood(runtime, [seed], ["paper-a"], GraphOptions(max_edges=1))
    assert neighborhood(runtime, [], ["paper-a"], GraphOptions())["node_ids"] == []
    assert neighborhood(runtime, [seed], ["paper-a"], GraphOptions(max_hops=0))["node_ids"] == [
        seed
    ]


@pytest.mark.parametrize(
    "options",
    [
        {"max_hops": 3},
        {"max_nodes": True},
        {"max_context_chars": 24001},
        {"guess_missing_source": True},
    ],
)
def test_strict_options(options):
    with pytest.raises(ValidationError):
        GraphOptions.model_validate(options)


def test_missing_index_explicit_two_sided_fallback(monkeypatch):
    def unavailable(*args):
        raise GraphRuntimeError("GRAPH_INDEX_MISSING", 503)

    monkeypatch.setattr(RuntimeGraph, "load", unavailable)
    result = SimpleNamespace()
    settings = SimpleNamespace(graph_rag_enabled=True, graph_rag_allow_hybrid_fallback=False)
    kwargs = dict(settings=settings, registry=None, chunks=[], collection="test")
    with pytest.raises(GraphRuntimeError, match="MISSING"):
        augment(result, "query", None, 5, GraphOptions(fallback_to_hybrid=True), **kwargs)
    settings.graph_rag_allow_hybrid_fallback = True
    with pytest.raises(GraphRuntimeError, match="MISSING"):
        augment(result, "query", None, 5, GraphOptions(), **kwargs)
    assert (
        augment(result, "query", None, 5, GraphOptions(fallback_to_hybrid=True), **kwargs) is result
    )
    assert result.graph_rag["status"] == "FALLBACK_HYBRID"

    def stale(*args):
        raise GraphRuntimeError("GRAPH_INDEX_STALE")

    monkeypatch.setattr(RuntimeGraph, "load", stale)
    with pytest.raises(GraphRuntimeError, match="STALE"):
        augment(result, "query", None, 5, GraphOptions(fallback_to_hybrid=True), **kwargs)


def test_rehashed_topology_and_sourceless_wiki_rejected(tmp_path, monkeypatch, runtime):
    import paper_research.evidence_graph.runtime as module

    monkeypatch.setattr(module, "active_sources", lambda *a: (runtime.sources, runtime.binding))
    settings = SimpleNamespace(data_dir=tmp_path)
    path = graph_path(settings)
    path.parent.mkdir()
    original = runtime.record()
    for change in ("edge", "wiki", "node"):
        record = copy.deepcopy(original)
        if change == "edge":
            record["edges"][0]["target"] = "unapproved"
        elif change == "wiki":
            next(iter(record["wiki_pages"].values()))["evidence"] = []
        else:
            record["nodes"].append({"type": "Claim", "text": "gold"})
        record["graph_digest"] = digest(record)
        path.write_text(json.dumps(record), encoding="utf8")
        with pytest.raises(GraphRuntimeError, match="TOPOLOGY_MISMATCH"):
            RuntimeGraph.load(settings, None, [], "test")
    path.write_text(json.dumps(original), encoding="utf8")
    assert RuntimeGraph.load(settings, None, [], "test").record() == original


def test_default_api_and_checkpoint_opt_in():
    assert HybridRetrievalRequest(query="test").graph_rag is None
    assert ResearchAgentRequest(query="test").graph_rag is None
    for other in ("include_evidence_contract", "multimodal_enabled"):
        with pytest.raises(ValidationError):
            ResearchAgentRequest(query="test", graph_rag={}, **{other: True})
    with pytest.raises(ResearchSafetyError, match="GRAPH_CHECKPOINT_INVALID"):
        checkpoint_kind({"graph_rag": {"options": {}, "requests": []}})


def test_real_fusion_dedup_and_context_budget(monkeypatch, runtime):
    monkeypatch.setattr(RuntimeGraph, "load", lambda *a: runtime)
    chunk = Chunk.model_validate(runtime.sources.chunks["paper-a-0"])
    context = ContextBuilder(include_neighbors=False).build([FusedResult(chunk, 1.0)])
    trace = RetrievalTrace(
        query="canonical",
        filters={},
        dense_results=[],
        sparse_results=[],
        fusion_results=[],
        rerank_results=[],
        final_context=context,
        latency_ms=0,
    )
    baseline = HybridRetrievalResult(context, trace)
    settings = SimpleNamespace(graph_rag_enabled=True, qa_context_token_budget=4000)
    result = augment(
        baseline,
        "canonical",
        ["paper-a"],
        3,
        GraphOptions(max_context_chars=256),
        settings=settings,
        registry=None,
        chunks=[],
        collection="test",
    )
    assert len(result.context) == 3
    assert len({c.chunk_id for c in result.context}) == 3
    assert result.graph_rag["added_chunk_ids"]
    assert result.graph_rag["lost_hybrid_chunk_ids"] == []
    assert all(c.paper_id == "paper-a" for c in result.context)
    assert sum(len(c.evidence) for c in result.context) <= 256
    assert baseline.context == context and baseline.trace.trace_id != result.trace.trace_id
    assert result.graph_rag["model_requests"] == 0


def test_api_rejects_disabled_and_unsupported_filters_before_retrieval(monkeypatch):
    from paper_research.api.routes import rag

    monkeypatch.setattr(rag, "get_settings", lambda: SimpleNamespace(graph_rag_enabled=False))
    with pytest.raises(GraphRuntimeError, match="DISABLED"):
        rag._run_hybrid(HybridRetrievalRequest(query="test", graph_rag={}))
    monkeypatch.setattr(rag, "get_settings", lambda: SimpleNamespace(graph_rag_enabled=True))
    with pytest.raises(GraphRuntimeError, match="NON_PAPER_FILTERS_UNSUPPORTED"):
        rag._run_hybrid(
            HybridRetrievalRequest(query="test", graph_rag={}, filters={"sections": ["Methods"]})
        )
