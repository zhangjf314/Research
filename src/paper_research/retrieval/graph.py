"""Explicit post-Hybrid structural expansion, bounded traversal and source-only fusion."""

import time
import uuid
from collections import deque

from pydantic import BaseModel, ConfigDict, Field

from paper_research.chunking.types import Chunk
from paper_research.evidence_graph.runtime import GraphRuntimeError, RuntimeGraph
from paper_research.evidence_graph.schema import node_id

from .context_builder import ContextBuilder
from .fusion import FusedResult
from .hybrid import HybridRetrievalResult
from .sparse import BM25Retriever


class GraphOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    max_hops: int = Field(default=2, ge=0, le=2)
    max_nodes: int = Field(default=256, ge=1, le=256)
    max_edges: int = Field(default=256, ge=1, le=512)
    max_context_chars: int = Field(default=16000, ge=256, le=24000)
    fallback_to_hybrid: bool = False


class GraphAgentRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    options: GraphOptions
    requests: list[dict] = Field(default_factory=list, max_length=200)


def scoped_graph(runtime, paper_ids):
    allowed = set(runtime.sources.metadata)
    if paper_ids and (len(set(paper_ids)) != len(paper_ids) or not set(paper_ids) <= allowed):
        raise GraphRuntimeError("GRAPH_PAPER_SCOPE_NOT_AUTHORIZED", 403)
    return set(paper_ids or allowed)


def neighborhood(runtime, seeds, paper_ids, options):
    graph = runtime.graph
    scope = scoped_graph(runtime, paper_ids)
    queue, seen, edges, paths = deque(), set(), set(), {}
    for seed in seeds:
        graph.require(seed)
        if graph.nodes[seed].properties["paper_id"] not in scope:
            raise GraphRuntimeError("GRAPH_SEED_OUTSIDE_SCOPE", 403)
        queue.append((seed, 0))
        seen.add(seed)
        paths[seed] = [seed]
    if len(seen) > options.max_nodes:
        raise GraphRuntimeError("GRAPH_QUERY_BUDGET_EXCEEDED", 422)
    while queue:
        identity, depth = queue.popleft()
        if depth >= options.max_hops:
            continue
        for edge in sorted(
            [*graph.outgoing[identity], *graph.incoming[identity]], key=lambda e: e.id
        ):
            target = edge.target if edge.source == identity else edge.source
            if graph.nodes[target].properties["paper_id"] not in scope:
                continue
            edges.add(edge.id)
            if len(edges) > options.max_edges:
                raise GraphRuntimeError("GRAPH_QUERY_BUDGET_EXCEEDED", 422)
            if target not in seen:
                if len(seen) >= options.max_nodes:
                    raise GraphRuntimeError("GRAPH_QUERY_BUDGET_EXCEEDED", 422)
                seen.add(target)
                paths[target] = [*paths[identity], target]
                queue.append((target, depth + 1))
    # Reuse existing bounded-neighborhood semantics after the stricter resource guard.
    queried = set(seeds)
    for seed in seeds:
        queried.update(graph.neighbors(seed, depth=options.max_hops)["node_ids"])
    if queried != seen:
        raise GraphRuntimeError("GRAPH_SCOPE_OR_TRAVERSAL_MISMATCH")
    return {
        "node_ids": sorted(seen),
        "edge_ids": sorted(edges),
        "paths": {k: paths[k] for k in sorted(paths)},
        "nodes_visited": len(seen),
        "edges_visited": len(edges),
        "max_hops": options.max_hops,
    }


def augment(result, query, paper_ids, top_k, options, *, settings, registry, chunks, collection):
    started = time.perf_counter()
    if not settings.graph_rag_enabled:
        raise GraphRuntimeError("GRAPH_RAG_DISABLED", 503)
    try:
        runtime = RuntimeGraph.load(settings, registry, chunks, collection)
        scope = scoped_graph(runtime, paper_ids)
        canonical = {
            cid: Chunk.model_validate(c)
            for cid, c in runtime.sources.chunks.items()
            if c["paper_id"] in scope
        }
        seeds = [node_id("Chunk", c.paper_id, c.chunk_id) for c in result.context[:5]]
        trace = neighborhood(runtime, seeds, paper_ids, options)
        expanded = [
            canonical[runtime.graph.nodes[n].properties["chunk_id"]]
            for n in trace["node_ids"]
            if runtime.graph.nodes[n].type == "Chunk"
        ]
        lexical = BM25Retriever(expanded).retrieve(query, top_k=len(expanded)) if expanded else []
        ranking = [r.chunk.chunk_id for r in lexical]
        ranking += [c.chunk_id for c in expanded if c.chunk_id not in set(ranking)]
        baseline = [c.chunk_id for c in result.context]
        scores = {}
        for channel in (baseline, ranking):
            for rank, cid in enumerate(channel, 1):
                scores[cid] = scores.get(cid, 0) + 1 / (60 + rank)
        fused = [
            FusedResult(canonical[cid], scores[cid], None, None)
            for cid in sorted(scores, key=lambda x: (-scores[x], x))
        ]
        builder = ContextBuilder(
            include_neighbors=False,
            max_characters=min(options.max_context_chars, settings.qa_context_token_budget * 4),
            max_tokens=settings.qa_context_token_budget,
        )
        context = builder.build(fused[:top_k])
        receipt = {
            "status": "USED" if seeds else "NO_GRAPH_PATH",
            "schema_version": runtime.binding["schema_version"],
            "binding": runtime.binding,
            "graph_digest": runtime.graph.graph_digest,
            "relation_kind": "DETERMINISTIC_CANONICAL_CONTAINMENT_NOT_SEMANTIC_SUPPORT",
            "hybrid_trace_id": result.trace.trace_id,
            "options": options.model_dump(),
            "traversal": trace,
            "hybrid_chunk_ids": baseline,
            "graph_ranked_chunk_ids": ranking,
            "final_chunk_ids": [c.chunk_id for c in context],
            "added_chunk_ids": [c.chunk_id for c in context if c.chunk_id not in baseline],
            "lost_hybrid_chunk_ids": [
                cid for cid in baseline if cid not in {c.chunk_id for c in context}
            ],
            "context_characters": sum(len(c.evidence) for c in context),
            "context_budget": builder.max_characters,
            "context_truncated_chunk_id": builder.last_trace.truncated_chunk_id,
            "model_forwards": 0,
            "model_requests": 0,
            "latency_ms": (time.perf_counter() - started) * 1000,
        }
        final = result.trace.model_copy(
            deep=True,
            update={
                "trace_id": str(uuid.uuid4()),
                "final_context": context,
                "context_strategy": builder.last_trace,
                "final_top_k": len(context),
                "latency_ms": result.trace.latency_ms + receipt["latency_ms"],
                "stage_latency_ms": {
                    **result.trace.stage_latency_ms,
                    "graph_rag": receipt["latency_ms"],
                },
            },
        )
        receipt["final_trace_id"] = final.trace_id
        enriched = HybridRetrievalResult(context, final)
        enriched.graph_rag = receipt
        return enriched
    except GraphRuntimeError as exc:
        # Source/topology/scope failures never fall back; only missing/unavailable graph does.
        if (
            exc.code not in {"GRAPH_INDEX_MISSING", "GRAPH_INDEX_UNAVAILABLE"}
            or not options.fallback_to_hybrid
            or not settings.graph_rag_allow_hybrid_fallback
        ):
            raise
        result.graph_rag = {
            "status": "FALLBACK_HYBRID",
            "reason": exc.code,
            "options": options.model_dump(),
            "model_requests": 0,
            "latency_ms": (time.perf_counter() - started) * 1000,
        }
        return result
