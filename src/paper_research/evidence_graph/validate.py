"""Fail closed on identity, exact source text, provenance and mapping completeness."""

from collections import Counter

from .builder import BENCHMARK, SourceData, span_id, validate_sources
from .schema import Edge, Node, canonical, edge_id, node_id


def validate_graph(nodes: list[Node], edges: list[Edge], d: SourceData) -> dict:
    validate_sources(d)
    if len({n.id for n in nodes}) != len(nodes):
        raise ValueError("DUPLICATE_NODE")
    if len({e.id for e in edges}) != len(edges):
        raise ValueError("DUPLICATE_EDGE")
    index = {n.id: n for n in nodes}
    expected_nodes = {}
    expected_edges = set()

    def expected_node(kind, identity, properties, provenance):
        expected_nodes[identity] = (kind, properties, provenance.model_dump())

    def expected_edge(kind, source, target, provenance):
        expected_edges.add((kind, source, target, canonical(provenance.model_dump())))

    for paper, metadata in d.metadata.items():
        expected_node(
            "Paper",
            node_id("Paper", paper),
            {"paper_id": paper, **metadata},
            d.provenance("metadata", paper),
        )
    for chunk in d.chunks.values():
        cid = node_id("Chunk", chunk["paper_id"], chunk["chunk_id"])
        p = d.provenance("chunks", chunk["chunk_id"], chunk=chunk)
        expected_node("Chunk", cid, chunk, p)
        expected_edge("CONTAINS", node_id("Paper", chunk["paper_id"]), cid, p)
    origins = {}
    for task, row in sorted(d.evidence.items()):
        for e in sorted(row["evidence"], key=lambda e: e["evidence_id"]):
            c = d.chunks[e["chunk_id"]]
            origins.setdefault(span_id(c), (task, e["evidence_id"], c))
    for sid, (task, eid, c) in origins.items():
        p = d.provenance("evidence", task + "/" + eid, chunk=c)
        expected_node(
            "EvidenceSpan",
            sid,
            {
                "paper_id": c["paper_id"],
                "chunk_id": c["chunk_id"],
                "start": 0,
                "end": len(c["chunk_text"]),
                "text": c["chunk_text"],
                "page_start": c["page_start"],
                "page_end": c["page_end"],
                "block_ids": c["block_ids"],
            },
            p,
        )
        expected_edge("CONTAINS", node_id("Chunk", c["paper_id"], c["chunk_id"]), sid, p)
        expected_edge("SOURCED_FROM", sid, node_id("Paper", c["paper_id"]), p)
    for task, q in d.questions.items():
        qid = node_id("Question", BENCHMARK, task)
        expected_node("Question", qid, q, d.provenance("questions", task))
        for kind, key, group, idkey in (
            ("Requirement", "requirements", "requirements", "requirement_id"),
            ("Claim", "claims", "claims", "claim_id"),
        ):
            for record in getattr(d, key)[task][group]:
                identity = node_id(kind, BENCHMARK, task, record[idkey])
                p = d.provenance(key, task + "/" + record[idkey])
                expected_node(kind, identity, {"sample_id": task, **record}, p)
                if kind == "Requirement":
                    expected_edge("REQUIRES", qid, identity, p)
                for eid in record["evidence_ids"]:
                    chunk = d.chunks[eid]
                    expected_edge(
                        "GROUNDED_BY",
                        identity,
                        span_id(chunk),
                        d.provenance(key, task + "/" + record[idkey], semantic=True, chunk=chunk),
                    )
    for n in nodes:
        Node.model_validate(n.model_dump())
        if n.id not in expected_nodes:
            raise ValueError("UNKNOWN_NODE_IDENTITY_OR_REQUIREMENT_OR_CLAIM")
        if n.type == "EvidenceSpan":
            p = n.properties
            chunk = d.chunks.get(p.get("chunk_id"))
            if chunk is None:
                raise ValueError("BAD_CHUNK_ID")
            if (
                type(p.get("start")) is not int
                or type(p.get("end")) is not int
                or not 0 <= p["start"] < p["end"] <= len(chunk["chunk_text"])
            ):
                raise ValueError("BAD_SPAN_OFFSET")
            if p.get("text") != chunk["chunk_text"][p["start"] : p["end"]]:
                raise ValueError("SPAN_TEXT_MISMATCH")
        kind, props, provenance = expected_nodes[n.id]
        if n.type != kind or canonical(n.properties) != canonical(props):
            raise ValueError("NODE_SOURCE_RECORD_MISMATCH")
        if n.provenance.model_dump() != provenance:
            raise ValueError("INVALID_NODE_PROVENANCE")
    if set(index) != set(expected_nodes):
        raise ValueError("MISSING_SOURCE_NODE")
    observed_edges = set()
    for e in edges:
        Edge.model_validate(e.model_dump())
        if e.source not in index or e.target not in index:
            raise ValueError("DANGLING_EDGE")
        if e.id != edge_id(e) or e.properties:
            raise ValueError("EDGE_ID_OR_PROPERTIES_MISMATCH")
        signature = (e.type, e.source, e.target, canonical(e.provenance.model_dump()))
        if signature not in expected_edges:
            raise ValueError("INVALID_EDGE_SOURCE_MAPPING_OR_PROVENANCE")
        if signature in observed_edges:
            raise ValueError("DUPLICATE_EDGE_ASSERTION")
        observed_edges.add(signature)
    if observed_edges != expected_edges:
        raise ValueError("MISSING_FROZEN_SOURCE_EDGE")
    return {
        "PASS": True,
        "dangling_edges": 0,
        "semantic_edges_without_provenance": 0,
        "invalid_provenance": 0,
        "span_source_mismatch": 0,
        "node_count_by_type": dict(sorted(Counter(n.type for n in nodes).items())),
        "edge_count_by_type": dict(sorted(Counter(e.type for e in edges).items())),
        "node_count": len(nodes),
        "edge_count": len(edges),
    }
