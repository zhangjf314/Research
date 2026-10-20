"""Indexed deterministic lookups and bounded neighborhoods; no semantic inference."""

import json
from collections import defaultdict
from pathlib import Path

from .builder import SourceData, sha
from .schema import Edge, GraphManifest, Node, digest
from .validate import validate_graph


def result(nodes=(), edges=(), paths=()):
    return {
        "node_ids": sorted(set(nodes)),
        "edge_ids": sorted(set(edges)),
        "paths": sorted({tuple(p) for p in paths}),
    }


class GraphQuery:
    def __init__(self, nodes: list[Node], edges: list[Edge], sources: SourceData):
        validate_graph(nodes, edges, sources)
        self.nodes = {n.id: Node.model_validate(n.model_dump()) for n in nodes}
        self.edges = {e.id: Edge.model_validate(e.model_dump()) for e in edges}
        self.outgoing, self.incoming = defaultdict(list), defaultdict(list)
        for edge in sorted(self.edges.values(), key=lambda e: e.id):
            self.outgoing[edge.source].append(edge)
            self.incoming[edge.target].append(edge)

    @classmethod
    def from_artifact(cls, directory: Path, sources: SourceData):
        manifest = GraphManifest.model_validate_json((directory / "manifest.json").read_bytes())
        if (
            manifest.source_hashes != sources.source_hashes
            or sha(directory / "nodes.jsonl") != manifest.nodes_sha256
            or sha(directory / "edges.jsonl") != manifest.edges_sha256
        ):
            raise ValueError("GRAPH_ARTIFACT_HASH_MISMATCH")
        import hashlib

        if (
            hashlib.sha256(
                (directory / "nodes.jsonl").read_bytes() + (directory / "edges.jsonl").read_bytes()
            ).hexdigest()
            != manifest.graph_digest
        ):
            raise ValueError("GRAPH_DIGEST_MISMATCH")
        scientific = manifest.model_dump()
        scientific.pop("scientific_digest")
        if digest(scientific) != manifest.scientific_digest:
            raise ValueError("SCIENTIFIC_MANIFEST_MISMATCH")
        nodes = [
            Node.model_validate(json.loads(line))
            for line in (directory / "nodes.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        edges = [
            Edge.model_validate(json.loads(line))
            for line in (directory / "edges.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        if len(nodes) != manifest.node_count or len(edges) != manifest.edge_count:
            raise ValueError("MANIFEST_COUNT_MISMATCH")
        return cls(nodes, edges, sources)

    def require(self, identity, kind=None):
        if identity not in self.nodes:
            raise KeyError("UNKNOWN_GRAPH_NODE")
        if kind and self.nodes[identity].type != kind:
            raise ValueError("QUERY_NODE_TYPE_MISMATCH")
        return self.nodes[identity]

    def evidence_for_requirement(self, identity):
        self.require(identity, "Requirement")
        return self.evidence_for_claim_or_requirement(identity)

    def evidence_for_claim_or_requirement(self, identity):
        n = self.require(identity)
        if n.type not in ("Claim", "Requirement"):
            raise ValueError("QUERY_NODE_TYPE_MISMATCH")
        edges = [e for e in self.outgoing[identity] if e.type == "GROUNDED_BY"]
        return result(
            (e.target for e in edges), (e.id for e in edges), ([identity, e.target] for e in edges)
        )

    def _using_paper(self, paper_id, kind):
        self.require(paper_id, "Paper")
        spans = {e.source for e in self.incoming[paper_id] if e.type == "SOURCED_FROM"}
        ids = {
            e.source
            for span in spans
            for e in self.incoming[span]
            if e.type == "GROUNDED_BY" and self.nodes[e.source].type == kind
        }
        traces = [self.trace_provenance(identity) for identity in sorted(ids)]
        # Only include paths through the requested paper, not other source papers.
        paths = [p for t in traces for p in t["paths"] if p[-1] == paper_id]
        path_nodes = {n for p in paths for n in p}
        edge_ids = {
            e.id
            for e in self.edges.values()
            if e.source in path_nodes
            and e.target in path_nodes
            and any(e.source in p and e.target in p for p in paths)
        }
        return result(ids, edge_ids, paths)

    def claims_grounded_by_paper(self, identity):
        return self._using_paper(identity, "Claim")

    def requirements_using_paper(self, identity):
        return self._using_paper(identity, "Requirement")

    def trace_provenance(self, identity):
        node = self.require(identity)
        if node.type == "Paper":
            return result([identity], paths=[[identity]])
        if node.type == "Question":
            targets = [e for e in self.outgoing[identity] if e.type == "REQUIRES"]
        elif node.type in ("Claim", "Requirement"):
            targets = [e for e in self.outgoing[identity] if e.type == "GROUNDED_BY"]
        else:
            targets = [e for e in self.incoming[identity] if e.type == "CONTAINS"]
        all_nodes, all_edges, paths = {identity}, set(), []
        for edge in targets:
            next_id = (
                edge.target if node.type in ("Question", "Claim", "Requirement") else edge.source
            )
            child = self.trace_provenance(next_id)
            all_nodes.update(child["node_ids"])
            all_edges.update(child["edge_ids"])
            all_edges.add(edge.id)
            paths += [[identity, *path] for path in child["paths"]]
        if node.type == "EvidenceSpan":
            all_edges.update(e.id for e in self.outgoing[identity] if e.type == "SOURCED_FROM")
        return result(all_nodes, all_edges, paths)

    def multi_source_requirements(self):
        selected, edges, paths = [], [], []
        for node in self.nodes.values():
            if node.type != "Requirement":
                continue
            evidence = self.evidence_for_requirement(node.id)
            papers = {self.nodes[s].properties["paper_id"] for s in evidence["node_ids"]}
            if len(papers) > 1:
                selected.append(node.id)
                edges += evidence["edge_ids"]
                paths += evidence["paths"]
        return result(selected, edges, paths)

    def neighbors(self, identity, depth=1):
        self.require(identity)
        if type(depth) is not int or not 0 <= depth <= 2:
            raise ValueError("NEIGHBOR_DEPTH_MUST_BE_0_TO_2")
        visited, found_edges, frontier = {identity}, set(), {identity}
        for _ in range(depth):
            upcoming = set()
            for n in frontier:
                for e in [*self.outgoing[n], *self.incoming[n]]:
                    found_edges.add(e.id)
                    upcoming.update((e.source, e.target))
            frontier = upcoming - visited
            visited.update(upcoming)
        return result(visited - {identity}, found_edges)
