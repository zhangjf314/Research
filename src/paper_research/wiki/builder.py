"""Qualified graph in; deterministic projections out. Never opens Phase2B sources."""

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from paper_research.evidence_graph.schema import (
    Edge,
    GraphManifest,
    Node,
    canonical,
    digest,
    edge_id,
)

from .schema import PAGE_TYPES, VERSION, Footer, Page, filename, page_id, uri


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def title(node):
    key = {"Paper": "title", "Question": "question", "Requirement": "description", "Claim": "text"}
    return node.properties[key[node.type]]


@dataclass
class GraphSnapshot:
    nodes: dict[str, Node]
    edges: dict[str, Edge]
    graph_digest: str
    source_hashes: dict[str, str]
    synthetic: bool = False

    @classmethod
    def from_phase3a(cls, directory: Path):
        decision = json.loads((directory / "decision.json").read_text(encoding="utf-8"))
        if decision["phase_status"] != "PHASE3A_EVIDENCE_GRAPH_FOUNDATION_QUALIFIED":
            raise ValueError("GRAPH_NOT_QUALIFIED")
        freeze = json.loads((directory / "completion_freeze.json").read_text(encoding="utf-8"))
        permitted = [
            "graph/nodes.jsonl",
            "graph/edges.jsonl",
            "graph/manifest.json",
            "identity/source_hashes.json",
            "identity/corpus.json",
            "identity/benchmark.json",
            "decision.json",
        ]
        for rel in permitted:
            if sha(directory / rel) != freeze["artifact_sha256_excludes_self"][rel]:
                raise ValueError("PHASE3A_FROZEN_GRAPH_TAMPERED")
        m = GraphManifest.model_validate_json((directory / "graph/manifest.json").read_bytes())
        nb, eb = (
            (directory / "graph/nodes.jsonl").read_bytes(),
            (directory / "graph/edges.jsonl").read_bytes(),
        )
        scientific = m.model_dump()
        scientific.pop("scientific_digest")
        if (
            hashlib.sha256(nb).hexdigest() != m.nodes_sha256
            or hashlib.sha256(eb).hexdigest() != m.edges_sha256
            or hashlib.sha256(nb + eb).hexdigest() != m.graph_digest
            or digest(scientific) != m.scientific_digest
        ):
            raise ValueError("GRAPH_MANIFEST_MISMATCH")
        nodes = [Node.model_validate_json(line) for line in nb.splitlines()]
        edges = [Edge.model_validate_json(line) for line in eb.splitlines()]
        if len(nodes) != m.node_count or len(edges) != m.edge_count:
            raise ValueError("GRAPH_COUNT_MISMATCH")
        if len({n.id for n in nodes}) != len(nodes) or len({e.id for e in edges}) != len(edges):
            raise ValueError("DUPLICATE_GRAPH_ID")
        result = cls(
            {n.id: n for n in nodes}, {e.id: e for e in edges}, m.graph_digest, m.source_hashes
        )
        result.validate()
        return result

    def validate(self):
        allowed = {
            "CONTAINS": {("Paper", "Chunk"), ("Chunk", "EvidenceSpan")},
            "SOURCED_FROM": {("EvidenceSpan", "Paper")},
            "GROUNDED_BY": {("Requirement", "EvidenceSpan"), ("Claim", "EvidenceSpan")},
            "REQUIRES": {("Question", "Requirement")},
        }
        for n in self.nodes.values():
            Node.model_validate(n.model_dump())
            if n.type == "EvidenceSpan":
                p = n.properties
                chunks = [
                    c
                    for c in self.nodes.values()
                    if c.type == "Chunk" and c.properties["chunk_id"] == p["chunk_id"]
                ]
                if (
                    len(chunks) != 1
                    or p["text"] != chunks[0].properties["chunk_text"][p["start"] : p["end"]]
                ):
                    raise ValueError("SPAN_SOURCE_MISMATCH")
        for e in self.edges.values():
            Edge.model_validate(e.model_dump())
            if e.id != edge_id(e) or e.source not in self.nodes or e.target not in self.nodes:
                raise ValueError("INVALID_GRAPH_EDGE")
            if (self.nodes[e.source].type, self.nodes[e.target].type) not in allowed[e.type]:
                raise ValueError("INVALID_GRAPH_EDGE_TYPES")
            if (
                not self.synthetic
                and e.provenance.source_sha256 != self.source_hashes[e.provenance.source_artifact]
            ):
                raise ValueError("INVALID_SOURCE_HASH")


def descriptors(g: GraphSnapshot):
    """Topology/dependency pass; does not render pages or assemble page content."""
    g.validate()
    out, incoming = defaultdict(list), defaultdict(list)
    for e in g.edges.values():
        out[e.source].append(e)
        incoming[e.target].append(e)
    requirements, claims, questions = {}, {}, {}
    for n in g.nodes.values():
        if n.type == "Requirement":
            requirements[(n.properties["sample_id"], n.properties["requirement_id"])] = n.id
        if n.type == "Claim":
            claims.setdefault(n.properties["sample_id"], set()).add(n.id)
        if n.type == "Question":
            questions[n.properties["sample_id"]] = n.id
    grounded = {
        n.id: {e.target for e in out[n.id] if e.type == "GROUNDED_BY"}
        for n in g.nodes.values()
        if n.type in ("Claim", "Requirement")
    }
    span_papers, span_chunks, span_path_edges = {}, {}, {}
    for n in g.nodes.values():
        if n.type != "EvidenceSpan":
            continue
        paper = [e for e in out[n.id] if e.type == "SOURCED_FROM"]
        chunk = [e for e in incoming[n.id] if e.type == "CONTAINS"]
        if len(paper) != 1 or len(chunk) != 1:
            raise ValueError("INCOMPLETE_SPAN_PROVENANCE")
        root = [
            e
            for e in incoming[chunk[0].source]
            if e.type == "CONTAINS" and e.source == paper[0].target
        ]
        if len(root) != 1:
            raise ValueError("INCOMPLETE_CHUNK_PROVENANCE")
        span_papers[n.id], span_chunks[n.id] = paper[0].target, chunk[0].source
        span_path_edges[n.id] = {paper[0].id, chunk[0].id, root[0].id}
    desc = {}
    for n in sorted(g.nodes.values(), key=lambda n: n.id):
        if n.type not in PAGE_TYPES:
            continue
        links, subjects, spans = {}, set(), set()
        if n.type == "Paper":
            spans = {s for s, paper in span_papers.items() if paper == n.id}
            subjects = {s for s, targets in grounded.items() if targets & spans}
            for kind in ("Claim", "Requirement"):
                links[kind.lower() + "s"] = {s for s in subjects if g.nodes[s].type == kind}
            links["questions"] = {questions[g.nodes[s].properties["sample_id"]] for s in subjects}
        elif n.type == "Question":
            reqs = {e.target for e in out[n.id] if e.type == "REQUIRES"}
            cls = claims.get(n.properties["sample_id"], set())
            links = {"requirements": reqs, "claims": cls}
            subjects = reqs | cls
        else:
            task = n.properties["sample_id"]
            links["questions"] = {questions[task]}
            subjects = {n.id}
            if n.type == "Claim":
                links["requirements"] = {requirements[(task, n.properties["requirement_id"])]}
                links["shared_evidence_with"] = {
                    s
                    for s, targets in grounded.items()
                    if s != n.id and g.nodes[s].type == "Claim" and targets & grounded[n.id]
                }
            else:
                links["claims"] = {
                    s
                    for s in claims.get(task, set())
                    if g.nodes[s].properties["requirement_id"] == n.properties["requirement_id"]
                }
        if n.type != "Paper":
            spans = set().union(*(grounded[s] for s in subjects)) if subjects else set()
            links["papers"] = {span_papers[s] for s in spans}
        links = {
            k: sorted(page_id(g.nodes[s].type, s) for s in v) for k, v in sorted(links.items())
        }
        path_edges, paths = set(), []
        for s in sorted(spans):
            path_edges |= span_path_edges[s]
            matching = [e for e in incoming[s] if e.type == "GROUNDED_BY" and e.source in subjects]
            path_edges |= {e.id for e in matching}
            if matching:
                paths += [[e.source, s, span_chunks[s], span_papers[s]] for e in matching]
            else:
                paths.append([s, span_chunks[s], span_papers[s]])
        for related in set().union(*(set(v) for v in links.values())) if links else set():
            rid = related.split(":", 2)[2]
            if g.nodes[rid].type == "Requirement":
                path_edges |= {e.id for e in incoming[rid] if e.type == "REQUIRES"}
            if g.nodes[rid].type == "Claim" and "shared_evidence_with" in links:
                path_edges |= {
                    e.id for e in out[rid] if e.type == "GROUNDED_BY" and e.target in spans
                }
        if n.type == "Requirement":
            path_edges |= {e.id for e in incoming[n.id] if e.type == "REQUIRES"}
        desc[page_id(n.type, n.id)] = {
            "source": n.id,
            "links": links,
            "spans": sorted(spans),
            "edges": sorted(path_edges),
            "paths": sorted(set(tuple(p) for p in paths)),
            "backlinks": [],
        }
    for pid, record in desc.items():
        for target in (
            set().union(*(set(v) for v in record["links"].values())) if record["links"] else set()
        ):
            if target not in desc:
                raise ValueError("DANGLING_WIKI_LINK")
            desc[target]["backlinks"].append(pid)
    for _pid, record in desc.items():
        record["backlinks"] = sorted(set(record["backlinks"]))
        references = (
            set(record["backlinks"]) | set().union(*(set(v) for v in record["links"].values()))
            if record["links"]
            else set(record["backlinks"])
        )
        node_ids = {record["source"], *(p.split(":", 2)[2] for p in references)}
        for s in record["spans"]:
            node_ids.update((s, span_chunks[s], span_papers[s]))
        for e in record["edges"]:
            node_ids.update((g.edges[e].source, g.edges[e].target))
        record["nodes"] = sorted(node_ids)
        record["dependency_digest"] = digest(
            {
                "descriptor": record,
                "nodes": {i: g.nodes[i].model_dump() for i in sorted(node_ids)},
                "edges": {i: g.edges[i].model_dump() for i in record["edges"]},
            }
        )
    return desc


def _project_page(g: GraphSnapshot, pid: str, record: dict, anchor: str | None = None) -> Page:
    node = g.nodes[record["source"]]
    refs = set(record["backlinks"])
    for values in record["links"].values():
        refs.update(values)
    linked = [
        {
            "page_id": r,
            "uri": uri(r),
            "source_node_id": r.split(":", 2)[2],
            "title": title(g.nodes[r.split(":", 2)[2]]),
            "source_record": g.nodes[r.split(":", 2)[2]].properties,
        }
        for r in sorted(refs)
    ]
    evidence = []
    for sid in record["spans"]:
        span = g.nodes[sid]
        chain = next(p for p in record["paths"] if sid in p)
        cid, paper = chain[-2:]
        evidence.append(
            {
                "span_id": sid,
                **span.properties,
                "graph_chunk_node_id": cid,
                "graph_paper_node_id": paper,
                "paper_metadata": g.nodes[paper].properties,
                "source_artifact": span.provenance.source_artifact,
                "provenance": span.provenance.model_dump(),
            }
        )
    artifact_hashes = {}
    for n in record["nodes"]:
        prov = g.nodes[n].provenance
        artifact_hashes[prov.source_artifact] = prov.source_sha256
    for e in record["edges"]:
        prov = g.edges[e].provenance
        artifact_hashes[prov.source_artifact] = prov.source_sha256
    return Page(
        id=pid,
        type=PAGE_TYPES[node.type],
        source_node_id=node.id,
        title=title(node),
        source_record=node.properties,
        links=record["links"],
        backlinks=record["backlinks"],
        linked_records=linked,
        evidence=evidence,
        provenance_paths=[list(p) for p in record["paths"]],
        evidence_status="FROZEN_MODEL_GOLD_NOT_HUMAN_TRUTH",
        provenance=Footer(
            graph_digest=anchor or g.graph_digest,
            source_node_id=node.id,
            source_edge_ids=record["edges"],
            source_artifact_hashes=artifact_hashes,
            wiki_builder_version=VERSION,
            dependency_digest=record["dependency_digest"],
        ),
    )


def make_page(g: GraphSnapshot, pid: str, record: dict, anchor: str | None = None) -> Page:
    """Materialize a page. Invariant reconstruction is separate from incremental builds."""
    return _project_page(g, pid, record, anchor)


def build_pages(g: GraphSnapshot):
    desc = descriptors(g)
    return {pid: make_page(g, pid, d) for pid, d in desc.items()}, desc


def dependency_index(desc):
    nodes, edges = defaultdict(list), defaultdict(list)
    for pid, d in sorted(desc.items()):
        for n in d["nodes"]:
            nodes[n].append(pid)
        for e in d["edges"]:
            edges[e].append(pid)
    return {
        "nodes": dict(sorted(nodes.items())),
        "edges": dict(sorted(edges.items())),
        "pages": {
            p: {
                "nodes": d["nodes"],
                "edges": d["edges"],
                "dependency_digest": d["dependency_digest"],
            }
            for p, d in sorted(desc.items())
        },
    }


def validate_pages(g, pages, desc=None, lineage=None, validated_reuse=None):
    desc = descriptors(g) if desc is None else desc
    if set(pages) != set(desc):
        raise ValueError("MISSING_OR_DUPLICATE_PAGE")
    for pid, page in pages.items():
        Page.model_validate(page.model_dump())
        anchor = page.provenance.graph_digest
        if lineage is None:
            if anchor != g.graph_digest:
                raise ValueError("INVALID_GRAPH_PIN")
        elif lineage.get(anchor, {}).get(pid) != desc[pid]["dependency_digest"]:
            raise ValueError("INVALID_PIN_COMPATIBILITY_PROOF")
        if validated_reuse is not None and pid in validated_reuse:
            if canonical(page.model_dump()) != canonical(validated_reuse[pid].model_dump()):
                raise ValueError("UNAFFECTED_PAGE_CHANGED")
        elif canonical(page.model_dump()) != canonical(
            _project_page(g, pid, desc[pid], anchor).model_dump()
        ):
            raise ValueError("PAGE_GRAPH_ROUNDTRIP_MISMATCH")
        if page.id != pid or page.source_node_id not in g.nodes:
            raise ValueError("FORGED_SOURCE_NODE")
        if page.type != "PaperPage" and not page.backlinks:
            raise ValueError("ORPHAN_PAGE")
    return {
        "PASS": True,
        "broken_internal_links": 0,
        "orphan_pages": 0,
        "invalid_provenance": 0,
        "roundtrip_pages_exact": len(pages),
    }


def write_wiki(directory, g, pages, desc, lineage=None, previous=None, rebuilt=None):
    from .render import render_markdown, render_root

    lineage = lineage or {g.graph_digest: {pid: d["dependency_digest"] for pid, d in desc.items()}}
    reuse = {pid: page for pid, page in pages.items() if previous and pid not in rebuilt}
    validate_pages(g, pages, desc, lineage, reuse or None)
    if directory.exists():
        raise ValueError("IMMUTABLE_WIKI_OUTPUT_EXISTS")
    directory.mkdir(parents=True)
    for sub in ("pages/json", "pages/md"):
        (directory / sub).mkdir(parents=True)
    entries = []
    previous_entries = {}
    if previous:
        old_index = json.loads((previous / "index.json").read_text(encoding="utf-8"))
        old_manifest = json.loads((previous / "manifest.json").read_text(encoding="utf-8"))
        if (
            digest(old_index["entries"]) != old_index["index_digest"]
            or old_index["index_digest"] != old_manifest["index_digest"]
        ):
            raise ValueError("PREVIOUS_WIKI_INDEX_TAMPERED")
        previous_entries = {e["page_id"]: e for e in old_index["entries"]}
    for pid, page in sorted(pages.items()):
        name = filename(pid)
        json_path, md_path = "pages/json/" + name + ".json", "pages/md/" + name + ".md"
        if previous and pid not in rebuilt:
            # Unaffected pages are reused byte-for-byte; no renderer/builder invoked.
            jb = (previous / json_path).read_bytes()
            mb = (previous / md_path).read_bytes()
            old = previous_entries[pid]
            if (
                hashlib.sha256(jb).hexdigest() != old["digest"]
                or hashlib.sha256(mb).hexdigest() != old["markdown_digest"]
                or canonical(json.loads(jb)) != canonical(page.model_dump())
            ):
                raise ValueError("TAMPERED_UNAFFECTED_PAGE_REUSE")
        else:
            jb = canonical(page.model_dump()) + b"\n"
            mb = render_markdown(page).encode("utf-8")
        (directory / json_path).write_bytes(jb)
        (directory / md_path).write_bytes(mb)
        entries.append(
            {
                "page_id": pid,
                "page_type": page.type,
                "source_node_id": page.source_node_id,
                "title": page.title,
                "path": json_path,
                "markdown_path": md_path,
                "digest": hashlib.sha256(jb).hexdigest(),
                "markdown_digest": hashlib.sha256(mb).hexdigest(),
            }
        )
    index = {"version": VERSION, "entries": entries, "index_digest": digest(entries)}
    deps = dependency_index(desc)
    content = {
        "version": VERSION,
        "graph_digest": g.graph_digest,
        "page_count": len(pages),
        "index_digest": index["index_digest"],
        "dependency_index_digest": digest(deps),
        "lineage_digest": digest(lineage),
        "semantic_truth_qualification": "NOT_ASSESSED",
        "evidence_status": "FROZEN_MODEL_GOLD_NOT_HUMAN_TRUTH",
        "synthetic": g.synthetic,
    }
    manifest = {**content, "scientific_digest": digest(content)}
    for name, data in (
        ("index.json", index),
        ("manifest.json", manifest),
        ("dependency_index.json", deps),
        ("lineage.json", lineage),
    ):
        (directory / name).write_bytes(canonical(data) + b"\n")
    (directory / "README.md").write_text(render_root(entries), encoding="utf-8")
    return manifest
