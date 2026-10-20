"""Build exclusively from an explicit frozen corpus/benchmark allowlist."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .schema import Edge, GraphManifest, Node, Provenance, canonical, digest, edge_id, node_id

BENCHMARK = "PHASE2B_FROZEN_60_V1"
INPUTS = {
    "metadata": "artifacts/paperresearch-v2/phase1q-r2/corpus/metadata.json",
    "chunks": "artifacts/paperresearch-v2/phase1q-r2/corpus/chunks.jsonl",
    "questions": "artifacts/paperresearch-v2/phase2b/benchmark/questions.jsonl",
    "requirements": "artifacts/paperresearch-v2/phase2b/benchmark/gold_requirements.jsonl",
    "claims": "artifacts/paperresearch-v2/phase2b/benchmark/gold_claims.jsonl",
    "evidence": "artifacts/paperresearch-v2/phase2b/benchmark/gold_evidence.jsonl",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unique(rows, key):
    result = {}
    for row in rows:
        identity = row[key]
        if not isinstance(identity, str) or not identity or identity in result:
            raise ValueError("DUPLICATE_OR_INVALID_SOURCE_ID")
        result[identity] = row
    return result


@dataclass(frozen=True)
class SourceData:
    metadata: dict
    chunks: dict
    questions: dict
    requirements: dict
    claims: dict
    evidence: dict
    source_hashes: dict

    def provenance(self, key, record, *, semantic=False, chunk=None):
        fields = {}
        if chunk is not None:
            fields = {
                "paper_id": chunk["paper_id"],
                "chunk_id": chunk["chunk_id"],
                "span_id": span_id(chunk),
            }
        return Provenance(
            source_phase="Phase1Q-R2" if key in ("metadata", "chunks") else "Phase2B",
            source_artifact=INPUTS[key],
            source_sha256=self.source_hashes[INPUTS[key]],
            source_record_id=record,
            construction_method="frozen_gold_mapping" if semantic else "deterministic_identity",
            evidence_status="FROZEN_MODEL_GOLD_NOT_HUMAN_TRUTH"
            if semantic
            else "STRUCTURAL_SOURCE",
            **fields,
        )


def span_id(chunk):
    return node_id(
        "EvidenceSpan", chunk["paper_id"], chunk["chunk_id"], 0, len(chunk["chunk_text"])
    )


def load_inputs(root: Path, expected_hashes: dict[str, str]) -> SourceData:
    if set(expected_hashes) != set(INPUTS.values()):
        raise ValueError("INPUT_ALLOWLIST_MISMATCH")
    values = {}
    for key, relative in INPUTS.items():
        path = root / relative
        if sha(path) != expected_hashes[relative]:
            raise ValueError("FROZEN_INPUT_HASH_MISMATCH")
        text = path.read_text(encoding="utf-8")
        values[key] = (
            json.loads(text)
            if key == "metadata"
            else [json.loads(line) for line in text.splitlines() if line]
        )
    data = SourceData(
        metadata=values["metadata"],
        chunks=unique(values["chunks"], "chunk_id"),
        questions=unique(values["questions"], "sample_id"),
        requirements=unique(values["requirements"], "sample_id"),
        claims=unique(values["claims"], "sample_id"),
        evidence=unique(values["evidence"], "sample_id"),
        source_hashes=dict(expected_hashes),
    )
    validate_sources(data)
    return data


def validate_sources(d: SourceData):
    if any(set(table) != set(d.questions) for table in (d.requirements, d.claims, d.evidence)):
        raise ValueError("TASK_IDENTITY_MISMATCH")
    for chunk in d.chunks.values():
        if chunk["paper_id"] not in d.metadata:
            raise ValueError("MISSING_PAPER")
        if not isinstance(chunk["chunk_text"], str) or not chunk["chunk_text"]:
            raise ValueError("INVALID_CHUNK_TEXT")
    for task, question in d.questions.items():
        if any(p not in d.metadata for p in question["paper_scope"]):
            raise ValueError("UNKNOWN_PAPER_SCOPE")
        evidence = unique(d.evidence[task]["evidence"], "evidence_id")
        unique(d.evidence[task]["evidence"], "source_id")
        for identity, e in evidence.items():
            if identity != e["chunk_id"] or e["chunk_id"] not in d.chunks:
                raise ValueError("BAD_CHUNK_ID")
            c = d.chunks[e["chunk_id"]]
            if e["paper_id"] != c["paper_id"] or e["text"] != c["chunk_text"]:
                raise ValueError("SOURCE_TEXT_OR_PAPER_MISMATCH")
            if (e["page"], e["page_end"], e["block_ids"]) != (
                c["page_start"],
                c["page_end"],
                c["block_ids"],
            ):
                raise ValueError("PAGE_BLOCK_PROVENANCE_MISMATCH")
        requirements = unique(d.requirements[task]["requirements"], "requirement_id")
        claims = unique(d.claims[task]["claims"], "claim_id")
        for row in [*requirements.values(), *claims.values()]:
            if any(e not in evidence for e in row["evidence_ids"]):
                raise ValueError("UNKNOWN_EVIDENCE_MAPPING")
        if any(c["requirement_id"] not in requirements for c in claims.values()):
            raise ValueError("UNKNOWN_REQUIREMENT")


def build_graph(d: SourceData) -> tuple[list[Node], list[Edge]]:
    validate_sources(d)
    nodes, edges = {}, {}

    def add_node(kind, identity, props, provenance):
        node = Node(id=identity, type=kind, properties=props, provenance=provenance)
        if identity in nodes:
            raise ValueError("DUPLICATE_NODE")
        nodes[identity] = node

    def add_edge(kind, source, target, provenance):
        fields = {
            "type": kind,
            "kind": "evidence-semantic" if kind == "GROUNDED_BY" else "structural",
            "source": source,
            "target": target,
            "properties": {},
            "provenance": provenance.model_dump(),
        }
        edge = Edge(id=edge_id(fields), **fields)
        if edge.id in edges:
            raise ValueError("DUPLICATE_EDGE")
        edges[edge.id] = edge

    for p, metadata in sorted(d.metadata.items()):
        add_node(
            "Paper", node_id("Paper", p), {"paper_id": p, **metadata}, d.provenance("metadata", p)
        )
    for c in sorted(d.chunks.values(), key=lambda c: c["chunk_id"]):
        identity = node_id("Chunk", c["paper_id"], c["chunk_id"])
        provenance = d.provenance("chunks", c["chunk_id"], chunk=c)
        add_node("Chunk", identity, c, provenance)
        add_edge("CONTAINS", node_id("Paper", c["paper_id"]), identity, provenance)

    # One reusable span per unique frozen evidence chunk, independent of task traversal order.
    span_origins = {}
    for task, row in sorted(d.evidence.items()):
        for e in sorted(row["evidence"], key=lambda e: e["evidence_id"]):
            c = d.chunks[e["chunk_id"]]
            span_origins.setdefault(span_id(c), (task, e["evidence_id"], c))
    for identity, (task, evidence_id, c) in sorted(span_origins.items()):
        provenance = d.provenance("evidence", task + "/" + evidence_id, chunk=c)
        add_node(
            "EvidenceSpan",
            identity,
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
            provenance,
        )
        add_edge("CONTAINS", node_id("Chunk", c["paper_id"], c["chunk_id"]), identity, provenance)
        add_edge("SOURCED_FROM", identity, node_id("Paper", c["paper_id"]), provenance)

    for task, question in sorted(d.questions.items()):
        qid = node_id("Question", BENCHMARK, task)
        add_node("Question", qid, question, d.provenance("questions", task))
        for kind, key, group, idkey in (
            ("Requirement", "requirements", "requirements", "requirement_id"),
            ("Claim", "claims", "claims", "claim_id"),
        ):
            for row in sorted(getattr(d, key)[task][group], key=lambda r: r[idkey]):
                identity = node_id(kind, BENCHMARK, task, row[idkey])
                provenance = d.provenance(key, task + "/" + row[idkey])
                add_node(kind, identity, {"sample_id": task, **row}, provenance)
                if kind == "Requirement":
                    add_edge("REQUIRES", qid, identity, provenance)
                for evidence_id in sorted(set(row["evidence_ids"])):
                    c = d.chunks[evidence_id]
                    add_edge(
                        "GROUNDED_BY",
                        identity,
                        span_id(c),
                        d.provenance(key, task + "/" + row[idkey], semantic=True, chunk=c),
                    )
    result = ([nodes[k] for k in sorted(nodes)], [edges[k] for k in sorted(edges)])
    from .validate import validate_graph

    validate_graph(*result, d)
    return result


def graph_bytes(records):
    return b"".join(canonical(record.model_dump()) + b"\n" for record in records)


def write_graph(directory: Path, nodes: list[Node], edges: list[Edge], sources: SourceData):
    from .validate import validate_graph

    validate_graph(nodes, edges, sources)
    if directory.exists():
        raise ValueError("IMMUTABLE_BUILD_OUTPUT_EXISTS")
    nb, eb = graph_bytes(nodes), graph_bytes(edges)
    content = {
        "schema_version": "EVIDENCE_GRAPH_V1",
        "benchmark": BENCHMARK,
        "source_hashes": sources.source_hashes,
        "span_policy": "FULL_CANONICAL_CHUNK_HALF_OPEN_UNICODE_CODEPOINTS",
        "node_count": len(nodes),
        "edge_count": len(edges),
        "nodes_sha256": hashlib.sha256(nb).hexdigest(),
        "edges_sha256": hashlib.sha256(eb).hexdigest(),
        "graph_digest": hashlib.sha256(nb + eb).hexdigest(),
        "verifier_predictions_as_truth": False,
    }
    manifest = GraphManifest(**content, scientific_digest=digest(content))
    directory.mkdir(parents=True)
    (directory / "nodes.jsonl").write_bytes(nb)
    (directory / "edges.jsonl").write_bytes(eb)
    (directory / "manifest.json").write_bytes(canonical(manifest.model_dump()) + b"\n")
    return manifest
