"""Independent mandatory synthesis projection; the generic Wiki API is unchanged."""

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from paper_research.evidence_graph.schema import Node, canonical, digest, edge_id
from paper_research.wiki_synthesis import Strict
from paper_research.wiki_synthesis_r1 import R1Synthesis, validate_r1

from .builder import GraphSnapshot, sha
from .query import WikiQuery
from .schema import page_id

VERSION = "SYNTHESIS_BUNDLE_V1"
MAX_CHARS = 24000
FIELDS = ("question", "source_units", "evidence", "source_papers", "provenance")


class QuestionCore(Strict):
    question_id: str
    question_text: str
    stratum: str
    question_page_id: str


class UnitRecord(Strict):
    """Semantic lineage anchor, not proof, a truth label or a support judgment."""

    type: Literal["CLAIM", "REQUIREMENT"]
    graph_node_id: str
    wiki_page_id: str
    text: str
    direct_grounded_evidence_span_ids: list[str] = Field(min_length=1)


class EvidenceRecord(Strict):
    span_id: str
    text: str
    chunk_id: str
    paper_id: str
    start: int
    end: int
    page_start: int | None
    page_end: int | None
    block_ids: list[str]


class PaperRecord(Strict):
    paper_id: str
    title: str
    arxiv_id: str | None = None


class BundleProvenance(Strict):
    projection_version: Literal["SYNTHESIS_BUNDLE_V1"]
    bundle_id: str
    graph_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    wiki_manifest_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_edge_ids: list[str]
    # span ID -> [Graph Chunk node ID, Graph Paper node ID]. All edges are pinned above.
    span_paths: dict[str, list[str]]


class SynthesisBundle(Strict):
    question: QuestionCore
    source_units: list[UnitRecord] = Field(min_length=1)
    evidence: list[EvidenceRecord] = Field(min_length=1)
    source_papers: list[PaperRecord] = Field(min_length=1)
    provenance: BundleProvenance

    @model_validator(mode="after")
    def unique_sorted_records(self):
        for values in [
            [u.graph_node_id for u in self.source_units],
            [e.span_id for e in self.evidence],
            [p.paper_id for p in self.source_papers],
            self.provenance.source_edge_ids,
        ]:
            if values != sorted(set(values)):
                raise ValueError("DUPLICATE_OR_UNSORTED_PROJECTION_ID")
        for unit in self.source_units:
            if unit.direct_grounded_evidence_span_ids != sorted(
                set(unit.direct_grounded_evidence_span_ids)
            ):
                raise ValueError("DUPLICATE_OR_UNSORTED_GROUNDING_ID")
        return self


def bundle_bytes(bundle):
    """Fixed top-level field order; nested keys use the frozen canonical JSON encoder."""
    model = (
        bundle if isinstance(bundle, SynthesisBundle) else SynthesisBundle.model_validate(bundle)
    )
    value = model.model_dump()
    return b"{" + b",".join(canonical(k) + b":" + canonical(value[k]) for k in FIELDS) + b"}"


def bundle_sha256(bundle):
    return hashlib.sha256(bundle_bytes(bundle)).hexdigest()


class OverBudget(ValueError):
    """The complete mandatory object is retained for diagnostics, never truncated."""

    def __init__(self, bundle):
        self.bundle = bundle
        self.serialized_chars = len(bundle_bytes(bundle).decode("utf-8"))
        super().__init__(f"MANDATORY_PROJECTION_OVER_BUDGET:{self.serialized_chars}>{MAX_CHARS}")


class SynthesisBundleBuilder:
    def __init__(self, graph, wiki, wiki_manifest_digest):
        if wiki.graph.graph_digest != graph.graph_digest:
            raise ValueError("GRAPH_WIKI_IDENTITY_MISMATCH")
        self.graph, self.wiki, self.wiki_manifest_digest = graph, wiki, wiki_manifest_digest

    @classmethod
    def from_artifacts(cls, phase3a: Path, phase3b: Path):
        # Reads qualified Graph/Wiki artifacts only; never opens their Phase2B sources.
        graph = GraphSnapshot.from_phase3a(phase3a)
        freeze = json.loads((phase3b / "completion_freeze.json").read_text("utf-8"))
        for rel in [
            "decision.json",
            "wiki/manifest.json",
            "wiki/index.json",
            "wiki/lineage.json",
            "wiki/dependency_index.json",
        ]:
            if sha(phase3b / rel) != freeze["artifact_sha256_excludes_self"][rel]:
                raise ValueError("PHASE3B_FROZEN_IDENTITY_MISMATCH")
        decision = json.loads((phase3b / "decision.json").read_text("utf-8"))
        if decision["phase_status"] != "PHASE3B_AGENT_NATIVE_WIKI_FOUNDATION_QUALIFIED":
            raise ValueError("WIKI_NOT_QUALIFIED")
        wiki = WikiQuery.from_artifact(phase3b / "wiki", graph)
        manifest = json.loads((phase3b / "wiki/manifest.json").read_text("utf-8"))
        return cls(graph, wiki, manifest["scientific_digest"])

    def node(self, identity, kind):
        if identity not in self.graph.nodes or self.graph.nodes[identity].type != kind:
            raise ValueError("UNKNOWN_OR_WRONG_GRAPH_NODE:" + kind)
        node = self.graph.nodes[identity]
        Node.model_validate(node.model_dump())
        if node.id != identity or not identity.startswith(kind.lower() + ":"):
            raise ValueError("GRAPH_NODE_IDENTITY_MISMATCH")
        if (
            not self.graph.synthetic
            and self.graph.source_hashes.get(node.provenance.source_artifact)
            != node.provenance.source_sha256
        ):
            raise ValueError("NODE_SOURCE_PROVENANCE_MISMATCH")
        return node

    def page(self, identity, kind, graph_identity):
        if identity not in self.wiki.pages:
            raise ValueError("UNKNOWN_WIKI_PAGE:" + kind)
        page = self.wiki.pages[identity]
        node = self.node(graph_identity, kind)
        if (page.id, page.type, page.source_node_id) != (
            page_id(kind, graph_identity),
            kind + "Page",
            graph_identity,
        ):
            raise ValueError("PAGE_NODE_IDENTITY_MISMATCH")
        if (
            canonical(page.source_record) != canonical(node.properties)
            or page.provenance.graph_digest != self.graph.graph_digest
        ):
            raise ValueError("PAGE_SOURCE_RECORD_OR_PROVENANCE_MISMATCH")
        return page

    def edges(self, *, source=None, target=None, kind=None):
        found = []
        for edge in self.graph.edges.values():
            if (
                (source is None or edge.source == source)
                and (target is None or edge.target == target)
                and (kind is None or edge.type == kind)
            ):
                if edge.id != edge_id(edge):
                    raise ValueError("GRAPH_EDGE_IDENTITY_MISMATCH")
                if (
                    not self.graph.synthetic
                    and self.graph.source_hashes.get(edge.provenance.source_artifact)
                    != edge.provenance.source_sha256
                ):
                    raise ValueError("EDGE_SOURCE_PROVENANCE_MISMATCH")
                found.append(edge)
        return sorted(found, key=lambda e: e.id)

    def direct_nodes(self, question_page_id):
        if question_page_id not in self.wiki.pages:
            raise ValueError("UNKNOWN_QUESTION_PAGE")
        supplied = self.wiki.pages[question_page_id]
        page = self.page(question_page_id, "Question", supplied.source_node_id)
        sample = page.source_record["sample_id"]
        claims = sorted(
            n.id
            for n in self.graph.nodes.values()
            if n.type == "Claim" and n.properties["sample_id"] == sample
        )
        requires = self.edges(source=page.source_node_id, kind="REQUIRES")
        requirements = sorted(e.target for e in requires)
        expected_requirements = sorted(
            n.id
            for n in self.graph.nodes.values()
            if n.type == "Requirement" and n.properties["sample_id"] == sample
        )
        if requirements != expected_requirements:
            raise ValueError("QUESTION_REQUIREMENT_GRAPH_BINDING_MISMATCH")
        for kind, identities in [("Claim", claims), ("Requirement", requirements)]:
            links = page.links.get(kind.lower() + "s", [])
            if len(set(links)) != len(links):
                raise ValueError("DUPLICATE_DIRECT_SOURCE_UNIT")
            resolved = []
            for pid in links:
                if pid not in self.wiki.pages:
                    raise ValueError("UNKNOWN_DIRECT_PAGE")
                direct = self.wiki.pages[pid]
                self.page(pid, kind, direct.source_node_id)
                resolved.append(direct.source_node_id)
            if sorted(resolved) != identities:
                raise ValueError("DIRECT_GRAPH_WIKI_UNIT_SET_MISMATCH")
        return page, claims, requirements, requires

    def _mandatory(self, question_page_id):
        page, claims, requirements, requires = self.direct_nodes(question_page_id)
        selected, kind = (claims, "Claim") if claims else (requirements, "Requirement")
        if not selected:
            raise ValueError("NO_DIRECT_SEMANTIC_SOURCE_UNIT")
        units, evidence, papers, paths = [], {}, {}, {}
        used_edges = {e.id for e in requires}
        for identity in selected:
            node = self.node(identity, kind)
            unit_page = self.page(page_id(kind, identity), kind, identity)
            grounded = self.edges(source=identity, kind="GROUNDED_BY")
            spans = sorted({e.target for e in grounded})
            if not spans:
                raise ValueError("MISSING_DIRECT_GROUNDED_BY")
            if {e["span_id"] for e in unit_page.evidence} != set(spans):
                raise ValueError("WIKI_GRAPH_GROUNDING_SET_MISMATCH")
            used_edges.update(e.id for e in grounded)
            units.append(
                UnitRecord(
                    type=kind.upper(),
                    graph_node_id=identity,
                    wiki_page_id=unit_page.id,
                    text=node.properties["text" if kind == "Claim" else "description"],
                    direct_grounded_evidence_span_ids=spans,
                )
            )
            for sid in spans:
                span = self.node(sid, "EvidenceSpan")
                to_paper = self.edges(source=sid, kind="SOURCED_FROM")
                from_chunk = self.edges(target=sid, kind="CONTAINS")
                if len(to_paper) != 1 or len(from_chunk) != 1:
                    raise ValueError("MISSING_OR_AMBIGUOUS_SPAN_PROVENANCE")
                paper = self.node(to_paper[0].target, "Paper")
                chunk = self.node(from_chunk[0].source, "Chunk")
                parent = self.edges(source=paper.id, target=chunk.id, kind="CONTAINS")
                if len(parent) != 1:
                    raise ValueError("MISSING_CHUNK_PAPER_PROVENANCE")
                p, c, s = paper.properties, chunk.properties, span.properties
                if (
                    s["paper_id"] != p["paper_id"]
                    or c["paper_id"] != p["paper_id"]
                    or s["chunk_id"] != c["chunk_id"]
                ):
                    raise ValueError("SPAN_CHUNK_PAPER_BINDING_MISMATCH")
                if (
                    s["text"] != c["chunk_text"][s["start"] : s["end"]]
                    or s["block_ids"] != c["block_ids"]
                    or (s["page_start"], s["page_end"]) != (c["page_start"], c["page_end"])
                ):
                    raise ValueError("SPAN_TEXT_OR_BLOCK_PROVENANCE_MISMATCH")
                if (
                    span.provenance.span_id != sid
                    or span.provenance.chunk_id != s["chunk_id"]
                    or span.provenance.paper_id != s["paper_id"]
                ):
                    raise ValueError("SPAN_SOURCE_PROVENANCE_MISMATCH")
                self.page(page_id("Paper", paper.id), "Paper", paper.id)
                evidence[sid] = EvidenceRecord(span_id=sid, **s)
                papers[p["paper_id"]] = PaperRecord(**p)
                paths[sid] = [chunk.id, paper.id]
                used_edges.update([to_paper[0].id, from_chunk[0].id, parent[0].id])
        bundle_identity = "synthesis-bundle:" + digest(
            [VERSION, self.graph.graph_digest, self.wiki_manifest_digest, page.id]
        )
        return SynthesisBundle(
            question=QuestionCore(
                question_id=page.source_node_id,
                question_text=page.source_record["question"],
                stratum=page.source_record["complexity_stratum"],
                question_page_id=page.id,
            ),
            source_units=units,
            evidence=[evidence[s] for s in sorted(evidence)],
            source_papers=[papers[p] for p in sorted(papers)],
            provenance=BundleProvenance(
                projection_version=VERSION,
                bundle_id=bundle_identity,
                graph_digest=self.graph.graph_digest,
                wiki_manifest_digest=self.wiki_manifest_digest,
                source_edge_ids=sorted(used_edges),
                span_paths=paths,
            ),
        )

    def build(self, question_page_id):
        bundle = self._mandatory(question_page_id)
        if len(bundle_bytes(bundle).decode("utf-8")) > MAX_CHARS:
            raise OverBudget(bundle)
        return bundle

    def validate(self, bundle, expected_sha256=None):
        model = (
            bundle
            if isinstance(bundle, SynthesisBundle)
            else SynthesisBundle.model_validate(bundle)
        )
        raw = bundle_bytes(model)
        if expected_sha256 is not None and hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise ValueError("TAMPERED_BUNDLE_DIGEST")
        expected = self.build(model.question.question_page_id)
        if raw != bundle_bytes(expected):
            raise ValueError("PROJECTION_GRAPH_WIKI_ROUNDTRIP_MISMATCH")
        return model


def candidate_scope(bundle):
    """Validation-only view of R2's selected units; never a generator input bundle."""
    model = (
        bundle if isinstance(bundle, SynthesisBundle) else SynthesisBundle.model_validate(bundle)
    )
    serialized_chars = len(bundle_bytes(model).decode("utf-8"))
    if serialized_chars > MAX_CHARS:
        raise OverBudget(model)
    return {
        "linked_pages": [
            {
                "type": "ClaimPage" if u.type == "CLAIM" else "RequirementPage",
                "source_node_id": u.graph_node_id,
            }
            for u in model.source_units
        ],
        "evidence": [e.model_dump() for e in model.evidence],
    }


def validate_candidate(value, bundle):
    # The candidate schema and its coverage rules are the frozen R1 implementation.
    return validate_r1(value, candidate_scope(bundle))


def structural_candidate(bundle):
    model = (
        bundle if isinstance(bundle, SynthesisBundle) else SynthesisBundle.model_validate(bundle)
    )
    return validate_candidate(
        {
            "statements": [
                {
                    "text": "Offline structural witness.",
                    "source_units": [
                        {"type": u.type, "id": u.graph_node_id} for u in model.source_units
                    ],
                    "evidence_span_ids": [e.span_id for e in model.evidence],
                }
            ]
        },
        model,
    )


assert R1Synthesis.model_config["extra"] == "forbid"
