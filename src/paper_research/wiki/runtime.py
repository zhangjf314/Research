"""Truthful canonical Wiki projection; reuse WikiQuery APIs without legacy gold labels."""

import copy
from typing import Literal

from paper_research.evidence_graph.schema import digest, node_id

from .query import WikiQuery
from .schema import Footer, Page, page_id


class RuntimePage(Page):
    evidence_status: Literal["STRUCTURAL_SOURCE"] = "STRUCTURAL_SOURCE"


class RuntimeWikiQuery(WikiQuery):
    """Reuse search/get_page/pages_for_paper/backlinks/trace_sources/context budgets.

    The old validate_pages requires frozen gold. This constructor instead derives
    every page from an already validated runtime graph; no caller pages are accepted.
    RuntimeGraph.load compares the persisted entire projection to this derivation.
    """

    def __init__(self, graph):
        self.graph, self.pages = graph, {}
        for paper in sorted(
            (n for n in graph.nodes.values() if n.type == "Paper"), key=lambda n: n.id
        ):
            chunks = sorted(
                (
                    n
                    for n in graph.nodes.values()
                    if n.type == "Chunk"
                    and n.properties["paper_id"] == paper.properties["paper_id"]
                ),
                key=lambda n: (n.properties["page_start"], n.id),
            )
            evidence, paths, edge_ids = [], [], set()
            for chunk in chunks:
                trace = graph.trace_provenance(chunk.id)
                edge_ids.update(trace["edge_ids"])
                paths.extend([list(p) for p in trace["paths"]])
                c = chunk.properties
                evidence.append(
                    {
                        "evidence_id": c["chunk_id"],
                        "chunk_id": c["chunk_id"],
                        "paper_id": c["paper_id"],
                        "start": 0,
                        "end": len(c["chunk_text"]),
                        "text": c["chunk_text"],
                        "page_start": c["page_start"],
                        "page_end": c["page_end"],
                        "block_ids": c["block_ids"],
                        "block_page_map": c["block_page_map"],
                        "pdf_sha256": paper.properties["pdf_sha256"],
                        "graph_paper_node_id": node_id("Paper", c["paper_id"]),
                    }
                )
            identity = page_id("Paper", paper.id)
            self.pages[identity] = RuntimePage(
                id=identity,
                type="PaperPage",
                source_node_id=paper.id,
                title=paper.properties["title"],
                source_record=copy.deepcopy(paper.properties),
                links={},
                backlinks=[],
                linked_records=[],
                evidence=evidence,
                provenance_paths=paths,
                provenance=Footer(
                    graph_digest=graph.graph_digest,
                    source_node_id=paper.id,
                    source_edge_ids=sorted(edge_ids),
                    source_artifact_hashes=graph.source_hashes,
                    wiki_builder_version="AGENT_NATIVE_WIKI_FOUNDATION_V1",
                    dependency_digest=digest({"paper": paper.id, "paths": paths}),
                ),
            )
