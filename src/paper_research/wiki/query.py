"""Agent-facing deterministic APIs, literal search and record-only context bundles."""

import copy
import json
import unicodedata
from pathlib import Path

from paper_research.evidence_graph.schema import canonical, digest

from .builder import dependency_index, descriptors, sha, validate_pages
from .schema import DEFAULT_BUDGET, Page, filename, page_id


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def page_core(page):
    return {
        k: copy.deepcopy(page.model_dump()[k])
        for k in (
            "id",
            "type",
            "title",
            "source_node_id",
            "source_record",
            "evidence_status",
            "provenance",
        )
    }


class WikiQuery:
    def __init__(self, graph, pages, lineage=None):
        validate_pages(graph, pages, lineage=lineage)
        self.graph = graph
        self.pages = {k: Page.model_validate(v.model_dump()) for k, v in pages.items()}

    @classmethod
    def from_artifact(cls, directory: Path, graph):
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        index = json.loads((directory / "index.json").read_text(encoding="utf-8"))
        lineage = json.loads((directory / "lineage.json").read_text(encoding="utf-8"))
        deps = json.loads((directory / "dependency_index.json").read_text(encoding="utf-8"))
        content = dict(manifest)
        content.pop("scientific_digest")
        if (
            manifest["graph_digest"] != graph.graph_digest
            or digest(content) != manifest["scientific_digest"]
        ):
            raise ValueError("WIKI_MANIFEST_MISMATCH")
        if (
            digest(index["entries"]) != index["index_digest"]
            or index["index_digest"] != manifest["index_digest"]
            or digest(deps) != manifest["dependency_index_digest"]
            or digest(lineage) != manifest["lineage_digest"]
        ):
            raise ValueError("WIKI_INDEX_MISMATCH")
        pages = {}
        for entry in index["entries"]:
            pid = entry["page_id"]
            if pid in pages:
                raise ValueError("DUPLICATE_PAGE_ID")
            if (
                entry["path"] != "pages/json/" + filename(pid) + ".json"
                or entry["markdown_path"] != "pages/md/" + filename(pid) + ".md"
            ):
                raise ValueError("INVALID_PAGE_PATH")
            if (
                sha(directory / entry["path"]) != entry["digest"]
                or sha(directory / entry["markdown_path"]) != entry["markdown_digest"]
            ):
                raise ValueError("TAMPERED_CANONICAL_JSON_OR_MARKDOWN")
            page = Page.model_validate_json((directory / entry["path"]).read_bytes())
            from .render import render_markdown

            if (directory / entry["markdown_path"]).read_bytes() != render_markdown(page).encode(
                "utf-8"
            ):
                raise ValueError("MARKDOWN_DERIVED_VIEW_MISMATCH")
            if (page.id, page.type, page.source_node_id, page.title) != (
                pid,
                entry["page_type"],
                entry["source_node_id"],
                entry["title"],
            ):
                raise ValueError("PAGE_INDEX_MISMATCH")
            pages[pid] = page
        if len(pages) != manifest["page_count"] or canonical(deps) != canonical(
            dependency_index(descriptors(graph))
        ):
            raise ValueError("DEPENDENCY_INDEX_OR_PAGE_COUNT_MISMATCH")
        return cls(graph, pages, lineage)

    def get_page(self, identity):
        if identity not in self.pages:
            raise KeyError("UNKNOWN_WIKI_PAGE")
        return copy.deepcopy(self.pages[identity].model_dump())

    def page_for_claim(self, identity):
        return self.get_page(page_id("Claim", identity))

    def page_for_requirement(self, identity):
        return self.get_page(page_id("Requirement", identity))

    def pages_for_paper(self, identity):
        if identity not in self.graph.nodes or self.graph.nodes[identity].type != "Paper":
            raise KeyError("UNKNOWN_GRAPH_PAPER")
        pid = page_id("Paper", identity)
        found = {pid}
        for p in self.pages.values():
            if pid in p.links.get("papers", []):
                found.add(p.id)
        return [self.get_page(p) for p in sorted(found)]

    def backlinks(self, identity):
        return self.get_page(identity)["backlinks"]

    def trace_sources(self, identity):
        page = self.get_page(identity)
        nodes = {page["source_node_id"]}
        for path in page["provenance_paths"]:
            nodes.update(path)
        return {
            "page_id": identity,
            "graph_digest": page["provenance"]["graph_digest"],
            "paths": page["provenance_paths"],
            "nodes": [self.graph.nodes[n].model_dump() for n in sorted(nodes)],
            "edges": [
                self.graph.edges[e].model_dump() for e in page["provenance"]["source_edge_ids"]
            ],
            "provenance": page["provenance"],
        }

    def search(self, query):
        if not isinstance(query, str) or not query.strip():
            return []
        return sorted(
            p.id
            for p in self.pages.values()
            if query in (p.id, p.source_node_id, p.title)
            or normalized(query) in normalized(p.title)
        )

    def build_context_bundle(self, identity, **limits):
        budget = {**DEFAULT_BUDGET, **limits}
        if set(budget) != set(DEFAULT_BUDGET) or any(
            type(v) is not int or v < 0 for v in budget.values()
        ):
            raise ValueError("INVALID_BUNDLE_BUDGET")
        page = self.get_page(identity)
        paper_ids = sorted(page["links"].get("papers", []))
        if page["type"] == "PaperPage":
            paper_ids = [identity]
        direct = sorted({p for values in page["links"].values() for p in values} - set(paper_ids))
        candidates = list(dict.fromkeys([*paper_ids, *direct, *page["backlinks"]]))
        bundle = {
            "page": page_core(self.pages[identity]),
            "linked_pages": [],
            "evidence": [],
            "paper_provenance": [],
            "budget": budget,
            "omitted": {"evidence": len(page["evidence"]), "linked_pages": len(candidates)},
        }

        def length(value):
            return len(canonical(value).decode("utf-8"))

        if length(bundle) > budget["max_chars"]:
            raise ValueError("BUNDLE_CORE_EXCEEDS_BUDGET")
        seen_papers = set()
        for evidence in page["evidence"][: budget["max_evidence_blocks"]]:
            candidate = copy.deepcopy(bundle)
            candidate["evidence"].append(evidence)
            candidate["omitted"]["evidence"] -= 1
            paper = evidence["graph_paper_node_id"]
            if paper not in seen_papers:
                candidate["paper_provenance"].append(self.graph.nodes[paper].model_dump())
            if length(candidate) > budget["max_chars"]:
                break
            bundle = candidate
            seen_papers.add(paper)
        for pid in candidates[: budget["max_linked_pages"]]:
            candidate = copy.deepcopy(bundle)
            candidate["linked_pages"].append(page_core(self.pages[pid]))
            candidate["omitted"]["linked_pages"] -= 1
            if length(candidate) > budget["max_chars"]:
                break
            bundle = candidate
        return bundle
