"""D6-only structural graph adapter. Frozen benchmark builders remain unchanged."""

import argparse
import hashlib
import json
from dataclasses import dataclass

from paper_research.indexing.registry import IndexIdentityError

from .builder import SourceData, build_graph
from .query import GraphQuery
from .schema import Provenance, canonical, digest

VERSION = "D6_STRUCTURAL_GRAPH_WIKI_V1"
MAX_CHUNKS = 5000
MAX_FILE_BYTES = 32 * 1024 * 1024


class GraphRuntimeError(IndexIdentityError):
    def __init__(self, code, status_code=409):
        super().__init__(code, status_code=status_code)


@dataclass(frozen=True)
class RuntimeSources(SourceData):
    def provenance(self, key, record, *, semantic=False, chunk=None):
        if semantic or key not in {"metadata", "chunks"}:
            raise GraphRuntimeError("GRAPH_RUNTIME_GOLD_FORBIDDEN")
        return Provenance(
            source_phase="D6_ACTIVE_CANONICAL",
            source_artifact="runtime:" + key,
            source_sha256=self.source_hashes["runtime:" + key],
            source_record_id=record,
            construction_method="deterministic_identity",
            evidence_status="STRUCTURAL_SOURCE",
            paper_id=chunk["paper_id"] if chunk else None,
            chunk_id=chunk["chunk_id"] if chunk else None,
        )


def active_sources(registry, chunks, collection):
    """Call only inside query_admission: canonical and physical bytes are already verified."""
    if not chunks or len(chunks) > MAX_CHUNKS:
        raise GraphRuntimeError("GRAPH_CORPUS_SIZE_UNSUPPORTED", 422)
    binding = registry.read()["indexes"][collection]["trust_binding"]
    inventory = registry.inventory(binding["generation_id"])
    bundle = registry.authority_bundle(binding["generation_id"], inventory=inventory)
    pdfs = {d["paper_id"]: d["pdf_sha256"] for d in bundle["commitment"]["documents"]}
    metadata = {
        pid: {"title": "Paper " + pid, "pdf_sha256": sha} for pid, sha in sorted(pdfs.items())
    }
    # Some canonical Chunk payloads omit block_page_map. Never substitute the
    # first page for every block: recover it only from sealed parser blocks.
    blocks = {}
    for doc in bundle["commitment"]["documents"]:
        from paper_research.indexing.identity import trust_json_loads
        from paper_research.parsing.types import PaperBlock

        raw = registry.file_bytes(
            registry.protected_path(f"parsed/{doc['paper_id']}/paper_blocks.jsonl"),
            "GRAPH_SOURCE_UNAVAILABLE",
        )
        if hashlib.sha256(raw).hexdigest() != doc["parse_blocks_sha256"]:
            raise GraphRuntimeError("GRAPH_CANONICAL_BLOCKS_CHANGED")
        rows = [
            PaperBlock.model_validate(trust_json_loads(line)) for line in raw.splitlines() if line
        ]
        blocks[doc["paper_id"]] = {b.block_id: b for b in rows}
        if len(blocks[doc["paper_id"]]) != len(rows):
            raise GraphRuntimeError("GRAPH_CANONICAL_BLOCKS_INVALID")
        metadata[doc["paper_id"]]["blocks_sha256"] = doc["parse_blocks_sha256"]
    records = {}
    for c in chunks:
        try:
            pages = {bid: blocks[c.paper_id][bid].page_start for bid in c.block_ids}
        except KeyError as exc:
            raise GraphRuntimeError("GRAPH_CANONICAL_BLOCKS_INVALID") from exc
        if (
            not pages
            or any(not c.page_start <= page <= c.page_end for page in pages.values())
            or (c.block_page_map is not None and c.block_page_map != pages)
        ):
            raise GraphRuntimeError("GRAPH_CANONICAL_BLOCKS_INVALID")
        records[c.chunk_id] = c.model_copy(update={"block_page_map": pages}).model_dump(mode="json")
    if len(records) != len(chunks) or {c.paper_id for c in chunks} != set(metadata):
        raise GraphRuntimeError("GRAPH_CANONICAL_MEMBERSHIP_INVALID")
    sources = RuntimeSources(
        metadata,
        records,
        {},
        {},
        {},
        {},
        {
            "runtime:metadata": digest(metadata),
            "runtime:chunks": digest(records),
        },
    )
    identity = {
        "schema_version": VERSION,
        "generation_id": binding["generation_id"],
        "inventory_digest": binding["inventory_digest"],
        "seal_digest": binding["seal_digest"],
        "index_identity": binding["index_identity"],
        "source_hashes": sources.source_hashes,
    }
    return sources, identity


def graph_path(settings):
    root = settings.data_dir.resolve()
    path = (root / "graph-runtime/index.json").resolve()
    if not path.is_relative_to(root):
        raise GraphRuntimeError("GRAPH_PATH_OUTSIDE_DATA_DIR")
    return path


class RuntimeGraph:
    def __init__(self, sources, binding):
        self.sources, self.binding = sources, binding
        nodes, edges = build_graph(sources)
        # Real GraphQuery source validation is reused, not bypassed.
        self.graph = GraphQuery(nodes, edges, sources)
        self.graph.graph_digest = digest(
            {
                "binding": binding,
                "nodes": [n.model_dump() for n in nodes],
                "edges": [e.model_dump() for e in edges],
            }
        )
        self.graph.source_hashes = sources.source_hashes
        from paper_research.wiki.runtime import RuntimeWikiQuery

        self.wiki = RuntimeWikiQuery(self.graph)

    def record(self):
        return {
            "schema_version": VERSION,
            "binding": self.binding,
            "graph_digest": self.graph.graph_digest,
            "nodes": [self.graph.nodes[n].model_dump() for n in sorted(self.graph.nodes)],
            "edges": [self.graph.edges[e].model_dump() for e in sorted(self.graph.edges)],
            "wiki_pages": {k: p.model_dump() for k, p in sorted(self.wiki.pages.items())},
        }

    @classmethod
    def load(cls, settings, registry, chunks, collection):
        sources, binding = active_sources(registry, chunks, collection)
        path = graph_path(settings)
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                raise GraphRuntimeError("GRAPH_INDEX_TOO_LARGE")
            raw = path.read_bytes()
        except FileNotFoundError as exc:
            raise GraphRuntimeError("GRAPH_INDEX_MISSING", 503) from exc
        except OSError as exc:
            raise GraphRuntimeError("GRAPH_INDEX_UNAVAILABLE", 503) from exc
        if len(raw) > MAX_FILE_BYTES:
            raise GraphRuntimeError("GRAPH_INDEX_TOO_LARGE")
        try:
            from paper_research.indexing.identity import trust_json_loads

            stored = trust_json_loads(raw)
            if stored.get("schema_version") != VERSION:
                raise GraphRuntimeError("GRAPH_RUNTIME_GOLD_OR_SCHEMA_FORBIDDEN")
            if stored.get("binding") != binding:
                raise GraphRuntimeError("GRAPH_INDEX_STALE")
            expected = cls(sources, binding)
            # Re-derive exact structure and Wiki from current canonical records. A rehashed
            # malicious edge, orphan page, copied gold node or altered quote still fails.
            if canonical(stored) != canonical(expected.record()):
                raise GraphRuntimeError("GRAPH_SOURCE_OR_TOPOLOGY_MISMATCH")
            return expected
        except GraphRuntimeError:
            raise
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            raise GraphRuntimeError("GRAPH_INDEX_INVALID") from exc


def build_active(provider):
    with provider._admission() as (chunks, _):
        sources, binding = active_sources(provider._registry, chunks, provider._collection)
        runtime = RuntimeGraph(sources, binding)
        raw = canonical(runtime.record()) + b"\n"
        if len(raw) > MAX_FILE_BYTES:
            raise GraphRuntimeError("GRAPH_INDEX_TOO_LARGE")
        provider._registry.durable_bytes(graph_path(provider.settings), raw)
        return {
            "status": "BUILT",
            "schema_version": VERSION,
            "binding": binding,
            "graph_digest": runtime.graph.graph_digest,
            "nodes": len(runtime.graph.nodes),
            "edges": len(runtime.graph.edges),
            "wiki_pages": len(runtime.wiki.pages),
            "bytes": len(raw),
            "semantic_relations": 0,
            "model_requests": 0,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["build"])
    parser.parse_args()
    from paper_research.agents.providers import HybridLocalResearchProvider
    from paper_research.config import get_settings

    provider = HybridLocalResearchProvider(get_settings(), require_identity=True)
    try:
        print(json.dumps(build_active(provider), ensure_ascii=False))
    finally:
        provider._client.close()


if __name__ == "__main__":
    main()
