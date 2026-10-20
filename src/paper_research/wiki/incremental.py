"""Dependency-local invalidation and page reuse. No render-all incremental fallback."""

from paper_research.evidence_graph.schema import canonical, digest

from .builder import dependency_index, descriptors, make_page, validate_pages


def incremental_build(old_graph, new_graph, old_pages, old_dependencies, old_lineage):
    old_desc, new_desc = descriptors(old_graph), descriptors(new_graph)
    if canonical(old_dependencies) != canonical(dependency_index(old_desc)):
        raise ValueError("DEPENDENCY_INDEX_OMISSION_OR_TAMPER")
    validate_pages(old_graph, old_pages, old_desc, old_lineage)
    # Route record changes through the old/new reverse dependency indexes. A topology
    # selector pass also catches added/removed relationships absent from the old index.
    new_dependencies = dependency_index(new_desc)
    candidates = set(old_desc) ^ set(new_desc)
    for key, old_records, new_records in (
        ("nodes", old_graph.nodes, new_graph.nodes),
        ("edges", old_graph.edges, new_graph.edges),
    ):
        changed = {
            identity
            for identity in set(old_records) | set(new_records)
            if identity not in old_records
            or identity not in new_records
            or canonical(old_records[identity].model_dump())
            != canonical(new_records[identity].model_dump())
        }
        for identity in changed:
            candidates.update(old_dependencies[key].get(identity, []))
            candidates.update(new_dependencies[key].get(identity, []))
    for pid in set(old_desc) & set(new_desc):
        if any(
            old_desc[pid][field] != new_desc[pid][field]
            for field in ("links", "backlinks", "spans", "edges", "paths", "nodes")
        ):
            candidates.add(pid)
    affected = {
        pid
        for pid in candidates
        if pid not in old_desc
        or pid not in new_desc
        or old_desc[pid]["dependency_digest"] != new_desc[pid]["dependency_digest"]
    }
    updated = {}
    for pid in sorted(new_desc):
        updated[pid] = (
            make_page(new_graph, pid, new_desc[pid]) if pid in affected else old_pages[pid]
        )
    lineage = {anchor: dict(pages) for anchor, pages in old_lineage.items()}
    lineage[new_graph.graph_digest] = {pid: d["dependency_digest"] for pid, d in new_desc.items()}
    for pid in set(updated) - affected:
        if canonical(updated[pid].model_dump()) != canonical(old_pages[pid].model_dump()):
            raise ValueError("UNAFFECTED_PAGE_CHANGED")
    validate_pages(
        new_graph,
        updated,
        new_desc,
        lineage,
        {pid: old_pages[pid] for pid in set(updated) - affected},
    )
    return (
        updated,
        new_desc,
        lineage,
        {
            "affected_pages": sorted(affected),
            "rebuilt_pages": sorted(affected & set(updated)),
            "deleted_pages": sorted(affected - set(updated)),
            "reused_pages": sorted(set(updated) - affected),
            "page_builder_calls": len(affected & set(updated)),
            "dependency_index_digest": digest(dependency_index(new_desc)),
        },
    )


def validate_invalidation(actual, expected):
    if set(actual) != set(expected):
        raise ValueError("INCREMENTAL_OVER_OR_UNDER_INVALIDATION")
