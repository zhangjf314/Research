"""Uniform native geometry projection; all V1 detector predicates/config reused."""

from pathlib import Path

import geometry_v2 as g
import pymupdf as fitz

from paper_research.pdf_multimodal import extraction as old
from paper_research.pdf_multimodal.geometry import gap, overlap, q, union, valid
from paper_research.pdf_multimodal.identity import canonical, digest, file_sha, native


def unit(
    kind, components, box, paper, sha, page, width, height, c, caption=None, text=None, branch=None
):
    identities = [s["identity"] for s in components]
    result = old.unit(
        kind, components, box, paper, sha, page, width, height, c, caption, text, branch
    )
    result["provenance"]["protocol_digest"] = g.protocol_digest(c)
    result["unit_id"] = g.unit_id(sha, page, kind, box, identities)
    result["provenance"]["source_geometries"] = [
        {
            "component_id": s["component_id"],
            "raw_bbox_native": s["raw_bbox_native"],
            "raw_bbox": s["raw_bbox"],
            "visible_bbox": s["visible_bbox"],
            "visibility_class": s["visibility_class"],
        }
        for s in components
    ]
    return result


def construct(components, pages, papers, c):
    units, anchors = [], []
    for p in pages:
        paper, sha, pi, w, h = (
            p["paper_id"],
            p["pdf_sha256"],
            p["page_index"],
            p["width_pt"],
            p["height_pt"],
        )
        local = [s for s in components if s["paper_id"] == paper and s["page_index"] == pi]
        visible = [s for s in local if s["evidence_visible"]]
        blocks = [s for s in visible if s["identity"]["primitive_type"] == "TEXT_BLOCK"]
        caps, local_units = [], []
        for s in blocks:
            local_units.append(
                unit(
                    "TEXT_BLOCK",
                    [s],
                    s["visible_bbox"],
                    paper,
                    sha,
                    pi,
                    w,
                    h,
                    c,
                    text=old.raw_text(s["raw"]),
                )
            )
            # Exact frozen caption function / regex, projected component bbox only.
            cap = old.caption_anchor(s, c)
            if cap:
                caps.append(cap)
        for s in visible:
            if s["identity"]["primitive_type"] == "TEXT_LINE":
                parent = s["identity"]["references"]["block_component_id"]
                branch = old.equation_branch(
                    s["raw"], w, any(a["source_component_id"] == parent for a in caps), c
                )
                if branch:
                    local_units.append(
                        unit(
                            "EQUATION_CANDIDATE",
                            [s],
                            s["visible_bbox"],
                            paper,
                            sha,
                            pi,
                            w,
                            h,
                            c,
                            text="".join(span["text"] for span in s["raw"]["spans"]),
                            branch=branch,
                        )
                    )
        visuals = [
            s
            for s in visible
            if s["identity"]["primitive_type"] in ("RASTER_OBJECT", "VECTOR_OBJECT")
        ]
        for cap in caps:
            if cap["caption_type"] != "FIGURE":
                continue
            rule = c["figures"]
            passing = [
                s
                for s in visuals
                if float((s["bbox"][2] - s["bbox"][0]) * (s["bbox"][3] - s["bbox"][1]))
                >= float(rule["area_min"]) * float(w * h)
                and overlap(s["bbox"], cap["bbox"]) >= float(rule["overlap_min"])
                and gap(s["bbox"], cap["bbox"]) <= float(rule["gap_max_height"]) * float(h)
            ]
            passing.sort(
                key=lambda s: (
                    gap(s["bbox"], cap["bbox"]) / float(h) + 1 - overlap(s["bbox"], cap["bbox"]),
                    s["bbox"][3] > cap["bbox"][1],
                    s["component_id"],
                )
            )
            if passing:
                block = next(s for s in local if s["component_id"] == cap["source_component_id"])
                local_units.append(
                    unit(
                        "FIGURE_CANDIDATE",
                        passing + [block],
                        union([s["bbox"] for s in passing]),
                        paper,
                        sha,
                        pi,
                        w,
                        h,
                        c,
                        caption=cap,
                    )
                )
        for s in visible:
            if s["identity"]["primitive_type"] != "TABLE_PRIMITIVE":
                continue
            rule = c["tables"]
            eligible = [
                a
                for a in caps
                if a["caption_type"] == "TABLE"
                and overlap(a["bbox"], s["bbox"]) >= float(rule["caption_overlap_min"])
                and gap(a["bbox"], s["bbox"]) <= float(rule["caption_gap_max_height"]) * float(h)
            ]
            eligible.sort(
                key=lambda a: (
                    gap(a["bbox"], s["bbox"]),
                    a["bbox"][3] > s["bbox"][1],
                    a["caption_id"],
                )
            )
            cap = eligible[0] if eligible else None
            cs = [s] + (
                [next(b for b in local if b["component_id"] == cap["source_component_id"])]
                if cap
                else []
            )
            text = "\n".join("\t".join(v or "" for v in row) for row in s["raw"]["text"])
            local_units.append(
                unit(
                    "TABLE_CANDIDATE",
                    cs,
                    s["bbox"],
                    paper,
                    sha,
                    pi,
                    w,
                    h,
                    c,
                    caption=cap,
                    text=text,
                )
            )
        p.update(
            text_block_ids=[u["unit_id"] for u in local_units if u["unit_type"] == "TEXT_BLOCK"],
            raster_object_ids=[
                s["component_id"]
                for s in local
                if s["identity"]["primitive_type"] == "RASTER_OBJECT"
            ],
            vector_object_ids=[
                s["component_id"]
                for s in local
                if s["identity"]["primitive_type"] == "VECTOR_OBJECT"
            ],
            caption_anchor_ids=[a["caption_id"] for a in caps],
            candidate_ids=[u["unit_id"] for u in local_units if u["unit_type"] != "TEXT_BLOCK"],
        )
        chars = sum(not char.isspace() for s in blocks for char in old.raw_text(s["raw"]))
        largest = max(
            [
                float((s["bbox"][2] - s["bbox"][0]) * (s["bbox"][3] - s["bbox"][1])) / float(w * h)
                for s in visuals
                if s["identity"]["primitive_type"] == "RASTER_OBJECT"
            ]
            or [0]
        )
        dominant = chars < c["image_dominant"]["visible_text_chars_lt"] and largest >= float(
            c["image_dominant"]["largest_raster_area_min"]
        )
        p.update(image_dominant=dominant, OCR_REQUIRED=dominant, ocr_executed=False)
        units.extend(local_units)
        anchors.extend(caps)
    return {
        "papers": papers,
        "pages": pages,
        "components": components,
        "anchors": anchors,
        "units": units,
        "protocol_digest": g.protocol_digest(c),
    }


def table_dump(table, index, c):
    return {
        "bbox": native(table.bbox),
        "cells": native(table.cells),
        "rows": [native(r.cells) for r in table.rows],
        "text": table.extract(),
        "row_count": table.row_count,
        "col_count": table.col_count,
        "header": {
            "bbox": native(table.header.bbox),
            "cells": native(table.header.cells),
            "names": table.header.names,
            "external": table.header.external,
        },
        "finder_identity": {"table_index": index, "config_digest": digest(c["calls"]["tables"])},
    }


def extract_synthetic(path, paper, expected, c, authorized_shas):
    g.guard(c)
    sha = file_sha(path)
    if sha != expected:
        raise ValueError("PDF_SHA_MISMATCH")
    root = Path(__file__).resolve().parents[4]
    import json

    formal = {
        json.loads(line)["expected_sha256"]
        for line in (
            root / "artifacts/paperresearch-v2/phase4a-r1-source-recovery/final_corpus/papers.jsonl"
        )
        .read_text(encoding="utf-8")
        .splitlines()
    }
    if sha in formal or sha not in authorized_shas or not paper.startswith("synthetic:"):
        raise ValueError("FORMAL_EXTRACTION_NOT_AUTHORIZED")
    components, pages = [], []
    with fitz.open(path) as doc:
        for pi, page in enumerate(doc):
            rect = page.rect * page.derotation_matrix
            width, height = rect.width, rect.height
            p = {
                "paper_id": paper,
                "pdf_sha256": sha,
                "page_index": pi,
                "page_number": pi + 1,
                "width_pt": q(width),
                "height_pt": q(height),
                "width_native": width,
                "height_native": height,
                "page_bounds_native": [0, 0, width, height],
                "rotation": page.rotation,
                "mediabox": native(page.mediabox),
                "cropbox": native(page.cropbox),
            }
            calls = c["calls"]
            text = page.get_text(**calls["text"]["kwargs"])
            images = page.get_image_info(**calls["images"]["kwargs"])
            drawings = page.get_drawings(**calls["vectors"]["kwargs"])
            finder = old.find_tables_unrotated(page, calls["tables"]["kwargs"])
            for bi, block in enumerate(text["blocks"]):
                if block["type"] != 0:
                    continue
                s = g.source(
                    "TEXT_BLOCK",
                    block,
                    block["bbox"],
                    paper,
                    sha,
                    pi,
                    width,
                    height,
                    {"block_index": bi},
                )
                components.append(s)
                cap = old.caption_anchor(s, c) if s["evidence_visible"] else None
                for li, line in enumerate(block["lines"]):
                    if old.equation_branch(line, q(width), cap is not None, c):
                        components.append(
                            g.source(
                                "TEXT_LINE",
                                line,
                                line["bbox"],
                                paper,
                                sha,
                                pi,
                                width,
                                height,
                                {
                                    "block_component_id": s["component_id"],
                                    "block_index": bi,
                                    "line_index": li,
                                    "span_indices": list(range(len(line["spans"]))),
                                },
                            )
                        )
            for raw in images:
                components.append(
                    g.source(
                        "RASTER_OBJECT",
                        raw,
                        raw["bbox"],
                        paper,
                        sha,
                        pi,
                        width,
                        height,
                        {
                            "xref": raw.get("xref"),
                            "digest": native(raw.get("digest")),
                            "raw_placement_bbox": [q(v) for v in raw["bbox"]],
                        },
                    )
                )
            for di, raw in enumerate(drawings):
                components.append(
                    g.source(
                        "VECTOR_OBJECT",
                        raw,
                        raw["rect"],
                        paper,
                        sha,
                        pi,
                        width,
                        height,
                        {"drawing_index": di},
                    )
                )
            for ti, table in enumerate(finder.tables):
                raw = table_dump(table, ti, c)
                components.append(
                    g.source(
                        "TABLE_PRIMITIVE",
                        raw,
                        table.bbox,
                        paper,
                        sha,
                        pi,
                        width,
                        height,
                        {"table_index": ti, "config_digest": digest(calls["tables"])},
                    )
                )
            pages.append(p)
    return construct(components, pages, {paper: {"pdf_sha256": sha, "page_count": len(pages)}}, c)


def oracle_raw(path, s, c):
    ref = s["identity"]["references"]
    kind = s["identity"]["primitive_type"]
    with fitz.open(path) as doc:
        page = doc[s["page_index"]]
        if kind in ("TEXT_BLOCK", "TEXT_LINE"):
            block = page.get_text(**c["calls"]["text"]["kwargs"])["blocks"][ref["block_index"]]
            return native(block if kind == "TEXT_BLOCK" else block["lines"][ref["line_index"]])
        if kind == "RASTER_OBJECT":
            raw = next(
                (
                    raw
                    for raw in page.get_image_info(**c["calls"]["images"]["kwargs"])
                    if digest(raw) == s["identity"]["native_primitive_digest"]
                ),
                None,
            )
            if raw is None:
                raise ValueError("RAW_PRIMITIVE_NOT_FOUND")
            return native(raw)
        if kind == "VECTOR_OBJECT":
            return native(
                page.get_drawings(**c["calls"]["vectors"]["kwargs"])[ref["drawing_index"]]
            )
        if kind == "TABLE_PRIMITIVE":
            return table_dump(
                old.find_tables_unrotated(page, c["calls"]["tables"]["kwargs"]).tables[
                    ref["table_index"]
                ],
                ref["table_index"],
                c,
            )
    raise ValueError("UNKNOWN_SOURCE_KIND")


def validate(bundle, c, source_paths):
    g.guard(c)
    if bundle["protocol_digest"] != g.protocol_digest(c):
        raise ValueError("PARSER_CONFIG_DIGEST_MISMATCH")
    known = bundle["papers"]
    page_registry = {}
    for schema_name, records in [
        ("page_record", bundle["pages"]),
        ("evidence_unit", bundle["units"]),
    ]:
        if any(set(record) != set(c["schemas"][schema_name]["required"]) for record in records):
            raise ValueError("SCHEMA_FIELDS")
    for paper, info in known.items():
        if paper not in source_paths or file_sha(source_paths[paper]) != info["pdf_sha256"]:
            raise ValueError("PDF_SHA_OR_UNKNOWN_PAPER")
        with fitz.open(source_paths[paper]) as doc:
            if doc.page_count != info["page_count"]:
                raise ValueError("PAGE_COUNT")
    for p in bundle["pages"]:
        key = p["paper_id"], p["page_index"]
        if (
            key in page_registry
            or p["paper_id"] not in known
            or not 0 <= p["page_index"] < known[p["paper_id"]]["page_count"]
        ):
            raise ValueError("INVALID_PAGE")
        with fitz.open(source_paths[p["paper_id"]]) as doc:
            page = doc[p["page_index"]]
            rect = page.rect * page.derotation_matrix
            if canonical(
                [
                    p["width_native"],
                    p["height_native"],
                    p["width_pt"],
                    p["height_pt"],
                    p["page_bounds_native"],
                    p["rotation"],
                    p["cropbox"],
                    p["mediabox"],
                ]
            ) != canonical(
                [
                    rect.width,
                    rect.height,
                    q(rect.width),
                    q(rect.height),
                    [0, 0, rect.width, rect.height],
                    page.rotation,
                    native(page.cropbox),
                    native(page.mediabox),
                ]
            ):
                raise ValueError("WRONG_PAGE_BOUNDS")
        if (
            p["pdf_sha256"] != known[p["paper_id"]]["pdf_sha256"]
            or p["page_number"] != p["page_index"] + 1
        ):
            raise ValueError("PAGE_IDENTITY")
        page_registry[key] = p
    seen = set()
    for s in bundle["components"]:
        p = page_registry.get((s["paper_id"], s["page_index"]))
        if not p or s["pdf_sha256"] != p["pdf_sha256"]:
            raise ValueError("CROSS_PAGE_COMPONENT")
        raw = oracle_raw(source_paths[s["paper_id"]], s, c)
        raw_box = raw["rect"] if s["identity"]["primitive_type"] == "VECTOR_OBJECT" else raw["bbox"]
        projection = g.project(list(raw_box), p["width_native"], p["height_native"])
        for k, v in projection.items():
            if canonical(s[k]) != canonical(v):
                raise ValueError("RAW_OR_VISIBLE_PROJECTION_TAMPER: " + k)
        cls, visible = g.independent_projection(raw_box, p["width_native"], p["height_native"])
        if cls != s["visibility_class"] or visible != s["visible_bbox"]:
            raise ValueError("INDEPENDENT_PROJECTION_MISMATCH")
        expected = g.source(
            s["identity"]["primitive_type"],
            raw,
            raw_box,
            s["paper_id"],
            s["pdf_sha256"],
            s["page_index"],
            p["width_native"],
            p["height_native"],
            s["identity"]["references"],
        )
        if canonical(expected) != canonical(s) or s["component_id"] in seen:
            raise ValueError("SOURCE_COMPONENT_TAMPER_OR_DUPLICATE")
        seen.add(s["component_id"])
    registry = {s["component_id"]: s for s in bundle["components"]}
    for s in bundle["components"]:
        if s["identity"]["primitive_type"] == "TEXT_LINE":
            refs = s["identity"]["references"]
            parent = registry.get(refs["block_component_id"])
            if not parent or (parent["paper_id"], parent["page_index"]) != (
                s["paper_id"],
                s["page_index"],
            ):
                raise ValueError("MISSING_OR_CROSS_PAGE_PARENT")
            if canonical(parent["raw"]["lines"][refs["line_index"]]) != canonical(s["raw"]) or refs[
                "span_indices"
            ] != list(range(len(s["raw"]["spans"]))):
                raise ValueError("SOURCE_LINE_REFERENCES")
    unit_seen = set()
    for u in bundle["units"]:
        if u["unit_id"] in unit_seen:
            raise ValueError("DUPLICATE_CANONICAL_ID")
        unit_seen.add(u["unit_id"])
        p = page_registry.get((u["paper_id"], u["page_index"]))
        if not p or u["pdf_sha256"] != p["pdf_sha256"]:
            raise ValueError("UNKNOWN_UNIT_PAPER_OR_PAGE")
        valid(u["bbox"], p["width_pt"], p["height_pt"])
    expected_pages = [
        {
            k: v
            for k, v in p.items()
            if k
            not in (
                "text_block_ids",
                "raster_object_ids",
                "vector_object_ids",
                "caption_anchor_ids",
                "candidate_ids",
                "image_dominant",
                "OCR_REQUIRED",
                "ocr_executed",
            )
        }
        for p in bundle["pages"]
    ]
    rebuilt = construct(bundle["components"], expected_pages, bundle["papers"], c)
    # Rebuild from verified raw primitives, never copy candidate text/provenance as oracle.
    if canonical(rebuilt) != canonical(bundle):
        raise ValueError("TEXT_PROVENANCE_OR_CANDIDATE_TAMPER")
    textual = sum(u["raw_text"] is not None for u in bundle["units"]) + len(bundle["anchors"])
    denominators = {k: len(bundle["units"]) for k in c["metrics"] if k != "scope"}
    denominators.update(
        page_identity_exact=len(bundle["pages"]),
        source_component_exact=len(bundle["components"]),
        raw_source_bbox_exact=len(bundle["components"]),
        visible_projection_exact=len(bundle["components"]),
        text_roundtrip_exact=textual,
    )
    return {
        k: {"denominator": n, "numerator": n, "value": 1.0 if n else "NOT_ASSESSED"}
        for k, n in denominators.items()
    }
