"""Fail-closed identity and raw primitive roundtrip validation."""

import copy

import pymupdf as fitz

from .contract import TYPES, contract_guard
from .extraction import (
    canonical_text,
    caption_anchor,
    equation_branch,
    find_tables_unrotated,
    raw_text,
)
from .geometry import bbox, gap, normalized, overlap, union, valid
from .identity import canonical, digest, file_sha, native, unit_id


def primitive_lookup(path, component, contract):
    """Reread native source, never use a final unit as its own oracle."""
    ref = component["identity"]["references"]
    with fitz.open(path) as doc:
        page = doc[component["page_index"]]
        kind = component["identity"]["kind"]
        calls = contract["calls"]
        if kind in ("TEXT_BLOCK", "TEXT_LINE"):
            blocks = page.get_text(**calls["text"]["kwargs"])["blocks"]
            if kind == "TEXT_BLOCK":
                return native(blocks[ref["block_index"]])
            # Line reference includes the block identity; resolve via digest rather than guess.
            parent = ref["block_component_id"]
            for bi, b in enumerate(blocks):
                if b["type"] != 0:
                    continue
                ident = {
                    "kind": "TEXT_BLOCK",
                    "page_index": component["page_index"],
                    "raw_digest": digest(b),
                    "bbox": bbox(b["bbox"], page.cropbox.width, page.cropbox.height),
                    "references": {"block_index": bi},
                }
                if "src:" + digest({"pdf_sha256": component["pdf_sha256"], **ident}) == parent:
                    return native(b["lines"][ref["line_index"]])
        if kind == "RASTER_OBJECT":
            for raw in page.get_image_info(**calls["images"]["kwargs"]):
                if digest(raw) == component["identity"]["raw_digest"]:
                    return native(raw)
        if kind == "VECTOR_OBJECT":
            return native(page.get_drawings(**calls["vectors"]["kwargs"])[ref["drawing_index"]])
        if kind == "TABLE_PRIMITIVE":
            table = find_tables_unrotated(page, calls["tables"]["kwargs"]).tables[
                ref["table_index"]
            ]
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
                "finder_identity": {
                    "table_index": ref["table_index"],
                    "config_digest": digest(calls["tables"]),
                },
            }
    raise ValueError("SOURCE_COMPONENT_NOT_FOUND")


def validate(bundle, contract, source_paths, reread=True):
    contract_guard(contract)
    for schema_name, records in [
        ("page_record", bundle["pages"]),
        ("evidence_unit", bundle["units"]),
    ]:
        schema = contract["schemas"][schema_name]
        for record in records:
            if set(record) != set(schema["required"]):
                raise ValueError("SCHEMA_REQUIRED_OR_EXTRA_FIELDS")
    for record in bundle["units"]:
        if set(record["provenance"]) != set(contract["schemas"]["provenance"]["required"]):
            raise ValueError("PROVENANCE_SCHEMA_FIELDS")
    if bundle["protocol_digest"] != digest(contract):
        raise ValueError("PARSER_CONFIG_DIGEST_MISMATCH")
    papers = bundle["papers"]
    for paper, p in papers.items():
        if paper not in source_paths or file_sha(source_paths[paper]) != p["pdf_sha256"]:
            raise ValueError("WRONG_PDF_SHA_OR_UNKNOWN_PAPER")
        with fitz.open(source_paths[paper]) as doc:
            if doc.page_count != p["page_count"]:
                raise ValueError("PAGE_COUNT_MISMATCH")
    pages = {}
    for p in bundle["pages"]:
        key = (p["paper_id"], p["page_index"])
        if (
            key in pages
            or p["paper_id"] not in papers
            or not 0 <= p["page_index"] < papers[p["paper_id"]]["page_count"]
        ):
            raise ValueError("INVALID_PAGE")
        if (
            p["pdf_sha256"] != papers[p["paper_id"]]["pdf_sha256"]
            or p["page_number"] != p["page_index"] + 1
        ):
            raise ValueError("PAGE_IDENTITY_MISMATCH")
        with fitz.open(source_paths[p["paper_id"]]) as doc:
            page = doc[p["page_index"]]
            rect = page.rect * page.derotation_matrix
            from .geometry import q

            if (p["width_pt"], p["height_pt"], p["rotation"], p["mediabox"], p["cropbox"]) != (
                q(rect.width),
                q(rect.height),
                page.rotation,
                native(page.mediabox),
                native(page.cropbox),
            ):
                raise ValueError("PAGE_GEOMETRY_MISMATCH")
        pages[key] = p
    components = {}
    for c in bundle["components"]:
        key = (c["paper_id"], c["page_index"])
        if key not in pages or c["pdf_sha256"] != papers[c["paper_id"]]["pdf_sha256"]:
            raise ValueError("CROSS_PAGE_SOURCE")
        ident = c["identity"]
        if (
            c["component_id"] in components
            or ident["page_index"] != c["page_index"]
            or ident["bbox"] != c["bbox"]
        ):
            raise ValueError("COMPONENT_IDENTITY_MISMATCH")
        if digest(c["raw"]) != ident["raw_digest"] or c["component_id"] != "src:" + digest(
            {"pdf_sha256": c["pdf_sha256"], **ident}
        ):
            raise ValueError("TAMPERED_SOURCE_COMPONENT")
        p = pages[key]
        valid(c["bbox"], p["width_pt"], p["height_pt"])
        raw_box = c["raw"]["rect"] if ident["kind"] == "VECTOR_OBJECT" else c["raw"]["bbox"]
        if c["bbox"] != bbox(raw_box, p["width_pt"], p["height_pt"]):
            raise ValueError("SOURCE_GEOMETRY_MISMATCH")
        if ident["kind"] == "RASTER_OBJECT" and ident["references"] != {
            "xref": c["raw"].get("xref"),
            "digest": c["raw"].get("digest"),
            "placement_bbox": c["bbox"],
        }:
            raise ValueError("RASTER_PLACEMENT_REFERENCE_MISMATCH")
        if ident["kind"] == "TEXT_LINE" and ident["references"]["span_indices"] != list(
            range(len(c["raw"]["spans"]))
        ):
            raise ValueError("SPAN_REFERENCE_MISMATCH")
        if reread and canonical(
            primitive_lookup(source_paths[c["paper_id"]], c, contract)
        ) != canonical(c["raw"]):
            raise ValueError("RAW_SOURCE_ROUNDTRIP_MISMATCH")
        components[c["component_id"]] = c
    anchors = {}
    for a in bundle["anchors"]:
        c = components.get(a["source_component_id"])
        if (
            not c
            or canonical(caption_anchor(c, contract)) != canonical(a)
            or a["caption_id"] in anchors
        ):
            raise ValueError("CAPTION_SOURCE_MISMATCH")
        anchors[a["caption_id"]] = a
    seen = set()
    for u in bundle["units"]:
        if u["unit_id"] in seen:
            raise ValueError("DUPLICATE_CANONICAL_ID")
        seen.add(u["unit_id"])
        p = pages.get((u["paper_id"], u["page_index"]))
        if p is None or u["pdf_sha256"] != p["pdf_sha256"] or u["page_number"] != p["page_number"]:
            raise ValueError("INVALID_UNIT_PAGE_OR_PAPER")
        valid(u["bbox"], p["width_pt"], p["height_pt"])
        if (
            u["normalized_bbox"] != normalized(u["bbox"], p["width_pt"], p["height_pt"])
            or u["unit_type"] not in TYPES
        ):
            raise ValueError("UNIT_SCHEMA_OR_GEOMETRY_MISMATCH")
        cs = [components.get(cid) for cid in u["component_ids"]]
        if not cs or any(c is None for c in cs):
            raise ValueError("MISSING_SOURCE_COMPONENT")
        if any(
            (c["paper_id"], c["pdf_sha256"], c["page_index"])
            != (u["paper_id"], u["pdf_sha256"], u["page_index"])
            for c in cs
        ):
            raise ValueError("CROSS_PAGE_OR_PAPER_COMPONENT")
        identities = [c["identity"] for c in cs]
        if canonical(identities) != canonical(u["source_component_identity"]) or u[
            "unit_id"
        ] != unit_id(u["pdf_sha256"], u["page_index"], u["unit_type"], u["bbox"], identities):
            raise ValueError("UNIT_IDENTITY_MISMATCH")
        if (
            u["provenance"]["component_ids"] != u["component_ids"]
            or u["provenance"]["protocol_digest"] != digest(contract)
            or u["rule_version"] != contract["rule_version"]
        ):
            raise ValueError("BROKEN_PROVENANCE")
        cap = anchors.get(u["caption_id"]) if u["caption_id"] else None
        if u["caption_id"] and (
            cap is None or (cap["paper_id"], cap["page_index"]) != (u["paper_id"], u["page_index"])
        ):
            raise ValueError("CROSS_PAPER_CAPTION")
        if u["provenance"]["caption_source_block"] != (cap["source_component_id"] if cap else None):
            raise ValueError("CAPTION_PROVENANCE_MISMATCH")
        kind = u["unit_type"]
        if u["union_bbox"] != (u["bbox"] if kind == "FIGURE_CANDIDATE" else None):
            raise ValueError("UNION_BBOX_MISMATCH")
        expected_text = None
        if kind == "TEXT_BLOCK":
            if (
                len(cs) != 1
                or cs[0]["identity"]["kind"] != "TEXT_BLOCK"
                or u["bbox"] != cs[0]["bbox"]
            ):
                raise ValueError("TEXT_SOURCE_MISMATCH")
            expected_text = raw_text(cs[0]["raw"])
        elif kind == "FIGURE_CANDIDATE":
            visuals = [c for c in cs if c["identity"]["kind"] in ("RASTER_OBJECT", "VECTOR_OBJECT")]
            if (
                not visuals
                or cap is None
                or cap["caption_type"] != "FIGURE"
                or cap["source_component_id"] not in u["component_ids"]
                or u["bbox"] != union([v["bbox"] for v in visuals])
            ):
                raise ValueError("FIGURE_SOURCE_MISMATCH")
            rule = contract["figures"]
            all_visuals = [
                v
                for v in components.values()
                if v["paper_id"] == u["paper_id"]
                and v["page_index"] == u["page_index"]
                and v["identity"]["kind"] in ("RASTER_OBJECT", "VECTOR_OBJECT")
            ]
            expected = [
                v
                for v in all_visuals
                if float((v["bbox"][2] - v["bbox"][0]) * (v["bbox"][3] - v["bbox"][1]))
                >= float(rule["area_min"]) * float(p["width_pt"] * p["height_pt"])
                and overlap(v["bbox"], cap["bbox"]) >= float(rule["overlap_min"])
                and gap(v["bbox"], cap["bbox"])
                <= float(rule["gap_max_height"]) * float(p["height_pt"])
            ]
            expected.sort(
                key=lambda v: (
                    gap(v["bbox"], cap["bbox"]) / float(p["height_pt"])
                    + 1
                    - overlap(v["bbox"], cap["bbox"]),
                    v["bbox"][3] > cap["bbox"][1],
                    v["component_id"],
                )
            )
            if u["component_ids"] != [v["component_id"] for v in expected] + [
                cap["source_component_id"]
            ]:
                raise ValueError("FIGURE_ASSOCIATION_RULE_MISMATCH")
        elif kind == "TABLE_CANDIDATE":
            if (
                cs[0]["identity"]["kind"] != "TABLE_PRIMITIVE"
                or u["bbox"] != cs[0]["bbox"]
                or cs[0]["raw"]["finder_identity"]["config_digest"]
                != digest(contract["calls"]["tables"])
            ):
                raise ValueError("TABLE_PRIMITIVE_MISMATCH")
            rule = contract["tables"]
            eligible = [
                a
                for a in anchors.values()
                if a["paper_id"] == u["paper_id"]
                and a["page_index"] == u["page_index"]
                and a["caption_type"] == "TABLE"
                and overlap(a["bbox"], u["bbox"]) >= float(rule["caption_overlap_min"])
                and gap(a["bbox"], u["bbox"])
                <= float(rule["caption_gap_max_height"]) * float(p["height_pt"])
            ]
            eligible.sort(
                key=lambda a: (
                    gap(a["bbox"], u["bbox"]),
                    a["bbox"][3] > u["bbox"][1],
                    a["caption_id"],
                )
            )
            expected_cap = eligible[0] if eligible else None
            if cap != expected_cap or u["component_ids"] != [cs[0]["component_id"]] + (
                [cap["source_component_id"]] if cap else []
            ):
                raise ValueError("TABLE_CAPTION_RULE_MISMATCH")
            expected_text = "\n".join(
                "\t".join(v or "" for v in row) for row in cs[0]["raw"]["text"]
            )
        else:
            if (
                len(cs) != 1
                or cs[0]["identity"]["kind"] != "TEXT_LINE"
                or u["bbox"] != cs[0]["bbox"]
            ):
                raise ValueError("EQUATION_SOURCE_MISMATCH")
            line = cs[0]["raw"]
            expected_text = "".join(s["text"] for s in line["spans"])
            parent = cs[0]["identity"]["references"]["block_component_id"]
            if (
                parent not in components
                or u["provenance"]["rule_branch"] not in ("A", "B")
                or equation_branch(
                    line,
                    p["width_pt"],
                    any(a["source_component_id"] == parent for a in anchors.values()),
                    contract,
                )
                != u["provenance"]["rule_branch"]
            ):
                raise ValueError("EQUATION_BRANCH_MISMATCH")
        if u["raw_text"] != expected_text or u["canonical_text"] != (
            canonical_text(expected_text) if expected_text is not None else None
        ):
            raise ValueError("TEXT_ROUNDTRIP_MISMATCH")
    for p in pages.values():
        local = [
            u
            for u in bundle["units"]
            if (u["paper_id"], u["page_index"]) == (p["paper_id"], p["page_index"])
        ]
        if set(p["text_block_ids"] + p["candidate_ids"]) != {u["unit_id"] for u in local}:
            raise ValueError("DANGLING_PAGE_REFS")
        expected_raster = {
            c["component_id"]
            for c in components.values()
            if c["paper_id"] == p["paper_id"]
            and c["page_index"] == p["page_index"]
            and c["identity"]["kind"] == "RASTER_OBJECT"
        }
        expected_vector = {
            c["component_id"]
            for c in components.values()
            if c["paper_id"] == p["paper_id"]
            and c["page_index"] == p["page_index"]
            and c["identity"]["kind"] == "VECTOR_OBJECT"
        }
        if (
            set(p["raster_object_ids"]) != expected_raster
            or set(p["vector_object_ids"]) != expected_vector
            or set(p["caption_anchor_ids"])
            != {
                a["caption_id"]
                for a in anchors.values()
                if a["paper_id"] == p["paper_id"] and a["page_index"] == p["page_index"]
            }
        ):
            raise ValueError("DANGLING_PRIMITIVE_REFS")
        local_components = [
            c
            for c in components.values()
            if c["paper_id"] == p["paper_id"] and c["page_index"] == p["page_index"]
        ]
        chars = sum(
            not ch.isspace()
            for c in local_components
            if c["identity"]["kind"] == "TEXT_BLOCK"
            for ch in raw_text(c["raw"])
        )
        largest = max(
            [
                float((c["bbox"][2] - c["bbox"][0]) * (c["bbox"][3] - c["bbox"][1]))
                / float(p["width_pt"] * p["height_pt"])
                for c in local_components
                if c["identity"]["kind"] == "RASTER_OBJECT"
            ]
            or [0]
        )
        dominant = chars < contract["image_dominant"]["visible_text_chars_lt"] and largest >= float(
            contract["image_dominant"]["largest_raster_area_min"]
        )
        if p["image_dominant"] != dominant or p["OCR_REQUIRED"] != dominant or p["ocr_executed"]:
            raise ValueError("IMAGE_DOMINANT_OR_OCR_POLICY_MISMATCH")
    denominators = {
        "page_identity_exact": len(pages),
        "bbox_validity": len(components) + len(seen),
        "source_component_exact": len(components),
        "provenance_roundtrip_exact": len(seen),
        "text_roundtrip_exact": sum(u["raw_text"] is not None for u in bundle["units"])
        + len(anchors),
    }
    return {k: 1.0 if n else "NOT_ASSESSED" for k, n in denominators.items()}


def mutated(bundle, change):
    result = copy.deepcopy(bundle)
    change(result)
    return result
