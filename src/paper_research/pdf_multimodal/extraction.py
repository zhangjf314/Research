"""Native primitives and candidate construction, accessible only to fixture CLI."""

import json
import re
import unicodedata
from pathlib import Path

import pymupdf as fitz

from .contract import contract_guard
from .geometry import bbox, gap, normalized, overlap, q, union
from .identity import digest, file_sha, native, unit_id


def raw_text(block):
    return "".join("".join(s["text"] for s in line["spans"]) + "\n" for line in block["lines"])


def canonical_text(text):
    return text.replace("\r\n", "\n").replace("\r", "\n")


def find_tables_unrotated(page, kwargs):
    """Avoid 1.28.0 internal page content derotation; only mutate unsaved page metadata."""
    rotation = page.rotation
    cropbox = fitz.Rect(page.cropbox)
    try:
        page.set_rotation(0)
        return page.find_tables(**kwargs)
    finally:
        page.set_cropbox(cropbox)
        page.set_rotation(rotation)


def source(kind, raw, box, paper, sha, page, refs=None):
    identity = {
        "kind": kind,
        "page_index": page,
        "raw_digest": digest(raw),
        "bbox": box,
        "references": refs or {},
    }
    return {
        "component_id": "src:" + digest({"pdf_sha256": sha, **identity}),
        "identity": identity,
        "raw": native(raw),
        "paper_id": paper,
        "pdf_sha256": sha,
        "page_index": page,
        "bbox": box,
    }


def unit(
    kind,
    components,
    box,
    paper,
    sha,
    page,
    width,
    height,
    contract,
    caption=None,
    text=None,
    branch=None,
):
    identities = [c["identity"] for c in components]
    uid = unit_id(sha, page, kind, box, identities)
    return {
        "unit_id": uid,
        "unit_type": kind,
        "paper_id": paper,
        "pdf_sha256": sha,
        "page_index": page,
        "page_number": page + 1,
        "bbox": box,
        "normalized_bbox": normalized(box, width, height),
        "union_bbox": box if kind == "FIGURE_CANDIDATE" else None,
        "raw_text": text,
        "canonical_text": canonical_text(text) if text is not None else None,
        "caption_id": caption["caption_id"] if caption else None,
        "component_ids": [c["component_id"] for c in components],
        "source_component_identity": identities,
        "extraction_method": {
            "TEXT_BLOCK": "Page.get_text",
            "FIGURE_CANDIDATE": "caption+native visuals",
            "TABLE_CANDIDATE": "Page.find_tables",
            "EQUATION_CANDIDATE": "native line rules",
        }[kind],
        "rule_version": contract["rule_version"],
        "provenance": {
            "component_ids": [c["component_id"] for c in components],
            "caption_source_block": caption["source_component_id"] if caption else None,
            "rule_branch": branch,
            "protocol_digest": digest(contract),
        },
    }


def caption_anchor(component, contract):
    text = raw_text(component["raw"])
    match = re.match(contract["captions"]["regex"], text, re.IGNORECASE)
    if not match:
        return None
    anchor = {
        "source_component_id": component["component_id"],
        "paper_id": component["paper_id"],
        "pdf_sha256": component["pdf_sha256"],
        "page_index": component["page_index"],
        "bbox": component["bbox"],
        "exact_block_text": text,
        "matched_prefix": match.group("prefix"),
        "number_literal": match.group("number"),
        "caption_type": "TABLE" if match.group("prefix").lower() == "table" else "FIGURE",
        "label": "CAPTION_ANCHOR",
    }
    anchor["caption_id"] = "cap:" + digest(anchor)
    return anchor


def equation_branch(line, width, is_caption, contract):
    text = "".join(s["text"] for s in line["spans"])
    rule = contract["equations"]
    operators = sum(c in rule["operators"] for c in text)
    greek = sum(
        unicodedata.category(c).startswith("L")
        and any(a <= ord(c) <= b for a, b in rule["greek_ranges"])
        for c in text
    )
    if not operators or is_caption:
        return None
    numbered = re.search(rule["anchor_regex"], text, re.IGNORECASE)
    if numbered:
        return "A"
    b = rule["B"]
    if (
        0 < len(text) <= b["max_codepoints"]
        and operators + greek >= b["min_math_chars"]
        and (operators + greek) / len(text) >= b["min_ratio"]
        and line["bbox"][2] - line["bbox"][0] <= float(b["max_width_ratio"]) * float(width)
    ):
        return "B"
    return None


def extract_fixture(path, paper, expected_sha, contract, allowed_fixture_shas):
    contract_guard(contract)
    actual = file_sha(path)
    if actual != expected_sha:
        raise ValueError("PDF_SHA_MISMATCH")
    root = Path(__file__).resolve().parents[3]
    corpus = (
        root / "artifacts/paperresearch-v2/phase4a-r1-source-recovery/final_corpus/papers.jsonl"
    )
    formal_shas = {
        json.loads(line)["expected_sha256"]
        for line in corpus.read_text(encoding="utf-8").splitlines()
    }
    if actual in formal_shas:
        raise ValueError("FORMAL_EXTRACTION_NOT_AUTHORIZED")
    if actual not in allowed_fixture_shas or not paper.startswith("synthetic:"):
        raise ValueError("FORMAL_EXTRACTION_NOT_AUTHORIZED")
    pages, components, anchors, units = [], [], [], []
    with fitz.open(path) as doc:
        for pi, page in enumerate(doc):
            rect = page.rect * page.derotation_matrix
            # 1.28.0 find_tables may reset the in-memory cropbox on rotated pages.
            # Preserve page geometry before detector calls; PDF bytes are never saved.
            original_mediabox = native(page.mediabox)
            original_cropbox = native(page.cropbox)
            original_rotation = page.rotation
            width, height = q(rect.width), q(rect.height)
            calls = contract["calls"]
            text = page.get_text(**calls["text"]["kwargs"])
            images = page.get_image_info(**calls["images"]["kwargs"])
            drawings = page.get_drawings(**calls["vectors"]["kwargs"])
            finder = find_tables_unrotated(page, calls["tables"]["kwargs"])
            local, caps, local_units = [], [], []
            for bi, block in enumerate(text["blocks"]):
                if block["type"] != 0:
                    continue
                box = bbox(block["bbox"], rect.width, rect.height)
                c = source("TEXT_BLOCK", block, box, paper, actual, pi, {"block_index": bi})
                local.append(c)
                local_units.append(
                    unit(
                        "TEXT_BLOCK",
                        [c],
                        box,
                        paper,
                        actual,
                        pi,
                        width,
                        height,
                        contract,
                        text=raw_text(block),
                    )
                )
                cap = caption_anchor(c, contract)
                if cap:
                    caps.append(cap)
                for li, line in enumerate(block["lines"]):
                    branch = equation_branch(line, width, cap is not None, contract)
                    if branch:
                        lb = bbox(line["bbox"], rect.width, rect.height)
                        lc = source(
                            "TEXT_LINE",
                            line,
                            lb,
                            paper,
                            actual,
                            pi,
                            {
                                "block_component_id": c["component_id"],
                                "line_index": li,
                                "span_indices": list(range(len(line["spans"]))),
                            },
                        )
                        local.append(lc)
                        local_units.append(
                            unit(
                                "EQUATION_CANDIDATE",
                                [lc],
                                lb,
                                paper,
                                actual,
                                pi,
                                width,
                                height,
                                contract,
                                text="".join(s["text"] for s in line["spans"]),
                                branch=branch,
                            )
                        )
            visuals = []
            for raw in images:
                box = bbox(raw["bbox"], rect.width, rect.height)
                c = source(
                    "RASTER_OBJECT",
                    raw,
                    box,
                    paper,
                    actual,
                    pi,
                    {
                        "xref": raw.get("xref"),
                        "digest": native(raw.get("digest")),
                        "placement_bbox": box,
                    },
                )
                visuals.append(c)
            for di, raw in enumerate(drawings):
                box = bbox(raw["rect"], rect.width, rect.height)
                c = source("VECTOR_OBJECT", raw, box, paper, actual, pi, {"drawing_index": di})
                visuals.append(c)
            local.extend(visuals)
            for cap in caps:
                if cap["caption_type"] != "FIGURE":
                    continue
                rule = contract["figures"]
                passing = [
                    v
                    for v in visuals
                    if float((v["bbox"][2] - v["bbox"][0]) * (v["bbox"][3] - v["bbox"][1]))
                    >= rule["area_min"] * float(width * height)
                    and overlap(v["bbox"], cap["bbox"]) >= rule["overlap_min"]
                    and gap(v["bbox"], cap["bbox"]) <= rule["gap_max_height"] * float(height)
                ]
                passing.sort(
                    key=lambda v: (
                        gap(v["bbox"], cap["bbox"]) / float(height)
                        + 1
                        - overlap(v["bbox"], cap["bbox"]),
                        v["bbox"][3] > cap["bbox"][1],
                        v["component_id"],
                    )
                )
                if passing:
                    # Caption block is a real source component, separate from visual union geometry.
                    block = next(
                        c for c in local if c["component_id"] == cap["source_component_id"]
                    )
                    local_units.append(
                        unit(
                            "FIGURE_CANDIDATE",
                            passing + [block],
                            union([v["bbox"] for v in passing]),
                            paper,
                            actual,
                            pi,
                            width,
                            height,
                            contract,
                            caption=cap,
                        )
                    )
            for ti, table in enumerate(finder.tables):
                primitive = {
                    "bbox": native(table.bbox),
                    "cells": native(table.cells),
                    "rows": [native(row.cells) for row in table.rows],
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
                        "table_index": ti,
                        "config_digest": digest(calls["tables"]),
                    },
                }
                box = bbox(table.bbox, rect.width, rect.height)
                tc = source(
                    "TABLE_PRIMITIVE",
                    primitive,
                    box,
                    paper,
                    actual,
                    pi,
                    {"table_index": ti, "config_digest": digest(calls["tables"])},
                )
                local.append(tc)
                rule = contract["tables"]
                eligible = [
                    a
                    for a in caps
                    if a["caption_type"] == "TABLE"
                    and overlap(a["bbox"], box) >= rule["caption_overlap_min"]
                    and gap(a["bbox"], box) <= rule["caption_gap_max_height"] * float(height)
                ]
                eligible.sort(
                    key=lambda a: (gap(a["bbox"], box), a["bbox"][3] > box[1], a["caption_id"])
                )
                cap = eligible[0] if eligible else None
                cs = [tc] + (
                    [next(c for c in local if c["component_id"] == cap["source_component_id"])]
                    if cap
                    else []
                )
                local_units.append(
                    unit(
                        "TABLE_CANDIDATE",
                        cs,
                        box,
                        paper,
                        actual,
                        pi,
                        width,
                        height,
                        contract,
                        caption=cap,
                        text="\n".join(
                            "\t".join(v or "" for v in row) for row in primitive["text"]
                        ),
                    )
                )
            chars = sum(
                not ch.isspace() for b in text["blocks"] if b["type"] == 0 for ch in raw_text(b)
            )
            largest = max(
                [
                    float((v["bbox"][2] - v["bbox"][0]) * (v["bbox"][3] - v["bbox"][1]))
                    / float(width * height)
                    for v in visuals
                    if v["identity"]["kind"] == "RASTER_OBJECT"
                ]
                or [0]
            )
            dominant = (
                chars < contract["image_dominant"]["visible_text_chars_lt"]
                and largest >= contract["image_dominant"]["largest_raster_area_min"]
            )
            pages.append(
                {
                    "paper_id": paper,
                    "pdf_sha256": actual,
                    "page_index": pi,
                    "page_number": pi + 1,
                    "width_pt": width,
                    "height_pt": height,
                    "rotation": original_rotation,
                    "mediabox": original_mediabox,
                    "cropbox": original_cropbox,
                    "text_block_ids": [
                        u["unit_id"] for u in local_units if u["unit_type"] == "TEXT_BLOCK"
                    ],
                    "raster_object_ids": [
                        v["component_id"]
                        for v in visuals
                        if v["identity"]["kind"] == "RASTER_OBJECT"
                    ],
                    "vector_object_ids": [
                        v["component_id"]
                        for v in visuals
                        if v["identity"]["kind"] == "VECTOR_OBJECT"
                    ],
                    "caption_anchor_ids": [a["caption_id"] for a in caps],
                    "candidate_ids": [
                        u["unit_id"] for u in local_units if u["unit_type"] != "TEXT_BLOCK"
                    ],
                    "image_dominant": dominant,
                    "OCR_REQUIRED": dominant,
                    "ocr_executed": False,
                }
            )
            components.extend(local)
            anchors.extend(caps)
            units.extend(local_units)
    return {
        "papers": {paper: {"pdf_sha256": actual, "page_count": len(pages)}},
        "pages": pages,
        "components": components,
        "anchors": anchors,
        "units": units,
        "protocol_digest": digest(contract),
    }
