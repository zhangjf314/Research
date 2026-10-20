"""Explicit parser calls and preregistered scientific rules."""

import inspect

import pymupdf as fitz
from pymupdf import table

from .identity import digest, native
from .schema import schemas

VERSION = "1.28.0"
TYPES = ["TEXT_BLOCK", "FIGURE_CANDIDATE", "TABLE_CANDIDATE", "EQUATION_CANDIDATE"]
CAPTION = (
    r"^\s*(?P<prefix>Figure|Fig\.|Table)\s+(?P<number>[A-Z]\d+|\d+[a-z]?|[IVXLCDM]+)(?=$|[\s.:])"
)
MATH = "=+−-×*/<>≤≥≈≠∈∑∏∫√∞±∂∇"
SEED = "PHASE4A_MULTIMODAL_EVIDENCE_UNIT_V1_20261005"


def parser_guard():
    if fitz.VersionBind != VERSION:
        raise ValueError("PARSER_VERSION_MISMATCH")


def make_contract():
    parser_guard()
    # Every public kwarg including the underlying wrapped table API is frozen.
    table_kwargs = {
        n: native(p.default)
        for n, p in inspect.signature(table.find_tables).parameters.items()
        if n != "page"
    }
    table_kwargs.update(strategy="lines", vertical_strategy="lines", horizontal_strategy="lines")
    for prefix in ("snap", "join", "intersection"):
        table_kwargs[prefix + "_x_tolerance"] = 3
        table_kwargs[prefix + "_y_tolerance"] = 3
    return {
        "schemas": schemas(),
        "parser": {"name": "PyMuPDF", "version": VERSION},
        "calls": {
            "text": {
                "method": "Page.get_text",
                "kwargs": {
                    "option": "dict",
                    "clip": None,
                    "flags": int(fitz.TEXTFLAGS_DICT & ~fitz.TEXT_PRESERVE_IMAGES),
                    "textpage": None,
                    "sort": False,
                    "delimiters": None,
                    "tolerance": 3,
                },
            },
            "images": {"method": "Page.get_image_info", "kwargs": {"hashes": True, "xrefs": True}},
            "vectors": {"method": "Page.get_drawings", "kwargs": {"extended": False}},
            "tables": {"method": "Page.find_tables", "kwargs": table_kwargs},
        },
        "geometry": {
            "space": "PyMuPDF unrotated crop-relative page space",
            "origin": "top-left",
            "x_axis": "right",
            "y_axis": "down",
            "units": "pt",
            "rotation": (
                "native extraction unrotated; dimensions Page.rect * Page.derotation_matrix"
            ),
            "cropbox": (
                "bbox relative to crop origin; mediabox/cropbox stored raw in their native frame"
            ),
            "point_precision": "0.001",
            "normalized_precision": "0.000001",
            "rounding": "ROUND_HALF_EVEN",
            "raw_tolerance_pt": "0.000001",
            "canonical_bounds": "compare against quantized width/height after raw validation",
        },
        "text": {
            "unicode_normalization": "NONE",
            "line_endings": "CRLF/CR to LF in canonical_text only",
            "raw": "unaltered span text joined in native line order, LF after each line",
            "reading_order": "source parser block/line/span order; sort=false",
        },
        "captions": {
            "regex": CAPTION,
            "flags": ["IGNORECASE"],
            "scope": "start of block",
            "label": "CAPTION_ANCHOR",
            "numbers": ["decimal", "decimal suffix", "A1", "Roman"],
        },
        "figures": {
            "area_min": 0.005,
            "overlap_min": 0.25,
            "gap_max_height": 0.25,
            "overlap_denominator": "min(visual width, caption width)",
            "vertical_sides": ["above", "below", "overlap"],
            "ranking": [
                "gap/height + 1-overlap",
                "above preference on equal score",
                "component_id",
            ],
            "binding": "all passing same-page objects; shared components allowed",
            "union_bbox": "visual components only",
            "vector_clustering": "NO_VECTOR_CLUSTERING_V1",
        },
        "tables": {
            "caption_overlap_min": 0.25,
            "caption_gap_max_height": 0.20,
            "caption_ranking": ["vertical gap", "caption above preference", "caption_id"],
            "captionless": "retain with caption_id null",
            "primitive": "TableFinder table/cells/rows/header/extract",
            "geometry_snapshot": "capture page boxes/rotation before find_tables; PDF never saved",
            "rotated_table_adapter": {
                "pre_call": "Page.set_rotation(0) on unsaved document",
                "post_call": (
                    "restore original Page.set_cropbox(rect) and "
                    "Page.set_rotation(rotation) in finally"
                ),
                "reason": (
                    "avoid 1.28.0 internal rotated-content transform and cropbox reset; "
                    "detector receives native unrotated coordinate space"
                ),
            },
        },
        "equations": {
            "operators": MATH,
            "greek_ranges": [[880, 1023], [7936, 8191]],
            "greek_policy": "Unicode letter in these ranges counts, operator still required",
            "anchor_regex": r"\(\d+[a-z]?\)\s*$",
            "anchor_flags": ["IGNORECASE"],
            "A": "at least one operator AND terminal number anchor",
            "B": {
                "max_codepoints": 180,
                "min_math_chars": 2,
                "min_ratio": 0.12,
                "max_width_ratio": 0.90,
                "exclude_caption": True,
                "operator_required": True,
            },
            "number_right_side_gate": False,
            "logical_line": "one parser-native line; no geometric joining",
        },
        "image_dominant": {
            "visible_text_chars_lt": 80,
            "largest_raster_area_min": 0.60,
            "visible": "non-whitespace codepoints",
            "ocr_required": True,
            "ocr_execute": False,
        },
        "ids": {
            "prefix": "mmu:",
            "algorithm": "SHA256",
            "payload": [
                "pdf_sha256",
                "page_index",
                "unit_type",
                "canonical_bbox",
                "source_component_identity",
            ],
        },
        "serialization": {
            "encoding": "UTF-8",
            "ensure_ascii": False,
            "sort_keys": True,
            "separators": [",", ":"],
            "newline": "LF",
            "decimal_numbers": "fixed precision JSON numeric tokens",
            "forbidden": ["timestamp", "absolute machine path", "temp path", "random UUID"],
        },
        "benchmark": {
            "seed": SEED,
            "target_per_type": 10,
            "rank": "SHA256(canonical JSON {seed,unit_id}) then unit_id",
            "diversity": "successive passes max 1,2,3,... per paper until 10 or exhausted",
            "shortfall": "take all; no substitution across types",
            "cases_now": "NOT_CREATED",
        },
        "metrics": {
            "page_identity_exact": 1.0,
            "bbox_validity": 1.0,
            "source_component_exact": 1.0,
            "provenance_roundtrip_exact": 1.0,
            "text_roundtrip_exact": 1.0,
            "scope": "applicable cases only; zero denominator NOT_ASSESSED; not detector precision",
        },
        "bridge": {
            "priority": [
                "exact identity",
                "exact canonical text + paper/page",
                "unique normalized text + paper/page",
            ],
            "normalize": ["NFC", "Unicode whitespace collapse to ASCII space", "strip"],
            "ambiguity": {"0": "UNLINKED", "1": "LINKED", ">1": "AMBIGUOUS"},
            "ambiguous_at_any_priority": "stop without bbox tie-break",
            "execution": "NOT_RUN",
        },
        "eligibility": {"units_min": 24, "distinct_papers_min": 8, "scope": "coverage only"},
        "integrity": {
            k: 0
            for k in [
                "dangling_refs",
                "unknown_paper_refs",
                "invalid_pages",
                "out_of_page_bbox",
                "duplicate_canonical_ids",
                "broken_provenance",
            ]
        },
        "reproducibility": {
            "order": [
                "paper_id lexical",
                "page_index ascending",
                "unit_type fixed TYPES order",
                "unit_id lexical",
            ],
            "two_clean_builds": [
                "byte-identical JSONL",
                "byte-identical manifest",
                "same scientific digest",
            ],
        },
        "formal_extraction": "FORMAL_EXTRACTION_NOT_AUTHORIZED",
        "rule_version": "MULTIMODAL_EXTRACTION_PROTOCOL_V1",
    }


def contract_guard(contract):
    parser_guard()
    if digest(contract) != digest(make_contract()):
        raise ValueError("PARSER_OR_PROTOCOL_CONFIG_MISMATCH")
