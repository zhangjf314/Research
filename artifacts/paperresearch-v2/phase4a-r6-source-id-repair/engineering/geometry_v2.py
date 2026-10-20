# ruff: noqa: E501 -- Frozen protocol and diagnostic prose.
"""Raw source geometry and page-visible projection are different identities."""

import copy
from decimal import Decimal

from paper_research.pdf_multimodal import contract as v1
from paper_research.pdf_multimodal.geometry import normalized, q, valid
from paper_research.pdf_multimodal.identity import canonical, digest, native


def raw_valid(box):
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise ValueError("HARD_INVALID_SOURCE_GEOMETRY: malformed rect")
    try:
        values = [Decimal(str(v)) for v in box]
    except Exception as exc:
        raise ValueError("HARD_INVALID_SOURCE_GEOMETRY: malformed coordinate") from exc
    if not all(v.is_finite() for v in values):
        raise ValueError("HARD_INVALID_SOURCE_GEOMETRY: nonfinite")
    if values[0] > values[2] or values[1] > values[3]:
        raise ValueError("HARD_INVALID_SOURCE_GEOMETRY: inverted")
    return values


def project(box, width, height):
    values = raw_valid(box)
    w, h = Decimal(str(width)), Decimal(str(height))
    if not w.is_finite() or not h.is_finite() or w <= 0 or h <= 0:
        raise ValueError("INVALID_PAGE_BOUNDS")
    raw = [q(v) for v in values]
    result = {
        "raw_bbox_native": native(list(box)),
        "raw_bbox": raw,
        "page_bounds_native": [0, 0, native(width), native(height)],
        "visible_bbox_native": None,
        "visible_bbox": None,
        "normalized_visible_bbox": None,
        "evidence_visible": False,
    }
    if values[0] == values[2] or values[1] == values[3]:
        result["visibility_class"] = "ZERO_AREA_SOURCE"
        return result
    intersection = [
        max(values[0], Decimal(0)),
        max(values[1], Decimal(0)),
        min(values[2], w),
        min(values[3], h),
    ]
    if intersection[0] >= intersection[2] or intersection[1] >= intersection[3]:
        result["visibility_class"] = "OUTSIDE_PAGE"
        return result
    contained = values[0] >= 0 and values[1] >= 0 and values[2] <= w and values[3] <= h
    visible = [q(v) for v in intersection]
    valid(visible, q(w), q(h))
    result.update(
        visibility_class="FULLY_VISIBLE" if contained else "PARTIALLY_VISIBLE",
        visible_bbox_native=intersection,
        visible_bbox=visible,
        normalized_visible_bbox=normalized(visible, q(w), q(h)),
        evidence_visible=True,
    )
    return result


def source(kind, raw, raw_box, paper, sha, page, width, height, refs=None):
    geometry = project(list(raw_box), width, height)
    identity = {
        "pdf_sha256": sha,
        "page_index": page,
        "primitive_type": kind,
        "kind": kind,
        "raw_bbox": geometry["raw_bbox"],
        "native_primitive_digest": digest(raw),
        "raw_digest": digest(raw),
        "native_index_identity": refs or {},
        "references": refs or {},
    }
    record = {
        "identity": identity,
        "raw": native(raw),
        "paper_id": paper,
        "pdf_sha256": sha,
        "page_index": page,
        **geometry,
        "bbox": geometry["visible_bbox"],
    }
    record["component_id"] = source_id(record)
    return record


def unit_id(sha, page, kind, box, identities):
    return "mmu:" + digest(
        {
            "pdf_sha256": sha,
            "page_index": page,
            "unit_type": kind,
            "canonical_visible_bbox": box,
            "source_component_identity": identities,
        }
    )


def make_protocol():
    c = copy.deepcopy(v1.make_contract())
    c["rule_version"] = "MULTIMODAL_EXTRACTION_PROTOCOL_V2"
    c["geometry"] = {
        **c["geometry"],
        "contract": "PDF_GEOMETRY_CONTRACT_V2",
        "raw_validity": "four finite coordinates, x0<=x1,y0<=y1; no page-bounds gate",
        "projection": "Decimal(str(native value)); intersection before quantization",
        "classification_priority": [
            "HARD_INVALID_SOURCE_GEOMETRY fail",
            "ZERO_AREA_SOURCE",
            "OUTSIDE_PAGE",
            "FULLY_VISIBLE",
            "PARTIALLY_VISIBLE",
        ],
        "outside_representation": "visible_bbox=null; retained raw provenance",
        "promotion": "native positive-area page intersection only; OUTSIDE_PAGE/ZERO_AREA_SOURCE never units",
        "raw_native_preservation": "full precision raw_bbox_native and original raw primitive; canonical raw_bbox quantized separately",
        "page_bounds": "[0,0,unrotated native page width,native page height]",
        "figure_geometry": "visible_bbox for area/overlap/gap/union",
        "table_caption_geometry": "visible_bbox for table/caption association",
        "text_equation": "content and source order unchanged; frozen equation predicate reads original native line; final unit visible geometry",
        "image_area": "largest page-visible raster bbox area; original numeric threshold unchanged",
        "tiny_positive_intersection": "classify before quantization; no new rounding-size threshold",
        "source_identity": [
            "pdf_sha256",
            "page_index",
            "primitive_type",
            "raw_bbox",
            "native_primitive_digest",
            "native_index_identity",
        ],
    }
    c["ids"]["payload"] = [
        "pdf_sha256",
        "page_index",
        "unit_type",
        "canonical_visible_bbox",
        "source_component_identity",
    ]
    c["metrics"].update(raw_source_bbox_exact=1.0, visible_projection_exact=1.0)
    c["integrity"].pop("out_of_page_bbox")
    c["integrity"].update(visible_bbox_out_of_page=0, invalid_raw_geometry=0)
    c["schemas"]["evidence_unit"]["properties"]["provenance"]["properties"]["source_geometries"] = {
        "type": "array",
        "items": {"type": "object"},
    }
    c["schemas"]["evidence_unit"]["properties"]["provenance"]["required"].append(
        "source_geometries"
    )
    c["schemas"]["provenance"]["properties"]["source_geometries"] = {
        "type": "array",
        "items": {"type": "object"},
    }
    c["schemas"]["provenance"]["required"].append("source_geometries")
    c["schemas"]["source_component"] = {
        "geometry_required": [
            "raw_bbox_native",
            "raw_bbox",
            "visible_bbox_native",
            "visible_bbox",
            "visibility_class",
            "evidence_visible",
            "page_bounds_native",
        ],
        "visibility_enum": [
            "FULLY_VISIBLE",
            "PARTIALLY_VISIBLE",
            "OUTSIDE_PAGE",
            "ZERO_AREA_SOURCE",
        ],
        "native_raw": "unaltered original parser primitive",
        "identity": c["geometry"]["source_identity"],
    }
    for name in ["width_native", "height_native"]:
        c["schemas"]["page_record"]["properties"][name] = {"type": "number", "exclusiveMinimum": 0}
        c["schemas"]["page_record"]["required"].append(name)
    c["schemas"]["page_record"]["properties"]["page_bounds_native"] = {
        "type": "array",
        "items": {"type": "number"},
        "minItems": 4,
        "maxItems": 4,
    }
    c["schemas"]["page_record"]["required"].append("page_bounds_native")
    return c


def guard(c):
    v1.parser_guard()
    if canonical(c) != canonical(make_protocol()):
        raise ValueError("V2_PROTOCOL_CONFIG_MISMATCH")


def protocol_digest(c):
    return digest({"contract": c, "schemas": c["schemas"]})


def independent_projection(box, width, height):
    """Oracle implementation, no call to production project()."""
    x0, y0, x1, y1 = [Decimal(str(v)) for v in box]
    w, h = Decimal(str(width)), Decimal(str(height))
    if x0 == x1 or y0 == y1:
        return "ZERO_AREA_SOURCE", None
    left = x0 if x0 > 0 else Decimal(0)
    top = y0 if y0 > 0 else Decimal(0)
    right = x1 if x1 < w else w
    bottom = y1 if y1 < h else h
    if right <= left or bottom <= top:
        return "OUTSIDE_PAGE", None
    classification = (
        "FULLY_VISIBLE" if (0 <= x0 <= x1 <= w and 0 <= y0 <= y1 <= h) else "PARTIALLY_VISIBLE"
    )
    return classification, [q(v) for v in [left, top, right, bottom]]


SOURCE_ID_FIELDS = (
    "pdf_sha256",
    "page_index",
    "primitive_type",
    "raw_bbox",
    "native_primitive_digest",
    "native_index_identity",
)


def validate_identity_record(record):
    identity = record["identity"]
    for field in SOURCE_ID_FIELDS:
        if field not in identity or identity[field] is None:
            raise ValueError("SOURCE_ID_REQUIRED_FIELD: " + field)
    for alias, field in (
        ("kind", "primitive_type"),
        ("raw_digest", "native_primitive_digest"),
        ("references", "native_index_identity"),
    ):
        if alias in identity and canonical(identity[alias]) != canonical(identity[field]):
            raise ValueError("SOURCE_ID_ALIAS_INCONSISTENCY: " + alias)


def source_id_payload(record):
    validate_identity_record(record)
    identity = record["identity"]
    return {
        "pdf_sha256": identity["pdf_sha256"],
        "page_index": identity["page_index"],
        "primitive_type": identity["primitive_type"],
        "raw_bbox": identity["raw_bbox"],
        "native_primitive_digest": identity["native_primitive_digest"],
        "native_index_identity": identity["native_index_identity"],
    }


def source_id(record):
    return "src:" + digest(source_id_payload(record))
