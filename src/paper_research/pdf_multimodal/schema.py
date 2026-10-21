"""Frozen JSON schemas; cross-record constraints enforced in provenance validation."""

TYPES = ["TEXT_BLOCK", "FIGURE_CANDIDATE", "TABLE_CANDIDATE", "EQUATION_CANDIDATE"]


def schemas():
    string = {"type": "string"}
    integer = {"type": "integer", "minimum": 0}
    box = {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4}
    ids = {"type": "array", "items": string, "uniqueItems": True}
    common = {
        "paper_id": string,
        "pdf_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "page_index": integer,
        "page_number": {"type": "integer", "minimum": 1},
    }
    page = {
        **common,
        "width_pt": {"type": "number", "exclusiveMinimum": 0},
        "height_pt": {"type": "number", "exclusiveMinimum": 0},
        "rotation": {"enum": [0, 90, 180, 270]},
        "mediabox": box,
        "cropbox": box,
        **{
            k: ids
            for k in [
                "text_block_ids",
                "raster_object_ids",
                "vector_object_ids",
                "caption_anchor_ids",
                "candidate_ids",
            ]
        },
        "image_dominant": {"type": "boolean"},
        "OCR_REQUIRED": {"type": "boolean"},
        "ocr_executed": {"const": False},
    }
    prov = {
        "component_ids": ids,
        "caption_source_block": {"type": ["string", "null"]},
        "rule_branch": {"enum": ["A", "B", None]},
        "protocol_digest": string,
    }
    unit = {
        **common,
        "unit_id": {"type": "string", "pattern": "^mmu:[0-9a-f]{64}$"},
        "unit_type": {"enum": TYPES},
        "bbox": box,
        "union_bbox": {"anyOf": [box, {"type": "null"}]},
        "normalized_bbox": box,
        "raw_text": {"type": ["string", "null"]},
        "canonical_text": {"type": ["string", "null"]},
        "caption_id": {"type": ["string", "null"]},
        "component_ids": ids,
        "source_component_identity": {"type": "array", "minItems": 1, "items": {"type": "object"}},
        "extraction_method": string,
        "rule_version": string,
        "provenance": {
            "type": "object",
            "required": list(prov),
            "properties": prov,
            "additionalProperties": False,
        },
    }
    result = {}
    for name, props in [("page_record", page), ("evidence_unit", unit), ("provenance", prov)]:
        result[name] = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "required": list(props),
            "properties": props,
            "additionalProperties": False,
        }
    return result
