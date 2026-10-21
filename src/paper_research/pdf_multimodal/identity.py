"""Canonical serialization with fixed numeric precision, not float repr."""

import hashlib
import json
import math
from decimal import Decimal

import pymupdf as fitz


def native(value):
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if isinstance(value, (fitz.Rect, fitz.Point, fitz.Quad, fitz.Matrix)):
        return native(list(value))
    if isinstance(value, dict):
        return {str(k): native(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [native(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("NONFINITE_SOURCE")
    if value is None or isinstance(value, (str, int, float, bool, Decimal)):
        return value
    raise TypeError(type(value).__name__)


def canonical(value):
    value = native(value)
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("NONFINITE_DECIMAL")
        return format(value, "f")
    if isinstance(value, list):
        return "[" + ",".join(canonical(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ",".join(canonical(k) + ":" + canonical(value[k]) for k in sorted(value)) + "}"
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()


def unit_id(pdf_sha256, page_index, unit_type, bbox, source_identity):
    return "mmu:" + digest(
        {
            "pdf_sha256": pdf_sha256,
            "page_index": page_index,
            "unit_type": unit_type,
            "canonical_bbox": bbox,
            "source_component_identity": source_identity,
        }
    )
