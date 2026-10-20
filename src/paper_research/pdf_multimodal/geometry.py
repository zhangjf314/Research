"""Unrotated, crop-relative PyMuPDF geometry."""

from decimal import ROUND_HALF_EVEN, Decimal


def q(value, precision="0.001"):
    d = Decimal(str(value))
    if not d.is_finite():
        raise ValueError("NONFINITE_BBOX")
    result = d.quantize(Decimal(precision), rounding=ROUND_HALF_EVEN)
    return abs(result) if result == 0 else result


def valid(box, width, height, tolerance="0.000001"):
    b = [Decimal(str(v)) for v in box]
    w, h, t = Decimal(str(width)), Decimal(str(height)), Decimal(tolerance)
    if len(b) != 4 or not all(v.is_finite() for v in b) or w <= 0 or h <= 0:
        raise ValueError("INVALID_BBOX")
    if b[0] > b[2] or b[1] > b[3] or min(b) < -t or b[2] > w + t or b[3] > h + t:
        raise ValueError("OUT_OF_PAGE_BBOX")


def bbox(box, width, height):
    valid(box, width, height)
    # Only the preregistered 1e-6 raw rounding tolerance is clipped.
    return [
        q(max(0, min(float(v), float(width if i % 2 == 0 else height)))) for i, v in enumerate(box)
    ]


def normalized(box, width, height):
    return [
        q(Decimal(str(v)) / Decimal(str(width if i % 2 == 0 else height)), "0.000001")
        for i, v in enumerate(box)
    ]


def union(boxes):
    return [
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    ]


def overlap(a, b):
    denominator = min(float(a[2] - a[0]), float(b[2] - b[0]))
    return max(0, float(min(a[2], b[2]) - max(a[0], b[0]))) / denominator if denominator > 0 else 0


def gap(a, b):
    return float(max(0, max(a[1], b[1]) - min(a[3], b[3])))
