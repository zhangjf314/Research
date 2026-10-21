"""Selection, bridge and coverage contracts. Never create formal cases here."""

import re
import unicodedata

from .contract import SEED, TYPES
from .identity import digest


def select_synthetic(units):
    if any(not u["paper_id"].startswith("synthetic:") for u in units):
        raise ValueError("FORMAL_MECHANISM_CASES_NOT_AUTHORIZED")
    result = {}
    for kind in TYPES:
        ranked = sorted(
            [u for u in units if u["unit_type"] == kind],
            key=lambda u: (digest({"seed": SEED, "unit_id": u["unit_id"]}), u["unit_id"]),
        )
        selected = []
        used = set()
        counts = {}
        limit = 1
        while len(selected) < min(10, len(ranked)):
            for u in ranked:
                if u["unit_id"] not in used and counts.get(u["paper_id"], 0) < limit:
                    selected.append(u)
                    used.add(u["unit_id"])
                    counts[u["paper_id"]] = counts.get(u["paper_id"], 0) + 1
                    if len(selected) == 10:
                        break
            limit += 1
        result[kind] = [u["unit_id"] for u in selected]
    return result


def bridge_normalize(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()


def bridge_synthetic(record, units):
    if not record["paper_id"].startswith("synthetic:"):
        raise ValueError("FORMAL_BRIDGE_NOT_AUTHORIZED")
    local = [
        u
        for u in units
        if u["paper_id"] == record["paper_id"]
        and u["page_index"] == record["page_index"]
        and u["unit_type"] == "TEXT_BLOCK"
    ]
    stages = [
        lambda u: u["unit_id"] == record.get("unit_id"),
        lambda u: u["canonical_text"] == record["canonical_text"],
        lambda u: (
            bridge_normalize(u["canonical_text"]) == bridge_normalize(record["canonical_text"])
        ),
    ]
    for index, predicate in enumerate(stages):
        matched = [u["unit_id"] for u in local if predicate(u)]
        if matched:
            return {
                "status": "LINKED" if len(matched) == 1 else "AMBIGUOUS",
                "priority": index + 1,
                "candidates": matched,
            }
    return {"status": "UNLINKED", "priority": None, "candidates": []}
