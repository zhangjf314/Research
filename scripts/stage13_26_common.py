"""Shared helpers for Stage 13.26 offline audits."""

from __future__ import annotations

import importlib
from typing import Any

try:
    common25 = importlib.import_module("scripts.stage13_25_common")
except (ImportError, ModuleNotFoundError):
    common25 = importlib.import_module("stage13_25_common")

DATA = common25.DATA
DOCS = common25.DOCS
ROOT = common25.ROOT
RUN_ROOT = common25.RUN_ROOT
canonical_hash = common25.canonical_hash
citation_keys = common25.citation_keys
file_hash = common25.file_hash
iter_claim_contexts = common25.iter_claim_contexts
read_json = common25.read_json
write_csv = common25.write_csv
write_json = common25.write_json
write_jsonl = common25.write_jsonl


def relation_key_from_candidate(candidate: Any) -> str:
    return f"{candidate.paper_id}|{candidate.page}|{candidate.block_id}"


def safe_fraction(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0
