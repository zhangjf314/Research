"""Versioned deterministic identities for the RAGQ3 executable matrix."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections.abc import Mapping
from typing import Any

IDENTITY_SCHEMA_VERSION = "ragq3-evidence-id-v1"


def normalize_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).replace("\r\n", "\n").split())


def serialize_identity(fields: Mapping[str, Any]) -> str:
    """Canonical UTF-8 JSON; identities never include filesystem paths or timestamps."""

    def normalized(value: Any) -> Any:
        if isinstance(value, str):
            return normalize_text(value)
        if isinstance(value, Mapping):
            return {key: normalized(item) for key, item in value.items()}
        if isinstance(value, list):
            return [normalized(item) for item in value]
        return value

    payload = normalized({"identity_schema_version": IDENTITY_SCHEMA_VERSION, **dict(fields)})
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_id(prefix: str, fields: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(serialize_identity(fields).encode("utf-8")).hexdigest()
    return f"{prefix}:{digest}"


def sentence_boundaries(text: str) -> list[str]:
    """Frozen regex splitter: terminal .?! only; semicolon/colon stay intra-sentence."""
    normalized = normalize_text(text)
    if not normalized:
        return []
    pieces: list[str] = []
    start = 0
    for index, char in enumerate(normalized):
        if char not in ".?!" or (char == "." and index and normalized[index - 1].isdigit()):
            continue
        if index + 1 < len(normalized) and not normalized[index + 1].isspace():
            continue
        pieces.append(normalized[start : index + 1].strip())
        start = index + 1
    tail = normalized[start:].strip()
    return [piece for piece in [*pieces, tail] if piece]
