"""Versioned, cross-platform hashes for text-based evaluation artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

CANONICAL_HASH_VERSION = "canonical-hash-v1"
SOURCE_HASH_SCHEMA_VERSION = "source-hash-v2"
HashMode = Literal[
    "raw_sha256",
    "canonical_text_v1",
    "canonical_json_v1",
    "canonical_jsonl_v1",
]


class CanonicalHashError(ValueError):
    """Raised when an artifact cannot be hashed under the requested protocol."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_raw_file(path: Path) -> str:
    return _sha256(path.read_bytes())


def canonicalize_text_bytes(data: bytes) -> bytes:
    """Normalize UTF-8 BOM/newlines and use exactly one terminal LF for non-empty text."""
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise CanonicalHashError("canonical text input must be valid UTF-8") from exc
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    if not normalized:
        return b""
    return (normalized.rstrip("\n") + "\n").encode("utf-8")


def sha256_canonical_text_file(path: Path) -> str:
    return _sha256(canonicalize_text_bytes(path.read_bytes()))


def canonicalize_json_value(value: Any) -> bytes:
    try:
        serialized = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise CanonicalHashError("value is not canonical JSON") from exc
    return (serialized + "\n").encode("utf-8")


def _load_json_bytes(data: bytes) -> Any:
    try:
        return json.loads(data.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CanonicalHashError("invalid UTF-8 JSON") from exc


def sha256_canonical_json_file(path: Path) -> str:
    return _sha256(canonicalize_json_value(_load_json_bytes(path.read_bytes())))


def canonicalize_jsonl_bytes(data: bytes) -> bytes:
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise CanonicalHashError("canonical JSONL input must be valid UTF-8") from exc
    records: list[bytes] = []
    for line_number, line in enumerate(
        text.replace("\r\n", "\n").replace("\r", "\n").split("\n"), 1
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CanonicalHashError(f"invalid JSONL record at line {line_number}") from exc
        records.append(canonicalize_json_value(value).rstrip(b"\n"))
    return b"\n".join(records) + (b"\n" if records else b"")


def sha256_canonical_jsonl_file(path: Path) -> str:
    return _sha256(canonicalize_jsonl_bytes(path.read_bytes()))


def hash_with_metadata(path: Path, mode: str) -> dict[str, str]:
    functions = {
        "raw_sha256": sha256_raw_file,
        "canonical_text_v1": sha256_canonical_text_file,
        "canonical_json_v1": sha256_canonical_json_file,
        "canonical_jsonl_v1": sha256_canonical_jsonl_file,
    }
    try:
        function = functions[mode]
    except KeyError as exc:
        raise CanonicalHashError(f"unsupported canonical hash mode: {mode}") from exc
    return {
        "algorithm": "sha256",
        "mode": mode,
        "value": function(path),
        "raw_value_at_review": sha256_raw_file(path),
        "schema_version": SOURCE_HASH_SCHEMA_VERSION,
        "canonicalization_version": CANONICAL_HASH_VERSION,
    }


def legacy_text_hash_variants(path: Path) -> dict[str, str]:
    """Return auditable byte representations; never use these as semantic acceptance."""
    data = path.read_bytes()
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise CanonicalHashError("legacy text input must be valid UTF-8") from exc
    lf = text.replace("\r\n", "\n").replace("\r", "\n")
    encoded = text.encode("utf-8")
    return {
        "raw": _sha256(data),
        "lf": _sha256(lf.encode("utf-8")),
        "crlf": _sha256(lf.replace("\n", "\r\n").encode("utf-8")),
        "bom_removed": _sha256(encoded),
        "bom_added": _sha256(b"\xef\xbb\xbf" + encoded),
        "single_terminal_lf": _sha256(canonicalize_text_bytes(data)),
    }


def verify_legacy_raw_hash(path: Path, expected: str) -> str:
    matches = [name for name, value in legacy_text_hash_variants(path).items() if value == expected]
    if not matches:
        raise CanonicalHashError(
            f"legacy raw hash is not explained by text normalization: {path}"
        )
    return matches[0]
