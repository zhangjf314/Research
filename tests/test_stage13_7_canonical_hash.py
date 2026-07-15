from __future__ import annotations

import json
from pathlib import Path

import pytest

from paper_research.evaluation.canonical_hash import (
    CanonicalHashError,
    hash_with_metadata,
    sha256_canonical_json_file,
    sha256_canonical_jsonl_file,
    sha256_canonical_text_file,
    verify_legacy_raw_hash,
)


def write_bytes(path: Path, value: bytes) -> Path:
    path.write_bytes(value)
    return path


def test_lf_crlf_and_bom_have_same_canonical_text_hash(tmp_path: Path) -> None:
    lf = write_bytes(tmp_path / "lf.md", b"heading\nbody\n")
    crlf = write_bytes(tmp_path / "crlf.md", b"heading\r\nbody\r\n")
    bom = write_bytes(tmp_path / "bom.md", b"\xef\xbb\xbfheading\nbody\n")
    assert sha256_canonical_text_file(lf) == sha256_canonical_text_file(crlf)
    assert sha256_canonical_text_file(lf) == sha256_canonical_text_file(bom)


def test_json_formatting_is_ignored_but_values_are_not(tmp_path: Path) -> None:
    compact = write_bytes(tmp_path / "compact.json", b'{"b":2,"a":1}')
    indented = write_bytes(tmp_path / "indented.json", b'{\r\n  "a": 1,\r\n  "b": 2\r\n}')
    changed = write_bytes(tmp_path / "changed.json", b'{"b":3,"a":1}')
    assert sha256_canonical_json_file(compact) == sha256_canonical_json_file(indented)
    assert sha256_canonical_json_file(compact) != sha256_canonical_json_file(changed)


def test_jsonl_newlines_are_ignored_but_record_order_is_not(tmp_path: Path) -> None:
    lf = write_bytes(tmp_path / "lf.jsonl", b'{"a":1}\n{"b":2}\n')
    crlf = write_bytes(tmp_path / "crlf.jsonl", b'{"a":1}\r\n{"b":2}\r\n')
    reversed_rows = write_bytes(tmp_path / "reversed.jsonl", b'{"b":2}\n{"a":1}\n')
    assert sha256_canonical_jsonl_file(lf) == sha256_canonical_jsonl_file(crlf)
    assert sha256_canonical_jsonl_file(lf) != sha256_canonical_jsonl_file(reversed_rows)


def test_markdown_content_spaces_remain_significant(tmp_path: Path) -> None:
    plain = write_bytes(tmp_path / "plain.md", b"alpha beta\n")
    changed = write_bytes(tmp_path / "changed.md", b"alpha  beta\n")
    assert sha256_canonical_text_file(plain) != sha256_canonical_text_file(changed)


def test_unknown_hash_mode_fails_closed(tmp_path: Path) -> None:
    path = write_bytes(tmp_path / "value.txt", b"value\n")
    with pytest.raises(CanonicalHashError, match="unsupported canonical hash mode"):
        hash_with_metadata(path, "unknown")


def test_legacy_crlf_hash_is_provable_and_unrelated_hash_fails(tmp_path: Path) -> None:
    path = write_bytes(tmp_path / "value.json", b'{\n  "a": 1\n}')
    crlf = write_bytes(tmp_path / "crlf.json", b'{\r\n  "a": 1\r\n}')
    legacy = hash_with_metadata(crlf, "raw_sha256")["value"]
    assert verify_legacy_raw_hash(path, legacy) == "crlf"
    with pytest.raises(CanonicalHashError, match="not explained"):
        verify_legacy_raw_hash(path, "0" * 64)


def test_migration_preserves_human_and_non_hash_fields() -> None:
    root = Path("data/evaluation")
    migration = json.loads(
        (root / "stage13-review-hash-migration-v1.json").read_text(encoding="utf-8")
    )
    integrity = json.loads(
        (root / "stage13-7-review-integrity-audit-v1.json").read_text(encoding="utf-8")
    )
    assert migration["affected_records"] == 81
    assert migration["label_field_changes"] == 0
    assert migration["reviewer_field_changes"] == 0
    assert migration["immutable_non_hash_field_changes"] == 0
    assert integrity["human_labels_changed"] == 0
    assert integrity["reviewer_fields_changed"] == 0
    assert integrity["immutable_non_hash_fields_changed"] == 0
