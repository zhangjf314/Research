# ruff: noqa: E501
"""Validate, import, and summarize externally reviewed Stage 13.2 Dev citations."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from scripts.review_evidence_qa_dev_citations_v1 import (
        AUDIT,
        DATA,
        HUMAN_FIELDS,
        LABELS,
        read_jsonl,
        source_record_hash,
        validate,
        write_jsonl,
    )
except ModuleNotFoundError:
    from review_evidence_qa_dev_citations_v1 import (  # type: ignore[no-redef]
        AUDIT,
        DATA,
        HUMAN_FIELDS,
        LABELS,
        read_jsonl,
        source_record_hash,
        validate,
        write_jsonl,
    )

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REVIEWED = ROOT / "artifacts/evidence-qa-dev-citation-human-audit-v1.jsonl"
SUMMARY = DATA / "evidence-qa-dev-citation-audit-summary-v1.json"
REPORT = ROOT / "docs/evidence-qa-dev-citation-audit-summary-v1.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviewed", type=Path, default=DEFAULT_REVIEWED)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    return parser.parse_args()


def validate_external(
    current: list[dict[str, Any]], reviewed: list[dict[str, Any]]
) -> dict[str, Any]:
    if len(reviewed) != 24:
        raise RuntimeError(f"expected 24 reviewed records, found {len(reviewed)}")
    identifiers = [row.get("sample_id") for row in reviewed]
    if None in identifiers or len(identifiers) != len(set(identifiers)):
        raise RuntimeError("sample_id values must be present and unique")
    current_by_id = {row["sample_id"]: row for row in current}
    reviewed_by_id = {row["sample_id"]: row for row in reviewed}
    if set(current_by_id) != set(reviewed_by_id):
        raise RuntimeError("reviewed sample IDs differ from the frozen pending set")
    immutable_changes: list[tuple[str, str]] = []
    for sample_id, source in current_by_id.items():
        target = reviewed_by_id[sample_id]
        if target.get("human_review_status") != "approved":
            raise RuntimeError(f"{sample_id}: human_review_status must be approved")
        if target.get("human_label") not in LABELS:
            raise RuntimeError(f"{sample_id}: invalid human_label")
        for field in ("reviewer", "reviewed_at", "review_notes"):
            if not str(target.get(field) or "").strip():
                raise RuntimeError(f"{sample_id}: {field} is required")
        if target.get("source_hashes") != source.get("source_hashes"):
            raise RuntimeError(f"{sample_id}: source_hashes changed")
        if target.get("source_record_hash") != source.get("source_record_hash"):
            raise RuntimeError(f"{sample_id}: source_record_hash changed")
        for field in set(source) - HUMAN_FIELDS:
            if target.get(field) != source.get(field):
                immutable_changes.append((sample_id, field))
        if source_record_hash(target) != target["source_record_hash"]:
            raise RuntimeError(f"{sample_id}: source_record_hash no longer validates")
    if immutable_changes:
        raise RuntimeError(f"immutable reviewed fields changed: {immutable_changes[:5]}")
    validate(reviewed)
    return {
        "records": len(reviewed),
        "unique_sample_ids": len(set(identifiers)),
        "approved": sum(row["human_review_status"] == "approved" for row in reviewed),
        "source_hashes_valid": True,
        "source_record_hashes_valid": True,
        "citation_triples_valid": True,
        "immutable_changes": 0,
    }


def stratum(row: dict[str, Any]) -> str:
    labels = row["automated_labels"]
    if labels.get("exact_gold"):
        return "exact_gold"
    if labels.get("same_page"):
        return "same_page_non_exact"
    if labels.get("semantic_support_signal"):
        return "semantic_support"
    return "unsupported_signal"


def group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = Counter(row["human_label"] for row in rows)
    valid = [
        row
        for row in rows
        if row["human_label"] not in {"ambiguous_claim", "malformed_evidence"}
    ]
    strict = sum(row["human_label"] == "fully_supported" for row in valid)
    lenient = sum(
        row["human_label"] in {"fully_supported", "partially_supported"}
        for row in valid
    )
    return {
        "total_reviewed": len(rows),
        "valid_reviewed": len(valid),
        "label_counts": {label: labels.get(label, 0) for label in sorted(LABELS)},
        "strict_support_rate": round(strict / len(valid), 6) if valid else None,
        "lenient_support_rate": round(lenient / len(valid), 6) if valid else None,
    }


def grouped(
    rows: list[dict[str, Any]], key
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(key(row))].append(row)
    return {name: group_summary(values) for name, values in sorted(groups.items())}


def build_summary(
    reviewed: list[dict[str, Any]], validation: dict[str, Any], source: Path
) -> dict[str, Any]:
    gold = {
        row["question_id"]: row for row in read_jsonl(DATA / "gold-set-v1.jsonl")
    }
    enriched = []
    for row in reviewed:
        enriched.append(
            {
                **row,
                "audit_stratum": stratum(row),
                "category": gold[row["question_id"]]["category"],
                "difficulty": gold[row["question_id"]]["difficulty"],
                "selection_origin": (
                    "adjacent_completion_selected_block"
                    if row["is_adjacent_completion_block"]
                    else "original_selected_block"
                ),
            }
        )
    overall = group_summary(enriched)
    complete = (
        len(enriched) == 24
        and all(row["human_review_status"] == "approved" for row in enriched)
        and all(row["human_label"] in LABELS for row in enriched)
    )
    strata = grouped(enriched, lambda row: row["audit_stratum"])
    empty = group_summary([])
    for name in (
        "exact_gold",
        "same_page_non_exact",
        "semantic_support",
        "unsupported_signal",
    ):
        strata.setdefault(name, empty)
    return {
        "schema_version": "evidence-qa-dev-citation-audit-summary-v1",
        "source_review_file": str(source),
        "review_method": "externally supplied AI-assisted manual citation audit",
        "representativeness_warning": "These 24 Stage 13.2 Dev claim-citation pairs must not be extrapolated as Full-50 or production citation precision.",
        "validation": validation,
        "overall": overall,
        "strata": dict(sorted(strata.items())),
        "selection_origin": grouped(enriched, lambda row: row["selection_origin"]),
        "block_type": grouped(enriched, lambda row: row["block_type"]),
        "category": grouped(enriched, lambda row: row["category"]),
        "difficulty": grouped(enriched, lambda row: row["difficulty"]),
        "phase_b_gain_questions": grouped(
            [
                row
                for row in enriched
                if row["question_id"] in {"q002", "q007", "q013", "q050"}
            ],
            lambda row: row["question_id"],
        ),
        "gold_annotation_too_narrow_found": overall["label_counts"][
            "gold_annotation_too_narrow"
        ]
        > 0,
        "human_citation_audit_complete": complete,
        "human_citation_audit_complete_marker": f"HUMAN_CITATION_AUDIT_COMPLETE={str(complete).lower()}",
        "dev_v2_run": False,
        "full_qa_run": False,
        "deep_research_run": False,
        "dev_v2_authorized": False,
    }


def write_report(payload: dict[str, Any]) -> None:
    overall = payload["overall"]
    lines = [
        "# Evidence QA Dev Citation Audit Summary v1",
        "",
        f"- `{payload['human_citation_audit_complete_marker']}`",
        f"- Total / valid reviewed: {overall['total_reviewed']} / {overall['valid_reviewed']}",
        f"- Strict support rate: {overall['strict_support_rate']:.6f}",
        f"- Lenient support rate: {overall['lenient_support_rate']:.6f}",
        f"- Gold annotation too narrow found: {payload['gold_annotation_too_narrow_found']}",
        "- Dev v2 run: **False**",
        "",
        "> This is an externally supplied AI-assisted manual audit of 24 Stage 13.2 Dev claim-citation pairs. It is not an independent double-blind review and must not be extrapolated to Full-50 or production precision.",
        "",
        "## Label counts",
        "",
        "| Label | Count |",
        "|---|---:|",
    ]
    for label, count in overall["label_counts"].items():
        lines.append(f"| {label} | {count} |")
    for title, key in [
        ("Automated stratum", "strata"),
        ("Selection origin", "selection_origin"),
        ("Block type", "block_type"),
        ("Category", "category"),
        ("Difficulty", "difficulty"),
    ]:
        lines += [
            "",
            f"## {title}",
            "",
            "| Group | N | Strict | Lenient |",
            "|---|---:|---:|---:|",
        ]
        for name, values in payload[key].items():
            lines.append(
                f"| {name} | {values['total_reviewed']} | "
                f"{values['strict_support_rate']} | {values['lenient_support_rate']} |"
            )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.reviewed.exists():
        print("WAITING_FOR_EXTERNAL_HUMAN_CITATION_AUDIT")
        raise SystemExit(2)
    current = read_jsonl(AUDIT)
    reviewed = read_jsonl(args.reviewed)
    validation = validate_external(current, reviewed)
    if args.validate_only:
        print(json.dumps(validation))
        return
    if args.summary_only:
        payload = build_summary(reviewed, validation, args.reviewed)
        existing = (
            json.loads(SUMMARY.read_text(encoding="utf-8"))
            if SUMMARY.exists()
            else {}
        )
        payload["pending_backup"] = existing.get("pending_backup")
        SUMMARY.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        write_report(payload)
        print(json.dumps({"status": payload["human_citation_audit_complete_marker"]}))
        return
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    backup = AUDIT.with_name(f"{AUDIT.name}.pre-human-import.{stamp}.bak")
    shutil.copy2(AUDIT, backup)
    write_jsonl(AUDIT, reviewed)
    payload = build_summary(reviewed, validation, args.reviewed)
    payload["pending_backup"] = str(backup)
    SUMMARY.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(payload)
    print(
        json.dumps(
            {
                "status": payload["human_citation_audit_complete_marker"],
                "strict_support_rate": payload["overall"]["strict_support_rate"],
                "lenient_support_rate": payload["overall"]["lenient_support_rate"],
                "backup": str(backup),
                "dev_v2_run": False,
            }
        )
    )


if __name__ == "__main__":
    main()
