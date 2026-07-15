# ruff: noqa: E501
"""Prepare and manually review the 24 Stage 13.2 Dev citation samples."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from paper_research.evaluation.canonical_hash import (
    SOURCE_HASH_SCHEMA_VERSION,
    hash_with_metadata,
    verify_legacy_raw_hash,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/evaluation"
DOCS = ROOT / "docs"
ARTIFACTS = ROOT / "artifacts"
AUDIT = DATA / "evidence-qa-dev-citation-audit-v1.jsonl"
GUIDE = DOCS / "evidence-qa-dev-citation-review-guide-v1.md"
PACK = ARTIFACTS / "stage13-3-dev-citation-review-pack.zip"
LABELS = {
    "fully_supported",
    "partially_supported",
    "related_but_insufficient",
    "unsupported",
    "gold_annotation_too_narrow",
    "ambiguous_claim",
    "malformed_evidence",
}
HUMAN_FIELDS = {
    "human_review_status",
    "human_label",
    "reviewer",
    "reviewed_at",
    "review_notes",
    "human_reviewer",
    "human_reviewed_at",
}
SOURCE_SPECS = {
    "evidence_corpus": (DATA / "evidence-corpus-v1.jsonl", "canonical_jsonl_v1"),
    "gold_set": (DATA / "gold-set-v1.jsonl", "canonical_jsonl_v1"),
    "retrieval_gold": (DATA / "retrieval-gold-v2.jsonl", "canonical_jsonl_v1"),
    "dev_summary": (DATA / "evidence-qa-dev-v1.json", "canonical_json_v1"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list-pending", action="store_true")
    parser.add_argument("--sample-id")
    parser.add_argument("--label", choices=sorted(LABELS))
    parser.add_argument("--reviewer")
    parser.add_argument("--notes")
    parser.add_argument("--export", nargs="?", const="artifacts/stage13-3-dev-citation-review-export.md")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--build-pack", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_source_hashes() -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, (path, mode) in SOURCE_SPECS.items():
        metadata = hash_with_metadata(path, mode)
        output.update(
            {
                f"{name}_sha256": metadata["raw_value_at_review"],
                f"{name}_canonical_sha256": metadata["value"],
                f"{name}_hash_mode": metadata["mode"],
                f"{name}_hash_schema_version": metadata["schema_version"],
                f"{name}_raw_sha256_at_review": metadata["raw_value_at_review"],
                f"{name}_legacy_raw_hash_verified_via_newline_normalization": False,
            }
        )
    return output


def validate_source_hashes(recorded: dict[str, Any]) -> None:
    for name, (path, expected_mode) in SOURCE_SPECS.items():
        legacy = recorded.get(f"{name}_sha256")
        raw_at_review = recorded.get(f"{name}_raw_sha256_at_review")
        mode = recorded.get(f"{name}_hash_mode")
        schema = recorded.get(f"{name}_hash_schema_version")
        canonical = recorded.get(f"{name}_canonical_sha256")
        if not all((legacy, raw_at_review, mode, schema, canonical)):
            raise RuntimeError(f"incomplete source hash metadata: {name}")
        if schema != SOURCE_HASH_SCHEMA_VERSION:
            raise RuntimeError(f"unsupported source hash schema: {schema}")
        if mode != expected_mode:
            raise RuntimeError(f"unsupported source hash mode for {name}: {mode}")
        current = hash_with_metadata(path, mode)
        if canonical != current["value"]:
            raise RuntimeError(f"canonical source hash invalid: {name}")
        if legacy != raw_at_review:
            raise RuntimeError(f"legacy/raw-at-review hash mismatch: {name}")
        operation = verify_legacy_raw_hash(path, legacy)
        migrated = recorded.get(
            f"{name}_legacy_raw_hash_verified_via_newline_normalization"
        )
        if bool(migrated) != (operation != "raw"):
            raise RuntimeError(f"legacy source hash migration evidence invalid: {name}")


def source_record_hash(row: dict[str, Any]) -> str:
    source = {
        key: value
        for key, value in row.items()
        if key not in HUMAN_FIELDS | {"source_record_hash", "source_hashes"}
    }
    body = json.dumps(source, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode()).hexdigest()


def _run_by_question() -> dict[str, Path]:
    output = {}
    root = DATA / "evidence-qa-dev-v1/runs/retrieval_only"
    for path in root.glob("*/result.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row["status"] == "completed":
            output[row["question_id"]] = path.parent
    return output


def prepare() -> dict[str, Any]:
    rows = read_jsonl(AUDIT)
    if len(rows) != 24:
        raise RuntimeError(f"expected 24 citation audit rows, found {len(rows)}")
    evidence_rows = read_jsonl(DATA / "evidence-corpus-v1.jsonl")
    by_triple = {
        (row["paper_id"], int(row["page"]), row["block_id"]): row
        for row in evidence_rows
    }
    by_block = {(row["paper_id"], row["block_id"]): row for row in evidence_rows}
    run_by_q = _run_by_question()
    source_hashes = build_source_hashes()
    changed = False
    for row in rows:
        triple = (
            row["citation"]["paper_id"],
            int(row["citation"]["page"]),
            row["citation"]["block_id"],
        )
        if triple not in by_triple:
            raise RuntimeError(f"citation triple not found: {triple}")
        unit = by_triple[triple]
        run_dir = run_by_q[row["question_id"]]
        result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
        context = json.loads((run_dir / "context-trace.json").read_text(encoding="utf-8"))
        adjacent_ids = {
            item["block_id"] for item in context["adjacent_completion_blocks"]
        }

        def neighbor(
            block_id: str | None,
            paper_id: str = unit["paper_id"],
        ) -> dict[str, Any] | None:
            if not block_id:
                return None
            item = by_block.get((paper_id, block_id))
            if item is None:
                return {"block_id": block_id, "missing": True}
            return {
                "paper_id": item["paper_id"],
                "page": int(item["page"]),
                "block_id": item["block_id"],
                "text": item["text"],
            }

        additions = {
            "retrieval_variant": result["retrieval_variant"],
            "prompt_version": result["prompt_version"],
            "allowed_citation_triples": context["allowed_citation_triples"],
            "block_type": unit["block_type"],
            "is_adjacent_completion_block": unit["block_id"] in adjacent_ids,
            "adjacent_evidence_context": {
                "previous": neighbor(unit.get("previous_block_id")),
                "next": neighbor(unit.get("next_block_id")),
            },
            "source_hashes": source_hashes,
        }
        for key, value in additions.items():
            if row.get(key) != value:
                row[key] = value
                changed = True
        calculated = source_record_hash(row)
        if row.get("source_record_hash") != calculated:
            row["source_record_hash"] = calculated
            changed = True
    if changed:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        backup = AUDIT.with_name(f"{AUDIT.name}.pending.{stamp}.bak")
        shutil.copy2(AUDIT, backup)
        write_jsonl(AUDIT, rows)
    else:
        backup = None
    validate(rows)
    return {"records": len(rows), "pending": sum(row["human_review_status"] == "pending" for row in rows), "backup": str(backup) if backup else None, "changed": changed}


def validate(rows: list[dict[str, Any]]) -> None:
    required = {
        "sample_id", "variant", "question_id", "claim_id", "claim_text",
        "citation", "cited_evidence_text", "adjacent_evidence_context",
        "gold_blocks", "gold_pages", "automated_labels", "retrieval_variant",
        "prompt_version", "allowed_citation_triples", "human_review_status",
        "human_label", "reviewer", "reviewed_at", "review_notes",
        "source_hashes", "source_record_hash",
    }
    evidence = {
        (row["paper_id"], int(row["page"]), row["block_id"])
        for row in read_jsonl(DATA / "evidence-corpus-v1.jsonl")
    }
    for row in rows:
        missing = required - set(row)
        if missing:
            raise RuntimeError(f"{row.get('sample_id')} missing fields: {sorted(missing)}")
        triple = (row["citation"]["paper_id"], int(row["citation"]["page"]), row["citation"]["block_id"])
        if triple not in evidence:
            raise RuntimeError(f"missing citation triple: {triple}")
        if list(triple) not in row["allowed_citation_triples"]:
            raise RuntimeError(f"citation not allowed in original run: {triple}")
        if source_record_hash(row) != row["source_record_hash"]:
            raise RuntimeError(f"source record hash invalid: {row['sample_id']}")
        validate_source_hashes(row["source_hashes"])


def review(args: argparse.Namespace) -> None:
    if not all((args.sample_id, args.label, args.reviewer, args.notes)):
        raise RuntimeError("--sample-id, --label, --reviewer and --notes are required")
    rows = read_jsonl(AUDIT)
    validate(rows)
    target = next((row for row in rows if row["sample_id"] == args.sample_id), None)
    if target is None:
        raise RuntimeError(f"unknown sample: {args.sample_id}")
    if target["human_review_status"] != "pending" and not args.overwrite:
        raise RuntimeError("sample is already reviewed; use --overwrite")
    backup = AUDIT.with_name(
        f"{AUDIT.name}.review.{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}.bak"
    )
    shutil.copy2(AUDIT, backup)
    target.update(
        human_review_status="reviewed",
        human_label=args.label,
        reviewer=args.reviewer,
        reviewed_at=datetime.now(UTC).isoformat(),
        review_notes=args.notes,
    )
    write_jsonl(AUDIT, rows)


def export(rows: list[dict[str, Any]], path: Path) -> None:
    selected = rows
    lines = ["# Stage 13.3 Dev Citation Review Export", ""]
    for row in selected:
        lines += [
            f"## {row['sample_id']} — {row['question_id']} / {row['claim_id']}",
            "",
            f"Claim: {row['claim_text']}",
            "",
            f"Citation: `{json.dumps(row['citation'], sort_keys=True)}`",
            "",
            "Cited evidence:",
            "",
            row["cited_evidence_text"],
            "",
            f"Automated labels: `{json.dumps(row['automated_labels'], sort_keys=True)}`",
            "",
            "Human label: pending",
            "",
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_pack() -> None:
    files = [
        AUDIT,
        DATA / "evidence-corpus-v1.jsonl",
        DATA / "gold-set-v1.jsonl",
        DATA / "retrieval-gold-v2.jsonl",
        DATA / "claim-units-v1.jsonl",
        DATA / "evidence-qa-dev-v1.json",
        GUIDE,
    ]
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(PACK, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, arcname=path.name)


def write_guide() -> None:
    GUIDE.write_text(
        "# Evidence QA Dev Citation Review Guide v1\n\n"
        "Review all 24 claim-citation pairs against the cited block and adjacent context. "
        "Automated exact/page/semantic fields are routing aids, not human labels.\n\n"
        "Valid labels: `fully_supported`, `partially_supported`, "
        "`related_but_insufficient`, `unsupported`, `gold_annotation_too_narrow`, "
        "`ambiguous_claim`, `malformed_evidence`.\n\n"
        "Example:\n\n"
        "```powershell\n"
        ".\\.venv\\Scripts\\python.exe scripts\\review_evidence_qa_dev_citations_v1.py "
        "--sample-id evidence-qa-dev-citation-001 --label fully_supported "
        "--reviewer <name> --notes \"Directly supported by the cited block.\"\n"
        "```\n\n"
        "Reviewer and notes are mandatory. Existing reviews require `--overwrite`. Every update "
        "creates a backup and fails if source hashes or citation triples changed. No model or "
        "automated signal may fill the human label.\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.prepare:
        write_guide()
        print(json.dumps(prepare()))
    rows = read_jsonl(AUDIT)
    if args.list_pending:
        for row in rows:
            if row["human_review_status"] == "pending":
                print(row["sample_id"], row["question_id"], row["claim_id"])
    if args.sample_id and args.label:
        review(args)
    if args.export:
        export(rows, ROOT / args.export)
    if args.build_pack:
        validate(read_jsonl(AUDIT))
        build_pack()
        print(str(PACK))


if __name__ == "__main__":
    main()
