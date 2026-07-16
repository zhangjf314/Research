"""Human-only CLI for Stage 13.10 claim-level Gold adjudication."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from paper_research.evaluation.canonical_hash import hash_with_metadata

try:
    from scripts.build_claim_evidence_gold_dev_v1 import (
        DATA,
        GOLD_VERSION,
        MUTABLE_RELATION_FIELDS,
        OUTPUT,
        SOURCE_FILES,
        immutable_payload,
    )
    from scripts.evidence_qa_dev_lib_v1 import canonical_hash, read_jsonl
except ModuleNotFoundError:
    from build_claim_evidence_gold_dev_v1 import (  # type: ignore[no-redef]
        DATA,
        GOLD_VERSION,
        MUTABLE_RELATION_FIELDS,
        OUTPUT,
        SOURCE_FILES,
        immutable_payload,
    )
    from evidence_qa_dev_lib_v1 import canonical_hash, read_jsonl  # type: ignore[no-redef]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, default=OUTPUT)
    parser.add_argument("--list-pending", action="store_true")
    parser.add_argument("--required-claim-id")
    parser.add_argument("--approve-core", action="append", default=[])
    parser.add_argument("--approve-supporting", action="append", default=[])
    parser.add_argument("--mark-equivalent", action="append", default=[])
    parser.add_argument("--reject", action="append", default=[])
    parser.add_argument("--no-valid-evidence", action="store_true")
    parser.add_argument("--reviewer")
    parser.add_argument("--notes")
    parser.add_argument("--export", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _flatten(values: list[str]) -> list[str]:
    return [item.strip() for value in values for item in value.split(",") if item.strip()]


def validate(rows: list[dict[str, Any]]) -> None:
    if len(rows) != 27 or len({row["required_claim_id"] for row in rows}) != 27:
        raise RuntimeError("expected 27 unique required claims")
    evidence = {
        (row["paper_id"], int(row["page"]), row["block_id"]): row
        for row in read_jsonl(DATA / "evidence-corpus-v1.jsonl")
    }
    claims = {row["claim_id"]: row for row in read_jsonl(DATA / "claim-units-v1.jsonl")}
    for row in rows:
        if row["gold_version"] != GOLD_VERSION:
            raise RuntimeError(f"unexpected Gold version: {row['required_claim_id']}")
        source_claim = claims.get(row["required_claim_id"])
        if source_claim is None or canonical_hash(source_claim) != row["source_record_hash"]:
            raise RuntimeError(f"source claim changed: {row['required_claim_id']}")
        if canonical_hash(immutable_payload(row)) != row["immutable_record_hash"]:
            raise RuntimeError(f"immutable fields changed: {row['required_claim_id']}")
        expected_hashes = {
            name: hash_with_metadata(
                path,
                "canonical_jsonl_v1" if path.suffix == ".jsonl" else "canonical_json_v1",
            )
            for name, path in SOURCE_FILES.items()
        }
        if row["source_hashes"] != expected_hashes:
            raise RuntimeError(f"source hashes changed: {row['required_claim_id']}")
        relation_ids: set[str] = set()
        for relation in row["candidate_evidence_relations"]:
            if relation["relation_id"] in relation_ids:
                raise RuntimeError(f"duplicate relation: {relation['relation_id']}")
            relation_ids.add(relation["relation_id"])
            triple = (relation["paper_id"], int(relation["page"]), relation["block_id"])
            unit = evidence.get(triple)
            if unit is None or unit["text"] != relation["evidence_text"]:
                raise RuntimeError(f"relation triple/text changed: {relation['relation_id']}")
        core_ids = {
            relation_id
            for core_set in row["approved_core_relations"]
            for relation_id in core_set["required_relations"]
        }
        groups = [
            core_ids,
            set(row["approved_supporting_relations"]),
            set(row["equivalent_non_gold_relations"]),
            set(row["rejected_relations"]),
        ]
        if any(not group <= relation_ids for group in groups):
            raise RuntimeError(f"unknown adjudicated relation: {row['required_claim_id']}")
        if sum(len(group) for group in groups) != len(set().union(*groups)):
            raise RuntimeError(f"relation assigned to multiple roles: {row['required_claim_id']}")
        approved = bool(core_ids or groups[1] or groups[2])
        if row["no_valid_gold_evidence"] and approved:
            raise RuntimeError(
                "no-valid-evidence conflicts with approved relations: "
                f"{row['required_claim_id']}"
            )
        if row["adjudication_status"] == "approved":
            if not row["reviewer"] or not row["reviewed_at"] or not row["review_notes"]:
                raise RuntimeError(f"incomplete human metadata: {row['required_claim_id']}")
            if not row["no_valid_gold_evidence"] and not core_ids:
                raise RuntimeError(
                    "approved answerable claim requires a core set: "
                    f"{row['required_claim_id']}"
                )


def _set_relation(
    relation: dict[str, Any], role: str, scope: str, label: str, notes: str
) -> None:
    relation.update(
        relation_role=role,
        support_scope=scope,
        adjudication_label=label,
        adjudication_notes=notes,
    )


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.path)
    validate(rows)
    if args.list_pending:
        for row in rows:
            if row["adjudication_status"] == "pending":
                print(f"{row['question_id']}\t{row['required_claim_id']}\t{row['required_claim_text']}")
        return
    if args.export:
        if args.export.exists() and not args.overwrite:
            raise RuntimeError("export exists; pass --overwrite")
        args.export.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.path, args.export)
        if not args.required_claim_id:
            return
    if not args.required_claim_id:
        raise RuntimeError("--required-claim-id is required for a review update")
    if not args.reviewer or not args.notes:
        raise RuntimeError("--reviewer and --notes are required")
    row = next(
        (item for item in rows if item["required_claim_id"] == args.required_claim_id),
        None,
    )
    if row is None:
        raise RuntimeError("unknown required claim")
    if row["adjudication_status"] == "approved" and not args.overwrite:
        raise RuntimeError("reviewed row requires --overwrite")
    core = _flatten(args.approve_core)
    supporting = _flatten(args.approve_supporting)
    equivalent = _flatten(args.mark_equivalent)
    rejected = _flatten(args.reject)
    selected = core + supporting + equivalent + rejected
    if len(selected) != len(set(selected)):
        raise RuntimeError("a relation cannot be assigned to multiple roles")
    known = {relation["relation_id"] for relation in row["candidate_evidence_relations"]}
    if not set(selected) <= known:
        raise RuntimeError("unknown relation ID")
    if args.no_valid_evidence and selected:
        raise RuntimeError("--no-valid-evidence is mutually exclusive with relation selections")
    if not args.no_valid_evidence and not core:
        raise RuntimeError("explicit core relation(s) or --no-valid-evidence required")
    relation_map = {
        relation["relation_id"]: relation for relation in row["candidate_evidence_relations"]
    }
    for relation in row["candidate_evidence_relations"]:
        for field in MUTABLE_RELATION_FIELDS:
            relation[field] = None
    for relation_id in core:
        _set_relation(
            relation_map[relation_id], "core", "fully_supports_claim", "core_gold", args.notes
        )
    for relation_id in supporting:
        _set_relation(
            relation_map[relation_id],
            "supporting",
            "partially_supports_claim",
            "supporting_gold",
            args.notes,
        )
    for relation_id in equivalent:
        _set_relation(
            relation_map[relation_id],
            "equivalent",
            "fully_supports_claim",
            "equivalent_valid_evidence",
            args.notes,
        )
    for relation_id in rejected:
        _set_relation(
            relation_map[relation_id], "rejected", "unrelated", "unrelated", args.notes
        )
    core_set = []
    if core:
        core_set = [{
            "core_set_id": f"cs-{row['question_id']}-{row['required_claim_id'][-6:]}-01",
            "required_relations": core,
            "minimum_complete_set": True,
        }]
    row.update(
        approved_core_relations=core_set,
        approved_supporting_relations=supporting,
        equivalent_non_gold_relations=equivalent,
        rejected_relations=rejected,
        no_valid_gold_evidence=bool(args.no_valid_evidence),
        adjudication_status="approved",
        reviewer=args.reviewer.strip(),
        reviewed_at=datetime.now(UTC).isoformat(),
        review_notes=args.notes.strip(),
    )
    validate(rows)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    backup = args.path.with_name(f"{args.path.name}.pre-review.{stamp}.bak")
    shutil.copy2(args.path, backup)
    args.path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in rows),
        encoding="utf-8",
    )
    print(json.dumps({"updated": args.required_claim_id, "backup": str(backup)}))


if __name__ == "__main__":
    main()
