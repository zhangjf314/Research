"""Validate, import, freeze, and summarize Stage 13.10 reviewed claim Gold."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import zipfile
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from paper_research.evaluation.canonical_hash import hash_with_metadata

try:
    from scripts.build_claim_evidence_gold_dev_v1 import (
        DATA,
        DOCS,
        GOLD_VERSION,
        OUTPUT,
        SCHEMA_VERSION,
        SOURCE_FILES,
        immutable_payload,
    )
    from scripts.evidence_qa_dev_lib_v1 import canonical_hash, read_jsonl
    from scripts.review_claim_evidence_gold_dev_v1 import core_relation_ids
except ModuleNotFoundError:
    from build_claim_evidence_gold_dev_v1 import (  # type: ignore[no-redef]
        DATA,
        DOCS,
        GOLD_VERSION,
        OUTPUT,
        SCHEMA_VERSION,
        SOURCE_FILES,
        immutable_payload,
    )
    from evidence_qa_dev_lib_v1 import canonical_hash, read_jsonl  # type: ignore[no-redef]
    from review_claim_evidence_gold_dev_v1 import (  # type: ignore[no-redef]
        core_relation_ids,
    )

ROOT = DATA.parents[1]
EXPECTED_PACKAGE_HASH = "a1dd665133e0dd0f6a4fbbfe1101702adec835a652fac9d37239a362d1fc1015"
EXPECTED_MEMBERS = {
    "claim-evidence-gold-dev-v1-reviewed.jsonl",
    "stage13-10-human-claim-gold-review-summary.json",
    "stage13-10-human-claim-gold-review-summary.md",
}
IMPORT_ROOT = ROOT / "artifacts/imports/stage13-10-human-claim-gold-review-results"
FREEZE_JSON = DATA / "claim-evidence-gold-dev-v1-freeze.json"
FREEZE_DOC = DOCS / "claim-evidence-gold-dev-v1-freeze.md"
SUMMARY_JSON = DATA / "claim-evidence-gold-dev-v1-summary.json"
SUMMARY_CSV = DATA / "claim-evidence-gold-dev-v1-summary.csv"
SUMMARY_DOC = DOCS / "claim-evidence-gold-dev-v1-summary.md"
ANSWERABLE_IDS = ["q001", "q002", "q004", "q007", "q008", "q013", "q015", "q019", "q050"]
SECRET_PATTERN = re.compile(
    r"(?i)(api[_ -]?key|authorization\s*:|bearer\s+[a-z0-9._-]+|cookie\s*:|"
    r"provider\s*header|postgres(?:ql)?://|redis://|sqlite)"
)
VALID_LABELS = {
    "core_gold",
    "supporting_gold",
    "equivalent_valid_evidence",
    "partially_relevant",
    "insufficient",
    "unrelated",
    "malformed_evidence",
    "gold_claim_ambiguous",
    "no_valid_evidence",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "artifacts/stage13-10-human-claim-gold-review-results.zip",
    )
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def extract_package(package: Path) -> Path:
    if sha256(package) != EXPECTED_PACKAGE_HASH:
        raise RuntimeError("CLAIM_GOLD_REVIEW_PACKAGE_HASH_MISMATCH")
    with zipfile.ZipFile(package) as archive:
        names = set(archive.namelist())
        if names != EXPECTED_MEMBERS:
            raise RuntimeError(f"unexpected review package members: {sorted(names)}")
        for info in archive.infolist():
            if Path(info.filename).name != info.filename:
                raise RuntimeError("nested or unsafe review package member")
            content = archive.read(info)
            text = content.decode("utf-8")
            if SECRET_PATTERN.search(text):
                raise RuntimeError(f"secret-like content in {info.filename}")
        IMPORT_ROOT.mkdir(parents=True, exist_ok=True)
        for info in archive.infolist():
            (IMPORT_ROOT / info.filename).write_bytes(archive.read(info))
    return IMPORT_ROOT


def _relation_map(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    relations = row["candidate_evidence_relations"]
    mapping = {relation["relation_id"]: relation for relation in relations}
    if len(mapping) != len(relations):
        raise RuntimeError(f"duplicate relation ID: {row['required_claim_id']}")
    return mapping


def _expected_source_hashes() -> dict[str, dict[str, str]]:
    return {
        name: hash_with_metadata(
            path,
            "canonical_jsonl_v1" if path.suffix == ".jsonl" else "canonical_json_v1",
        )
        for name, path in SOURCE_FILES.items()
    }


def validate_review(
    pending: list[dict[str, Any]], reviewed: list[dict[str, Any]]
) -> dict[str, Any]:
    if len(pending) != 27 or len(reviewed) != 27:
        raise RuntimeError("expected 27 pending and 27 reviewed records")
    pending_by_id = {row["required_claim_id"]: row for row in pending}
    reviewed_by_id = {row["required_claim_id"]: row for row in reviewed}
    if len(pending_by_id) != 27 or len(reviewed_by_id) != 27:
        raise RuntimeError("duplicate required claim")
    if set(pending_by_id) != set(reviewed_by_id):
        raise RuntimeError("partial claim review import")
    question_counts = Counter(row["question_id"] for row in reviewed)
    if question_counts != Counter({question_id: 3 for question_id in ANSWERABLE_IDS}):
        raise RuntimeError("fixed answerable question/claim coverage changed")

    evidence = {
        (row["paper_id"], int(row["page"]), row["block_id"]): row
        for row in read_jsonl(DATA / "evidence-corpus-v1.jsonl")
    }
    claims = {row["claim_id"]: row for row in read_jsonl(DATA / "claim-units-v1.jsonl")}
    expected_hashes = _expected_source_hashes()
    immutable_changes = 0
    relation_count = 0
    label_counts: Counter[str] = Counter()
    core_count = supporting_count = equivalent_count = 0
    multi_sets = no_valid = 0
    for claim_id, row in reviewed_by_id.items():
        original = pending_by_id[claim_id]
        if row["adjudication_status"] != "approved":
            raise RuntimeError(f"record is not approved: {claim_id}")
        if not row.get("reviewer") or not row.get("reviewed_at") or not row.get("review_notes"):
            raise RuntimeError(f"incomplete review metadata: {claim_id}")
        if row["schema_version"] != SCHEMA_VERSION or row["gold_version"] != GOLD_VERSION:
            raise RuntimeError(f"schema/Gold version changed: {claim_id}")
        if row["source_hashes"] != expected_hashes:
            raise RuntimeError(f"source hashes changed: {claim_id}")
        source_claim = claims.get(claim_id)
        if source_claim is None or canonical_hash(source_claim) != row["source_record_hash"]:
            raise RuntimeError(f"source claim changed: {claim_id}")
        if canonical_hash(immutable_payload(row)) != row["immutable_record_hash"]:
            raise RuntimeError(f"immutable hash invalid: {claim_id}")
        if row["immutable_record_hash"] != original["immutable_record_hash"]:
            immutable_changes += 1
        original_relations = original["candidate_evidence_relations"]
        reviewed_relations = row["candidate_evidence_relations"]
        if len(original_relations) != len(reviewed_relations):
            raise RuntimeError(f"candidate relation count changed: {claim_id}")
        for before, after in zip(original_relations, reviewed_relations, strict=True):
            before_immutable = {
                key: value
                for key, value in before.items()
                if key not in {
                    "relation_role",
                    "support_scope",
                    "adjudication_label",
                    "adjudication_notes",
                }
            }
            after_immutable = {
                key: value
                for key, value in after.items()
                if key not in {
                    "relation_role",
                    "support_scope",
                    "adjudication_label",
                    "adjudication_notes",
                }
            }
            if before_immutable != after_immutable:
                raise RuntimeError(f"candidate relation immutable fields changed: {claim_id}")
        relation_map = _relation_map(row)
        relation_count += len(relation_map)
        core_ids = core_relation_ids(row)
        supporting_ids = set(row["approved_supporting_relations"])
        equivalent_ids = set(row["equivalent_non_gold_relations"])
        rejected_ids = set(row["rejected_relations"])
        groups = [core_ids, supporting_ids, equivalent_ids, rejected_ids]
        if any(not group <= relation_map.keys() for group in groups):
            raise RuntimeError(f"unknown adjudicated relation: {claim_id}")
        if sum(len(group) for group in groups) != len(set().union(*groups)):
            raise RuntimeError(f"relation assigned to multiple outcomes: {claim_id}")
        if set().union(*groups) != relation_map.keys():
            raise RuntimeError(f"not every candidate relation was adjudicated: {claim_id}")
        if row["no_valid_gold_evidence"] and (core_ids or supporting_ids or equivalent_ids):
            raise RuntimeError(f"no-valid-evidence conflict: {claim_id}")
        if not row["no_valid_gold_evidence"] and not (core_ids or equivalent_ids):
            raise RuntimeError(f"approved claim lacks core/equivalent evidence: {claim_id}")
        if len(core_ids | equivalent_ids) > 1:
            multi_sets += 1
        no_valid += bool(row["no_valid_gold_evidence"])
        core_count += len(core_ids)
        supporting_count += len(supporting_ids)
        equivalent_count += len(equivalent_ids)
        for relation_id, relation in relation_map.items():
            triple = (relation["paper_id"], int(relation["page"]), relation["block_id"])
            unit = evidence.get(triple)
            if (
                unit is None
                or unit["text"] != relation["evidence_text"]
                or unit["block_type"] != relation["block_type"]
            ):
                raise RuntimeError(f"relation triple/text/type invalid: {relation_id}")
            label = relation["adjudication_label"]
            if label not in VALID_LABELS or not relation["adjudication_notes"]:
                raise RuntimeError(f"incomplete relation adjudication: {relation_id}")
            label_counts[label] += 1
            if relation_id in core_ids and (
                label != "core_gold"
                or relation["relation_role"] != "core"
                or not relation["support_scope"]
            ):
                raise RuntimeError(f"invalid core relation: {relation_id}")
            if relation_id in supporting_ids and (
                label != "supporting_gold"
                or relation["relation_role"] != "supporting"
                or not relation["support_scope"]
            ):
                raise RuntimeError(f"invalid supporting relation: {relation_id}")
            if relation_id in equivalent_ids and (
                label != "equivalent_valid_evidence"
                or relation["relation_role"] != "equivalent"
                or relation["support_scope"] in {None, "unrelated"}
            ):
                raise RuntimeError(f"invalid equivalent relation: {relation_id}")
            if relation_id in rejected_ids and relation["relation_role"] != "rejected":
                raise RuntimeError(f"invalid rejected relation: {relation_id}")
    if immutable_changes:
        raise RuntimeError(f"immutable record changes: {immutable_changes}")
    return {
        "required_claims": len(reviewed),
        "approved_claims": sum(row["adjudication_status"] == "approved" for row in reviewed),
        "candidate_relations": relation_count,
        "core_gold": core_count,
        "supporting_gold": supporting_count,
        "equivalent_valid_evidence": equivalent_count,
        "partially_relevant": label_counts["partially_relevant"],
        "insufficient": label_counts["insufficient"],
        "rejected_relations": len(reviewed) and sum(
            len(row["rejected_relations"]) for row in reviewed
        ),
        "multi_relation_minimum_sets": multi_sets,
        "no_valid_gold_evidence": no_valid,
        "immutable_changes": immutable_changes,
        "label_counts": dict(sorted(label_counts.items())),
        "relation_triples_valid": True,
        "source_hashes_valid": True,
    }


def validate_external_summary(summary: dict[str, Any], result: dict[str, Any]) -> None:
    expected = {
        "required_claims_total": result["required_claims"],
        "approved_claims": result["approved_claims"],
        "candidate_relations_total": result["candidate_relations"],
        "approved_core_relations": result["core_gold"],
        "approved_supporting_relations": result["supporting_gold"],
        "equivalent_non_gold_relations": result["equivalent_valid_evidence"],
        "claims_with_multi_relation_minimum_set": result["multi_relation_minimum_sets"],
        "claims_with_no_valid_gold_evidence": result["no_valid_gold_evidence"],
        "immutable_fields_changed": result["immutable_changes"],
    }
    mismatches = {
        key: {"expected": value, "actual": summary.get(key)}
        for key, value in expected.items()
        if summary.get(key) != value
    }
    for label in ("core_gold", "equivalent_valid_evidence", "partially_relevant", "insufficient"):
        if summary.get("adjudication_label_counts", {}).get(label) != result[label]:
            mismatches[f"adjudication_label_counts.{label}"] = {
                "expected": result[label],
                "actual": summary.get("adjudication_label_counts", {}).get(label),
            }
    if summary.get("historical_gold_modified") is not False:
        mismatches["historical_gold_modified"] = {
            "expected": False,
            "actual": summary.get("historical_gold_modified"),
        }
    if mismatches:
        raise RuntimeError(f"SUMMARY_MISMATCH: {json.dumps(mismatches, sort_keys=True)}")


def summarize(rows: list[dict[str, Any]], validation: dict[str, Any]) -> dict[str, Any]:
    per_question: dict[str, Counter[str]] = defaultdict(Counter)
    per_paper: dict[str, Counter[str]] = defaultdict(Counter)
    per_role: dict[str, Counter[str]] = defaultdict(Counter)
    candidate_distribution: Counter[int] = Counter()
    core_set_distribution: Counter[int] = Counter()
    detail_rows: list[dict[str, Any]] = []
    inherited_retained = inherited_rejected = equivalent_non_historical = 0
    single_core = multi_core = with_supporting = with_equivalent = ambiguous = 0
    for row in rows:
        relation_map = _relation_map(row)
        core = core_relation_ids(row)
        supporting = set(row["approved_supporting_relations"])
        equivalent = set(row["equivalent_non_gold_relations"])
        rejected = set(row["rejected_relations"])
        candidate_distribution[len(relation_map)] += 1
        valid_set = core or equivalent
        core_set_distribution[len(valid_set)] += 1
        single_core += len(core) == 1
        multi_core += len(valid_set) > 1
        with_supporting += bool(supporting)
        with_equivalent += bool(equivalent)
        ambiguous += any(
            relation["adjudication_label"] == "gold_claim_ambiguous"
            for relation in relation_map.values()
        )
        for relation_id, relation in relation_map.items():
            if relation["source_question_gold"]:
                if relation_id in core or relation_id in supporting:
                    inherited_retained += 1
                elif relation_id in rejected:
                    inherited_rejected += 1
            if relation_id in equivalent and not relation["source_question_gold"]:
                equivalent_non_historical += 1
        for dimension in (
            per_question[row["question_id"]],
            per_role[row["claim_role"]],
        ):
            dimension.update(
                claims=1,
                core=len(core),
                supporting=len(supporting),
                equivalent=len(equivalent),
                rejected=len(rejected),
            )
        for paper_id in row["target_papers"]:
            per_paper[paper_id].update(
                claims=1, core=len(core), supporting=len(supporting), equivalent=len(equivalent)
            )
        detail_rows.append(
            {
                "question_id": row["question_id"],
                "required_claim_id": row["required_claim_id"],
                "claim_role": row["claim_role"],
                "candidate_count": len(relation_map),
                "core_count": len(core),
                "supporting_count": len(supporting),
                "equivalent_count": len(equivalent),
                "rejected_count": len(rejected),
                "minimum_complete_set_size": len(valid_set),
                "no_valid_gold_evidence": row["no_valid_gold_evidence"],
            }
        )
    return {
        "schema_version": "claim-evidence-gold-dev-summary-v1",
        "review_type": "AI-assisted manual claim-level Gold adjudication",
        "gold_version": GOLD_VERSION,
        "total_required_claims": len(rows),
        "approved_claims": validation["approved_claims"],
        "candidate_relations": validation["candidate_relations"],
        "single_core_claims": single_core,
        "multi_relation_core_set_claims": multi_core,
        "claims_with_supporting_relations": with_supporting,
        "claims_with_equivalent_evidence": with_equivalent,
        "no_valid_evidence_claims": validation["no_valid_gold_evidence"],
        "ambiguous_claims": ambiguous,
        "inherited_question_gold_retained": inherited_retained,
        "inherited_question_gold_rejected": inherited_rejected,
        "equivalent_non_historical_relations_approved": equivalent_non_historical,
        "label_counts": validation["label_counts"],
        "core_set_size_distribution": dict(sorted(core_set_distribution.items())),
        "candidate_count_distribution": dict(sorted(candidate_distribution.items())),
        "per_question": {key: dict(value) for key, value in sorted(per_question.items())},
        "per_paper": {key: dict(value) for key, value in sorted(per_paper.items())},
        "per_claim_role": {key: dict(value) for key, value in sorted(per_role.items())},
        "focus_questions": {
            key: dict(per_question[key]) for key in ("q001", "q004", "q015", "q019", "q050")
        },
        "records": detail_rows,
        "limitations": [
            "Fixed Dev scope: 27 required claims across 9 answerable questions.",
            (
                "AI-assisted manual claim-level Gold adjudication, "
                "not independent double-blind review."
            ),
            "Not Full-50 Gold, Production Gold, or general-corpus Gold.",
            "Equivalent valid evidence is separate from historical exact Gold.",
        ],
    }


def write_summary(summary: dict[str, Any]) -> None:
    SUMMARY_JSON.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with SUMMARY_CSV.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary["records"][0]))
        writer.writeheader()
        writer.writerows(summary["records"])
    SUMMARY_DOC.write_text(
        f"""# Claim-level Gold Dev v1 summary

- Review type: **AI-assisted manual claim-level Gold adjudication**
- Required claims: {summary['approved_claims']}/{summary['total_required_claims']}
- Candidate relations: {summary['candidate_relations']}
- Core Gold relations: {summary['label_counts'].get('core_gold', 0)}
- Supporting Gold relations: {summary['label_counts'].get('supporting_gold', 0)}
- Equivalent valid evidence: {summary['label_counts'].get('equivalent_valid_evidence', 0)}
- Multi-relation minimum sets: {summary['multi_relation_core_set_claims']}
- No-valid-evidence claims: {summary['no_valid_evidence_claims']}
- Inherited question Gold retained: {summary['inherited_question_gold_retained']}
- Inherited question Gold rejected: {summary['inherited_question_gold_rejected']}

The adjudication covers only the fixed Dev 27 required claims. It is not independent double-blind
human review, Full-50 Gold, Production Gold, or a general-corpus estimate. Historical question-level
Gold remains unchanged. Equivalent evidence is reported separately and never rewrites exact Gold.

## Focus questions

```json
{json.dumps(summary['focus_questions'], ensure_ascii=False, indent=2)}
```
""",
        encoding="utf-8",
    )


def write_freeze(rows: list[dict[str, Any]], validation: dict[str, Any]) -> dict[str, Any]:
    if FREEZE_JSON.exists():
        raise RuntimeError("claim Gold freeze already exists; automatic overwrite forbidden")
    source_hashes = rows[0]["source_hashes"]
    reviewed_hash = hash_with_metadata(OUTPUT, "canonical_jsonl_v1")
    freeze = {
        "schema_version": "claim-evidence-gold-freeze-v1",
        "gold_version": GOLD_VERSION,
        "claim_gold_schema_version": SCHEMA_VERSION,
        "review_type": "AI-assisted manual claim-level Gold adjudication",
        "reviewed_record_count": len(rows),
        "relation_count": validation["candidate_relations"],
        "core_relation_count": validation["core_gold"],
        "supporting_relation_count": validation["supporting_gold"],
        "equivalent_relation_count": validation["equivalent_valid_evidence"],
        "partially_relevant_count": validation["partially_relevant"],
        "insufficient_count": validation["insufficient"],
        "rejected_relation_count": validation["rejected_relations"],
        "core_set_count": validation["multi_relation_minimum_sets"],
        "no_valid_evidence_count": validation["no_valid_gold_evidence"],
        "reviewed_file_hash": reviewed_hash,
        "source_corpus_hash": source_hashes["evidence_corpus"],
        "claim_units_hash": source_hashes["claim_units"],
        "historical_gold_hash": source_hashes["gold_set"],
        "retrieval_gold_hash": source_hashes["retrieval_gold"],
        "frozen_at": datetime.now(UTC).isoformat(),
        "frozen_before_next_live_run": True,
        "scope": "fixed_dev_27_required_claims",
        "external_validity_limitation": (
            "AI-assisted manual claim-level Gold adjudication; not independent double-blind, "
            "Full-50, Production, or general-corpus Gold."
        ),
        "automatic_overwrite_allowed": False,
    }
    FREEZE_JSON.write_text(
        json.dumps(freeze, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    FREEZE_DOC.write_text(
        f"""# Claim-level Gold Dev v1 freeze

- Gold version: `{GOLD_VERSION}`
- Schema version: `{SCHEMA_VERSION}`
- Records: {len(rows)}
- Candidate relations: {validation['candidate_relations']}
- Reviewed file canonical SHA-256: `{reviewed_hash['value']}`
- Frozen before next live run: `true`
- Scope: `fixed_dev_27_required_claims`

This freeze is an AI-assisted manual claim-level Gold adjudication. It is not independent
double-blind review and cannot be extrapolated to Full-50, Production, or the general corpus.
Automatic overwrite is forbidden.
""",
        encoding="utf-8",
    )
    return freeze


def main() -> None:
    args = parse_args()
    import_root = extract_package(args.input)
    pending = read_jsonl(OUTPUT)
    reviewed_path = import_root / "claim-evidence-gold-dev-v1-reviewed.jsonl"
    reviewed = read_jsonl(reviewed_path)
    validation = validate_review(pending, reviewed)
    external_summary = json.loads(
        (import_root / "stage13-10-human-claim-gold-review-summary.json").read_text(
            encoding="utf-8"
        )
    )
    validate_external_summary(external_summary, validation)
    if args.validate_only:
        print(json.dumps(validation, indent=2))
        return
    if all(row["adjudication_status"] == "approved" for row in pending):
        raise RuntimeError("reviewed claim Gold is already imported")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    backup = OUTPUT.with_name(f"{OUTPUT.name}.pre-human-import.{stamp}.bak")
    shutil.copy2(OUTPUT, backup)
    backup_hash = sha256(backup)
    shutil.copy2(reviewed_path, OUTPUT)
    imported = read_jsonl(OUTPUT)
    post_validation = validate_review(pending, imported)
    summary = summarize(imported, post_validation)
    write_summary(summary)
    freeze = write_freeze(imported, post_validation)
    print(
        json.dumps(
            {
                "status": "CLAIM_GOLD_REVIEW_IMPORTED",
                "validation": post_validation,
                "backup": str(backup),
                "backup_sha256": backup_hash,
                "freeze_hash": freeze["reviewed_file_hash"]["value"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
