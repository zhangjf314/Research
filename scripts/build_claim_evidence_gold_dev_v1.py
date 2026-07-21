"""Build the offline Stage 13.10 claim-level Gold adjudication pack."""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from paper_research.evaluation.canonical_hash import hash_with_metadata

try:
    from scripts.evidence_qa_dev_lib_v1 import canonical_hash, read_jsonl
except ModuleNotFoundError:
    from evidence_qa_dev_lib_v1 import canonical_hash, read_jsonl  # type: ignore[no-redef]

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/evaluation"
DOCS = ROOT / "docs"
ARTIFACTS = ROOT / "artifacts"
OUTPUT = DATA / "claim-evidence-gold-dev-v1.jsonl"
SCHEMA = DATA / "claim-evidence-gold-dev-v1.schema.json"
AUDIT_DOC = DOCS / "claim-evidence-gold-dev-v1.md"
GUIDE = DOCS / "claim-gold-review-guide-v1.md"
PACK = ARTIFACTS / "stage13-10-claim-gold-review-pack.zip"
SCHEMA_VERSION = "claim-evidence-gold-dev-schema-v1"
GOLD_VERSION = "claim-evidence-gold-dev-v1"
MAX_CANDIDATES = 12

SOURCE_FILES = {
    "claim_units": DATA / "claim-units-v1.jsonl",
    "evidence_corpus": DATA / "evidence-corpus-v1.jsonl",
    "gold_set": DATA / "gold-set-v1.jsonl",
    "retrieval_gold": DATA / "retrieval-gold-v2.jsonl",
    "gold_relation_audit": DATA / "citation-recall-gold-relation-audit-v1.jsonl",
    "dev_v2_summary": DATA / "evidence-qa-dev-v2.json",
    "dev_v3_1_summary": DATA / "evidence-qa-dev-v3-1.json",
    "dev_v2_citation_audit": DATA / "evidence-qa-dev-v2-citation-audit-v1.jsonl",
    "dev_v3_1_citation_audit": DATA / "evidence-qa-dev-v3-1-citation-audit-v1.jsonl",
}

MUTABLE_RECORD_FIELDS = {
    "approved_core_relations",
    "approved_supporting_relations",
    "equivalent_non_gold_relations",
    "rejected_relations",
    "no_valid_gold_evidence",
    "adjudication_status",
    "reviewer",
    "reviewed_at",
    "review_notes",
}
MUTABLE_RELATION_FIELDS = {
    "relation_role",
    "support_scope",
    "adjudication_label",
    "adjudication_notes",
}


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 2
    }


def immutable_payload(row: dict[str, Any]) -> dict[str, Any]:
    payload = {key: value for key, value in row.items() if key not in MUTABLE_RECORD_FIELDS}
    payload.pop("immutable_record_hash", None)
    payload["candidate_evidence_relations"] = [
        {key: value for key, value in relation.items() if key not in MUTABLE_RELATION_FIELDS}
        for relation in row["candidate_evidence_relations"]
    ]
    return payload


def relation_id(question_id: str, claim_id: str, triple: tuple[str, int, str]) -> str:
    digest = hashlib.sha256(
        f"{question_id}|{claim_id}|{triple[0]}|{triple[1]}|{triple[2]}".encode()
    ).hexdigest()[:20]
    return f"rel-{question_id}-{digest}"


def _selected_triples(version: str) -> tuple[set[tuple[str, int, str]], set[tuple[str, int, str]]]:
    summary = json.loads((DATA / f"{version}.json").read_text(encoding="utf-8"))
    selected: set[tuple[str, int, str]] = set()
    adjacent: set[tuple[str, int, str]] = set()
    for run_id in summary["selected_runs"]:
        trace = json.loads(
            (DATA / version / "runs" / run_id / "retrieval-trace.json").read_text(
                encoding="utf-8"
            )
        )
        selected.update(
            (item[0], int(item[1]), item[2])
            for item in trace["allowed_citation_triples"]
        )
        adjacent.update(
            (item["paper_id"], int(item["page"]), item["block_id"])
            for item in trace["adjacent_completion_blocks"]
        )
    return selected, adjacent


def _claim_for_v2(row: dict[str, Any]) -> str | None:
    best = (row.get("required_claim_match") or {}).get("best") or {}
    return best.get("required_claim_id")


def build_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    claims_all = {row["claim_id"]: row for row in read_jsonl(SOURCE_FILES["claim_units"])}
    historical = read_jsonl(SOURCE_FILES["gold_relation_audit"])
    claim_ids = sorted({row["required_claim_id"] for row in historical})
    if len(claim_ids) != 27:
        raise RuntimeError(f"expected 27 frozen Dev claims, got {len(claim_ids)}")
    evidence_rows = read_jsonl(SOURCE_FILES["evidence_corpus"])
    evidence = {
        (row["paper_id"], int(row["page"]), row["block_id"]): row for row in evidence_rows
    }
    by_paper_page: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for unit in evidence_rows:
        by_paper_page[(unit["paper_id"], int(unit["page"]))].append(unit)

    historical_by_claim: dict[str, list[tuple[str, int, str]]] = defaultdict(list)
    for row in historical:
        triple = (row["paper_id"], int(row["page"]), row["block_id"])
        if triple not in evidence:
            raise RuntimeError(f"historical Gold triple missing: {triple}")
        historical_by_claim[row["required_claim_id"]].append(triple)

    audits: dict[str, dict[str, list[dict[str, Any]]]] = {
        "v2": defaultdict(list),
        "v3_1": defaultdict(list),
    }
    for row in read_jsonl(SOURCE_FILES["dev_v2_citation_audit"]):
        claim_id = _claim_for_v2(row)
        if claim_id in claim_ids:
            audits["v2"][claim_id].append(row)
    for row in read_jsonl(SOURCE_FILES["dev_v3_1_citation_audit"]):
        if row["required_claim_id"] in claim_ids:
            audits["v3_1"][row["required_claim_id"]].append(row)
    selected_v2, adjacent_v2 = _selected_triples("evidence-qa-dev-v2")
    selected_v3, adjacent_v3 = _selected_triples("evidence-qa-dev-v3-1")

    source_hashes = {
        name: hash_with_metadata(
            path, "canonical_jsonl_v1" if path.suffix == ".jsonl" else "canonical_json_v1"
        )
        for name, path in SOURCE_FILES.items()
    }
    output: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    before_dedupe = 0
    for claim_id in claim_ids:
        claim = claims_all[claim_id]
        question_id = claim["question_id"]
        inherited = list(dict.fromkeys(historical_by_claim[claim_id]))
        candidate_meta: dict[tuple[str, int, str], dict[str, Any]] = {}

        def add(
            triple: tuple[str, int, str],
            provenance: str,
            priority: int,
            candidate_store: dict[tuple[str, int, str], dict[str, Any]] = candidate_meta,
        ) -> None:
            nonlocal before_dedupe
            before_dedupe += 1
            if triple not in evidence:
                raise RuntimeError(f"candidate triple missing: {triple}")
            item = candidate_store.setdefault(
                triple, {"priority": priority, "provenance": set()}
            )
            item["priority"] = min(item["priority"], priority)
            item["provenance"].add(provenance)

        for triple in inherited:
            add(triple, "inherited_question_gold", 0)
        for version, rows_by_claim in audits.items():
            for audit in rows_by_claim.get(claim_id, []):
                triple_value = audit["citation_triple"]
                triple = (
                    triple_value["paper_id"],
                    int(triple_value["page"]),
                    triple_value["block_id"],
                )
                priority = (
                    1
                    if audit["human_label"] in {"fully_supported", "partially_supported"}
                    else 2
                )
                add(triple, f"{version}_cited", priority)
                if audit["human_label"] in {"fully_supported", "partially_supported"}:
                    add(triple, f"{version}_human_supported", 1)

        claim_tokens = _tokens(claim["claim_text"])
        lexical: list[tuple[float, tuple[str, int, str]]] = []
        for paper_id in claim["target_paper_ids"]:
            for page in claim["gold_pages"]:
                for unit in by_paper_page.get((paper_id, int(page)), []):
                    triple = (paper_id, int(page), unit["block_id"])
                    score = len(claim_tokens & _tokens(unit["text"])) / max(len(claim_tokens), 1)
                    if score:
                        lexical.append((score, triple))
        for _, triple in sorted(lexical, key=lambda item: (-item[0], item[1]))[:6]:
            add(triple, "same_gold_page_lexical", 4)

        for triple in list(candidate_meta):
            unit = evidence[triple]
            for neighbor_id in (unit.get("previous_block_id"), unit.get("next_block_id")):
                if not neighbor_id:
                    continue
                neighbor = (triple[0], triple[1], neighbor_id)
                if neighbor in evidence:
                    add(neighbor, "same_page_adjacent", 5)

        ordered = sorted(
            candidate_meta,
            key=lambda triple: (candidate_meta[triple]["priority"], triple),
        )
        retained = list(inherited)
        retained.extend(triple for triple in ordered if triple not in retained)
        if len(retained) > MAX_CANDIDATES:
            stats["claims_over_candidate_cap_before_truncation"] += 1
        retained = retained[: max(MAX_CANDIDATES, len(inherited))]
        relations: list[dict[str, Any]] = []
        for triple in retained:
            unit = evidence[triple]
            labels = sorted(
                {
                    audit["human_label"]
                    for version in audits.values()
                    for audit in version.get(claim_id, [])
                    if (
                        audit["citation_triple"]["paper_id"],
                        int(audit["citation_triple"]["page"]),
                        audit["citation_triple"]["block_id"],
                    )
                    == triple
                }
            )
            previous = evidence.get((triple[0], triple[1], unit.get("previous_block_id")))
            following = evidence.get((triple[0], triple[1], unit.get("next_block_id")))
            relation = {
                "relation_id": relation_id(question_id, claim_id, triple),
                "paper_id": triple[0],
                "page": triple[1],
                "block_id": triple[2],
                "evidence_text": unit["text"],
                "neighboring_context": {
                    "previous": previous["text"] if previous else None,
                    "next": following["text"] if following else None,
                },
                "block_type": unit["block_type"],
                "relation_role": None,
                "support_scope": None,
                "provenance": sorted(candidate_meta[triple]["provenance"]),
                "source_question_gold": triple in inherited,
                "retrieved_in_dev_v2": triple in selected_v2,
                "selected_in_dev_v2": triple in selected_v2,
                "cited_in_dev_v2": any(
                    (
                        row["citation_triple"]["paper_id"],
                        int(row["citation_triple"]["page"]),
                        row["citation_triple"]["block_id"],
                    )
                    == triple
                    for row in audits["v2"].get(claim_id, [])
                ),
                "retrieved_in_dev_v3_1": triple in selected_v3,
                "selected_in_dev_v3_1": triple in selected_v3,
                "cited_in_dev_v3_1": any(
                    (
                        row["citation_triple"]["paper_id"],
                        int(row["citation_triple"]["page"]),
                        row["citation_triple"]["block_id"],
                    )
                    == triple
                    for row in audits["v3_1"].get(claim_id, [])
                ),
                "adjacent_in_dev_v2": triple in adjacent_v2,
                "adjacent_in_dev_v3_1": triple in adjacent_v3,
                "human_citation_support_labels": labels,
                "adjudication_label": None,
                "adjudication_notes": None,
            }
            relations.append(relation)
            stats["historical_gold_candidates"] += triple in inherited
            stats["dev_v2_cited_candidates"] += relation["cited_in_dev_v2"]
            stats["dev_v3_1_cited_candidates"] += relation["cited_in_dev_v3_1"]
            stats["human_supported_candidates"] += bool(
                {"fully_supported", "partially_supported"} & set(labels)
            )
            stats["adjacent_candidates"] += (
                relation["adjacent_in_dev_v2"] or relation["adjacent_in_dev_v3_1"]
            )
        row = {
            "record_id": f"claim-gold-{claim_id}",
            "question_id": question_id,
            "required_claim_id": claim_id,
            "required_claim_text": claim["claim_text"],
            "claim_role": claim["claim_role"],
            "answerable": bool(claim["expected_answerability"]),
            "target_papers": claim["target_paper_ids"],
            "inherited_question_gold_blocks": claim["gold_block_ids"],
            "inherited_question_gold_pages": claim["gold_pages"],
            "candidate_evidence_relations": relations,
            "approved_core_relations": [],
            "approved_supporting_relations": [],
            "equivalent_non_gold_relations": [],
            "rejected_relations": [],
            "no_valid_gold_evidence": False,
            "adjudication_status": "pending",
            "reviewer": None,
            "reviewed_at": None,
            "review_notes": None,
            "source_hashes": source_hashes,
            "source_record_hash": canonical_hash(claim),
            "immutable_record_hash": None,
            "schema_version": SCHEMA_VERSION,
            "gold_version": GOLD_VERSION,
        }
        row["immutable_record_hash"] = canonical_hash(immutable_payload(row))
        output.append(row)
    stats["candidate_relations"] = sum(len(row["candidate_evidence_relations"]) for row in output)
    stats["deduplicated_candidates"] = before_dedupe - stats["candidate_relations"]
    stats["pending"] = len(output)
    return output, dict(stats)


def schema_document() -> dict[str, Any]:
    relation_roles = ["core", "supporting", "equivalent", "rejected"]
    support_scopes = [
        "fully_supports_claim", "partially_supports_claim", "supports_subclaim",
        "supports_method", "supports_result", "supports_limitation",
        "supports_comparison_side_a", "supports_comparison_side_b", "numeric_support",
        "contextual_only", "unrelated",
    ]
    labels = [
        "core_gold", "supporting_gold", "equivalent_valid_evidence",
        "partially_relevant", "insufficient", "unrelated", "malformed_evidence",
        "gold_claim_ambiguous", "no_valid_evidence",
    ]
    relation = {
        "type": "object",
        "required": [
            "relation_id", "paper_id", "page", "block_id", "evidence_text",
            "neighboring_context", "block_type", "relation_role", "support_scope",
            "provenance", "source_question_gold", "retrieved_in_dev_v2",
            "selected_in_dev_v2", "cited_in_dev_v2", "retrieved_in_dev_v3_1",
            "selected_in_dev_v3_1", "cited_in_dev_v3_1",
            "human_citation_support_labels", "adjudication_label", "adjudication_notes",
        ],
        "properties": {
            "relation_id": {"type": "string"},
            "paper_id": {"type": "string"},
            "page": {"type": "integer", "minimum": 1},
            "block_id": {"type": "string"},
            "evidence_text": {"type": "string"},
            "neighboring_context": {"type": "object"},
            "block_type": {"type": "string"},
            "relation_role": {"type": ["string", "null"], "enum": relation_roles + [None]},
            "support_scope": {"type": ["string", "null"], "enum": support_scopes + [None]},
            "provenance": {"type": "array", "items": {"type": "string"}},
            "source_question_gold": {"type": "boolean"},
            "retrieved_in_dev_v2": {"type": "boolean"},
            "selected_in_dev_v2": {"type": "boolean"},
            "cited_in_dev_v2": {"type": "boolean"},
            "retrieved_in_dev_v3_1": {"type": "boolean"},
            "selected_in_dev_v3_1": {"type": "boolean"},
            "cited_in_dev_v3_1": {"type": "boolean"},
            "human_citation_support_labels": {"type": "array", "items": {"type": "string"}},
            "adjudication_label": {"type": ["string", "null"], "enum": labels + [None]},
            "adjudication_notes": {"type": ["string", "null"]},
        },
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": SCHEMA_VERSION,
        "type": "object",
        "required": [
            "record_id", "question_id", "required_claim_id", "required_claim_text",
            "claim_role", "answerable", "target_papers", "inherited_question_gold_blocks",
            "inherited_question_gold_pages", "candidate_evidence_relations",
            "approved_core_relations", "approved_supporting_relations",
            "equivalent_non_gold_relations", "rejected_relations",
            "no_valid_gold_evidence", "adjudication_status", "reviewer", "reviewed_at",
            "review_notes", "source_hashes", "source_record_hash",
            "immutable_record_hash", "schema_version", "gold_version",
        ],
        "properties": {
            "candidate_evidence_relations": {"type": "array", "items": relation},
            "adjudication_status": {"enum": ["pending", "approved"]},
            "approved_core_relations": {"type": "array"},
            "approved_supporting_relations": {"type": "array", "items": {"type": "string"}},
            "equivalent_non_gold_relations": {"type": "array", "items": {"type": "string"}},
            "rejected_relations": {"type": "array", "items": {"type": "string"}},
            "no_valid_gold_evidence": {"type": "boolean"},
        },
        "additionalProperties": True,
    }


def write_outputs(rows: list[dict[str, Any]], stats: dict[str, Any]) -> None:
    OUTPUT.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    SCHEMA.write_text(
        json.dumps(schema_document(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    distribution = Counter(len(row["candidate_evidence_relations"]) for row in rows)
    AUDIT_DOC.write_text(
        f"""# Claim-level Gold Dev v1

Stage 13.10 Phase A created a human-only adjudication layer for the frozen 27
answerable Dev required claims. It does not modify question-level Gold or any historical metric.

- Schema: `{SCHEMA_VERSION}`
- Gold version: `{GOLD_VERSION}`
- Required claims: {len(rows)}
- Candidate relations: {stats['candidate_relations']}
- Candidate count distribution: {dict(sorted(distribution.items()))}
- Historical question-level Gold candidates retained: {stats['historical_gold_candidates']}
- Pending adjudications: {stats['pending']}
- Candidate cap exceeded before truncation:
  {stats.get('claims_over_candidate_cap_before_truncation', 0)}

Candidate provenance is diagnostic only. Retrieval hits, model citations, and prior human citation
support labels are never converted automatically into claim-level Gold. Multi-block claims may be
approved as a `minimum_complete_set` containing multiple relation IDs. Equivalent valid evidence
remains separate from historical exact Gold.

`READY_FOR_HUMAN_CLAIM_GOLD_ADJUDICATION=true`

`WAITING_FOR_EXTERNAL_CLAIM_GOLD_REVIEW`

`READY_FOR_DEV_V3_2=false`
""",
        encoding="utf-8",
    )
    GUIDE.write_text(
        """# Claim Gold review guide v1

Review every required claim independently. Candidate provenance and automated/human citation
signals are context, not Gold labels.

Use `core_gold` for a relation that alone supports the claim, or place all relations needed for a
minimum complete multi-block set in one core set. Use `supporting_gold` only when the block belongs
to a complete evidence set but is not sufficient alone. Use `equivalent_valid_evidence` for valid
evidence outside historical question-level Gold; it does not rewrite historical exact Gold.
`partially_relevant`, `insufficient`, and `unrelated` are not formal Gold. If the corpus contains no
valid evidence, select `no_valid_evidence`; do not force a block.

Pay special attention to q001 decomposition, q004's GPU/Adam/warmup set, q015 paper identity versus
future work, q019 numeric and experimental subclaims, and q050 comparison-side completeness.

Example:

```
python scripts/review_claim_evidence_gold_dev_v1.py --required-claim-id <id> \
  --approve-core <relation-id> --reviewer <name> --notes "<reason>"
```

Repeat `--approve-core` to create one multi-relation minimum complete core set. The tool always
backs up the JSONL before a write and never approves a record without an explicit human action.
""",
        encoding="utf-8",
    )
    members = [
        OUTPUT, SOURCE_FILES["evidence_corpus"], SOURCE_FILES["claim_units"],
        SOURCE_FILES["gold_set"], SOURCE_FILES["retrieval_gold"],
        SOURCE_FILES["dev_v2_summary"], SOURCE_FILES["dev_v3_1_summary"],
        SOURCE_FILES["dev_v2_citation_audit"], SOURCE_FILES["dev_v3_1_citation_audit"],
        SOURCE_FILES["gold_relation_audit"], GUIDE,
    ]
    PACK.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(PACK, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in members:
            info = zipfile.ZipInfo(path.relative_to(ROOT).as_posix(), (2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())


def main() -> None:
    rows, stats = build_rows()
    write_outputs(rows, stats)
    print(json.dumps({"records": len(rows), **stats, "pack": str(PACK)}, indent=2))


if __name__ == "__main__":
    main()
