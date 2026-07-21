# ruff: noqa: E501
"""Prepare the offline Stage 13.9 recall audit and external human-review pack."""

from __future__ import annotations

import csv
import hashlib
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any
from urllib.parse import urlparse

from paper_research.config import Settings
from paper_research.evaluation.canonical_hash import hash_with_metadata

try:
    from scripts.evidence_qa_dev_lib_v1 import DATA, DEV_IDS, DOCS, canonical_hash, read_jsonl
    from scripts.evidence_qa_dev_v3_1_lib import RUN_ROOT
except ModuleNotFoundError:
    from evidence_qa_dev_lib_v1 import (  # type: ignore[no-redef]
        DATA,
        DEV_IDS,
        DOCS,
        canonical_hash,
        read_jsonl,
    )
    from evidence_qa_dev_v3_1_lib import RUN_ROOT  # type: ignore[no-redef]

ROOT = DATA.parents[1]
SUMMARY = DATA / "evidence-qa-dev-v3-1.json"
CITATION_AUDIT = DATA / "evidence-qa-dev-v3-1-citation-audit-v1.jsonl"
RECALL_JSONL = DATA / "dev-v3-1-citation-recall-audit-v1.jsonl"
RECALL_CSV = DATA / "dev-v3-1-citation-recall-audit-v1.csv"
RECALL_DOC = DOCS / "dev-v3-1-citation-recall-audit-v1.md"
GUIDE = DOCS / "evidence-qa-dev-v3-1-citation-review-guide-v1.md"
PACK = ROOT / "artifacts/stage13-9-dev-v3-1-citation-review-pack.zip"
LABELS = {
    "fully_supported", "partially_supported", "related_but_insufficient",
    "unsupported", "gold_annotation_too_narrow", "ambiguous_claim",
    "malformed_evidence",
}
HUMAN_FIELDS = {
    "human_review_status", "human_label", "reviewer", "reviewed_at", "review_notes",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_citation_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != 33 or len({row["sample_id"] for row in rows}) != 33:
        raise RuntimeError("expected 33 unique citation audit samples")
    evidence_path = DATA / "evidence-corpus-v1.jsonl"
    evidence = {
        (row["paper_id"], int(row["page"]), row["block_id"]): row
        for row in read_jsonl(evidence_path)
    }
    source = hash_with_metadata(evidence_path, "canonical_jsonl_v1")
    run_registries: dict[str, dict[str, Any]] = {}
    for row in rows:
        required = {
            "sample_id", "question_id", "required_claim_id", "generated_claim",
            "citation_id", "citation_triple", "cited_evidence_text",
            "cited_evidence_context", "evidence_source", "gold_blocks", "gold_pages",
            "automated_signal", "registry_hash", "source_hash", "source_record_hash",
            "immutable_record_hash", "human_review_status", "human_label",
        }
        if not required <= set(row):
            raise RuntimeError(f"citation audit schema incomplete: {row['sample_id']}")
        status = row["human_review_status"]
        if status not in {"pending", "approved"}:
            raise RuntimeError(f"invalid review status: {row['sample_id']}")
        if status == "pending" and (
            row["human_label"] is not None
            or any(
                row.get(field) is not None
                for field in {"reviewer", "reviewed_at", "review_notes"}
            )
        ):
            raise RuntimeError(f"pending review metadata populated: {row['sample_id']}")
        if status == "approved" and (
            row["human_label"] not in LABELS
            or not row.get("reviewer")
            or not row.get("reviewed_at")
            or not row.get("review_notes")
        ):
            raise RuntimeError(f"approved review metadata incomplete: {row['sample_id']}")
        triple = row["citation_triple"]
        key = (triple["paper_id"], int(triple["page"]), triple["block_id"])
        unit = evidence.get(key)
        if unit is None or row["source_record_hash"] != canonical_hash(unit):
            raise RuntimeError(f"citation triple/source record mismatch: {row['sample_id']}")
        if row["source_canonical_sha256"] != source["value"]:
            raise RuntimeError(f"citation source hash changed: {row['sample_id']}")
        immutable = {
            key: value for key, value in row.items()
            if key not in HUMAN_FIELDS | {"immutable_record_hash"}
        }
        if row["immutable_record_hash"] != canonical_hash(immutable):
            raise RuntimeError(f"immutable audit fields changed: {row['sample_id']}")
        registry = run_registries.setdefault(
            row["run_id"],
            json.loads((RUN_ROOT / row["run_id"] / "citation-registry.json").read_text(encoding="utf-8")),
        )
        entry = next((item for item in registry["entries"] if item["citation_id"] == row["citation_id"]), None)
        if entry is None or any(entry[field] != triple[field] for field in ("paper_id", "page", "block_id")):
            raise RuntimeError(f"registry triple mismatch: {row['sample_id']}")
        if registry["registry_hash"] != row["registry_hash"]:
            raise RuntimeError(f"registry hash mismatch: {row['sample_id']}")
    return {
        "records": len(rows),
        "unique_sample_ids": 33,
        "pending": sum(row["human_review_status"] == "pending" for row in rows),
        "approved": sum(row["human_review_status"] == "approved" for row in rows),
        "source_hash_valid": True, "source_record_hash_valid": True,
        "immutable_hash_valid": True, "registry_hash_valid": True,
        "citation_triples_valid": True,
    }


def build_claim_rows(summary: dict[str, Any], audit: list[dict[str, Any]]) -> list[dict[str, Any]]:
    audit_by_claim: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in audit:
        audit_by_claim[(row["question_id"], row["required_claim_id"])].append(row)
    gold = {row["question_id"]: row for row in read_jsonl(DATA / "gold-set-v1.jsonl")}
    output: list[dict[str, Any]] = []
    for run_id in summary["selected_runs"]:
        result = json.loads((RUN_ROOT / run_id / "result.json").read_text(encoding="utf-8"))
        question_id = result["question_id"]
        if not gold[question_id]["answerable"]:
            continue
        for slot in result["answer"]["required_claim_results"]:
            selected_ids = list(slot["citation_ids"])
            citations = audit_by_claim[(question_id, slot["required_claim_id"])]
            triples = [row["citation_triple"] for row in citations]
            gold_blocks = list(gold[question_id]["gold_block_ids"])
            gold_pages = list(gold[question_id]["gold_pages"])
            exact_hits = sorted({
                triple["block_id"] for triple in triples
                if triple["paper_id"] in gold[question_id]["gold_paper_ids"]
                and triple["page"] in gold_pages and triple["block_id"] in gold_blocks
            })
            page_hits = [triple for triple in triples if triple["paper_id"] in gold[question_id]["gold_paper_ids"] and triple["page"] in gold_pages]
            duplicates = sorted(citation_id for citation_id, count in Counter(selected_ids).items() if count > 1)
            denominator = len(gold_blocks)
            output.append({
                "question_id": question_id,
                "required_claim_id": slot["required_claim_id"],
                "status": slot["status"],
                "generated_claim": slot["claim_text"],
                "selected_citation_ids": selected_ids,
                "citation_triples": triples,
                "gold_blocks": gold_blocks,
                "gold_pages": gold_pages,
                "gold_block_count": denominator,
                "exact_gold_hits": exact_hits,
                "page_hits": page_hits,
                "duplicate_citations": duplicates,
                "recall_numerator": len(exact_hits),
                "recall_denominator": denominator,
                "recall_contribution": len(exact_hits) / denominator if denominator else None,
                "aggregation_weight": {"claim_macro": 1 / 27, "micro_gold_block": denominator},
                "included_in_macro": bool(denominator),
                "included_in_micro": bool(denominator),
                "exclusion_reason": None if denominator else "empty_gold_blocks",
                "empty_gold_handling": "not_empty" if denominator else "excluded_and_reported",
                "automated_support_signal": [row["automated_signal"] for row in citations],
                "human_citation_labels": [row["human_label"] for row in citations],
            })
    if len(output) != 27:
        raise RuntimeError(f"expected 27 required-claim rows, got {len(output)}")
    return output


def aggregate(summary: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    per_query = {row["question_id"]: row for row in summary["per_query"]}
    per_question_recall = {
        qid: float(per_query[qid]["citation_recall"]) for qid in DEV_IDS
    }
    formal_values = [per_question_recall[qid] for qid in DEV_IDS]
    answerable_ids = [qid for qid in DEV_IDS if per_query[qid]["required_claim_denominator"] > 0]
    answerable_values = [float(per_query[qid]["citation_recall"]) for qid in answerable_ids]
    claim_values = [float(row["recall_contribution"]) for row in rows if row["included_in_macro"]]
    micro_num = len({
        (row["question_id"], block)
        for row in rows
        for block in row["exact_gold_hits"]
    })
    gold_by_question = {
        (row["question_id"], block) for row in rows for block in row["gold_blocks"]
    }
    v2 = json.loads((DATA / "evidence-qa-dev-v2.json").read_text(encoding="utf-8"))
    v2_rows = [
        row for row in v2["comparison"]["per_query"]
        if row["status"] == "completed"
        and row["question_id"] in answerable_ids
        and row["dev_v2"].get("citation_recall") is not None
    ]
    v2_values = [float(row["dev_v2"]["citation_recall"]) for row in v2_rows]
    citation_claims: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        for citation_id in row["selected_citation_ids"]:
            citation_claims[(row["question_id"], citation_id)].add(row["required_claim_id"])
    shared = [
        {
            "question_id": key[0],
            "citation_id": key[1],
            "required_claim_ids": sorted(claim_ids),
        }
        for key, claim_ids in sorted(citation_claims.items())
        if len(claim_ids) > 1
    ]
    formal = mean(formal_values)
    v2_reconstructed = mean(v2_values)
    return {
        "dev_v3_1_formal": {
            "formula": "mean(per_question_exact_block_recall for fixed 10 questions; unanswerable q005 contributes refusal_correct=1)",
            "values": formal_values,
            "numerator_sum": sum(formal_values),
            "denominator_questions": 10,
            "value": formal,
            "per_question": per_question_recall,
        },
        "macro_question_recall_answerable_only_diagnostic": {
            "formula": "mean(per_question_exact_block_recall for 9 answerable questions)",
            "question_ids": answerable_ids,
            "value": mean(answerable_values),
        },
        "macro_required_claim_recall_diagnostic": {
            "formula": "mean(unique exact cited gold blocks / question gold blocks for each of 27 required-claim slots)",
            "value": mean(claim_values),
        },
        "micro_question_gold_block_recall_diagnostic": {
            "formula": "unique (question_id, exact cited gold block) hits / unique (question_id, gold block) targets",
            "numerator": micro_num,
            "denominator": len(gold_by_question),
            "value": micro_num / len(gold_by_question),
        },
        "dev_v2_formal": {
            "formula": "mean(per_question exact citation recall over completed answerable questions only)",
            "question_ids": [row["question_id"] for row in v2_rows],
            "values": v2_values,
            "numerator_sum": sum(v2_values),
            "denominator_questions": len(v2_values),
            "value": v2_reconstructed,
        },
        "published_values_match": {
            "dev_v3_1": abs(formal - 0.295) < 1e-12,
            "dev_v2": abs(v2_reconstructed - 0.29583333333333334) < 1e-6,
        },
        "metric_protocols_comparable": False,
        "protocol_difference": "Dev v3.1 averages all fixed 10 questions and awards q005 refusal recall=1; Dev v2 averages only 8 completed answerable questions and excludes q005 plus failed q050.",
        "edge_case_audit": {
            "empty_gold_required_claims": sum(not row["gold_blocks"] for row in rows),
            "duplicate_citations_within_claim": sum(len(row["duplicate_citations"]) for row in rows),
            "shared_citations_across_required_claims": shared,
            "multiple_gold_blocks": "unique exact block IDs are the denominator; page-only hits do not count",
            "exact_vs_page": {
                "exact_hit_instances": sum(len(row["exact_gold_hits"]) for row in rows),
                "page_hit_instances": sum(len(row["page_hits"]) for row in rows),
            },
            "floating_point_or_serialization_changed_decision": False,
        },
        "decision": "CITATION_RECALL_METRIC_INCOMPARABLE",
        "formal_gate_modified": False,
        "calculation_bug_found": False,
    }


def write_outputs(rows: list[dict[str, Any]], metrics: dict[str, Any], validation: dict[str, Any]) -> None:
    RECALL_JSONL.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    with RECALL_CSV.open("w", encoding="utf-8", newline="") as stream:
        fieldnames = list(rows[0])
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (list, dict)) else value for key, value in row.items()})
    RECALL_DOC.write_text(
        "# Dev v3.1 Citation Recall Audit\n\n"
        f"- Required-claim rows: {len(rows)}\n"
        f"- Formal Dev v3.1 recall: {metrics['dev_v3_1_formal']['value']:.6f}\n"
        f"- Answerable-only macro question recall (diagnostic): {metrics['macro_question_recall_answerable_only_diagnostic']['value']:.6f}\n"
        f"- Macro required-claim recall (diagnostic): {metrics['macro_required_claim_recall_diagnostic']['value']:.6f}\n"
        f"- Micro question-gold-block recall (diagnostic): {metrics['micro_question_gold_block_recall_diagnostic']['numerator']}/{metrics['micro_question_gold_block_recall_diagnostic']['denominator']} = {metrics['micro_question_gold_block_recall_diagnostic']['value']:.6f}\n"
        f"- Dev v2 published-path reconstruction: {metrics['dev_v2_formal']['value']:.6f}\n"
        f"- Per-question formal recalls: `{json.dumps(metrics['dev_v3_1_formal']['per_question'], sort_keys=True)}`\n"
        f"- Duplicate citations within claim / shared citations across claims: {metrics['edge_case_audit']['duplicate_citations_within_claim']}/{len(metrics['edge_case_audit']['shared_citations_across_required_claims'])}\n"
        f"- Exact/page hit instances across claim rows: {metrics['edge_case_audit']['exact_vs_page']['exact_hit_instances']}/{metrics['edge_case_audit']['exact_vs_page']['page_hit_instances']}\n"
        "- Dev v3.1 formula: mean of all ten fixed per-question recalls; q005 refusal contributes 1 and q050 contributes 0.\n"
        "- Dev v2 formula: mean over eight completed answerable questions; q005 is excluded as unanswerable and failed q050 is excluded.\n"
        "- Decision: **CITATION_RECALL_METRIC_INCOMPARABLE**. The 0.295000 versus 0.295833 difference is caused by denominator/protocol differences, not float serialization or rounding.\n"
        "- Formal Stage 13.8 metric and gate were not changed.\n"
        f"- Citation audit schema/hash/triple validation: {validation}\n",
        encoding="utf-8",
    )
    GUIDE.write_text(
        "# Evidence QA Dev v3.1 Citation Review Guide\n\n"
        "Review all 33 rows independently. Do not infer a label from exact/page/semantic automated signals. Verify whether the cited evidence supports the generated claim, using previous/current/next context and the supplied Gold only as reference.\n\n"
        "Allowed labels:\n\n"
        "- `fully_supported`\n- `partially_supported`\n- `related_but_insufficient`\n"
        "- `unsupported`\n- `gold_annotation_too_narrow`\n- `ambiguous_claim`\n"
        "- `malformed_evidence`\n\n"
        "For every approved row fill `human_label`, `reviewer`, `reviewed_at`, and a non-empty `review_notes`, then set `human_review_status=approved`. Do not edit sample IDs, claims, citations, evidence, Gold, automated labels, source hashes, registry hashes, or immutable hashes.\n",
        encoding="utf-8",
    )


def build_pack() -> None:
    members = [
        CITATION_AUDIT,
        DATA / "evidence-corpus-v1.jsonl",
        DATA / "claim-units-v1.jsonl",
        DATA / "gold-set-v1.jsonl",
        DATA / "retrieval-gold-v2.jsonl",
        SUMMARY,
        RECALL_JSONL,
        GUIDE,
    ]
    PACK.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(PACK, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in members:
            info = zipfile.ZipInfo(path.name, date_time=(2026, 7, 15, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())
    with zipfile.ZipFile(PACK) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or len(names) != 8:
            raise RuntimeError("review pack member set invalid")
        forbidden = [name for name in names if any(token in name.lower() for token in (".env", ".sqlite", "raw-provider", "runs/"))]
        if forbidden:
            raise RuntimeError(f"forbidden review pack members: {forbidden}")
        settings = Settings()
        database_password = urlparse(settings.database_url).password or ""
        secrets = [settings.llm_api_key or "", settings.database_url or ""]
        if len(database_password) >= 12:
            secrets.append(database_password)
        for name in names:
            body = archive.read(name)
            if any(secret and secret.encode() in body for secret in secrets):
                raise RuntimeError(f"configured secret leaked into review pack: {name}")
            if b"Authorization: Bearer" in body or b'"Authorization"' in body:
                raise RuntimeError(f"authorization header leaked into review pack: {name}")


def main() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    if summary["metrics"]["all_manifest_conservative"]["citation_recall"] != 0.295:
        raise RuntimeError("Stage 13.8 formal citation recall changed")
    if summary["dev_v3_1_quality_candidate_gate"] or summary["ready_for_full_qa"]:
        raise RuntimeError("Stage 13.8 frozen gate changed")
    audit = read_jsonl(CITATION_AUDIT)
    validation = validate_citation_audit(audit)
    rows = build_claim_rows(summary, audit)
    metrics = aggregate(summary, rows)
    write_outputs(rows, metrics, validation)
    build_pack()
    print(json.dumps({
        "citation_audit": validation,
        "required_claim_rows": len(rows),
        "formal_recall": metrics["dev_v3_1_formal"]["value"],
        "macro_question_answerable": metrics["macro_question_recall_answerable_only_diagnostic"]["value"],
        "macro_claim": metrics["macro_required_claim_recall_diagnostic"]["value"],
        "micro_block": metrics["micro_question_gold_block_recall_diagnostic"]["value"],
        "dev_v2_reconstructed": metrics["dev_v2_formal"]["value"],
        "decision": metrics["decision"],
        "review_pack": str(PACK.relative_to(ROOT)).replace("\\", "/"),
        "review_pack_sha256": sha256(PACK),
        "ready_for_human_audit": True,
    }))


if __name__ == "__main__":
    main()
