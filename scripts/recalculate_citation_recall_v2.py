# ruff: noqa: E501
"""Freeze citation-recall-v2 and recalculate historical Dev experiments offline."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any

try:
    from scripts.evidence_qa_dev_lib_v1 import DATA, DEV_IDS, DOCS, overlap, read_jsonl
    from scripts.evidence_qa_dev_v3_1_lib import RUN_ROOT as V31_RUN_ROOT
except ModuleNotFoundError:
    from evidence_qa_dev_lib_v1 import (  # type: ignore[no-redef]
        DATA,
        DEV_IDS,
        DOCS,
        overlap,
        read_jsonl,
    )
    from evidence_qa_dev_v3_1_lib import RUN_ROOT as V31_RUN_ROOT  # type: ignore[no-redef]

ANSWERABLE_IDS = [
    "q001", "q002", "q004", "q007", "q008", "q013", "q015", "q019", "q050"
]
METRIC_JSON = DATA / "citation-recall-metric-v2.json"
METRIC_DOC = DOCS / "citation-recall-metric-v2.md"
RELATION_JSONL = DATA / "citation-recall-gold-relation-audit-v1.jsonl"
RELATION_CSV = DATA / "citation-recall-gold-relation-audit-v1.csv"
RELATION_DOC = DOCS / "citation-recall-gold-relation-audit-v1.md"
COMPARISON_JSON = DATA / "citation-recall-v2-comparison.json"
COMPARISON_CSV = DATA / "citation-recall-v2-comparison.csv"
COMPARISON_DOC = DOCS / "citation-recall-v2-comparison.md"
CLAIM_MATCH_THRESHOLD = 0.35


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value, ensure_ascii=False, sort_keys=True)
                if isinstance(value, (dict, list))
                else value
                for key, value in row.items()
            })


def build_relations() -> tuple[list[dict[str, Any]], dict[str, set[tuple[str, int, str]]]]:
    claims = [
        row for row in read_jsonl(DATA / "claim-units-v1.jsonl")
        if row["question_id"] in ANSWERABLE_IDS
    ]
    evidence = {
        (row["paper_id"], row["block_id"]): row
        for row in read_jsonl(DATA / "evidence-corpus-v1.jsonl")
    }
    block_claims: dict[tuple[str, str], set[str]] = {}
    for claim in claims:
        paper_id = claim["target_paper_ids"][0]
        for block_id in claim["gold_block_ids"]:
            block_claims.setdefault((paper_id, block_id), set()).add(claim["claim_id"])
    rows: list[dict[str, Any]] = []
    by_claim: dict[str, set[tuple[str, int, str]]] = {}
    for claim in claims:
        relations: set[tuple[str, int, str]] = set()
        paper_id = claim["target_paper_ids"][0]
        inherited = "pending review" in claim["derivation_trace"]["gold_mapping"]
        for block_id in claim["gold_block_ids"]:
            unit = evidence.get((paper_id, block_id))
            page = int(unit["page"]) if unit else int(claim["gold_pages"][0])
            relation = (paper_id, page, block_id)
            relations.add(relation)
            rows.append({
                "question_id": claim["question_id"],
                "required_claim_id": claim["claim_id"],
                "paper_id": paper_id,
                "page": page,
                "block_id": block_id,
                "relation_key": f"{claim['question_id']}|{claim['claim_id']}|{paper_id}|{page}|{block_id}",
                "claim_role": claim["claim_role"],
                "block_type": unit.get("block_type") if unit else None,
                "answerable": True,
                "relation_source": "claim-units-v1 question-level candidate Gold mapping",
                "duplicated_across_claims": len(block_claims[(paper_id, block_id)]) > 1,
                "duplicated_within_claim": False,
                "page_only_relation": False,
                "exact_block_available": unit is not None,
                "evidence_corpus_triple_exists": bool(unit and int(unit["page"]) == page),
                "ambiguity": inherited,
                "ambiguity_reason": (
                    "Gold blocks were copied as question-level candidates to every required claim; claim-level mapping is explicitly pending review."
                    if inherited else None
                ),
                "included_in_v2": True,
                "included_in_primary_question_metric": True,
                "included_in_claim_metric": not inherited,
                "exclusion_reason": (
                    "claim-level relation ambiguity" if inherited else None
                ),
            })
        by_claim[claim["claim_id"]] = relations
    return rows, by_claim


def triples_from_answer(answer: dict[str, Any]) -> list[tuple[str, int, str]]:
    triples: list[tuple[str, int, str]] = []
    for claim in answer.get("claims", []):
        for citation in claim.get("citations", []):
            triples.append((
                citation["paper_id"], int(citation["page"]), citation["block_id"]
            ))
    return triples


def generated_claims(answer: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "text": claim.get("claim_text") or claim.get("text") or "",
            "triples": {
                (item["paper_id"], int(item["page"]), item["block_id"])
                for item in claim.get("citations", [])
            },
        }
        for claim in answer.get("claims", [])
    ]


def load_experiments() -> list[dict[str, Any]]:
    qa = json.loads((DATA / "qa-production-v1.json").read_text(encoding="utf-8"))
    stage11 = {
        row["question_id"]: {
            "status": row["status"],
            "answer": row.get("answer", {}),
        }
        for row in qa["queries"] if row["question_id"] in DEV_IDS
    }
    dev1 = json.loads((DATA / "evidence-qa-dev-v1.json").read_text(encoding="utf-8"))
    stage132 = {}
    root132 = DATA / "evidence-qa-dev-v1/runs/retrieval_only"
    for run_id in dev1["selected_runs"]:
        row = json.loads((root132 / run_id / "result.json").read_text(encoding="utf-8"))
        stage132[row["question_id"]] = {
            "status": row["status"],
            "answer": row.get("answer", {}),
        }
    dev2 = json.loads((DATA / "evidence-qa-dev-v2.json").read_text(encoding="utf-8"))
    stage133 = {}
    root2 = DATA / "evidence-qa-dev-v2/runs"
    for run_id in dev2["selected_runs"]:
        row = json.loads((root2 / run_id / "result.json").read_text(encoding="utf-8"))
        stage133[row["question_id"]] = {
            "status": row["status"],
            "answer": row.get("answer", {}),
        }
    dev31 = json.loads((DATA / "evidence-qa-dev-v3-1.json").read_text(encoding="utf-8"))
    audit = read_jsonl(DATA / "evidence-qa-dev-v3-1-citation-audit-v1.jsonl")
    audit_by_question: dict[str, list[dict[str, Any]]] = {}
    for row in audit:
        audit_by_question.setdefault(row["question_id"], []).append(row)
    stage138 = {}
    for run_id in dev31["selected_runs"]:
        row = json.loads((V31_RUN_ROOT / run_id / "result.json").read_text(encoding="utf-8"))
        answer = row.get("answer", {})
        claims = []
        for slot in answer.get("required_claim_results", []):
            citations = [
                item["citation_triple"]
                for item in audit_by_question.get(row["question_id"], [])
                if item["required_claim_id"] == slot["required_claim_id"]
            ]
            claims.append({
                "required_claim_id": slot["required_claim_id"],
                "claim_text": slot["claim_text"],
                "citations": citations,
            })
        stage138[row["question_id"]] = {
            "status": row["status"],
            "answer": {"answerable": answer.get("answerable"), "claims": claims},
        }
    return [
        {
            "evaluation_version": "stage11c_a",
            "rows": stage11,
            "historical_reported_recall": json.loads(
                (DATA / "evidence-qa-dev-v1.json").read_text(encoding="utf-8")
            )["variants"]["historical_stage11c"]["metrics"]["citation_recall"],
            "historical_formula": "completed answerable macro in historical helper",
        },
        {
            "evaluation_version": "stage13_2_b",
            "rows": stage132,
            "historical_reported_recall": dev1["variants"]["retrieval_only"]["metrics"]["citation_recall"],
            "historical_formula": "completed answerable macro; failures dynamically excluded",
        },
        {
            "evaluation_version": "stage13_3_dev_v2",
            "rows": stage133,
            "historical_reported_recall": dev2["metrics"]["all_manifest_conservative"]["citation_recall"],
            "historical_formula": "completed answerable macro; q050 failure dynamically excluded",
        },
        {
            "evaluation_version": "stage13_8_dev_v3_1",
            "rows": stage138,
            "historical_reported_recall": dev31["metrics"]["all_manifest_conservative"]["citation_recall"],
            "historical_formula": "fixed ten-question macro with q005 refusal=1",
        },
    ]


def calculate(
    experiment: dict[str, Any],
    claims_by_question: dict[str, list[dict[str, Any]]],
    relations_by_claim: dict[str, set[tuple[str, int, str]]],
) -> dict[str, Any]:
    rows = experiment["rows"]
    per_question: dict[str, float] = {}
    claim_scores: list[float] = []
    relation_hits: set[str] = set()
    relation_total: set[str] = set()
    failures = []
    for question_id in DEV_IDS:
        run = rows.get(question_id)
        if not run or str(run["status"]).lower() != "completed":
            failures.append(question_id)
            per_question[question_id] = 0.0
            for claim in claims_by_question.get(question_id, []):
                claim_scores.append(0.0)
                for triple in relations_by_claim[claim["claim_id"]]:
                    relation_total.add(
                        f"{question_id}|{claim['claim_id']}|{triple[0]}|{triple[1]}|{triple[2]}"
                    )
            continue
        answer = run.get("answer", {})
        if question_id == "q005":
            per_question[question_id] = float(answer.get("answerable") is False)
            continue
        question_claims = claims_by_question[question_id]
        expected_question = set().union(
            *(relations_by_claim[claim["claim_id"]] for claim in question_claims)
        )
        cited_question = set(triples_from_answer(answer))
        per_question[question_id] = (
            len(expected_question & cited_question) / len(expected_question)
            if expected_question else 0.0
        )
        direct = {
            claim.get("required_claim_id"): {
                (item["paper_id"], int(item["page"]), item["block_id"])
                for item in claim.get("citations", [])
            }
            for claim in answer.get("claims", [])
            if claim.get("required_claim_id")
        }
        legacy = generated_claims(answer)
        for claim in question_claims:
            claim_id = claim["claim_id"]
            expected = relations_by_claim[claim_id]
            cited = direct.get(claim_id)
            if cited is None:
                match = max(
                    legacy,
                    key=lambda item: overlap(claim["claim_text"], item["text"]),
                    default=None,
                )
                cited = (
                    match["triples"]
                    if match
                    and overlap(claim["claim_text"], match["text"])
                    >= CLAIM_MATCH_THRESHOLD
                    else set()
                )
            hits = expected & cited
            claim_scores.append(len(hits) / len(expected) if expected else 0.0)
            for triple in expected:
                key = f"{question_id}|{claim_id}|{triple[0]}|{triple[1]}|{triple[2]}"
                relation_total.add(key)
                if triple in hits:
                    relation_hits.add(key)
    answerable_macro = mean(per_question[qid] for qid in ANSWERABLE_IDS)
    return {
        "evaluation_version": experiment["evaluation_version"],
        "available_questions": sorted(rows),
        "missing_questions": sorted(set(DEV_IDS) - set(rows)),
        "provider_or_schema_failures": failures,
        "manifest_questions": DEV_IDS,
        "answerable_fixed_denominator": 9,
        "question_macro_exact_recall": mean(per_question.values()),
        "answerable_question_macro_exact_recall_v2": answerable_macro,
        "required_claim_macro_exact_recall": mean(claim_scores),
        "micro_exact_gold_relation_recall": len(relation_hits) / len(relation_total),
        "micro_relation_numerator": len(relation_hits),
        "micro_relation_denominator": len(relation_total),
        "per_question": per_question,
        "historical_reported_recall": experiment["historical_reported_recall"],
        "historical_formula": experiment["historical_formula"],
        "directly_comparable_under_v2": True,
        "limitations": [
            "Required-claim and micro relation metrics use inherited question-level Gold candidates and are diagnostic because claim-level Gold mapping is ambiguous.",
            "Legacy generated claims are mapped to required claims using the frozen lexical overlap threshold 0.35.",
        ],
    }


def main() -> None:
    relation_rows, relations_by_claim = build_relations()
    claims = [
        row for row in read_jsonl(DATA / "claim-units-v1.jsonl")
        if row["question_id"] in ANSWERABLE_IDS
    ]
    claims_by_question: dict[str, list[dict[str, Any]]] = {}
    for claim in claims:
        claims_by_question.setdefault(claim["question_id"], []).append(claim)
    RELATION_JSONL.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in relation_rows),
        encoding="utf-8",
    )
    write_csv(RELATION_CSV, relation_rows)
    ambiguity_count = sum(row["ambiguity"] for row in relation_rows)
    RELATION_DOC.write_text(
        "# Citation Recall Gold Relation Audit v1\n\n"
        f"- Relations: {len(relation_rows)}\n"
        f"- Ambiguous claim-level relations: {ambiguity_count}\n"
        "- Every claim currently inherits question-level candidate Gold blocks; claim-level mapping is explicitly pending review.\n"
        "- Question-level exact recall remains computable, but required-claim and micro relation metrics are diagnostic.\n"
        "- Gold and Retrieval Gold were not modified.\n"
        + (
            "- Status: **CITATION_RECALL_V2_BLOCKED_BY_GOLD_RELATION_AMBIGUITY**\n"
            if ambiguity_count else "- Status: no blocking ambiguity\n"
        ),
        encoding="utf-8",
    )
    protocol = {
        "metric_id": "answerable_question_macro_exact_recall_v2",
        "protocol": "citation-recall-v2",
        "version": "2.0",
        "selected_primary_metric": "answerable_question_macro_exact_recall_v2",
        "rationale": "Fixed answerable question set, stable denominator, failures score zero, and unanswerable refusal is not mixed into citation recall.",
        "formula": "mean over fixed 9 answerable questions of unique exact Gold triples hit / fixed question Gold triples",
        "fixed_manifest": DEV_IDS,
        "fixed_answerable_questions": ANSWERABLE_IDS,
        "fixed_denominator": 9,
        "failure_handling": "zero",
        "unanswerable_handling": "excluded",
        "empty_gold_handling": "score zero and report; protocol blocked if unexpected",
        "duplicate_relation_handling": "dedupe by full relation key",
        "relation_key": [
            "question_id", "required_claim_id", "paper_id", "page", "block_id"
        ],
        "question_relation_key": ["question_id", "paper_id", "page", "block_id"],
        "gold_version": "gold-set-v1-human-reviewed-2026-07-13 / claim-units-v1",
        "frozen_at": "2026-07-16",
        "frozen_before_next_live_run": True,
        "backward_gate_effect": False,
        "stage13_8_historical_gate": "FAILED",
        "dev_v3_1_comparable_recall_status": "RECALCULATED_DIAGNOSTIC",
        "metrics": {
            "question_macro_exact_recall": "fixed 10 questions; correct unanswerable refusal scores 1, failure scores 0",
            "answerable_question_macro_exact_recall_v2": "primary; fixed 9 answerable questions",
            "required_claim_macro_exact_recall": "fixed 27 claims; diagnostic while claim Gold mapping is ambiguous",
            "micro_exact_gold_relation_recall": "full relation-key micro recall; diagnostic while claim Gold mapping is ambiguous",
        },
    }
    METRIC_JSON.write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    METRIC_DOC.write_text(
        "# Citation Recall Metric v2\n\n"
        "- Protocol: `citation-recall-v2`\n"
        "- Primary: `answerable_question_macro_exact_recall_v2`\n"
        "- Fixed denominator: 9 answerable questions\n"
        "- q005: excluded from citation recall\n"
        "- q050/provider/schema/validation failure: recall 0, never dynamically excluded\n"
        "- Duplicate handling: exact full relation key\n"
        "- Historical Stage 13.8 Gate effect: none; it remains FAILED.\n"
        "- Frozen before any Dev v3.2 live run: true\n",
        encoding="utf-8",
    )
    comparisons = [
        calculate(experiment, claims_by_question, relations_by_claim)
        for experiment in load_experiments()
    ]
    payload = {
        "schema_version": "citation-recall-v2-comparison-v1",
        "protocol": protocol,
        "gold_relation_count": len(relation_rows),
        "gold_relation_ambiguity_count": ambiguity_count,
        "status": (
            "CITATION_RECALL_V2_BLOCKED_BY_GOLD_RELATION_AMBIGUITY"
            if ambiguity_count else "complete"
        ),
        "experiments": comparisons,
    }
    COMPARISON_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(COMPARISON_CSV, comparisons)
    table = "\n".join(
        f"| {row['evaluation_version']} | {row['historical_reported_recall']:.6f} | "
        f"{row['answerable_question_macro_exact_recall_v2']:.6f} | "
        f"{row['required_claim_macro_exact_recall']:.6f} | "
        f"{row['micro_exact_gold_relation_recall']:.6f} |"
        for row in comparisons
    )
    COMPARISON_DOC.write_text(
        "# Citation Recall v2 Historical Comparison\n\n"
        "| Evaluation | Historical | Answerable macro v2 | Claim macro | Micro relation |\n"
        "|---|---:|---:|---:|---:|\n"
        f"{table}\n\n"
        "All v2 primary values use the same fixed nine-question denominator and score failures as zero. Claim-level auxiliary metrics remain diagnostic due to Gold relation ambiguity.\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "relations": len(relation_rows),
        "ambiguities": ambiguity_count,
        "status": payload["status"],
        "experiments": {
            row["evaluation_version"]: row["answerable_question_macro_exact_recall_v2"]
            for row in comparisons
        },
    }))


if __name__ == "__main__":
    main()
