"""Merge Stage 13.1 human review and run offline Phase B ablation.

No LLM, reranker, Deep Research, embedding request, or Dev QA is executed.
"""

# ruff: noqa: E501 -- report strings stay readable as Markdown source.

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

from paper_research.evidence.claims import ClaimUnit
from paper_research.evidence.schema import EvidenceUnit
from paper_research.retrieval.context_completion import complete_with_adjacent_same_page
from paper_research.retrieval.evidence_retriever import EvidenceCandidate, EvidenceRetriever
from paper_research.retrieval.filters import RetrievalFilter
from paper_research.retrieval.query_router import route_query

try:
    from scripts.run_evidence_retrieval_v1 import (
        _deduplicate,
        _evaluate_row,
        _jsonl,
        _selection_claim,
        _source_scores,
        _summary,
    )
except ModuleNotFoundError:
    from run_evidence_retrieval_v1 import (
        _deduplicate,
        _evaluate_row,
        _jsonl,
        _selection_claim,
        _source_scores,
        _summary,
    )

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/evaluation"
DOCS = ROOT / "docs"
ARTIFACTS = ROOT / "artifacts/stage13-1-human-review"

GAP_PENDING = DATA / "evidence-gap-cases-v1.jsonl"
PILOT_PENDING = DATA / "claim-evidence-gold-pilot-v1.jsonl"
GAP_REVIEWED = ARTIFACTS / "evidence-gap-cases-v1-reviewed.jsonl"
PILOT_REVIEWED = ARTIFACTS / "claim-evidence-gold-pilot-v1-reviewed.jsonl"

GAP_CSV = DATA / "evidence-gap-cases-v1.csv"
GAP_MD = DOCS / "evidence-gap-cases-v1.md"
PILOT_MD = DOCS / "claim-evidence-gold-pilot-v1.md"
TAXONOMY_JSON = DATA / "evidence-gap-taxonomy-phase-b-v1.json"
TAXONOMY_MD = DOCS / "evidence-gap-taxonomy-phase-b-v1.md"
ABLATION_JSON = DATA / "evidence-retrieval-phase-b-ablation-v1.json"
ABLATION_CSV = DATA / "evidence-retrieval-phase-b-ablation-v1.csv"
ABLATION_MD = DOCS / "evidence-retrieval-phase-b-ablation-v1.md"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def backup(path: Path) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    target = path.with_name(f"{path.name}.pending.{stamp}.bak")
    shutil.copy2(path, target)
    return target


def stable_row_hash(row: dict[str, Any]) -> str:
    stable = dict(row)
    for key in (
        "annotation_status",
        "decision",
        "reviewer",
        "reviewed_at",
        "review_notes",
        "approved_evidence_sets",
        "approved_alternative_evidence_sets",
        "multi_block_required",
        "claim_role_after_review",
    ):
        stable.pop(key, None)
    return hashlib.sha256(json.dumps(stable, sort_keys=True).encode("utf-8")).hexdigest()


def validate_reviewed() -> dict[str, Any]:
    gaps_pending = read_jsonl(GAP_PENDING)
    gaps_reviewed = read_jsonl(GAP_REVIEWED)
    pilot_pending = read_jsonl(PILOT_PENDING)
    pilot_reviewed = read_jsonl(PILOT_REVIEWED)
    evidence = read_jsonl(DATA / "evidence-corpus-v1.jsonl")
    evidence_triples = {(row["paper_id"], int(row["page"]), row["block_id"]) for row in evidence}
    current_hashes = {
        "claim_units_sha256": file_hash(DATA / "claim-units-v1.jsonl"),
        "evidence_corpus_sha256": file_hash(DATA / "evidence-corpus-v1.jsonl"),
        "gold_sha256": file_hash(DATA / "gold-set-v1.jsonl"),
    }

    if len(gaps_reviewed) != 17 or len(pilot_reviewed) != 40:
        raise RuntimeError("reviewed artifact counts must be 17 gaps and 40 Pilot samples")
    if set(gaps_pending[0]) != set(gaps_reviewed[0]):
        raise RuntimeError("gap reviewed schema differs from pending schema")
    if set(pilot_pending[0]) != set(pilot_reviewed[0]):
        raise RuntimeError("Pilot reviewed schema differs from pending schema")
    if {row["question_id"] for row in gaps_pending} != {row["question_id"] for row in gaps_reviewed}:
        raise RuntimeError("gap question_id set changed")
    if {row["pilot_sample_id"] for row in pilot_pending} != {
        row["pilot_sample_id"] for row in pilot_reviewed
    }:
        raise RuntimeError("Pilot sample_id set changed")
    if Counter(row["human_review_status"] for row in gaps_reviewed) != {"reviewed": 17}:
        raise RuntimeError("all gap rows must be reviewed")
    if Counter(row["annotation_status"] for row in pilot_reviewed) != {"reviewed": 40}:
        raise RuntimeError("all Pilot rows must be reviewed")
    if Counter(row["decision"] for row in pilot_reviewed) != {"approved": 40}:
        raise RuntimeError("all Pilot rows must have approved decisions")

    pending_record_hashes = {
        row["pilot_sample_id"]: row["source_record_hash"] for row in pilot_pending
    }
    triple_count = 0
    for row in pilot_reviewed:
        if row["source_hashes"] != current_hashes:
            raise RuntimeError(f"source_hashes invalidated for {row['pilot_sample_id']}")
        if row["source_record_hash"] != pending_record_hashes[row["pilot_sample_id"]]:
            raise RuntimeError(f"source_record_hash invalidated for {row['pilot_sample_id']}")
        allowed = set(row["target_papers"])
        for field in ("approved_evidence_sets", "approved_alternative_evidence_sets"):
            for group in row.get(field) or []:
                for item in group:
                    triple_count += 1
                    triple = (item["paper_id"], int(item["page"]), item["block_id"])
                    if triple not in evidence_triples:
                        raise RuntimeError(f"missing Pilot evidence triple: {triple}")
                    if item["paper_id"] not in allowed:
                        raise RuntimeError(f"Pilot evidence outside target paper: {triple}")
        if row["claim_role_after_review"] != "verify_absence" and not row["approved_evidence_sets"]:
            raise RuntimeError(f"approved non-absence sample lacks evidence: {row['pilot_sample_id']}")
    for row in gaps_reviewed:
        for field in ("gold_block_text", "selected_evidence_text", "candidate_pool_top_30"):
            for item in row.get(field) or []:
                if not all(key in item for key in ("paper_id", "page", "block_id")):
                    continue
                triple_count += 1
                triple = (item["paper_id"], int(item["page"]), item["block_id"])
                if triple not in evidence_triples:
                    raise RuntimeError(f"missing gap evidence triple: {triple}")
    return {
        "gap_schema_valid": True,
        "pilot_schema_valid": True,
        "source_hashes_valid": True,
        "source_record_hash_valid": True,
        "triples_valid": True,
        "triples_checked": triple_count,
        "gap_review_status": dict(Counter(row["human_review_status"] for row in gaps_reviewed)),
        "pilot_review_status": dict(Counter(row["annotation_status"] for row in pilot_reviewed)),
        "pilot_decisions": dict(Counter(row["decision"] for row in pilot_reviewed)),
    }


def merge_reviewed() -> dict[str, str]:
    gap_backup = backup(GAP_PENDING)
    pilot_backup = backup(PILOT_PENDING)
    write_jsonl(GAP_PENDING, read_jsonl(GAP_REVIEWED))
    write_jsonl(PILOT_PENDING, read_jsonl(PILOT_REVIEWED))
    return {"gap_pending_backup": str(gap_backup), "pilot_pending_backup": str(pilot_backup)}


def write_review_reports() -> dict[str, Any]:
    gaps = read_jsonl(GAP_PENDING)
    pilot = read_jsonl(PILOT_PENDING)
    with GAP_CSV.open("w", encoding="utf-8", newline="") as stream:
        fieldnames = [
            "question_id",
            "category",
            "difficulty",
            "retrieval_scope",
            "gold_page_hit",
            "initial_failure_category",
            "human_review_status",
            "human_failure_category",
            "reviewer",
            "reviewed_at",
            "review_notes",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in gaps:
            writer.writerow({key: row.get(key) for key in fieldnames})

    gap_lines = [
        "# Evidence Gap Cases v1",
        "",
        "> Human adjudication has been merged from `artifacts/stage13-1-human-review`.",
        "",
        "| Question | Initial category | Human category | Selected support | Reviewer |",
        "|---|---|---|---|---|",
    ]
    for row in gaps:
        adjudication = row["human_adjudication"]
        gap_lines.append(
            f"| {row['question_id']} | {row['initial_failure_category']} | "
            f"{row['human_failure_category']} | {adjudication['selected_evidence_support']} | "
            f"{row['reviewer']} |"
        )
    GAP_MD.write_text("\n".join(gap_lines) + "\n", encoding="utf-8")

    pilot_lines = [
        "# Claim-Evidence Gold Pilot v1",
        "",
        "> Human adjudication has been merged. These 40 samples are still a Pilot, not the full 146-claim Gold set.",
        "",
        f"- Samples: {len(pilot)}",
        f"- Reviewed: {sum(row['annotation_status'] == 'reviewed' for row in pilot)}",
        f"- Approved decisions: {sum(row['decision'] == 'approved' for row in pilot)}",
        f"- Multi-block required: {sum(bool(row['multi_block_required']) for row in pilot)}",
        "",
        "| Sample | Question | Stratum | Role after review | Evidence sets |",
        "|---|---|---|---|---:|",
    ]
    for row in pilot:
        pilot_lines.append(
            f"| {row['pilot_sample_id']} | {row['question_id']} | {row['sampling_stratum']} | "
            f"{row['claim_role_after_review']} | {len(row['approved_evidence_sets'])} |"
        )
    PILOT_MD.write_text("\n".join(pilot_lines) + "\n", encoding="utf-8")

    taxonomy = {
        "schema_version": "evidence-gap-taxonomy-phase-b-v1",
        "source": "human-reviewed evidence-gap-cases-v1",
        "gap_count": len(gaps),
        "human_failure_categories": dict(Counter(row["human_failure_category"] for row in gaps)),
        "initial_failure_categories": dict(Counter(row["initial_failure_category"] for row in gaps)),
        "selected_support": dict(
            Counter(row["human_adjudication"]["selected_evidence_support"] for row in gaps)
        ),
        "issue_classes": dict(Counter(row["human_adjudication"]["issue_class"] for row in gaps)),
        "gold_too_broad": sum(row["human_adjudication"]["gold_too_broad"] for row in gaps),
        "equivalent_non_gold": sum(
            row["human_adjudication"]["equivalent_non_gold_evidence_exists"] for row in gaps
        ),
        "phase_b_rule": {
            "name": "adjacent_same_page_completion",
            "rationale": "Human review found page-hit/block-miss, parsing-boundary, and partially supported selected evidence patterns. The rule keeps the frozen routed ranking and adds immediate same-page neighbors around high-ranked selected evidence without reading Gold labels.",
            "uses_gold_for_selection": False,
            "uses_human_pilot_for_selection": False,
        },
    }
    TAXONOMY_JSON.write_text(json.dumps(taxonomy, ensure_ascii=False, indent=2), encoding="utf-8")
    TAXONOMY_MD.write_text(
        "# Evidence Gap Taxonomy Phase B v1\n\n"
        f"- Gap rows reviewed: {taxonomy['gap_count']}\n"
        f"- Human categories: `{json.dumps(taxonomy['human_failure_categories'], sort_keys=True)}`\n"
        f"- Selected support: `{json.dumps(taxonomy['selected_support'], sort_keys=True)}`\n"
        f"- Issue classes: `{json.dumps(taxonomy['issue_classes'], sort_keys=True)}`\n"
        f"- Phase B rule: `{taxonomy['phase_b_rule']['name']}`\n\n"
        "The Phase B rule is a general context-boundary completion rule. It does not branch on question IDs, block IDs, Gold labels, approved Pilot evidence, or answer text.\n",
        encoding="utf-8",
    )
    return taxonomy


def selected_triples(selected: list[EvidenceCandidate]) -> set[tuple[str, int, str]]:
    return {
        (item.evidence.paper_id, item.evidence.page, item.evidence.block_id)
        for item in selected
    }


def pilot_metrics(rows: list[dict[str, Any]], pilot: list[dict[str, Any]]) -> dict[str, Any]:
    selected_by_q = {
        row["question_id"]: {
            tuple(item) for item in row["metrics"]["citation_triples"]
        }
        for row in rows
    }
    reviewed = [
        row
        for row in pilot
        if row["decision"] == "approved" and row["claim_role_after_review"] != "verify_absence"
    ]
    hits = 0
    by_question: dict[str, list[bool]] = defaultdict(list)
    for row in reviewed:
        selected = selected_by_q[row["question_id"]]
        set_hit = any(
            {
                (item["paper_id"], item["page"], item["block_id"])
                for item in evidence_set
            }.issubset(selected)
            for evidence_set in row["approved_evidence_sets"]
        )
        hits += int(set_hit)
        by_question[row["question_id"]].append(set_hit)
    all_claims = [all(values) for values in by_question.values()]
    return {
        "pilot_reviewed_samples": len(pilot),
        "approved_non_absence_samples": len(reviewed),
        "claim_evidence_set_recall": round(hits / len(reviewed), 6) if reviewed else None,
        "all_required_claims_evidence_available": round(mean(all_claims), 6)
        if all_claims
        else None,
    }


def slices(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output = {}
    for field in ("retrieval_scope", "category", "difficulty"):
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[row[field]].append(row)
        output[field] = {key: _summary(value) for key, value in sorted(grouped.items())}
    return output


def run_ablation() -> dict[str, Any]:
    started = time.perf_counter()
    units = [EvidenceUnit.model_validate(row) for row in _jsonl(DATA / "evidence-corpus-v1.jsonl")]
    claims = [ClaimUnit.model_validate(row) for row in _jsonl(DATA / "claim-units-v1.jsonl")]
    gold = {row["question_id"]: row for row in _jsonl(DATA / "gold-set-v1.jsonl")}
    protocol = {row["question_id"]: row for row in _jsonl(DATA / "retrieval-gold-v2.jsonl")}
    baseline = json.loads((DATA / "qa-production-v1.json").read_text(encoding="utf-8"))
    baseline_rows = {row["question_id"]: row for row in baseline["queries"]}
    pilot = read_jsonl(PILOT_PENDING)
    claims_by_question: dict[str, list[ClaimUnit]] = defaultdict(list)
    for claim in claims:
        claims_by_question[claim.question_id].append(claim)
    retriever = EvidenceRetriever(units)

    variants: dict[str, list[dict[str, Any]]] = {
        "stage13_routed_baseline": [],
        "phase_b_adjacent_same_page_completion": [],
    }
    for question_id in sorted(gold):
        gold_row = gold[question_id]
        protocol_row = protocol[question_id]
        selection_claim = _selection_claim(protocol_row, gold_row)
        retrieval_filter = RetrievalFilter(**protocol_row["retrieval_filter"])
        decision = route_query(protocol_row["retrieval_query"], [selection_claim], retrieval_filter)
        source = _source_scores(baseline_rows[question_id]["context"])
        routed_started = time.perf_counter()
        pool = retriever.score_candidates(selection_claim, decision, source_scores=source)
        pool = pool[: decision.profile.candidate_pool_k]
        selected = _deduplicate(sorted(pool, key=lambda item: -item.total_score), 10)
        routed_latency = (time.perf_counter() - routed_started) * 1000
        phase_b_started = time.perf_counter()
        completed = complete_with_adjacent_same_page(selected, units, seed_limit=5, window=1)
        phase_b_latency = routed_latency + (time.perf_counter() - phase_b_started) * 1000
        common = {
            "question_id": question_id,
            "retrieval_scope": protocol_row["retrieval_scope"],
            "category": gold_row["category"],
            "difficulty": gold_row["difficulty"],
        }
        variants["stage13_routed_baseline"].append(
            {
                **common,
                "metrics": _evaluate_row(
                    selected,
                    gold_row,
                    claims_by_question[question_id],
                    candidate_count=len(pool),
                    truncated=False,
                    latency_ms=routed_latency,
                ),
            }
        )
        variants["phase_b_adjacent_same_page_completion"].append(
            {
                **common,
                "metrics": _evaluate_row(
                    completed,
                    gold_row,
                    claims_by_question[question_id],
                    candidate_count=len(pool),
                    truncated=False,
                    latency_ms=phase_b_latency,
                ),
                "phase_b_trace": {
                    "base_selected_count": len(selected),
                    "completed_selected_count": len(completed),
                    "added_adjacent_count": len(completed) - len(selected),
                    "uses_gold_for_selection": False,
                    "uses_human_pilot_for_selection": False,
                },
            }
        )

    results = []
    for name, rows in variants.items():
        metrics = _summary(rows)
        metrics.update(pilot_metrics(rows, pilot))
        results.append({"name": name, "metrics": metrics, "slices": slices(rows), "queries": rows})
    baseline_metrics = results[0]["metrics"]
    candidate_metrics = results[1]["metrics"]
    hit_gain_questions = [
        row["question_id"]
        for row, base in zip(results[1]["queries"], results[0]["queries"], strict=True)
        if row["metrics"]["answerable"]
        and row["metrics"]["exact_gold_block_available"]
        and not base["metrics"]["exact_gold_block_available"]
    ]
    hit_loss_questions = [
        row["question_id"]
        for row, base in zip(results[1]["queries"], results[0]["queries"], strict=True)
        if row["metrics"]["answerable"]
        and not row["metrics"]["exact_gold_block_available"]
        and base["metrics"]["exact_gold_block_available"]
    ]
    gates = {
        "exact_block_at_least_0_65": candidate_metrics["exact_gold_block_availability"] >= 0.65,
        "gold_page_at_least_0_80": candidate_metrics["gold_page_availability"] >= 0.80,
        "metadata_not_above_baseline": candidate_metrics["metadata_contamination_rate"]
        <= baseline_metrics["metadata_contamination_rate"],
        "context_tokens_within_150_percent_of_routed": candidate_metrics[
            "mean_context_token_count"
        ]
        <= baseline_metrics["mean_context_token_count"] * 1.5,
        "offline_p95_below_500_ms": candidate_metrics["p95_latency_ms"] < 500,
        "no_exact_hit_regressions": not hit_loss_questions,
        "claim_evidence_set_recall_available": candidate_metrics["claim_evidence_set_recall"]
        is not None,
        "citation_triple_trace_complete": all(
            len(row["metrics"]["citation_triples"]) == row["metrics"]["selected_count"]
            for row in results[1]["queries"]
        ),
        "no_oracle_or_gold_injection": True,
        "reranker_disabled": True,
        "llm_not_called": True,
        "deep_research_not_called": True,
        "no_new_embedding_requests": True,
    }
    payload = {
        "status": "COMPLETED_OFFLINE",
        "protocol_version": "stage13-1-phase-b-v1",
        "variants": results,
        "candidate": "phase_b_adjacent_same_page_completion",
        "hit_gain_questions": hit_gain_questions,
        "hit_loss_questions": hit_loss_questions,
        "offline_candidate_gates": gates,
        "allow_dev_qa": all(gates.values()),
        "dev_qa_run": False,
        "llm_called": False,
        "reranker_enabled": False,
        "deep_research_called": False,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
    }
    ABLATION_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with ABLATION_CSV.open("w", encoding="utf-8", newline="") as stream:
        fieldnames = ["variant", *results[0]["metrics"].keys()]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for item in results:
            writer.writerow({"variant": item["name"], **item["metrics"]})
    lines = [
        "# Evidence Retrieval Phase B Ablation v1",
        "",
        "> Offline only. Dev QA was not run. No LLM, reranker, Deep Research, or new embedding request was made.",
        "",
        "| Variant | Exact block | Gold page | Block recall | Claim evidence recall | Mean tokens | P95 ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in results:
        metric = item["metrics"]
        claim_recall = metric["claim_evidence_set_recall"]
        lines.append(
            f"| {item['name']} | {metric['exact_gold_block_availability']:.6f} | "
            f"{metric['gold_page_availability']:.6f} | {metric['gold_block_recall']:.6f} | "
            f"{claim_recall:.6f} | {metric['mean_context_token_count']:.2f} | "
            f"{metric['p95_latency_ms']:.6f} |"
        )
    lines += [
        "",
        f"- Hit gains: `{hit_gain_questions}`",
        f"- Hit losses: `{hit_loss_questions}`",
        f"- Offline candidate gates: `{json.dumps(gates, sort_keys=True)}`",
        f"- Allow Dev QA: **{payload['allow_dev_qa']}**",
        "- Actual Dev QA: **False**",
    ]
    ABLATION_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    validation = validate_reviewed()
    backups = merge_reviewed()
    taxonomy = write_review_reports()
    ablation = run_ablation()
    print(
        json.dumps(
            {
                "validation": validation,
                "backups": backups,
                "human_failure_categories": taxonomy["human_failure_categories"],
                "candidate": ablation["candidate"],
                "candidate_exact": ablation["variants"][1]["metrics"][
                    "exact_gold_block_availability"
                ],
                "allow_dev_qa": ablation["allow_dev_qa"],
                "dev_qa_run": ablation["dev_qa_run"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
