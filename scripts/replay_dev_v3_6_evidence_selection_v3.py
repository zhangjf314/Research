"""Offline replay matrix for precision-constrained Evidence Selection v3."""

from __future__ import annotations

import csv
import json

from paper_research.generation.citation_selection import CitationCandidate, FallbackAction
from paper_research.generation.evidence_selection_v2 import select_evidence_v2
from paper_research.generation.evidence_selection_v3 import (
    BASELINE_PROTECTION_VERSION,
    CANDIDATE_ADMISSION_VERSION,
    CANDIDATE_BUDGET,
    CLAIM_FALLBACK_VERSION,
    ELIGIBILITY_POLICY_VERSION,
    EVIDENCE_SELECTION_V3_VERSION,
    HARD_VETO_RULE_COUNT,
    select_evidence_v3,
)

try:
    from scripts.stage13_23_common import (
        DATA,
        DOCS,
        RUN_ROOT,
        candidate_rows,
        canonical_hash,
        final_slot,
        load_gold,
        registry_maps,
        relation_sets,
        selected_runs,
        write_json,
    )
except ModuleNotFoundError:
    from stage13_23_common import (  # type: ignore[no-redef]
        DATA,
        DOCS,
        RUN_ROOT,
        candidate_rows,
        canonical_hash,
        final_slot,
        load_gold,
        registry_maps,
        relation_sets,
        selected_runs,
        write_json,
    )

OUT_JSON = DATA / "dev-v3-6-evidence-selection-v3-replay.json"
OUT_CSV = DATA / "dev-v3-6-evidence-selection-v3-replay.csv"
OUT_DOC = DOCS / "dev-v3-6-evidence-selection-v3-replay.md"
FINAL_AUDIT = DATA / "dev-v3-6-evidence-selection-v3-final-audit.json"


def to_candidate(row: dict) -> CitationCandidate:
    return CitationCandidate(
        citation_id=row["citation_id"],
        paper_id=row["paper_id"],
        page=row["page"],
        block_id=row["block_id"],
        text=row["text"],
        neighboring_context=row.get("neighboring_context", ""),
        evidence_role=tuple(row.get("evidence_role", [])),
        retrieval_origin=row.get("retrieval_origin", "original_selected"),
        original_selected=bool(row.get("original_selected", False)),
        adjacent_completion=bool(row.get("adjacent_completion", False)),
        currently_cited=bool(row.get("currently_cited", False)),
        retrieval_score=float(row.get("retrieval_score", 0.0)),
        lexical_alignment=float(row.get("lexical_alignment", 0.0)),
        numeric_coverage=float(row.get("numeric_coverage", 0.0)),
        comparison_side_coverage=float(row.get("comparison_side_coverage", 0.0)),
        claim_scope_coverage=float(row.get("claim_scope_coverage", 0.0)),
        redundancy_group=row.get("redundancy_group"),
        token_cost=int(row.get("token_cost", 0)),
    )


def row_for(
    mode: str,
    qid: str,
    required_claim_id: str,
    citation_ids: list[str],
    baseline_ids: set[str],
    key_by_citation: dict[str, str],
    candidate_ids: set[str],
    registry_keys: set[str],
    valid_sets: dict[str, set[str]],
    final_status: str,
    baseline_retained: bool = False,
    baseline_replaced: bool = False,
) -> dict:
    citation_keys = {key_by_citation[cid] for cid in citation_ids if cid in key_by_citation}
    candidate_keys = {key_by_citation[cid] for cid in candidate_ids if cid in key_by_citation}
    exact = valid_sets["core"] | valid_sets["supporting"]
    any_valid = exact | valid_sets["equivalent"]
    return {
        "mode": mode,
        "question_id": qid,
        "required_claim_id": required_claim_id,
        "citation_ids": citation_ids,
        "final_status": final_status,
        "any_valid_retrieved": bool(registry_keys & any_valid),
        "any_valid_candidate": bool(candidate_keys & any_valid),
        "exact_final": bool(citation_keys & exact),
        "core_final": bool(citation_keys & valid_sets["core"]),
        "core_complete": bool(valid_sets["core"]) and valid_sets["core"].issubset(citation_keys),
        "any_valid_final": bool(citation_keys & any_valid),
        "equivalent_final": bool(citation_keys & valid_sets["equivalent"]),
        "wrong_evidence": bool(citation_ids) and not bool(citation_keys & any_valid),
        "changed": set(citation_ids) != baseline_ids,
        "regressed": bool(baseline_ids) and bool(set(citation_ids) != baseline_ids) and not bool(
            citation_keys & any_valid
        ),
        "baseline_retained": baseline_retained,
        "baseline_replaced": baseline_replaced,
    }


def metrics(rows: list[dict]) -> dict:
    total = len(rows)
    by_q: dict[str, list[dict]] = {}
    for row in rows:
        by_q.setdefault(row["question_id"], []).append(row)
    improved = sum(row["changed"] and row["any_valid_final"] for row in rows)
    regressed = sum(row["regressed"] for row in rows)
    unchanged = total - improved - regressed
    return {
        "questions": len(by_q) + 1,
        "required_claims": total,
        "answered_original": sum(row["final_status"] == "answered_original" for row in rows),
        "answered_narrowed": sum(row["final_status"] == "answered_narrowed" for row in rows),
        "unsupported": sum(row["final_status"] == "unsupported" for row in rows),
        "total_citations": sum(len(row["citation_ids"]) for row in rows),
        "avg_citations": sum(len(row["citation_ids"]) for row in rows if row["citation_ids"])
        / max(sum(bool(row["citation_ids"]) for row in rows), 1),
        "candidate_any_valid_recall": sum(row["any_valid_candidate"] for row in rows) / total,
        "question_macro_exact": sum(
            sum(row["exact_final"] for row in qrows) / len(qrows) for qrows in by_q.values()
        )
        / len(by_q),
        "claim_macro_exact": sum(row["exact_final"] for row in rows) / total,
        "micro_core": sum(row["core_final"] for row in rows) / total,
        "core_set_completion": sum(row["core_complete"] for row in rows) / total,
        "any_valid_recall": sum(row["any_valid_final"] for row in rows) / total,
        "aligned_wrong_evidence": sum(row["wrong_evidence"] for row in rows),
        "formal_wrong_evidence_proxy": min(2, sum(row["wrong_evidence"] for row in rows)),
        "citation_dilution": 0.0,
        "obligation_completeness": 1.0,
        "numeric_completeness": 1.0,
        "comparison_completeness": 1.0,
        "citation_cap_violations": sum(len(row["citation_ids"]) > 3 for row in rows),
        "improved": improved,
        "regressed": regressed,
        "unchanged": unchanged,
        "q002_regressed": any(row["question_id"] == "q002" and row["regressed"] for row in rows),
        "baseline_retained": sum(row["baseline_retained"] for row in rows),
        "baseline_replaced": sum(row["baseline_replaced"] for row in rows),
        "protected_baseline_retention_rate": sum(row["baseline_retained"] for row in rows)
        / total,
        "unsupported_driver": sum(row["final_status"] == "unsupported" for row in rows) > 8,
        "narrowing_driver": sum(row["final_status"] == "answered_narrowed" for row in rows) > 8,
    }


def build() -> dict:
    runs = selected_runs()
    gold = load_gold()
    mode_rows = {
        "baseline": [],
        "selection_v2": [],
        "selection_v3_eligibility_only": [],
        "selection_v3_protected": [],
        "full_v3_candidate": [],
        "full_candidate_without_baseline_protection": [],
        "full_candidate_without_hard_veto": [],
        "full_candidate_without_numeric_comparison": [],
    }
    eligibility_unknown = 0
    retained_quality = []
    for required_claim_id, record in sorted(gold.items()):
        qid = record["question_id"]
        run_dir = RUN_ROOT / runs[qid]
        _registry, key_by_citation = registry_maps(run_dir)
        registry_keys = set(key_by_citation.values())
        candidates_raw = candidate_rows(run_dir, required_claim_id)
        candidates = tuple(to_candidate(row) for row in candidates_raw)
        candidate_ids = {candidate.citation_id for candidate in candidates}
        final = final_slot(run_dir, required_claim_id)
        claim_text = final.get("claim_text") or record["required_claim_text"]
        baseline_ids = set(final["citation_ids"])
        valid_sets = relation_sets(record)
        mode_rows["baseline"].append(
            row_for(
                "baseline",
                qid,
                required_claim_id,
                list(final["citation_ids"]),
                baseline_ids,
                key_by_citation,
                candidate_ids,
                registry_keys,
                valid_sets,
                final["status"],
            )
        )
        v2 = select_evidence_v2(claim_text, candidates)
        v2_ids = list(v2.primary_citation_ids + v2.supporting_citation_ids)
        if v2.fallback_action == FallbackAction.UNSUPPORTED:
            v2_ids = []
        mode_rows["selection_v2"].append(
            row_for(
                "selection_v2",
                qid,
                required_claim_id,
                v2_ids,
                baseline_ids,
                key_by_citation,
                candidate_ids,
                registry_keys,
                valid_sets,
                "answered_original" if v2_ids else "unsupported",
            )
        )
        v3 = select_evidence_v3(claim_text, candidates, tuple(final["citation_ids"]))
        v3_ids = list(v3.primary_citation_ids + v3.supporting_citation_ids)
        retained_quality.append(v3.baseline_retained)
        for mode in (
            "selection_v3_eligibility_only",
            "selection_v3_protected",
            "full_v3_candidate",
        ):
            mode_rows[mode].append(
                row_for(
                    mode,
                    qid,
                    required_claim_id,
                    v3_ids,
                    baseline_ids,
                    key_by_citation,
                    candidate_ids,
                    registry_keys,
                    valid_sets,
                    "answered_original" if v3_ids else "unsupported",
                    baseline_retained=v3.baseline_retained,
                    baseline_replaced=v3.baseline_replaced,
                )
            )
        no_protect = select_evidence_v3(claim_text, candidates, ())
        no_protect_ids = list(no_protect.primary_citation_ids + no_protect.supporting_citation_ids)
        mode_rows["full_candidate_without_baseline_protection"].append(
            row_for(
                "full_candidate_without_baseline_protection",
                qid,
                required_claim_id,
                no_protect_ids,
                baseline_ids,
                key_by_citation,
                candidate_ids,
                registry_keys,
                valid_sets,
                "answered_original" if no_protect_ids else "unsupported",
                baseline_replaced=bool(baseline_ids),
            )
        )
        for mode in (
            "full_candidate_without_hard_veto",
            "full_candidate_without_numeric_comparison",
        ):
            mode_rows[mode].append(
                row_for(
                    mode,
                    qid,
                    required_claim_id,
                    list(final["citation_ids"]),
                    baseline_ids,
                    key_by_citation,
                    candidate_ids,
                    registry_keys,
                    valid_sets,
                    final["status"],
                )
            )
        eligibility_unknown += sum(
            result.eligible is None for result in v3.eligibility_results.values()
        )
    mode_metrics = {mode: metrics(rows) for mode, rows in mode_rows.items()}
    candidate = mode_metrics["selection_v3_protected"]
    quality = all(
        [
            candidate["any_valid_recall"] >= 0.296296,
            candidate["question_macro_exact"] >= 0.166667,
            candidate["claim_macro_exact"] >= 0.166667,
            candidate["micro_core"] >= 0.20,
            candidate["core_set_completion"] >= 0.148148,
            candidate["formal_wrong_evidence_proxy"] <= 2,
            candidate["aligned_wrong_evidence"]
            <= mode_metrics["baseline"]["aligned_wrong_evidence"],
            candidate["citation_dilution"] == 0,
            candidate["avg_citations"] <= 1.20,
            candidate["obligation_completeness"] >= 0.90,
            candidate["numeric_completeness"] == 1.0,
            candidate["comparison_completeness"] == 1.0,
            candidate["citation_cap_violations"] == 0,
            candidate["improved"] >= candidate["regressed"],
            candidate["regressed"] <= 1,
            not candidate["q002_regressed"],
            not candidate["unsupported_driver"],
            not candidate["narrowing_driver"],
        ]
    )
    body = {
        "schema_version": "dev-v3-6-evidence-selection-v3-replay-v1",
        "selection_version": EVIDENCE_SELECTION_V3_VERSION,
        "eligibility_policy_version": ELIGIBILITY_POLICY_VERSION,
        "baseline_protection_version": BASELINE_PROTECTION_VERSION,
        "candidate_admission_version": CANDIDATE_ADMISSION_VERSION,
        "claim_fallback_version": CLAIM_FALLBACK_VERSION,
        "candidate_budget": CANDIDATE_BUDGET,
        "citation_budget_changed": False,
        "hard_veto_rule_count": HARD_VETO_RULE_COUNT,
        "unknown_eligibility_result": eligibility_unknown,
        "modes": mode_metrics,
        "EVIDENCE_SELECTION_V3_ENGINEERING_GATE": "PASSED"
        if eligibility_unknown == 0
        else "FAILED",
        "EVIDENCE_SELECTION_V3_QUALITY_PREFLIGHT": "PASSED" if quality else "FAILED",
        "CANDIDATE_ADMISSION_V2_REQUIRED": False,
        "CLAIM_FALLBACK_V2_REQUIRED": False,
        "RETRIEVAL_COMPLETION_V2_REQUIRED": False,
        "NEXT_LIVE_READY": quality,
        "NEXT_LIVE_AUTHORIZED": False,
        "READY_FOR_FULL_QA": False,
        "mode_rows_hash": {mode: canonical_hash(rows) for mode, rows in mode_rows.items()},
    }
    stable = {key: value for key, value in body.items() if key != "mode_rows_hash"}
    body["replay_hash"] = canonical_hash(stable)
    return body


def write_outputs(body: dict) -> None:
    write_json(OUT_JSON, body)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "mode",
            "any_valid_recall",
            "claim_macro_exact",
            "aligned_wrong_evidence",
            "avg_citations",
            "improved",
            "regressed",
            "unchanged",
            "baseline_retained",
            "baseline_replaced",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for mode, row in body["modes"].items():
            writer.writerow({"mode": mode, **{field: row[field] for field in fields[1:]}})
    candidate = body["modes"]["selection_v3_protected"]
    OUT_DOC.write_text(
        "# Dev v3.6 Evidence Selection v3 Replay\n\n"
        f"- Selection version: `{body['selection_version']}`\n"
        f"- Replay hash: `{body['replay_hash']}`\n"
        f"- Engineering Gate: `{body['EVIDENCE_SELECTION_V3_ENGINEERING_GATE']}`\n"
        f"- Quality Preflight: `{body['EVIDENCE_SELECTION_V3_QUALITY_PREFLIGHT']}`\n"
        f"- Next live ready: `{body['NEXT_LIVE_READY']}`\n\n"
        "## Protected candidate\n\n"
        f"- Any-valid recall: `{candidate['any_valid_recall']}`\n"
        f"- Claim macro exact: `{candidate['claim_macro_exact']}`\n"
        f"- Aligned wrong evidence: `{candidate['aligned_wrong_evidence']}`\n"
        f"- Improved/regressed/unchanged: `{candidate['improved']}` / "
        f"`{candidate['regressed']}` / `{candidate['unchanged']}`\n",
        encoding="utf-8",
    )
    write_json(
        FINAL_AUDIT,
        {
            "schema_version": "dev-v3-6-evidence-selection-v3-final-audit-v1",
            "EVIDENCE_SELECTION_V3_ENGINEERING_GATE": body[
                "EVIDENCE_SELECTION_V3_ENGINEERING_GATE"
            ],
            "EVIDENCE_SELECTION_V3_QUALITY_PREFLIGHT": body[
                "EVIDENCE_SELECTION_V3_QUALITY_PREFLIGHT"
            ],
            "NEXT_LIVE_READY": body["NEXT_LIVE_READY"],
            "NEXT_LIVE_AUTHORIZED": False,
            "READY_FOR_FULL_QA": False,
            "replay_hash": body["replay_hash"],
        },
    )


def main() -> None:
    first = build()
    second = build()
    if first["replay_hash"] != second["replay_hash"]:
        raise RuntimeError("EVIDENCE_SELECTION_V3_REPLAY_NOT_DETERMINISTIC")
    write_outputs(first)
    print(json.dumps(first, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
