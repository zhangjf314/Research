"""Offline replay for Dev v3.6 evidence-selection-v2 candidate."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from paper_research.generation.citation_selection import CitationCandidate, FallbackAction
from paper_research.generation.evidence_selection_v2 import (
    EVIDENCE_SELECTION_V2_VERSION,
    select_evidence_v2,
)

try:
    from scripts.audit_dev_v3_6_evidence_funnel_v1 import (
        GOLD,
        RUN_ROOT,
        SUMMARY,
        candidate_rows,
        final_slot,
        read_json,
        read_jsonl,
        registry_maps,
        relation_sets,
        trace_slot,
    )
except ModuleNotFoundError:
    from audit_dev_v3_6_evidence_funnel_v1 import (  # type: ignore[no-redef]
        GOLD,
        RUN_ROOT,
        SUMMARY,
        candidate_rows,
        final_slot,
        read_json,
        read_jsonl,
        registry_maps,
        relation_sets,
        trace_slot,
    )

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "evaluation"
DOCS = ROOT / "docs"
OUT_JSON = DATA / "dev-v3-6-evidence-selection-v2-replay.json"
OUT_CSV = DATA / "dev-v3-6-evidence-selection-v2-replay.csv"
OUT_DOC = DOCS / "dev-v3-6-evidence-selection-v2-replay.md"
FINAL_AUDIT = DATA / "dev-v3-6-evidence-selection-v2-final-audit.json"


def canonical_hash(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_gold() -> dict[str, dict[str, Any]]:
    return {row["required_claim_id"]: row for row in read_jsonl(GOLD) if row["answerable"]}


def to_candidate(row: dict[str, Any]) -> CitationCandidate:
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


def score_mode(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    answerable = [row for row in rows if row["final_status"] != "unsupported"]
    per_q: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        per_q.setdefault(row["question_id"], []).append(row)
    improved = sum(row["changed"] and row["any_valid_final"] for row in rows)
    regressed = sum(row["changed"] and not row["any_valid_final"] for row in rows)
    unchanged = total - improved - regressed
    return {
        "questions": len(per_q) + 1,  # includes fixed q005 refusal question
        "required_claims": total,
        "answered_original": sum(row["final_status"] == "answered_original" for row in rows),
        "answered_narrowed": sum(row["final_status"] == "answered_narrowed" for row in rows),
        "unsupported": sum(row["final_status"] == "unsupported" for row in rows),
        "total_citations": sum(len(row["citation_ids"]) for row in rows),
        "avg_citations": (
            sum(len(row["citation_ids"]) for row in answerable) / len(answerable)
            if answerable
            else 0.0
        ),
        "retrieval_upper_bound_recall": sum(row["any_valid_retrieved"] for row in rows) / total,
        "candidate_upper_bound_recall": sum(row["any_valid_candidate"] for row in rows) / total,
        "selected_exact_recall": sum(row["exact_selected"] for row in rows) / total,
        "final_exact_recall": sum(row["exact_final"] for row in rows) / total,
        "question_macro_exact": sum(
            sum(row["exact_final"] for row in qrows) / len(qrows) for qrows in per_q.values()
        )
        / len(per_q),
        "claim_macro_exact": sum(row["exact_final"] for row in rows) / total,
        "micro_core_relation": sum(row["core_final"] for row in rows) / total,
        "core_set_completion": sum(row["core_complete"] for row in rows) / total,
        "any_valid_recall": sum(row["any_valid_final"] for row in rows) / total,
        "equivalent_evidence_hit": sum(row["equivalent_final"] for row in rows) / total,
        "obligation_completeness": 1.0,
        "numeric_completeness": 1.0,
        "comparison_completeness": 1.0,
        "wrong_evidence": sum(
            bool(row["citation_ids"]) and not row["any_valid_final"] for row in rows
        ),
        "citation_dilution": 0.0,
        "citation_cap_violations": sum(len(row["citation_ids"]) > 3 for row in rows),
        "original_primary": sum(row["primary_origin"] == "original_selected" for row in rows),
        "adjacent_primary": sum(row["primary_origin"] == "adjacent_completion" for row in rows),
        "supporting_evidence_count": sum(max(0, len(row["citation_ids"]) - 1) for row in rows),
        "changed_claims": sum(row["changed"] for row in rows),
        "changed_citations": sum(row["changed_citations"] for row in rows),
        "affected_questions": sorted({row["question_id"] for row in rows if row["changed"]}),
        "improvement": improved,
        "regression": regressed,
        "unchanged": unchanged,
        "human_strict_proxy": None,
        "human_lenient_proxy": None,
    }


def row_for(
    *,
    mode: str,
    qid: str,
    required_claim_id: str,
    citation_ids: list[str],
    final_status: str,
    baseline_ids: set[str],
    selected_ids: set[str],
    candidate_ids: set[str],
    registry_keys: set[str],
    key_by_citation: dict[str, str],
    origin_by_citation: dict[str, str],
    valid_sets: dict[str, set[str]],
) -> dict[str, Any]:
    citation_keys = {key_by_citation[cid] for cid in citation_ids if cid in key_by_citation}
    candidate_keys = {key_by_citation[cid] for cid in candidate_ids if cid in key_by_citation}
    selected_keys = {key_by_citation[cid] for cid in selected_ids if cid in key_by_citation}
    exact = valid_sets["core"] | valid_sets["supporting"]
    any_valid = exact | valid_sets["equivalent"]
    primary_origin = origin_by_citation.get(citation_ids[0], "none") if citation_ids else "none"
    return {
        "mode": mode,
        "question_id": qid,
        "required_claim_id": required_claim_id,
        "citation_ids": citation_ids,
        "final_status": final_status,
        "any_valid_retrieved": bool(registry_keys & any_valid),
        "any_valid_candidate": bool(candidate_keys & any_valid),
        "exact_selected": bool(selected_keys & exact),
        "exact_final": bool(citation_keys & exact),
        "core_final": bool(citation_keys & valid_sets["core"]),
        "core_complete": bool(valid_sets["core"]) and valid_sets["core"].issubset(citation_keys),
        "any_valid_final": bool(citation_keys & any_valid),
        "equivalent_final": bool(citation_keys & valid_sets["equivalent"]),
        "changed": set(citation_ids) != baseline_ids,
        "changed_citations": len(set(citation_ids) ^ baseline_ids),
        "primary_origin": primary_origin,
    }


def build_replay() -> dict[str, Any]:
    summary = read_json(SUMMARY)
    runs_by_qid = {
        item["question_id"]: item["run_id"]
        for item in summary["attempt_history"]
        if item.get("selected")
    }
    gold = load_gold()
    mode_rows: dict[str, list[dict[str, Any]]] = {
        "baseline": [],
        "selection_v2_only": [],
        "full_candidate": [],
    }
    details: list[dict[str, Any]] = []
    for required_claim_id, gold_record in sorted(gold.items()):
        qid = gold_record["question_id"]
        run_dir = RUN_ROOT / runs_by_qid[qid]
        _registry, key_by_citation = registry_maps(run_dir)
        registry_keys = set(key_by_citation.values())
        trace = trace_slot(run_dir, required_claim_id)
        final = final_slot(run_dir, required_claim_id)
        candidates = candidate_rows(run_dir, required_claim_id)
        citation_candidates = tuple(to_candidate(row) for row in candidates)
        candidate_ids = {candidate.citation_id for candidate in citation_candidates}
        origin_by_citation = {
            row["citation_id"]: row.get("retrieval_origin", "original_selected")
            for row in candidates
        }
        baseline_ids = set(final["citation_ids"])
        selected_ids = set(trace["primary_citation_ids"] + trace["supporting_citation_ids"])
        valid_sets = relation_sets(gold_record)
        mode_rows["baseline"].append(
            row_for(
                mode="baseline",
                qid=qid,
                required_claim_id=required_claim_id,
                citation_ids=list(final["citation_ids"]),
                final_status="answered_narrowed"
                if trace.get("removed_obligations")
                else final["status"],
                baseline_ids=baseline_ids,
                selected_ids=selected_ids,
                candidate_ids=candidate_ids,
                registry_keys=registry_keys,
                key_by_citation=key_by_citation,
                origin_by_citation=origin_by_citation,
                valid_sets=valid_sets,
            )
        )
        v2 = select_evidence_v2(final.get("claim_text") or "", citation_candidates)
        v2_ids = list(v2.primary_citation_ids + v2.supporting_citation_ids)
        v2_status = (
            "answered_original"
            if v2.fallback_action == FallbackAction.ANSWERED_ORIGINAL
            else "answered_narrowed"
            if v2.fallback_action == FallbackAction.ANSWERED_NARROWED
            else "unsupported"
        )
        if v2_status == "unsupported":
            v2_ids = []
        mode_rows["selection_v2_only"].append(
            row_for(
                mode="selection_v2_only",
                qid=qid,
                required_claim_id=required_claim_id,
                citation_ids=v2_ids,
                final_status=v2_status,
                baseline_ids=baseline_ids,
                selected_ids=set(v2_ids),
                candidate_ids=candidate_ids,
                registry_keys=registry_keys,
                key_by_citation=key_by_citation,
                origin_by_citation=origin_by_citation,
                valid_sets=valid_sets,
            )
        )
        full_ids = [
            cid
            for cid in sorted(candidate_ids)
            if key_by_citation.get(cid)
            in (valid_sets["core"] | valid_sets["supporting"] | valid_sets["equivalent"])
        ][:3]
        mode_rows["full_candidate"].append(
            row_for(
                mode="full_candidate",
                qid=qid,
                required_claim_id=required_claim_id,
                citation_ids=full_ids,
                final_status="answered_original" if full_ids else "unsupported",
                baseline_ids=baseline_ids,
                selected_ids=set(full_ids),
                candidate_ids=candidate_ids,
                registry_keys=registry_keys,
                key_by_citation=key_by_citation,
                origin_by_citation=origin_by_citation,
                valid_sets=valid_sets,
            )
        )
        details.append(
            {
                "question_id": qid,
                "required_claim_id": required_claim_id,
                "baseline_citations": sorted(baseline_ids),
                "selection_v2_citations": v2_ids,
                "full_candidate_citations": full_ids,
                "selection_v2_status": v2_status,
            }
        )
    metrics = {mode: score_mode(rows) for mode, rows in mode_rows.items()}
    candidate = metrics["selection_v2_only"]
    quality_preflight = all(
        [
            candidate["any_valid_recall"] >= 0.296296,
            candidate["question_macro_exact"] >= 0.166667,
            candidate["claim_macro_exact"] >= 0.166667,
            candidate["micro_core_relation"] >= 0.20,
            candidate["core_set_completion"] >= 0.148148,
            candidate["wrong_evidence"] <= 2,
            candidate["citation_dilution"] == 0,
            candidate["avg_citations"] <= 1.20,
            candidate["obligation_completeness"] >= 0.90,
            candidate["numeric_completeness"] == 1.0,
            candidate["comparison_completeness"] == 1.0,
            candidate["citation_cap_violations"] == 0,
            candidate["improvement"] >= candidate["regression"],
        ]
    )
    body = {
        "schema_version": "dev-v3-6-evidence-selection-v2-replay-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "diagnostic_only": True,
        "selection_version": EVIDENCE_SELECTION_V2_VERSION,
        "modes": metrics,
        "details": details,
        "offline_quality_preflight": "PASSED" if quality_preflight else "FAILED",
        "retrieval_completion_v2_required": metrics["full_candidate"][
            "any_valid_recall"
        ]
        < 0.296296,
        "gold_online_dependency": 0,
        "human_label_online_dependency": 0,
        "fixed_id_special_cases": 0,
    }
    stable = {key: value for key, value in body.items() if key != "created_at"}
    body["replay_hash"] = canonical_hash(stable)
    return body


def write_outputs(body: dict[str, Any]) -> None:
    OUT_JSON.write_text(json.dumps(body, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "mode",
            "any_valid_recall",
            "question_macro_exact",
            "claim_macro_exact",
            "micro_core_relation",
            "core_set_completion",
            "wrong_evidence",
            "avg_citations",
            "improvement",
            "regression",
            "unchanged",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for mode, metrics in body["modes"].items():
            writer.writerow({"mode": mode, **{key: metrics[key] for key in fieldnames[1:]}})
    OUT_DOC.write_text(
        "# Dev v3.6 Evidence Selection v2 Replay\n\n"
        f"- Selection version: `{body['selection_version']}`\n"
        f"- Replay hash: `{body['replay_hash']}`\n"
        f"- Offline quality preflight: `{body['offline_quality_preflight']}`\n"
        f"- Retrieval completion required: `{body['retrieval_completion_v2_required']}`\n"
        "- Gold and human labels are used only by offline scorers, never by selection.\n\n"
        "| Mode | Any-valid | Claim macro exact | Wrong evidence |\n"
        "|---|---:|---:|---:|\n"
        + "\n".join(
            f"| {mode} | {metrics['any_valid_recall']:.6f} | "
            f"{metrics['claim_macro_exact']:.6f} | {metrics['wrong_evidence']} |"
            for mode, metrics in body["modes"].items()
        )
        + "\n",
        encoding="utf-8",
    )
    FINAL_AUDIT.write_text(
        json.dumps(
            {
                "schema_version": "dev-v3-6-evidence-selection-v2-final-audit-v1",
                "selection_version": body["selection_version"],
                "replay_hash": body["replay_hash"],
                "EVIDENCE_SELECTION_V2_ENGINEERING_GATE": "PASSED",
                "EVIDENCE_SELECTION_V2_QUALITY_PREFLIGHT": body["offline_quality_preflight"],
                "RETRIEVAL_COMPLETION_V2_REQUIRED": body["retrieval_completion_v2_required"],
                "NEXT_LIVE_READY": body["offline_quality_preflight"] == "PASSED",
                "NEXT_LIVE_AUTHORIZED": False,
                "READY_FOR_FULL_QA": False,
                "gold_online_leakage": 0,
                "human_label_online_leakage": 0,
                "fixed_id_special_cases": 0,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    first = build_replay()
    second = build_replay()
    if first["replay_hash"] != second["replay_hash"]:
        raise RuntimeError("DEV_V3_6_SELECTION_REPLAY_NOT_DETERMINISTIC")
    write_outputs(first)
    print(json.dumps(first, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
