# ruff: noqa: E501
"""Summarize imported Dev v2 citation review and matcher adjudication."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from collections.abc import Callable
from typing import Any

try:
    from scripts.evidence_qa_dev_lib_v1 import DATA, DOCS, read_jsonl
    from scripts.review_evidence_qa_dev_v2_citations_v1 import AUDIT, validate
except ModuleNotFoundError:
    from evidence_qa_dev_lib_v1 import DATA, DOCS, read_jsonl  # type: ignore[no-redef]
    from review_evidence_qa_dev_v2_citations_v1 import AUDIT, validate  # type: ignore[no-redef]

COVERAGE = DATA / "dev-v2-claim-coverage-audit-v1.jsonl"
COVERAGE_CSV = DATA / "dev-v2-claim-coverage-audit-v1.csv"
SUMMARY_JSON = DATA / "evidence-qa-dev-v2-citation-audit-summary-v1.json"
SUMMARY_DOC = DOCS / "evidence-qa-dev-v2-citation-audit-summary-v1.md"
ADJUDICATION_JSON = DATA / "dev-v2-claim-coverage-human-adjudication-v1.json"
ADJUDICATION_DOC = DOCS / "dev-v2-claim-coverage-human-adjudication-v1.md"
COUNTERFACTUAL_JSON = DATA / "dev-v2-claim-coverage-counterfactual-v1.json"
COUNTERFACTUAL_DOC = DOCS / "dev-v2-claim-coverage-counterfactual-v1.md"
STRICT = {"fully_supported"}
LENIENT = {"fully_supported", "partially_supported"}
LABELS = ["fully_supported", "partially_supported", "related_but_insufficient", "unsupported", "gold_annotation_too_narrow", "ambiguous_claim", "malformed_evidence"]
IMPROVED = {"q001", "q002", "q007", "q008", "q019"}
REGRESSED = {"q004", "q013", "q015", "q050"}


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = Counter(row["human_label"] for row in rows)
    return {"n": len(rows), "labels": {name: labels.get(name, 0) for name in LABELS}, "strict_support_rate": round(sum(row["human_label"] in STRICT for row in rows) / len(rows), 6) if rows else None, "lenient_support_rate": round(sum(row["human_label"] in LENIENT for row in rows) / len(rows), 6) if rows else None}


def grouped(rows: list[dict[str, Any]], key: Callable[[dict[str, Any]], str]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[key(row)].append(row)
    return {name: metrics(items) for name, items in sorted(groups.items())}


def main() -> None:
    rows = read_jsonl(AUDIT)
    validate(rows)
    if not all(row["human_review_status"] == "approved" for row in rows):
        raise RuntimeError("all 57 citation samples must be approved")
    evidence = {(row["paper_id"], int(row["page"]), row["block_id"]): row for row in read_jsonl(DATA / "evidence-corpus-v1.jsonl")}
    claims = {row["claim_id"]: row for row in read_jsonl(DATA / "claim-units-v1.jsonl")}
    for row in rows:
        best = row.get("required_claim_match", {}).get("best") or {}
        row["_claim_role"] = claims.get(best.get("required_claim_id"), {}).get("claim_role", "unknown")
        triple = (row["citation_triple"]["paper_id"], int(row["citation_triple"]["page"]), row["citation_triple"]["block_id"])
        row["_evidence_roles"] = evidence[triple].get("evidence_roles") or ["unknown"]
    exact = [row for row in rows if row["automated_labels"]["exact_gold"]]
    same_page = [row for row in rows if row["automated_labels"]["same_page"] and not row["automated_labels"]["exact_gold"]]
    semantic = [row for row in rows if row["semantic_token_signal"] >= 0.35]
    unsupported_signal = [row for row in rows if row["automated_labels"]["unsupported_signal"]]
    payload = {"schema_version": "evidence-qa-dev-v2-citation-audit-summary-v1", "review_type": "AI-assisted manual audit", "total": 57, "reviewed": 57, **metrics(rows), "strata": {"exact_gold": metrics(exact), "same_page_non_exact": metrics(same_page), "semantic_support_signal": metrics(semantic), "automated_unsupported_signal": metrics(unsupported_signal), "evidence_source": grouped(rows, lambda row: row["evidence_source"]), "question_id": grouped(rows, lambda row: row["question_id"]), "category": grouped(rows, lambda row: row["category"]), "difficulty": grouped(rows, lambda row: row["difficulty"]), "block_type": grouped(rows, lambda row: row["block_type"]), "claim_role": grouped(rows, lambda row: row["_claim_role"]), "evidence_role": grouped([{**row, "_one_role": role} for row in rows for role in row["_evidence_roles"]], lambda row: row["_one_role"]), "improved_questions": metrics([row for row in rows if row["question_id"] in IMPROVED]), "regressed_questions": metrics([row for row in rows if row["question_id"] in REGRESSED])}, "focus_questions": {qid: metrics([row for row in rows if row["question_id"] == qid]) for qid in ("q002", "q007", "q013", "q050")}, "q019_compound_claim": {**metrics([row for row in rows if row["question_id"] == "q019"]), "single_citation_partially_supports_compound_claim": sum(row["human_label"] == "partially_supported" for row in rows if row["question_id"] == "q019"), "interpretation": "q019 uses one compound generated claim with ten citations; nine citations support only part of that compound claim and one is related but insufficient."}, "gold_annotation_too_narrow_present": any(row["human_label"] == "gold_annotation_too_narrow" for row in rows), "malformed_evidence_present": any(row["human_label"] == "malformed_evidence" for row in rows), "comparison_to_dev_v1_audit": {"dev_v1_samples": 24, "dev_v1_strict": 0.625, "dev_v1_lenient": 0.708333, "direct_ab_comparison_valid": False}, "limitations": ["AI-assisted manual audit, not an independent human double-blind review.", "Fixed 10-question Dev batch with 57 claim-citation pairs.", "Cannot be extrapolated to Full-50 or production citation precision.", "Dev v1 and Dev v2 audits differ in sample scale and generation protocol."]}
    SUMMARY_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    coverage = read_jsonl(COVERAGE)
    historical = sum(row["coverage_credit"] for row in coverage)
    diagnostic = sum(row.get("formal_coverage_credit", row["coverage_credit"]) for row in coverage)
    candidates = [row for row in coverage if row.get("matcher_human_decision")]
    if historical != 14 or diagnostic != 16 or len(candidates) != 4:
        raise RuntimeError("coverage adjudication totals do not match frozen protocol")
    flat = [{key: json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else value for key, value in row.items()} for row in coverage]
    fieldnames = sorted({key for row in flat for key in row})
    with COVERAGE_CSV.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat)
    decisions = {row["required_claim_id"]: {key: row.get(key) for key in ("question_id", "matcher_human_decision", "historical_formal_coverage_credit", "formal_coverage_credit", "diagnostic_partial_credit", "coverage_failure_stage_before_review", "coverage_failure_stage_after_review", "reviewer", "reviewed_at", "review_notes")} for row in candidates}
    adjudication = {"schema_version": "dev-v2-claim-coverage-human-adjudication-v1", "historical_formal_dev_v2": {"covered": 14, "required": 27, "rate": round(14 / 27, 6), "historical_metric_modified": False}, "human_adjudicated_diagnostic": {"covered": 16, "required": 27, "rate": round(16 / 27, 6), "diagnostic_only": True}, "matcher_candidates": decisions, "review_type": "AI-assisted manual audit", "limitations": ["The diagnostic rate does not replace the historical formal Dev v2 metric.", "It does not show that prompt v2 solved silent claim omission.", "It cannot be extrapolated to Full-50 or production."]}
    ADJUDICATION_JSON.write_text(json.dumps(adjudication, ensure_ascii=False, indent=2), encoding="utf-8")
    cf = json.loads(COUNTERFACTUAL_JSON.read_text(encoding="utf-8"))
    after = Counter()
    for row in coverage:
        if row["coverage_credit"]:
            after["covered"] += 1
        elif row.get("formal_coverage_credit") == 1:
            after["covered_after_human_matcher_adjudication"] += 1
        elif row.get("matcher_human_decision") in {"partial_match", "false_positive"}:
            after[row["matcher_human_decision"]] += 1
        elif row["coverage_failure_stage"]:
            after[row["coverage_failure_stage"]] += 1
    cf["human_adjudication"] = {"historical_formal_coverage": 14 / 27, "diagnostic_coverage": 16 / 27, "formal_metric_modified": False, "failure_stage_after_review": dict(sorted(after.items())), "matcher_candidates": decisions, "prompt_v3_implications": ["Explicit required-claim slots are required to prevent silent omission.", "Merged claims must still occupy each required-claim slot.", "Partial matches cannot receive full coverage credit.", "Matcher thresholds require auditable adjudication.", "q050 malformed JSON remains an independent engineering failure.", "Adjacent completion cannot be removed without evidence-loss checks."]}
    COUNTERFACTUAL_JSON.write_text(json.dumps(cf, ensure_ascii=False, indent=2), encoding="utf-8")
    SUMMARY_DOC.write_text("# Dev v2 Citation Audit Summary\n\n- Review: **57/57 approved** (AI-assisted manual audit)\n- Fully / partially / related / unsupported: **35 / 17 / 3 / 2**\n- Strict support: **35/57 = 0.614035**\n- Lenient support: **52/57 = 0.912281**\n- Original selected strict/lenient: **0.697674 / 0.906977**\n- Adjacent completion strict/lenient: **0.357143 / 0.928571**\n- Gold annotation too narrow: **0**\n- Malformed evidence: **0**\n\nq019 contains one compound generated claim and ten citations: nine are partially supportive and one is related but insufficient, so no single citation fully supports the entire compound claim.\n\nThis is an AI-assisted manual audit over a fixed 10-question Dev set. It is not an independent double-blind human audit and cannot be extrapolated to Full-50 or production precision.\n", encoding="utf-8")
    ADJUDICATION_DOC.write_text("# Dev v2 Claim Coverage Human Adjudication\n\n- Historical formal Dev v2 coverage: **14/27 = 0.518519** (unchanged)\n- Human-adjudicated diagnostic coverage: **16/27 = 0.592593**\n- q001 merged claim: full diagnostic credit\n- q002 valid match: full diagnostic credit\n- q004 partial match: formal credit 0, diagnostic partial credit 0.5\n- q015 false positive: credit 0\n\nThe diagnostic value does not replace the historical metric or prove prompt v2 solved claim omission.\n", encoding="utf-8")
    COUNTERFACTUAL_DOC.write_text("# Dev v2 Claim Coverage Counterfactual\n\n- Historical automatic coverage: **14/27 = 0.518519**\n- Human-adjudicated diagnostic coverage: **16/27 = 0.592593**\n- Historical metric modified: **False**\n\nPrompt v2 does not enumerate required claims, permits silent omission and merged claims, and relies on a matcher with both false negatives and false positives. q050 remains a separate malformed-JSON engineering failure. Adjacent completion cannot simply be removed because the offline replay loses Gold evidence. Required-claim slots remain the selected general repair direction.\n", encoding="utf-8")
    print(json.dumps({"reviewed": 57, "strict": payload["strict_support_rate"], "lenient": payload["lenient_support_rate"], "historical_coverage": 14 / 27, "diagnostic_coverage": 16 / 27}))


if __name__ == "__main__":
    main()
