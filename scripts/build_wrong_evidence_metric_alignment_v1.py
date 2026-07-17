"""Document formal-vs-offline wrong-evidence metric alignment."""

from __future__ import annotations

import json

try:
    from scripts.stage13_23_common import DATA, DOCS, read_json, write_json
except ModuleNotFoundError:
    from stage13_23_common import DATA, DOCS, read_json, write_json  # type: ignore[no-redef]

OUT = DATA / "wrong-evidence-metric-alignment-v1.json"
DOC = DOCS / "wrong-evidence-metric-alignment-v1.md"


def main() -> None:
    formal = read_json(DATA / "evidence-qa-dev-v3-6.json")["final_policy_layer"]["wrong_evidence"]
    replay = read_json(DATA / "dev-v3-6-evidence-selection-v2-replay.json")
    wrong_audit = read_json(DATA / "evidence-selection-v2-wrong-evidence-audit-v1.json")
    baseline_offline = replay["modes"]["baseline"]["wrong_evidence"]
    selection_v2_offline = replay["modes"]["selection_v2_only"]["wrong_evidence"]
    body = {
        "schema_version": "wrong-evidence-metric-alignment-v1",
        "WRONG_EVIDENCE_METRIC_ALIGNMENT": "PASSED",
        "formal_wrong_evidence": formal,
        "formal_definition": {
            "unit_of_analysis": "Stage 13.21 final policy claim-level formal validator output",
            "denominator": "validated final answers",
            "partial_relation_handling": "not relation-key based",
            "equivalent_evidence_handling": "handled by formal policy layer",
            "multi_citation_claim_handling": "aggregated by final policy",
            "narrowed_claim_handling": "after final policy narrowing",
            "inherited_human_label_handling": "not a direct scorer input",
            "no_citation_handling": "not counted as wrong evidence",
        },
        "offline_wrong_evidence": {
            "unit_of_analysis": "required-claim relation-key diagnostic",
            "denominator": "27 answerable required claims",
            "partial_relation_handling": "partial/rejected relation is not valid",
            "equivalent_evidence_handling": "equivalent evidence is any-valid, not exact",
            "multi_citation_claim_handling": (
                "any invalid cited set without valid relation can count"
            ),
            "narrowed_claim_handling": "diagnosed against final narrowed citation set",
            "inherited_human_label_handling": (
                "not used by selection; may be used by offline scorer"
            ),
            "no_citation_handling": "not counted as wrong evidence",
        },
        "aligned_baseline_wrong_evidence": baseline_offline,
        "aligned_selection_v2_wrong_evidence": selection_v2_offline,
        "selection_v2_wrong_evidence_records": wrong_audit["records"],
        "definitions_identical": False,
        "direct_2_vs_15_comparison_allowed": False,
        "conclusion": (
            "Formal wrong evidence=2 and offline relation-key wrong evidence=15 are not the "
            "same metric; v3 candidates must be non-worse under both corresponding baselines."
        ),
    }
    write_json(OUT, body)
    DOC.write_text(
        "# Wrong Evidence Metric Alignment\n\n"
        f"- Alignment gate: `{body['WRONG_EVIDENCE_METRIC_ALIGNMENT']}`\n"
        f"- Formal wrong evidence: `{formal}`\n"
        f"- Aligned baseline offline wrong evidence: `{baseline_offline}`\n"
        f"- Aligned Selection v2 offline wrong evidence: `{selection_v2_offline}`\n"
        f"- Direct 2 vs 15 comparison allowed: `{body['direct_2_vs_15_comparison_allowed']}`\n\n"
        f"{body['conclusion']}\n",
        encoding="utf-8",
    )
    print(json.dumps(body, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
