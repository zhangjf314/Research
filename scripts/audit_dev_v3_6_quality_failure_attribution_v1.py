"""Finalize Stage 13.22 Dev v3.6 quality-failure attribution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "evaluation"
DOCS = ROOT / "docs"

OUT_JSON = DATA / "dev-v3-6-quality-failure-attribution-v1.json"
OUT_DOC = DOCS / "dev-v3-6-quality-failure-attribution-v1.md"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    summary = read_json(DATA / "evidence-qa-dev-v3-6.json")
    funnel = read_json(DATA / "dev-v3-6-evidence-funnel-metrics-v1.json")
    replay = read_json(DATA / "dev-v3-6-evidence-selection-v2-replay.json")
    selection_audit = read_json(DATA / "dev-v3-6-evidence-selection-v2-final-audit.json")
    leakage = read_json(DATA / "evidence-selection-v2-feature-leakage-audit.json")
    final = {
        "schema_version": "dev-v3-6-quality-failure-attribution-v1",
        "DEV_V3_6_QUALITY_FAILURE_ATTRIBUTION": "COMPLETE"
        if funnel["total_required_claims"] == 27
        and "UNKNOWN" not in funnel["primary_root_cause_distribution"]
        else "INCOMPLETE",
        "PRIMARY_QUALITY_BOTTLENECK": funnel["primary_quality_bottleneck"],
        "EVIDENCE_SELECTION_V2_ENGINEERING_GATE": selection_audit[
            "EVIDENCE_SELECTION_V2_ENGINEERING_GATE"
        ],
        "EVIDENCE_SELECTION_V2_QUALITY_PREFLIGHT": selection_audit[
            "EVIDENCE_SELECTION_V2_QUALITY_PREFLIGHT"
        ],
        "RETRIEVAL_COMPLETION_V2_REQUIRED": selection_audit[
            "RETRIEVAL_COMPLETION_V2_REQUIRED"
        ],
        "NEXT_LIVE_READY": False,
        "NEXT_LIVE_AUTHORIZED": False,
        "READY_FOR_FULL_QA": False,
        "HUMAN_CITATION_REVIEW_DEFERRED": True,
        "full_qa_executed": False,
        "deep_research_executed": False,
        "new_live_executed": False,
        "llm_called_in_stage13_22": False,
        "embedding_api_called": False,
        "reranker_called": False,
        "payload_v4_modified": False,
        "evidence_presentation_v2_modified": False,
        "stage13_21_results_modified": False,
        "production_ready": False,
        "v1_0": False,
        "current_release": "v0.9.0-rc3",
        "funnel": funnel,
        "replay": {
            "selection_version": replay["selection_version"],
            "replay_hash": replay["replay_hash"],
            "modes": replay["modes"],
        },
        "feature_leakage": leakage,
        "accounting": {
            "effective_active_reservations": summary["all_manifest_conservative"][
                "effective_active_reservations"
            ],
            "double_settlement_count": summary["all_manifest_conservative"][
                "double_settlement_count"
            ],
        },
    }
    OUT_JSON.write_text(json.dumps(final, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    candidate = replay["modes"]["selection_v2_only"]
    OUT_DOC.write_text(
        "# Dev v3.6 Quality Failure Attribution\n\n"
        f"- Attribution: `{final['DEV_V3_6_QUALITY_FAILURE_ATTRIBUTION']}`\n"
        f"- Primary bottleneck: `{final['PRIMARY_QUALITY_BOTTLENECK']}`\n"
        f"- Evidence Selection v2 Engineering Gate: "
        f"`{final['EVIDENCE_SELECTION_V2_ENGINEERING_GATE']}`\n"
        f"- Evidence Selection v2 Quality Preflight: "
        f"`{final['EVIDENCE_SELECTION_V2_QUALITY_PREFLIGHT']}`\n"
        f"- Retrieval completion required: `{final['RETRIEVAL_COMPLETION_V2_REQUIRED']}`\n"
        f"- Next live ready: `{final['NEXT_LIVE_READY']}`\n"
        f"- Next live authorized: `{final['NEXT_LIVE_AUTHORIZED']}`\n"
        f"- Human citation review deferred: `{final['HUMAN_CITATION_REVIEW_DEFERRED']}`\n\n"
        "## Funnel counts\n\n"
        f"- F2 retrieval misses: `{funnel['f2_retrieval_misses']}`\n"
        f"- F3 candidate pruning misses: `{funnel['f3_candidate_pruning_misses']}`\n"
        f"- F5 selection misses: `{funnel['f5_policy_selection_misses']}`\n"
        f"- F6 selected-not-cited: `{funnel['f6_selected_not_cited']}`\n"
        f"- F7 support completeness failures: "
        f"`{funnel['f7_support_completeness_failures']}`\n"
        f"- Narrowing losses: `{funnel['narrowing_losses']}`\n"
        f"- Unsupported losses: `{funnel['unsupported_losses']}`\n\n"
        "## Selection v2 candidate\n\n"
        f"- Any-valid recall: `{candidate['any_valid_recall']}`\n"
        f"- Question macro exact: `{candidate['question_macro_exact']}`\n"
        f"- Claim macro exact: `{candidate['claim_macro_exact']}`\n"
        f"- Micro core relation: `{candidate['micro_core_relation']}`\n"
        f"- Core-set completion: `{candidate['core_set_completion']}`\n"
        f"- Wrong evidence: `{candidate['wrong_evidence']}`\n"
        f"- Improvement/regression/unchanged: `{candidate['improvement']}` / "
        f"`{candidate['regression']}` / `{candidate['unchanged']}`\n\n"
        "Selection v2 remains an offline candidate. Because quality preflight failed, no new "
        "Dev live run, Human Citation Audit, Full QA, or Deep Research is authorized.\n",
        encoding="utf-8",
    )
    print(json.dumps(final, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
