"""Finalize Stage 13.23 readiness without authorizing live execution."""

from __future__ import annotations

import json

try:
    from scripts.stage13_23_common import DATA, DOCS, read_json, write_json
except ModuleNotFoundError:
    from stage13_23_common import DATA, DOCS, read_json, write_json  # type: ignore[no-redef]

OUT = DATA / "stage13-23-selection-v3-readiness-v1.json"
DOC = DOCS / "stage13-23-selection-v3-readiness-v1.md"


def main() -> None:
    wrong = read_json(DATA / "evidence-selection-v2-wrong-evidence-audit-v1.json")
    alignment = read_json(DATA / "wrong-evidence-metric-alignment-v1.json")
    replay = read_json(DATA / "dev-v3-6-evidence-selection-v3-replay.json")
    leakage = read_json(DATA / "evidence-selection-v3-feature-leakage-audit.json")
    candidate = replay["modes"]["selection_v3_protected"]
    ready = (
        wrong["unknown"] == 0
        and alignment["WRONG_EVIDENCE_METRIC_ALIGNMENT"] == "PASSED"
        and replay["EVIDENCE_SELECTION_V3_ENGINEERING_GATE"] == "PASSED"
        and replay["EVIDENCE_SELECTION_V3_QUALITY_PREFLIGHT"] == "PASSED"
        and leakage["gate"] == "PASSED"
    )
    body = {
        "schema_version": "stage13-23-selection-v3-readiness-v1",
        "SELECTION_V2_FAILURE_ATTRIBUTION": "COMPLETE"
        if wrong["records"] > 0 and wrong["unknown"] == 0
        else "INCOMPLETE",
        "PRIMARY_SELECTION_V2_FAILURE": max(
            wrong["root_cause_distribution"],
            key=wrong["root_cause_distribution"].get,
        )
        if wrong["root_cause_distribution"]
        else "none",
        "WRONG_EVIDENCE_METRIC_ALIGNMENT": alignment["WRONG_EVIDENCE_METRIC_ALIGNMENT"],
        "EVIDENCE_SELECTION_V3_ENGINEERING_GATE": replay[
            "EVIDENCE_SELECTION_V3_ENGINEERING_GATE"
        ],
        "EVIDENCE_SELECTION_V3_QUALITY_PREFLIGHT": replay[
            "EVIDENCE_SELECTION_V3_QUALITY_PREFLIGHT"
        ],
        "CANDIDATE_ADMISSION_V2_REQUIRED": replay["CANDIDATE_ADMISSION_V2_REQUIRED"],
        "CLAIM_FALLBACK_V2_REQUIRED": replay["CLAIM_FALLBACK_V2_REQUIRED"],
        "RETRIEVAL_COMPLETION_V2_REQUIRED": False,
        "NEXT_LIVE_READY": ready,
        "NEXT_LIVE_AUTHORIZED": False,
        "READY_FOR_FULL_QA": False,
        "HUMAN_CITATION_REVIEW_DEFERRED": True,
        "production_ready": False,
        "v1_0": False,
        "current_release": "v0.9.0-rc3",
        "selection_v2_wrong_evidence_total": wrong["records"],
        "wrong_evidence_root_cause_distribution": wrong["root_cause_distribution"],
        "unknown_classification": wrong["unknown"],
        "aligned_baseline_wrong_evidence": alignment["aligned_baseline_wrong_evidence"],
        "aligned_selection_v2_wrong_evidence": alignment[
            "aligned_selection_v2_wrong_evidence"
        ],
        "selection_v3_metrics": candidate,
        "selection_v3_replay_hash": replay["replay_hash"],
        "feature_leakage": leakage,
        "stage13_21_results_modified": False,
        "stage13_22_results_modified": False,
        "payload_v4_modified": False,
        "evidence_presentation_v2_modified": False,
        "prompt_modified": False,
        "live_llm_executed": False,
        "embedding_api_executed": False,
        "reranker_executed": False,
        "new_live_executed": False,
        "full_qa_executed": False,
        "deep_research_executed": False,
    }
    write_json(OUT, body)
    DOC.write_text(
        "# Stage 13.23 Selection v3 Readiness\n\n"
        f"- Selection v2 attribution: `{body['SELECTION_V2_FAILURE_ATTRIBUTION']}`\n"
        f"- Primary Selection v2 failure: `{body['PRIMARY_SELECTION_V2_FAILURE']}`\n"
        f"- Metric alignment: `{body['WRONG_EVIDENCE_METRIC_ALIGNMENT']}`\n"
        f"- Selection v3 Engineering Gate: `{body['EVIDENCE_SELECTION_V3_ENGINEERING_GATE']}`\n"
        f"- Selection v3 Quality Preflight: `{body['EVIDENCE_SELECTION_V3_QUALITY_PREFLIGHT']}`\n"
        f"- Next live ready: `{body['NEXT_LIVE_READY']}`\n"
        f"- Next live authorized: `{body['NEXT_LIVE_AUTHORIZED']}`\n\n"
        "No live LLM, Embedding API, Reranker, Human Citation Audit, Full QA, or Deep Research "
        "was run in Stage 13.23.\n",
        encoding="utf-8",
    )
    print(json.dumps(body, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
