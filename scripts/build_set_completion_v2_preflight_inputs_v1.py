"""Freeze Stage 13.25 set-completion v2 inputs."""

from __future__ import annotations

import json

try:
    from scripts.stage13_25_common import DATA, DOCS, ROOT, canonical_hash, file_hash, write_json
except ModuleNotFoundError:
    from stage13_25_common import (  # type: ignore[no-redef]
        DATA,
        DOCS,
        ROOT,
        canonical_hash,
        file_hash,
        write_json,
    )

OUT_JSON = DATA / "set-completion-v2-preflight-inputs-v1.json"
OUT_DOC = DOCS / "set-completion-v2-preflight-inputs-v1.md"

FILES = {
    "stage13_21_results": DATA / "evidence-qa-dev-v3-6.json",
    "stage13_21_citation_sets": DATA / "evidence-qa-dev-v3-6-citation-audit-v1.jsonl",
    "stage13_22_evidence_funnel": DATA / "dev-v3-6-evidence-funnel-v1.jsonl",
    "stage13_22_attribution": DATA / "dev-v3-6-quality-failure-attribution-v1.json",
    "stage13_23_selection_v3_replay": DATA / "dev-v3-6-evidence-selection-v3-replay.json",
    "stage13_24_selection_v4_replay": DATA / "dev-v3-6-evidence-selection-v4-replay.json",
    "stage13_24_final_audit": DATA / "dev-v3-6-evidence-selection-v4-final-audit.json",
    "stage13_24_recall_collapse": DATA / "evidence-selection-v3-recall-collapse-v1.json",
    "stage13_24_improvement_concentration": DATA
    / "evidence-selection-v4-improvement-concentration-v1.json",
    "stage13_24_feature_leakage": DATA / "evidence-selection-v4-feature-leakage-audit.json",
    "claim_gold_freeze": DATA / "claim-evidence-gold-dev-v1-freeze.json",
    "payload_v4": DATA / "payload-contract-v4-protocol.json",
    "evidence_presentation_v2": DATA / "evidence-presentation-v2-protocol.json",
    "prompt_v3_7": DATA / "dev-v3-6-prompt-rendering-preflight-v1.json",
    "citation_budget": ROOT / "src" / "paper_research" / "generation" / "citation_selection.py",
}


def meta(path):
    return {
        "path": str(path.relative_to(ROOT)),
        "exists": path.exists(),
        "sha256": file_hash(path) if path.exists() else "",
    }


def build() -> dict:
    body = {
        "schema_version": "set-completion-v2-preflight-inputs-v1",
        "stage": "13.25",
        "baseline_head": "42d816fcf935e6e24b9fe1db375419dd0a88124e",
        "frozen_files": {name: meta(path) for name, path in FILES.items()},
        "candidate_budget": 12,
        "citation_budget": {"max_total": 3},
        "gold_freeze_offline_scoring_only": True,
        "live_llm_executed": False,
        "embedding_api_executed": False,
        "reranker_executed": False,
        "new_live_executed": False,
        "human_citation_review_deferred": True,
        "full_qa_executed": False,
        "deep_research_executed": False,
    }
    body["preflight_signature"] = canonical_hash(body)
    return body


def main() -> None:
    first = build()
    second = build()
    if first["preflight_signature"] != second["preflight_signature"]:
        raise RuntimeError("SET_COMPLETION_V2_PREFLIGHT_NOT_DETERMINISTIC")
    write_json(OUT_JSON, first)
    OUT_DOC.write_text(
        "# Set Completion v2 Preflight Inputs\n\n"
        f"- Signature: `{first['preflight_signature']}`\n"
        f"- Candidate budget: `{first['candidate_budget']}`\n"
        f"- Citation budget: `{first['citation_budget']}`\n"
        "- Gold is recorded for offline scoring only.\n\n"
        "## Frozen files\n\n"
        + "\n".join(
            f"- `{name}`: `{row['sha256']}` (`{row['path']}`)"
            for name, row in first["frozen_files"].items()
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(first, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
