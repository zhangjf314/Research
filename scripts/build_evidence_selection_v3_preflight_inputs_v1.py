"""Freeze Stage 13.23 replay inputs without modifying prior results."""

from __future__ import annotations

import json

from paper_research.generation.citation_selection import (
    CITATION_BUDGET_VERSION,
    COMPARISON_VALIDATION_VERSION,
    NUMERIC_VALIDATION_VERSION,
    OBLIGATION_POLICY_VERSION,
)

try:
    from scripts.stage13_23_common import (
        DATA,
        DOCS,
        RUN_ROOT,
        canonical_hash,
        file_hash,
        read_json,
        write_json,
    )
except ModuleNotFoundError:
    from stage13_23_common import (  # type: ignore[no-redef]
        DATA,
        DOCS,
        RUN_ROOT,
        canonical_hash,
        file_hash,
        read_json,
        write_json,
    )

OUT = DATA / "evidence-selection-v3-preflight-inputs-v1.json"
DOC = DOCS / "evidence-selection-v3-preflight-inputs-v1.md"


def build() -> dict:
    protocol = read_json(DATA / "evidence-qa-dev-v3-6-protocol-freeze-v1.json")
    summary = read_json(DATA / "evidence-qa-dev-v3-6.json")
    run_hashes = []
    for run_id in summary["selected_runs"]:
        run_dir = RUN_ROOT / run_id
        run_hashes.append(
            {
                "run_id": run_id,
                "citation_registry_hash": file_hash(run_dir / "citation-registry.json"),
                "candidate_evidence_hash": file_hash(run_dir / "candidate-evidence-local.json"),
                "claim_proposal_hash": file_hash(run_dir / "raw-model-payload.json"),
            }
        )
    body = {
        "schema_version": "evidence-selection-v3-preflight-inputs-v1",
        "stage13_21_formal_result_hash": file_hash(DATA / "evidence-qa-dev-v3-6.json"),
        "stage13_21_final_audit_hash": file_hash(DATA / "evidence-qa-dev-v3-6-final-audit.json"),
        "stage13_21_quality_failure_freeze_hash": file_hash(
            DATA / "dev-v3-6-quality-failure-freeze-v1.json"
        ),
        "stage13_22_evidence_funnel_hash": file_hash(
            DATA / "dev-v3-6-evidence-funnel-v1.json"
        ),
        "stage13_22_attribution_hash": file_hash(
            DATA / "dev-v3-6-quality-failure-attribution-v1.json"
        ),
        "selection_v2_implementation_hash": file_hash(
            DATA.parent.parent
            / "src"
            / "paper_research"
            / "generation"
            / "evidence_selection_v2.py"
        ),
        "selection_v2_replay_hash": read_json(
            DATA / "dev-v3-6-evidence-selection-v2-replay.json"
        )["replay_hash"],
        "payload_v4_hash": protocol["payload_v4_hash"],
        "envelope_v4_hash": protocol["envelope_v4_hash"],
        "evidence_presentation_v2_hash": protocol["evidence_presentation_hash"],
        "prompt_v3_7_hash": protocol["prompt_hash"],
        "run_hashes": run_hashes,
        "citation_budget_hash": canonical_hash(CITATION_BUDGET_VERSION),
        "obligation_policy_hash": canonical_hash(OBLIGATION_POLICY_VERSION),
        "numeric_validator_hash": canonical_hash(NUMERIC_VALIDATION_VERSION),
        "comparison_validator_hash": canonical_hash(COMPARISON_VALIDATION_VERSION),
        "gold_freeze_hash_offline_scoring_only": file_hash(
            DATA / "claim-evidence-gold-dev-v1-freeze.json"
        ),
    }
    body["preflight_signature"] = canonical_hash(body)
    return body


def main() -> None:
    first = build()
    second = build()
    if first["preflight_signature"] != second["preflight_signature"]:
        raise RuntimeError("EVIDENCE_SELECTION_V3_PREFLIGHT_NOT_DETERMINISTIC")
    write_json(OUT, first)
    DOC.write_text(
        "# Evidence Selection v3 Preflight Inputs\n\n"
        f"- Preflight signature: `{first['preflight_signature']}`\n"
        f"- Payload v4 hash: `{first['payload_v4_hash']}`\n"
        f"- Evidence Presentation v2 hash: `{first['evidence_presentation_v2_hash']}`\n"
        f"- Prompt v3.7 hash: `{first['prompt_v3_7_hash']}`\n"
        "- Gold freeze is recorded only for offline scoring.\n",
        encoding="utf-8",
    )
    print(json.dumps(first, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
