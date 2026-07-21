"""Freeze Stage 13.24 Evidence Selection v4 preflight inputs."""

from __future__ import annotations

import json
from pathlib import Path

try:
    from scripts.stage13_23_common import DATA, DOCS, ROOT, canonical_hash, file_hash, write_json
except ModuleNotFoundError:
    from stage13_23_common import (  # type: ignore[no-redef]
        DATA,
        DOCS,
        ROOT,
        canonical_hash,
        file_hash,
        write_json,
    )

OUT_JSON = DATA / "evidence-selection-v4-preflight-inputs-v1.json"
OUT_DOC = DOCS / "evidence-selection-v4-preflight-inputs-v1.md"

FILES = {
    "stage13_21_summary": DATA / "evidence-qa-dev-v3-6.json",
    "stage13_21_final_audit": DATA / "evidence-qa-dev-v3-6-final-audit.json",
    "stage13_21_citation_traces": DATA / "evidence-qa-dev-v3-6-citation-audit-v1.jsonl",
    "stage13_22_evidence_funnel": DATA / "dev-v3-6-evidence-funnel-v1.jsonl",
    "stage13_22_attribution": DATA / "dev-v3-6-quality-failure-attribution-v1.json",
    "stage13_23_wrong_evidence_audit": DATA / "evidence-selection-v2-wrong-evidence-audit-v1.json",
    "wrong_evidence_metric_alignment": DATA / "wrong-evidence-metric-alignment-v1.json",
    "selection_v2_replay": DATA / "dev-v3-6-evidence-selection-v2-replay.json",
    "selection_v3_replay": DATA / "dev-v3-6-evidence-selection-v3-replay.json",
    "selection_v3_readiness": DATA / "stage13-23-selection-v3-readiness-v1.json",
    "claim_gold_freeze": DATA / "claim-evidence-gold-dev-v1-freeze.json",
    "claim_gold": DATA / "claim-evidence-gold-dev-v1.jsonl",
    "payload_v4_protocol": DATA / "payload-contract-v4-protocol.json",
    "payload_v4_preflight": DATA / "payload-contract-v4-preflight-inputs-v1.json",
    "envelope_v4": DATA / "dev-v3-4-payload-contract-v4-final-audit.json",
    "evidence_presentation_v2": DATA / "evidence-presentation-v2-protocol.json",
    "prompt_v3_7": DATA / "dev-v3-6-prompt-rendering-preflight-v1.json",
    "citation_budget": ROOT / "src" / "paper_research" / "generation" / "citation_selection.py",
}


def safe_hash(path: Path) -> dict[str, str | bool]:
    return {
        "path": str(path.relative_to(ROOT)),
        "exists": path.exists(),
        "sha256": file_hash(path) if path.exists() else "",
    }


def build() -> dict:
    body = {
        "schema_version": "evidence-selection-v4-preflight-inputs-v1",
        "stage": "13.24",
        "baseline_head": "5ec5a4f81657487431c05a8caabb489186a81237",
        "frozen_files": {name: safe_hash(path) for name, path in FILES.items()},
        "candidate_budget": 12,
        "citation_budget": {"max_primary": 1, "max_supporting": 2, "max_total": 3},
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


def write_outputs(body: dict) -> None:
    write_json(OUT_JSON, body)
    OUT_DOC.write_text(
        "# Evidence Selection v4 Preflight Inputs\n\n"
        f"- Signature: `{body['preflight_signature']}`\n"
        f"- Candidate budget: `{body['candidate_budget']}`\n"
        f"- Citation budget: `{body['citation_budget']}`\n"
        "- Gold freeze is recorded for offline scoring only.\n"
        "- No live LLM, embedding API, reranker, Human Citation Audit, Full QA, "
        "or Deep Research was executed by this freeze step.\n\n"
        "## Frozen file hashes\n\n"
        + "\n".join(
            f"- `{name}`: `{meta['sha256']}` (`{meta['path']}`)"
            for name, meta in body["frozen_files"].items()
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    first = build()
    second = build()
    if first["preflight_signature"] != second["preflight_signature"]:
        raise RuntimeError("EVIDENCE_SELECTION_V4_PREFLIGHT_NOT_DETERMINISTIC")
    write_outputs(first)
    print(json.dumps(first, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
