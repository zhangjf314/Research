"""Freeze Stage 13.26 offline inputs."""

from __future__ import annotations

import json

from paper_research.generation.bounded_set_search import (
    BOUNDED_SET_SEARCH_VERSION,
    MINIMAL_GAIN_PROOF_V3_VERSION,
    SET_SUFFICIENCY_V3_VERSION,
)
from paper_research.generation.evidence_selection_v4 import CANDIDATE_BUDGET
from paper_research.retrieval.obligation_query_builder_v1 import (
    OBLIGATION_QUERY_BUILDER_VERSION,
)

try:
    from scripts.stage13_26_common import DATA, DOCS, ROOT, canonical_hash, file_hash, write_json
except ModuleNotFoundError:
    from stage13_26_common import DATA, DOCS, ROOT, canonical_hash, file_hash, write_json

OUT_JSON = DATA / "stage13-26-preflight-inputs-v1.json"
OUT_DOC = DOCS / "stage13-26-preflight-inputs-v1.md"


def tracked(path: str) -> dict[str, object]:
    target = ROOT / path
    return {
        "path": path,
        "exists": target.exists(),
        "sha256": file_hash(target) if target.exists() else None,
    }


def build() -> dict[str, object]:
    files = {
        "stage13_21_results": tracked("data/evaluation/evidence-qa-dev-v3-6.json"),
        "stage13_21_baseline_citation_sets": tracked(
            "data/evaluation/evidence-qa-dev-v3-6-citation-audit-v1.jsonl"
        ),
        "stage13_22_evidence_funnel": tracked(
            "data/evaluation/dev-v3-6-evidence-funnel-v1.jsonl"
        ),
        "stage13_23_selection_v3_replay": tracked(
            "data/evaluation/dev-v3-6-evidence-selection-v3-replay.json"
        ),
        "stage13_24_selection_v4_replay": tracked(
            "data/evaluation/dev-v3-6-evidence-selection-v4-replay.json"
        ),
        "stage13_25_set_completion_v2_replay": tracked(
            "data/evaluation/dev-v3-6-set-completion-v2-replay.json"
        ),
        "stage13_25_final_audit": tracked(
            "data/evaluation/dev-v3-6-set-completion-v2-final-audit.json"
        ),
        "oracle_gap_audit": tracked("data/evaluation/evidence-selection-v4-oracle-gap-v1.json"),
        "obligation_alignment_audit": tracked(
            "data/evaluation/obligation-definition-alignment-v1.json"
        ),
        "numeric_completeness_audit": tracked(
            "data/evaluation/selection-v4-numeric-completeness-audit-v1.jsonl"
        ),
        "comparison_completeness_audit": tracked(
            "data/evaluation/selection-v4-comparison-completeness-audit-v1.jsonl"
        ),
        "claim_gold_freeze": tracked("data/evaluation/claim-evidence-gold-dev-v1-freeze.json"),
        "payload_v4": tracked("data/evaluation/payload-contract-v4-protocol.json"),
        "evidence_presentation_v2": tracked(
            "data/evaluation/evidence-presentation-v2-protocol.json"
        ),
        "prompt_v3_7": tracked(
            "data/evaluation/dev-v3-6-prompt-rendering-preflight-v1.json"
        ),
        "bounded_set_search": tracked("src/paper_research/generation/bounded_set_search.py"),
        "obligation_query_builder": tracked(
            "src/paper_research/retrieval/obligation_query_builder_v1.py"
        ),
    }
    body = {
        "schema_version": "stage13-26-preflight-inputs-v1",
        "baseline_head": "cb59dff2fad8d977ba89ff455fe1eb9ca08dad9e",
        "frozen_files": files,
        "collection_name": "papers_production_v1",
        "embedding_model_identity": "jina-embeddings-v5-text-small",
        "citation_budget": 3,
        "candidate_budget": CANDIDATE_BUDGET,
        "bounded_set_search_version": BOUNDED_SET_SEARCH_VERSION,
        "set_sufficiency_version": SET_SUFFICIENCY_V3_VERSION,
        "minimal_gain_proof_version": MINIMAL_GAIN_PROOF_V3_VERSION,
        "targeted_query_builder_version": OBLIGATION_QUERY_BUILDER_VERSION,
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
        raise RuntimeError("STAGE13_26_PREFLIGHT_NOT_DETERMINISTIC")
    write_json(OUT_JSON, first)
    OUT_DOC.write_text(
        "# Stage 13.26 Preflight Inputs\n\n"
        f"- Signature: `{first['preflight_signature']}`\n"
        f"- Candidate budget: `{first['candidate_budget']}`\n"
        f"- Citation budget: `{first['citation_budget']}`\n"
        "- Gold is offline-scoring only.\n",
        encoding="utf-8",
    )
    print(json.dumps(first, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
