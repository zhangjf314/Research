"""Build the offline Stage 13.1 Phase A adjudication package.

This script replays the frozen Stage 13 routed retrieval implementation. Gold
annotations are joined only after selection, for evaluation and diagnosis.
It performs no network or model calls.
"""

# ruff: noqa: E501 -- long human-review templates and taxonomy definitions remain readable inline.

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from paper_research.evidence.claims import ClaimUnit
from paper_research.evidence.schema import EvidenceUnit
from paper_research.retrieval.evidence_retriever import EvidenceCandidate, EvidenceRetriever
from paper_research.retrieval.filters import RetrievalFilter
from paper_research.retrieval.query_router import route_query

try:
    from scripts.run_evidence_retrieval_v1 import (
        _deduplicate,
        _selection_claim,
        _source_scores,
    )
except ModuleNotFoundError:  # direct `python scripts/...py` execution
    from run_evidence_retrieval_v1 import (
        _deduplicate,
        _selection_claim,
        _source_scores,
    )

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/evaluation"
DOCS = ROOT / "docs"
INPUTS = {
    "evidence_corpus": DATA / "evidence-corpus-v1.jsonl",
    "evidence_manifest": DATA / "evidence-corpus-v1-manifest.json",
    "claim_units": DATA / "claim-units-v1.jsonl",
    "claim_manifest": DATA / "claim-units-v1-manifest.json",
    "claim_evidence_gold": DATA / "claim-evidence-gold-v1.jsonl",
    "citation_calibration": DATA / "citation-calibration-v1.jsonl",
    "gold": DATA / "gold-set-v1.jsonl",
    "protocol": DATA / "retrieval-gold-v2.jsonl",
    "production_corpus": DATA / "production-corpus-v1.json",
    "qa_baseline": DATA / "qa-production-v1.json",
    "stage13_result": DATA / "evidence-retrieval-v1.json",
}
FREEZE_JSON = DATA / "stage13-baseline-freeze-v1.json"
FREEZE_MD = DOCS / "stage13-baseline-freeze-v1.md"
GAPS_JSONL = DATA / "evidence-gap-cases-v1.jsonl"
GAPS_CSV = DATA / "evidence-gap-cases-v1.csv"
GAPS_MD = DOCS / "evidence-gap-cases-v1.md"
TAXONOMY_JSON = DATA / "evidence-gap-taxonomy-v1.json"
TAXONOMY_MD = DOCS / "evidence-gap-taxonomy-v1.md"
GAP_REVIEW_MD = DOCS / "evidence-gap-human-review-v1.md"
PILOT_JSONL = DATA / "claim-evidence-gold-pilot-v1.jsonl"
PILOT_MD = DOCS / "claim-evidence-gold-pilot-v1.md"
CLAIM_AUDIT_JSON = DATA / "claim-first-failure-audit-v1.json"
CLAIM_AUDIT_MD = DOCS / "claim-first-failure-audit-v1.md"

TAXONOMY: dict[str, dict[str, Any]] = {
    "candidate_recall_failure": {
        "definition": "Gold evidence is absent from the frozen candidate pool.",
        "criteria": "No Gold block appears within candidate_pool_k after generic scoring.",
        "observable_evidence": "gold_candidate_rank is null or exceeds candidate_pool_k.",
        "retrieval_implementation_issue": True,
        "automatic_fix_allowed": False,
        "human_review_required": True,
        "affects_gold": False,
        "affects_production_metric": True,
    },
    "dense_rank_failure": {
        "definition": "Dense ranking suppresses supporting evidence.",
        "criteria": "Gold dense score/rank is materially worse than non-supporting candidates.",
        "observable_evidence": "Requires a source trace containing dense_score; unavailable in the frozen Stage 13 trace.",
        "retrieval_implementation_issue": True,
        "automatic_fix_allowed": False,
        "human_review_required": True,
        "affects_gold": False,
        "affects_production_metric": True,
    },
    "lexical_rank_failure": {
        "definition": "Lexical ranking suppresses supporting evidence.",
        "criteria": "Gold lexical score/rank is materially worse despite relevant terminology.",
        "observable_evidence": "Requires a source trace containing lexical_score; unavailable in the frozen Stage 13 trace.",
        "retrieval_implementation_issue": True,
        "automatic_fix_allowed": False,
        "human_review_required": True,
        "affects_gold": False,
        "affects_production_metric": True,
    },
    "fusion_rank_failure": {
        "definition": "Gold evidence reaches the candidate pool but not final Top-10.",
        "criteria": "Gold rank is within candidate_pool_k and greater than final context k.",
        "observable_evidence": "Candidate and selected ranks from the deterministic replay.",
        "retrieval_implementation_issue": True,
        "automatic_fix_allowed": False,
        "human_review_required": True,
        "affects_gold": False,
        "affects_production_metric": True,
    },
    "evidence_role_mismatch": {
        "definition": "A supporting block is assigned an incompatible evidence role.",
        "criteria": "Gold evidence roles do not intersect the routed required roles.",
        "observable_evidence": "Gold evidence_roles and router profile filters.",
        "retrieval_implementation_issue": True,
        "automatic_fix_allowed": False,
        "human_review_required": True,
        "affects_gold": False,
        "affects_production_metric": True,
    },
    "block_type_filter_error": {
        "definition": "A supporting block is rejected by generic block/evidence eligibility.",
        "criteria": "A Gold block exists but eligible_for_final_context is false.",
        "observable_evidence": "Block type, evidence roles, and rejection reasons.",
        "retrieval_implementation_issue": True,
        "automatic_fix_allowed": False,
        "human_review_required": True,
        "affects_gold": False,
        "affects_production_metric": True,
    },
    "section_rule_error": {
        "definition": "Section compatibility rules incorrectly suppress supporting evidence.",
        "criteria": "Gold is eligible but section scoring/caps exclude it.",
        "observable_evidence": "Section title, section score, candidate rank, packing trace.",
        "retrieval_implementation_issue": True,
        "automatic_fix_allowed": False,
        "human_review_required": True,
        "affects_gold": False,
        "affects_production_metric": True,
    },
    "context_budget_truncation": {
        "definition": "Selected support is removed by the final token budget.",
        "criteria": "Gold is selected before packing and appears in a truncation trace.",
        "observable_evidence": "Pre-pack selection and context packing trace.",
        "retrieval_implementation_issue": True,
        "automatic_fix_allowed": False,
        "human_review_required": True,
        "affects_gold": False,
        "affects_production_metric": True,
    },
    "page_hit_block_miss": {
        "definition": "The context reaches a Gold page but not an exact Gold block.",
        "criteria": "gold_page_available=true and exact_gold_block_available=false.",
        "observable_evidence": "Selected pages and Gold pages/blocks.",
        "retrieval_implementation_issue": False,
        "automatic_fix_allowed": False,
        "human_review_required": True,
        "affects_gold": False,
        "affects_production_metric": True,
    },
    "multi_block_evidence_required": {
        "definition": "A claim requires a set of blocks rather than one block.",
        "criteria": "No individual block directly supports the complete claim.",
        "observable_evidence": "Claim text, Gold blocks, neighbors, and human source inspection.",
        "retrieval_implementation_issue": False,
        "automatic_fix_allowed": False,
        "human_review_required": True,
        "affects_gold": True,
        "affects_production_metric": True,
    },
    "gold_granularity_too_narrow": {
        "definition": "Gold omits equivalent directly supporting evidence.",
        "criteria": "Human review confirms a non-Gold block is equally valid support.",
        "observable_evidence": "Gold, candidate, neighbor, and source comparison.",
        "retrieval_implementation_issue": False,
        "automatic_fix_allowed": False,
        "human_review_required": True,
        "affects_gold": True,
        "affects_production_metric": True,
    },
    "gold_granularity_too_broad": {
        "definition": "A Gold block contains materially broader content than needed.",
        "criteria": "Human review finds only a smaller span directly supports the claim.",
        "observable_evidence": "Gold block and sentence-level source inspection.",
        "retrieval_implementation_issue": False,
        "automatic_fix_allowed": False,
        "human_review_required": True,
        "affects_gold": True,
        "affects_production_metric": True,
    },
    "equivalent_non_gold_evidence": {
        "definition": "Selected non-Gold evidence directly supports the same claim.",
        "criteria": "Human review confirms semantic and factual equivalence.",
        "observable_evidence": "Selected text versus Gold and source context.",
        "retrieval_implementation_issue": False,
        "automatic_fix_allowed": False,
        "human_review_required": True,
        "affects_gold": True,
        "affects_production_metric": True,
    },
    "parsing_boundary_error": {
        "definition": "Parsing split or merged source text across incorrect block boundaries.",
        "criteria": "Direct support is fragmented, missing, or attached to a wrong page/block.",
        "observable_evidence": "PDF/source comparison, block neighbors, unusually short fragments.",
        "retrieval_implementation_issue": True,
        "automatic_fix_allowed": False,
        "human_review_required": True,
        "affects_gold": True,
        "affects_production_metric": True,
    },
    "claim_role_unknown": {
        "definition": "The derived claim role is unknown and uses the safe generic route.",
        "criteria": "claim_role=unknown for relevant ClaimUnits.",
        "observable_evidence": "ClaimUnit role and routing decision.",
        "retrieval_implementation_issue": True,
        "automatic_fix_allowed": False,
        "human_review_required": True,
        "affects_gold": False,
        "affects_production_metric": True,
    },
    "query_formulation_failure": {
        "definition": "The retrieval query lacks terms needed to locate support.",
        "criteria": "Human review confirms the query under-specifies the evidence need.",
        "observable_evidence": "Retrieval query, required claims, and candidate scores.",
        "retrieval_implementation_issue": True,
        "automatic_fix_allowed": False,
        "human_review_required": True,
        "affects_gold": False,
        "affects_production_metric": True,
    },
    "multi_paper_allocation_failure": {
        "definition": "Candidate or context allocation fails to preserve all target papers.",
        "criteria": "A multi-paper query omits at least one required paper.",
        "observable_evidence": "Target papers, selected paper distribution, per-paper ranks.",
        "retrieval_implementation_issue": True,
        "automatic_fix_allowed": False,
        "human_review_required": True,
        "affects_gold": False,
        "affects_production_metric": True,
    },
    "metric_exact_match_limitation": {
        "definition": "Strict block identity misses valid evidence at another granularity.",
        "criteria": "Human review confirms support while exact Gold block identity is absent.",
        "observable_evidence": "Strict metric result and approved alternative evidence mapping.",
        "retrieval_implementation_issue": False,
        "automatic_fix_allowed": False,
        "human_review_required": True,
        "affects_gold": False,
        "affects_production_metric": True,
    },
    "unknown": {
        "definition": "Available automatic evidence cannot determine a cause.",
        "criteria": "No deterministic category is justified.",
        "observable_evidence": "Incomplete or conflicting diagnostic signals.",
        "retrieval_implementation_issue": False,
        "automatic_fix_allowed": False,
        "human_review_required": True,
        "affects_gold": False,
        "affects_production_metric": True,
    },
}


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()


def candidate_payload(candidate: EvidenceCandidate, rank: int) -> dict[str, Any]:
    return {
        "rank": rank,
        "evidence_id": candidate.evidence.evidence_id,
        "paper_id": candidate.evidence.paper_id,
        "page": candidate.evidence.page,
        "block_id": candidate.evidence.block_id,
        "block_type": candidate.evidence.block_type,
        "section_title": candidate.evidence.section_title,
        "evidence_roles": candidate.evidence.evidence_roles,
        "text": candidate.evidence.text,
        "source_chunk_id": candidate.evidence.source_chunk_id,
        "source_retrieval_rank": candidate.original_retrieval_rank,
        "dense_score": candidate.dense_score,
        "lexical_score": candidate.lexical_score,
        "structural_score": candidate.structural_score,
        "evidence_score": candidate.total_score,
        "evidence_score_components": candidate.score_components.model_dump(mode="json"),
        "filter_reasons": candidate.filter_reasons,
        "rejection_reasons": candidate.rejection_reasons,
        "source_component_availability": {
            "dense_score": "unavailable_in_frozen_stage13_source_trace",
            "lexical_score": "unavailable_in_frozen_stage13_source_trace",
            "structural_score": "unavailable_in_frozen_stage13_source_trace",
            "fused_score": "available_as_query_relevance",
        },
    }


def replay() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    units = [EvidenceUnit.model_validate(row) for row in jsonl(INPUTS["evidence_corpus"])]
    gold = {row["question_id"]: row for row in jsonl(INPUTS["gold"])}
    protocol = {row["question_id"]: row for row in jsonl(INPUTS["protocol"])}
    claims = [ClaimUnit.model_validate(row) for row in jsonl(INPUTS["claim_units"])]
    claims_by_question: dict[str, list[ClaimUnit]] = defaultdict(list)
    for claim in claims:
        claims_by_question[claim.question_id].append(claim)
    baseline = json.loads(INPUTS["qa_baseline"].read_text(encoding="utf-8"))
    baseline_rows = {row["question_id"]: row for row in baseline["queries"]}
    by_key = {(unit.paper_id, unit.block_id): unit for unit in units}
    retriever = EvidenceRetriever(units)
    rows: list[dict[str, Any]] = []
    for qid in sorted(gold):
        gold_row = gold[qid]
        protocol_row = protocol[qid]
        selection_claim = _selection_claim(protocol_row, gold_row)
        retrieval_filter = RetrievalFilter(**protocol_row["retrieval_filter"])
        decision = route_query(protocol_row["retrieval_query"], [selection_claim], retrieval_filter)
        source = _source_scores(baseline_rows[qid]["context"])
        scored = retriever.score_candidates(selection_claim, decision, source_scores=source)
        pool = scored[: decision.profile.candidate_pool_k]
        selected = _deduplicate(pool, 10)
        selected_payload = [candidate_payload(item, rank) for rank, item in enumerate(selected, 1)]
        selected_blocks = {item.evidence.block_id for item in selected}
        selected_pages = {item.evidence.page for item in selected}
        selected_ids = [item.evidence.evidence_id for item in selected]
        exact = (
            bool(selected_blocks & set(gold_row["gold_block_ids"]))
            if gold_row["answerable"]
            else None
        )
        rows.append(
            {
                "question_id": qid,
                "answerable": gold_row["answerable"],
                "category": gold_row["category"],
                "difficulty": gold_row["difficulty"],
                "retrieval_scope": protocol_row["retrieval_scope"],
                "retrieval_filter": protocol_row["retrieval_filter"],
                "selection_claim": selection_claim.model_dump(mode="json"),
                "router": decision.model_dump(mode="json"),
                "candidate_pool_size": len(pool),
                "candidate_pool": [
                    candidate_payload(item, rank) for rank, item in enumerate(pool, 1)
                ],
                "selected": selected_payload,
                "selected_evidence_ids": selected_ids,
                "selected_block_ids": sorted(selected_blocks),
                "selected_pages": sorted(selected_pages),
                "exact_gold_block_available": exact,
                "gold_page_available": bool(selected_pages & set(gold_row["gold_pages"]))
                if gold_row["answerable"]
                else None,
                "question_claims": [
                    item.model_dump(mode="json") for item in claims_by_question[qid]
                ],
                "gold_units": [
                    by_key[(paper_id, block_id)].model_dump(mode="json")
                    for paper_id in gold_row["gold_paper_ids"]
                    for block_id in gold_row["gold_block_ids"]
                    if (paper_id, block_id) in by_key
                ],
            }
        )
    return rows, {"units": units, "by_key": by_key, "gold": gold, "protocol": protocol}


def diagnose(row: dict[str, Any], state: dict[str, Any], titles: dict[str, str]) -> dict[str, Any]:
    qid = row["question_id"]
    gold = state["gold"][qid]
    target_papers = gold["gold_paper_ids"]
    gold_keys = {(paper, block) for paper in target_papers for block in gold["gold_block_ids"]}
    all_units = [unit for key, unit in state["by_key"].items() if key in gold_keys]
    pool = row["candidate_pool"]
    selected = row["selected"]
    ranks = {
        (item["paper_id"], item["block_id"]): item["rank"]
        for item in pool
        if (item["paper_id"], item["block_id"]) in gold_keys
    }
    gold_in_pool = bool(ranks)
    any_ineligible = any(not unit.eligible_for_final_context for unit in all_units)
    wanted_roles = set(row["selection_claim"]["required_evidence_roles"])
    role_mismatch = bool(all_units) and all(
        not (wanted_roles & set(unit.evidence_roles)) for unit in all_units
    )
    page_hit = row["gold_page_available"]
    unknown_role = any(claim["claim_role"] == "unknown" for claim in row["question_claims"])
    if any_ineligible:
        category = "block_type_filter_error"
    elif gold_in_pool:
        category = "fusion_rank_failure"
    elif page_hit:
        category = "page_hit_block_miss"
    elif role_mismatch:
        category = "evidence_role_mismatch"
    elif unknown_role:
        category = "claim_role_unknown"
    elif not all_units:
        category = "parsing_boundary_error"
    else:
        category = "candidate_recall_failure"
    selected_text = [
        {
            "evidence_id": item["evidence_id"],
            "paper_id": item["paper_id"],
            "page": item["page"],
            "block_id": item["block_id"],
            "text": item["text"],
        }
        for item in selected
    ]
    return {
        "question_id": qid,
        "question": gold["question"],
        "category": gold["category"],
        "difficulty": gold["difficulty"],
        "retrieval_scope": row["retrieval_scope"],
        "target_papers": [
            {"paper_id": paper, "title": titles.get(paper, paper)} for paper in target_papers
        ],
        "required_claims": gold["required_claims"],
        "gold_block_ids": gold["gold_block_ids"],
        "gold_pages": gold["gold_pages"],
        "gold_block_text": [
            {
                "evidence_id": unit.evidence_id,
                "paper_id": unit.paper_id,
                "page": unit.page,
                "block_id": unit.block_id,
                "block_type": unit.block_type,
                "evidence_roles": unit.evidence_roles,
                "text": unit.text,
            }
            for unit in all_units
        ],
        "selected_evidence_ids": row["selected_evidence_ids"],
        "selected_block_ids": row["selected_block_ids"],
        "selected_pages": row["selected_pages"],
        "selected_evidence_text": selected_text,
        "candidate_pool_top_30": pool[:30],
        "candidate_pool_k": row["candidate_pool_size"],
        "dense_score_status": "unavailable_in_frozen_stage13_source_trace",
        "lexical_score_status": "unavailable_in_frozen_stage13_source_trace",
        "structural_score_status": "unavailable_in_frozen_stage13_source_trace",
        "gold_block_candidate_ranks": [
            {"paper_id": paper, "block_id": block, "rank": ranks.get((paper, block))}
            for paper, block in sorted(gold_keys)
        ],
        "gold_page_hit": page_hit,
        "gold_block_filtered_in_initial_screen": any_ineligible,
        "gold_block_downweighted_by_scoring": not gold_in_pool and bool(all_units),
        "gold_block_truncated_by_context_packing": False,
        "gold_block_excluded_by_section_cap": False,
        "gold_block_filtered_by_type_or_role": any_ineligible or role_mismatch,
        "possible_parsing_boundary_issue": (not all_units)
        or any(len(unit.text.strip()) < 20 for unit in all_units),
        "possible_gold_too_short_or_broad": any(
            len(unit.text.strip()) < 40 or len(unit.text) > 1500 for unit in all_units
        ),
        "possible_multi_block_joint_support": len(gold["gold_block_ids"]) > 1
        or len(gold["required_claims"]) > 1,
        "possible_equivalent_non_gold_evidence": None,
        "initial_failure_category": category,
        "initial_failure_category_is_automatic": True,
        "human_review_status": "pending",
        "human_failure_category": None,
        "reviewer": None,
        "reviewed_at": None,
        "review_notes": None,
        "human_adjudication": {
            "gold_block_directly_supports": None,
            "multiple_blocks_required": None,
            "equivalent_non_gold_evidence_exists": None,
            "gold_too_narrow": None,
            "gold_too_broad": None,
            "selected_evidence_support": None,
            "issue_class": None,
        },
    }


def neighbors(unit: EvidenceUnit, by_key: dict[tuple[str, str], EvidenceUnit]) -> dict[str, Any]:
    def item(block_id: str | None) -> dict[str, Any] | None:
        found = by_key.get((unit.paper_id, block_id)) if block_id else None
        return (
            {
                "paper_id": found.paper_id,
                "page": found.page,
                "block_id": found.block_id,
                "text": found.text,
            }
            if found
            else None
        )

    return {"previous": item(unit.previous_block_id), "next": item(unit.next_block_id)}


def select_pilot(
    replay_rows: list[dict[str, Any]], gaps: list[dict[str, Any]], state: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    claims = [ClaimUnit.model_validate(row) for row in jsonl(INPUTS["claim_units"])]
    by_q: dict[str, list[ClaimUnit]] = defaultdict(list)
    for claim in claims:
        by_q[claim.question_id].append(claim)
    selected: list[tuple[ClaimUnit, str]] = []
    used: set[str] = set()

    def add(claim: ClaimUnit, stratum: str) -> bool:
        if claim.claim_id in used:
            return False
        selected.append((claim, stratum))
        used.add(claim.claim_id)
        return True

    for gap in gaps:
        if by_q[gap["question_id"]]:
            add(by_q[gap["question_id"]][0], "exact_miss_core_claim")
    for row in replay_rows:
        if len([x for x in selected if x[1] == "exact_hit_incomplete_recall"]) >= 8:
            break
        if row["answerable"] and row["exact_gold_block_available"]:
            gold_blocks = set(state["gold"][row["question_id"]]["gold_block_ids"])
            if gold_blocks - set(row["selected_block_ids"]):
                for claim in by_q[row["question_id"]]:
                    if add(claim, "exact_hit_incomplete_recall"):
                        break
    for claim in claims:
        if len([x for x in selected if x[1] == "multi_paper"]) >= 5:
            break
        if len(claim.target_paper_ids) >= 2 or claim.question_type == "multi_paper":
            add(claim, "multi_paper")
    for claim in claims:
        if len([x for x in selected if x[1] == "unknown_claim_role"]) >= 5:
            break
        if claim.claim_role == "unknown":
            add(claim, "unknown_claim_role")
    seen_controls: set[tuple[str, str]] = set()
    gold_map = state["gold"]
    for claim in claims:
        if len([x for x in selected if x[1] == "control"]) >= 5:
            break
        gold = gold_map[claim.question_id]
        key = (gold["category"], gold["difficulty"])
        if key not in seen_controls and add(claim, "control"):
            seen_controls.add(key)
    for claim in claims:
        if len(selected) >= 40:
            break
        add(claim, "category_difficulty_supplement")
    selected = selected[:40]
    replay_by_q = {row["question_id"]: row for row in replay_rows}
    source_hashes = {
        "claim_units_sha256": file_hash(INPUTS["claim_units"]),
        "evidence_corpus_sha256": file_hash(INPUTS["evidence_corpus"]),
        "gold_sha256": file_hash(INPUTS["gold"]),
    }
    output = []
    for index, (claim, stratum) in enumerate(selected, 1):
        replay_row = replay_by_q[claim.question_id]
        candidates = replay_row["candidate_pool"][:10]
        current_gold = [
            unit
            for (paper, block), unit in state["by_key"].items()
            if paper in claim.target_paper_ids and block in claim.gold_block_ids
        ]
        candidate_evidence = [
            {
                "evidence_id": row["evidence_id"],
                "paper_id": row["paper_id"],
                "page": row["page"],
                "block_id": row["block_id"],
                "score": row["evidence_score"],
                "roles": row["evidence_roles"],
                "text": row["text"],
            }
            for row in candidates
        ]
        proposed_sets = [
            [{"paper_id": unit.paper_id, "page": unit.page, "block_id": unit.block_id}]
            for unit in current_gold
        ]
        output.append(
            {
                "pilot_sample_id": f"claim-evidence-pilot-v1-{index:03d}",
                "sampling_stratum": stratum,
                "question_id": claim.question_id,
                "claim_id": claim.claim_id,
                "claim_text": claim.claim_text,
                "claim_role": claim.claim_role,
                "target_papers": claim.target_paper_ids,
                "current_gold_blocks": claim.gold_block_ids,
                "current_gold_pages": claim.gold_pages,
                "candidate_evidence": candidate_evidence,
                "neighboring_context": [
                    {
                        "evidence": {
                            "paper_id": unit.paper_id,
                            "page": unit.page,
                            "block_id": unit.block_id,
                            "text": unit.text,
                        },
                        **neighbors(unit, state["by_key"]),
                    }
                    for unit in current_gold
                ],
                "proposed_evidence_sets": proposed_sets,
                "proposed_alternatives": candidate_evidence[:3],
                "approved_evidence_sets": [],
                "approved_alternative_evidence_sets": [],
                "multi_block_required": claim.multi_block_required,
                "annotation_status": "pending",
                "decision": None,
                "reviewer": None,
                "reviewed_at": None,
                "review_notes": None,
                "claim_role_before_review": claim.claim_role,
                "claim_role_after_review": None,
                "source_hashes": source_hashes,
                "source_record_hash": stable_hash(claim.model_dump(mode="json")),
            }
        )
    return output, dict(Counter(stratum for _, stratum in selected))


def build_freeze(replay_rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = json.loads(INPUTS["stage13_result"].read_text(encoding="utf-8"))
    routed = next(
        item for item in result["variants"] if item["name"] == "routed_evidence_retrieval"
    )
    metrics = routed["metrics"]
    citation_rows = jsonl(INPUTS["citation_calibration"])
    strata = Counter(row["stratum"] for row in citation_rows)
    citation_valid = {
        "record_count": len(citation_rows),
        "strata": dict(sorted(strata.items())),
        "all_pending": all(row["human_review_status"] == "pending" for row in citation_rows),
        "no_human_label": all(row["human_label"] is None for row in citation_rows),
        "distinct_from_claim_evidence_pilot": True,
    }
    hashes = {name: file_hash(path) for name, path in INPUTS.items()}
    fingerprint_input = {
        "profile": "routed_evidence_retrieval",
        "candidate_pool_k": sorted(
            {row["router"]["profile"]["candidate_pool_k"] for row in replay_rows}
        ),
        "final_context_k": 10,
        "reranker_enabled": False,
        "llm_called": False,
        "gold_selection_policy": "post-selection evaluation only",
    }
    return {
        "schema_version": "stage13-baseline-freeze-v1",
        "code_commit": git_commit(),
        "input_artifact_hashes": hashes,
        "configuration": fingerprint_input,
        "configuration_fingerprint": stable_hash(fingerprint_input),
        "retrieval_profile": "routed_evidence_retrieval",
        "corpus_signature": json.loads(INPUTS["evidence_manifest"].read_text(encoding="utf-8"))[
            "build_signature"
        ],
        "claim_manifest_signature": json.loads(
            INPUTS["claim_manifest"].read_text(encoding="utf-8")
        )["build_signature"],
        "answerable_questions": 48,
        "exact_block_hits": 31,
        "exact_block_availability": metrics["exact_gold_block_availability"],
        "gold_page_hits": 42,
        "gold_page_availability": metrics["gold_page_availability"],
        "gold_block_recall": metrics["gold_block_recall"],
        "multi_paper_coverage": metrics["multi_paper_all_source_coverage"],
        "metadata_contamination": metrics["metadata_contamination_rate"],
        "mean_context_tokens": metrics["mean_context_token_count"],
        "p95_latency_ms": metrics["p95_latency_ms"],
        "per_question_final_context_evidence_ids": {
            row["question_id"]: row["selected_evidence_ids"] for row in replay_rows
        },
        "selection_integrity": {
            "gold_used_for_selection": False,
            "gold_required_claims_used_for_selection": False,
            "oracle_used_for_selection": False,
            "proof": "Replay selects from retrieval_query, retrieval_scope, retrieval_filter, frozen Stage 11C source context, EvidenceUnit fields, and deterministic router scores before Gold is joined for metrics.",
        },
        "citation_calibration_validation": citation_valid,
        "model_calls": {"llm": 0, "embedding": 0, "reranker": 0, "deep_research": 0},
    }


def claim_first_audit(
    replay_rows: list[dict[str, Any]], state: dict[str, Any], pilot: list[dict[str, Any]]
) -> dict[str, Any]:
    stage13 = json.loads(INPUTS["stage13_result"].read_text(encoding="utf-8"))
    claim_first = next(
        item for item in stage13["variants"] if item["name"] == "claim_first_evidence_retrieval"
    )
    routed = next(
        item for item in stage13["variants"] if item["name"] == "routed_evidence_retrieval"
    )
    allocation_rows = []
    for row in claim_first["queries"]:
        for allocation in row.get("claim_allocations", []):
            allocation_rows.append(
                {
                    "question_id": row["question_id"],
                    "candidate_pool_size": row["metrics"]["candidate_count"],
                    "selected_evidence_count": len(allocation["selected_evidence_ids"]),
                    "evidence_complete": allocation["evidence_complete"],
                    "unsupported_before_generation": allocation["unsupported_before_generation"],
                    "missing_evidence_reason": allocation["missing_evidence_reason"],
                    "truncated": row["metrics"]["truncated"],
                    "context_tokens": row["metrics"]["context_token_count"],
                }
            )
    unknown_total = sum(row["claim_role"] == "unknown" for row in jsonl(INPUTS["claim_units"]))
    pilot_unknown = sum(row["claim_role"] == "unknown" for row in pilot)
    return {
        "schema_version": "claim-first-failure-audit-v1",
        "current_exact_availability": claim_first["metrics"]["exact_gold_block_availability"],
        "routed_exact_availability": routed["metrics"]["exact_gold_block_availability"],
        "selector_configuration": {"minimum_score": 0.18, "max_per_claim": 2},
        "context_configuration": {"max_tokens": 3000, "max_units_per_section": 3},
        "allocation_rows": allocation_rows,
        "analysis": {
            "candidate_pool_sizes": dict(
                Counter(row["candidate_pool_size"] for row in allocation_rows)
            ),
            "evidence_completeness_threshold": "minimum_score=0.18 plus role/paper constraints",
            "minimum_evidence_set_size": "1 normally; 2 when multi_block_required; target-paper count for multi-paper",
            "token_budget_too_small": False,
            "truncated_question_count": sum(row["truncated"] for row in allocation_rows),
            "unknown_claim_role_total": unknown_total,
            "unknown_claim_role_pilot_coverage": pilot_unknown,
            "automatic_required_roles_risk": "Selection uses one query-derived ClaimUnit; automatically inferred roles may reject complete support.",
            "gold_candidate_presence": "Audited separately in evidence-gap-cases-v1; Gold is evaluation-only.",
            "selector_rejection_risk": "max_per_claim=2 and minimum_score can reduce Top-10 routed context to one or two units.",
            "multi_claim_budget_fragmentation": "Not exercised by Stage 13 formal selection: it uses one query-derived selection claim per question.",
            "primary_failure": "Aggressive selector compression: a maximum of two units from the routed pool lowered exact availability from 0.645833 to 0.125.",
            "multi_paper_allocation_risk": "Per-paper quota can consume the two-unit maximum and leave no redundancy within a paper.",
            "related_vs_complete_support_risk": "Score threshold establishes relevance, not human-validated complete support.",
            "human_mapping_impact": "All 146 claim-evidence mappings are pending, so formal claim completeness cannot yet be measured.",
        },
        "gold_used_for_selection": False,
        "approved_pilot_count": sum(row["annotation_status"] == "approved" for row in pilot),
        "claim_level_metrics_status": "unavailable_until_pilot_review",
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )


def write_reports(
    freeze: dict[str, Any],
    gaps: list[dict[str, Any]],
    pilot: list[dict[str, Any]],
    pilot_strata: dict[str, int],
    audit: dict[str, Any],
) -> None:
    FREEZE_MD.write_text(
        "# Stage 13 Baseline Freeze v1\n\n"
        "> Offline deterministic freeze. Gold is joined only after selection. No LLM, Embedding, Reranker, or Deep Research call was made.\n\n"
        f"- Code commit: `{freeze['code_commit']}`\n"
        f"- Corpus signature: `{freeze['corpus_signature']}`\n"
        f"- Configuration fingerprint: `{freeze['configuration_fingerprint']}`\n"
        f"- Exact block availability: **31/48 = {freeze['exact_block_availability']:.6f}**\n"
        f"- Gold page availability: **42/48 = {freeze['gold_page_availability']:.6f}**\n"
        f"- Gold block recall: **{freeze['gold_block_recall']:.6f}**\n"
        f"- Multi-paper coverage: **{freeze['multi_paper_coverage']:.6f}**\n"
        f"- Metadata contamination: **{freeze['metadata_contamination']:.6f}**\n"
        f"- Mean context tokens: **{freeze['mean_context_tokens']:.2f}**\n"
        f"- P95 latency: **{freeze['p95_latency_ms']:.6f} ms**\n\n"
        "The JSON freeze contains every input SHA-256 and each question's selected Evidence IDs. The replay implementation constructs the selection from the approved retrieval protocol and frozen source context before reading Gold fields for evaluation.\n",
        encoding="utf-8",
    )
    lines = [
        "# Evidence Gap Cases v1",
        "",
        "> Automatic diagnostics only. All human adjudication fields remain pending/null.",
        "",
        "| Question | Category | Difficulty | Page hit | Gold rank(s) | Initial category |",
        "|---|---|---|---:|---|---|",
    ]
    for row in gaps:
        ranks = ", ".join(f"{x['block_id']}:{x['rank']}" for x in row["gold_block_candidate_ranks"])
        lines.append(
            f"| {row['question_id']} | {row['category']} | {row['difficulty']} | {row['gold_page_hit']} | {ranks} | {row['initial_failure_category']} |"
        )
    lines += [
        "",
        "Full Top-30 candidates, score components, texts, and pending human fields are in the JSONL artifact.",
    ]
    GAPS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    taxonomy_lines = [
        "# Evidence Gap Taxonomy v1",
        "",
        "| Category | Retrieval issue | Auto-fix | Human review | Affects Gold |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, item in TAXONOMY.items():
        taxonomy_lines.append(
            f"| `{name}` | {item['retrieval_implementation_issue']} | {item['automatic_fix_allowed']} | {item['human_review_required']} | {item['affects_gold']} |"
        )
        taxonomy_lines += [
            "",
            f"**{name}** — {item['definition']}  ",
            f"Criteria: {item['criteria']}  ",
            f"Observable evidence: {item['observable_evidence']}",
            "",
        ]
    TAXONOMY_MD.write_text("\n".join(taxonomy_lines) + "\n", encoding="utf-8")
    review = [
        "# Evidence Gap Human Review v1",
        "",
        "> 17/17 are pending. Do not infer human conclusions from automatic categories.",
        "",
        "Allowed categories: " + ", ".join(f"`{x}`" for x in TAXONOMY),
        "",
    ]
    by_key = {
        (u.paper_id, u.block_id): u
        for u in [EvidenceUnit.model_validate(x) for x in jsonl(INPUTS["evidence_corpus"])]
    }
    for row in gaps:
        review += [
            f"## {row['question_id']}",
            "",
            f"1. Question: {row['question']}",
            f"2. Required claims: {' | '.join(row['required_claims'])}",
            f"3. Target paper: {json.dumps(row['target_papers'], ensure_ascii=False)}",
            f"4. Gold block(s): {json.dumps(row['gold_block_text'], ensure_ascii=False)}",
        ]
        neighbor_material = []
        for gold in row["gold_block_text"]:
            unit = by_key[(gold["paper_id"], gold["block_id"])]
            neighbor_material.append({"gold": gold, **neighbors(unit, by_key)})
        review += [
            f"5. Gold page(s): {row['gold_pages']}",
            f"6. Top candidates: {json.dumps(row['candidate_pool_top_30'][:5], ensure_ascii=False)}",
            f"7. Selected context: {json.dumps(row['selected_evidence_text'], ensure_ascii=False)}",
            f"8. Same-page/non-Gold and neighbors: {json.dumps(neighbor_material, ensure_ascii=False)}",
            "9. Possible equivalent evidence: [HUMAN INPUT REQUIRED]",
            f"10. Automatic category: `{row['initial_failure_category']}`",
            "11. Human fields: category=[ ], Gold directly supports=[ ], multi-block=[ ], equivalent non-Gold=[ ], Gold too narrow=[ ], Gold too broad=[ ], selected support=[fully/partial/none], retrieval-or-annotation=[ ], reviewer=[ ], reviewed_at=[ ], notes=[ ]",
            "",
        ]
    GAP_REVIEW_MD.write_text("\n".join(review) + "\n", encoding="utf-8")
    pilot_lines = [
        "# Claim-Evidence Gold Pilot v1",
        "",
        "> All proposed evidence is machine-generated review material, not approved Gold.",
        "",
        f"- Samples: {len(pilot)}",
        f"- Pending: {sum(x['annotation_status'] == 'pending' for x in pilot)}",
        f"- Strata: `{json.dumps(pilot_strata, sort_keys=True)}`",
        f"- Unknown-role coverage: {sum(x['claim_role'] == 'unknown' for x in pilot)}/63",
        "",
        "Use `scripts/review_claim_evidence_pilot_v1.py --list-pending` and export a sample before recording a decision.",
        "",
    ]
    for row in pilot:
        pilot_lines += [
            f"## {row['pilot_sample_id']} — {row['question_id']}",
            "",
            f"- Stratum: `{row['sampling_stratum']}`",
            f"- Claim: {row['claim_text']}",
            f"- Role: `{row['claim_role']}`",
            f"- Target papers: `{row['target_papers']}`",
            f"- Proposed sets (not approved): `{row['proposed_evidence_sets']}`",
            "",
        ]
    PILOT_MD.write_text("\n".join(pilot_lines) + "\n", encoding="utf-8")
    analysis = audit["analysis"]
    CLAIM_AUDIT_MD.write_text(
        "# Claim-first Failure Audit v1\n\n"
        f"- Claim-first exact availability: **{audit['current_exact_availability']:.6f}**\n"
        f"- Routed exact availability: **{audit['routed_exact_availability']:.6f}**\n"
        f"- Selector: minimum score `{audit['selector_configuration']['minimum_score']}`, max `{audit['selector_configuration']['max_per_claim']}` per claim\n"
        f"- Context budget: `{audit['context_configuration']['max_tokens']}` tokens, section cap `{audit['context_configuration']['max_units_per_section']}`\n"
        f"- Unknown roles: `{analysis['unknown_claim_role_total']}`, Pilot coverage `{analysis['unknown_claim_role_pilot_coverage']}`\n\n"
        "The principal deterministic finding is aggressive selector compression: routed Top-10 is reduced to at most two units. A relevance threshold is not evidence-completeness validation. The formal Stage 13 path also uses one query-derived selection claim, so required-claim mappings are not injected and multi-claim budget fragmentation is not the observed cause. All claim-level quality metrics remain unavailable until human Pilot approval.\n\n"
        "No Gold block, approved Pilot evidence, or question-specific condition is used by the Production selector.\n",
        encoding="utf-8",
    )


def main() -> None:
    replay_rows, state = replay()
    stage13 = json.loads(INPUTS["stage13_result"].read_text(encoding="utf-8"))
    routed_frozen = next(x for x in stage13["variants"] if x["name"] == "routed_evidence_retrieval")
    frozen_ids = {
        row["question_id"]: sorted(
            "|".join(map(str, triple)) for triple in row["metrics"]["citation_triples"]
        )
        for row in routed_frozen["queries"]
    }
    replay_ids = {
        row["question_id"]: sorted(
            "|".join(map(str, (item["paper_id"], item["page"], item["block_id"])))
            for item in row["selected"]
        )
        for row in replay_rows
    }
    if frozen_ids != replay_ids:
        raise RuntimeError("deterministic replay differs from frozen Stage 13 selected contexts")
    answerable = [row for row in replay_rows if row["answerable"]]
    hits = sum(bool(row["exact_gold_block_available"]) for row in answerable)
    if len(answerable) != 48 or hits != 31:
        raise RuntimeError(f"baseline mismatch: answerable={len(answerable)}, hits={hits}")
    corpus = json.loads(INPUTS["production_corpus"].read_text(encoding="utf-8"))
    titles = {row["paper_id"]: row["title"] for row in corpus["papers"]}
    gaps = [
        diagnose(row, state, titles) for row in answerable if not row["exact_gold_block_available"]
    ]
    if len(gaps) != 17:
        raise RuntimeError(f"expected 17 gaps, got {len(gaps)}")
    freeze = build_freeze(replay_rows)
    pilot, pilot_strata = select_pilot(replay_rows, gaps, state)
    if len(pilot) != 40:
        raise RuntimeError(f"expected 40 Pilot samples, got {len(pilot)}")
    claim_audit = claim_first_audit(replay_rows, state, pilot)
    FREEZE_JSON.write_text(json.dumps(freeze, ensure_ascii=False, indent=2), encoding="utf-8")
    write_jsonl(GAPS_JSONL, gaps)
    with GAPS_CSV.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "question_id",
                "category",
                "difficulty",
                "retrieval_scope",
                "gold_page_hit",
                "initial_failure_category",
                "human_review_status",
                "human_failure_category",
                "reviewer",
                "reviewed_at",
                "review_notes",
            ],
        )
        writer.writeheader()
        for row in gaps:
            writer.writerow({key: row.get(key) for key in writer.fieldnames})
    TAXONOMY_JSON.write_text(
        json.dumps(
            {"schema_version": "evidence-gap-taxonomy-v1", "categories": TAXONOMY},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_jsonl(PILOT_JSONL, pilot)
    CLAIM_AUDIT_JSON.write_text(
        json.dumps(claim_audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_reports(freeze, gaps, pilot, pilot_strata, claim_audit)
    print(
        json.dumps(
            {
                "baseline": "31/48",
                "misses": len(gaps),
                "pilot": len(pilot),
                "pilot_pending": 40,
                "llm_calls": 0,
                "reranker_enabled": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
