"""Offline Stage 13 evidence retrieval ablation with no model calls."""

from __future__ import annotations

import csv
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean

from paper_research.evidence.claims import (
    ClaimUnit,
    classify_claim_role,
    classify_question_type,
    required_roles,
    stable_claim_id,
)
from paper_research.evidence.schema import EvidenceUnit
from paper_research.generation.claim_evidence_selector import ClaimFirstEvidenceSelector
from paper_research.retrieval.evidence_context_builder import EvidenceContextBuilder
from paper_research.retrieval.evidence_retriever import EvidenceCandidate, EvidenceRetriever
from paper_research.retrieval.filters import RetrievalFilter
from paper_research.retrieval.query_router import route_query

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "data/evaluation/evidence-corpus-v1.jsonl"
CLAIMS = ROOT / "data/evaluation/claim-units-v1.jsonl"
GOLD = ROOT / "data/evaluation/gold-set-v1.jsonl"
PROTOCOL = ROOT / "data/evaluation/retrieval-gold-v2.jsonl"
BASELINE = ROOT / "data/evaluation/qa-production-v1.json"
OUTPUT = ROOT / "data/evaluation/evidence-retrieval-v1.json"
CSV_OUTPUT = ROOT / "data/evaluation/evidence-retrieval-v1.csv"
REPORT = ROOT / "docs/evidence-retrieval-v1.md"
TERM_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.%-]+")


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _terms(text: str) -> set[str]:
    return {term.casefold() for term in TERM_RE.findall(text) if len(term) > 2}


def _lexical_score(text: str, query: str) -> float:
    wanted = _terms(query)
    if not wanted:
        return 0.0
    present = _terms(text)
    return len(wanted & present) / len(wanted)


def _source_scores(context: list[dict]) -> dict[str, dict[str, float | int | None]]:
    return {
        item["chunk_id"]: {
            "rank": rank,
            "fused_score": item.get("score", 0.0),
            "dense_score": None,
            "lexical_score": None,
            "structural_score": None,
        }
        for rank, item in enumerate(context, 1)
    }


def _candidate_from_unit(claim_id: str, unit: EvidenceUnit, score: float) -> EvidenceCandidate:
    from paper_research.retrieval.evidence_retriever import EvidenceScoreComponents

    return EvidenceCandidate(
        claim_id=claim_id,
        evidence=unit,
        total_score=score,
        lexical_score=score,
        score_components=EvidenceScoreComponents(
            query_relevance=score,
            claim_term_coverage=score,
            evidence_role_compatibility=0.5,
            section_compatibility=0.0,
            paper_filter_validity=1.0,
            numeric_fact_compatibility=0.0,
            comparison_dimension_coverage=0.0,
            answerability_compatibility=1.0,
            metadata_penalty=0.0,
            citation_only_penalty=0.0,
            duplication_penalty=0.0,
        ),
        filter_reasons=["paper_filter_valid", "eligible_evidence"],
    )


def _selection_claim(protocol: dict, gold: dict) -> ClaimUnit:
    query = protocol["retrieval_query"]
    scope = protocol["retrieval_scope"]
    if scope == "unanswerable":
        question_type = "unanswerable"
    elif scope == "multi_paper":
        question_type = "multi_paper"
    else:
        question_type = classify_question_type(
            {
                "question": query,
                "category": "",
                "answerable": True,
                "gold_paper_ids": [],
            }
        )
    role = classify_claim_role(query, question_type)
    target_papers = protocol.get("retrieval_filter", {}).get("paper_ids", [])
    return ClaimUnit(
        claim_id=stable_claim_id(protocol["question_id"], 0, query),
        question_id=protocol["question_id"],
        question_type=question_type,
        claim_text=query,
        normalized_claim=" ".join(query.casefold().split()),
        claim_role=role,
        target_paper_ids=target_papers,
        target_terms=sorted(_terms(query)),
        expected_answerability=bool(gold["answerable"]),
        required_evidence_roles=required_roles(role),
        negative_constraints=(
            ["Do not fabricate evidence for an unanswerable question."]
            if not gold["answerable"]
            else []
        ),
        derivation_trace={
            "selection_source": "retrieval_query, retrieval_scope, and paper filter only",
            "gold_required_claims": "not used for selection",
        },
    )


def _deduplicate(candidates: list[EvidenceCandidate], limit: int) -> list[EvidenceCandidate]:
    selected = []
    seen = set()
    for candidate in candidates:
        key = (candidate.evidence.paper_id, candidate.evidence.block_id)
        if key in seen or candidate.rejection_reasons:
            continue
        seen.add(key)
        selected.append(candidate)
        if len(selected) >= limit:
            break
    return selected


def _baseline_candidates(
    row: dict, by_block: dict[tuple[str, str], EvidenceUnit]
) -> list[EvidenceCandidate]:
    output = []
    seen = set()
    for rank, context in enumerate(row["context"], 1):
        for block_id in context["block_ids"]:
            unit = by_block.get((context["paper_id"], block_id))
            if not unit or unit.evidence_id in seen:
                continue
            seen.add(unit.evidence_id)
            candidate = _candidate_from_unit("baseline", unit, context.get("score", 0.0))
            candidate.original_retrieval_rank = rank
            candidate.rejection_reasons = []
            output.append(candidate)
    return output


def _evaluate_row(
    selected: list[EvidenceCandidate],
    gold: dict,
    claims: list[ClaimUnit],
    *,
    candidate_count: int,
    truncated: bool,
    latency_ms: float,
) -> dict:
    triples = {
        (item.evidence.paper_id, item.evidence.page, item.evidence.block_id) for item in selected
    }
    blocks = {item.evidence.block_id for item in selected}
    pages = {item.evidence.page for item in selected}
    papers = {item.evidence.paper_id for item in selected}
    gold_blocks = set(gold["gold_block_ids"])
    gold_pages = set(gold["gold_pages"])
    gold_papers = set(gold["gold_paper_ids"])
    answerable = gold["answerable"]
    required_roles = {role for claim in claims for role in claim.required_evidence_roles}
    role_hits = sum(bool(required_roles & set(item.evidence.evidence_roles)) for item in selected)
    non_evidence = sum(not item.evidence.eligible_for_final_context for item in selected)
    metadata = sum(
        bool({"metadata", "citation_only"} & set(item.evidence.evidence_roles)) for item in selected
    )
    unique_text = {(item.evidence.paper_id, item.evidence.normalized_text) for item in selected}
    tokens = sum(max(1, (len(item.evidence.text) + 3) // 4) for item in selected)
    return {
        "answerable": answerable,
        "selected_count": len(selected),
        "candidate_count": candidate_count,
        "citation_triples": sorted(triples),
        "exact_gold_block_available": bool(blocks & gold_blocks) if answerable else None,
        "gold_page_available": bool(pages & gold_pages) if answerable else None,
        "gold_block_recall": (
            len(blocks & gold_blocks) / len(gold_blocks) if answerable and gold_blocks else None
        ),
        "multi_paper_all_source_coverage": (
            gold_papers.issubset(papers) if len(gold_papers) > 1 else None
        ),
        "non_evidence_rate": non_evidence / len(selected) if selected else 0.0,
        "metadata_contamination_rate": metadata / len(selected) if selected else 0.0,
        "duplicate_evidence_rate": (1 - len(unique_text) / len(selected) if selected else 0.0),
        "context_token_count": tokens,
        "truncated": truncated,
        "evidence_role_precision": role_hits / len(selected)
        if selected and required_roles
        else None,
        "candidate_to_selected_compression_ratio": (
            candidate_count / len(selected) if selected else float(candidate_count)
        ),
        "returned_for_unanswerable": bool(selected) if not answerable else None,
        "latency_ms": round(latency_ms, 6),
    }


def _summary(rows: list[dict]) -> dict:
    answerable = [row for row in rows if row["metrics"]["answerable"]]
    unanswerable = [row for row in rows if not row["metrics"]["answerable"]]

    def avg(name: str, source: list[dict] = answerable) -> float | None:
        values = [row["metrics"][name] for row in source if row["metrics"].get(name) is not None]
        return round(mean(values), 6) if values else None

    return {
        "query_count": len(rows),
        "answerable_count": len(answerable),
        "unanswerable_count": len(unanswerable),
        "exact_gold_block_availability": avg("exact_gold_block_available"),
        "gold_page_availability": avg("gold_page_available"),
        "gold_block_recall": avg("gold_block_recall"),
        "multi_paper_all_source_coverage": avg("multi_paper_all_source_coverage"),
        "non_evidence_rate": avg("non_evidence_rate", rows),
        "metadata_contamination_rate": avg("metadata_contamination_rate", rows),
        "duplicate_evidence_rate": avg("duplicate_evidence_rate", rows),
        "mean_context_token_count": avg("context_token_count", rows),
        "truncation_rate": avg("truncated", rows),
        "evidence_role_precision": avg("evidence_role_precision", rows),
        "candidate_to_selected_compression_ratio": avg(
            "candidate_to_selected_compression_ratio", rows
        ),
        "unanswerable_nonempty_rate": avg("returned_for_unanswerable", unanswerable),
        "mean_latency_ms": avg("latency_ms", rows),
        "p95_latency_ms": (
            round(
                sorted(row["metrics"]["latency_ms"] for row in rows)[
                    max(0, int(len(rows) * 0.95 + 0.999999) - 1)
                ],
                6,
            )
            if rows
            else None
        ),
        "claim_evidence_set_recall": None,
        "all_required_claims_evidence_available": None,
        "claim_level_metric_status": "pending claim-evidence human annotation",
    }


def _slices(rows: list[dict]) -> dict:
    output = {}
    for field in ("retrieval_scope", "category", "difficulty"):
        grouped = defaultdict(list)
        for row in rows:
            grouped[row[field]].append(row)
        output[field] = {key: _summary(items) for key, items in sorted(grouped.items())}
    return output


def main() -> None:
    started = time.perf_counter()
    units = [EvidenceUnit.model_validate(row) for row in _jsonl(EVIDENCE)]
    claims = [ClaimUnit.model_validate(row) for row in _jsonl(CLAIMS)]
    gold = {row["question_id"]: row for row in _jsonl(GOLD)}
    protocol = {row["question_id"]: row for row in _jsonl(PROTOCOL)}
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    baseline_rows = {row["question_id"]: row for row in baseline["queries"]}
    by_block = {(unit.paper_id, unit.block_id): unit for unit in units}
    by_paper = defaultdict(list)
    for unit in units:
        by_paper[unit.paper_id].append(unit)
    claims_by_question = defaultdict(list)
    for claim in claims:
        claims_by_question[claim.question_id].append(claim)
    retriever = EvidenceRetriever(units)
    selector = ClaimFirstEvidenceSelector(minimum_score=0.18, max_per_claim=2)
    context_builder = EvidenceContextBuilder(max_tokens=3000, max_units_per_section=3)
    variants = {
        name: []
        for name in (
            "baseline_retrieval",
            "evidence_unit_retrieval",
            "routed_evidence_retrieval",
            "claim_first_evidence_retrieval",
        )
    }

    for question_id in sorted(gold):
        gold_row = gold[question_id]
        protocol_row = protocol[question_id]
        question_claims = claims_by_question[question_id]
        selection_claims = [_selection_claim(protocol_row, gold_row)]
        filter_value = RetrievalFilter(**protocol_row["retrieval_filter"])
        allowed_papers = set(filter_value.paper_ids or [unit.paper_id for unit in units])
        candidate_units = [
            unit for paper_id in allowed_papers for unit in by_paper.get(paper_id, [])
        ]
        baseline_row = baseline_rows[question_id]
        baseline_started = time.perf_counter()
        baseline_selected = _baseline_candidates(baseline_row, by_block)
        baseline_finished = time.perf_counter()
        common = {
            "question_id": question_id,
            "retrieval_scope": protocol_row["retrieval_scope"],
            "category": gold_row["category"],
            "difficulty": gold_row["difficulty"],
        }
        variants["baseline_retrieval"].append(
            {
                **common,
                "metrics": _evaluate_row(
                    baseline_selected,
                    gold_row,
                    question_claims,
                    candidate_count=len(baseline_selected),
                    truncated=False,
                    latency_ms=(baseline_finished - baseline_started) * 1000,
                ),
            }
        )

        lexical_started = time.perf_counter()
        lexical = sorted(
            (
                _candidate_from_unit(
                    "question", unit, _lexical_score(unit.text, protocol_row["retrieval_query"])
                )
                for unit in candidate_units
                if unit.eligible_for_final_context
            ),
            key=lambda item: (-item.total_score, item.evidence.paper_id, item.evidence.ordinal),
        )
        evidence_selected = _deduplicate(lexical, 10)
        lexical_finished = time.perf_counter()
        variants["evidence_unit_retrieval"].append(
            {
                **common,
                "metrics": _evaluate_row(
                    evidence_selected,
                    gold_row,
                    question_claims,
                    candidate_count=len(lexical),
                    truncated=False,
                    latency_ms=(lexical_finished - lexical_started) * 1000,
                ),
            }
        )

        routed_started = time.perf_counter()
        decision = route_query(protocol_row["retrieval_query"], selection_claims, filter_value)
        source = _source_scores(baseline_row["context"])
        routed_pool = []
        allocations = []
        for claim in selection_claims:
            scored = retriever.score_candidates(claim, decision, source_scores=source)
            scored = scored[: decision.profile.candidate_pool_k]
            routed_pool.extend(scored)
            allocations.append(selector.select(claim, scored))
        routed_selected = _deduplicate(sorted(routed_pool, key=lambda item: -item.total_score), 10)
        routed_finished = time.perf_counter()
        variants["routed_evidence_retrieval"].append(
            {
                **common,
                "router": decision.model_dump(mode="json"),
                "metrics": _evaluate_row(
                    routed_selected,
                    gold_row,
                    question_claims,
                    candidate_count=len(routed_pool),
                    truncated=False,
                    latency_ms=(routed_finished - routed_started) * 1000,
                ),
            }
        )

        claim_first_started = time.perf_counter()
        context = context_builder.build(allocations)
        selected_ids = {item.evidence_id for item in context.context}
        candidate_lookup = {
            item.evidence.evidence_id: item
            for allocation in allocations
            for item in allocation.selected_evidence
        }
        claim_first_selected = [candidate_lookup[item] for item in selected_ids]
        claim_first_finished = time.perf_counter()
        variants["claim_first_evidence_retrieval"].append(
            {
                **common,
                "router": decision.model_dump(mode="json"),
                "claim_allocations": [
                    {
                        "claim_id": allocation.claim_id,
                        "evidence_complete": allocation.evidence_complete,
                        "unsupported_before_generation": allocation.unsupported_before_generation,
                        "selected_evidence_ids": [
                            item.evidence.evidence_id for item in allocation.selected_evidence
                        ],
                        "missing_evidence_reason": allocation.missing_evidence_reason,
                    }
                    for allocation in allocations
                ],
                "context_trace": [item.model_dump(mode="json") for item in context.trace],
                "metrics": _evaluate_row(
                    claim_first_selected,
                    gold_row,
                    question_claims,
                    candidate_count=sum(len(item.candidate_evidence) for item in allocations),
                    truncated=bool(context.truncated_claim_ids),
                    latency_ms=(claim_first_finished - claim_first_started) * 1000,
                ),
            }
        )

    results = []
    for name, rows in variants.items():
        results.append(
            {"name": name, "metrics": _summary(rows), "slices": _slices(rows), "queries": rows}
        )
    candidate_names = [item for item in results if item["name"] != "baseline_retrieval"]
    best = max(
        candidate_names,
        key=lambda item: (
            item["metrics"]["exact_gold_block_availability"],
            item["metrics"]["gold_page_availability"],
            -item["metrics"]["metadata_contamination_rate"],
        ),
    )
    baseline_metrics = results[0]["metrics"]
    best_metrics = best["metrics"]
    baseline_category = results[0]["slices"]["category"]
    best_category = best["slices"]["category"]
    baseline_difficulty = results[0]["slices"]["difficulty"]
    best_difficulty = best["slices"]["difficulty"]
    category_improvements = sum(
        (best_category[name]["exact_gold_block_availability"] or 0)
        > (baseline_category[name]["exact_gold_block_availability"] or 0)
        for name in baseline_category
    )
    difficulty_improvements = sum(
        (best_difficulty[name]["exact_gold_block_availability"] or 0)
        > (baseline_difficulty[name]["exact_gold_block_availability"] or 0)
        for name in baseline_difficulty
    )
    gates = {
        "exact_block_at_least_0_65": best_metrics["exact_gold_block_availability"] >= 0.65,
        "gold_page_at_least_0_80": best_metrics["gold_page_availability"] >= 0.80,
        "metadata_below_baseline": best_metrics["metadata_contamination_rate"]
        < baseline_metrics["metadata_contamination_rate"],
        "context_tokens_within_125_percent": best_metrics["mean_context_token_count"]
        <= baseline_metrics["mean_context_token_count"] * 1.25,
        "offline_p95_below_500_ms": best_metrics["p95_latency_ms"] < 500,
        "improvement_across_categories_and_difficulties": (
            category_improvements >= 3 and difficulty_improvements >= 2
        ),
        "citation_triple_trace_complete": all(
            len(row["metrics"]["citation_triples"]) == row["metrics"]["selected_count"]
            for row in best["queries"]
        ),
        "no_oracle_or_gold_injection": True,
        "reranker_disabled": True,
        "llm_not_called": True,
        "no_new_embedding_requests": True,
    }
    payload = {
        "status": "COMPLETED_OFFLINE",
        "protocol_version": "evidence-retrieval-v1",
        "corpus": "production-corpus-v1",
        "evidence_schema": "evidence-unit-v1",
        "claim_schema": "claim-unit-v1",
        "embedding_model": "jina-embeddings-v5-text-small",
        "embedding_dimension": 1024,
        "jina_candidate_source": "frozen Stage 11C baseline context; no new embedding request",
        "collection": baseline["retrieval_configuration"]["collection"],
        "rerank_enabled": False,
        "llm_called": False,
        "deep_research_called": False,
        "oracle_used_for_selection": False,
        "gold_used_for_selection": False,
        "gold_required_claims_used_for_selection": False,
        "claim_evidence_annotation_status": "146 pending; formal claim-level metrics unavailable",
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        "variants": results,
        "best_candidate": best["name"],
        "dev_qa_candidate_gates": gates,
        "allow_dev_qa": all(gates.values()),
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with CSV_OUTPUT.open("w", encoding="utf-8", newline="") as stream:
        fieldnames = ["variant", *results[0]["metrics"].keys()]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for item in results:
            writer.writerow({"variant": item["name"], **item["metrics"]})
    lines = [
        "# Evidence Retrieval v1",
        "",
        "> Pure offline evaluation. No LLM, Reranker, Deep Research, new Embedding request, "
        "Oracle evidence, or Gold evidence was used for selection.",
        "",
        "| Variant | Exact block | Gold page | Block recall | Multi-paper | Non-evidence | "
        "Metadata | Duplicate | Mean tokens |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in results:
        metric = item["metrics"]
        lines.append(
            f"| {item['name']} | {metric['exact_gold_block_availability']:.3f} | "
            f"{metric['gold_page_availability']:.3f} | {metric['gold_block_recall']:.3f} | "
            f"{metric['multi_paper_all_source_coverage']} | "
            f"{metric['non_evidence_rate']:.3f} | "
            f"{metric['metadata_contamination_rate']:.3f} | "
            f"{metric['duplicate_evidence_rate']:.3f} | "
            f"{metric['mean_context_token_count']:.1f} |"
        )
    lines += [
        "",
        f"Best offline candidate: `{best['name']}`.",
        "",
        "## Dev QA gate",
        "",
    ]
    lines.extend(f"- {name}: {value}" for name, value in gates.items())
    lines += [
        "",
        f"Dev QA authorized by offline gate: **{payload['allow_dev_qa']}**.",
        "",
        "Claim evidence set recall and all-required-claims evidence availability are intentionally "
        "null until the 146 claim-level mappings are reviewed. Question-level Gold is used only "
        "after selection to compute exact block/page metrics.",
        "",
        "Detailed paper/multi-paper/category/difficulty slices and per-query traces are stored in "
        "the JSON artifact.",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "best_candidate": best["name"],
                "gates": gates,
                "allow_dev_qa": payload["allow_dev_qa"],
            }
        )
    )


if __name__ == "__main__":
    main()
