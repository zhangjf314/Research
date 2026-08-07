from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from paper_research.config import get_settings
from paper_research.evaluation.rag_official_baseline import percentile

RAG_ROOT = Path("data/evaluation/rag-benchmark")
ANSWERABLE_FAILURE_STAGES = (
    "R0_RETRIEVAL_MISS",
    "C0_CONTEXT_SELECTION_DROP",
    "G0_GENERATION_OMISSION",
    "G1_GENERATION_UNSUPPORTED_OR_INCORRECT",
    "CITATION_FAILURE",
    "SUCCESS",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_chunk_texts(index_version: str | None = None) -> dict[str, str]:
    settings = get_settings()
    names = [
        f"paper_chunks.{index_version or settings.index_version}.jsonl",
        "paper_chunks.jsonl",
    ]
    chunks: dict[str, str] = {}
    for paper_dir in settings.parsed_papers_dir.glob("*"):
        if not paper_dir.is_dir():
            continue
        for name in names:
            path = paper_dir / name
            if not path.exists():
                continue
            for row in read_jsonl(path):
                chunks[str(row["chunk_id"])] = str(row.get("chunk_text") or "")
            break
    return chunks


def enrich_context_items(
    items: list[dict[str, Any]], chunk_texts: dict[str, str]
) -> list[dict[str, Any]]:
    enriched = []
    for item in items:
        copy = dict(item)
        text = chunk_texts.get(str(item.get("chunk_id")), "")
        copy["estimated_tokens"] = (
            max(1, (len(text) + 3) // 4) if text else _fallback_tokens(item)
        )
        copy["evidence_text"] = text
        enriched.append(copy)
    return enriched


def _fallback_tokens(item: dict[str, Any]) -> int:
    return max(1, len(item.get("block_ids") or []) * 80)


def reconstruct_baseline_context(
    ranked_results: list[dict[str, Any]],
    chunk_texts: dict[str, str],
    *,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    return enrich_context_items(ranked_results[:top_k], chunk_texts)


def score_budgeted_deduplicated_context(
    ranked_results: list[dict[str, Any]],
    chunk_texts: dict[str, str],
    *,
    token_budget: int,
) -> list[dict[str, Any]]:
    selected = []
    used_tokens = 0
    seen_blocks: set[str] = set()
    for item in enrich_context_items(ranked_results, chunk_texts):
        blocks = set(str(block) for block in item.get("block_ids", []))
        if blocks and blocks <= seen_blocks:
            continue
        tokens = int(item["estimated_tokens"])
        if used_tokens + tokens > token_budget and selected:
            continue
        selected.append(item)
        used_tokens += tokens
        seen_blocks.update(blocks)
        if used_tokens >= token_budget:
            break
    return selected


def diversity_aware_context(
    ranked_results: list[dict[str, Any]],
    chunk_texts: dict[str, str],
    *,
    token_budget: int,
    paper_cap: int = 2,
    section_cap: int = 2,
) -> list[dict[str, Any]]:
    candidates = enrich_context_items(ranked_results, chunk_texts)
    selected: list[dict[str, Any]] = []
    used_tokens = 0
    paper_counts: Counter[str] = Counter()
    section_counts: Counter[str] = Counter()
    selected_chunks: set[str] = set()

    def add(item: dict[str, Any]) -> bool:
        nonlocal used_tokens
        tokens = int(item["estimated_tokens"])
        if used_tokens + tokens > token_budget and selected:
            return False
        selected.append(item)
        used_tokens += tokens
        selected_chunks.add(str(item.get("chunk_id")))
        paper_counts[str(item.get("paper_id"))] += 1
        section_counts[_section_key(item)] += 1
        return True

    for item in candidates:
        if paper_counts[str(item.get("paper_id"))] >= paper_cap:
            continue
        if section_counts[_section_key(item)] >= section_cap:
            continue
        add(item)
        if used_tokens >= token_budget:
            break
    for item in candidates:
        if used_tokens >= token_budget:
            break
        if str(item.get("chunk_id")) in selected_chunks:
            continue
        add(item)
    return selected


def _section_key(item: dict[str, Any]) -> str:
    return " > ".join(str(part) for part in item.get("section_path", [])) or "unknown"


def build_trace(
    gold: dict[str, Any],
    retrieval: dict[str, Any],
    generation: dict[str, Any] | None,
    final_context: list[dict[str, Any]],
) -> dict[str, Any]:
    retrieved_top10 = _blocks(retrieval.get("ranked_results", [])[:10])
    retrieved_top20 = _blocks(retrieval.get("ranked_results", [])[:20])
    final_blocks = _blocks(final_context)
    answer = (generation or {}).get("answer") or {}
    required = []
    for claim in gold.get("required_claims", []):
        claim_blocks = set(claim.get("gold_block_ids") or gold.get("gold_block_ids") or [])
        generated = _claim_generated(claim, answer)
        cited = _claim_cited(claim_blocks, answer)
        required.append(
            {
                "claim_id": claim.get("claim_id"),
                "gold_block_ids": sorted(claim_blocks),
                "retrieved": bool(claim_blocks & set(retrieved_top20)),
                "in_final_context": bool(claim_blocks & set(final_blocks)),
                "generated": generated,
                "cited": cited,
            }
        )
    return {
        "question_id": gold["question_id"],
        "retrieved_block_ids": retrieval.get("retrieved_block_ids", []),
        "retrieved_top10_block_ids": retrieved_top10,
        "retrieved_top20_block_ids": retrieved_top20,
        "final_context_block_ids": final_blocks,
        "final_context_paper_ids": [item.get("paper_id") for item in final_context],
        "context_token_count": sum(
            int(item.get("estimated_tokens") or 0) for item in final_context
        ),
        "context_block_count": len(final_context),
        "gold_block_ids": gold.get("gold_block_ids", []),
        "required_claims": required,
        "answerable": gold.get("answerable"),
        "category": gold.get("category"),
        "difficulty": gold.get("difficulty"),
    }


def _blocks(items: list[dict[str, Any]]) -> list[str]:
    return [
        str(block)
        for item in items
        for block in (item.get("block_ids") or [item.get("chunk_id")])
        if block
    ]


def _claim_generated(claim: dict[str, Any], answer: dict[str, Any]) -> bool:
    required = str(claim.get("text") or "")
    generated_claims = answer.get("claims") or []
    return any(
        _term_overlap(required, str(item.get("text") or "")) >= 0.35
        for item in generated_claims
    )


def _claim_cited(claim_blocks: set[str], answer: dict[str, Any]) -> bool:
    for generated_claim in answer.get("claims") or []:
        for citation in generated_claim.get("citations") or []:
            if str(citation.get("block_id")) in claim_blocks:
                return True
    return False


def _term_overlap(expected: str, actual: str) -> float:
    expected_terms = {term for term in expected.lower().split() if len(term) > 2}
    actual_terms = {term for term in actual.lower().split() if len(term) > 2}
    return len(expected_terms & actual_terms) / max(1, len(expected_terms))


def aggregate_traces(traces: list[dict[str, Any]]) -> dict[str, Any]:
    answerable = [trace for trace in traces if trace.get("answerable")]
    claim_total = 0
    claim_retrieved = 0
    claim_context = 0
    claim_generated = 0
    claim_cited = 0
    retrieved_claims = 0
    retained_retrieved_claims = 0
    retrieved_gold_blocks = 0
    retained_gold_blocks = 0
    dropped_gold_blocks = 0
    token_counts = []
    block_counts = []
    gold_density_values = []
    duplicate_rates = []
    paper_concentration = []
    unique_paper_counts = []
    unique_section_counts = []
    for trace in answerable:
        token_counts.append(float(trace["context_token_count"]))
        block_counts.append(float(trace["context_block_count"]))
        final_blocks = set(trace["final_context_block_ids"])
        retrieved_blocks = set(trace["retrieved_top20_block_ids"])
        gold_blocks = set(trace["gold_block_ids"])
        retrieved_gold = gold_blocks & retrieved_blocks
        retained_gold = retrieved_gold & final_blocks
        retrieved_gold_blocks += len(retrieved_gold)
        retained_gold_blocks += len(retained_gold)
        dropped_gold_blocks += len(retrieved_gold - final_blocks)
        if trace["context_block_count"]:
            gold_density_values.append(
                len(gold_blocks & final_blocks) / trace["context_block_count"]
            )
        duplicate_rates.append(_duplicate_rate(trace["final_context_block_ids"]))
        papers = trace["final_context_paper_ids"]
        unique_paper_counts.append(len(set(papers)))
        unique_section_counts.append(0)
        paper_concentration.append(_max_frequency_rate(papers))
        for claim in trace["required_claims"]:
            claim_total += 1
            if claim["retrieved"]:
                claim_retrieved += 1
                retrieved_claims += 1
                if claim["in_final_context"]:
                    retained_retrieved_claims += 1
            if claim["in_final_context"]:
                claim_context += 1
            if claim["generated"]:
                claim_generated += 1
            if claim["cited"]:
                claim_cited += 1
    funnel = exclusive_failure_funnel(answerable)
    answerable_with_gold_at20 = sum(
        1
        for trace in answerable
        if set(trace["gold_block_ids"]) & set(trace["retrieved_top20_block_ids"])
    )
    full_context_count = sum(
        1
        for trace in answerable
        if all(claim["in_final_context"] for claim in trace["required_claims"])
    )
    required_claims_dropped = retrieved_claims - retained_retrieved_claims
    same_paper_concentration = (
        round(mean(paper_concentration), 6) if paper_concentration else 0.0
    )
    unique_paper_mean = (
        round(mean(unique_paper_counts), 6) if unique_paper_counts else 0.0
    )
    unique_section_mean = (
        round(mean(unique_section_counts), 6) if unique_section_counts else 0.0
    )
    return {
        "answerable_count": len(answerable),
        "gold_evidence_available_at_20": _rate(
            answerable_with_gold_at20,
            len(answerable),
        ),
        "gold_evidence_retained_final_context": _rate(
            retained_gold_blocks, retrieved_gold_blocks
        ),
        "required_claim_evidence_available_at_20": _rate(claim_retrieved, claim_total),
        "required_claim_evidence_retained_final_context": _rate(
            claim_context, claim_total
        ),
        "required_claim_context_retention": _rate(retained_retrieved_claims, retrieved_claims),
        "required_claim_evidence_coverage_in_final_context": _rate(
            claim_context, claim_total
        ),
        "full_required_claim_evidence_coverage_in_final_context": _rate(
            full_context_count,
            len(answerable),
        ),
        "retrieved_gold_blocks_dropped_by_context": dropped_gold_blocks,
        "retrieved_required_claims_dropped_by_context": required_claims_dropped,
        "claim_generated_rate": _rate(claim_generated, claim_total),
        "claim_cited_rate": _rate(claim_cited, claim_total),
        "context_token_p50": percentile(token_counts, 0.5),
        "context_token_p95": percentile(token_counts, 0.95),
        "context_block_p50": percentile(block_counts, 0.5),
        "context_block_p95": percentile(block_counts, 0.95),
        "context_gold_density": (
            round(mean(gold_density_values), 6) if gold_density_values else 0.0
        ),
        "context_redundancy": round(mean(duplicate_rates), 6) if duplicate_rates else 0.0,
        "same_paper_block_concentration": same_paper_concentration,
        "unique_paper_count_mean": unique_paper_mean,
        "unique_section_count_mean": unique_section_mean,
        "exclusive_failure_funnel": funnel,
    }


def exclusive_failure_funnel(answerable_traces: list[dict[str, Any]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    items = []
    for trace in answerable_traces:
        claims = trace["required_claims"]
        if not any(claim["retrieved"] for claim in claims):
            stage = "R0_RETRIEVAL_MISS"
        elif any(claim["retrieved"] and not claim["in_final_context"] for claim in claims):
            stage = "C0_CONTEXT_SELECTION_DROP"
        elif any(claim["in_final_context"] and not claim["generated"] for claim in claims):
            stage = "G0_GENERATION_OMISSION"
        elif any(claim["generated"] and not claim["cited"] for claim in claims):
            stage = "CITATION_FAILURE"
        elif all(claim["generated"] and claim["cited"] for claim in claims):
            stage = "SUCCESS"
        else:
            stage = "G1_GENERATION_UNSUPPORTED_OR_INCORRECT"
        counts[stage] += 1
        items.append({"question_id": trace["question_id"], "primary_failure_stage": stage})
    return {
        "counts": {stage: counts.get(stage, 0) for stage in ANSWERABLE_FAILURE_STAGES},
        "items": items,
    }


def context_selection_hypothesis_supported(
    metrics: dict[str, Any], pre_availability: float
) -> bool:
    answerable = metrics["answerable_count"] or 1
    c0_count = metrics["exclusive_failure_funnel"]["counts"]["C0_CONTEXT_SELECTION_DROP"]
    c0_rate = c0_count / answerable
    retention = metrics["required_claim_context_retention"]
    final_coverage = metrics["required_claim_evidence_coverage_in_final_context"]
    return c0_rate >= 0.15 or retention < 0.90 or pre_availability - final_coverage >= 0.10


def offline_selector_gate(c0: dict[str, Any], candidate: dict[str, Any]) -> bool:
    coverage_gain = (
        candidate["required_claim_evidence_coverage_in_final_context"]
        - c0["required_claim_evidence_coverage_in_final_context"]
    )
    full_gain = (
        candidate["full_required_claim_evidence_coverage_in_final_context"]
        - c0["full_required_claim_evidence_coverage_in_final_context"]
    )
    token_ok = (candidate["context_token_p95"] or 0) <= (
        c0["context_token_p95"] or 0
    ) * 1.10
    return (coverage_gain >= 0.05 or full_gain >= 0.05) and token_ok


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _duplicate_rate(values: list[str]) -> float:
    if not values:
        return 0.0
    return 1.0 - (len(set(values)) / len(values))


def _max_frequency_rate(values: list[Any]) -> float:
    if not values:
        return 0.0
    counts = Counter(str(value) for value in values)
    return max(counts.values()) / len(values)


def write_markdown(path: Path, payload: dict[str, Any], title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {title}",
        "",
        f"- dev_questions: `{payload.get('dev_questions')}`",
        f"- dev_answerable: `{payload.get('dev_answerable')}`",
        f"- test_questions_evaluated: `{payload.get('test_questions_evaluated')}`",
        f"- context_trace_source: `{payload.get('context_trace_source')}`",
        "",
        "```json",
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
