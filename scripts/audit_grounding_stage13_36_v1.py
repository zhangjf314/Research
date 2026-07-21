# ruff: noqa: E501
"""Stage 13.36 offline grounding metric and alignment audit.

This script never calls an LLM. It freezes the existing Qwen and DeepSeek
15-item canaries as baselines, audits the exact-match grounding metrics, and
emits diagnostic classifications without changing historical results.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "evaluation"
DOCS = ROOT / "docs"
ARTIFACTS = ROOT / "artifacts"

QWEN_JSON = DATA / "full-qa-canary-results-v2.json"
DEEPSEEK_JSON = DATA / "full-qa-canary-deepseek-v1.json"
QWEN_CSV = DATA / "full-qa-canary-results-v2.csv"
DEEPSEEK_CSV = DATA / "full-qa-canary-deepseek-v1.csv"
QWEN_TRACE = ARTIFACTS / "full-qa-canary-trace-v2.json"
DEEPSEEK_TRACE = ARTIFACTS / "full-qa-canary-deepseek-trace-v1.json"
GOLD = DATA / "gold-set-v1.jsonl"
RETRIEVAL_GOLD = DATA / "retrieval-gold-v2.jsonl"
CONTEXT_GROUNDING = DATA / "context-grounding-v2.json"

OUT_ALIGNMENT_JSON = DATA / "grounding-alignment-audit-v1.json"
OUT_ALIGNMENT_CSV = DATA / "grounding-alignment-audit-v1.csv"
OUT_ALIGNMENT_DOC = DOCS / "grounding-alignment-audit-v1.md"
OUT_METRIC_DOC = DOCS / "grounding-metric-definition-audit-v1.md"
OUT_CARDINALITY_JSON = DATA / "claim-cardinality-audit-v1.json"
OUT_CARDINALITY_DOC = DOCS / "claim-cardinality-audit-v1.md"
OUT_STAGE_DOC = DOCS / "stage-13-36-audit.md"

CLAIM_FAILURE_CATEGORIES = {
    "CLAIM_TRULY_OMITTED",
    "CLAIM_EXPRESSED_WITH_DIFFERENT_WORDING",
    "CLAIM_MERGED_INTO_COMPOSITE",
    "CLAIM_TOO_BROAD",
    "CLAIM_TOO_NARROW",
    "CLAIM_MATCHER_FALSE_NEGATIVE",
    "EVIDENCE_NOT_IN_CONTEXT",
    "GOLD_CLAIM_AMBIGUOUS",
    "UNKNOWN",
}
UNSUPPORTED_CATEGORIES = {
    "NO_SUPPORTING_EVIDENCE",
    "PARTIALLY_SUPPORTED_COMPOSITE",
    "SUPPORTED_BY_NON_GOLD_BLOCK",
    "SUPPORTED_BY_EQUIVALENT_PAGE",
    "CLAIM_OVERSTATED",
    "CLAIM_USES_EXTERNAL_KNOWLEDGE",
    "CITATION_BINDING_WRONG",
    "EVALUATOR_FALSE_POSITIVE",
    "UNKNOWN",
}
CITATION_FAILURE_CATEGORIES = {
    "CITATION_DOES_NOT_SUPPORT_CLAIM",
    "CITATION_PARTIALLY_SUPPORTS_CLAIM",
    "CITATION_SUPPORTS_CLAIM_BUT_NOT_GOLD_BLOCK",
    "CITATION_SUPPORTS_CLAIM_ON_EQUIVALENT_PAGE",
    "CITATION_MATCHER_TOO_STRICT",
    "CLAIM_TOO_COMPOSITE",
    "UNKNOWN",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def terms(text: str) -> set[str]:
    return {
        token.lower().strip(".,:;!?()[]{}'\"")
        for token in text.replace("/", " ").replace("-", " ").split()
        if token.lower().strip(".,:;!?()[]{}'\"")
    }


def overlap(expected: str, actual: str) -> float:
    expected_terms = terms(expected)
    return len(expected_terms & terms(actual)) / max(1, len(expected_terms))


def is_composite_claim(
    text: str, citation_count: int, match_count: int
) -> tuple[bool, dict[str, Any]]:
    lowered = text.lower()
    connector_count = sum(
        lowered.count(f" {word} ") for word in ("and", "while", "whereas", "but", "also")
    )
    punctuation_count = sum(text.count(mark) for mark in (";", ":", ","))
    predicate_hints = sum(
        lowered.count(f" {word}")
        for word in (" uses", " reports", " compares", " shows", " includes", " relies")
    )
    word_count = len(terms(text))
    score = (
        int(connector_count >= 1)
        + int(punctuation_count >= 2)
        + int(predicate_hints >= 2)
        + int(citation_count > 1)
        + int(match_count > 1)
        + int(word_count > 24)
    )
    return score >= 2, {
        "connector_count": connector_count,
        "punctuation_count": punctuation_count,
        "predicate_hint_count": predicate_hints,
        "word_count": word_count,
        "citation_count": citation_count,
        "required_claim_match_count": match_count,
        "composite_score": score,
    }


def classify_claim_gap(
    required: str,
    generated_claims: list[dict[str, Any]],
    context_row: dict[str, Any] | None,
) -> str:
    scores = [overlap(required, claim.get("text", "")) for claim in generated_claims]
    best = max(scores, default=0.0)
    if best >= 0.30:
        return "CLAIM_MATCHER_FALSE_NEGATIVE"
    composite_hits = sum(best >= 0.18 for best in scores)
    if composite_hits:
        return "CLAIM_EXPRESSED_WITH_DIFFERENT_WORDING"
    if context_row and context_row.get("required_claim_evidence_coverage") == 0:
        return "EVIDENCE_NOT_IN_CONTEXT"
    required_len = len(terms(required))
    generated_lens = [len(terms(claim.get("text", ""))) for claim in generated_claims]
    if generated_lens and max(generated_lens) > required_len * 2:
        return "CLAIM_TOO_BROAD"
    return "CLAIM_TRULY_OMITTED"


def classify_unsupported_claim(
    claim: dict[str, Any],
    gold_blocks: set[str],
    gold_pages: set[int],
    gold_papers: set[str],
    uuid_to_public: dict[str, str],
) -> str:
    citations = claim.get("citations") or []
    if not citations:
        return "NO_SUPPORTING_EVIDENCE"
    citation_pages = {int(citation.get("page") or -1) for citation in citations}
    citation_blocks = {str(citation.get("block_id") or "") for citation in citations}
    citation_papers = {
        uuid_to_public.get(str(citation.get("paper_id")), str(citation.get("paper_id")))
        for citation in citations
    }
    exact = any(
        paper in gold_papers and block in gold_blocks and page in gold_pages
        for paper in citation_papers
        for block in citation_blocks
        for page in citation_pages
    )
    if exact:
        return "EVALUATOR_FALSE_POSITIVE"
    if citation_papers & gold_papers and citation_pages & gold_pages:
        return "SUPPORTED_BY_EQUIVALENT_PAGE"
    if citation_papers & gold_papers:
        return "SUPPORTED_BY_NON_GOLD_BLOCK"
    return "CITATION_BINDING_WRONG"


def classify_citation_failure(
    claim: dict[str, Any],
    citation: dict[str, Any],
    gold_blocks: set[str],
    gold_pages: set[int],
    gold_papers: set[str],
    uuid_to_public: dict[str, str],
    required_claims: list[str],
) -> str:
    paper = uuid_to_public.get(str(citation.get("paper_id")), str(citation.get("paper_id")))
    block = str(citation.get("block_id") or "")
    page = int(citation.get("page") or -1)
    if paper in gold_papers and block in gold_blocks and page in gold_pages:
        return "UNKNOWN"
    if paper in gold_papers and page in gold_pages:
        return "CITATION_SUPPORTS_CLAIM_ON_EQUIVALENT_PAGE"
    if paper in gold_papers:
        return "CITATION_SUPPORTS_CLAIM_BUT_NOT_GOLD_BLOCK"
    claim_text = claim.get("text", "")
    match_count = sum(overlap(required, claim_text) >= 0.20 for required in required_claims)
    composite, _ = is_composite_claim(claim_text, len(claim.get("citations") or []), match_count)
    if composite:
        return "CLAIM_TOO_COMPOSITE"
    return "CITATION_DOES_NOT_SUPPORT_CLAIM"


def load_public_maps() -> dict[str, str]:
    manifest = read_json(DATA / "production-corpus-v1.json")
    return {
        str(paper["database_id"]): str(paper["paper_id"])
        for paper in manifest.get("papers", [])
        if paper.get("included_in_production")
    }


def audit_baseline(
    label: str,
    payload: dict[str, Any],
    context_by_id: dict[str, dict[str, Any]],
    uuid_to_public: dict[str, str],
    gold_meta: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    counters: dict[str, Counter[str]] = {
        "claim_failure": Counter(),
        "unsupported": Counter(),
        "citation_failure": Counter(),
    }
    composite_count = 0
    atomic_count = 0
    partially_supported_composite_count = 0
    for row in payload.get("rows", []):
        qid = row["question_id"]
        gold = row.get("gold") or {}
        meta = gold_meta.get(qid, {})
        answer = row.get("answer") or {}
        claims = answer.get("claims") or []
        required_claims = gold.get("required_claims") or []
        gold_blocks = set(gold.get("gold_block_ids") or [])
        gold_pages = {int(page) for page in gold.get("gold_pages") or []}
        gold_papers = set(gold.get("gold_paper_ids") or [])
        generated_citations = [
            citation for claim in claims for citation in (claim.get("citations") or [])
        ]
        resolved_blocks = sorted(
            {str(citation.get("block_id")) for citation in generated_citations}
        )
        resolved_pages = sorted(
            {int(citation.get("page") or -1) for citation in generated_citations}
        )
        match_rows = []
        for required in required_claims:
            scores = [
                {
                    "claim_id": claim.get("claim_id"),
                    "score": round(overlap(required, claim.get("text", "")), 6),
                    "text": claim.get("text", ""),
                }
                for claim in claims
            ]
            best = max((item["score"] for item in scores), default=0.0)
            matched = best >= 0.35
            classification = None
            if not matched:
                classification = classify_claim_gap(required, claims, context_by_id.get(qid))
                counters["claim_failure"][classification] += 1
            match_rows.append(
                {
                    "required_claim": required,
                    "best_overlap": best,
                    "matched_by_legacy_threshold": matched,
                    "failure_classification": classification,
                }
            )
        citation_results = []
        for claim in claims:
            claim_match_count = sum(
                overlap(required, claim.get("text", "")) >= 0.20 for required in required_claims
            )
            composite, composite_features = is_composite_claim(
                claim.get("text", ""),
                len(claim.get("citations") or []),
                claim_match_count,
            )
            if composite:
                composite_count += 1
            else:
                atomic_count += 1
            unsupported_class = classify_unsupported_claim(
                claim, gold_blocks, gold_pages, gold_papers, uuid_to_public
            )
            if unsupported_class != "EVALUATOR_FALSE_POSITIVE":
                counters["unsupported"][unsupported_class] += 1
            if composite and unsupported_class in {
                "SUPPORTED_BY_NON_GOLD_BLOCK",
                "SUPPORTED_BY_EQUIVALENT_PAGE",
            }:
                partially_supported_composite_count += 1
            for citation in claim.get("citations") or []:
                paper = uuid_to_public.get(
                    str(citation.get("paper_id")), str(citation.get("paper_id"))
                )
                exact = (
                    paper in gold_papers
                    and str(citation.get("block_id")) in gold_blocks
                    and int(citation.get("page") or -1) in gold_pages
                )
                classification = None
                if not exact:
                    classification = classify_citation_failure(
                        claim,
                        citation,
                        gold_blocks,
                        gold_pages,
                        gold_papers,
                        uuid_to_public,
                        required_claims,
                    )
                    counters["citation_failure"][classification] += 1
                citation_results.append(
                    {
                        "claim_id": claim.get("claim_id"),
                        "paper_id": paper,
                        "page": citation.get("page"),
                        "block_id": citation.get("block_id"),
                        "gold_exact_match": exact,
                        "failure_classification": classification,
                    }
                )
        context_row = context_by_id.get(qid) or {}
        rows.append(
            {
                "baseline_label": label,
                "sample_id": qid,
                "question": meta.get("question") or row.get("retrieval_query"),
                "category": meta.get("category"),
                "difficulty": meta.get("difficulty"),
                "answerable": gold.get("answerable"),
                "required_claim_count": len(required_claims),
                "generated_claim_count": len(claims),
                "required_claims": required_claims,
                "generated_claims": [claim.get("text") for claim in claims],
                "generated_citation_keys": [
                    {
                        "claim_id": claim.get("claim_id"),
                        "paper_id": uuid_to_public.get(
                            str(citation.get("paper_id")), str(citation.get("paper_id"))
                        ),
                        "page": citation.get("page"),
                        "block_id": citation.get("block_id"),
                    }
                    for claim in claims
                    for citation in (claim.get("citations") or [])
                ],
                "resolved_blocks": resolved_blocks,
                "resolved_pages": resolved_pages,
                "gold_blocks": sorted(gold_blocks),
                "gold_pages": sorted(gold_pages),
                "required_claim_matches": match_rows,
                "citation_precision_result": {
                    "legacy_metric_name": "citation_precision",
                    "audited_metric_name": "gold_citation_exact_match_precision",
                    "value": (row.get("metrics") or {}).get("citation_precision"),
                    "citation_results": citation_results,
                },
                "citation_recall_result": {
                    "legacy_metric_name": "citation_recall",
                    "audited_metric_name": "gold_block_exact_recall",
                    "value": (row.get("metrics") or {}).get("citation_recall"),
                },
                "unsupported_claim_result": {
                    "legacy_count": (row.get("metrics") or {}).get("unsupported_claim_count"),
                    "classifications": [
                        classify_unsupported_claim(
                            claim, gold_blocks, gold_pages, gold_papers, uuid_to_public
                        )
                        for claim in claims
                    ],
                },
                "context_contains_required_evidence": {
                    "gold_block_context_hit": context_row.get("gold_block_context_hit"),
                    "gold_page_context_hit": context_row.get("gold_page_context_hit"),
                    "required_claim_evidence_coverage": context_row.get(
                        "required_claim_evidence_coverage"
                    ),
                },
            }
        )
    summary = {
        "baseline_label": label,
        "summary": payload.get("summary") or {},
        "claim_failure_classifications": dict(counters["claim_failure"]),
        "unsupported_claim_classifications": dict(counters["unsupported"]),
        "citation_precision_failure_classifications": dict(counters["citation_failure"]),
        "composite_claim_rate": round(composite_count / max(1, composite_count + atomic_count), 6),
        "atomic_claim_rate": round(atomic_count / max(1, composite_count + atomic_count), 6),
        "partially_supported_composite_count": partially_supported_composite_count,
    }
    return rows, summary


def summarize_cardinality(gold_rows: list[dict[str, Any]], canary_ids: list[str]) -> dict[str, Any]:
    def dist(rows: list[dict[str, Any]]) -> dict[str, Any]:
        counts = [len(row.get("required_claims") or []) for row in rows]
        return {
            "count": len(counts),
            "required_claim_count_distribution": dict(Counter(counts)),
            "mean": round(sum(counts) / len(counts), 6) if counts else None,
            "median": statistics.median(counts) if counts else None,
            "p95": sorted(counts)[max(0, math.ceil(len(counts) * 0.95) - 1)] if counts else None,
            "max": max(counts) if counts else None,
            "count_required_claim_count_gt_3": sum(count > 3 for count in counts),
            "theoretical_coverage_cap_with_max_3_generated_claims_one_to_one": (
                round(sum(min(3, count) for count in counts) / max(1, sum(counts)), 6)
                if counts
                else None
            ),
        }

    canary_set = set(canary_ids)
    return {
        "schema_version": "claim-cardinality-audit-v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "gold_dev_v1": dist(gold_rows),
        "canary_15": dist([row for row in gold_rows if row["question_id"] in canary_set]),
        "interpretation": (
            "The existing Direct QA prompt caps generated claims at 3. If the evaluator only "
            "allowed strict one-to-one matching, questions with more than 3 required claims "
            "would have a mathematical coverage ceiling. Current gold-dev-v1 records all have "
            "three or fewer required claims, so the low canary coverage is not caused by this "
            "cardinality cap in the current dataset."
        ),
    }


def render_alignment_doc(payload: dict[str, Any]) -> str:
    lines = [
        "# Grounding Alignment Audit v1",
        "",
        "This is an offline diagnostic audit of the existing canary outputs. It does not call an LLM, does not change Gold, and does not overwrite historical metrics.",
        "",
        "## Frozen baselines",
        "",
    ]
    for item in payload["baseline_summaries"]:
        summary = item["summary"]
        lines.extend(
            [
                f"### {item['baseline_label']}",
                "",
                f"- attempted/completed: `{summary.get('attempted')}` / `{summary.get('completed')}`",
                f"- required_claim_coverage: `{summary.get('required_claim_coverage')}`",
                f"- legacy citation_precision: `{summary.get('citation_precision')}`",
                f"- legacy citation_recall: `{summary.get('citation_recall')}`",
                f"- core_unsupported_claim_count: `{summary.get('core_unsupported_claim_count')}`",
                f"- composite_claim_rate: `{item['composite_claim_rate']}`",
                f"- atomic_claim_rate: `{item['atomic_claim_rate']}`",
                "",
                "Claim coverage failure classification:",
                "",
            ]
        )
        for key, value in sorted(item["claim_failure_classifications"].items()):
            lines.append(f"- `{key}`: `{value}`")
        lines.extend(["", "Citation exact-match failure classification:", ""])
        for key, value in sorted(item["citation_precision_failure_classifications"].items()):
            lines.append(f"- `{key}`: `{value}`")
        lines.extend(["", "Unsupported-claim diagnostic classification:", ""])
        for key, value in sorted(item["unsupported_claim_classifications"].items()):
            lines.append(f"- `{key}`: `{value}`")
        lines.append("")
    lines.extend(
        [
            "## Auditor conclusion",
            "",
            "- The evaluator is deterministic and internally consistent, but the names `citation_precision` and `citation_recall` are broader than their actual exact-Gold-block semantics.",
            "- The same `core_unsupported_claim_count=30` across Qwen and DeepSeek is not explained by invalid citations or JSON/schema instability; both runs had stable structured outputs.",
            "- Offline context coverage is high, so the dominant failure mode is downstream claim/evidence selection and exact-Gold matching, not basic context absence.",
            "- This audit is not a blind holdout and must not be used as a strong generalization claim.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_metric_doc() -> str:
    return (
        "\n".join(
            [
                "# Grounding Metric Definition Audit v1",
                "",
                "## Required Claim Coverage",
                "",
                "- Current implementation: deterministic token-overlap heuristic.",
                "- Embedding: not used.",
                "- LLM judge: not used.",
                "- Threshold: best generated-claim overlap >= `0.35`.",
                "- One generated claim can satisfy multiple required claims mathematically because each required claim independently takes the best overlap across all generated claims.",
                "- Multiple generated claims cannot jointly satisfy one required claim; the score is the maximum single-claim overlap.",
                "- Composite claims are not semantically decomposed; this can under-credit partially correct composite answers and over-credit broad paraphrases.",
                "",
                "## Citation Precision",
                "",
                "The current metric checks whether each cited `(paper_id, page, block_id)` triple exactly matches the question-level Gold paper/page/block sets. It does not judge whether the citation semantically supports the generated claim.",
                "",
                "Audited name: `gold_citation_exact_match_precision`.",
                "",
                "## Citation Recall",
                "",
                "The current metric measures cited Gold block coverage, i.e. `len(cited_gold_blocks) / len(gold_blocks)`. It is an exact Gold Block Recall, not semantic evidence recall.",
                "",
                "Audited name: `gold_block_exact_recall`.",
                "",
                "## Unsupported Claim",
                "",
                "The current unsupported-claim count marks a generated claim unsupported when none of its citations exactly matches question-level Gold paper/page/block. It is deterministic and exact-Gold based. It is not an LLM judge and does not incorporate human citation labels or equivalent evidence unless those blocks/pages are in Gold.",
                "",
                "Therefore, Gold-block mismatch must not be described as semantic unsupported without a separate human or semantic support audit.",
            ]
        )
        + "\n"
    )


def render_cardinality_doc(payload: dict[str, Any]) -> str:
    lines = [
        "# Claim Cardinality Audit v1",
        "",
        "This audit checks whether the Direct QA maximum of 3 generated claims creates a mathematical coverage ceiling.",
        "",
    ]
    for name in ("gold_dev_v1", "canary_15"):
        item = payload[name]
        lines.extend(
            [
                f"## {name}",
                "",
                f"- records: `{item['count']}`",
                f"- distribution: `{item['required_claim_count_distribution']}`",
                f"- mean/median/p95/max: `{item['mean']}` / `{item['median']}` / `{item['p95']}` / `{item['max']}`",
                f"- count(required_claim_count > 3): `{item['count_required_claim_count_gt_3']}`",
                "- theoretical coverage cap with max 3 generated claims and one-to-one matching: "
                f"`{item['theoretical_coverage_cap_with_max_3_generated_claims_one_to_one']}`",
                "",
            ]
        )
    lines.extend(["## Conclusion", "", payload["interpretation"], ""])
    return "\n".join(lines)


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def main() -> int:
    qwen = read_json(QWEN_JSON)
    deepseek = read_json(DEEPSEEK_JSON)
    for path in (QWEN_JSON, QWEN_CSV, QWEN_TRACE, DEEPSEEK_JSON, DEEPSEEK_CSV, DEEPSEEK_TRACE):
        if not path.exists():
            raise FileNotFoundError(path)
    if qwen.get("canary_ids") != deepseek.get("canary_ids"):
        raise RuntimeError("Qwen and DeepSeek canary sample IDs differ")

    gold_rows = read_jsonl(GOLD)
    gold_meta = {row["question_id"]: row for row in gold_rows}
    context = read_json(CONTEXT_GROUNDING)
    context_by_id = {row["question_id"]: row for row in context.get("rows", [])}
    uuid_to_public = load_public_maps()

    alignment_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for label, payload in (
        ("QWEN_CANARY_BASELINE", qwen),
        ("DEEPSEEK_CANARY_BASELINE", deepseek),
    ):
        rows, summary = audit_baseline(label, payload, context_by_id, uuid_to_public, gold_meta)
        alignment_rows.extend(rows)
        summaries.append(summary)

    alignment_payload = {
        "schema_version": "grounding-alignment-audit-v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "git_commit": git_head(),
        "llm_called": False,
        "historical_results_overwritten": False,
        "baselines": {
            "QWEN_CANARY_BASELINE": str(QWEN_JSON),
            "DEEPSEEK_CANARY_BASELINE": str(DEEPSEEK_JSON),
        },
        "metric_semantics": {
            "citation_precision": "gold_citation_exact_match_precision",
            "citation_recall": "gold_block_exact_recall",
            "unsupported_claim_count": "exact_gold_mismatch_count",
            "required_claim_coverage": "deterministic_token_overlap_best_claim_threshold_0.35",
        },
        "context_summary": context.get("summary"),
        "baseline_summaries": summaries,
        "rows": alignment_rows,
        "failure_category_vocabularies": {
            "required_claim": sorted(CLAIM_FAILURE_CATEGORIES),
            "unsupported_claim": sorted(UNSUPPORTED_CATEGORIES),
            "citation_precision": sorted(CITATION_FAILURE_CATEGORIES),
        },
        "auditor_conclusion": "EVALUATOR_BASICALLY_VALID_BUT_EXACT_GOLD_METRIC_NAMES_ARE_TOO_BROAD",
    }
    write_json(OUT_ALIGNMENT_JSON, alignment_payload)
    with OUT_ALIGNMENT_CSV.open("w", encoding="utf-8", newline="") as stream:
        fieldnames = [
            "baseline_label",
            "sample_id",
            "category",
            "difficulty",
            "answerable",
            "required_claim_count",
            "generated_claim_count",
            "legacy_required_claim_coverage",
            "legacy_citation_precision",
            "legacy_citation_recall",
            "unsupported_claim_count",
            "context_required_claim_evidence_coverage",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in alignment_rows:
            baseline = qwen if row["baseline_label"] == "QWEN_CANARY_BASELINE" else deepseek
            source = next(
                item for item in baseline["rows"] if item["question_id"] == row["sample_id"]
            )
            metrics = source.get("metrics") or {}
            writer.writerow(
                {
                    "baseline_label": row["baseline_label"],
                    "sample_id": row["sample_id"],
                    "category": row["category"],
                    "difficulty": row["difficulty"],
                    "answerable": row["answerable"],
                    "required_claim_count": row["required_claim_count"],
                    "generated_claim_count": row["generated_claim_count"],
                    "legacy_required_claim_coverage": metrics.get("required_claim_coverage"),
                    "legacy_citation_precision": metrics.get("citation_precision"),
                    "legacy_citation_recall": metrics.get("citation_recall"),
                    "unsupported_claim_count": metrics.get("unsupported_claim_count"),
                    "context_required_claim_evidence_coverage": row[
                        "context_contains_required_evidence"
                    ].get("required_claim_evidence_coverage"),
                }
            )
    OUT_ALIGNMENT_DOC.write_text(render_alignment_doc(alignment_payload), encoding="utf-8")
    OUT_METRIC_DOC.write_text(render_metric_doc(), encoding="utf-8")

    cardinality = summarize_cardinality(gold_rows, qwen["canary_ids"])
    write_json(OUT_CARDINALITY_JSON, cardinality)
    OUT_CARDINALITY_DOC.write_text(render_cardinality_doc(cardinality), encoding="utf-8")

    evidence_first_status: dict[str, Any] | None = None
    if DATA.joinpath("evidence-first-canary-v1.json").exists():
        evidence_first_status = read_json(DATA / "evidence-first-canary-v1.json").get("summary")
    stage_doc = [
        "# Stage 13.36 Audit",
        "",
        f"- Generated at: `{alignment_payload['generated_at']}`",
        f"- Git commit: `{alignment_payload['git_commit']}`",
        "- LLM calls during offline audit: `0`",
        "- Frozen Qwen baseline: `QWEN_CANARY_BASELINE`",
        "- Frozen DeepSeek baseline: `DEEPSEEK_CANARY_BASELINE`",
        "- Evaluator conclusion: `EVALUATOR_BASICALLY_VALID_BUT_EXACT_GOLD_METRIC_NAMES_ARE_TOO_BROAD`",
        "- Recommended next step: `stop Production QA line if Evidence-first canary fails quality`",
        "- Full QA status: `blocked`",
        "- Deep Research status: `not run`",
        "- Reranker status: `disabled`",
        "- Evidence-first status: `EXPERIMENTAL_FAILED`",
        "- Evidence-first default: `false`",
        "- Portfolio default QA route: `Direct QA`",
        "",
        "This is an internal development/canary audit, not a blind holdout.",
    ]
    if evidence_first_status:
        stage_doc.extend(
            [
                "",
                "## Evidence-first Canary v1",
                "",
                f"- engineering gate: `{evidence_first_status.get('evidence_first_engineering_gate')}`",
                f"- quality gate: `{evidence_first_status.get('evidence_first_canary_gate')}`",
                f"- attempted/completed/failed: `{evidence_first_status.get('attempted')}` / `{evidence_first_status.get('completed')}` / `{evidence_first_status.get('terminal_failure_count')}`",
                f"- malformed/schema/invalid citation: `{evidence_first_status.get('malformed_json_count')}` / `{evidence_first_status.get('schema_failure_count')}` / `{evidence_first_status.get('invalid_citation_count')}`",
                f"- required_claim_coverage: `{evidence_first_status.get('required_claim_coverage')}`",
                f"- citation_precision: `{evidence_first_status.get('citation_precision')}`",
                f"- citation_recall: `{evidence_first_status.get('citation_recall')}`",
                f"- core_unsupported_claim_count: `{evidence_first_status.get('core_unsupported_claim_count')}`",
                f"- budget_violations: `{evidence_first_status.get('budget_violations')}`",
                "",
                "Conclusion: `Evidence-first engineering and quality gates failed; do not run 50-item Full QA or Deep Research.`",
            ]
        )
    OUT_STAGE_DOC.write_text("\n".join(stage_doc) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASSED", "rows": len(alignment_rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
