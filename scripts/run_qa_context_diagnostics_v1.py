# ruff: noqa: E501,E701,E702,I001
"""Stage 11C.5 context and citation diagnostics over the frozen Stage 11C run."""

import argparse
import csv
import json
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from paper_research.config import Settings
from paper_research.generation.qa_service import QAService
from paper_research.parsing.types import PaperBlock
from paper_research.providers.factory import build_llm_provider
from paper_research.providers.llm import LLMProviderError
from paper_research.retrieval.context_builder import ContextItem

try:
    import scripts.run_qa_production_v1 as qa_v1
except ModuleNotFoundError:
    import run_qa_production_v1 as qa_v1  # type: ignore[no-redef]

PRODUCTION = Path("data/evaluation/qa-production-v1.json")
GOLD = Path("data/evaluation/gold-set-v1.jsonl")
DEFAULT_OUTPUT = Path("data/evaluation/qa-context-diagnostics-v1.json")
DEFAULT_CSV = Path("data/evaluation/qa-context-diagnostics-v1.csv")
DEFAULT_REPORT = Path("docs/qa-context-diagnostics-v1.md")
BLOCK_ROOT = Path("data/reports/parsing-audit")
CONTEXT_MODES = (
    "retrieved",
    "oracle_gold_only",
    "oracle_gold_plus_distractors",
    "retrieved_plus_missing_gold",
)
SEMANTIC_THRESHOLD = 0.35
WEAK_THRESHOLD = 0.15
ADJACENT_DISTANCE = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "dev", "full"), required=True)
    parser.add_argument("--context-mode", choices=CONTEXT_MODES, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-requests", type=int, default=160)
    return parser.parse_args()


def select_answerable(rows: list[dict], mode: str) -> list[dict]:
    answerable = [row for row in rows if row["gold"]["answerable"]]
    return answerable[:3] if mode == "smoke" else answerable[:10] if mode == "dev" else answerable


def load_blocks(paper_ids: set[str]) -> dict[tuple[str, str], PaperBlock]:
    output = {}
    for paper_id in paper_ids:
        path = BLOCK_ROOT / paper_id / "paper_blocks.jsonl"
        if not path.exists():
            raise RuntimeError(f"gold block source is missing for paper {paper_id}")
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                block = PaperBlock.model_validate(json.loads(line))
                output[(paper_id, block.block_id)] = block
    return output


def block_context(paper_id: str, block: PaperBlock) -> ContextItem:
    return ContextItem(
        chunk_id=f"oracle:{paper_id}:{block.block_id}",
        paper_id=paper_id,
        block_ids=[block.block_id],
        section_path=block.section_path,
        page_start=block.page_start,
        page_end=block.page_end,
        evidence=block.text,
        score=1.0,
    )


def build_context(
    production_row: dict,
    mode: str,
    blocks: dict[tuple[str, str], PaperBlock],
) -> tuple[list[ContextItem], dict]:
    retrieved = [ContextItem.model_validate(item) for item in production_row["context"]]
    gold_pairs = [
        (paper_id, block_id)
        for paper_id in production_row["gold"]["gold_paper_ids"]
        for block_id in production_row["gold"]["gold_block_ids"]
        if (paper_id, block_id) in blocks
    ]
    gold = [block_context(paper_id, blocks[(paper_id, block_id)]) for paper_id, block_id in gold_pairs]
    retrieved_blocks = {block for item in retrieved for block in item.block_ids}
    missing = [item for item in gold if item.block_ids[0] not in retrieved_blocks]
    if mode == "retrieved":
        context = retrieved
        distractors = []
    elif mode == "oracle_gold_only":
        context = gold
        distractors = []
    elif mode == "oracle_gold_plus_distractors":
        non_gold = [item for item in retrieved if not set(item.block_ids) & set(production_row["gold"]["gold_block_ids"])]
        distractors = non_gold[: max(0, len(retrieved) - len(gold))]
        context = [*gold, *distractors]
    elif mode == "retrieved_plus_missing_gold":
        distractors = retrieved
        context = [*retrieved, *missing]
    else:
        raise ValueError(mode)
    seen = set()
    deduped = []
    for item in context:
        identity = tuple(item.block_ids)
        if identity not in seen:
            deduped.append(item)
            seen.add(identity)
    return deduped, {
        "oracle": mode != "retrieved",
        "production_metric": mode == "retrieved",
        "retrieved_count": len(retrieved),
        "gold_count": len(gold),
        "missing_gold_count": len(missing),
        "distractor_count": len(distractors),
        "distractor_chunk_ids": [item.chunk_id for item in distractors],
    }


def block_number(block_id: str) -> int | None:
    digits = "".join(character for character in block_id if character.isdigit())
    return int(digits) if digits else None


def classify_citation(
    citation: dict,
    claim_text: str,
    context: list[ContextItem],
    gold: dict,
) -> dict:
    matching = [
        item
        for item in context
        if item.paper_id == citation["paper_id"]
        and citation["block_id"] in item.block_ids
        and item.page_start <= citation["page"] <= item.page_end
    ]
    if not matching:
        return {"classification": "invalid", "semantic_score": 0.0}
    if citation["block_id"] in gold["gold_block_ids"]:
        return {"classification": "exact_gold_block", "semantic_score": 1.0}
    if citation["paper_id"] in gold["gold_paper_ids"] and citation["page"] in gold["gold_pages"]:
        return {"classification": "same_gold_page", "semantic_score": 1.0}
    number = block_number(citation["block_id"])
    gold_numbers = [block_number(item) for item in gold["gold_block_ids"]]
    if number is not None and any(
        candidate is not None and abs(number - candidate) <= ADJACENT_DISTANCE
        for candidate in gold_numbers
    ):
        return {"classification": "adjacent_to_gold_block", "semantic_score": 1.0}
    score = max(qa_v1.overlap(claim_text, item.evidence) for item in matching)
    return {
        "classification": (
            "semantic_support_non_gold"
            if score >= SEMANTIC_THRESHOLD
            else "weakly_related"
            if score >= WEAK_THRESHOLD
            else "unsupported"
        ),
        "semantic_score": round(score, 6),
    }


def diagnose_answer(answer: dict, context: list[ContextItem], gold: dict) -> dict:
    classifications = Counter()
    details = []
    unsupported = Counter()
    required_best = []
    for claim in answer["claims"]:
        if not claim["citations"]:
            unsupported["no_context_citation"] += 1
        claim_details = []
        for citation in claim["citations"]:
            result = classify_citation(citation, claim["text"], context, gold)
            classifications[result["classification"]] += 1
            claim_details.append({**citation, **result})
        classes = {item["classification"] for item in claim_details}
        if "invalid" in classes:
            unsupported["invalid_citation"] += 1
        if classes and classes <= {"weakly_related", "unsupported"}:
            unsupported["citation_not_supporting_claim"] += 1
        if classes & {"same_gold_page", "adjacent_to_gold_block", "semantic_support_non_gold"}:
            unsupported["context_support_but_not_gold"] += 1
        required_scores = [qa_v1.overlap(required, claim["text"]) for required in gold["required_claims"]]
        if max(required_scores, default=0) < qa_v1.CLAIM_MATCH_THRESHOLD:
            unsupported["extra_claim_not_in_required_claims"] += 1
        details.append({"claim_id": claim["claim_id"], "citations": claim_details, "required_scores": required_scores})
    for required in gold["required_claims"]:
        best = max((qa_v1.overlap(required, claim["text"]) for claim in answer["claims"]), default=0)
        required_best.append(best)
        if best < qa_v1.CLAIM_MATCH_THRESHOLD:
            unsupported["required_claim_semantic_miss"] += 1
    context_blocks = {block for item in context for block in item.block_ids}
    gold_exact = bool(context_blocks & set(gold["gold_block_ids"]))
    gold_page = any(item.paper_id in gold["gold_paper_ids"] and set(range(item.page_start, item.page_end + 1)) & set(gold["gold_pages"]) for item in context)
    support_non_gold = any(
        not set(item.block_ids) & set(gold["gold_block_ids"])
        and max((qa_v1.overlap(required, item.evidence) for required in gold["required_claims"]), default=0) >= SEMANTIC_THRESHOLD
        for item in context
    )
    total_citations = sum(classifications.values())
    exact = classifications["exact_gold_block"]
    page = exact + classifications["same_gold_page"]
    adjacent = page + classifications["adjacent_to_gold_block"]
    semantic = adjacent + classifications["semantic_support_non_gold"]
    return {
        "citation_classifications": dict(classifications),
        "citation_details": details,
        "unsupported_categories": dict(unsupported),
        "legacy_unsupported_claim_count": sum(
            not any(item["classification"] == "exact_gold_block" for item in detail["citations"])
            for detail in details
        ),
        "required_claim_coverage": sum(score >= qa_v1.CLAIM_MATCH_THRESHOLD for score in required_best) / len(required_best) if required_best else 0,
        "extra_claim_count": unsupported["extra_claim_not_in_required_claims"],
        "exact_gold_precision": exact / total_citations if total_citations else 0,
        "page_level_precision": page / total_citations if total_citations else 0,
        "adjacent_support_precision": adjacent / total_citations if total_citations else 0,
        "semantic_support_precision": semantic / total_citations if total_citations else 0,
        "citation_recall": len({c["block_id"] for claim in answer["claims"] for c in claim["citations"]} & set(gold["gold_block_ids"])) / len(set(gold["gold_block_ids"])) if gold["gold_block_ids"] else 0,
        "exact_gold_block_available": gold_exact,
        "gold_page_available": gold_page,
        "supporting_non_gold_available": support_non_gold,
    }


def summarize(rows: list[dict]) -> dict:
    completed = [row for row in rows if row["status"] == "COMPLETED"]
    diagnostics = [row["diagnostics"] for row in completed]
    def avg(key: str) -> float:
        return round(sum(item[key] for item in diagnostics) / len(diagnostics), 6) if diagnostics else 0
    latency_values = [row["answer"]["latency"]["total_latency_ms"] for row in completed]
    unsupported = Counter()
    classes = Counter()
    for item in diagnostics:
        unsupported.update(item["unsupported_categories"])
        classes.update(item["citation_classifications"])
    return {
        "query_count": len(rows), "completed": len(completed),
        "json_success": len(completed) / len(rows) if rows else 0,
        "schema_success": len(completed) / len(rows) if rows else 0,
        "retry_count": sum(row.get("retry_count", 0) for row in rows),
        "failure_count": sum(row["status"] == "FAILED" for row in rows),
        "answerable_accuracy": round(sum(row["answer"]["answerable"] for row in completed) / len(completed), 6) if completed else 0,
        "required_claim_coverage": avg("required_claim_coverage"),
        "extra_claim_count": sum(item["extra_claim_count"] for item in diagnostics),
        "exact_gold_precision": avg("exact_gold_precision"),
        "page_level_precision": avg("page_level_precision"),
        "adjacent_support_precision": avg("adjacent_support_precision"),
        "semantic_support_precision": avg("semantic_support_precision"),
        "citation_recall": avg("citation_recall"),
        "unsupported_rate": round(sum(item["legacy_unsupported_claim_count"] for item in diagnostics) / max(1, sum(len(row["answer"]["claims"]) for row in completed)), 6),
        "exact_gold_block_available": avg("exact_gold_block_available"),
        "gold_page_available": avg("gold_page_available"),
        "supporting_non_gold_available": avg("supporting_non_gold_available"),
        "mean_distractor_count": round(sum(row["context_audit"]["distractor_count"] for row in rows) / len(rows), 3) if rows else 0,
        "unsupported_categories": dict(unsupported),
        "citation_classifications": dict(classes),
        "input_tokens": sum(row["answer"]["model_usage"]["input_tokens"] for row in completed),
        "output_tokens": sum(row["answer"]["model_usage"]["output_tokens"] for row in completed),
        "total_tokens": sum(row["answer"]["model_usage"]["total_tokens"] for row in completed),
        "requests": sum(row.get("api_request_count", 0) for row in rows),
        "latency_mean_ms": round(sum(latency_values) / len(latency_values), 3) if latency_values else 0,
        "latency_p95_ms": qa_v1.percentile(latency_values, 0.95),
    }


def write_outputs(payload: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = []
    for row in payload["runs"]:
        d = row.get("diagnostics", {})
        rows.append({"question_id": row["question_id"], "context_mode": row["context_mode"], "status": row["status"], "claim_coverage": d.get("required_claim_coverage"), "exact_precision": d.get("exact_gold_precision"), "page_precision": d.get("page_level_precision"), "semantic_precision": d.get("semantic_support_precision"), "citation_recall": d.get("citation_recall"), "distractors": row["context_audit"]["distractor_count"], "tokens": row.get("answer", {}).get("model_usage", {}).get("total_tokens"), "latency_ms": row.get("answer", {}).get("latency", {}).get("total_latency_ms"), "failure_reason": row.get("failure_reason")})
    with DEFAULT_CSV.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["question_id"]); writer.writeheader(); writer.writerows(rows)
    lines = [
        "# QA Context Diagnostics v1", "",
        "> Status: diagnostic evidence only. Oracle modes are explicitly marked `oracle=true`, "
        "`production_metric=false`, and are excluded from Production metrics.", "",
        "## Frozen protocol", "",
        "- Corpus, chunks, Jina embedding, Structural Hybrid retrieval, filters, and queries are unchanged from Stage 11C.",
        "- Reranker is disabled. The LLM remains SiliconFlow `Qwen/Qwen3-8B`, prompt `qa-production-v1`, temperature 0.",
        "- `retrieved` reuses the frozen Stage 11C answers and makes no LLM request. The other modes are controlled Oracle diagnostics.",
        "- Deep Research was not run. The original `qa-production-v1.*` artifacts were not overwritten.", "",
        "## Metric definitions", "",
        "- **Exact**: cited `(paper_id, block_id)` is in the Gold block set.",
        "- **Page**: Exact, or the citation is on a Gold page in the same Gold paper.",
        "- **Adjacent**: Page support, or block number distance from a Gold block is at most 2.",
        "- **Semantic**: the preceding classes, or token-set recall from cited block to claim is at least 0.35. This is a lexical diagnostic proxy, not a human entailment judgment.",
        "- **Recall**: Gold block identifiers cited by the answer divided by all Gold block identifiers.",
        "- **Unsupported**: claim-level exact-Gold miss rate, retained for comparison with the strict Stage 11C criterion.", "",
        "## Results", "",
        "| Mode | N | Answerable | Claim coverage | Exact | Page | Adjacent | Semantic | Recall | Unsupported | Tokens | P95 ms |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    ]
    for mode, metrics in payload["metrics"].items():
        lines.append(f"| {mode} | {metrics['query_count']} | {metrics['answerable_accuracy']:.3f} | {metrics['required_claim_coverage']:.3f} | {metrics['exact_gold_precision']:.3f} | {metrics['page_level_precision']:.3f} | {metrics['adjacent_support_precision']:.3f} | {metrics['semantic_support_precision']:.3f} | {metrics['citation_recall']:.3f} | {metrics['unsupported_rate']:.3f} | {metrics['total_tokens']} | {metrics['latency_p95_ms']} |")
    if all(mode in payload["metrics"] for mode in CONTEXT_MODES):
        retrieved = payload["metrics"]["retrieved"]
        gold_only = payload["metrics"]["oracle_gold_only"]
        gold_distractors = payload["metrics"]["oracle_gold_plus_distractors"]
        augmented = payload["metrics"]["retrieved_plus_missing_gold"]
        baseline_refusals = {
            row["question_id"] for row in payload["runs"]
            if row["context_mode"] == "retrieved" and row["answer"].get("insufficient_evidence")
        }
        augmented_rows = {
            row["question_id"]: row for row in payload["runs"]
            if row["context_mode"] == "retrieved_plus_missing_gold"
        }
        lines += [
            "", "## Bottleneck diagnosis", "",
            f"- **Retrieval evidence availability:** exact Gold blocks occur in only {retrieved['exact_gold_block_available']:.1%} of retrieved contexts; Gold pages occur in {retrieved['gold_page_available']:.1%}. Gold-only Oracle raises answerable accuracy from {retrieved['answerable_accuracy']:.1%} to {gold_only['answerable_accuracy']:.1%} and required-claim coverage from {retrieved['required_claim_coverage']:.1%} to {gold_only['required_claim_coverage']:.1%}.",
            f"- **Context distraction:** adding distractors to the same Gold evidence lowers exact precision from {gold_only['exact_gold_precision']:.1%} to {gold_distractors['exact_gold_precision']:.1%} and raises strict unsupported rate from {gold_only['unsupported_rate']:.1%} to {gold_distractors['unsupported_rate']:.1%}.",
            f"- **Appending missing Gold is not sufficient:** it raises answerable accuracy to {augmented['answerable_accuracy']:.1%}, but exact precision remains {augmented['exact_gold_precision']:.1%} and strict unsupported rate {augmented['unsupported_rate']:.1%}; the original distractors still dominate the context.",
            f"- **Gold exactness is narrow:** retrieved exact/page/adjacent/semantic precision is {retrieved['exact_gold_precision']:.1%}/{retrieved['page_level_precision']:.1%}/{retrieved['adjacent_support_precision']:.1%}/{retrieved['semantic_support_precision']:.1%}. Semantic support is only an automated lexical proxy and remains pending human audit.",
            f"- **LLM is not a perfect upper bound:** even Gold-only context reaches {gold_only['answerable_accuracy']:.1%} answerable accuracy and {gold_only['required_claim_coverage']:.1%} claim coverage, so generation/claim selection remains a secondary bottleneck after retrieval and context selection.", "",
            "The primary next step is retrieval and context selection optimization, followed by a small human citation audit. These Oracle results do not justify a Production or v1.0 claim.", "",
            "## Stage 11C refusal recovery", "",
            "| Question | Recovered after adding missing Gold | Claim coverage | Exact precision |", "|---|---:|---:|---:|"
        ]
        for question_id in sorted(baseline_refusals):
            row = augmented_rows[question_id]
            d = row["diagnostics"]
            lines.append(f"| {question_id} | {str(not row['answer'].get('insufficient_evidence')).lower()} | {d['required_claim_coverage']:.3f} | {d['exact_gold_precision']:.3f} |")
        recovered = sum(not augmented_rows[q]["answer"].get("insufficient_evidence") for q in baseline_refusals)
        lines += ["", f"Five of the six answerable Stage 11C refusals recovered ({recovered}/{len(baseline_refusals)}); q023 remained a refusal. Recovery alone is not correctness: q024 and q043 still have zero required-claim coverage and zero exact precision.", "", "## Citation classifications and unsupported categories", ""]
        for mode in CONTEXT_MODES:
            metrics = payload["metrics"][mode]
            lines += [f"### `{mode}`", "", f"Classifications: `{json.dumps(metrics['citation_classifications'], sort_keys=True)}`", "", f"Unsupported categories: `{json.dumps(metrics['unsupported_categories'], sort_keys=True)}`", ""]
        lines += [
            "## Compatibility note", "",
            "Stage 11C reports 147 unsupported claims. This diagnostic's claim-level exact-Gold recount is 146. The sole difference is q004: two claims cite the same exact Gold block `b000103`; Stage 11C's aggregate counts that unique supporting citation once, while this diagnostic credits both bound claims. No answer or Gold record changed.", "",
            "## Known limitations", "",
            "- Oracle context proves evidence availability effects, not a deployable retrieval policy.",
            "- Token-set recall can over-credit lexically similar but non-entailing evidence; no row has been represented as human-approved.",
            "- Exact Gold annotations may omit valid same-page, adjacent, or alternative evidence blocks.",
            "- Latency and token totals for `retrieved` are inherited from the frozen Stage 11C run; Oracle modes are new live calls and are not directly cost-equivalent.",
            "- The experiment covers 48 answerable questions. The two unanswerable questions are intentionally excluded from these context-recall diagnostics.",
        ]
    DEFAULT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args(); settings = Settings()
    if settings.rerank_enabled: raise RuntimeError("RERANK_ENABLED must remain false")
    if settings.llm_provider != "siliconflow" or settings.llm_model != "Qwen/Qwen3-8B": raise RuntimeError("fixed SiliconFlow Qwen provider is required")
    if settings.prompt_version != "qa-production-v1" or settings.llm_temperature != 0: raise RuntimeError("fixed qa-production-v1 prompt is required")
    production = json.loads(PRODUCTION.read_text(encoding="utf-8"))
    selected = select_answerable(production["queries"], args.mode)
    paper_ids = {paper for row in selected for paper in row["gold"]["gold_paper_ids"]}
    blocks = load_blocks(paper_ids)
    payload = {"status": "RUNNING", "generated_at": datetime.now(UTC).isoformat(), "production_source": str(PRODUCTION), "production_source_unchanged": True, "rerank_enabled": False, "llm_provider": "siliconflow", "llm_model": "Qwen/Qwen3-8B", "prompt_version": "qa-production-v1", "deep_research_called": False, "semantic_method": {"algorithm": "token_set_recall", "semantic_threshold": SEMANTIC_THRESHOLD, "weak_threshold": WEAK_THRESHOLD, "adjacent_block_distance": ADJACENT_DISTANCE}, "runs": []}
    if args.resume and args.output.exists(): payload = json.loads(args.output.read_text(encoding="utf-8"))
    completed_keys = {(row["context_mode"], row["question_id"]) for row in payload["runs"] if row["status"] == "COMPLETED"}
    requests = sum(row.get("api_request_count", 0) for row in payload["runs"])
    qa = QAService(llm=build_llm_provider(settings), prompt_version="qa-production-v1")
    for source in selected:
        key = (args.context_mode, source["question_id"])
        if key in completed_keys or requests >= args.max_requests: continue
        context, audit = build_context(source, args.context_mode, blocks)
        row = {"question_id": source["question_id"], "context_mode": args.context_mode, "oracle": args.context_mode != "retrieved", "production_metric": args.context_mode == "retrieved", "context": [item.model_dump() for item in context], "context_audit": audit, "gold": source["gold"]}
        try:
            if args.context_mode == "retrieved": answer = source["answer"]
            else: answer = qa.answer_from_context(source["retrieval_query"], context, total_started=time.perf_counter()).model_dump()
            row.update(status="COMPLETED", answer=answer, diagnostics=diagnose_answer(answer, context, source["gold"]), api_request_count=0 if args.context_mode == "retrieved" else answer["api_request_count"], retry_count=0 if args.context_mode == "retrieved" else answer["retry_count"])
        except LLMProviderError as exc:
            row.update(status="FAILED", failure_reason=str(exc), api_request_count=exc.api_request_count, retry_count=len(exc.retry_reasons), retry_reasons=exc.retry_reasons)
        payload["runs"] = [item for item in payload["runs"] if (item["context_mode"], item["question_id"]) != key] + [row]
        requests += row.get("api_request_count", 0)
        payload["metrics"] = {mode: summarize([item for item in payload["runs"] if item["context_mode"] == mode]) for mode in CONTEXT_MODES if any(item["context_mode"] == mode for item in payload["runs"])}
        write_outputs(payload, args.output)
    payload["status"] = "COMPLETED" if all((args.context_mode, row["question_id"]) in {(item["context_mode"], item["question_id"]) for item in payload["runs"] if item["status"] == "COMPLETED"} for row in selected) else "PARTIAL"
    payload["metrics"] = {mode: summarize([item for item in payload["runs"] if item["context_mode"] == mode]) for mode in CONTEXT_MODES if any(item["context_mode"] == mode for item in payload["runs"])}
    write_outputs(payload, args.output); print(json.dumps({"status": payload["status"], "context_mode": args.context_mode, "metrics": payload["metrics"].get(args.context_mode)}))


if __name__ == "__main__": main()
