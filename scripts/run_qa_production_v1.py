# ruff: noqa: E501
"""Run Stage 11C evidence-bound QA evaluation with fixed Stage 11A.5 retrieval."""

import argparse
import csv
import json
import math
import time
from datetime import UTC, datetime
from pathlib import Path

from qdrant_client import QdrantClient

from paper_research.config import Settings
from paper_research.generation.qa_service import QAService
from paper_research.indexing.registry import IndexRegistry
from paper_research.indexing.vector_store import QdrantVectorStore
from paper_research.providers.factory import build_embedding_provider, build_llm_provider
from paper_research.providers.llm import LLMProviderError
from paper_research.retrieval.context_builder import ContextBuilder, ContextItem
from paper_research.retrieval.dense import DenseRetriever
from paper_research.retrieval.filters import RetrievalFilter
from paper_research.retrieval.hybrid import HybridRetriever
from paper_research.retrieval.reranker import DisabledReranker
from paper_research.retrieval.sparse import BM25Retriever

try:
    import scripts.run_retrieval_ablation_v2 as v2
except ModuleNotFoundError:
    import run_retrieval_ablation_v2 as v2  # type: ignore[no-redef]

PROTOCOL = Path("data/evaluation/retrieval-gold-v2.jsonl")
GOLD = Path("data/evaluation/gold-set-v1.jsonl")
CORPUS = Path("data/evaluation/production-corpus-v1.json")
INDEX_MANIFEST = Path("data/evaluation/retrieval-index-v2.json")
DEFAULT_OUTPUT = Path("data/evaluation/qa-production-v1.json")
DEFAULT_CSV = Path("data/evaluation/qa-production-v1.csv")
DEFAULT_REPORT = Path("docs/qa-production-v1.md")
PROMPT_VERSION = "qa-production-v1"
RECALL_K = 20
TOP_K = 10
CLAIM_MATCH_THRESHOLD = 0.35
SMOKE_IDS = ["q001", "q005", "q030"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "dev", "full"), required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-cost", type=float, default=None)
    parser.add_argument("--max-requests", type=int, default=160)
    return parser.parse_args()


def select_records(records: list[dict], mode: str) -> list[dict]:
    by_id = {record["question_id"]: record for record in records}
    if mode == "smoke":
        return [by_id[identifier] for identifier in SMOKE_IDS]
    if mode == "dev":
        identifiers = [record["question_id"] for record in records[:9]] + ["q005", "q030"]
        return [by_id[identifier] for identifier in dict.fromkeys(identifiers)]
    return records


def order_rows(selected: list[dict], rows: list[dict]) -> list[dict]:
    by_id = {row["question_id"]: row for row in rows}
    return [by_id[item["question_id"]] for item in selected if item["question_id"] in by_id]


def terms(text: str) -> set[str]:
    return {
        token.lower().strip(".,:;!?()[]{}'\"")
        for token in text.split()
        if token.strip(".,:;!?()[]{}'\"")
    }


def overlap(expected: str, actual: str) -> float:
    expected_terms = terms(expected)
    return len(expected_terms & terms(actual)) / max(1, len(expected_terms))


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[max(0, math.ceil(len(ordered) * fraction) - 1)], 3)


def latency(values: list[float]) -> dict:
    return {
        "mean_ms": round(sum(values) / len(values), 3) if values else None,
        "p50_ms": percentile(values, 0.5),
        "p95_ms": percentile(values, 0.95),
    }


def evaluate_answer(row: dict, gold: dict, context: list[ContextItem]) -> dict:
    answer = row["answer"]
    context_citations = {
        (item.paper_id, page, block_id)
        for item in context
        for page in range(item.page_start, item.page_end + 1)
        for block_id in (item.block_ids or [item.chunk_id])
    }
    citations = [citation for claim in answer["claims"] for citation in claim["citations"]]
    gold_blocks = set(gold["gold_block_ids"])
    gold_pages = set(gold["gold_pages"])
    gold_papers = set(gold["gold_paper_ids"])
    valid = [
        (item["paper_id"], item["page"], item["block_id"]) in context_citations
        for item in citations
    ]
    precise = [
        item["paper_id"] in gold_papers
        and item["block_id"] in gold_blocks
        and item["page"] in gold_pages
        for item in citations
    ]
    claim_scores = []
    for required in gold["required_claims"]:
        scores = [overlap(required, claim["text"]) for claim in answer["claims"]]
        claim_scores.append({"required_claim": required, "scores": scores, "best": max(scores, default=0)})
    unsupported = sum(
        not any(
            citation["paper_id"] in gold_papers
            and citation["block_id"] in gold_blocks
            and citation["page"] in gold_pages
            for citation in claim["citations"]
        )
        for claim in answer["claims"]
    )
    cited_gold_blocks = {item["block_id"] for item in citations} & gold_blocks
    return {
        "answerable_correct": answer["answerable"] == gold["answerable"],
        "required_claim_scores": claim_scores,
        "required_claim_coverage": (
            sum(item["best"] >= CLAIM_MATCH_THRESHOLD for item in claim_scores) / len(claim_scores)
            if claim_scores
            else None
        ),
        "unsupported_claim_count": unsupported,
        "citation_presence": (
            all(claim["citations"] for claim in answer["claims"])
            if answer["answerable"]
            else not citations
        ),
        "citation_id_validity": sum(valid) / len(valid) if valid else (1.0 if not answer["answerable"] else 0.0),
        "citation_precision": sum(precise) / len(precise) if precise else (1.0 if not gold["answerable"] else 0.0),
        "citation_recall": len(cited_gold_blocks) / len(gold_blocks) if gold_blocks else (1.0 if not citations else 0.0),
        "claim_citation_binding_rate": (
            sum(bool(claim["citations"]) for claim in answer["claims"]) / len(answer["claims"])
            if answer["claims"]
            else (1.0 if not answer["answerable"] else 0.0)
        ),
        "cited_block_in_context_rate": sum(valid) / len(valid) if valid else 1.0,
    }


def summarize(rows: list[dict]) -> dict:
    completed = [row for row in rows if row["status"] == "COMPLETED"]
    answerable = [row for row in completed if row["gold"]["answerable"]]
    unanswerable = [row for row in completed if not row["gold"]["answerable"]]
    metric_rows = [row["metrics"] for row in completed]
    answerable_metrics = [row["metrics"] for row in answerable]

    def average(name: str, source: list[dict] = metric_rows) -> float | None:
        values = [row[name] for row in source if row.get(name) is not None]
        return round(sum(values) / len(values), 6) if values else None

    total_latency = [row["answer"]["latency"]["total_latency_ms"] for row in completed]
    first_latency = [
        row["answer"]["latency"]["llm_first_token_latency_ms"]
        for row in completed
        if row["answer"]["latency"]["llm_first_token_latency_ms"] is not None
    ]
    usage = [row["answer"]["model_usage"] for row in completed]
    cost_is_configured = bool(usage) and all(
        item["estimated_cost_usd"] is not None for item in usage
    )
    return {
        "attempted": len(rows),
        "completed": len(completed),
        "json_parse_success_rate": round(len(completed) / len(rows), 6) if rows else 0,
        "schema_validation_success_rate": round(len(completed) / len(rows), 6) if rows else 0,
        "retry_count": sum(row.get("retry_count", 0) for row in rows),
        "failure_count": sum(row["status"] == "FAILED" for row in rows),
        "answerable_accuracy": (
            round(sum(row["answer"]["answerable"] for row in answerable) / len(answerable), 6)
            if answerable
            else None
        ),
        "refusal_accuracy": (
            round(sum(not row["answer"]["answerable"] for row in unanswerable) / len(unanswerable), 6)
            if unanswerable
            else None
        ),
        "required_claim_coverage": average("required_claim_coverage", answerable_metrics),
        "unsupported_claim_count": sum(row["unsupported_claim_count"] for row in metric_rows),
        "citation_presence_rate": average("citation_presence", answerable_metrics),
        "citation_id_validity": average("citation_id_validity", answerable_metrics),
        "citation_precision": average("citation_precision", answerable_metrics),
        "citation_recall": average("citation_recall", answerable_metrics),
        "claim_citation_binding_rate": average(
            "claim_citation_binding_rate", answerable_metrics
        ),
        "cited_block_in_retrieved_context_rate": average(
            "cited_block_in_context_rate", answerable_metrics
        ),
        "gold_block_retrieved_rate": (
            round(sum(row["gold_block_present"] for row in answerable) / len(answerable), 6)
            if answerable
            else None
        ),
        "answer_failure_gold_present": sum(row["status"] != "COMPLETED" or not row.get("answer", {}).get("answerable", False) for row in rows if row["gold_block_present"] and row["gold"]["answerable"]),
        "answer_failure_gold_absent": sum(row["status"] != "COMPLETED" or not row.get("answer", {}).get("answerable", False) for row in rows if not row["gold_block_present"] and row["gold"]["answerable"]),
        "total_latency": latency(total_latency),
        "first_token_latency": latency(first_latency),
        "input_tokens": sum(item["input_tokens"] for item in usage),
        "output_tokens": sum(item["output_tokens"] for item in usage),
        "total_tokens": sum(item["total_tokens"] for item in usage),
        "estimated_cost_usd": (
            round(sum(item["estimated_cost_usd"] for item in usage), 8)
            if cost_is_configured
            else None
        ),
        "cost_is_configured": cost_is_configured,
        "api_requests": sum(row.get("api_request_count", 0) for row in rows),
        "rate_limit_events": sum(row.get("rate_limit_events", 0) for row in rows),
    }


def write_outputs(payload: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output)
    csv_path = DEFAULT_CSV if output == DEFAULT_OUTPUT else output.with_suffix(".csv")
    report_path = DEFAULT_REPORT if output == DEFAULT_OUTPUT else output.with_suffix(".md")
    rows = []
    for row in payload["queries"]:
        rows.append(
            {
                "question_id": row["question_id"],
                "status": row["status"],
                "gold_answerable": row["gold"]["answerable"],
                "predicted_answerable": row.get("answer", {}).get("answerable"),
                "gold_block_present": row["gold_block_present"],
                "required_claim_coverage": row.get("metrics", {}).get("required_claim_coverage"),
                "citation_precision": row.get("metrics", {}).get("citation_precision"),
                "citation_recall": row.get("metrics", {}).get("citation_recall"),
                "unsupported_claim_count": row.get("metrics", {}).get("unsupported_claim_count"),
                "total_latency_ms": row.get("answer", {}).get("latency", {}).get("total_latency_ms"),
                "input_tokens": row.get("answer", {}).get("model_usage", {}).get("input_tokens"),
                "output_tokens": row.get("answer", {}).get("model_usage", {}).get("output_tokens"),
                "retry_count": row.get("retry_count", 0),
                "failure_reason": row.get("failure_reason"),
            }
        )
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["question_id"])
        writer.writeheader()
        writer.writerows(rows)
    m = payload["metrics"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "\n".join(
            [
                "# Production QA Evaluation v1",
                "",
                f"- Mode: {payload['mode']}",
                "- Model: `Qwen/Qwen3-8B` via SiliconFlow",
                "- Prompt: `qa-production-v1`",
                "- Retrieval: Jina 1024d + Structural Hybrid, Recall 20 / Top 10",
                "- Reranker: disabled",
                "- Deep Research: not run",
                f"- Completed/failures/retries: {m['completed']}/{m['failure_count']}/{m['retry_count']}",
                "",
                "## Metrics",
                "",
                f"- JSON parse success: {m['json_parse_success_rate']}",
                f"- Schema validation success: {m['schema_validation_success_rate']}",
                f"- Answerable accuracy: {m['answerable_accuracy']}",
                f"- Refusal accuracy: {m['refusal_accuracy']}",
                f"- Required claim coverage: {m['required_claim_coverage']}",
                f"- Unsupported claims: {m['unsupported_claim_count']}",
                f"- Citation presence / validity: {m['citation_presence_rate']} / {m['citation_id_validity']}",
                f"- Citation precision / recall: {m['citation_precision']} / {m['citation_recall']}",
                f"- Claim-citation binding: {m['claim_citation_binding_rate']}",
                f"- Gold block retrieved: {m['gold_block_retrieved_rate']}",
                f"- Total latency mean/p50/p95 ms: {m['total_latency']}",
                f"- First-token latency: {m['first_token_latency']}",
                f"- Tokens input/output/total: {m['input_tokens']}/{m['output_tokens']}/{m['total_tokens']}",
                f"- Estimated cost USD: {m['estimated_cost_usd']} (configured={m['cost_is_configured']})",
                f"- API requests / rate limits: {m['api_requests']} / {m['rate_limit_events']}",
                "",
                "Rule-based required-claim matching uses token-set recall with threshold 0.35 and stores every raw score. No LLM judge is used.",
                "",
                "## Smoke / Dev / Full progression",
                "",
                "| Mode | Completed | Failures | Retries | Answerable | Refusal | Claim coverage | Citation precision | Citation recall |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
                *[
                    f"| {mode} | {values['completed']} | {values['failure_count']} | {values['retry_count']} | {values['answerable_accuracy']} | {values['refusal_accuracy']} | {values['required_claim_coverage']} | {values['citation_precision']} | {values['citation_recall']} |"
                    for mode, values in payload["mode_results"].items()
                ],
                "",
                "## Final retry and failure records",
                "",
                *(
                    [
                        f"- `{row['question_id']}`: retries={row.get('retry_count', 0)}, reasons={row.get('retry_reasons', [])}"
                        for row in payload["queries"]
                        if row.get("retry_count", 0)
                    ]
                    or ["- No retry was required in the retained successful rows."]
                ),
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    settings = Settings()
    if settings.app_profile != "production":
        raise RuntimeError("APP_PROFILE=production is required")
    if settings.embedding_provider != "jina" or settings.embedding_dimensions != 1024:
        raise RuntimeError("fixed Jina 1024-dimensional embedding is required")
    if settings.rerank_enabled:
        raise RuntimeError("RERANK_ENABLED must remain false")
    if settings.llm_provider != "siliconflow" or settings.llm_model != "Qwen/Qwen3-8B":
        raise RuntimeError("LLM_PROVIDER=siliconflow and LLM_MODEL=Qwen/Qwen3-8B are required")
    if settings.prompt_version != PROMPT_VERSION or settings.llm_temperature != 0:
        raise RuntimeError("qa-production-v1 with temperature=0 is required")
    protocol = v2.load_jsonl(PROTOCOL)
    gold_by_id = {item["question_id"]: item for item in v2.load_jsonl(GOLD)}
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    manifest = json.loads(INDEX_MANIFEST.read_text(encoding="utf-8"))
    v2.validate_inputs(protocol, corpus, manifest)
    selected = select_records(protocol, args.mode)
    included = [paper for paper in corpus["papers"] if paper["included_in_production"]]
    public_to_raw = {paper["paper_id"]: paper["database_id"] for paper in included}
    raw_to_public = {paper["database_id"]: paper["paper_id"] for paper in included}
    collection = manifest["collections"]["jina"]
    if collection["name"] != "papers_jina_eval34_v2__20260713152149":
        raise RuntimeError("fixed Stage 11A.5 Jina collection changed")
    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key, check_compatibility=False)
    chunks = v2.load_chunks(client, collection["name"])
    if len(chunks) != 2062 or v2.chunk_signature(chunks) != collection["chunk_signature"]:
        raise RuntimeError("live fixed collection does not match the signed manifest")
    store = QdrantVectorStore(
        client,
        IndexRegistry(settings.data_dir / "index_registry.json").resolve(collection["name"]),
        1024,
    )
    retriever = HybridRetriever(
        DenseRetriever(build_embedding_provider(settings), store),
        BM25Retriever(chunks),
        DisabledReranker(),
        ContextBuilder(include_neighbors=False, max_tokens=settings.qa_context_token_budget),
        provider_metadata=settings.provider_metadata,
    )
    qa = QAService(llm=build_llm_provider(settings), prompt_version=PROMPT_VERSION)
    existing = {}
    if args.resume and args.output.exists():
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        existing = {row["question_id"]: row for row in previous.get("queries", []) if row["status"] == "COMPLETED"}
    rows = [existing[item["question_id"]] for item in selected if item["question_id"] in existing]
    for record in selected:
        if record["question_id"] in existing:
            continue
        requests_used = sum(row.get("api_request_count", 0) for row in rows)
        cost_used = sum(row.get("answer", {}).get("model_usage", {}).get("estimated_cost_usd") or 0 for row in rows)
        if requests_used >= args.max_requests or (args.max_cost is not None and cost_used >= args.max_cost):
            break
        gold = gold_by_id[record["question_id"]]
        raw_filter = [public_to_raw[item] for item in record["retrieval_filter"]["paper_ids"]]
        started = time.perf_counter()
        result = retriever.retrieve(
            record["retrieval_query"],
            RetrievalFilter(paper_ids=raw_filter),
            recall_k=RECALL_K,
            top_k=TOP_K,
            retrieval_scope=record["retrieval_scope"],
        )
        context = [item.model_copy(update={"paper_id": raw_to_public[item.paper_id]}) for item in result.context]
        context_blocks = {block for item in context for block in item.block_ids}
        row = {
            "question_id": record["question_id"],
            "retrieval_query": record["retrieval_query"],
            "retrieval_scope": record["retrieval_scope"],
            "retrieval_filter": record["retrieval_filter"],
            "gold": {
                "answerable": gold["answerable"],
                "gold_paper_ids": gold["gold_paper_ids"],
                "gold_block_ids": gold["gold_block_ids"],
                "gold_pages": gold["gold_pages"],
                "required_claims": gold["required_claims"],
            },
            "context": [item.model_dump() for item in context],
            "context_strategy": result.trace.context_strategy.model_dump(),
            "gold_block_present": bool(context_blocks & set(gold["gold_block_ids"])),
            "retrieval_latency_ms": result.trace.retrieval_latency_ms,
        }
        try:
            answer = qa.answer_from_context(
                record["retrieval_query"],
                context,
                retrieval_latency_ms=result.trace.retrieval_latency_ms,
                rerank_latency_ms=0,
                context_build_latency_ms=result.trace.context_build_latency_ms,
                total_started=started,
            )
            row.update(
                status="COMPLETED",
                answer=answer.model_dump(),
                metrics=evaluate_answer({"answer": answer.model_dump()}, gold, context),
                api_request_count=answer.api_request_count,
                retry_count=answer.retry_count,
                retry_reasons=answer.retry_reasons,
                rate_limit_events=answer.rate_limit_events,
            )
        except LLMProviderError as exc:
            row.update(
                status="FAILED",
                failure_reason=str(exc),
                api_request_count=exc.api_request_count,
                retry_count=len(exc.retry_reasons),
                retry_reasons=exc.retry_reasons,
                rate_limit_events=exc.rate_limit_events,
            )
        rows.append(row)
        rows = order_rows(selected, rows)
        payload = build_payload(args.mode, settings, collection, rows)
        write_outputs(payload, args.output)
    rows = order_rows(selected, rows)
    payload = build_payload(args.mode, settings, collection, rows)
    write_outputs(payload, args.output)
    print(json.dumps({"status": payload["status"], "mode": args.mode, "metrics": payload["metrics"]}))


def build_payload(mode: str, settings: Settings, collection: dict, rows: list[dict]) -> dict:
    protocol = v2.load_jsonl(PROTOCOL)
    rows_by_id = {row["question_id"]: row for row in rows}
    mode_results = {}
    for candidate_mode in ("smoke", "dev", "full"):
        identifiers = [item["question_id"] for item in select_records(protocol, candidate_mode)]
        if set(identifiers).issubset(rows_by_id):
            mode_results[candidate_mode] = summarize([rows_by_id[item] for item in identifiers])
    return {
        "status": (
            "COMPLETED_WITH_FAILURES"
            if len(rows) == len(select_records(v2.load_jsonl(PROTOCOL), mode))
            and any(row["status"] == "FAILED" for row in rows)
            else "COMPLETED"
            if len(rows) == len(select_records(v2.load_jsonl(PROTOCOL), mode))
            else "PARTIAL_BUDGET_STOP"
        ),
        "generated_at": datetime.now(UTC).isoformat(),
        "mode": mode,
        "model_configuration": {
            "provider": "siliconflow",
            "model": settings.llm_model,
            "temperature": settings.llm_temperature,
            "max_output_tokens": settings.llm_max_output_tokens,
            "prompt_version": PROMPT_VERSION,
        },
        "retrieval_configuration": {
            "embedding": "jina-embeddings-v5-text-small",
            "dimension": 1024,
            "retriever": "structural_hybrid",
            "recall_k": RECALL_K,
            "top_k": TOP_K,
            "rerank_enabled": False,
            "collection": collection["name"],
            "corpus": "production-corpus-v1",
        },
        "llm_called": True,
        "deep_research_called": False,
        "queries": rows,
        "metrics": summarize(rows),
        "mode_results": mode_results,
    }


if __name__ == "__main__":
    main()
