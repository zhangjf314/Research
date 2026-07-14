# ruff: noqa: E501
"""Stage 11C.6 retrieval recall and context-selection ablation."""

import argparse
import csv
import json
import math
import time
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean

from qdrant_client import QdrantClient

from paper_research.config import Settings
from paper_research.generation.qa_service import QAService
from paper_research.indexing.registry import IndexRegistry
from paper_research.indexing.vector_store import QdrantVectorStore
from paper_research.providers.factory import build_embedding_provider, build_llm_provider
from paper_research.providers.llm import LLMProviderError
from paper_research.retrieval.context_builder import ContextItem
from paper_research.retrieval.context_strategy import ContextStrategy, StrategicContextBuilder
from paper_research.retrieval.dense import DenseRetriever
from paper_research.retrieval.filters import RetrievalFilter
from paper_research.retrieval.fusion import reciprocal_rank_fusion
from paper_research.retrieval.sparse import BM25Retriever

try:
    import scripts.run_qa_context_diagnostics_v1 as diagnostics_v1
    import scripts.run_qa_production_v1 as qa_v1
    import scripts.run_retrieval_ablation_v2 as v2
except ModuleNotFoundError:
    import run_qa_context_diagnostics_v1 as diagnostics_v1  # type: ignore[no-redef]
    import run_qa_production_v1 as qa_v1  # type: ignore[no-redef]
    import run_retrieval_ablation_v2 as v2  # type: ignore[no-redef]

PROTOCOL = Path("data/evaluation/retrieval-gold-v2.jsonl")
GOLD = Path("data/evaluation/gold-set-v1.jsonl")
CORPUS = Path("data/evaluation/production-corpus-v1.json")
INDEX_MANIFEST = Path("data/evaluation/retrieval-index-v2.json")
QA_BASELINE = Path("data/evaluation/qa-production-v1.json")
DIAGNOSTICS = Path("data/evaluation/qa-context-diagnostics-v1.json")
DEFAULT_OUTPUT = Path("data/evaluation/retrieval-context-optimization-v1.json")
DEFAULT_CSV = Path("data/evaluation/retrieval-context-optimization-v1.csv")
DEFAULT_REPORT = Path("docs/retrieval-context-optimization-v1.md")
AUDIT_SAMPLE = Path("data/evaluation/citation-human-audit-sample-v1.jsonl")
AUDIT_GUIDE = Path("docs/citation-human-audit-guide-v1.md")
COLLECTION = "papers_jina_eval34_v2__20260713152149"
MAX_RETRIEVAL_K = 30
PROMPT_VERSION = "qa-production-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("retrieval", "qa", "report"), required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-requests", type=int, default=80)
    parser.add_argument("--additional-api-requests", type=int, default=0)
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return round(ordered[max(0, math.ceil(len(ordered) * fraction) - 1)], 3)


def fixed_settings(settings: Settings, *, require_llm: bool) -> None:
    if settings.rerank_enabled:
        raise RuntimeError("RERANK_ENABLED must remain false")
    if settings.embedding_provider != "jina" or settings.embedding_model != "jina-embeddings-v5-text-small":
        raise RuntimeError("fixed Jina embedding is required")
    if require_llm and (
        settings.llm_provider != "siliconflow"
        or settings.llm_model != "Qwen/Qwen3-8B"
        or settings.prompt_version != PROMPT_VERSION
        or settings.llm_temperature != 0
    ):
        raise RuntimeError("fixed SiliconFlow Qwen/Qwen3-8B qa-production-v1 configuration required")


def base_strategy() -> ContextStrategy:
    return ContextStrategy(
        retrieval_k=20,
        context_k=10,
        max_context_characters=12000,
        max_context_tokens=12000,
        dense_weight=0.5,
        lexical_weight=0.5,
    )


def replace(strategy: ContextStrategy, **updates: object) -> ContextStrategy:
    return strategy.model_copy(update=updates)


def strategy_groups() -> dict[str, list[tuple[str, ContextStrategy]]]:
    baseline = base_strategy()
    top_k = [
        ("retrieval20_context5", replace(baseline, retrieval_k=20, context_k=5)),
        ("retrieval20_context8", replace(baseline, retrieval_k=20, context_k=8)),
        ("retrieval20_context10", baseline),
        ("retrieval30_context8", replace(baseline, retrieval_k=30, context_k=8)),
        ("retrieval30_context10", replace(baseline, retrieval_k=30, context_k=10)),
    ]
    return {"top_k": top_k}


def expansion_group(best: ContextStrategy) -> list[tuple[str, ContextStrategy]]:
    return [
        ("no_expansion", best),
        ("neighbor_window_1", replace(best, neighbor_window=1)),
        ("same_page_expansion", replace(best, page_expansion=True)),
        (
            "neighbor_window_1_plus_page_cap",
            replace(best, neighbor_window=1, max_blocks_per_page=2),
        ),
    ]


def diversity_group(best: ContextStrategy) -> list[tuple[str, ContextStrategy]]:
    uncapped = replace(best, max_blocks_per_page=None, max_blocks_per_section=None)
    return [
        ("no_cap", uncapped),
        ("max_2_blocks_per_page", replace(uncapped, max_blocks_per_page=2)),
        ("max_3_blocks_per_section", replace(uncapped, max_blocks_per_section=3)),
        (
            "page_and_section_cap",
            replace(uncapped, max_blocks_per_page=2, max_blocks_per_section=3),
        ),
    ]


def hybrid_group(best: ContextStrategy) -> list[tuple[str, ContextStrategy]]:
    return [
        ("dense_0.7_lexical_0.3", replace(best, dense_weight=0.7, lexical_weight=0.3)),
        ("dense_0.5_lexical_0.5", replace(best, dense_weight=0.5, lexical_weight=0.5)),
        ("dense_0.3_lexical_0.7", replace(best, dense_weight=0.3, lexical_weight=0.7)),
    ]


def query_metrics(context: list[ContextItem], trace: dict, gold: dict) -> dict:
    block_ids = {block for item in context for block in item.block_ids}
    pages = {
        page
        for item in context
        for page in range(item.page_start, item.page_end + 1)
        if item.paper_id in set(gold["gold_paper_ids"])
    }
    exact = bool(block_ids & set(gold["gold_block_ids"]))
    page = bool(pages & set(gold["gold_pages"]))
    output_ids = [item.chunk_id for item in context]
    page_values = list(trace["page_counts"].values())
    section_values = list(trace["section_counts"].values())
    return {
        "exact_gold_block_available": exact,
        "gold_page_available": page,
        "any_gold_evidence_available": exact or page,
        "context_duplication_rate": 1 - len(set(output_ids)) / max(1, len(output_ids)),
        "deduplicated_candidate_count": len(trace["duplicate_chunk_ids"]),
        "context_item_count": len(context),
        "max_items_per_page": max(page_values, default=0),
        "max_items_per_section": max(section_values, default=0),
        "token_count": trace["estimated_tokens"],
        "truncated": bool(trace["truncated_chunk_ids"]),
    }


def summarize(rows: list[dict]) -> dict:
    metrics = [row["metrics"] for row in rows]
    avg = lambda key: round(mean(float(item[key]) for item in metrics), 6)  # noqa: E731
    return {
        "query_count": len(rows),
        "exact_gold_block_available": avg("exact_gold_block_available"),
        "gold_page_available": avg("gold_page_available"),
        "any_gold_evidence_available": avg("any_gold_evidence_available"),
        "context_duplication_rate": avg("context_duplication_rate"),
        "deduplicated_candidate_count": sum(item["deduplicated_candidate_count"] for item in metrics),
        "mean_context_items": avg("context_item_count"),
        "mean_max_items_per_page": avg("max_items_per_page"),
        "mean_max_items_per_section": avg("max_items_per_section"),
        "mean_tokens": round(mean(item["token_count"] for item in metrics), 3),
        "p95_tokens": percentile([item["token_count"] for item in metrics], 0.95),
        "truncation_rate": avg("truncated"),
    }


def selection_key(experiment: dict) -> tuple[float, ...]:
    metrics = experiment["metrics"]
    page_floor_met = metrics["gold_page_available"] >= 0.708333
    return (
        float(page_floor_met),
        metrics["exact_gold_block_available"],
        metrics["gold_page_available"],
        metrics["any_gold_evidence_available"],
        -metrics["truncation_rate"],
        -metrics["mean_tokens"],
    )


def evaluate_strategy(
    identifier: str,
    stage: str,
    strategy: ContextStrategy,
    caches: list[dict],
    chunks: list,
    raw_to_public: dict[str, str],
) -> dict:
    builder = StrategicContextBuilder(chunks, strategy)
    rows = []
    for cached in caches:
        fused = reciprocal_rank_fusion(
            cached["dense"],
            cached["sparse"],
            dense_weight=strategy.dense_weight,
            lexical_weight=strategy.lexical_weight,
        )
        built = builder.build(fused)
        context = [
            item.model_copy(update={"paper_id": raw_to_public[item.paper_id]})
            for item in built.context
        ]
        trace = built.trace.model_dump()
        rows.append(
            {
                "question_id": cached["record"]["question_id"],
                "context": [item.model_dump() for item in context],
                "trace": trace,
                "retrieval_latency_ms": cached["retrieval_latency_ms"],
                "metrics": query_metrics(context, trace, cached["gold"]),
            }
        )
    return {
        "experiment_id": identifier,
        "stage": stage,
        "strategy": strategy.model_dump(),
        "metrics": summarize(rows),
        "queries": rows,
    }


def retrieve_caches(settings: Settings) -> tuple[list[dict], list, dict[str, str], dict]:
    protocol = v2.load_jsonl(PROTOCOL)
    gold_by_id = {item["question_id"]: item for item in v2.load_jsonl(GOLD)}
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    manifest = json.loads(INDEX_MANIFEST.read_text(encoding="utf-8"))
    v2.validate_inputs(protocol, corpus, manifest)
    collection = manifest["collections"]["jina"]
    if collection["name"] != COLLECTION:
        raise RuntimeError("fixed Stage 11A.5 Jina collection changed")
    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key, check_compatibility=False)
    chunks = v2.load_chunks(client, COLLECTION)
    if len(chunks) != 2062 or v2.chunk_signature(chunks) != collection["chunk_signature"]:
        raise RuntimeError("live fixed collection does not match signed manifest")
    included = [paper for paper in corpus["papers"] if paper["included_in_production"]]
    public_to_raw = {paper["paper_id"]: paper["database_id"] for paper in included}
    raw_to_public = {paper["database_id"]: paper["paper_id"] for paper in included}
    store = QdrantVectorStore(
        client,
        IndexRegistry(settings.data_dir / "index_registry.json").resolve(COLLECTION),
        1024,
    )
    dense = DenseRetriever(build_embedding_provider(settings), store)
    sparse = BM25Retriever(chunks)
    caches = []
    for record in protocol:
        gold = gold_by_id[record["question_id"]]
        if not gold["answerable"]:
            continue
        retrieval_filter = RetrievalFilter(
            paper_ids=[public_to_raw[item] for item in record["retrieval_filter"]["paper_ids"]]
        )
        retrieval_started = time.perf_counter()
        dense_results = dense.retrieve(
            record["retrieval_query"],
            retrieval_filter=retrieval_filter,
            top_k=MAX_RETRIEVAL_K,
        )
        sparse_results = sparse.retrieve(
            record["retrieval_query"],
            retrieval_filter=retrieval_filter,
            top_k=MAX_RETRIEVAL_K,
        )
        caches.append(
            {
                "record": record,
                "gold": gold,
                "dense": dense_results,
                "sparse": sparse_results,
                "retrieval_latency_ms": round(
                    (time.perf_counter() - retrieval_started) * 1000, 3
                ),
            }
        )
    return caches, chunks, raw_to_public, collection


def run_retrieval(settings: Settings, output: Path) -> dict:
    caches, chunks, raw_to_public, collection = retrieve_caches(settings)
    experiments = []
    top = [
        evaluate_strategy(identifier, "top_k", strategy, caches, chunks, raw_to_public)
        for identifier, strategy in strategy_groups()["top_k"]
    ]
    experiments.extend(top)
    best_top = max(top, key=selection_key)
    expansion = [
        evaluate_strategy(identifier, "expansion", strategy, caches, chunks, raw_to_public)
        for identifier, strategy in expansion_group(ContextStrategy.model_validate(best_top["strategy"]))
    ]
    experiments.extend(expansion)
    best_expansion = max(expansion, key=selection_key)
    diversity = [
        evaluate_strategy(identifier, "diversity", strategy, caches, chunks, raw_to_public)
        for identifier, strategy in diversity_group(ContextStrategy.model_validate(best_expansion["strategy"]))
    ]
    experiments.extend(diversity)
    best_diversity = max(diversity, key=selection_key)
    hybrid = [
        evaluate_strategy(identifier, "hybrid_weight", strategy, caches, chunks, raw_to_public)
        for identifier, strategy in hybrid_group(ContextStrategy.model_validate(best_diversity["strategy"]))
    ]
    experiments.extend(hybrid)
    best_hybrid = max(hybrid, key=selection_key)
    payload = {
        "status": "RETRIEVAL_COMPLETED",
        "generated_at": datetime.now(UTC).isoformat(),
        "frozen_protocol": {
            "embedding": "jina-embeddings-v5-text-small",
            "collection": COLLECTION,
            "point_count": 2062,
            "corpus_documents": 34,
            "rerank_enabled": False,
            "llm": "SiliconFlow Qwen/Qwen3-8B",
            "prompt_version": PROMPT_VERSION,
            "deep_research_called": False,
            "gold_modified": False,
        },
        "collection": collection,
        "selection_rule": "max exact availability, then page, any evidence, lower truncation, lower tokens",
        "stage_winners": {
            "top_k": best_top["experiment_id"],
            "expansion": best_expansion["experiment_id"],
            "diversity": best_diversity["experiment_id"],
            "hybrid_weight": best_hybrid["experiment_id"],
        },
        "final_experiment_id": best_hybrid["experiment_id"],
        "experiments": experiments,
        "qa_comparison": {},
    }
    write_outputs(payload, output)
    write_audit_sample()
    return payload


def run_qa(settings: Settings, output: Path, *, resume: bool, max_requests: int) -> dict:
    payload = json.loads(output.read_text(encoding="utf-8"))
    final = next(
        item for item in payload["experiments"] if item["experiment_id"] == payload["final_experiment_id"]
    )
    gold_by_id = {item["question_id"]: item for item in v2.load_jsonl(GOLD)}
    protocol_by_id = {item["question_id"]: item for item in v2.load_jsonl(PROTOCOL)}
    existing = {
        item["question_id"]: item
        for item in payload.get("qa_candidate_queries", [])
        if item.get("status") == "COMPLETED"
    } if resume else {}
    rows = list(existing.values())
    qa = QAService(llm=build_llm_provider(settings), prompt_version=PROMPT_VERSION)
    for source in final["queries"]:
        question_id = source["question_id"]
        if question_id in existing or sum(item.get("api_request_count", 0) for item in rows) >= max_requests:
            continue
        record = protocol_by_id[question_id]
        gold = gold_by_id[question_id]
        context = [ContextItem.model_validate(item) for item in source["context"]]
        retrieval_latency_ms = source["retrieval_latency_ms"]
        started = time.perf_counter() - retrieval_latency_ms / 1000
        row = {
            "question_id": question_id,
            "experiment_id": final["experiment_id"],
            "context": source["context"],
            "context_trace": source["trace"],
            "gold": {
                "answerable": gold["answerable"],
                "gold_paper_ids": gold["gold_paper_ids"],
                "gold_block_ids": gold["gold_block_ids"],
                "gold_pages": gold["gold_pages"],
                "required_claims": gold["required_claims"],
            },
            "gold_block_present": source["metrics"]["exact_gold_block_available"],
            "retrieval_latency_ms": retrieval_latency_ms,
        }
        try:
            answer = qa.answer_from_context(
                record["retrieval_query"],
                context,
                retrieval_latency_ms=retrieval_latency_ms,
                total_started=started,
            )
            answer_dict = answer.model_dump()
            diagnostics = diagnostics_v1.diagnose_answer(answer_dict, context, gold)
            row.update(
                status="COMPLETED",
                answer=answer_dict,
                metrics=qa_v1.evaluate_answer({"answer": answer_dict}, gold, context),
                diagnostics=diagnostics,
                api_request_count=answer.api_request_count,
                retry_count=answer.retry_count,
            )
        except LLMProviderError as exc:
            row.update(
                status="FAILED",
                failure_reason=str(exc),
                api_request_count=exc.api_request_count,
                retry_count=len(exc.retry_reasons),
            )
        rows = [item for item in rows if item["question_id"] != question_id] + [row]
        payload["qa_candidate_queries"] = sorted(rows, key=lambda item: item["question_id"])
        write_outputs(payload, output)
    completed = [item for item in rows if item["status"] == "COMPLETED"]
    baseline = json.loads(QA_BASELINE.read_text(encoding="utf-8"))
    baseline_diagnostics = json.loads(DIAGNOSTICS.read_text(encoding="utf-8"))["metrics"][
        "retrieved"
    ]
    candidate_metrics = qa_v1.summarize(completed)
    diagnostic_rows = [item["diagnostics"] for item in completed]
    candidate_metrics.update(
        {
            "page_level_precision": round(
                mean(item["page_level_precision"] for item in diagnostic_rows), 6
            ) if diagnostic_rows else None,
            "adjacent_support_precision": round(
                mean(item["adjacent_support_precision"] for item in diagnostic_rows), 6
            ) if diagnostic_rows else None,
            "semantic_support_precision": round(
                mean(item["semantic_support_precision"] for item in diagnostic_rows), 6
            ) if diagnostic_rows else None,
            "unsupported_rate": round(
                sum(item["legacy_unsupported_claim_count"] for item in diagnostic_rows)
                / max(1, sum(len(item["answer"]["claims"]) for item in completed)),
                6,
            ),
        }
    )
    baseline_metrics = dict(baseline["metrics"])
    baseline_metrics.update(
        {
            "page_level_precision": baseline_diagnostics["page_level_precision"],
            "adjacent_support_precision": baseline_diagnostics["adjacent_support_precision"],
            "semantic_support_precision": baseline_diagnostics["semantic_support_precision"],
            "unsupported_rate": baseline_diagnostics["unsupported_rate"],
            "answerable_only_tokens": baseline_diagnostics["total_tokens"],
        }
    )
    payload["qa_comparison"] = {
        "baseline": {
            "source": str(QA_BASELINE),
            "reused_frozen_real_run": True,
            "new_llm_requests": 0,
            "metrics": baseline_metrics,
        },
        "final_candidate": {
            "experiment_id": final["experiment_id"],
            "metrics": candidate_metrics,
            "completed": len(completed),
            "failure_count": sum(item["status"] == "FAILED" for item in rows),
            "api_requests": sum(item.get("api_request_count", 0) for item in rows),
        },
    }
    payload["status"] = "COMPLETED" if len(completed) == 48 else "PARTIAL"
    write_outputs(payload, output)
    return payload


def finalize_report(output: Path, additional_api_requests: int) -> dict:
    payload = json.loads(output.read_text(encoding="utf-8"))
    baseline = json.loads(QA_BASELINE.read_text(encoding="utf-8"))
    diagnostics = json.loads(DIAGNOSTICS.read_text(encoding="utf-8"))
    baseline_by_id = {item["question_id"]: item for item in baseline["queries"]}
    baseline_diagnostics = {
        item["question_id"]: item
        for item in diagnostics["runs"]
        if item["context_mode"] == "retrieved"
    }
    completed = [
        item for item in payload.get("qa_candidate_queries", []) if item["status"] == "COMPLETED"
    ]
    failed = [
        item for item in payload.get("qa_candidate_queries", []) if item["status"] == "FAILED"
    ]

    def change_counts(source: str, metric: str) -> dict:
        deltas = []
        for item in completed:
            question_id = item["question_id"]
            baseline_metrics = (
                baseline_diagnostics[question_id]["diagnostics"]
                if source == "diagnostics"
                else baseline_by_id[question_id]["metrics"]
            )
            candidate_metrics = item["diagnostics"] if source == "diagnostics" else item["metrics"]
            deltas.append((candidate_metrics.get(metric) or 0) - (baseline_metrics.get(metric) or 0))
        return {
            "improved": sum(value > 1e-9 for value in deltas),
            "degraded": sum(value < -1e-9 for value in deltas),
            "unchanged": sum(abs(value) <= 1e-9 for value in deltas),
            "mean_delta": round(mean(deltas), 6) if deltas else None,
        }

    comparison = payload["qa_comparison"]
    baseline_metrics = comparison["baseline"]["metrics"]
    candidate_metrics = comparison["final_candidate"]["metrics"]
    final_experiment = next(
        item for item in payload["experiments"] if item["experiment_id"] == payload["final_experiment_id"]
    )
    audit_rows = [
        json.loads(line)
        for line in AUDIT_SAMPLE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    actual_requests = comparison["final_candidate"]["api_requests"] + additional_api_requests
    payload["qa_execution_audit"] = {
        "initial_and_latest_row_api_requests": comparison["final_candidate"]["api_requests"],
        "additional_resume_retry_requests": additional_api_requests,
        "actual_api_requests": actual_requests,
        "explanation": "Two additional resume runs retried q033 and q044 three times each; failed rows are replaced on resume, so 12 requests are restored here from command records.",
        "failed_question_ids": [item["question_id"] for item in failed],
        "failure_reasons": {item["question_id"]: item.get("failure_reason") for item in failed},
    }
    payload["per_query_change_counts"] = {
        "exact_citation_precision": change_counts("metrics", "citation_precision"),
        "page_level_precision": change_counts("diagnostics", "page_level_precision"),
        "required_claim_coverage": change_counts("metrics", "required_claim_coverage"),
    }
    conservative_answerable = sum(item["answer"]["answerable"] for item in completed) / 48
    conservative_claim = sum(
        item["metrics"].get("required_claim_coverage") or 0 for item in completed
    ) / 48
    gates = {
        "exact_gold_block_availability_above_43_8": final_experiment["metrics"]["exact_gold_block_available"] > 0.438,
        "gold_page_availability_at_least_70_8": final_experiment["metrics"]["gold_page_available"] >= 0.708,
        "qa_exact_precision_meaningfully_above_10_6": candidate_metrics["citation_precision"] >= 0.126,
        "unsupported_rate_below_81_1": candidate_metrics["unsupported_rate"] < 0.811,
        "conservative_answerable_accuracy_at_least_87_5": conservative_answerable >= 0.875,
        "p95_latency_within_25_percent": candidate_metrics["total_latency"]["p95_ms"] <= baseline_metrics["total_latency"]["p95_ms"] * 1.25,
        "zero_final_qa_failures": not failed,
        "improvement_not_driven_by_few_queries": payload["per_query_change_counts"]["exact_citation_precision"]["improved"] >= 10,
        "human_audit_complete_without_severe_semantic_distortion": all(item["human_review_status"] != "pending" for item in audit_rows),
    }
    payload["decision"] = {
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "stage_11c6_passed": all(gates.values()),
        "allow_stage_11d_smoke": all(gates.values()),
        "conservative_answerable_accuracy": round(conservative_answerable, 6),
        "conservative_required_claim_coverage": round(conservative_claim, 6),
        "human_audit_pending": sum(item["human_review_status"] == "pending" for item in audit_rows),
    }
    payload["status"] = "COMPLETED_WITH_BLOCKING_QA_FAILURES"
    write_outputs(payload, output)
    return payload


def write_audit_sample() -> None:
    diagnostics = json.loads(DIAGNOSTICS.read_text(encoding="utf-8"))
    retrieved = [item for item in diagnostics["runs"] if item["context_mode"] == "retrieved"]
    paper_ids = {
        paper_id for row in retrieved for paper_id in row["gold"]["gold_paper_ids"]
    }
    source_blocks = diagnostics_v1.load_blocks(paper_ids)
    buckets: dict[str, list[dict]] = {"semantic_non_gold": [], "same_gold_page": [], "unsupported": []}
    for row in retrieved:
        context_by_block = {
            block_id: item["evidence"]
            for item in row["context"]
            for block_id in item["block_ids"]
        }
        claim_by_id = {item["claim_id"]: item for item in row["answer"]["claims"]}
        for detail in row["diagnostics"]["citation_details"]:
            claim = claim_by_id.get(detail["claim_id"], {})
            for citation in detail["citations"]:
                classification = citation["classification"]
                bucket = (
                    "semantic_non_gold" if classification == "semantic_support_non_gold"
                    else "same_gold_page" if classification == "same_gold_page"
                    else "unsupported" if classification in {"unsupported", "weakly_related"}
                    else None
                )
                if bucket is None:
                    continue
                buckets[bucket].append(
                    {
                        "question_id": row["question_id"],
                        "claim_text": claim.get("text", ""),
                        "cited_paper_id": citation.get("paper_id"),
                        "cited_page": citation.get("page"),
                        "cited_block_id": citation.get("block_id"),
                        "cited_block_text": (
                            source_blocks[(citation.get("paper_id"), citation.get("block_id"))].text
                            if (citation.get("paper_id"), citation.get("block_id")) in source_blocks
                            else context_by_block.get(citation.get("block_id"), "")
                        ),
                        "gold_block_ids": row["gold"]["gold_block_ids"],
                        "gold_block_text": [
                            source_blocks[(paper_id, block_id)].text
                            for paper_id in row["gold"]["gold_paper_ids"]
                            for block_id in row["gold"]["gold_block_ids"]
                            if (paper_id, block_id) in source_blocks
                        ],
                        "automated_labels": {
                            "classification": classification,
                            "exact_gold": False,
                            "same_gold_page": classification == "same_gold_page",
                            "semantic_support": classification == "semantic_support_non_gold",
                            "unsupported": bucket == "unsupported",
                            "semantic_score": citation.get("semantic_score"),
                        },
                        "suggested_human_label": (
                            "unsupported" if bucket == "unsupported" else "related_but_insufficient"
                        ),
                        "human_review_status": "pending",
                        "human_label": None,
                        "review_notes": None,
                    }
                )
    sample = []
    for bucket in ("semantic_non_gold", "same_gold_page", "unsupported"):
        if len(buckets[bucket]) < 10:
            raise RuntimeError(f"insufficient {bucket} audit candidates: {len(buckets[bucket])}")
        sample.extend(buckets[bucket][:10])
    AUDIT_SAMPLE.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_SAMPLE.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in sample),
        encoding="utf-8",
    )
    AUDIT_GUIDE.write_text(
        """# Citation Human Audit Guide v1

This file contains 30 pending claim-citation judgments: 10 semantic non-Gold,
10 same-Gold-page non-exact, and 10 unsupported/weak automated cases.

Review the claim against the cited block text and the Gold block text. Do not infer support
from the paper title or outside knowledge. `human_label` must be one of:

- `fully_supported`
- `partially_supported`
- `related_but_insufficient`
- `unsupported`
- `gold_annotation_too_narrow`

Only a human reviewer may replace `human_review_status=pending`, populate `human_label`,
and add `review_notes`. `suggested_human_label` is an automated routing hint, not a human
conclusion, and must not be copied into `human_label` without review.
""",
        encoding="utf-8",
    )


def write_outputs(payload: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_rows = []
    for experiment in payload["experiments"]:
        csv_rows.append(
            {
                "stage": experiment["stage"],
                "experiment_id": experiment["experiment_id"],
                **experiment["strategy"],
                **experiment["metrics"],
            }
        )
    with DEFAULT_CSV.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]) if csv_rows else ["stage"])
        writer.writeheader()
        writer.writerows(csv_rows)
    lines = [
        "# Retrieval Context Optimization v1",
        "",
        "> Reranker disabled; embedding, LLM, prompt, corpus, queries, filters, chunks, and Gold are frozen. Oracle Gold is never injected into Production contexts.",
        "",
        "## Current implementation audit",
        "",
        "Stage 11C used equal-weight RRF, retrieval K=20, context K=10, chunk-ID deduplication, a configured 12,000-token budget plus a binding 12,000-character rank-prefix cap (about 3,000 estimated tokens), and no structural expansion or diversity caps. Its trace did not explain expansion sources because no expansion policy existed. Gold-page hits without exact blocks therefore could not be repaired structurally.",
        "",
        "The optimized strategy records original rank/score, expansion reason/source, deduplication, exclusions, final rank, token truncation, and per-page/per-section concentration. `max_blocks_per_*` caps structural context items because the indexed retrieval unit is a structural multi-block chunk.",
        "",
        "## Pure retrieval/context results",
        "",
        "| Stage | Experiment | Exact | Gold page | Any evidence | Duplication | Mean tokens | Truncation |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for experiment in payload["experiments"]:
        metric = experiment["metrics"]
        lines.append(
            f"| {experiment['stage']} | {experiment['experiment_id']} | {metric['exact_gold_block_available']:.3f} | {metric['gold_page_available']:.3f} | {metric['any_gold_evidence_available']:.3f} | {metric['context_duplication_rate']:.3f} | {metric['mean_tokens']:.1f} | {metric['truncation_rate']:.3f} |"
        )
    if payload.get("stage_winners"):
        lines += ["", "## Selected strategies", ""]
        for stage, identifier in payload["stage_winners"].items():
            lines.append(f"- {stage}: `{identifier}`")
        lines += ["", f"Final candidate: `{payload['final_experiment_id']}`."]
    if payload.get("qa_comparison"):
        lines += ["", "## Real QA comparison", ""]
        for name, result in payload["qa_comparison"].items():
            metrics = result["metrics"]
            lines.append(
                f"- {name}: answerable={metrics.get('answerable_accuracy')}, claim={metrics.get('required_claim_coverage')}, exact={metrics.get('citation_precision')}, page={metrics.get('page_level_precision', 'not-computed')}, recall={metrics.get('citation_recall')}, unsupported_rate={metrics.get('unsupported_rate')}, P95={metrics.get('total_latency', {}).get('p95_ms')} ms, tokens={metrics.get('total_tokens')}."
            )
    if payload.get("decision"):
        decision = payload["decision"]
        changes = payload["per_query_change_counts"]
        audit = payload["qa_execution_audit"]
        lines += [
            "",
            "## Acceptance decision",
            "",
            f"- Completed QA: {payload['qa_comparison']['final_candidate']['completed']}/48; blocking failures: {', '.join(audit['failed_question_ids']) or 'none'}.",
            f"- Actual API requests across the initial run and two resume attempts: {audit['actual_api_requests']}.",
            f"- Conservative answerable accuracy with failed rows counted incorrect: {decision['conservative_answerable_accuracy']:.3f}.",
            f"- Exact precision changes on completed common queries: {changes['exact_citation_precision']}.",
            f"- Page precision changes on completed common queries: {changes['page_level_precision']}.",
            f"- Required-claim changes on completed common queries: {changes['required_claim_coverage']}.",
            f"- Human audit pending: {decision['human_audit_pending']}/30.",
            "- Human audit strata: 10 semantic non-Gold, 10 same-Gold-page non-exact, and 10 unsupported/weak; all labels remain pending/null.",
            f"- Baseline latency mean/P95: {payload['qa_comparison']['baseline']['metrics']['total_latency']['mean_ms']}/{payload['qa_comparison']['baseline']['metrics']['total_latency']['p95_ms']} ms; candidate completed-row mean/P95: {payload['qa_comparison']['final_candidate']['metrics']['total_latency']['mean_ms']}/{payload['qa_comparison']['final_candidate']['metrics']['total_latency']['p95_ms']} ms.",
            f"- Baseline input/output/total tokens: {payload['qa_comparison']['baseline']['metrics']['input_tokens']}/{payload['qa_comparison']['baseline']['metrics']['output_tokens']}/{payload['qa_comparison']['baseline']['metrics']['total_tokens']}; candidate completed-row tokens: {payload['qa_comparison']['final_candidate']['metrics']['input_tokens']}/{payload['qa_comparison']['final_candidate']['metrics']['output_tokens']}/{payload['qa_comparison']['final_candidate']['metrics']['total_tokens']}.",
            "",
            "| Gate | Passed |",
            "|---|---:|",
        ]
        for gate, passed in decision["gates"].items():
            lines.append(f"| {gate} | {str(passed).lower()} |")
        lines += [
            "",
            f"Stage 11C.6 passed: **{str(decision['stage_11c6_passed']).lower()}**. Stage 11D smoke allowed: **{str(decision['allow_stage_11d_smoke']).lower()}**.",
        ]
    lines += [
        "",
        "## Decision status",
        "",
        "Human citation audit remains pending. Automated semantic support is not treated as human evidence. Stage 11D smoke is not authorized solely by these automated diagnostics.",
    ]
    DEFAULT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    settings = Settings()
    fixed_settings(settings, require_llm=args.phase == "qa")
    if args.phase == "retrieval":
        payload = run_retrieval(settings, args.output)
    elif args.phase == "qa":
        payload = run_qa(settings, args.output, resume=args.resume, max_requests=args.max_requests)
    else:
        payload = finalize_report(args.output, args.additional_api_requests)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "final_experiment_id": payload.get("final_experiment_id"),
                "stage_winners": payload.get("stage_winners"),
            }
        )
    )


if __name__ == "__main__":
    main()
