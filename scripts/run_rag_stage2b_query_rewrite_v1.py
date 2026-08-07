from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from paper_research.config import get_settings
from paper_research.evaluation.rag_official_baseline import retrieval_row
from paper_research.evaluation.rag_stage2a import (
    BOOTSTRAP_SEED,
    OPT_DOCS,
    OPT_ROOT,
    bad_case_delta,
    bootstrap_delta_ci,
    category_metrics,
    difficulty_metrics,
    load_dev_gold,
    paired_comparison,
)
from paper_research.evaluation.rag_stage2b import (
    DECOMPOSITION_PROMPT_VERSION,
    SINGLE_REWRITE_PROMPT_VERSION,
    SYSTEM_PROMPT,
    Decomposition,
    SingleRewrite,
    add_extended_row_metrics,
    aggregate_rewrite_usage,
    assert_no_gold_leakage,
    cache_key,
    decomposition_user_prompt,
    drift_labels,
    extended_metrics,
    fuse_ranked_contexts,
    prompt_hash,
    queries_for_config,
    read_cache,
    rewrite_specific_metrics,
    selection_gate,
    single_rewrite_user_prompt,
    write_cache,
    write_json,
    write_jsonl,
)
from paper_research.providers.factory import build_llm_provider
from paper_research.providers.llm import LLMProviderError

PLAN_PATH = OPT_ROOT / "stage2b-query-rewrite-plan-v1.json"
PLAN_DOC = OPT_DOCS / "stage2b-query-rewrite-plan-v1.md"
RESULT_JSON = OPT_ROOT / "stage2b-query-rewrite-v1.json"
RESULT_MD = OPT_DOCS / "stage2b-query-rewrite-v1.md"
AUDIT_JSONL = OPT_ROOT / "stage2b-query-rewrite-audit-v1.jsonl"
Q0_ITEMS = OPT_ROOT / "R3_current_hybrid-items-v1.jsonl"

CONFIGS = [
    "Q0_CURRENT_HYBRID",
    "Q1_SINGLE_REWRITE_REPLACE",
    "Q2_ORIGINAL_PLUS_SINGLE_REWRITE",
    "Q3_ORIGINAL_PLUS_DECOMPOSITION",
]


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def paper_id_map() -> dict[str, str]:
    manifest = json.loads(
        Path("data/evaluation/production-corpus-v1.json").read_text(encoding="utf-8")
    )
    return {
        str(paper.get("database_id")): str(paper.get("paper_id"))
        for paper in manifest.get("papers", [])
        if paper.get("included_in_production")
    }


def plan_payload() -> dict[str, Any]:
    single_prompt = SYSTEM_PROMPT + "\n" + single_rewrite_user_prompt("{question}")
    decomp_prompt = SYSTEM_PROMPT + "\n" + decomposition_user_prompt("{question}")
    return {
        "schema_version": "stage2b-query-rewrite-plan-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "split": "dev",
        "dev_question_count": 98,
        "test_questions_allowed": False,
        "test_questions_evaluated": 0,
        "q0_source": "STAGE2A_FROZEN_DEV_RESULT",
        "selected_stage2a_candidate": "R3_current_hybrid",
        "base_chain": {
            "retrieval": "Structural Hybrid",
            "reranker": "disabled",
            "embedding": "unchanged from active production retrieval collection",
            "top_k": 20,
            "recall_k": 20,
        },
        "provider": {
            "preferred_provider": "DeepSeek",
            "preferred_model": "deepseek/deepseek-v4-flash",
            "response_format": "json_object",
            "temperature": 0,
            "stream": False,
            "thinking_enabled": False,
        },
        "prompt_versions": {
            "single_rewrite": SINGLE_REWRITE_PROMPT_VERSION,
            "decomposition": DECOMPOSITION_PROMPT_VERSION,
        },
        "prompt_hashes": {
            "single_rewrite": prompt_hash(single_prompt),
            "decomposition": prompt_hash(decomp_prompt),
        },
        "configs": {
            "Q0_CURRENT_HYBRID": "Original question, Stage 2A frozen Current Hybrid rows.",
            "Q1_SINGLE_REWRITE_REPLACE": "Single rewritten query replaces original.",
            "Q2_ORIGINAL_PLUS_SINGLE_REWRITE": "Original query plus single rewrite, fused.",
            "Q3_ORIGINAL_PLUS_DECOMPOSITION": (
                "Original query plus 0-3 decomposition queries, fused; max 4 retrieval "
                "queries per logical question."
            ),
        },
        "failure_policy": {
            "Q1": "rewrite failure records a failed retrieval row",
            "Q2_Q3": (
                "rewrite/decomposition failure falls back to original query only and is flagged"
            ),
        },
        "budget": {
            "max_single_rewrite_calls": 98,
            "max_decomposition_calls": 98,
            "max_total_provider_calls": 196,
            "suggested_max_cost_usd": 0.10,
        },
        "decision_gate": {
            "absolute_gain_any_of": [
                "recall_at_10 >= +0.05",
                "evidence_coverage_at_10 >= +0.05",
                "required_claim_evidence_coverage_at_10 >= +0.05",
            ],
            "new_miss_rate_max": 0.05,
            "rewrite_success_rate_min": 0.98,
            "paper_recall_at_10_not_significantly_down": True,
            "bootstrap_resamples": 1000,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "forbidden": [
            "test split evaluation",
            "reranker",
            "context selection changes",
            "chunking changes",
            "embedding changes",
            "QA/generation benchmark",
            "Research Agent",
            "gold leakage in rewrite prompt",
        ],
    }


def write_plan() -> None:
    payload = plan_payload()
    write_json(PLAN_PATH, payload)
    PLAN_DOC.parent.mkdir(parents=True, exist_ok=True)
    PLAN_DOC.write_text(plan_markdown(payload), encoding="utf-8")


def plan_markdown(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Stage 2B query rewrite / decomposition plan v1",
            "",
            f"- split: `{payload['split']}`",
            f"- dev_question_count: `{payload['dev_question_count']}`",
            f"- test_questions_allowed: `{payload['test_questions_allowed']}`",
            f"- q0_source: `{payload['q0_source']}`",
            f"- selected_stage2a_candidate: `{payload['selected_stage2a_candidate']}`",
            f"- reranker: `{payload['base_chain']['reranker']}`",
            "",
            "## Configs",
            "",
            *[f"- `{key}`: {value}" for key, value in payload["configs"].items()],
            "",
            "## Decision gate",
            "",
            "At least one candidate must satisfy the pre-registered gain threshold without "
            "unacceptable new misses or paper recall regression. Otherwise Q0 remains selected.",
            "",
            "## Protocol guardrails",
            "",
            *[f"- {item}" for item in payload["forbidden"]],
            "",
        ]
    )


def load_plan() -> dict[str, Any]:
    if not PLAN_PATH.exists():
        raise FileNotFoundError(f"missing preregistered Stage 2B plan: {PLAN_PATH}")
    payload = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    if payload.get("test_questions_allowed"):
        raise RuntimeError("TEST_PROTOCOL_VIOLATION")
    return payload


def generate_rewrites(gold_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    settings = get_settings()
    provider = build_llm_provider(settings)
    if not hasattr(provider, "generate_structured_json") or provider.provider_name == "template":
        raise RuntimeError("real structured JSON rewrite provider is required for Stage 2B")
    model = getattr(provider, "model_name", settings.llm_model)
    items = []
    for gold in gold_rows:
        question_id = gold["question_id"]
        original = gold["question"]
        item = {
            "question_id": question_id,
            "provider": provider.provider_name,
            "model": model,
            "original_query": original,
            "single_rewrite": None,
            "decomposition_queries": [],
            "single_status": "not_started",
            "decomposition_status": "not_started",
            "single_provider_requests": 0,
            "decomposition_provider_requests": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "estimated_cost_usd": 0.0,
            "usage_source": "provider_reported",
        }
        _run_single_rewrite(provider, gold, item)
        _run_decomposition(provider, gold, item)
        item["drift_labels"] = drift_labels(
            original,
            [
                query
                for query in [item.get("single_rewrite"), *item.get("decomposition_queries", [])]
                if query
            ],
        )
        items.append(item)
    return items


def _usage_to_item(item: dict[str, Any], result: Any) -> None:
    usage = result.usage
    item["input_tokens"] += int(usage.input_tokens or 0)
    item["output_tokens"] += int(usage.output_tokens or 0)
    item["total_tokens"] += int(usage.total_tokens or 0)
    item["estimated_cost_usd"] = round(
        float(item.get("estimated_cost_usd") or 0.0)
        + float(usage.estimated_cost_usd or 0.0),
        8,
    )


def _estimate_cached_usage(item: dict[str, Any], user_prompt: str, output_text: str) -> None:
    input_tokens = max(1, round((len(SYSTEM_PROMPT) + len(user_prompt)) / 4))
    output_tokens = max(1, round(len(output_text) / 4))
    item["input_tokens"] += input_tokens
    item["output_tokens"] += output_tokens
    item["total_tokens"] += input_tokens + output_tokens
    item["usage_source"] = "cache_text_estimated_after_interrupted_provider_run"


def _run_single_rewrite(provider: Any, gold: dict[str, Any], item: dict[str, Any]) -> None:
    user_prompt = single_rewrite_user_prompt(gold["question"])
    assert_no_gold_leakage(user_prompt, gold)
    digest = prompt_hash(SYSTEM_PROMPT + "\n" + user_prompt)
    key = cache_key(gold["question_id"], SINGLE_REWRITE_PROMPT_VERSION, digest, item["model"])
    cached = read_cache("single", key)
    if cached:
        item.update(cached)
        item["single_cache_historical_provider_requests"] = int(
            cached.get("single_provider_requests") or 0
        )
        item["single_provider_requests"] = 0
        item["single_cache_hit"] = True
        _estimate_cached_usage(item, user_prompt, json.dumps(cached, ensure_ascii=False))
        return
    started = time.perf_counter()
    try:
        result = provider.generate_structured_json(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            schema_name=SINGLE_REWRITE_PROMPT_VERSION,
            request_context={"question_id": gold["question_id"], "stage": "stage2b_single"},
            max_output_tokens=256,
        )
        parsed = SingleRewrite.model_validate(result.payload)
        payload = {
            "single_rewrite": parsed.rewritten_query,
            "single_status": "success",
            "single_latency_ms": result.total_latency_ms,
            "single_prompt_hash": digest,
            "single_provider_requests": result.request_attempt_count,
            "single_provider_request_id": result.provider_request_id,
        }
        _usage_to_item(item, result)
    except (LLMProviderError, ValueError) as exc:
        payload = {
            "single_rewrite": None,
            "single_status": "failed",
            "single_failure_reason": f"{type(exc).__name__}: {exc}",
            "single_latency_ms": round((time.perf_counter() - started) * 1000, 3),
            "single_prompt_hash": digest,
            "single_provider_requests": getattr(exc, "api_request_count", 0) or 0,
        }
    item.update(payload)
    write_cache("single", key, payload)


def _run_decomposition(provider: Any, gold: dict[str, Any], item: dict[str, Any]) -> None:
    user_prompt = decomposition_user_prompt(gold["question"])
    assert_no_gold_leakage(user_prompt, gold)
    digest = prompt_hash(SYSTEM_PROMPT + "\n" + user_prompt)
    key = cache_key(gold["question_id"], DECOMPOSITION_PROMPT_VERSION, digest, item["model"])
    cached = read_cache("decomposition", key)
    if cached:
        item.update(cached)
        item["decomposition_cache_historical_provider_requests"] = int(
            cached.get("decomposition_provider_requests") or 0
        )
        item["decomposition_provider_requests"] = 0
        item["decomposition_cache_hit"] = True
        _estimate_cached_usage(item, user_prompt, json.dumps(cached, ensure_ascii=False))
        return
    started = time.perf_counter()
    try:
        result = provider.generate_structured_json(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            schema_name=DECOMPOSITION_PROMPT_VERSION,
            request_context={"question_id": gold["question_id"], "stage": "stage2b_decomposition"},
            max_output_tokens=512,
        )
        parsed = Decomposition.model_validate(result.payload)
        payload = {
            "decomposition_queries": [query.query for query in parsed.queries],
            "decomposition_purposes": [query.purpose for query in parsed.queries],
            "decomposition_status": "success",
            "decomposition_latency_ms": result.total_latency_ms,
            "decomposition_prompt_hash": digest,
            "decomposition_provider_requests": result.request_attempt_count,
            "decomposition_provider_request_id": result.provider_request_id,
        }
        _usage_to_item(item, result)
    except (LLMProviderError, ValueError) as exc:
        payload = {
            "decomposition_queries": [],
            "decomposition_purposes": [],
            "decomposition_status": "failed",
            "decomposition_failure_reason": f"{type(exc).__name__}: {exc}",
            "decomposition_latency_ms": round((time.perf_counter() - started) * 1000, 3),
            "decomposition_prompt_hash": digest,
            "decomposition_provider_requests": getattr(exc, "api_request_count", 0) or 0,
        }
    item.update(payload)
    write_cache("decomposition", key, payload)


def retrieve_query(query: str) -> tuple[list[dict[str, Any]], float, str | None, bool]:
    key = prompt_hash(query)
    cached = read_cache("retrieval", key)
    if cached:
        return (
            list(cached.get("context") or []),
            float(cached.get("latency_ms") or 0.0),
            cached.get("failure"),
            True,
        )
    started = time.perf_counter()
    try:
        with httpx.Client(timeout=180) as client:
            response = client.post(
                "http://localhost/api/v1/retrieve",
                json={"query": query, "filters": {}, "recall_k": 20, "top_k": 20},
            )
            latency_ms = round((time.perf_counter() - started) * 1000, 3)
            response.raise_for_status()
            body = response.json()
        context = [item for item in body.get("context", [])]
        write_cache(
            "retrieval",
            key,
            {"context": context, "latency_ms": latency_ms, "failure": None},
        )
        return context, latency_ms, None, False
    except Exception as exc:  # noqa: BLE001 - experiment records failed query
        latency_ms = round((time.perf_counter() - started) * 1000, 3)
        failure = f"{type(exc).__name__}: {exc}"
        write_cache(
            "retrieval",
            key,
            {"context": [], "latency_ms": latency_ms, "failure": failure},
        )
        return [], latency_ms, failure, False


def collect_query_contexts(
    query_jobs: list[tuple[str, str, str]],
) -> dict[str, tuple[list[dict[str, Any]], float, str | None]]:
    results = {}
    for index, (job_id, config_id, query) in enumerate(query_jobs, start=1):
        context, latency_ms, failure, cache_hit = retrieve_query(query)
        results[job_id] = (context, latency_ms, failure)
        if index % 10 == 0 or index == len(query_jobs):
            print(
                json.dumps(
                    {
                        "stage": "retrieval_collect",
                        "completed": index,
                        "total": len(query_jobs),
                        "config_id": config_id,
                        "cache_hit": cache_hit,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    return results


def run_candidate_rows(
    config_id: str,
    gold_rows: list[dict[str, Any]],
    rewrite_items: dict[str, dict[str, Any]],
    id_map: dict[str, str],
    q0_context_by_id: dict[str, list[dict[str, Any]]],
    query_contexts: dict[str, tuple[list[dict[str, Any]], float, str | None]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    retrieval_calls = 0
    fallback_count = 0
    query_failures = 0
    for gold in gold_rows:
        queries, fallback = queries_for_config(
            config_id,
            gold["question"],
            rewrite_items.get(gold["question_id"]),
        )
        fallback_count += int(fallback)
        ranked_lists = []
        latency = 0.0
        failures = []
        if config_id in {"Q2_ORIGINAL_PLUS_SINGLE_REWRITE", "Q3_ORIGINAL_PLUS_DECOMPOSITION"}:
            ranked_lists.append(q0_context_by_id[gold["question_id"]])
        for query in queries:
            if query == gold["question"] and config_id != "Q1_SINGLE_REWRITE_REPLACE":
                continue
            retrieval_calls += 1
            context, latency_ms, failure = query_contexts[_query_job_id(gold["question_id"], query)]
            latency += latency_ms
            if failure:
                failures.append(failure)
            ranked_lists.append(context)
        if config_id == "Q1_SINGLE_REWRITE_REPLACE" and not queries:
            failures.append("single rewrite unavailable")
        fused = fuse_ranked_contexts(ranked_lists, top_k=20)
        failure_reason = "; ".join(failures) if failures and not fused else None
        query_failures += int(bool(failure_reason))
        row = retrieval_row(
            gold,
            fused,
            latency_ms=round(latency, 3),
            failure=failure_reason,
            paper_id_map=id_map,
        )
        row = add_extended_row_metrics(row, gold)
        row["experiment_id"] = config_id
        row["retrieval_mode"] = "hybrid_multi_query" if len(queries) > 1 else "hybrid"
        row["reranker"] = "none"
        row["retrieval_queries"] = queries
        row["rewrite_provider_failure"] = fallback
        rows.append(row)
        if len(rows) % 10 == 0 or len(rows) == len(gold_rows):
            print(
                json.dumps(
                    {
                        "stage": "retrieval",
                        "config_id": config_id,
                        "completed": len(rows),
                        "total": len(gold_rows),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    return rows, {
        "retrieval_call_count": retrieval_calls,
        "rewrite_provider_failure_fallback_count": fallback_count,
        "failed_query_count": query_failures,
    }


def _query_job_id(question_id: str, query: str) -> str:
    return f"{question_id}:{prompt_hash(query)}"


def build_query_jobs(
    gold_rows: list[dict[str, Any]],
    rewrite_items: dict[str, dict[str, Any]],
) -> list[tuple[str, str, str]]:
    jobs: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for config_id in CONFIGS[1:]:
        for gold in gold_rows:
            queries, _fallback = queries_for_config(
                config_id,
                gold["question"],
                rewrite_items.get(gold["question_id"]),
            )
            for query in queries:
                if config_id != "Q1_SINGLE_REWRITE_REPLACE" and query == gold["question"]:
                    continue
                job_id = _query_job_id(gold["question_id"], query)
                if job_id in seen:
                    continue
                seen.add(job_id)
                jobs.append((job_id, config_id, query))
    return jobs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-plan-only", action="store_true")
    args = parser.parse_args()
    write_plan()
    if args.write_plan_only:
        print(json.dumps({"status": "PLAN_WRITTEN", "path": str(PLAN_PATH)}))
        return 0
    plan = load_plan()
    gold_rows = load_dev_gold()
    if len(gold_rows) != 98 or any(row.get("split") != "dev" for row in gold_rows):
        raise RuntimeError("TEST_PROTOCOL_VIOLATION")
    q0_source_rows = read_jsonl(Q0_ITEMS)
    q0_by_id = {row["question_id"]: row for row in q0_source_rows}
    q0_rows = [
        add_extended_row_metrics(q0_by_id[gold["question_id"]], gold) for gold in gold_rows
    ]
    if len(q0_rows) != 98:
        raise RuntimeError(f"expected 98 Stage 2A Q0 rows, found {len(q0_rows)}")
    settings = get_settings()
    if settings.rerank_enabled:
        raise RuntimeError("Stage 2B requires RERANK_ENABLED=false")
    rewrite_items = generate_rewrites(gold_rows)
    write_jsonl(AUDIT_JSONL, public_rewrite_audit(rewrite_items))
    rewrite_by_id = {item["question_id"]: item for item in rewrite_items}
    q0_context_by_id = {row["question_id"]: row.get("ranked_results", []) for row in q0_rows}
    query_contexts = collect_query_contexts(build_query_jobs(gold_rows, rewrite_by_id))
    id_map = paper_id_map()
    experiment_rows: dict[str, list[dict[str, Any]]] = {"Q0_CURRENT_HYBRID": q0_rows}
    experiment_meta: dict[str, dict[str, Any]] = {
        "Q0_CURRENT_HYBRID": {"retrieval_call_count": 0, "q0_source": plan["q0_source"]}
    }
    for config_id in CONFIGS[1:]:
        rows, meta = run_candidate_rows(
            config_id,
            gold_rows,
            rewrite_by_id,
            id_map,
            q0_context_by_id,
            query_contexts,
        )
        experiment_rows[config_id] = rows
        experiment_meta[config_id] = meta
        write_jsonl(OPT_ROOT / f"{config_id}-items-v1.jsonl", rows)
    metrics = {
        config_id: extended_metrics(rows) for config_id, rows in experiment_rows.items()
    }
    for config_id, meta in experiment_meta.items():
        metrics[config_id].update(meta)
    rewrite_usage = aggregate_rewrite_usage(rewrite_items)
    specific = {
        config_id: rewrite_specific_metrics(q0_rows, rows, rewrite_items)
        for config_id, rows in experiment_rows.items()
        if config_id != "Q0_CURRENT_HYBRID"
    }
    paired = {
        config_id: paired_comparison(q0_rows, rows)
        for config_id, rows in experiment_rows.items()
        if config_id != "Q0_CURRENT_HYBRID"
    }
    bootstrap = {
        config_id: {
            metric: bootstrap_delta_ci(q0_rows, rows, metric=metric)
            for metric in (
                "recall_at_10",
                "mrr_at_10",
                "ndcg_at_10",
                "evidence_coverage_at_10",
                "required_claim_evidence_coverage_at_10",
            )
        }
        for config_id, rows in experiment_rows.items()
        if config_id != "Q0_CURRENT_HYBRID"
    }
    gate_results = {
        config_id: selection_gate(
            metrics["Q0_CURRENT_HYBRID"],
            metrics[config_id],
            specific[config_id],
        )
        and rewrite_usage["rewrite_success_rate"] >= 0.98
        for config_id in CONFIGS[1:]
    }
    selected = _select_candidate(metrics, gate_results)
    payload = {
        "schema_version": "stage2b-query-rewrite-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "benchmark_harness_commit": git_head(),
        "split": "dev",
        "dev_question_count": len(gold_rows),
        "test_questions_evaluated": 0,
        "test_protocol_violation": False,
        "selected_stage2a_candidate": "Current Hybrid",
        "q0_source": plan["q0_source"],
        "reranker_enabled": False,
        "llm_generation_requests": 0,
        "rewrite_provider": {
            "provider": settings.llm_provider_name or settings.llm_provider,
            "model": settings.llm_model,
            "temperature": settings.llm_temperature,
            "response_format": settings.llm_response_format,
            "stream": settings.llm_stream,
            "thinking_enabled": settings.llm_thinking_enabled,
        },
        "plan": plan,
        "metrics": metrics,
        "rewrite_usage": rewrite_usage,
        "rewrite_specific_metrics": specific,
        "by_category": {
            config_id: category_metrics(rows) for config_id, rows in experiment_rows.items()
        },
        "by_difficulty": {
            config_id: difficulty_metrics(rows) for config_id, rows in experiment_rows.items()
        },
        "paired_comparison_vs_q0": paired,
        "bootstrap_ci_vs_q0": bootstrap,
        "bad_case_delta_vs_q0": {
            config_id: bad_case_delta(q0_rows, rows)
            for config_id, rows in experiment_rows.items()
            if config_id != "Q0_CURRENT_HYBRID"
        },
        "selection_gate": gate_results,
        "selected_candidate": selected,
        "candidate_recommendation": (
            "QUERY_REWRITE_SELECTED" if selected != "Q0_CURRENT_HYBRID" else "KEEP_CURRENT_HYBRID"
        ),
        "gold_unchanged": True,
        "test_split_untouched": True,
        "production_defaults_changed": False,
        "lexical_reranker_production_default": False,
    }
    write_json(RESULT_JSON, payload)
    RESULT_MD.write_text(report_markdown(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "COMPLETED",
                "selected_candidate": selected,
                "test_questions_evaluated": 0,
                "rewrite_usage": rewrite_usage,
            },
            ensure_ascii=False,
        )
    )
    return 0


def public_rewrite_audit(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    safe = []
    for item in items:
        safe.append(
            {
                "question_id": item["question_id"],
                "single_rewrite": item.get("single_rewrite"),
                "decomposition_queries": item.get("decomposition_queries", []),
                "decomposition_purposes": item.get("decomposition_purposes", []),
                "provider": item.get("provider"),
                "model": item.get("model"),
                "single_status": item.get("single_status"),
                "decomposition_status": item.get("decomposition_status"),
                "single_prompt_hash": item.get("single_prompt_hash"),
                "decomposition_prompt_hash": item.get("decomposition_prompt_hash"),
                "input_tokens": item.get("input_tokens", 0),
                "output_tokens": item.get("output_tokens", 0),
                "total_tokens": item.get("total_tokens", 0),
                "estimated_cost_usd": item.get("estimated_cost_usd", 0.0),
                "usage_source": item.get("usage_source"),
                "single_cache_hit": item.get("single_cache_hit", False),
                "decomposition_cache_hit": item.get("decomposition_cache_hit", False),
                "single_cache_historical_provider_requests": item.get(
                    "single_cache_historical_provider_requests", 0
                ),
                "decomposition_cache_historical_provider_requests": item.get(
                    "decomposition_cache_historical_provider_requests", 0
                ),
                "drift_labels": item.get("drift_labels", []),
            }
        )
    return safe


def _select_candidate(metrics: dict[str, Any], gate_results: dict[str, bool]) -> str:
    passing = [config_id for config_id, passed in gate_results.items() if passed]
    if not passing:
        return "Q0_CURRENT_HYBRID"
    return max(
        passing,
        key=lambda config_id: (
            metrics[config_id].get("evidence_coverage_at_10") or 0.0,
            metrics[config_id].get("required_claim_evidence_coverage_at_10") or 0.0,
            metrics[config_id].get("recall_at_10") or 0.0,
        ),
    )


def report_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Stage 2B query rewrite / decomposition v1",
        "",
        f"- split: `{payload['split']}`",
        f"- dev_question_count: `{payload['dev_question_count']}`",
        f"- test_questions_evaluated: `{payload['test_questions_evaluated']}`",
        f"- test_protocol_violation: `{payload['test_protocol_violation']}`",
        f"- selected_stage2a_candidate: `{payload['selected_stage2a_candidate']}`",
        f"- selected_candidate: `{payload['selected_candidate']}`",
        f"- reranker_enabled: `{payload['reranker_enabled']}`",
        f"- llm_generation_requests: `{payload['llm_generation_requests']}`",
        "",
        "## Metrics",
        "",
        "| Config | Recall@10 | EvidenceCov@10 | ReqClaimCov@10 | FullCov@10 | "
        "MRR@10 | nDCG@10 | PaperRecall@10 | P95 latency |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for config_id in CONFIGS:
        metric = payload["metrics"][config_id]
        latency = metric.get("latency_ms") or {}
        lines.append(
            f"| {config_id} | {metric.get('recall_at_10')} | "
            f"{metric.get('evidence_coverage_at_10')} | "
            f"{metric.get('required_claim_evidence_coverage_at_10')} | "
            f"{metric.get('full_evidence_coverage_at_10')} | {metric.get('mrr_at_10')} | "
            f"{metric.get('ndcg_at_10')} | {metric.get('paper_recall_at_10')} | "
            f"{latency.get('p95')} |"
        )
    lines.extend(
        [
            "",
            "## Rewrite accounting",
            "",
            "```json",
            json.dumps(payload["rewrite_usage"], ensure_ascii=False, indent=2),
            "```",
            "",
            "## Selection gate",
            "",
            "```json",
            json.dumps(payload["selection_gate"], ensure_ascii=False, indent=2),
            "```",
            "",
            "## Interpretation",
            "",
            (
                "Q0 Current Hybrid remains selected because no rewrite/decomposition candidate "
                "met the pre-registered gate."
                if payload["selected_candidate"] == "Q0_CURRENT_HYBRID"
                else f"`{payload['selected_candidate']}` met the pre-registered gate."
            ),
            "",
            "This DEV-only experiment did not evaluate TEST, did not invoke QA generation, "
            "and did not enable reranking.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
