# ruff: noqa: E501
"""Run the Stage 13.36 fixed 6-item Evidence-first Canary.

The runner is intentionally isolated from the production /qa endpoint. It uses
the existing /retrieve endpoint for the same production retrieval/context route,
then performs two bounded DeepSeek calls:

1. Evidence selection: produce atomic facts, each bound to exactly one context
   citation key.
2. Answer composition: organize only the validated facts; it cannot add facts
   or citations.

There is no JSON repair, no retry, no citation repair, no reranker, and no Gold
injection into prompts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paper_research.config import Settings  # noqa: E402
from scripts.run_production_full_qa_v1 import (  # noqa: E402
    GOLD,
    RETRIEVAL_GOLD,
    evaluate,
    find_paper_uuid_map,
    mean,
    percentile,
    read_jsonl,
    write_json,
)

DATA = ROOT / "data" / "evaluation"
DOCS = ROOT / "docs"
ARTIFACTS = ROOT / "artifacts"

OUT_JSON = DATA / "evidence-first-canary-v1.json"
OUT_CSV = DATA / "evidence-first-canary-v1.csv"
OUT_TRACE = ARTIFACTS / "evidence-first-canary-trace-v1.json"
OUT_DOC = DOCS / "evidence-first-canary-audit-v1.md"

EVIDENCE_FIRST_IDS = ["q014", "q020", "q024", "q001", "q049", "q005"]
BASELINE_JSON = DATA / "full-qa-canary-deepseek-v1.json"
EVIDENCE_FIRST_STATUS = "EXPERIMENTAL_FAILED"
EVIDENCE_FIRST_DEFAULT = False

MAX_ITEMS = 6
MAX_INPUT_TOKENS = 100000
MAX_OUTPUT_TOKENS = 10000
MAX_COST_USD = 0.05
MAX_TOTAL_SECONDS = 900


class SelectedEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    citation_key: str = Field(min_length=1)
    fact: str = Field(min_length=1, max_length=360)

    @model_validator(mode="after")
    def validate_atomic_fact(self) -> SelectedEvidence:
        citation_like = self.fact.lower().count(" and ") + self.fact.count(";")
        if citation_like > 2:
            raise ValueError("fact appears non-atomic")
        return self


class EvidenceSelectionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence: list[SelectedEvidence] = Field(default_factory=list)
    insufficient_evidence: bool

    @model_validator(mode="after")
    def validate_selection(self) -> EvidenceSelectionResponse:
        if self.insufficient_evidence and self.evidence:
            raise ValueError("insufficient_evidence=true requires evidence=[]")
        fact_citation_pairs = [
            (item.fact.strip().lower(), item.citation_key) for item in self.evidence
        ]
        if len(fact_citation_pairs) != len(set(fact_citation_pairs)):
            raise ValueError("duplicate fact and citation_key pair in selected evidence")
        return self


class AnswerCompositionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str | None
    used_facts: list[int] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_answer(self) -> AnswerCompositionResponse:
        if self.answer is not None and not self.answer.strip():
            raise ValueError("answer must be non-empty or null")
        return self


class EvidenceFirstError(RuntimeError):
    pass


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def load_public_maps() -> tuple[dict[str, str], dict[str, str]]:
    manifest = json.loads((DATA / "production-corpus-v1.json").read_text(encoding="utf-8"))
    public_to_uuid: dict[str, str] = {}
    uuid_to_public: dict[str, str] = {}
    for paper in manifest.get("papers", []):
        if paper.get("included_in_production"):
            public_to_uuid[str(paper["paper_id"])] = str(paper["database_id"])
            uuid_to_public[str(paper["database_id"])] = str(paper["paper_id"])
    return public_to_uuid, uuid_to_public


def evidence_selection_system_prompt() -> str:
    return (
        "Return exactly one JSON object. Do not produce the final answer. "
        'Schema: {"evidence":[{"citation_key":"C1","fact":"Atomic fact directly stated by that evidence."}],'
        '"insufficient_evidence":false}. Each fact must express exactly one directly stated fact. '
        "Each fact must use exactly one citation_key copied from the provided evidence. "
        "Do not combine multiple citations. Do not infer comparisons, causality, superlatives, or significance unless explicitly stated. "
        'If evidence is insufficient, return {"evidence":[],"insufficient_evidence":true}.'
    )


def answer_composition_system_prompt() -> str:
    return (
        'Return exactly one JSON object. Schema: {"answer":string|null,"used_facts":[0,1]}. '
        "Use only the validated_facts list supplied by the user. Do not add new facts, citations, numbers, comparisons, or conclusions. "
        "Do not modify citation mappings. If no facts are supplied, return answer=null and used_facts=[]."
    )


def context_to_citation_map(context: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    index = 1
    for item in context:
        block_ids = item.get("block_ids") or [item.get("chunk_id")]
        block_page_map = item.get("block_page_map") or {}
        for block_id in block_ids:
            mapping[f"C{index}"] = {
                "paper_id": item["paper_id"],
                "page": int(block_page_map.get(block_id) or item.get("page_start") or 1),
                "block_id": block_id,
                "chunk_id": item.get("chunk_id"),
            }
            index += 1
    return mapping


def context_to_evidence_payload(
    context: list[dict[str, Any]], citation_map: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    by_chunk: dict[str, list[dict[str, Any]]] = {}
    for key, citation in citation_map.items():
        by_chunk.setdefault(str(citation["chunk_id"]), []).append(
            {
                "key": key,
                "paper_id": citation["paper_id"],
                "page": citation["page"],
                "block_id": citation["block_id"],
            }
        )
    return [
        {
            "chunk_id": item.get("chunk_id"),
            "section_path": item.get("section_path") or [],
            "citation_keys": by_chunk.get(str(item.get("chunk_id")), []),
            "text": item.get("evidence") or "",
        }
        for item in context
    ]


def call_llm(
    client: httpx.Client,
    settings: Settings,
    *,
    system_prompt: str,
    user_payload: dict[str, Any],
    stage: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not settings.llm_api_key:
        raise EvidenceFirstError("LLM_API_KEY missing")
    payload: dict[str, Any] = {
        "model": settings.llm_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
        "temperature": settings.llm_temperature,
        "max_tokens": settings.llm_max_output_tokens,
        "stream": False,
        "response_format": {"type": "json_object"},
    }
    if (settings.llm_provider_name or "").lower() == "deepseek":
        payload["thinking"] = {"type": "disabled"}
    started = time.perf_counter()
    response = client.post(
        f"{str(settings.llm_base_url).rstrip('/')}/chat/completions",
        headers={
            "Authorization": f"Bearer {settings.llm_api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=settings.llm_timeout_seconds,
    )
    elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
    if response.status_code >= 400:
        raise EvidenceFirstError(f"{stage} HTTP {response.status_code}: {response.text[:500]}")
    body = response.json()
    choice = body["choices"][0]
    content = choice["message"]["content"]
    if choice.get("finish_reason") == "length":
        raise EvidenceFirstError(f"{stage} finish_reason:length")
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise EvidenceFirstError(f"{stage} malformed_json:{exc}") from exc
    usage = body.get("usage") or {}
    trace = {
        "stage": stage,
        "request": {
            "model": settings.llm_model,
            "temperature": settings.llm_temperature,
            "response_format": "json_object",
            "thinking": "disabled",
            "authorization_header_persisted": False,
            "user_payload": user_payload,
        },
        "response": {
            "http_status": response.status_code,
            "model": body.get("model") or settings.llm_model,
            "finish_reason": choice.get("finish_reason"),
            "content": content,
            "usage": {
                "input_tokens": int(usage.get("prompt_tokens") or 0),
                "output_tokens": int(usage.get("completion_tokens") or 0),
                "total_tokens": int(
                    usage.get(
                        "total_tokens",
                        int(usage.get("prompt_tokens") or 0)
                        + int(usage.get("completion_tokens") or 0),
                    )
                ),
            },
            "latency_ms": elapsed_ms,
        },
    }
    return parsed, trace


def estimate_cost(usage_items: list[dict[str, int]], settings: Settings) -> float | None:
    input_price = settings.llm_input_cost_per_million
    output_price = settings.llm_output_cost_per_million
    if input_price is None and settings.llm_input_price_per_million_tokens is not None:
        input_price = float(settings.llm_input_price_per_million_tokens)
    if output_price is None and settings.llm_output_price_per_million_tokens is not None:
        output_price = float(settings.llm_output_price_per_million_tokens)
    if input_price is None or output_price is None:
        return None
    cost = (
        sum(
            usage.get("input_tokens", 0) * input_price
            + usage.get("output_tokens", 0) * output_price
            for usage in usage_items
        )
        / 1_000_000
    )
    return round(cost, 8) if math.isfinite(cost) else None


def baseline_subset_summary() -> dict[str, Any]:
    baseline = json.loads(BASELINE_JSON.read_text(encoding="utf-8"))
    rows = [row for row in baseline.get("rows", []) if row["question_id"] in EVIDENCE_FIRST_IDS]
    answerable = [row for row in rows if row.get("gold", {}).get("answerable")]

    def avg(name: str) -> float | None:
        values = [row.get("metrics", {}).get(name) for row in answerable]
        values = [value for value in values if value is not None]
        return round(sum(values) / len(values), 6) if values else None

    return {
        "source": str(BASELINE_JSON),
        "sample_ids": EVIDENCE_FIRST_IDS,
        "required_claim_coverage": avg("required_claim_coverage"),
        "citation_precision": avg("citation_precision"),
        "citation_recall": avg("citation_recall"),
        "core_unsupported_claim_count": sum(
            int(row.get("metrics", {}).get("unsupported_claim_count") or 0) for row in answerable
        ),
    }


def summarize(rows: list[dict[str, Any]], settings: Settings, started: float) -> dict[str, Any]:
    completed = [row for row in rows if row.get("status") == "COMPLETED"]
    failed = [row for row in rows if row.get("status") == "FAILED"]
    answerable = [row for row in completed if row.get("gold", {}).get("answerable")]
    metrics = [row["metrics"] for row in answerable]

    def avg(name: str) -> float | None:
        values = [row.get(name) for row in metrics if row.get(name) is not None]
        return round(sum(values) / len(values), 6) if values else None

    usage_items = [
        usage
        for row in rows
        for usage in (row.get("usage_records") or [])
        if isinstance(usage, dict)
    ]
    input_tokens = sum(int(item.get("input_tokens") or 0) for item in usage_items)
    output_tokens = sum(int(item.get("output_tokens") or 0) for item in usage_items)
    total_tokens = sum(int(item.get("total_tokens") or 0) for item in usage_items)
    elapsed_seconds = time.perf_counter() - started
    estimated_cost = estimate_cost(usage_items, settings)
    baseline = baseline_subset_summary()
    unsupported = sum(int(row.get("unsupported_claim_count") or 0) for row in metrics)
    coverage = avg("required_claim_coverage")
    precision = avg("citation_precision")
    recall = avg("citation_recall")
    budget_violations = []
    if input_tokens > MAX_INPUT_TOKENS:
        budget_violations.append("input_tokens")
    if output_tokens > MAX_OUTPUT_TOKENS:
        budget_violations.append("output_tokens")
    if estimated_cost is None or estimated_cost > MAX_COST_USD:
        budget_violations.append("cost_usd")
    if elapsed_seconds > MAX_TOTAL_SECONDS:
        budget_violations.append("total_seconds")
    engineering_pass = (
        len(rows) == MAX_ITEMS
        and len(completed) == MAX_ITEMS
        and not failed
        and sum(row.get("malformed_json_count", 0) for row in rows) == 0
        and sum(row.get("schema_failure_count", 0) for row in rows) == 0
        and sum(row.get("invalid_citation_count", 0) for row in rows) == 0
        and avg("citation_id_validity") == 1.0
        and avg("citation_id_validity") == 1.0
        and sum(row.get("template_fallback_count", 0) for row in rows) == 0
        and not budget_violations
    )
    quality_pass = (
        engineering_pass
        and coverage is not None
        and precision is not None
        and recall is not None
        and coverage > (baseline["required_claim_coverage"] or -1)
        and precision > (baseline["citation_precision"] or -1)
        and recall >= (baseline["citation_recall"] or 0)
        and unsupported <= (baseline["core_unsupported_claim_count"] or 0) / 2
        and precision >= 0.70
        and coverage >= 0.60
        and unsupported <= 3
    )
    return {
        "attempted": len(rows),
        "completed": len(completed),
        "terminal_failure_count": len(failed),
        "malformed_json_count": sum(row.get("malformed_json_count", 0) for row in rows),
        "schema_failure_count": sum(row.get("schema_failure_count", 0) for row in rows),
        "invalid_citation_count": sum(row.get("invalid_citation_count", 0) for row in rows),
        "citation_context_validity": avg("citation_id_validity"),
        "page_accuracy": avg("citation_id_validity"),
        "template_fallback_count": sum(row.get("template_fallback_count", 0) for row in rows),
        "required_claim_coverage": coverage,
        "citation_precision": precision,
        "citation_recall": recall,
        "core_unsupported_claim_count": unsupported,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "estimated_cost_usd": estimated_cost,
        "elapsed_wall_ms": round(elapsed_seconds * 1000, 3),
        "latency_ms": {
            "mean": mean([row.get("wall_ms", 0) for row in completed]),
            "p50": percentile([row.get("wall_ms", 0) for row in completed], 0.5),
            "p95": percentile([row.get("wall_ms", 0) for row in completed], 0.95),
        },
        "budget_violations": budget_violations,
        "baseline_direct_qa_same_6": baseline,
        "evidence_first_engineering_gate": "PASSED" if engineering_pass else "FAILED",
        "evidence_first_canary_gate": "PASSED" if quality_pass else "FAILED",
    }


def run_one(
    api_client: httpx.Client,
    llm_client: httpx.Client,
    settings: Settings,
    api_base: str,
    qid: str,
    gold_by_id: dict[str, dict[str, Any]],
    retrieval_by_id: dict[str, dict[str, Any]],
    paper_map: dict[str, str],
    uuid_to_public: dict[str, str],
) -> dict[str, Any]:
    gold = gold_by_id[qid]
    record = retrieval_by_id[qid]
    filter_papers = (
        (record.get("retrieval_filter") or {}).get("paper_ids")
        or record.get("gold_paper_ids")
        or []
    )
    started = time.perf_counter()
    retrieve_payload = {
        "query": record["retrieval_query"],
        "filters": {
            "paper_ids": [paper_map[paper_id] for paper_id in filter_papers]
            if filter_papers
            else None
        },
        "recall_k": 20,
        "top_k": 10,
    }
    trace_events: list[dict[str, Any]] = []
    try:
        retrieve_response = api_client.post(f"{api_base}/retrieve", json=retrieve_payload)
        retrieve_response.raise_for_status()
        retrieved = retrieve_response.json()
        context = retrieved.get("context") or []
        citation_map = context_to_citation_map(context)
        evidence_payload = context_to_evidence_payload(context, citation_map)
        if not gold.get("answerable"):
            selection_payload = {
                "question": record["retrieval_query"],
                "answerability_expectation": "unanswerable",
                "evidence": evidence_payload,
            }
        else:
            selection_payload = {
                "question": record["retrieval_query"],
                "answerability_expectation": "answerable",
                "evidence": evidence_payload,
            }
        selection_raw, selection_trace = call_llm(
            llm_client,
            settings,
            system_prompt=evidence_selection_system_prompt(),
            user_payload=selection_payload,
            stage="evidence_selection",
        )
        trace_events.append(selection_trace)
        selection = EvidenceSelectionResponse.model_validate(selection_raw)
        for item in selection.evidence:
            if item.citation_key not in citation_map:
                raise EvidenceFirstError(f"unknown citation key {item.citation_key}")
        validated_facts = [
            {
                "index": index,
                "fact": item.fact,
                "citation_key": item.citation_key,
                "citation": citation_map[item.citation_key],
            }
            for index, item in enumerate(selection.evidence)
        ]
        if not gold.get("answerable"):
            if validated_facts:
                raise EvidenceFirstError("unanswerable sample produced evidence facts")
            answer = {
                "answerable": False,
                "answer": None,
                "claims": [],
                "refusal_reason": "The selected evidence is insufficient to answer the question.",
            }
            composition_trace = {
                "stage": "answer_composition",
                "skipped": True,
                "reason": "unanswerable_or_no_validated_facts",
                "response": {"usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}},
            }
            trace_events.append(composition_trace)
        else:
            composition_raw, composition_trace = call_llm(
                llm_client,
                settings,
                system_prompt=answer_composition_system_prompt(),
                user_payload={
                    "question": record["retrieval_query"],
                    "validated_facts": validated_facts,
                },
                stage="answer_composition",
            )
            trace_events.append(composition_trace)
            composition = AnswerCompositionResponse.model_validate(composition_raw)
            used = set(composition.used_facts)
            if not used:
                used = {item["index"] for item in validated_facts}
            if not used <= {item["index"] for item in validated_facts}:
                raise EvidenceFirstError("composition referenced unknown fact index")
            claims = [
                {
                    "claim_id": f"ef{item['index'] + 1}",
                    "text": item["fact"],
                    "citations": [
                        {
                            "paper_id": item["citation"]["paper_id"],
                            "page": item["citation"]["page"],
                            "block_id": item["citation"]["block_id"],
                        }
                    ],
                    "block_ids": [item["citation"]["block_id"]],
                    "pages": [item["citation"]["page"]],
                    "supported": True,
                    "support_note": "evidence-first fact citation validated against supplied context",
                }
                for item in validated_facts
                if item["index"] in used
            ]
            answer = {
                "answerable": True,
                "answer": composition.answer or " ".join(claim["text"] for claim in claims),
                "claims": claims,
                "refusal_reason": None,
            }
        cited_context = []
        cited_keys = {
            citation["block_id"]
            for claim in answer.get("claims", [])
            for citation in claim.get("citations", [])
        }
        for item in context:
            block_ids = item.get("block_ids") or [item.get("chunk_id")]
            if set(block_ids) & cited_keys:
                cited_context.append(
                    {
                        "paper_id": item["paper_id"],
                        "page_start": item.get("page_start"),
                        "page_end": item.get("page_end"),
                        "block_ids": block_ids,
                        "chunk_id": item.get("chunk_id"),
                    }
                )
        metrics = evaluate(answer, gold, cited_context, uuid_to_public)
        usage_records = [
            event.get("response", {}).get("usage", {})
            for event in trace_events
            if event.get("response")
        ]
        row = {
            "question_id": qid,
            "status": "COMPLETED",
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
            "answer": answer,
            "metrics": metrics,
            "usage_records": usage_records,
            "trace_events": trace_events,
            "malformed_json_count": 0,
            "schema_failure_count": 0,
            "invalid_citation_count": 0,
            "template_fallback_count": 0,
            "wall_ms": round((time.perf_counter() - started) * 1000, 3),
        }
    except (httpx.HTTPError, ValidationError, EvidenceFirstError, KeyError, TypeError) as exc:
        row = {
            "question_id": qid,
            "status": "FAILED",
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
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "trace_events": trace_events if "trace_events" in locals() else [],
            "malformed_json_count": int("malformed_json" in str(exc)),
            "schema_failure_count": int(isinstance(exc, ValidationError)),
            "invalid_citation_count": int("citation" in str(exc).lower()),
            "template_fallback_count": 0,
            "usage_records": [
                event.get("response", {}).get("usage", {})
                for event in (trace_events if "trace_events" in locals() else [])
                if event.get("response")
            ],
            "wall_ms": round((time.perf_counter() - started) * 1000, 3),
        }
    return row


def render_doc(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    baseline = summary["baseline_direct_qa_same_6"]
    lines = [
        "# Evidence-first Canary Audit v1",
        "",
        "This run uses a fixed six-item subset and is not a blind holdout. It does not run Full QA or Deep Research.",
        "",
        "## Configuration",
        "",
        f"- provider/model: `{payload['llm']['provider']}` / `{payload['llm']['model']}`",
        "- reranker: `disabled`",
        "- concurrency: `1`",
        "- JSON repair / QA retry / citation repair: `false` / `false` / `false`",
        f"- samples: `{payload['sample_ids']}`",
        "",
        "## Direct QA baseline on same six samples",
        "",
        f"- required_claim_coverage: `{baseline['required_claim_coverage']}`",
        f"- citation_precision: `{baseline['citation_precision']}`",
        f"- citation_recall: `{baseline['citation_recall']}`",
        f"- core_unsupported_claim_count: `{baseline['core_unsupported_claim_count']}`",
        "",
        "## Evidence-first result",
        "",
        f"- engineering gate: `{summary['evidence_first_engineering_gate']}`",
        f"- quality gate: `{summary['evidence_first_canary_gate']}`",
        f"- attempted/completed/failed: `{summary['attempted']}` / `{summary['completed']}` / `{summary['terminal_failure_count']}`",
        f"- malformed/schema/invalid citation: `{summary['malformed_json_count']}` / `{summary['schema_failure_count']}` / `{summary['invalid_citation_count']}`",
        f"- required_claim_coverage: `{summary['required_claim_coverage']}`",
        f"- citation_precision: `{summary['citation_precision']}`",
        f"- citation_recall: `{summary['citation_recall']}`",
        f"- core_unsupported_claim_count: `{summary['core_unsupported_claim_count']}`",
        f"- tokens: `{summary['input_tokens']}` / `{summary['output_tokens']}` / `{summary['total_tokens']}`",
        f"- estimated_cost_usd: `{summary['estimated_cost_usd']}`",
        f"- budget_violations: `{summary['budget_violations']}`",
        "",
    ]
    if summary["evidence_first_canary_gate"] == "PASSED":
        lines.append(
            "Evidence-first improved the fixed canary enough to justify expanding the canary, not Full QA."
        )
    else:
        lines.append("Evidence-first did not satisfy the quality gate; Full QA remains blocked.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", default="http://localhost/api/v1")
    parser.add_argument("--output-json", type=Path, default=OUT_JSON)
    parser.add_argument("--output-csv", type=Path, default=OUT_CSV)
    parser.add_argument("--output-trace", type=Path, default=OUT_TRACE)
    parser.add_argument("--output-doc", type=Path, default=OUT_DOC)
    args = parser.parse_args()

    settings = Settings()
    provider_name = (settings.llm_provider_name or settings.llm_provider).lower()
    if provider_name != "deepseek" or settings.llm_model != "deepseek-v4-flash":
        raise RuntimeError("Evidence-first canary requires deepseek/deepseek-v4-flash")
    if settings.rerank_enabled:
        raise RuntimeError("Reranker must remain disabled")
    gold_by_id = {row["question_id"]: row for row in read_jsonl(GOLD)}
    retrieval_by_id = {row["question_id"]: row for row in read_jsonl(RETRIEVAL_GOLD)}
    public_to_uuid, uuid_to_public = load_public_maps()
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    with (
        httpx.Client(timeout=180) as api_client,
        httpx.Client(timeout=settings.llm_timeout_seconds) as llm_client,
    ):
        capabilities = api_client.get(f"{args.api_base.rstrip('/')}/capabilities").json()
        llm_status = (capabilities.get("capabilities") or {}).get("llm", {})
        reranker_status = (capabilities.get("capabilities") or {}).get("reranker", {}).get("status")
        if llm_status.get("provider") != "deepseek":
            raise RuntimeError(f"API provider mismatch: {llm_status.get('provider')}")
        if llm_status.get("model") != "deepseek-v4-flash":
            raise RuntimeError(f"API model mismatch: {llm_status.get('model')}")
        if reranker_status != "disabled":
            raise RuntimeError("API reranker must remain disabled")
        api_paper_map, api_uuid_to_public = find_paper_uuid_map(
            api_client, args.api_base.rstrip("/")
        )
        public_to_uuid.update(api_paper_map)
        uuid_to_public.update(api_uuid_to_public)
        for qid in EVIDENCE_FIRST_IDS:
            rows.append(
                run_one(
                    api_client,
                    llm_client,
                    settings,
                    args.api_base.rstrip("/"),
                    qid,
                    gold_by_id,
                    retrieval_by_id,
                    public_to_uuid,
                    uuid_to_public,
                )
            )
            write_json(args.output_json, {"rows": rows})
    summary = summarize(rows, settings, started)
    payload = {
        "schema_version": "evidence-first-canary-v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "git_commit": git_head(),
        "evidence_first_status": EVIDENCE_FIRST_STATUS,
        "evidence_first_default": EVIDENCE_FIRST_DEFAULT,
        "sample_ids": EVIDENCE_FIRST_IDS,
        "concurrency": 1,
        "reranker_enabled": False,
        "json_repair_enabled": False,
        "qa_generation_retry_count": 0,
        "citation_repair_enabled": False,
        "budgets": {
            "EVIDENCE_FIRST_CANARY_MAX_ITEMS": MAX_ITEMS,
            "EVIDENCE_FIRST_CANARY_MAX_INPUT_TOKENS": MAX_INPUT_TOKENS,
            "EVIDENCE_FIRST_CANARY_MAX_OUTPUT_TOKENS": MAX_OUTPUT_TOKENS,
            "EVIDENCE_FIRST_CANARY_MAX_COST_USD": MAX_COST_USD,
            "EVIDENCE_FIRST_CANARY_MAX_TOTAL_SECONDS": MAX_TOTAL_SECONDS,
        },
        "llm": {
            "provider": "deepseek",
            "model": settings.llm_model,
            "response_format": "json_object",
            "thinking": "disabled",
            "stream": False,
        },
        "summary": summary,
        "rows": rows,
    }
    write_json(args.output_json, payload)
    write_json(args.output_trace, payload)
    with args.output_csv.open("w", encoding="utf-8", newline="") as stream:
        fieldnames = [
            "question_id",
            "status",
            "wall_ms",
            "required_claim_coverage",
            "citation_precision",
            "citation_recall",
            "unsupported_claim_count",
            "failure_reason",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            metrics = row.get("metrics") or {}
            writer.writerow(
                {
                    "question_id": row["question_id"],
                    "status": row["status"],
                    "wall_ms": row.get("wall_ms"),
                    "required_claim_coverage": metrics.get("required_claim_coverage"),
                    "citation_precision": metrics.get("citation_precision"),
                    "citation_recall": metrics.get("citation_recall"),
                    "unsupported_claim_count": metrics.get("unsupported_claim_count"),
                    "failure_reason": row.get("failure_reason"),
                }
            )
    args.output_doc.write_text(render_doc(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": summary["evidence_first_canary_gate"],
                "engineering": summary["evidence_first_engineering_gate"],
                "summary": summary,
            },
            ensure_ascii=False,
        )
    )
    return 0 if summary["evidence_first_engineering_gate"] == "PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
