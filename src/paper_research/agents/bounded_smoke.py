"""Bounded, resumable engineering smoke runner for Stage 11D.

This module is intentionally isolated from the full research graph.  It proves
budget, checkpoint, resume, trace and strict citation invariants; it is not a
quality evaluation path.
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal, Protocol

from paper_research.config import Settings
from paper_research.providers.llm import (
    GenerationResult,
    LLMProviderError,
    SiliconFlowLLMProvider,
)
from paper_research.retrieval.context_builder import ContextItem

BillingMode = Literal["paid", "free", "local"]
UsageSource = Literal[
    "provider_reported",
    "tokenizer_estimated",
    "unavailable_not_sent",
    "unavailable_after_send_attempt",
    "reserved_conservative",
]
NODES = (
    "plan",
    "retrieve",
    "assess_evidence",
    "optional_refine",
    "synthesize",
    "validate_citations",
    "persist_trace",
)


class SmokeConfigurationError(ValueError):
    pass


class BudgetBlocked(RuntimeError):
    pass


class ElapsedTimeBlocked(BudgetBlocked):
    pass


class ProviderFailed(RuntimeError):
    pass


class SmokeLLM(Protocol):
    provider_name: str
    model_name: str

    def generate_claim_answer(
        self,
        question: str,
        context: list[ContextItem],
        prompt_version: str,
        audit_metadata: dict[str, Any] | None = None,
    ) -> GenerationResult: ...


@dataclass(frozen=True)
class BillingPolicy:
    mode: BillingMode
    input_price: Decimal
    output_price: Decimal
    max_cost: Decimal
    warning: str | None = None

    @property
    def cost_basis(self) -> str:
        return {
            "paid": "configured_token_prices",
            "free": "explicit_free_provider",
            "local": "local_inference_excludes_hardware_cost",
        }[self.mode]

    def cost(self, input_tokens: int, output_tokens: int) -> Decimal:
        if self.mode != "paid":
            return Decimal("0")
        million = Decimal(1_000_000)
        return (
            Decimal(input_tokens) * self.input_price
            + Decimal(output_tokens) * self.output_price
        ) / million


@dataclass(frozen=True)
class SmokeLimits:
    max_queries: int
    iterations_per_query: int
    requests_per_query: int
    requests_total: int
    tokens_per_query: int
    tokens_total: int
    elapsed_per_query: int
    elapsed_total: int


def smoke_configuration(settings: Settings) -> tuple[BillingPolicy, SmokeLimits]:
    if settings.deep_research_mode != "engineering_smoke":
        raise SmokeConfigurationError("DEEP_RESEARCH_MODE must be engineering_smoke")
    if settings.rerank_enabled:
        raise SmokeConfigurationError("RERANK_ENABLED must be false")
    if settings.embedding_provider != "jina" or settings.embedding_model != (
        "jina-embeddings-v5-text-small"
    ):
        raise SmokeConfigurationError("fixed Jina embedding configuration is required")
    provider = settings.llm_provider.strip().lower()
    provider_name = (settings.llm_provider_name or settings.llm_provider).strip().lower()
    allowed_live_models = {
        ("siliconflow", "siliconflow", "Qwen/Qwen3-8B"),
        ("openai_compatible", "deepseek", "deepseek-v4-flash"),
    }
    if (provider, provider_name, settings.llm_model) not in allowed_live_models:
        raise SmokeConfigurationError(
            "fixed SiliconFlow Qwen/Qwen3-8B or DeepSeek deepseek-v4-flash is required"
        )
    mode = settings.llm_billing_mode
    if mode not in {"paid", "free", "local"}:
        raise SmokeConfigurationError("LLM_BILLING_MODE must be paid, free, or local")
    input_price = settings.llm_input_price_per_million_tokens
    output_price = settings.llm_output_price_per_million_tokens
    max_cost = settings.deep_research_max_cost_usd
    if input_price is None or output_price is None or max_cost is None:
        raise SmokeConfigurationError("billing prices and DEEP_RESEARCH_MAX_COST_USD are required")
    if not all(value.is_finite() and value >= 0 for value in (input_price, output_price, max_cost)):
        raise SmokeConfigurationError("billing values must be non-negative finite decimals")
    warning = None
    if mode == "paid":
        if max_cost <= 0:
            raise SmokeConfigurationError("paid mode requires DEEP_RESEARCH_MAX_COST_USD > 0")
        if input_price == 0 and output_price == 0:
            warning = "paid mode has suspicious zero input and output prices"
    elif input_price != 0 or output_price != 0 or max_cost != 0:
        raise SmokeConfigurationError(f"{mode} mode requires exact zero prices and max cost")
    policy = BillingPolicy(mode, input_price, output_price, max_cost, warning)
    limits = SmokeLimits(
        settings.deep_research_max_queries,
        settings.deep_research_max_iterations_per_query,
        settings.deep_research_max_llm_requests_per_query,
        settings.deep_research_max_llm_requests_total,
        settings.deep_research_max_tokens_per_query,
        settings.deep_research_max_tokens_total,
        settings.deep_research_max_elapsed_seconds_per_query,
        settings.deep_research_max_elapsed_seconds_total,
    )
    return policy, limits


@dataclass
class UsageRecord:
    request_id: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    usage_source: UsageSource
    monetary_cost_usd: str
    cost_basis: str


@dataclass
class SmokeState:
    run_id: str
    question_id: str
    question: str
    retrieval_scope: str
    retrieval_filter: dict[str, Any]
    current_node: str = "plan"
    status: str = "running"
    nodes_visited: list[str] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    contexts: list[dict[str, Any]] = field(default_factory=list)
    iteration_count: int = 0
    retrieval_calls: int = 0
    llm_requests: int = 0
    request_attempt_count: int = 0
    provider_completed_request_count: int = 0
    usage_record_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    monetary_cost_usd: str = "0"
    usage_records: list[dict[str, Any]] = field(default_factory=list)
    request_records: list[dict[str, Any]] = field(default_factory=list)
    reserved_input_tokens: int = 0
    reserved_output_tokens: int = 0
    reserved_total_tokens: int = 0
    budget_accounting_status: str = "settled"
    answer: dict[str, Any] | None = None
    citation_validation: str = "not_run"
    errors: list[str] = field(default_factory=list)
    budget_stop_reason: str | None = None
    resume_count: int = 0
    started_at: float = field(default_factory=time.time)
    elapsed_seconds: float = 0.0


class SQLiteSmokeCheckpoint:
    """One atomic row per run; node commits make resume idempotent."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS smoke_runs "
                "(run_id TEXT PRIMARY KEY, state_json TEXT NOT NULL, updated_at REAL NOT NULL)"
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def save(self, state: SmokeState) -> None:
        payload = json.dumps(asdict(state), ensure_ascii=False, sort_keys=True)
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO smoke_runs(run_id,state_json,updated_at) VALUES(?,?,?) "
                "ON CONFLICT(run_id) DO UPDATE SET state_json=excluded.state_json, "
                "updated_at=excluded.updated_at",
                (state.run_id, payload, time.time()),
            )

    def load(self, run_id: str) -> SmokeState | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT state_json FROM smoke_runs WHERE run_id=?", (run_id,)
            ).fetchone()
        return SmokeState(**json.loads(row[0])) if row else None

    def load_many(self, run_ids: list[str]) -> list[SmokeState]:
        return [state for run_id in run_ids if (state := self.load(run_id)) is not None]

    def load_all(self) -> list[SmokeState]:
        with self._connect() as connection:
            rows = connection.execute("SELECT state_json FROM smoke_runs").fetchall()
        return [SmokeState(**json.loads(row[0])) for row in rows]


class BudgetGuard:
    def __init__(self, policy: BillingPolicy, limits: SmokeLimits) -> None:
        self.policy = policy
        self.limits = limits
        self.global_requests = 0
        self.global_tokens = 0
        self.global_reserved_tokens = 0
        self.global_cost = Decimal("0")
        self.started = time.monotonic()

    def restore(self, states: list[SmokeState]) -> None:
        self.global_requests = sum(
            state.request_attempt_count or state.llm_requests for state in states
        )
        self.global_tokens = sum(state.total_tokens for state in states)
        self.global_reserved_tokens = sum(state.reserved_total_tokens for state in states)
        self.global_cost = sum(
            (Decimal(state.monetary_cost_usd) for state in states), Decimal("0")
        )
        if states:
            prior_elapsed = sum(state.elapsed_seconds for state in states)
            self.started = time.monotonic() - prior_elapsed

    def preflight(self, state: SmokeState, estimated_input: int, max_output: int) -> None:
        projected_tokens = estimated_input + max_output
        elapsed_query = time.time() - state.started_at
        attempts = state.request_attempt_count or state.llm_requests
        if attempts + 1 > self.limits.requests_per_query:
            raise BudgetBlocked("per_query_request_budget")
        if self.global_requests + 1 > self.limits.requests_total:
            raise BudgetBlocked("global_request_budget")
        if (
            state.total_tokens + state.reserved_total_tokens + projected_tokens
            > self.limits.tokens_per_query
        ):
            raise BudgetBlocked("per_query_token_budget")
        if (
            self.global_tokens + self.global_reserved_tokens + projected_tokens
            > self.limits.tokens_total
        ):
            raise BudgetBlocked("global_token_budget")
        if elapsed_query >= self.limits.elapsed_per_query:
            raise ElapsedTimeBlocked("per_query_elapsed_budget")
        if time.monotonic() - self.started >= self.limits.elapsed_total:
            raise ElapsedTimeBlocked("global_elapsed_budget")
        projected_cost = self.policy.cost(estimated_input, max_output)
        if self.policy.mode == "paid" and self.global_cost + projected_cost > self.policy.max_cost:
            raise BudgetBlocked("global_cost_budget")

    def reserve(self, state: SmokeState, input_tokens: int, output_tokens: int) -> None:
        self.preflight(state, input_tokens, output_tokens)
        total = input_tokens + output_tokens
        state.reserved_input_tokens += input_tokens
        state.reserved_output_tokens += output_tokens
        state.reserved_total_tokens += total
        state.budget_accounting_status = "conservative_reserved"
        self.global_reserved_tokens += total

    def release_reservation(self, state: SmokeState, request: dict[str, Any]) -> None:
        if request.get("usage_status") in {
            "provider_reported",
            "released_after_provider_failure",
            "released_after_missing_usage",
        }:
            return
        total = int(request["reserved_total_tokens"])
        if total <= 0:
            return
        state.reserved_input_tokens -= int(request["reserved_input_tokens"])
        state.reserved_output_tokens -= int(request["reserved_output_tokens"])
        state.reserved_total_tokens -= total
        self.global_reserved_tokens -= total
        state.reserved_input_tokens = max(0, state.reserved_input_tokens)
        state.reserved_output_tokens = max(0, state.reserved_output_tokens)
        state.reserved_total_tokens = max(0, state.reserved_total_tokens)
        self.global_reserved_tokens = max(0, self.global_reserved_tokens)
        if state.reserved_total_tokens == 0:
            state.budget_accounting_status = "settled"

    def commit(
        self, state: SmokeState, record: UsageRecord, request: dict[str, Any]
    ) -> None:
        self.release_reservation(state, request)
        state.input_tokens += record.input_tokens
        state.output_tokens += record.output_tokens
        state.total_tokens += record.total_tokens
        cost = Decimal(record.monetary_cost_usd)
        state.monetary_cost_usd = str(Decimal(state.monetary_cost_usd) + cost)
        state.usage_records.append(asdict(record))
        state.provider_completed_request_count += 1
        state.usage_record_count += 1
        self.global_tokens += record.total_tokens
        self.global_cost += cost
        if state.total_tokens > self.limits.tokens_per_query:
            raise BudgetBlocked("actual_per_query_token_budget")
        if self.global_tokens > self.limits.tokens_total:
            raise BudgetBlocked("actual_global_token_budget")
        if self.policy.mode == "paid" and self.global_cost > self.policy.max_cost:
            raise BudgetBlocked("actual_global_cost_budget")


def context_to_dict(item: ContextItem) -> dict[str, Any]:
    return item.model_dump()


def context_from_dict(item: dict[str, Any]) -> ContextItem:
    return ContextItem.model_validate(item)


class BoundedSmokeRunner:
    def __init__(
        self,
        llm: SmokeLLM,
        checkpoint: SQLiteSmokeCheckpoint,
        guard: BudgetGuard,
        *,
        prompt_version: str,
        max_output_tokens: int,
        retrieval: Any,
        request_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        if llm.provider_name == "template":
            raise SmokeConfigurationError("Template fallback is forbidden")
        self.llm = llm
        self.checkpoint = checkpoint
        self.guard = guard
        self.prompt_version = prompt_version
        self.max_output_tokens = max_output_tokens
        self.retrieval = retrieval
        self.request_event = request_event or (lambda event: None)

    def run(
        self,
        sample: dict[str, Any],
        *,
        run_id: str,
        resume: bool = False,
        stop_after_node: str | None = None,
    ) -> SmokeState:
        state = self.checkpoint.load(run_id)
        if state:
            if not resume:
                raise SmokeConfigurationError("run_id exists; pass --resume")
            if state.question_id != sample["question_id"]:
                raise SmokeConfigurationError("run_id belongs to a different question")
            state.resume_count += 1
            if state.status in {"stopped", "interrupted"}:
                state.status = "running"
        else:
            state = SmokeState(
                run_id=run_id,
                question_id=sample["question_id"],
                question=sample["question"],
                retrieval_scope=sample["retrieval_scope"],
                retrieval_filter=sample["retrieval_filter"],
            )
        while state.status == "running":
            node = state.current_node
            if node not in NODES:
                raise RuntimeError(f"unknown node: {node}")
            started = time.perf_counter()
            try:
                next_node = getattr(self, f"_node_{node}")(state, sample)
                state.nodes_visited.append(node)
                state.events.append(
                    {
                        "event_id": f"{run_id}:{node}",
                        "node": node,
                        "status": "completed",
                        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
                    }
                )
                state.current_node = next_node
                if node == "persist_trace":
                    state.status = (
                        "refused"
                        if state.answer and not state.answer.get("answerable")
                        else "completed"
                    )
            except ElapsedTimeBlocked as exc:
                state.status = "elapsed_time_blocked"
                state.budget_stop_reason = str(exc)
                state.errors.append(str(exc))
                self._failure_event(state, run_id, node, started, state.status)
            except BudgetBlocked as exc:
                state.status = "budget_blocked"
                state.budget_stop_reason = str(exc)
                state.errors.append(str(exc))
                self._failure_event(state, run_id, node, started, state.status)
            except ProviderFailed as exc:
                state.status = "provider_failed"
                state.errors.append(str(exc))
                self._failure_event(state, run_id, node, started, state.status)
            except Exception as exc:
                state.status = (
                    "validation_failed" if node == "validate_citations" else "checkpoint_error"
                )
                state.errors.append(f"{type(exc).__name__}: {exc}")
                self._failure_event(state, run_id, node, started, state.status)
            state.elapsed_seconds = round(time.time() - state.started_at, 6)
            self.checkpoint.save(state)
            if stop_after_node == node:
                state.status = "interrupted"
                self.checkpoint.save(state)
                break
        return state

    @staticmethod
    def _failure_event(
        state: SmokeState, run_id: str, node: str, started: float, status: str
    ) -> None:
        state.events.append(
            {
                "event_id": f"{run_id}:{node}:{status}",
                "node": node,
                "status": status,
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
            }
        )

    @staticmethod
    def _node_plan(state: SmokeState, sample: dict[str, Any]) -> str:
        del state, sample
        return "retrieve"

    def _node_retrieve(self, state: SmokeState, sample: dict[str, Any]) -> str:
        contexts = self.retrieval(sample, state.iteration_count)
        state.contexts = [context_to_dict(item) for item in contexts]
        state.retrieval_calls += 1
        state.iteration_count += 1
        return "assess_evidence"

    @staticmethod
    def _node_assess_evidence(state: SmokeState, sample: dict[str, Any]) -> str:
        del sample
        return (
            "optional_refine"
            if not state.contexts and state.iteration_count < 2
            else "synthesize"
        )

    def _node_optional_refine(self, state: SmokeState, sample: dict[str, Any]) -> str:
        if not state.contexts and state.iteration_count < 2:
            contexts = self.retrieval(sample, state.iteration_count)
            state.contexts = [context_to_dict(item) for item in contexts]
            state.retrieval_calls += 1
            state.iteration_count += 1
        return "synthesize"

    def _node_synthesize(self, state: SmokeState, sample: dict[str, Any]) -> str:
        contexts = [context_from_dict(item) for item in state.contexts]
        estimated_input = conservative_token_estimate(state.question, contexts)
        attempt_number = state.request_attempt_count + 1
        request_id = (
            f"{state.run_id}:synthesize:{attempt_number}:{uuid.uuid4().hex[:12]}"
        )
        request = {
            "request_id": request_id,
            "reservation_id": f"{request_id}:reservation",
            "node": "synthesize",
            "request_status": "prepared",
            "usage_status": "reserved_conservative",
            "prompt_token_estimate": estimated_input,
            "maximum_output_tokens": self.max_output_tokens,
            "reserved_input_tokens": estimated_input,
            "reserved_output_tokens": self.max_output_tokens,
            "reserved_total_tokens": estimated_input + self.max_output_tokens,
            "provider": self.llm.provider_name,
            "model": self.llm.model_name,
            "prepared_at": time.time(),
            "started_at": None,
            "completed_at": None,
            "failure_type": None,
            "failure_message": None,
        }
        state.request_records.append(request)
        self.checkpoint.save(state)
        self.request_event({"event": "request_prepared", **request})
        try:
            self.guard.reserve(state, estimated_input, self.max_output_tokens)
        except BudgetBlocked:
            request["request_status"] = "failed_before_send"
            request["usage_status"] = "unavailable_not_sent"
            request["failure_type"] = "budget_guard"
            self.checkpoint.save(state)
            self.request_event({"event": "request_failed_before_send", **request})
            raise
        self.request_event(
            {
                "event": "TOKEN_BUDGET_RESERVED",
                "run_id": state.run_id,
                "request_id": request_id,
                "reservation_id": request["reservation_id"],
                "reserved_input_tokens": estimated_input,
                "reserved_output_tokens": self.max_output_tokens,
                "reserved_total_tokens": estimated_input + self.max_output_tokens,
                "settled_tokens": 0,
                "released_tokens": 0,
                "timestamp": time.time(),
            }
        )
        request["request_status"] = "started"
        request["started_at"] = time.time()
        state.request_attempt_count += 1
        state.llm_requests = state.request_attempt_count
        self.guard.global_requests += 1
        self.checkpoint.save(state)
        self.request_event({"event": "request_started", **request})
        try:
            result = self.llm.generate_claim_answer(
                state.question,
                contexts,
                self.prompt_version,
                audit_metadata={
                    "request_id": request_id,
                    "run_id": state.run_id,
                    "sample_id": state.question_id,
                    "thread_id": state.run_id,
                },
            )
        except LLMProviderError as exc:
            request["request_status"] = "failed_after_send_unknown"
            request["failure_type"] = (
                exc.error_details.get("classification")
                or (exc.retry_reasons[-1] if exc.retry_reasons else type(exc).__name__)
            )
            request["failure_message"] = str(exc)
            request["error_code"] = exc.error_code
            request["error_stage"] = exc.stage
            request["error_details"] = exc.error_details
            request["response_audit_path"] = exc.response_audit_path
            request["completed_at"] = time.time()
            self.guard.release_reservation(state, request)
            request["usage_status"] = "released_after_provider_failure"
            self.request_event(
                {
                    "event": "TOKEN_BUDGET_RELEASED",
                    "run_id": state.run_id,
                    "request_id": request_id,
                    "reservation_id": request["reservation_id"],
                    "reserved_input_tokens": request["reserved_input_tokens"],
                    "reserved_output_tokens": request["reserved_output_tokens"],
                    "reserved_total_tokens": request["reserved_total_tokens"],
                    "settled_tokens": 0,
                    "released_tokens": request["reserved_total_tokens"],
                    "timestamp": time.time(),
                }
            )
            self.checkpoint.save(state)
            self.request_event({"event": "request_failed", **request})
            raise ProviderFailed(f"provider_failure:{exc}") from exc
        usage = result.usage
        if usage.total_tokens <= 0:
            request["request_status"] = "failed_after_send_unknown"
            request["failure_type"] = "missing_provider_usage"
            request["failure_message"] = "provider response omitted reliable usage"
            request["completed_at"] = time.time()
            self.guard.release_reservation(state, request)
            request["usage_status"] = "released_after_missing_usage"
            self.request_event(
                {
                    "event": "TOKEN_BUDGET_RELEASED",
                    "run_id": state.run_id,
                    "request_id": request_id,
                    "reservation_id": request["reservation_id"],
                    "reserved_input_tokens": request["reserved_input_tokens"],
                    "reserved_output_tokens": request["reserved_output_tokens"],
                    "reserved_total_tokens": request["reserved_total_tokens"],
                    "settled_tokens": 0,
                    "released_tokens": request["reserved_total_tokens"],
                    "timestamp": time.time(),
                }
            )
            self.checkpoint.save(state)
            self.request_event({"event": "request_failed", **request})
            raise ProviderFailed("provider_failure:missing_provider_usage")
        source: UsageSource = "provider_reported"
        record = UsageRecord(
            request_id=request_id,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            usage_source=source,
            monetary_cost_usd=str(
                self.guard.policy.cost(usage.input_tokens, usage.output_tokens)
            ),
            cost_basis=self.guard.policy.cost_basis,
        )
        self.guard.commit(state, record, request)
        request["request_status"] = "completed"
        request["usage_status"] = source
        request["completed_at"] = time.time()
        request["actual_input_tokens"] = usage.input_tokens
        request["actual_output_tokens"] = usage.output_tokens
        request["actual_total_tokens"] = usage.total_tokens
        request["monetary_cost_usd"] = record.monetary_cost_usd
        self.request_event(
            {
                "event": "TOKEN_BUDGET_SETTLED",
                "run_id": state.run_id,
                "request_id": request_id,
                "reservation_id": request["reservation_id"],
                "reserved_input_tokens": request["reserved_input_tokens"],
                "reserved_output_tokens": request["reserved_output_tokens"],
                "reserved_total_tokens": request["reserved_total_tokens"],
                "settled_tokens": usage.total_tokens,
                "released_tokens": request["reserved_total_tokens"],
                "timestamp": time.time(),
            }
        )
        self.checkpoint.save(state)
        self.request_event({"event": "request_completed", **request})
        state.answer = result.model_dump(mode="json")
        return "validate_citations"

    @staticmethod
    def _node_validate_citations(state: SmokeState, sample: dict[str, Any]) -> str:
        del sample
        if state.answer is None:
            raise RuntimeError("missing answer")
        answerable = bool(state.answer["answerable"])
        claims = state.answer["claims"]
        if not answerable and any(claim.get("citations") for claim in claims):
            raise ValueError("unanswerable output contains citations")
        contexts = [context_from_dict(item) for item in state.contexts]
        answer_model = GenerationResult.model_validate(state.answer)
        SiliconFlowLLMProvider._validate_context_citations(answer_model, contexts)
        state.citation_validation = "passed"
        return "persist_trace"

    @staticmethod
    def _node_persist_trace(state: SmokeState, sample: dict[str, Any]) -> str:
        del state, sample
        return "END"


def conservative_token_estimate(question: str, contexts: list[ContextItem]) -> int:
    characters = len(question) + sum(len(item.evidence) for item in contexts)
    return max(1, math.ceil(characters / 2))
