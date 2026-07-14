import json
from decimal import Decimal
from pathlib import Path

import pytest

from paper_research.agents.bounded_smoke import (
    BillingPolicy,
    BoundedSmokeRunner,
    BudgetBlocked,
    BudgetGuard,
    SmokeConfigurationError,
    SmokeLimits,
    SmokeState,
    SQLiteSmokeCheckpoint,
    UsageRecord,
    smoke_configuration,
)
from paper_research.config import Settings
from paper_research.providers.llm import (
    GeneratedCitation,
    GeneratedClaim,
    GenerationResult,
    ModelUsage,
)
from paper_research.retrieval.context_builder import ContextItem


def settings(**updates: object) -> Settings:
    values = {
        "deep_research_mode": "engineering_smoke",
        "rerank_enabled": False,
        "embedding_provider": "jina",
        "embedding_model": "jina-embeddings-v5-text-small",
        "llm_provider": "siliconflow",
        "llm_model": "Qwen/Qwen3-8B",
        "llm_billing_mode": "free",
        "llm_input_price_per_million_tokens": Decimal("0"),
        "llm_output_price_per_million_tokens": Decimal("0"),
        "deep_research_max_cost_usd": Decimal("0"),
    }
    values.update(updates)
    return Settings(_env_file=None, **values)


def context() -> ContextItem:
    return ContextItem(
        chunk_id="c1",
        paper_id="p1",
        block_ids=["b1"],
        block_page_map={"b1": 2},
        section_path=["Method"],
        page_start=2,
        page_end=2,
        evidence="bounded evidence",
        score=1,
    )


class CountingLLM:
    provider_name = "siliconflow_mock"
    model_name = "Qwen/Qwen3-8B"

    def __init__(self, *, invalid=False, missing_usage=False):
        self.calls = 0
        self.invalid = invalid
        self.missing_usage = missing_usage

    def generate_claim_answer(self, question, contexts, prompt_version):
        del question, prompt_version
        self.calls += 1
        item = contexts[0]
        citation = GeneratedCitation(
            paper_id=item.paper_id,
            page=99 if self.invalid else 2,
            block_id="b1",
        )
        return GenerationResult(
            answerable=True,
            answer="claim",
            claims=[GeneratedClaim(claim_id="c1", text="claim", citations=[citation])],
            refusal_reason=None,
            usage=ModelUsage()
            if self.missing_usage
            else ModelUsage(input_tokens=10, output_tokens=5, total_tokens=15),
            raw_model=self.model_name,
            api_request_count=1,
        )


def limits(**updates: int) -> SmokeLimits:
    values = dict(
        max_queries=3,
        iterations_per_query=2,
        requests_per_query=4,
        requests_total=12,
        tokens_per_query=40000,
        tokens_total=120000,
        elapsed_per_query=300,
        elapsed_total=900,
    )
    values.update(updates)
    return SmokeLimits(**values)


@pytest.mark.parametrize("mode", ["paid", "free", "local"])
def test_billing_modes_are_explicit(mode):
    kwargs = {}
    if mode == "paid":
        kwargs = {
            "llm_input_price_per_million_tokens": Decimal("0.1"),
            "llm_output_price_per_million_tokens": Decimal("0.2"),
            "deep_research_max_cost_usd": Decimal("1"),
        }
    policy, _ = smoke_configuration(settings(llm_billing_mode=mode, **kwargs))
    assert policy.mode == mode


@pytest.mark.parametrize("mode", ["invalid", "", None])
def test_invalid_billing_mode_fails(mode):
    with pytest.raises(SmokeConfigurationError, match="LLM_BILLING_MODE"):
        smoke_configuration(settings(llm_billing_mode=mode))


def test_paid_missing_price_and_zero_budget_fail_closed():
    with pytest.raises(SmokeConfigurationError):
        smoke_configuration(
            settings(
                llm_billing_mode="paid",
                llm_input_price_per_million_tokens=None,
                deep_research_max_cost_usd=Decimal("1"),
            )
        )
    with pytest.raises(SmokeConfigurationError):
        smoke_configuration(settings(llm_billing_mode="paid"))


@pytest.mark.parametrize("mode", ["free", "local"])
def test_nonzero_free_or_local_price_fails(mode):
    with pytest.raises(SmokeConfigurationError):
        smoke_configuration(
            settings(llm_billing_mode=mode, llm_input_price_per_million_tokens=Decimal("0.1"))
        )


def test_decimal_paid_cost_is_exact():
    policy = BillingPolicy("paid", Decimal("0.1"), Decimal("0.2"), Decimal("1"))
    assert policy.cost(3, 7) == Decimal("0.0000017")


def test_free_budget_still_enforces_request_and_tokens():
    policy = BillingPolicy("free", Decimal(0), Decimal(0), Decimal(0))
    guard = BudgetGuard(policy, limits(requests_per_query=1, tokens_per_query=10))
    state = SmokeState("r", "q", "question", "paper", {})
    with pytest.raises(BudgetBlocked, match="token"):
        guard.preflight(state, 8, 3)
    state.llm_requests = 1
    with pytest.raises(BudgetBlocked, match="request"):
        guard.preflight(state, 1, 1)


def test_missing_usage_fails_and_is_not_zero(tmp_path):
    llm = CountingLLM(missing_usage=True)
    runner = BoundedSmokeRunner(
        llm,
        SQLiteSmokeCheckpoint(tmp_path / "checkpoint.sqlite"),
        BudgetGuard(BillingPolicy("free", Decimal(0), Decimal(0), Decimal(0)), limits()),
        prompt_version="qa-production-v1",
        max_output_tokens=100,
        retrieval=lambda sample, iteration: [context()],
    )
    state = runner.run(sample(), run_id="missing")
    assert state.status == "provider_failed"
    assert state.budget_stop_reason is None
    assert state.request_records[0]["usage_status"] == "unavailable_after_send_attempt"


def sample() -> dict:
    return {
        "question_id": "q003",
        "question": "method?",
        "retrieval_scope": "paper",
        "retrieval_filter": {"paper_ids": ["p1"]},
    }


def build_runner(tmp_path, llm=None):
    llm = llm or CountingLLM()
    return (
        BoundedSmokeRunner(
            llm,
            SQLiteSmokeCheckpoint(tmp_path / "checkpoint.sqlite"),
            BudgetGuard(BillingPolicy("free", Decimal(0), Decimal(0), Decimal(0)), limits()),
            prompt_version="qa-production-v1",
            max_output_tokens=100,
            retrieval=lambda row, iteration: [context()],
        ),
        llm,
    )


def test_checkpoint_resume_does_not_repeat_llm_usage_or_trace(tmp_path):
    runner, llm = build_runner(tmp_path)
    stopped = runner.run(sample(), run_id="resume", stop_after_node="synthesize")
    assert stopped.status == "interrupted"
    assert llm.calls == 1
    event_count = len(stopped.events)
    resumed = runner.run(sample(), run_id="resume", resume=True)
    assert resumed.status == "completed"
    assert llm.calls == 1
    assert resumed.total_tokens == 15
    assert len(resumed.events) == event_count + 2
    assert len({event["event_id"] for event in resumed.events}) == len(resumed.events)


def test_illegal_citation_strictly_fails_without_repair(tmp_path):
    runner, llm = build_runner(tmp_path, CountingLLM(invalid=True))
    state = runner.run(sample(), run_id="invalid")
    assert state.status == "validation_failed"
    assert state.citation_validation == "not_run"
    assert llm.calls == 1
    assert "CitationContextError" in state.errors[0]


def test_template_full_mode_and_reranker_are_forbidden():
    with pytest.raises(SmokeConfigurationError):
        smoke_configuration(settings(deep_research_mode="full"))
    with pytest.raises(SmokeConfigurationError):
        smoke_configuration(settings(rerank_enabled=True))
    with pytest.raises(SmokeConfigurationError):
        smoke_configuration(settings(llm_provider="template"))


def test_actual_usage_over_limit_stops_following_nodes():
    guard = BudgetGuard(
        BillingPolicy("free", Decimal(0), Decimal(0), Decimal(0)), limits(tokens_per_query=10)
    )
    state = SmokeState("r", "q", "question", "paper", {})
    request = {
        "reserved_input_tokens": 4,
        "reserved_output_tokens": 4,
        "reserved_total_tokens": 8,
    }
    guard.reserve(state, 4, 4)
    with pytest.raises(BudgetBlocked, match="actual_per_query"):
        guard.commit(
            state,
            UsageRecord("request", 7, 7, 14, "provider_reported", "0", "explicit_free_provider"),
            request,
        )


def test_fixed_manifest_is_three_bounded_questions_and_excludes_failures():
    path = Path("data/evaluation/deep-research-smoke-v1.jsonl")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 3
    assert {row["smoke_role"] for row in rows} == {
        "single_paper_method",
        "multi_paper_comparison",
        "unanswerable",
    }
    assert not {"q033", "q044"} & {row["question_id"] for row in rows}
    assert all(row["max_iterations"] <= 2 for row in rows)
    assert all(row["max_llm_requests"] <= 4 for row in rows)
