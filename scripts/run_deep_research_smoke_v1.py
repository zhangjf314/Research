"""Run the Stage 11D bounded engineering smoke (never a quality evaluation)."""

from __future__ import annotations

import argparse
import json
import uuid
from decimal import Decimal
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient

from paper_research.agents.bounded_smoke import (
    BillingPolicy,
    BoundedSmokeRunner,
    BudgetGuard,
    SmokeConfigurationError,
    SmokeLimits,
    SQLiteSmokeCheckpoint,
    smoke_configuration,
)
from paper_research.agents.smoke_artifacts import (
    DEFAULT_RUN_ROOT,
    RequestLedger,
    load_runs,
    state_result,
    write_run_artifacts,
)
from paper_research.config import Settings
from paper_research.parsing.types import PaperBlock
from paper_research.providers.factory import build_llm_provider
from paper_research.providers.llm import (
    GeneratedCitation,
    GeneratedClaim,
    GenerationResult,
    ModelUsage,
)
from paper_research.retrieval.context_builder import ContextItem

MANIFEST = Path("data/evaluation/deep-research-smoke-v1.jsonl")
PRODUCTION_CORPUS = Path("data/evaluation/production-corpus-v1.json")
INDEX_MANIFEST = Path("data/evaluation/retrieval-index-v2.json")
CONTEXT_SOURCE = Path("data/evaluation/retrieval-context-optimization-v1.json")
BLOCK_ROOT = Path("data/reports/parsing-audit")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("dry-run", "live"), required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--question-id")
    group.add_argument("--all", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument("--stop-after-node")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--attempt-number", type=int, default=1)
    parser.add_argument("--parent-run-id")
    parser.add_argument("--no-summary", action="store_true")
    parser.add_argument("--max-cost-usd")
    parser.add_argument("--max-total-tokens", type=int)
    parser.add_argument("--max-total-requests", type=int)
    parser.add_argument("--max-total-seconds", type=int)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def validate_manifest(rows: list[dict[str, Any]]) -> None:
    if len(rows) != 3 or len({row["question_id"] for row in rows}) != 3:
        raise SmokeConfigurationError("smoke manifest must contain exactly three unique questions")
    if {row["smoke_role"] for row in rows} != {
        "single_paper_method",
        "multi_paper_comparison",
        "unanswerable",
    }:
        raise SmokeConfigurationError("smoke manifest roles are invalid")
    if {"q033", "q044"} & {row["question_id"] for row in rows}:
        raise SmokeConfigurationError("q033/q044 are excluded")
    for row in rows:
        if row["max_iterations"] > 2 or row["max_llm_requests"] > 4:
            raise SmokeConfigurationError("manifest exceeds bounded limits")


def validate_corpus_and_index(settings: Settings) -> dict[str, Any]:
    corpus = json.loads(PRODUCTION_CORPUS.read_text(encoding="utf-8"))
    papers = corpus["papers"] if isinstance(corpus, dict) else corpus
    included = [row for row in papers if row.get("included_in_production")]
    if len(included) != 34:
        raise SmokeConfigurationError(
            f"production corpus must contain 34 papers, got {len(included)}"
        )
    index = json.loads(INDEX_MANIFEST.read_text(encoding="utf-8"))
    jina = index["collections"]["jina"]
    if jina["paper_count"] != 34 or jina["point_count"] != 2062 or jina["dimension"] != 1024:
        raise SmokeConfigurationError("fixed Jina index manifest metadata is invalid")
    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    actual_points = client.count(jina["name"], exact=True).count
    if actual_points != 2062:
        raise SmokeConfigurationError(
            f"fixed Jina collection point count is {actual_points}, expected 2062"
        )
    paper_ids: set[str] = set()
    offset = None
    while True:
        points, offset = client.scroll(
            jina["name"],
            limit=256,
            offset=offset,
            with_payload=["paper_id"],
            with_vectors=False,
        )
        paper_ids.update(
            str(point.payload["paper_id"])
            for point in points
            if point.payload and point.payload.get("paper_id")
        )
        if offset is None:
            break
    if len(paper_ids) != 34:
        raise SmokeConfigurationError(
            f"fixed Jina collection paper count is {len(paper_ids)}, expected 34"
        )
    return {
        "production_papers": 34,
        "index_manifest": str(INDEX_MANIFEST),
        "collection": jina["name"],
        "dimension": 1024,
        "point_count": actual_points,
        "collection_papers": len(paper_ids),
    }


def exact_contexts() -> dict[str, list[ContextItem]]:
    source = json.loads(CONTEXT_SOURCE.read_text(encoding="utf-8"))
    experiment = next(
        row
        for row in source["experiments"]
        if row["experiment_id"] == source["final_experiment_id"]
    )
    output: dict[str, list[ContextItem]] = {"q005": []}
    for question_id in ("q003", "q049"):
        row = next(item for item in experiment["queries"] if item["question_id"] == question_id)
        contexts = []
        block_cache: dict[str, dict[str, PaperBlock]] = {}
        for item in row["context"]:
            paper_id = item["paper_id"]
            if paper_id not in block_cache:
                block_path = BLOCK_ROOT / paper_id / "paper_blocks.jsonl"
                block_cache[paper_id] = {
                    block.block_id: block
                    for block in (
                        PaperBlock.model_validate_json(line)
                        for line in block_path.read_text(encoding="utf-8").splitlines()
                        if line
                    )
                }
            blocks = block_cache[paper_id]
            page_map = {block_id: blocks[block_id].page_start for block_id in item["block_ids"]}
            contexts.append(ContextItem.model_validate({**item, "block_page_map": page_map}))
        output[question_id] = contexts
    return output


class DryRunLLM:
    provider_name = "dry_run_mock_siliconflow"
    model_name = "Qwen/Qwen3-8B-dry-run"

    def generate_claim_answer(
        self, question: str, context: list[ContextItem], prompt_version: str
    ) -> GenerationResult:
        del question, prompt_version
        usage = ModelUsage(input_tokens=128, output_tokens=32, total_tokens=160)
        if not context:
            return GenerationResult(
                answerable=False,
                answer=None,
                claims=[],
                refusal_reason="Dry-run: no bounded evidence was supplied.",
                usage=usage,
                raw_model=self.model_name,
                api_request_count=1,
            )
        item = context[0]
        block_id = item.block_ids[0]
        page = item.block_page_map[block_id]
        claim = GeneratedClaim(
            claim_id="dry-run-c1",
            text="Dry-run structural claim; not a quality result.",
            citations=[GeneratedCitation(paper_id=item.paper_id, page=page, block_id=block_id)],
        )
        return GenerationResult(
            answerable=True,
            answer=claim.text,
            claims=[claim],
            refusal_reason=None,
            usage=usage,
            raw_model=self.model_name,
            api_request_count=1,
        )


def limits_with_overrides(
    policy: BillingPolicy, limits: SmokeLimits, args: argparse.Namespace
) -> tuple[BillingPolicy, SmokeLimits]:
    from decimal import Decimal

    if args.max_cost_usd is not None:
        policy = BillingPolicy(
            policy.mode,
            policy.input_price,
            policy.output_price,
            Decimal(args.max_cost_usd),
            policy.warning,
        )
    limits = SmokeLimits(
        limits.max_queries,
        limits.iterations_per_query,
        limits.requests_per_query,
        args.max_total_requests or limits.requests_total,
        limits.tokens_per_query,
        args.max_total_tokens or limits.tokens_total,
        limits.elapsed_per_query,
        args.max_total_seconds or limits.elapsed_total,
    )
    return policy, limits


def main() -> int:
    args = parse_args()
    if args.mode == "live" and not (args.all or args.question_id):
        raise SmokeConfigurationError("live mode requires --all or --question-id")
    if args.mode == "live" and args.all:
        raise SmokeConfigurationError("live --all is forbidden for the bounded smoke")
    if args.resume and not args.run_id:
        raise SmokeConfigurationError("--resume requires --run-id")
    if args.attempt_number < 1:
        raise SmokeConfigurationError("--attempt-number must be positive")
    if args.attempt_number > 1 and not args.parent_run_id:
        raise SmokeConfigurationError("later attempts require --parent-run-id")
    if args.attempt_number == 1 and args.parent_run_id:
        raise SmokeConfigurationError("first attempt cannot have --parent-run-id")
    rows = load_jsonl(MANIFEST)
    validate_manifest(rows)
    settings_updates: dict[str, Any] = {}
    if args.max_cost_usd is not None:
        settings_updates["deep_research_max_cost_usd"] = Decimal(args.max_cost_usd)
    settings = Settings(**settings_updates)
    validate_corpus_and_index(settings)
    selected = rows if args.all else [row for row in rows if row["question_id"] == args.question_id]
    if not selected:
        raise SmokeConfigurationError("select --all or a manifest question_id")
    policy, limits = smoke_configuration(settings)
    policy, limits = limits_with_overrides(policy, limits, args)
    if len(selected) > limits.max_queries:
        raise SmokeConfigurationError("query count exceeds configured limit")
    contexts = exact_contexts()
    checkpoint = SQLiteSmokeCheckpoint(settings.deep_research_checkpoint_path)
    guard = BudgetGuard(policy, limits)
    run_ids = [
        (
            f"{args.run_id}-{sample['question_id']}"
            if args.run_id and len(selected) > 1
            else args.run_id or f"{args.mode}-{sample['question_id']}-{uuid.uuid4().hex[:12]}"
        )
        for sample in selected
    ]
    live_runs = [
        run for run in load_runs(args.output_root) if run["metadata"]["mode"] == "live"
    ]
    effective_attempt_number = args.attempt_number
    effective_parent_run_id = args.parent_run_id
    if args.resume:
        resume_runs = [
            run for run in live_runs if run["metadata"]["run_id"] == args.run_id
        ]
        if len(resume_runs) != 1:
            raise SmokeConfigurationError(
                "--resume requires one matching isolated run directory"
            )
        resume_metadata = resume_runs[0]["metadata"]
        if resume_metadata["question_id"] != selected[0]["question_id"]:
            raise SmokeConfigurationError("resume run belongs to a different question")
        effective_attempt_number = int(resume_metadata["attempt_number"])
        effective_parent_run_id = resume_metadata["parent_run_id"]
    prior_ids = [run["metadata"]["run_id"] for run in live_runs]
    if args.parent_run_id and args.parent_run_id not in prior_ids:
        raise SmokeConfigurationError("--parent-run-id does not identify an isolated live run")
    if args.parent_run_id:
        parent = next(
            run for run in live_runs if run["metadata"]["run_id"] == args.parent_run_id
        )
        if len(selected) != 1 or parent["metadata"]["question_id"] != selected[0]["question_id"]:
            raise SmokeConfigurationError("parent run belongs to a different question")
    if args.mode == "live" and not args.resume:
        question_attempts = [
            int(run["metadata"]["attempt_number"])
            for run in live_runs
            if run["metadata"]["question_id"] == selected[0]["question_id"]
        ]
        expected_attempt = max(question_attempts, default=0) + 1
        if args.attempt_number != expected_attempt:
            raise SmokeConfigurationError(
                f"--attempt-number must be {expected_attempt} for this question"
            )
    prior_states = checkpoint.load_many(prior_ids)
    if args.resume:
        prior_states.extend(
            state
            for state in checkpoint.load_many(run_ids)
            if state.run_id not in {prior.run_id for prior in prior_states}
        )
    guard.restore(prior_states)
    prior_by_id = {state.run_id: state for state in prior_states}
    for run in live_runs:
        exported = int(run["result"].get("reserved_total_tokens") or 0)
        prior_state = prior_by_id.get(run["metadata"]["run_id"])
        checkpoint_reserved = prior_state.reserved_total_tokens if prior_state else 0
        guard.global_reserved_tokens += max(0, exported - checkpoint_reserved)
    if args.mode == "dry-run":
        llm: Any = DryRunLLM()
    else:
        # Stage 11D treats a provider retry as a separate budgeted request; until an
        # attempt callback exists, fail on the first invalid response instead.
        llm = build_llm_provider(settings.model_copy(update={"llm_max_retries": 0}))
    states = []
    results = []
    run_paths = []
    for index, (sample, run_id) in enumerate(zip(selected, run_ids, strict=True)):
        run_dir = args.output_root / run_id
        ledger = RequestLedger(run_dir / "request-ledger.jsonl")
        runner = BoundedSmokeRunner(
            llm,
            checkpoint,
            guard,
            prompt_version="qa-production-v1",
            max_output_tokens=settings.llm_max_output_tokens,
            retrieval=lambda row, iteration: contexts[row["question_id"]],
            request_event=ledger.emit,
        )
        state = runner.run(
            sample,
            run_id=run_id,
            resume=args.resume,
            stop_after_node=args.stop_after_node if index == 0 else None,
        )
        states.append(state)
        result = state_result(
            state, provider=llm.provider_name, model=llm.model_name, policy=policy
        )
        results.append(result)
        run_paths.append(
            write_run_artifacts(
                args.output_root,
                state,
                result,
                mode=args.mode,
                attempt_number=effective_attempt_number,
                parent_run_id=effective_parent_run_id,
                provider=llm.provider_name,
                model=llm.model_name,
                policy=policy,
                settings=settings,
            )
        )
    print(
        json.dumps(
            {
                "mode": args.mode,
                "billing_mode": policy.mode,
                "cost_basis": policy.cost_basis,
                "monetary_budget_usd": str(policy.max_cost),
                "max_requests": limits.requests_total,
                "max_tokens": limits.tokens_total,
                "max_seconds": limits.elapsed_total,
                "results": [
                    {
                        "question_id": row["question_id"],
                        "status": row["graph_status"],
                        "run_id": row["run_id"],
                        "run_directory": str(path),
                    }
                    for row, path in zip(results, run_paths, strict=True)
                ],
                "summary_updated": False,
                "summary_command": (
                    "python scripts/summarize_deep_research_smoke_v1.py "
                    "--selection-policy latest-successful"
                ),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
