"""Freeze the Stage 13.37 DeepSeek Direct Full QA final configuration."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paper_research.config import Settings  # noqa: E402

DATA = ROOT / "data" / "evaluation"
DOCS = ROOT / "docs"
RETRIEVAL_GOLD = DATA / "retrieval-gold-v2.jsonl"
OUT_JSON = DATA / "deepseek-full-qa-final-config-v1.json"
OUT_DOC = DOCS / "deepseek-full-qa-final-config-v1.md"


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def hash_payload(payload: dict[str, Any]) -> str:
    stable = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(stable.encode("utf-8")).hexdigest()


def main() -> int:
    settings = Settings()
    provider = settings.llm_provider_name or settings.llm_provider
    config: dict[str, Any] = {
        "schema_version": "deepseek-full-qa-final-config-v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "branch": subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=ROOT, text=True
        ).strip(),
        "commit": git_head(),
        "dataset": "gold-dev-v1",
        "approved_only": True,
        "expected_total": 50,
        "expected_answerable": 48,
        "expected_unanswerable": 2,
        "dataset_hash": sha256_path(RETRIEVAL_GOLD),
        "retrieval": {
            "recall_k": settings.retrieval_recall_k,
            "top_k": 10,
            "context_selector": "production-api-context-builder",
            "context_token_budget": settings.qa_context_token_budget,
            "index_version": settings.index_version,
            "collection": settings.active_collection,
            "score_threshold": settings.retrieval_score_threshold,
        },
        "embedding": {
            "provider": settings.embedding_provider,
            "model": settings.embedding_model,
            "dimension": settings.embedding_dimensions,
        },
        "qdrant": {"collection": settings.active_collection},
        "llm": {
            "provider": provider,
            "adapter": settings.llm_provider,
            "model": settings.llm_model,
            "base_url_host": "api.deepseek.com"
            if settings.llm_base_url and "api.deepseek.com" in settings.llm_base_url
            else None,
            "temperature": settings.llm_temperature,
            "max_tokens": settings.llm_max_output_tokens,
            "response_format": settings.llm_response_format,
            "thinking": "enabled" if settings.llm_thinking_enabled else "disabled",
            "stream": settings.llm_stream,
            "api_key_present": bool(settings.llm_api_key),
            "api_key_recorded": False,
        },
        "prompt": settings.prompt_version,
        "normalizer": "citation-key-minimal-contract-v1",
        "claim_schema": "StructuredQA",
        "citation_validator": "strict_context_paper_page_block_triple",
        "reranker": {"enabled": False},
        "retry_policy": {
            "qa_generation_retry_count": 0,
            "json_repair": False,
            "citation_repair": False,
            "transport_retry_count": min(settings.llm_max_retries, 1),
        },
        "budget": {
            "source": "/api/v1/capabilities stage13_30_budget",
            "deepseek_canary_max_input_tokens": settings.deepseek_canary_max_input_tokens,
            "deepseek_canary_max_output_tokens": settings.deepseek_canary_max_output_tokens,
            "deepseek_canary_max_total_tokens": settings.deepseek_canary_max_total_tokens,
            "deepseek_canary_max_cost_usd": str(settings.deepseek_canary_max_cost_usd)
            if settings.deepseek_canary_max_cost_usd is not None
            else None,
            "deepseek_canary_max_total_seconds": settings.deepseek_canary_max_total_seconds,
        },
        "metric_semantics": {
            "required_claim_exact_match_coverage": "diagnostic_non_blocking",
            "gold_citation_exact_match_precision": "diagnostic_non_blocking",
            "gold_block_exact_recall": "diagnostic_non_blocking",
            "exact_gold_mismatch_claim_count": "diagnostic_non_blocking",
            "semantic_claim_support_audit": "NOT_FORMALLY_VALIDATED",
            "strong_grounding_claim_allowed": False,
        },
        "not_blind_holdout": True,
    }
    config["config_hash"] = hash_payload(config)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    OUT_DOC.write_text(
        "\n".join(
            [
                "# DeepSeek Full QA Final Config v1",
                "",
                f"- Branch/commit: `{config['branch']}` / `{config['commit']}`",
                f"- Dataset/hash: `{config['dataset']}` / `{config['dataset_hash']}`",
                f"- Provider/model: `{provider}` / `{settings.llm_model}`",
                f"- Prompt: `{settings.prompt_version}`",
                f"- Retrieval: `{config['retrieval']}`",
                f"- Embedding: `{config['embedding']}`",
                f"- Qdrant collection: `{settings.active_collection}`",
                "- Reranker: `disabled`",
                "- QA retry / JSON repair / citation repair: `0` / `false` / `false`",
                f"- Transport retry count: `{config['retry_policy']['transport_retry_count']}`",
                f"- Config hash: `{config['config_hash']}`",
                "",
                "Exact-Gold metrics are diagnostic and do not block the Portfolio "
                "engineering gate.",
                "This is a 50-item human-reviewed internal development evaluation, "
                "not a blind holdout.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"config": str(OUT_JSON), "config_hash": config["config_hash"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
