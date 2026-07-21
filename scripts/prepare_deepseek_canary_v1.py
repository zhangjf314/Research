"""Freeze the Stage 13.35 DeepSeek canary configuration without secrets."""

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
from scripts.run_full_qa_canary_v2 import CANARY_IDS, RETRIEVAL_GOLD  # noqa: E402

OUT_JSON = ROOT / "data" / "evaluation" / "deepseek-full-qa-canary-config-v1.json"
OUT_DOC = ROOT / "docs" / "deepseek-full-qa-canary-config-v1.md"


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _hash_payload(payload: dict[str, Any]) -> str:
    stable = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(stable.encode("utf-8")).hexdigest()


def main() -> int:
    settings = Settings()
    provider = settings.llm_provider_name or settings.llm_provider
    config: dict[str, Any] = {
        "schema_version": "deepseek-full-qa-canary-config-v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "git_commit": _git_head(),
        "dataset": "gold-dev-v1",
        "dataset_hash": _sha256_path(RETRIEVAL_GOLD),
        "sample_ids": CANARY_IDS,
        "retrieval": {
            "recall_k": settings.retrieval_recall_k,
            "top_k": 10,
            "context_token_budget": settings.qa_context_token_budget,
            "index_version": settings.index_version,
            "collection": settings.active_collection,
        },
        "embedding": {
            "provider": settings.embedding_provider,
            "model": settings.embedding_model,
            "dimension": settings.embedding_dimensions,
        },
        "reranker": {"enabled": settings.rerank_enabled},
        "llm": {
            "provider": provider,
            "adapter": settings.llm_provider,
            "model": settings.llm_model,
            "base_url_host": "api.deepseek.com"
            if settings.llm_base_url and "api.deepseek.com" in settings.llm_base_url
            else None,
            "thinking": "enabled" if settings.llm_thinking_enabled else "disabled",
            "response_format": settings.llm_response_format,
            "stream": settings.llm_stream,
            "temperature": settings.llm_temperature,
            "max_tokens": settings.llm_max_output_tokens,
            "api_key_present": bool(settings.llm_api_key),
            "api_key_recorded": False,
        },
        "prompt_version": settings.prompt_version,
        "normalizer_version": "citation-key-minimal-contract-v1",
        "claim_schema_version": "StructuredQA",
        "budget": {
            "max_input_tokens": settings.deepseek_canary_max_input_tokens,
            "max_output_tokens": settings.deepseek_canary_max_output_tokens,
            "max_total_tokens": settings.deepseek_canary_max_total_tokens,
            "max_cost_usd": (
                str(settings.deepseek_canary_max_cost_usd)
                if settings.deepseek_canary_max_cost_usd is not None
                else None
            ),
            "max_total_seconds": settings.deepseek_canary_max_total_seconds,
        },
        "not_blind_holdout": True,
    }
    config["configuration_sha256"] = _hash_payload(config)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    OUT_DOC.write_text(
        "\n".join(
            [
                "# DeepSeek Full QA Canary Config v1",
                "",
                f"- Provider/model: `{provider}` / `{settings.llm_model}`",
                f"- Sample IDs: `{', '.join(CANARY_IDS)}`",
                f"- Retrieval config: `{config['retrieval']}`",
                f"- Reranker enabled: `{settings.rerank_enabled}`",
                f"- Thinking: `{config['llm']['thinking']}`",
                f"- Response format: `{settings.llm_response_format}`",
                f"- Configuration SHA-256: `{config['configuration_sha256']}`",
                "",
                "This canary reuses the same internal development samples as Qwen v2; "
                "it is not a blind benchmark.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"config": str(OUT_JSON), "sha256": config["configuration_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
