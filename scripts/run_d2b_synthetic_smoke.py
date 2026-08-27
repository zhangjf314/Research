"""D2B synthetic-only selector output-capability smoke; no evaluation questions."""
# ruff: noqa
from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv

from paper_research.config import Settings
from paper_research.providers.factory import build_llm_provider
from paper_research.providers.llm import LLMProviderError
from run_d2_listwise_evaluation import CANONICAL_ROOT, ROOT, SYSTEM_PROMPT, validate_selection

OUT = ROOT / "artifacts/rag-quality-v3/d2b/smoke/d2b-selector-synthetic-smoke-v1.json"
BUDGETS = (256, 512, 1024)
STABILITY_ATTEMPTS = 2


def synthetic_prompt() -> str:
    candidates = [
        {
            "candidate_id": f"synthetic-unit-{index:02d}",
            "candidate_text": f"Fabricated neutral evidence passage {index}; this text has no development or blind evaluation content.",
            "canonical_document_id": f"synthetic-document-{(index - 1) // 4 + 1}",
            "source_spans": [[index, index]],
            "reranker_rank": index,
            "reranker_score": round(1 - index / 100, 3),
        }
        for index in range(1, 21)
    ]
    return json.dumps(
        {
            "question": "Synthetic capability check: choose five complementary fabricated evidence candidates.",
            "candidates": candidates,
        },
        ensure_ascii=False,
    )


def save(value: dict) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    load_dotenv(CANONICAL_ROOT / ".env", override=True)
    settings = Settings()
    provider = build_llm_provider(settings)
    allowed = [f"synthetic-unit-{index:02d}" for index in range(1, 21)]
    prompt = synthetic_prompt()
    attempts: list[dict] = []
    selected_budget: int | None = None
    for budget in BUDGETS:
        valid = 0
        for number in range(1, STABILITY_ATTEMPTS + 1):
            try:
                response = provider.generate_structured_json(
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=prompt,
                    schema_name="ragq3-d2b-listwise-selector-synthetic-v1",
                    request_context={"task_id": f"d2b-synthetic-{budget}-{number}", "run_id": "ragq3-d2b-smoke"},
                    max_output_tokens=budget,
                )
                selected = validate_selection(response.payload, allowed)
                valid += 1
                attempts.append({"budget": budget, "attempt": number, "status": "VALID", "selected_candidate_ids": selected, "requests": response.request_attempt_count, "finish_reason": "non_length"})
            except LLMProviderError as exc:
                attempts.append({"budget": budget, "attempt": number, "status": "INVALID", "error_code": exc.error_code, "requests": exc.api_request_count, "finish_reason": "length" if "finish_reason:length" in exc.retry_reasons else "provider_or_schema_error"})
                break
            except (TypeError, ValueError) as exc:
                attempts.append({"budget": budget, "attempt": number, "status": "INVALID", "error_code": type(exc).__name__, "requests": 1, "finish_reason": "schema_invalid"})
                break
        if valid == STABILITY_ATTEMPTS:
            selected_budget = budget
            break
    result = {
        "schema_version": "ragq3-d2b-synthetic-selector-smoke-v1",
        "scope": "synthetic_only_no_dev_or_postblind_questions",
        "provider": {"name": settings.llm_provider_name or settings.llm_provider, "model": settings.llm_model, "temperature": settings.llm_temperature},
        "budgets_in_order": list(BUDGETS),
        "stability_requirement": f"{STABILITY_ATTEMPTS} consecutive valid strict-schema outputs at one budget",
        "attempts": attempts,
        "selected_max_output_tokens": selected_budget,
        "status": "PASS" if selected_budget is not None else "D2B_SELECTOR_OUTPUT_ENVIRONMENT_BLOCKED",
        "quality_calls_before_freeze": 0,
    }
    save(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
