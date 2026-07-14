import json
from pathlib import Path

import httpx
import pytest

import scripts.investigate_citation_failures_v1 as failures
import scripts.summarize_citation_human_audit_v1 as audit
from paper_research.generation.qa_service import ClaimEvidenceValidator, ClaimValidationError
from paper_research.providers.llm import LLMProviderError, SiliconFlowLLMProvider
from paper_research.retrieval.context_builder import ContextItem


def reviewed_rows() -> list[dict]:
    return audit.load_jsonl(audit.INPUT)


def response(body: dict) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": "Qwen/Qwen3-8B",
            "choices": [{"message": {"content": json.dumps(body)}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        },
    )


def answer(page: int) -> dict:
    return {
        "answerable": True,
        "answer": "A supported claim.",
        "claims": [
            {
                "claim_id": "c1",
                "text": "A supported claim.",
                "citations": [{"paper_id": "paper", "page": page, "block_id": "b1"}],
            }
        ],
        "refusal_reason": None,
    }


def exact_context() -> list[ContextItem]:
    return [
        ContextItem(
            chunk_id="chunk",
            paper_id="paper",
            block_ids=["b1", "b2"],
            block_page_map={"b1": 3, "b2": 4},
            section_path=["Method"],
            page_start=3,
            page_end=4,
            evidence="Evidence",
            score=1,
        )
    ]


def test_review_file_schema_distribution_and_immutable_fields() -> None:
    rows = reviewed_rows()
    result = audit.validate(rows)

    assert all(result["checks"].values())
    assert len(rows) == 30
    assert sum(row["human_review_status"] == "pending" for row in rows) == 0
    assert [row["sample_id"] for row in rows] == [
        f"citation-audit-v1-{index:03d}" for index in range(1, 31)
    ]


def test_strict_lenient_rates_and_related_is_not_supported() -> None:
    rows = audit.enrich(reviewed_rows())
    summary = audit.summarize_group(rows)

    assert audit.support_flags("related_but_insufficient") == (False, False)
    assert summary["strict_support_rate"] == 0.166667
    assert summary["lenient_support_rate"] == 0.233333
    assert summary["gold_annotation_too_narrow_count"] == 0


def test_confusion_matrix_and_automated_precision() -> None:
    rows = audit.enrich(reviewed_rows())
    metrics = audit.automated_metrics(rows)

    assert audit.confusion(rows, lenient=False) == {
        "true_positive": 5,
        "false_positive": 15,
        "true_negative": 10,
        "false_negative": 0,
    }
    assert audit.confusion(rows, lenient=True) == {
        "true_positive": 6,
        "false_positive": 14,
        "true_negative": 9,
        "false_negative": 1,
    }
    assert metrics["semantic_support_non_gold"]["strict_precision"] == 0.3
    assert metrics["semantic_support_non_gold"]["lenient_precision"] == 0.4
    assert metrics["same_gold_page"]["fully_unsupported_count"] == 6
    assert metrics["weakly_related"]["fully_unsupported_count"] == 6
    assert metrics["unsupported"]["unsupported_precision"] == 1.0


def test_exact_block_page_map_is_authoritative_and_never_auto_corrected() -> None:
    context = exact_context()
    payload = SiliconFlowLLMProvider._evidence_payload(context)

    assert payload[0]["block_page_map"] == {"b1": 3, "b2": 4}
    assert {tuple(item.values()) for item in payload[0]["allowed_citations"]} == {
        ("paper", 3, "b1"),
        ("paper", 4, "b2"),
    }
    generated = SiliconFlowLLMProvider._allowed_citations(context)
    assert ("paper", 4, "b1") not in generated
    with pytest.raises(ClaimValidationError):
        ClaimEvidenceValidator().validate(
            [
                type(
                    "Claim",
                    (),
                    {
                        "claim_id": "c1",
                        "text": "claim",
                        "citations": [
                            type(
                                "Citation",
                                (),
                                {"paper_id": "paper", "page": 4, "block_id": "b1"},
                            )()
                        ],
                    },
                )()
            ],
            context,
        )


def test_citation_retry_lists_legal_triples_and_remains_bounded(monkeypatch) -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        return response(answer(4))

    monkeypatch.setattr("paper_research.providers.llm.time.sleep", lambda _seconds: None)
    provider = SiliconFlowLLMProvider(
        "https://api.siliconflow.cn/v1",
        "test-secret",
        "Qwen/Qwen3-8B",
        max_retries=1,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(LLMProviderError) as captured:
        provider.generate_claim_answer("question", exact_context(), "qa-production-v1")

    assert captured.value.api_request_count == 2
    assert len(calls) == 2
    assert "allowed triples" in calls[1]["messages"][2]["content"]
    assert '"page": 3, "block_id": "b1"' in calls[1]["messages"][2]["content"]
    assert answer(4)["claims"][0]["citations"][0]["page"] == 4


def test_failure_artifacts_are_sanitized_and_frozen() -> None:
    for question_id in failures.QUESTION_IDS:
        artifact = json.loads(
            Path(f"data/evaluation/citation-failure-{question_id}-v1.json").read_text(
                encoding="utf-8"
            )
        )
        serialized = json.dumps(artifact)
        assert artifact["historical_raw_outputs"]["status"] == "NOT_RETAINED_BY_STAGE_11C6"
        assert artifact["replay"]["status"] == "COMPLETED"
        assert artifact["replay"]["api_request_count"] == 2
        assert artifact["replay"]["retry_count"] == 1
        assert artifact["replay"]["invalid_citations"]
        assert artifact["root_cause"]["strict_validation_retained"] is True
        assert artifact["root_cause"]["illegal_page_auto_corrected"] is False
        assert artifact["root_cause"]["provider_model_citation_mapping_limitation"] is False
        assert artifact["frozen_configuration"]["rerank_enabled"] is False
        assert artifact["frozen_configuration"]["deep_research_called"] is False
        assert "Authorization" not in serialized
        assert "api_key\"" not in serialized
