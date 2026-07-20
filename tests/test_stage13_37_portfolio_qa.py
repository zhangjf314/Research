from __future__ import annotations

import scripts.run_deepseek_full_qa_final_v1 as final_qa
import scripts.run_evidence_first_canary_v1 as evidence_first


def test_exact_gold_metrics_are_diagnostic_not_engineering_blockers() -> None:
    payload = {
        "total": 50,
        "completed_count": 50,
        "failed_count": 0,
        "metrics": {
            "completed": 50,
            "failed": 0,
            "citation_id_validity": 1.0,
            "template_fallback_count": 0,
            "required_claim_coverage": 0.1,
            "citation_precision": 0.1,
            "citation_recall": 0.1,
            "unsupported_claim_count": 100,
        },
    }
    rows = [{"question_id": f"q{i:03d}", "status": "COMPLETED"} for i in range(1, 51)]
    original_reader = final_qa.read_jsonl
    original_head = final_qa.git_head
    try:
        final_qa.read_jsonl = lambda _path: rows  # type: ignore[assignment]
        final_qa.git_head = lambda: "abc123"  # type: ignore[assignment]
        result = final_qa.apply_portfolio_semantics(payload)
    finally:
        final_qa.read_jsonl = original_reader  # type: ignore[assignment]
        final_qa.git_head = original_head  # type: ignore[assignment]
    assert result["portfolio_qa_engineering_gate"] == "PASSED"
    assert result["metrics"]["gold_citation_exact_match_precision"] == 0.1
    assert result["metrics"]["semantic_claim_support_audit"] == "NOT_FORMALLY_VALIDATED"


def test_engineering_failure_blocks_deep_research() -> None:
    payload = {
        "total": 50,
        "completed_count": 47,
        "failed_count": 3,
        "metrics": {
            "completed": 47,
            "failed": 3,
            "citation_id_validity": 1.0,
            "template_fallback_count": 0,
        },
    }
    rows = [{"question_id": f"q{i:03d}", "status": "COMPLETED"} for i in range(1, 48)] + [
        {
            "question_id": "q048",
            "status": "FAILED",
            "provider_error_code": "CLAIM_QA_PROVIDER_TIMEOUT",
        },
        {
            "question_id": "q049",
            "status": "FAILED",
            "provider_error_code": "CLAIM_QA_PROVIDER_TIMEOUT",
        },
        {
            "question_id": "q050",
            "status": "FAILED",
            "provider_error_code": "CLAIM_QA_PROVIDER_TIMEOUT",
        },
    ]
    original_reader = final_qa.read_jsonl
    original_head = final_qa.git_head
    try:
        final_qa.read_jsonl = lambda _path: rows  # type: ignore[assignment]
        final_qa.git_head = lambda: "abc123"  # type: ignore[assignment]
        result = final_qa.apply_portfolio_semantics(payload)
    finally:
        final_qa.read_jsonl = original_reader  # type: ignore[assignment]
        final_qa.git_head = original_head  # type: ignore[assignment]
    assert result["portfolio_qa_engineering_gate"] == "FAILED"
    assert result["ready_for_production_deep_research"] is False


def test_evidence_first_is_experimental_not_default() -> None:
    assert evidence_first.EVIDENCE_FIRST_STATUS == "EXPERIMENTAL_FAILED"
    assert evidence_first.EVIDENCE_FIRST_DEFAULT is False
    assert evidence_first.EVIDENCE_FIRST_IDS == [
        "q014",
        "q020",
        "q024",
        "q001",
        "q049",
        "q005",
    ]
