from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from paper_research.api.routes.rag import _with_block_page_map
from paper_research.chunking.types import Chunk
from paper_research.generation.prompts import qa_system_prompt
from paper_research.providers.llm import (
    LLMProviderError,
    SiliconFlowLLMProvider,
    StructuredQA,
    normalize_structured_qa_content,
)
from paper_research.retrieval.context_builder import ContextBuilder, ContextItem
from paper_research.retrieval.fusion import FusedResult
from paper_research.retrieval.hybrid import HybridRetriever


def test_prompt_misconfiguration_is_not_silent_template_fallback():
    provider = SiliconFlowLLMProvider(
        "https://example.invalid/v1",
        "sk-test",
        "Qwen/Qwen3-8B",
        max_retries=0,
    )

    with pytest.raises(LLMProviderError) as exc:
        provider.generate_claim_answer("question", [], "claim-qa-v1")

    assert exc.value.error_code == "CLAIM_QA_CONFIGURATION_ERROR"
    assert exc.value.stage == "LLM_REQUEST_BUILD"
    assert provider.provider_name == "siliconflow"


def test_unsupported_prompt_version_remains_explicit():
    with pytest.raises(ValueError, match="unsupported production QA prompt version"):
        qa_system_prompt("claim-qa-v1")


def test_production_prompt_constrains_claim_schema_keys():
    prompt = qa_system_prompt("qa-production-v1")

    assert '"text": string' in prompt
    assert '"citation_keys": ["C1"]' in prompt
    assert "Do not output claim_id, paper_id, block_id, page" in prompt
    assert '"claims":[{"text":"One atomic claim.","citation_keys":["C1"]}]' in prompt
    assert "Do not output <think>" in prompt
    assert "answer <= 80 words" in prompt
    assert "at most 3 claims" in prompt


def test_normalize_accepts_fenced_json_and_page_string():
    content = """```json
    {"answerable":true,"answer":"A","claims":{"claim_id":"c1","text":"A",
    "citations":[{"paper_id":"p1","page":"7","block_id":"b1"}]},"refusal_reason":null}
    ```"""

    payload, events = normalize_structured_qa_content(content)
    parsed = StructuredQA.model_validate(payload)

    assert "removed_markdown_fence" in events
    assert "wrapped_single_claim_object" in events
    assert "coerced_page_string_to_int" in events
    assert parsed.claims[0].citations[0].page == 7


def test_normalize_removes_qwen_think_block_without_repairing_citations():
    content = (
        "<think>private reasoning</think>\n"
        '{"answerable":false,"answer":null,"claims":[],"refusal_reason":"No evidence."}'
    )

    payload, events = normalize_structured_qa_content(content)
    parsed = StructuredQA.model_validate(payload)

    assert "removed_think_block" in events
    assert parsed.answerable is False
    assert parsed.claims == []


def test_normalize_maps_citation_keys_to_server_owned_triples():
    content = json.dumps(
        {
            "answer": "A",
            "insufficient_evidence": False,
            "claims": [{"text": "A", "citation_keys": ["C1"]}],
        }
    )

    payload, events = normalize_structured_qa_content(
        content,
        citation_key_map={
            "C1": {"paper_id": "p1", "page": 7, "block_id": "b1", "chunk_id": "chunk"}
        },
    )
    parsed = StructuredQA.model_validate(payload)

    assert "mapped_citation_keys_to_server_triples" in events
    assert parsed.claims[0].claim_id == "c1"
    assert parsed.claims[0].citations[0].page == 7


def test_normalize_rejects_free_page_fields_in_citation_key_contract():
    content = json.dumps(
        {
            "answer": "A",
            "insufficient_evidence": False,
            "claims": [{"text": "A", "citation_keys": ["C1"], "page": 99}],
        }
    )

    with pytest.raises(ValueError, match="service-owned fields"):
        normalize_structured_qa_content(
            content,
            citation_key_map={
                "C1": {"paper_id": "p1", "page": 7, "block_id": "b1", "chunk_id": "chunk"}
            },
        )


def test_answerable_empty_claims_still_fails_schema():
    content = '{"answerable":true,"answer":"A","claims":[],"refusal_reason":null}'
    payload, _events = normalize_structured_qa_content(content)

    with pytest.raises(ValidationError):
        StructuredQA.model_validate(payload)


def test_unanswerable_with_citation_still_fails_schema():
    content = json.dumps(
        {
            "answerable": False,
            "answer": None,
            "claims": [
                {
                    "claim_id": "c1",
                    "text": "A",
                    "citations": [{"paper_id": "p1", "page": 1, "block_id": "b1"}],
                }
            ],
            "refusal_reason": "No evidence.",
        }
    )
    payload, _events = normalize_structured_qa_content(content)

    with pytest.raises(ValidationError):
        StructuredQA.model_validate(payload)


def test_unknown_citation_triple_fails_without_repair():
    context = [
        ContextItem(
            chunk_id="c1",
            paper_id="p1",
            block_ids=["b1"],
            block_page_map={"b1": 1},
            section_path=["Methods"],
            page_start=1,
            page_end=1,
            evidence="Evidence",
            score=1.0,
        )
    ]
    answer = StructuredQA.model_validate(
        {
            "answerable": True,
            "answer": "A",
            "claims": [
                {
                    "claim_id": "c1",
                    "text": "A",
                    "citations": [{"paper_id": "p1", "page": 2, "block_id": "b1"}],
                }
            ],
            "refusal_reason": None,
        }
    )

    with pytest.raises(ValueError):
        SiliconFlowLLMProvider._validate_context_citations(answer, context)


def make_fused(chunk_id: str, section: str, page: int) -> FusedResult:
    return FusedResult(
        chunk=Chunk(
            chunk_id=chunk_id,
            paper_id="paper",
            block_ids=[chunk_id.replace("chunk", "block")],
            section_path=[section],
            block_type="paragraph",
            page_start=page,
            page_end=page,
            chunk_text=f"{section} text",
            token_count=10,
        ),
        score=1.0,
    )


def test_paper_contribution_context_prioritizes_introduction_without_gold_injection():
    results = [
        make_fused("chunk-ref-1", "References", 11),
        make_fused("chunk-ref-2", "References", 12),
        make_fused("chunk-visual", "Attention Visualizations", 15),
        make_fused("chunk-conclusion", "7\nConclusion", 10),
        make_fused("chunk-intro", "1\nIntroduction", 2),
    ]

    ordered = HybridRetriever._context_candidates(
        "What are the target paper's main contributions?",
        results,
        top_k=3,
        retrieval_scope="paper",
    )

    assert [item.chunk.chunk_id for item in ordered] == [
        "chunk-intro",
        "chunk-conclusion",
        "chunk-visual",
    ]


def test_non_contribution_context_keeps_retrieval_order():
    results = [
        make_fused("chunk-ref-1", "References", 11),
        make_fused("chunk-intro", "1\nIntroduction", 2),
    ]

    ordered = HybridRetriever._context_candidates(
        "What optimizer did the paper use?",
        results,
        top_k=2,
        retrieval_scope="paper",
    )

    assert [item.chunk.chunk_id for item in ordered] == ["chunk-ref-1", "chunk-intro"]


def test_experiment_design_context_prioritizes_design_evidence_without_gold_ids():
    results = [
        make_fused("chunk-ref", "References", 29),
        make_fused("chunk-appendix", "A\nSummary of Power Laws", 20),
        make_fused("chunk-late", "5\nScaling Laws with Model Size and Training Time", 14),
        FusedResult(
            chunk=Chunk(
                chunk_id="chunk-design",
                paper_id="paper",
                block_ids=["block-design"],
                section_path=["3\nEmpirical Results and Basic Power Laws"],
                block_type="paragraph",
                page_start=7,
                page_end=8,
                chunk_text=(
                    "To characterize language model scaling we train a wide variety "
                    "of models, varying a number of factors including model size, "
                    "dataset size, and shape including depth, width, attention heads, "
                    "and feed-forward dimension."
                ),
                token_count=60,
            ),
            score=0.01,
        ),
    ]

    ordered = HybridRetriever._context_candidates(
        "How are the target paper's experiments designed and evaluated?",
        results,
        top_k=2,
        retrieval_scope="paper",
    )

    assert [item.chunk.chunk_id for item in ordered] == ["chunk-design", "chunk-late"]


def test_experiment_design_context_prioritizes_results_evaluation_signals():
    results = [
        make_fused("chunk-approach", "2\nApproach", 9),
        FusedResult(
            chunk=Chunk(
                chunk_id="chunk-results-a",
                paper_id="paper",
                block_ids=["block-results-a"],
                section_path=["3\nResults"],
                block_type="paragraph",
                page_start=10,
                page_end=10,
                chunk_text=(
                    "Below, we evaluate the 8 models described in Section 2 on a "
                    "wide range of datasets. We group the datasets into 9 categories "
                    "representing roughly similar tasks."
                ),
                token_count=50,
            ),
            score=0.01,
        ),
        FusedResult(
            chunk=Chunk(
                chunk_id="chunk-results-b",
                paper_id="paper",
                block_ids=["block-results-b"],
                section_path=["3\nResults"],
                block_type="paragraph",
                page_start=11,
                page_end=11,
                chunk_text=(
                    "Training curves show that performance follows a power-law "
                    "trend, and the scaling curves are evaluated task by task."
                ),
                token_count=40,
            ),
            score=0.02,
        ),
    ]

    ordered = HybridRetriever._context_candidates(
        "How are the target paper's experiments designed and evaluated?",
        results,
        top_k=2,
        retrieval_scope="paper",
    )

    assert [item.chunk.chunk_id for item in ordered] == [
        "chunk-results-a",
        "chunk-results-b",
    ]


def test_context_builder_materializes_deterministic_block_page_map_for_legacy_chunks():
    chunk = Chunk(
        chunk_id="chunk-multipage",
        paper_id="paper",
        block_ids=["b1", "b2"],
        section_path=["Methods"],
        block_type="paragraph",
        page_start=15,
        page_end=16,
        chunk_text="multi page evidence",
        token_count=10,
    )
    context = ContextBuilder(include_neighbors=False).build([FusedResult(chunk=chunk, score=1.0)])

    assert context[0].block_page_map == {"b1": 15, "b2": 15}


def test_model_visible_evidence_payload_uses_allowed_pages_not_page_range():
    context = [
        ContextItem(
            chunk_id="chunk-multipage",
            paper_id="paper",
            block_ids=["b1", "b2"],
            block_page_map={"b1": 15, "b2": 15},
            section_path=["Methods"],
            page_start=15,
            page_end=16,
            evidence="multi page evidence",
            score=1.0,
        )
    ]

    payload = SiliconFlowLLMProvider._evidence_payload(context)

    assert "pages" not in payload[0]
    assert "allowed_citations" not in payload[0]
    assert payload[0]["citation_keys"] == [
        {"key": "C1", "paper_id": "paper", "block_id": "b1", "page": 15},
        {"key": "C2", "paper_id": "paper", "block_id": "b2", "page": 15},
    ]


def test_api_chunk_loader_backfills_true_block_page_map_for_multipage_chunks():
    chunk = Chunk(
        chunk_id="chunk-results",
        paper_id="paper",
        block_ids=["b113", "b115"],
        section_path=["3\nResults"],
        block_type="paragraph",
        page_start=10,
        page_end=11,
        chunk_text="results evidence",
        token_count=10,
    )

    enriched = _with_block_page_map(chunk, {"b113": 10, "b115": 11})

    assert enriched.block_page_map == {"b113": 10, "b115": 11}
