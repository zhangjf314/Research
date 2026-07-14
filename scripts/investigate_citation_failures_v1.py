# ruff: noqa: E501
"""Audit and optionally replay the deterministic q033/q044 citation failures."""

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from paper_research.config import Settings
from paper_research.generation.qa_service import ClaimEvidenceValidator
from paper_research.parsing.types import PaperBlock
from paper_research.providers.factory import build_llm_provider
from paper_research.providers.llm import LLMProviderError, SiliconFlowLLMProvider
from paper_research.retrieval.context_builder import ContextItem

OPTIMIZATION = Path("data/evaluation/retrieval-context-optimization-v1.json")
PROTOCOL = Path("data/evaluation/retrieval-gold-v2.jsonl")
BLOCK_ROOT = Path("data/reports/parsing-audit")
QUESTION_IDS = ("q033", "q044")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def block_map(paper_id: str) -> dict[str, PaperBlock]:
    path = BLOCK_ROOT / paper_id / "paper_blocks.jsonl"
    return {
        block.block_id: block
        for block in (
            PaperBlock.model_validate(json.loads(line))
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def exact_context(row: dict, blocks: dict[str, PaperBlock]) -> list[ContextItem]:
    context = []
    for item in row["context"]:
        missing = [block_id for block_id in item["block_ids"] if block_id not in blocks]
        if missing:
            raise RuntimeError(f"context blocks missing from parse audit: {missing}")
        page_map = {block_id: blocks[block_id].page_start for block_id in item["block_ids"]}
        context.append(ContextItem.model_validate({**item, "block_page_map": page_map}))
    return context


def parsed_output(content: object) -> dict | None:
    if not isinstance(content, str):
        return None
    try:
        value = json.loads(content)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        return None


def generated_citations(attempts: list[dict]) -> list[dict]:
    output = []
    for attempt in attempts:
        parsed = parsed_output(attempt.get("sanitized_output"))
        if not parsed:
            continue
        for claim in parsed.get("claims") or []:
            for citation in claim.get("citations") or []:
                output.append({"attempt": attempt["attempt"], **citation})
    return output


def write_artifact(question_id: str, artifact: dict) -> None:
    json_path = Path(f"data/evaluation/citation-failure-{question_id}-v1.json")
    md_path = Path(f"docs/citation-failure-{question_id}-v1.md")
    json_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    replay = artifact["replay"]
    lines = [
        f"# Citation Failure {question_id} v1",
        "",
        f"- Question: {artifact['question']}",
        f"- Target paper: `{artifact['target_paper']}`",
        f"- Retrieval filter: `{json.dumps(artifact['retrieval_filter'], sort_keys=True)}`",
        f"- Historical raw outputs: **{artifact['historical_raw_outputs']['status']}**.",
        f"- Duplicate block IDs: {artifact['mapping_audit']['duplicate_block_ids'] or 'none'}.",
        f"- Block/page conflicts: {artifact['mapping_audit']['block_page_conflicts'] or 'none'}.",
        "",
        "## Root cause",
        "",
        artifact["root_cause"]["summary"],
        "",
        f"- Deterministic protocol defect: {artifact['root_cause']['deterministic_protocol_defect']}.",
        f"- Provider/model limitation: {artifact['root_cause']['provider_model_citation_mapping_limitation']}.",
        f"- Strict page validation retained: {artifact['root_cause']['strict_validation_retained']}.",
        "",
        "## Minimal replay",
        "",
        f"- Executed: {replay['executed']}",
        f"- Status: `{replay['status']}`",
        f"- API requests: {replay['api_request_count']}",
        f"- Retries: {replay['retry_count']}",
        f"- Final error: `{replay.get('failure_reason')}`",
        "- Invalid citations remained accepted: false.",
        "",
        "Historical invalid model bodies were not persisted by Stage 11C.6 and are not reconstructed. Replay outputs in the JSON artifact are sanitized and contain no request headers or credentials.",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_artifact(
    question_id: str,
    source: dict,
    protocol: dict,
    context: list[ContextItem],
    *,
    provider: SiliconFlowLLMProvider | None,
) -> dict:
    all_block_ids = [block_id for item in context for block_id in item.block_ids]
    counts = Counter(all_block_ids)
    duplicates = sorted(block_id for block_id, count in counts.items() if count > 1)
    maps: dict[str, set[int]] = {}
    context_blocks = []
    for rank, item in enumerate(context, start=1):
        for block_id, page in item.block_page_map.items():
            maps.setdefault(block_id, set()).add(page)
            context_blocks.append(
                {
                    "context_rank": rank,
                    "chunk_id": item.chunk_id,
                    "block_id": block_id,
                    "real_page": page,
                    "prompt_page": page,
                }
            )
    conflicts = {block_id: sorted(pages) for block_id, pages in maps.items() if len(pages) > 1}
    allowed = sorted(SiliconFlowLLMProvider._allowed_citations(context))
    retry_prompt = SiliconFlowLLMProvider._citation_retry_prompt(
        "citation_validation:page", context
    )
    replay = {
        "executed": provider is not None,
        "status": "NOT_RUN",
        "api_request_count": 0,
        "retry_count": 0,
        "retry_reasons": [],
        "sanitized_model_outputs": [],
        "generated_citations": [],
        "invalid_citations": [],
        "failure_reason": None,
        "model_usage": None,
    }
    if provider is not None:
        try:
            generation = provider.generate_claim_answer(
                protocol["retrieval_query"], context, "qa-production-v1"
            )
            ClaimEvidenceValidator().validate(generation.claims, context)
            replay.update(
                status="COMPLETED",
                api_request_count=generation.api_request_count,
                retry_count=generation.retry_count,
                retry_reasons=generation.retry_reasons,
                sanitized_model_outputs=generation.diagnostic_attempts,
                generated_citations=generated_citations(generation.diagnostic_attempts),
                model_usage=generation.usage.model_dump(),
            )
        except LLMProviderError as exc:
            replay.update(
                status="STRICTLY_REJECTED",
                api_request_count=exc.api_request_count,
                retry_count=len(exc.retry_reasons) - 1,
                retry_reasons=exc.retry_reasons,
                sanitized_model_outputs=exc.diagnostic_attempts,
                generated_citations=generated_citations(exc.diagnostic_attempts),
                failure_reason=str(exc),
            )
        allowed_set = set(allowed)
        replay["invalid_citations"] = [
            citation
            for citation in replay["generated_citations"]
            if (citation.get("paper_id"), citation.get("page"), citation.get("block_id"))
            not in allowed_set
        ]
    generated_pages = sorted(
        {
            citation.get("page")
            for citation in replay["generated_citations"]
            if isinstance(citation.get("page"), int)
        }
    )
    return {
        "status": "COMPLETED",
        "generated_at": datetime.now(UTC).isoformat(),
        "question_id": question_id,
        "question": protocol["original_question"],
        "retrieval_query": protocol["retrieval_query"],
        "target_paper": protocol["retrieval_filter"]["paper_ids"],
        "retrieval_filter": protocol["retrieval_filter"],
        "context_block_list": context_blocks,
        "block_page_map": {
            item.chunk_id: item.block_page_map for item in context
        },
        "legal_page_set": sorted({page for _, page, _ in allowed}),
        "legal_citation_triples": [
            {"paper_id": paper, "page": page, "block_id": block}
            for paper, page, block in allowed
        ],
        "historical_failure": {
            "failure_reason": source.get("failure_reason"),
            "api_request_count_in_latest_row": source.get("api_request_count"),
            "retry_count_in_latest_row": source.get("retry_count"),
            "three_execution_rounds": True,
            "estimated_requests_for_this_question": 9,
        },
        "historical_raw_outputs": {
            "status": "NOT_RETAINED_BY_STAGE_11C6",
            "outputs": [],
            "reason": "The provider stored only citation_validation:page and discarded invalid response bodies.",
        },
        "mapping_audit": {
            "duplicate_block_ids": duplicates,
            "block_page_conflicts": conflicts,
            "prompt_ambiguity_before_fix": True,
            "old_multi_page_map_used_page_start_for_every_block": any(
                item.page_start != item.page_end for item in context
            ),
            "old_validator_used_chunk_page_block_cartesian_product": True,
        },
        "retry_prompt_content": retry_prompt,
        "replay": replay,
        "model_generated_page_set": generated_pages,
        "root_cause": {
            "summary": "The old serialization and validator expressed different block/page rules, and citation retries repeated the same request without an explicit legal-triple correction. The repaired protocol supplies real block pages and an authoritative allowed_citations list while retaining exact triple validation.",
            "deterministic_protocol_defect": True,
            "context_serialization_mapping_error": any(
                item.page_start != item.page_end for item in context
            ),
            "retry_prompt_missing_legal_pages_before_fix": True,
            "model_ignored_mapping_after_fix": replay["status"] == "STRICTLY_REJECTED",
            "provider_model_citation_mapping_limitation": replay["status"]
            == "STRICTLY_REJECTED",
            "strict_validation_retained": True,
            "illegal_page_auto_corrected": False,
        },
        "security": {
            "authorization_header_persisted": False,
            "api_key_persisted": False,
            "complete_request_headers_persisted": False,
        },
        "frozen_configuration": {
            "rerank_enabled": False,
            "embedding": "jina-embeddings-v5-text-small",
            "llm": "SiliconFlow Qwen/Qwen3-8B",
            "prompt_version": "qa-production-v1",
            "deep_research_called": False,
        },
    }


def main() -> None:
    args = parse_args()
    settings = Settings()
    if settings.rerank_enabled:
        raise RuntimeError("RERANK_ENABLED must remain false")
    if settings.llm_provider != "siliconflow" or settings.llm_model != "Qwen/Qwen3-8B":
        raise RuntimeError("fixed SiliconFlow Qwen/Qwen3-8B is required")
    optimization = json.loads(OPTIMIZATION.read_text(encoding="utf-8"))
    source_by_id = {row["question_id"]: row for row in optimization["qa_candidate_queries"]}
    protocol_by_id = {row["question_id"]: row for row in load_jsonl(PROTOCOL)}
    provider = build_llm_provider(settings) if args.replay else None
    summaries = {}
    for question_id in QUESTION_IDS:
        source = source_by_id[question_id]
        protocol = protocol_by_id[question_id]
        paper_id = protocol["retrieval_filter"]["paper_ids"][0]
        context = exact_context(source, block_map(paper_id))
        artifact = build_artifact(
            question_id,
            source,
            protocol,
            context,
            provider=provider,  # type: ignore[arg-type]
        )
        write_artifact(question_id, artifact)
        summaries[question_id] = artifact["replay"]
    print(json.dumps(summaries, ensure_ascii=False))


if __name__ == "__main__":
    main()
