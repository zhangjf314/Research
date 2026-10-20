"""Read-only source representation after the existing retrieval identity guard."""

import json
import logging
import math
import uuid
from pathlib import Path
from typing import Literal

from pydantic import Field

from paper_research.agents.research_agent.evidence_contract import (
    EvidenceContract,
    EvidenceGap,
    EvidenceRequirement,
    GapType,
    RequirementType,
    StrictModel,
    stable_contract_id,
)
from paper_research.agents.research_agent.models import EvidenceItem, EvidenceVerificationStatus
from paper_research.chunking.types import Chunk
from paper_research.parsing.types import PaperBlock
from paper_research.retrieval.context_builder import ContextItem
from paper_research.retrieval.trace import RetrievalTrace

logger = logging.getLogger(__name__)


class SourceLocator(StrictModel):
    paper_id: str
    block_id: str = Field(min_length=1)
    page: int = Field(ge=1)
    page_end: int = Field(ge=1)


class EvidenceProvenance(StrictModel):
    paper_id: str
    chunk_id: str = Field(min_length=1)
    block_ids: list[str] = Field(min_length=1)
    block_page_map: dict[str, int]
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    section_path: list[str]
    sources: list[SourceLocator] = Field(min_length=1)
    validation: Literal["CANONICAL_IDENTIFIERS_VALIDATED"] = "CANONICAL_IDENTIFIERS_VALIDATED"
    text_relation: Literal["UNCHANGED_CONTEXT_PREFIX_OF_CANONICAL_CHUNK"] = (
        "UNCHANGED_CONTEXT_PREFIX_OF_CANONICAL_CHUNK"
    )


class StructuredEvidence(StrictModel):
    item: EvidenceItem
    provenance: EvidenceProvenance


class SourceAvailability(StrictModel):
    requirement_id: Literal["R1"] = "R1"
    availability: Literal["AVAILABLE", "PARTIAL", "MISSING", "UNAVAILABLE"]
    evidence_ids: list[str]
    valid_evidence_count: int = Field(ge=0)
    invalid_evidence_count: int = Field(ge=0)
    unique_document_count: int = Field(ge=0)
    reason_code: Literal[
        "VALID_SOURCES_RETURNED",
        "SOME_SOURCES_REJECTED",
        "NO_RETURNED_EVIDENCE",
        "PROVENANCE_UNAVAILABLE",
        "ADAPTER_ERROR",
    ]
    semantic_status: Literal["NOT_ASSESSED"] = "NOT_ASSESSED"


class EvidenceAdapterError(StrictModel):
    code: Literal[
        "PROVENANCE_MISSING",
        "PROVENANCE_INVALID",
        "SOURCE_ID_UNKNOWN",
        "SOURCE_SCOPE_MISMATCH",
        "TEXT_MISMATCH",
        "ARTIFACT_INDEX_MISMATCH",
        "UNSUPPORTED_CONTEXT_MODE",
        "EVIDENCE_ADAPTER_ERROR",
    ]
    trace_id: str
    item_index: int | None
    paper_id: str | None
    chunk_id: str | None
    fields: list[str]


class RetrievalIdentity(StrictModel):
    embedding_provider: str
    embedding_model: str
    embedding_revision: str
    embedding_dimension: int
    index_version: str
    collection: str


class ReadOnlyEvidenceContractResult(StrictModel):
    schema_version: Literal["readonly-evidence-contract-v1"] = "readonly-evidence-contract-v1"
    status: Literal["OK", "PARTIAL", "ERROR"]
    trace_id: str
    retrieval_completed: Literal[True] = True
    identity_gate: Literal["UPSTREAM_P0_R1_SUCCEEDED"] = "UPSTREAM_P0_R1_SUCCEEDED"
    retrieval_identity: RetrievalIdentity
    semantic_status: Literal["NOT_ASSESSED"] = "NOT_ASSESSED"
    contract: EvidenceContract
    evidence: list[StructuredEvidence] = Field(max_length=20)
    coverage: list[SourceAvailability] = Field(min_length=1, max_length=1)
    gaps: list[EvidenceGap] = Field(max_length=1)
    errors: list[EvidenceAdapterError]


class _SourceError(ValueError):
    def __init__(self, code: str, fields: list[str]):
        self.code = code
        self.fields = fields
        super().__init__(code)


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _read_paper(root: Path, paper_id: str, identity: RetrievalIdentity):
    try:
        if str(uuid.UUID(paper_id)) != paper_id:
            raise ValueError("noncanonical paper UUID")
    except ValueError as exc:
        raise _SourceError("SOURCE_ID_UNKNOWN", ["paper_id"]) from exc
    root = root.resolve()
    folder = (root / paper_id).resolve()
    if not folder.is_relative_to(root) or folder.name != paper_id:
        raise _SourceError("PROVENANCE_INVALID", ["paper_id"])
    chunks_path = folder / f"paper_chunks.{identity.index_version}.jsonl"
    paths = [
        chunks_path,
        chunks_path.with_name(f"{chunks_path.stem}_manifest.json"),
        folder / "paper_blocks.jsonl",
    ]
    if any(not path.resolve().is_relative_to(folder) for path in paths):
        raise _SourceError("PROVENANCE_INVALID", ["index_version"])
    if any(not path.is_file() for path in paths):
        raise _SourceError("PROVENANCE_MISSING", ["canonical_artifacts"])
    try:
        manifest = _read_json(paths[1])
        expected = {**identity.model_dump(), "paper_id": paper_id}
        mismatches = [key for key, value in expected.items() if manifest.get(key) != value]
        if mismatches:
            raise _SourceError("ARTIFACT_INDEX_MISMATCH", mismatches)
        chunks = {}
        for raw in _read_jsonl(chunks_path):
            if not raw.get("chunk_id") or raw.get("paper_id") != paper_id:
                raise _SourceError("SOURCE_ID_UNKNOWN", ["paper_id", "chunk_id"])
            chunk = Chunk.model_validate(raw)
            if chunk.chunk_id in chunks:
                raise _SourceError("PROVENANCE_INVALID", ["chunk_id"])
            chunks[chunk.chunk_id] = chunk
        blocks = {}
        for raw in _read_jsonl(paths[2]):
            block = PaperBlock.model_validate(raw)
            if (
                not block.block_id
                or block.block_id in blocks
                or block.page_start < 1
                or block.page_end < block.page_start
                or (block.source_page is not None and block.source_page != block.page_start)
            ):
                raise _SourceError("PROVENANCE_INVALID", ["block_id", "page"])
            blocks[block.block_id] = block
        if manifest.get("chunk_count") != len(chunks):
            raise _SourceError("PROVENANCE_INVALID", ["chunk_count"])
        return chunks, blocks
    except _SourceError:
        raise
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        raise _SourceError("PROVENANCE_INVALID", ["canonical_artifacts"]) from exc


def _convert(item: ContextItem, chunks: dict, blocks: dict) -> StructuredEvidence:
    item = ContextItem.model_validate(item.model_dump())
    chunk = chunks.get(item.chunk_id)
    if chunk is None or chunk.paper_id != item.paper_id:
        raise _SourceError("SOURCE_ID_UNKNOWN", ["paper_id", "chunk_id"])
    fields = [
        key
        for key in ("block_ids", "section_path", "page_start", "page_end")
        if getattr(item, key) != getattr(chunk, key)
    ]
    if (
        fields
        or not item.chunk_id
        or not item.block_ids
        or len(set(item.block_ids)) != len(item.block_ids)
        or item.page_start < 1
        or item.page_end < item.page_start
    ):
        raise _SourceError("PROVENANCE_INVALID", fields or ["block_ids", "page_span"])
    if any(block_id not in blocks for block_id in item.block_ids):
        raise _SourceError("SOURCE_ID_UNKNOWN", ["block_ids"])
    expected_map = {block_id: blocks[block_id].page_start for block_id in item.block_ids}
    if (
        item.block_page_map != expected_map
        or (chunk.block_page_map is not None and chunk.block_page_map != expected_map)
        or any(
            not block_id or blocks[block_id].page_end > item.page_end for block_id in item.block_ids
        )
    ):
        raise _SourceError("PROVENANCE_INVALID", ["block_page_map"])
    if not item.evidence or item.evidence != chunk.chunk_text[: len(item.evidence)]:
        raise _SourceError("TEXT_MISMATCH", ["evidence"])
    if not math.isfinite(item.score):
        raise _SourceError("PROVENANCE_INVALID", ["score"])
    primary = item.block_ids[0]
    return StructuredEvidence(
        item=EvidenceItem(
            evidence_id=f"{item.paper_id}:{item.chunk_id}",
            paper_id=item.paper_id,
            block_id=primary,
            page=item.block_page_map[primary],
            section=" > ".join(item.section_path),
            text_or_reference=item.evidence,
            source_type="local_retrieval",
            supports=[],
            contradicts=[],
            discovered_by_tool="readonly_retrieve_adapter",
            discovered_at_step=0,
            relevance=item.score,
            verification_status=EvidenceVerificationStatus.CANDIDATE,
            chunk_id=item.chunk_id,
            block_ids=list(item.block_ids),
            block_page_map=dict(item.block_page_map),
            page_end=item.page_end,
        ),
        provenance=EvidenceProvenance(
            paper_id=item.paper_id,
            chunk_id=item.chunk_id,
            block_ids=list(item.block_ids),
            block_page_map=dict(item.block_page_map),
            page_start=item.page_start,
            page_end=item.page_end,
            section_path=list(item.section_path),
            sources=[
                SourceLocator(
                    paper_id=item.paper_id,
                    block_id=block_id,
                    page=blocks[block_id].page_start,
                    page_end=blocks[block_id].page_end,
                )
                for block_id in item.block_ids
            ],
        ),
    )


def _error(trace, index, item, code, fields):
    logger.warning(
        "Evidence adapter rejected code=%s trace_id=%s item_index=%s", code, trace.trace_id, index
    )
    return EvidenceAdapterError(
        code=code,
        trace_id=trace.trace_id,
        item_index=index,
        paper_id=item.paper_id if item else None,
        chunk_id=item.chunk_id if item else None,
        fields=fields,
    )


def _result(query, trace, evidence, errors, invalid_count=0, *, failed=False):
    if failed:
        status, availability, reason = "ERROR", "UNAVAILABLE", "ADAPTER_ERROR"
    elif errors:
        status = "PARTIAL" if evidence else "ERROR"
        availability = "PARTIAL" if evidence else "UNAVAILABLE"
        reason = "SOME_SOURCES_REJECTED" if evidence else "PROVENANCE_UNAVAILABLE"
    elif evidence:
        status, availability, reason = "OK", "AVAILABLE", "VALID_SOURCES_RETURNED"
    else:
        status, availability, reason = "OK", "MISSING", "NO_RETURNED_EVIDENCE"
    contract = EvidenceContract(
        contract_id=stable_contract_id(query),
        question=query,
        created_by="readonly_retrieve_adapter",
        requirements=[
            EvidenceRequirement(
                requirement_id="R1",
                description=(
                    "Observe resolvable document/block/page provenance for this retrieval response."
                ),
                requirement_type=RequirementType.SOURCE,
                source_scope=list(trace.filters.get("paper_ids") or []),
            )
        ],
    )
    gaps = []
    if availability != "AVAILABLE":
        gaps.append(
            EvidenceGap(
                gap_id=f"{contract.contract_id}:v1:R1:MISSING_EVIDENCE",
                requirement_id="R1",
                gap_type=GapType.MISSING_EVIDENCE,
                description=reason,
            )
        )
    return ReadOnlyEvidenceContractResult(
        status=status,
        trace_id=trace.trace_id,
        retrieval_identity=RetrievalIdentity(
            **{key: getattr(trace, key) for key in RetrievalIdentity.model_fields}
        ),
        contract=contract,
        evidence=evidence,
        gaps=gaps,
        errors=errors,
        coverage=[
            SourceAvailability(
                availability=availability,
                evidence_ids=[e.item.evidence_id for e in evidence],
                valid_evidence_count=len(evidence),
                invalid_evidence_count=invalid_count,
                unique_document_count=len({e.item.paper_id for e in evidence}),
                reason_code=reason,
            )
        ],
    )


def evidence_contract_failure(query: str, context: list[ContextItem], trace: RetrievalTrace):
    """Adapter-stage failure only; upstream retrieval exceptions must propagate."""
    return _result(
        query,
        trace,
        [],
        [_error(trace, None, None, "EVIDENCE_ADAPTER_ERROR", [])],
        len(context),
        failed=True,
    )


def build_readonly_evidence_contract(
    query: str,
    context: list[ContextItem],
    trace: RetrievalTrace,
    *,
    parsed_root: Path,
) -> ReadOnlyEvidenceContractResult:
    """Validate existing source identities without changing context or semantic status."""
    evidence, errors, papers, seen = [], [], {}, set()
    try:
        identity = RetrievalIdentity(
            **{key: getattr(trace, key) for key in RetrievalIdentity.model_fields}
        )
        scope = trace.filters.get("paper_ids") or []
        groups = {}
        for item in context:
            groups.setdefault((item.paper_id, item.chunk_id), []).append(item.model_dump())
        conflicting = {key for key, rows in groups.items() if any(row != rows[0] for row in rows)}
        for index, item in enumerate(context):
            try:
                if trace.contextual_retrieval or trace.contextual_sparse:
                    raise _SourceError("UNSUPPORTED_CONTEXT_MODE", ["contextual_retrieval"])
                key = (item.paper_id, item.chunk_id)
                if key in conflicting:
                    raise _SourceError("PROVENANCE_INVALID", ["duplicate_identity"])
                if scope and item.paper_id not in scope:
                    raise _SourceError("SOURCE_SCOPE_MISMATCH", ["paper_id"])
                if item.paper_id not in papers:
                    papers[item.paper_id] = _read_paper(parsed_root, item.paper_id, identity)
                converted = _convert(item, *papers[item.paper_id])
                if key not in seen:
                    evidence.append(converted)
                    seen.add(key)
            except _SourceError as exc:
                errors.append(_error(trace, index, item, exc.code, exc.fields))
            except ValueError:
                errors.append(_error(trace, index, item, "PROVENANCE_INVALID", ["context"]))
        return _result(query, trace, evidence, errors, len(errors))
    except Exception:
        return evidence_contract_failure(query, context, trace)
