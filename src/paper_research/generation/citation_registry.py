"""Deterministic citation-ID registry for the Stage 13.3 citation-id-v2 protocol."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable

from pydantic import BaseModel, Field, model_validator

from paper_research.retrieval.context_builder import ContextItem


class CitationRegistryError(ValueError):
    """Raised when model citation IDs cannot be resolved exactly."""


class CitationRegistryEntry(BaseModel):
    citation_id: str = Field(pattern=r"^E\d{3}$")
    evidence_id: str
    paper_id: str
    page: int = Field(ge=1)
    block_id: str
    claim_ids: list[str] = Field(default_factory=list)
    context_position: int = Field(ge=1)
    registry_hash: str = ""

    @property
    def triple(self) -> tuple[str, int, str]:
        return self.paper_id, self.page, self.block_id


class CitationResolution(BaseModel):
    entries: list[CitationRegistryEntry]
    duplicate_ids: list[str]


class CitationRegistry(BaseModel):
    schema_version: str = "citation-id-v2"
    entries: list[CitationRegistryEntry]
    registry_hash: str

    @model_validator(mode="after")
    def validate_registry(self) -> CitationRegistry:
        identifiers = [entry.citation_id for entry in self.entries]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("citation IDs must be unique within one run")
        expected = self.compute_hash(self.entries)
        if self.registry_hash != expected:
            raise ValueError("citation registry hash mismatch")
        if any(entry.registry_hash != expected for entry in self.entries):
            raise ValueError("citation entry registry hash mismatch")
        return self

    @staticmethod
    def compute_hash(entries: Iterable[CitationRegistryEntry]) -> str:
        body = [
            {
                "citation_id": item.citation_id,
                "evidence_id": item.evidence_id,
                "paper_id": item.paper_id,
                "page": item.page,
                "block_id": item.block_id,
                "claim_ids": sorted(item.claim_ids),
                "context_position": item.context_position,
            }
            for item in entries
        ]
        encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def from_context(
        cls,
        context: list[ContextItem],
        *,
        claim_allocations: dict[str, list[str]] | None = None,
    ) -> CitationRegistry:
        claims_by_evidence: dict[str, list[str]] = {}
        for claim_id, evidence_ids in (claim_allocations or {}).items():
            for evidence_id in evidence_ids:
                claims_by_evidence.setdefault(evidence_id, []).append(claim_id)
        raw_entries: list[CitationRegistryEntry] = []
        position = 0
        for context_position, item in enumerate(context, start=1):
            block_ids = item.block_ids or [item.chunk_id]
            page_map = item.block_page_map or {
                block_id: item.page_start for block_id in block_ids
            }
            for block_id in block_ids:
                position += 1
                raw_entries.append(
                    CitationRegistryEntry(
                        citation_id=f"E{position:03d}",
                        evidence_id=item.chunk_id,
                        paper_id=item.paper_id,
                        page=page_map[block_id],
                        block_id=block_id,
                        claim_ids=sorted(claims_by_evidence.get(item.chunk_id, [])),
                        context_position=context_position,
                    )
                )
        registry_hash = cls.compute_hash(raw_entries)
        entries = [
            entry.model_copy(update={"registry_hash": registry_hash})
            for entry in raw_entries
        ]
        return cls(entries=entries, registry_hash=registry_hash)

    def prompt_entries(self) -> list[dict[str, object]]:
        """Return the only citation information exposed to the model."""
        return [
            {
                "citation_id": entry.citation_id,
                "evidence_id": entry.evidence_id,
                "claim_ids": entry.claim_ids,
                "context_position": entry.context_position,
            }
            for entry in self.entries
        ]

    def resolve(
        self,
        citation_ids: list[str],
        *,
        claim_id: str | None = None,
    ) -> CitationResolution:
        by_id = {entry.citation_id: entry for entry in self.entries}
        resolved: list[CitationRegistryEntry] = []
        seen: set[str] = set()
        duplicates: list[str] = []
        for citation_id in citation_ids:
            if citation_id not in by_id:
                raise CitationRegistryError(f"unknown citation_id: {citation_id}")
            entry = by_id[citation_id]
            if claim_id is not None and entry.claim_ids and claim_id not in entry.claim_ids:
                raise CitationRegistryError(
                    f"citation_id {citation_id} is not allocated to claim {claim_id}"
                )
            if citation_id in seen:
                duplicates.append(citation_id)
                continue
            seen.add(citation_id)
            resolved.append(entry)
        self.validate_resolved_triples(resolved)
        return CitationResolution(entries=resolved, duplicate_ids=duplicates)

    def validate_resolved_triples(
        self, entries: Iterable[CitationRegistryEntry]
    ) -> None:
        allowed = {entry.triple for entry in self.entries}
        if any(entry.triple not in allowed for entry in entries):
            raise CitationRegistryError("resolved citation is outside the registry")


def reject_free_form_citation(value: object) -> None:
    """citation-id-v2 never accepts model-generated paper/page/block objects."""
    if isinstance(value, dict) and {"paper_id", "page", "block_id"} & set(value):
        raise CitationRegistryError(
            "citation-id-v2 accepts citation_id only; free-form triples are forbidden"
        )
