"""Deterministic document/block identities for evaluation-only Gold lineage."""

from __future__ import annotations

from dataclasses import dataclass

from paper_research.evaluation.ragq3_identity import stable_id


@dataclass(frozen=True)
class CanonicalDocumentIdentity:
    canonical_document_id: str
    source_namespace: str
    external_source_id: str | None
    source_file_sha256: str | None
    database_paper_uuid: str | None


def canonical_document_identity(
    *,
    source_namespace: str,
    external_source_id: str | None,
    source_file_sha256: str | None,
    database_paper_uuid: str | None = None,
) -> CanonicalDocumentIdentity:
    """Create an ID independent of paths and mutable database UUIDs."""
    if not external_source_id and not source_file_sha256:
        raise ValueError("authoritative external source ID or file hash is required")
    canonical = stable_id(
        "canonical_document",
        {
            "source_namespace": source_namespace,
            "external_source_id": external_source_id,
            "source_file_sha256": source_file_sha256,
        },
    )
    return CanonicalDocumentIdentity(
        canonical, source_namespace, external_source_id, source_file_sha256, database_paper_uuid
    )


def canonical_block_id(document: CanonicalDocumentIdentity, local_block_id: str) -> str:
    """Make document-local parser block labels globally unambiguous."""
    if not local_block_id:
        raise ValueError("local block ID is required")
    return stable_id(
        "canonical_block",
        {
            "canonical_document_id": document.canonical_document_id,
            "legacy_local_block_id": local_block_id,
        },
    )


class GoldLineageResolver:
    """Exact registry lookup only; no text, title, or model-based inference."""

    def __init__(self, documents: list[CanonicalDocumentIdentity]) -> None:
        self._by_external = {
            (item.source_namespace, item.external_source_id): item
            for item in documents
            if item.external_source_id
        }

    def resolve(
        self, *, source_namespace: str, external_source_id: str, local_block_id: str
    ) -> str:
        document = self._by_external.get((source_namespace, external_source_id))
        if document is None:
            raise LookupError("unresolved or ambiguous authoritative document identity")
        return canonical_block_id(document, local_block_id)
