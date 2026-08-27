import pytest

from paper_research.evaluation.ragq3_lineage import (
    GoldLineageResolver,
    canonical_block_id,
    canonical_document_identity,
)


def test_document_identity_is_path_and_uuid_independent() -> None:
    first = canonical_document_identity(
        source_namespace="arxiv",
        external_source_id="2302.13971",
        source_file_sha256="a" * 64,
        database_paper_uuid="one",
    )
    second = canonical_document_identity(
        source_namespace="arxiv",
        external_source_id="2302.13971",
        source_file_sha256="a" * 64,
        database_paper_uuid="two",
    )
    assert first.canonical_document_id == second.canonical_document_id


def test_same_local_block_in_different_documents_is_globally_distinct() -> None:
    left = canonical_document_identity(
        source_namespace="arxiv", external_source_id="a", source_file_sha256=None
    )
    right = canonical_document_identity(
        source_namespace="arxiv", external_source_id="b", source_file_sha256=None
    )
    assert canonical_block_id(left, "b000001") != canonical_block_id(right, "b000001")


def test_resolver_never_guesses_missing_identity() -> None:
    document = canonical_document_identity(
        source_namespace="arxiv", external_source_id="a", source_file_sha256=None
    )
    with pytest.raises(LookupError):
        GoldLineageResolver([document]).resolve(
            source_namespace="arxiv", external_source_id="b", local_block_id="b000001"
        )
