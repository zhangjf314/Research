import copy
from types import SimpleNamespace

import pytest

from paper_research.chunking.types import Chunk
from paper_research.indexing.identity import (
    assert_runtime_compatible,
    corpus_identity,
    equality_certificate,
    index_manifest,
)
from paper_research.retrieval.contextual import contextualize


def sample():
    return Chunk(
        chunk_id="chunk",
        paper_id="paper",
        block_ids=["block"],
        block_page_map={"block": 2},
        section_path=["Methods"],
        block_type="paragraph",
        page_start=2,
        page_end=2,
        chunk_text="Original source evidence",
        token_count=3,
    )


def provider():
    return SimpleNamespace(
        provider_name="siliconflow",
        model_name="Qwen/Qwen3-Embedding-0.6B",
        revision="v1",
        dimensions=1024,
    )


def manifest(role="raw"):
    chunks = [sample()] if role == "raw" else [contextualize(sample(), {"title": "Paper"})]
    return index_manifest(
        chunks,
        {"paper": {"title": "Paper"}},
        provider(),
        collection=role,
        embedding_input=role,
        source_corpus_identity={"sha256": "source"},
        git_commit="commit",
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("embedding_model", "another-1024-model"),
        ("embedding_provider", "jina"),
        ("embedding_revision", "v2"),
        ("embedding_dimension", 384),
        ("distance_metric", "Dot"),
        ("query_instruction", "modified instruction"),
        ("collection_name", "other-collection"),
    ],
)
def test_runtime_mismatch_rejected(field, value):
    identity = manifest()
    identity[field] = value
    with pytest.raises(ValueError, match="INDEX_EMBEDDING_IDENTITY_MISMATCH"):
        assert_runtime_compatible(identity, provider(), collection="raw")


def test_raw_contextual_certificate_and_digest_order():
    raw, contextual = manifest(), manifest("contextual")
    assert_runtime_compatible(raw, provider(), collection="raw")
    assert equality_certificate(raw, contextual)["status"] == "VALID"
    other = sample().model_copy(update={"chunk_id": "second"})
    assert corpus_identity([sample(), other], {}) == corpus_identity([other, sample()], {})


@pytest.mark.parametrize(
    "field", ["raw_text_digest", "canonical_metadata_digest", "sorted_chunk_id_digest"]
)
def test_certificate_rejects_corpus_drift(field):
    contextual = copy.deepcopy(manifest("contextual"))
    contextual[field] = "drift"
    assert equality_certificate(manifest(), contextual)["status"] == "C0_C1_COMPARISON_INVALID"


def test_duplicate_chunk_ids_cannot_be_certified():
    with pytest.raises(ValueError, match="unique"):
        corpus_identity([sample(), sample()], {})


def test_pipeline_guard_runs_before_query():
    from paper_research.retrieval.context_builder import ContextBuilder
    from paper_research.retrieval.pipeline import RetrievalPipeline
    from paper_research.retrieval.reranker import DisabledReranker
    from paper_research.retrieval.sparse import BM25Retriever

    identity = manifest()
    identity["embedding_model"] = "different-model-same-dimension"
    dense = SimpleNamespace(embedding=provider(), vector_store=SimpleNamespace(collection="raw"))
    with pytest.raises(ValueError, match="INDEX_EMBEDDING_IDENTITY_MISMATCH"):
        RetrievalPipeline(
            dense,
            BM25Retriever([sample()]),
            DisabledReranker(),
            ContextBuilder(False),
            index_manifest=identity,
        )
