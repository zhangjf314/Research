import json

import pytest
from qdrant_client import QdrantClient, models

from paper_research.chunking.structural_chunker import StructuralChunker
from paper_research.indexing.embedding import HashEmbeddingProvider
from paper_research.indexing.identity import canonical_digest
from paper_research.indexing.registry import (
    IndexIdentityError,
    IndexRegistry,
    runtime_index_identity,
)
from paper_research.indexing.service import IndexingService
from paper_research.indexing.vector_store import QdrantVectorStore


@pytest.fixture
def indexed(tmp_path):
    client = QdrantClient(":memory:")
    client.create_collection(
        "synthetic",
        vectors_config=models.VectorParams(size=32, distance=models.Distance.COSINE),
    )
    registry = IndexRegistry(tmp_path / "registry.json")
    identity = runtime_index_identity(
        {"embedding_provider": "hash", "embedding_model": "hash-v1",
         "embedding_revision": "v1", "embedding_dimension": 32, "index_version": "fixture-v1"},
        "synthetic",
    )
    registry.bind_identity("synthetic", identity, client)
    return registry, client, identity


@pytest.mark.parametrize("field,value", [
    ("embedding_provider", "other-provider"), ("embedding_model", "other-model"),
    ("embedding_revision", "other-revision"), ("embedding_dimension", 64),
    ("index_version", "other-version"), ("collection_name", "other-collection"),
])
def test_runtime_identity_conflict_is_409(indexed, field, value):
    registry, client, identity = indexed
    with pytest.raises(IndexIdentityError) as error:
        registry.verify_identity("synthetic", {**identity, field: value}, client)
    assert error.value.status_code == 409
    assert field in error.value.fields


def test_builder_identity_survives_registry_switch_and_reload(indexed):
    registry, client, identity = indexed
    registry.switch("logical", "synthetic", {"manual_switch": True})
    reloaded = IndexRegistry(registry.path)
    assert reloaded.resolve("logical") == "synthetic"
    assert reloaded.verify_identity("synthetic", identity, client) == identity


@pytest.mark.parametrize("corruption", ["missing", "invalid_json", "checksum", "physical", "name"])
def test_missing_invalid_or_physically_conflicting_identity_fails_closed(indexed, corruption):
    registry, client, identity = indexed
    data = registry.read()
    if corruption == "missing":
        del data["indexes"]["synthetic"]["embedding_identity"]
    elif corruption == "invalid_json":
        registry.path.write_text("broken", encoding="utf-8")
    else:
        record = data["indexes"]["synthetic"]["embedding_identity"]
        if corruption == "checksum":
            record["sha256"] = "bad"
        elif corruption == "physical":
            record["identity"]["distance_metric"] = "Dot"
            record["sha256"] = canonical_digest(record["identity"])
            identity = record["identity"]
        elif corruption == "name":
            record["identity"]["collection_name"] = "wrong-name"
            record["sha256"] = canonical_digest(record["identity"])
    if corruption != "invalid_json":
        registry.path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(IndexIdentityError) as error:
        registry.verify_identity("synthetic", identity, client)
    assert error.value.status_code == 503


def test_existing_unattested_collection_cannot_be_silently_adopted(tmp_path):
    client = QdrantClient(":memory:")
    store = QdrantVectorStore(client, "existing", 32)
    store.ensure_collection()
    registry = IndexRegistry(tmp_path / "registry.json")
    metadata = {"embedding_provider": "hash", "embedding_model": "hash-v1",
        "embedding_revision": "v1", "embedding_dimension": 32, "index_version": "v1"}
    with pytest.raises(IndexIdentityError, match="INDEX_IDENTITY_MISSING"):
        IndexingService(StructuralChunker(), HashEmbeddingProvider(32), store).index(
            "synthetic", tmp_path / "unused-blocks", tmp_path / "unused-chunks",
            metadata=metadata, registry=registry,
        )
    assert not registry.path.exists()
    assert store.count() == 0
