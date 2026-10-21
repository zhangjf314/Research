"""Frozen S0 synthetic boundary cases; no scientific inputs or semantic verifier."""

import ast
import hashlib
import json
import math
import re
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from paper_research.api.routes import rag
from paper_research.chunking.types import Chunk
from paper_research.generation import evidence_contract_adapter as adapter
from paper_research.generation.qa_service import ClaimEvidenceValidator, ClaimValidationError
from paper_research.parsing.types import BoundingBox, PaperBlock
from paper_research.providers.llm import GeneratedCitation, GeneratedClaim
from paper_research.retrieval.context_builder import ContextItem
from paper_research.retrieval.hybrid import HybridRetrievalResult
from paper_research.retrieval.trace import RetrievalTrace

ROOT = Path(__file__).resolve().parents[1]
S0 = ROOT / "artifacts/paperresearch-v2/project-p1-s0-evidence-contract-integration-preregistration"
SCHEMA = json.loads((S0 / "interface_contract.json").read_text(encoding="utf-8"))["output_schema"]
PAPER = "11111111-1111-4111-8111-111111111111"
OTHER = "22222222-2222-4222-8222-222222222222"


def validate_frozen_schema(value, schema=None, *, root_schema=None):
    """Evaluate the exact keywords present in the frozen output schema, without new deps."""
    schema = SCHEMA if schema is None else schema
    root_schema = SCHEMA if root_schema is None else root_schema
    if "$ref" in schema:
        target = root_schema
        for key in schema["$ref"][2:].split("/"):
            target = target[key]
        return validate_frozen_schema(value, target, root_schema=root_schema)
    if "const" in schema:
        assert value == schema["const"] and type(value) is type(schema["const"])
    if "enum" in schema:
        assert value in schema["enum"]
    kind = schema.get("type")
    predicates = {
        "object": lambda x: isinstance(x, dict),
        "array": lambda x: isinstance(x, list),
        "string": lambda x: isinstance(x, str),
        "boolean": lambda x: type(x) is bool,
        "integer": lambda x: type(x) is int,
        "number": lambda x: type(x) in (int, float) and math.isfinite(x),
        "null": lambda x: x is None,
    }
    if kind:
        assert any(predicates[k](value) for k in (kind if isinstance(kind, list) else [kind]))
    if isinstance(value, dict):
        assert set(schema.get("required", [])) <= value.keys()
        properties = schema.get("properties", {})
        additional = schema.get("additionalProperties", True)
        if additional is False:
            assert set(value) <= properties.keys()
        assert len(value) >= schema.get("minProperties", 0)
        for key, item in value.items():
            if key in properties:
                validate_frozen_schema(item, properties[key], root_schema=root_schema)
            elif isinstance(additional, dict):
                validate_frozen_schema(item, additional, root_schema=root_schema)
    if isinstance(value, list):
        assert len(value) >= schema.get("minItems", 0)
        assert len(value) <= schema.get("maxItems", math.inf)
        if schema.get("uniqueItems"):
            assert len({json.dumps(x, sort_keys=True) for x in value}) == len(value)
        for item in value:
            validate_frozen_schema(item, schema.get("items", {}), root_schema=root_schema)
    if isinstance(value, str):
        assert len(value) >= schema.get("minLength", 0)
        assert len(value) <= schema.get("maxLength", math.inf)
        if "pattern" in schema:
            assert re.search(schema["pattern"], value)
        if schema.get("format") == "uuid":
            uuid.UUID(value)
    if type(value) in (int, float):
        assert value >= schema.get("minimum", -math.inf)
        assert value <= schema.get("maximum", math.inf)


def write_paper(root, paper=PAPER, *, chunks=None):
    blocks = [
        PaperBlock(
            block_id=f"b{page}",
            block_type="paragraph",
            section_path=["Synthetic"],
            page_start=page,
            page_end=page,
            source_page=page,
            block_index=page,
            text=text,
            bbox=BoundingBox(x0=0, y0=0, x1=100, y1=20),
        )
        for page, text in [(1, "alpha synthetic"), (2, "beta synthetic")]
    ]
    if chunks is None:
        chunks = [
            Chunk(
                chunk_id="c1",
                paper_id=paper,
                block_ids=["b1", "b2"],
                section_path=["Synthetic"],
                block_type="paragraph",
                page_start=1,
                page_end=2,
                block_page_map={"b1": 1, "b2": 2},
                chunk_text="alpha synthetic\nbeta synthetic",
                token_count=6,
            )
        ]
    directory = root / paper
    directory.mkdir(parents=True)
    chunk_path = directory / "paper_chunks.fixture-v1.jsonl"
    chunk_path.write_text("\n".join(c.model_dump_json() for c in chunks), encoding="utf-8")
    (directory / "paper_blocks.jsonl").write_text(
        "\n".join(b.model_dump_json() for b in blocks), encoding="utf-8"
    )
    manifest = {
        "embedding_provider": "hash",
        "embedding_model": "hash-v1",
        "embedding_revision": "v1",
        "embedding_dimension": 384,
        "index_version": "fixture-v1",
        "collection": "fixture-collection",
        "paper_id": paper,
        "chunk_count": len(chunks),
    }
    (directory / "paper_chunks.fixture-v1_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return [
        ContextItem(
            chunk_id=c.chunk_id,
            paper_id=c.paper_id,
            block_ids=list(c.block_ids),
            block_page_map=dict(c.block_page_map),
            section_path=list(c.section_path),
            page_start=c.page_start,
            page_end=c.page_end,
            evidence=c.chunk_text,
            score=0.5,
        )
        for c in chunks
    ]


def trace_for(context, *, scope=None):
    return RetrievalTrace(
        query="synthetic",
        filters={"paper_ids": scope},
        dense_results=[],
        sparse_results=[],
        fusion_results=[],
        rerank_results=[],
        final_context=context,
        latency_ms=0,
        embedding_provider="hash",
        embedding_model="hash-v1",
        embedding_revision="v1",
        embedding_dimension=384,
        index_version="fixture-v1",
        collection="fixture-collection",
    )


def assert_mapping(context, sidecar):
    assert sidecar["semantic_status"] == "NOT_ASSESSED"
    for record in sidecar["evidence"]:
        item, provenance = record["item"], record["provenance"]
        original = next(
            c
            for c in context
            if c["paper_id"] == item["paper_id"] and c["chunk_id"] == item["chunk_id"]
        )
        for key in [
            "paper_id",
            "chunk_id",
            "block_ids",
            "block_page_map",
            "section_path",
            "page_start",
            "page_end",
        ]:
            assert provenance[key] == original[key]
        assert item["text_or_reference"] == original["evidence"]
        assert item["relevance"] == original["score"]
        assert item["verification_status"] == "CANDIDATE"
        assert item["supports"] == item["contradicts"] == []
        assert item["block_id"] == original["block_ids"][0]
        assert item["page"] == original["block_page_map"][item["block_id"]]
        assert "source_id" not in provenance
        assert [(s["paper_id"], s["block_id"], s["page"]) for s in provenance["sources"]] == [
            (original["paper_id"], block, original["block_page_map"][block])
            for block in original["block_ids"]
        ]


@pytest.fixture
def source(tmp_path):
    context = write_paper(tmp_path)
    trace = trace_for(context, scope=[PAPER])
    return tmp_path, context, trace


@pytest.fixture
def boundary(source, monkeypatch):
    root, context, trace = source
    calls = {"retrieve": 0, "adapter": 0, "source_reads": 0}

    def retrieve(payload):
        calls["retrieve"] += 1
        return HybridRetrievalResult(context, trace)

    original = adapter.build_readonly_evidence_contract
    read_json, read_lines = adapter._read_json, adapter._read_jsonl

    def construct(*args, **kwargs):
        calls["adapter"] += 1
        return original(*args, **kwargs)

    def json_read(path):
        calls["source_reads"] += 1
        return read_json(path)

    def lines_read(path):
        calls["source_reads"] += 1
        return read_lines(path)

    monkeypatch.setattr(rag, "_run_hybrid", retrieve)
    monkeypatch.setattr(rag, "get_settings", lambda: SimpleNamespace(parsed_papers_dir=root))
    monkeypatch.setattr(adapter, "build_readonly_evidence_contract", construct)
    monkeypatch.setattr(adapter, "_read_json", json_read)
    monkeypatch.setattr(adapter, "_read_jsonl", lines_read)
    app = FastAPI()
    app.include_router(rag.router, prefix="/api/v1")
    with TestClient(app) as client:
        yield client, calls, context, trace


def test_S02_cross_page_chunk_and_document_identity(source):
    root, context, trace = source
    second = write_paper(root, OTHER)
    directory = root / PAPER
    path = directory / "paper_chunks.fixture-v1.jsonl"
    raw = json.loads(path.read_text())
    split = {**raw, "chunk_id": "c2"}
    path.write_text(json.dumps(raw) + "\n" + json.dumps(split), encoding="utf-8")
    manifest_path = directory / "paper_chunks.fixture-v1_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["chunk_count"] = 2
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    context += [context[0].model_copy(update={"chunk_id": "c2"}), *second]
    trace.filters = {"paper_ids": [PAPER, OTHER]}
    result = adapter.build_readonly_evidence_contract("synthetic", context, trace, parsed_root=root)
    data = result.model_dump(mode="json")
    assert data["status"] == "OK" and len(data["evidence"]) == 3
    assert len({e["item"]["evidence_id"] for e in data["evidence"]}) == 3
    assert_mapping([c.model_dump(mode="json") for c in context], data)
    assert data["evidence"][0]["provenance"]["block_page_map"] == {"b1": 1, "b2": 2}
    validate_frozen_schema(
        {
            "context": [c.model_dump(mode="json") for c in context],
            "trace": trace.model_dump(mode="json"),
            "evidence_contract": data,
        }
    )


def test_S03_empty_success_has_typed_gap(boundary, monkeypatch):
    client, calls, _, trace = boundary
    monkeypatch.setattr(rag, "_run_hybrid", lambda payload: HybridRetrievalResult([], trace))
    response = client.post(
        "/api/v1/retrieve", json={"query": "empty", "include_evidence_contract": True}
    )
    assert response.status_code == 200
    sidecar = response.json()["evidence_contract"]
    assert sidecar["status"] == "OK" and sidecar["coverage"][0]["availability"] == "MISSING"
    assert sidecar["evidence"] == [] and sidecar["gaps"][0]["gap_type"] == "MISSING_EVIDENCE"
    assert calls["adapter"] == 1 and calls["source_reads"] == 0
    validate_frozen_schema(response.json())


@pytest.mark.parametrize(
    "mutation",
    [
        {"paper_id": "unknown"},
        {"chunk_id": "unknown"},
        {"block_ids": ["unknown"], "block_page_map": {"unknown": 1}},
        {"page_start": 3, "page_end": 3, "block_page_map": {"b1": 3, "b2": 3}},
        {"block_page_map": {}},
        {"block_page_map": {"b1": 1}},
        {"block_page_map": {"b1": 1, "b2": 2, "extra": 1}},
        {"block_ids": ["b1", "b1"], "block_page_map": {"b1": 1}},
        {"block_page_map": {"b1": 1, "b2": 1}},
    ],
)
def test_S04_invalid_source_never_becomes_evidence(source, mutation):
    root, context, trace = source
    candidate = context[0].model_copy(update=mutation)
    result = adapter.build_readonly_evidence_contract(
        "synthetic", [candidate], trace, parsed_root=root
    )
    assert result.status == "ERROR" and result.evidence == []
    assert result.coverage[0].evidence_ids == []
    assert result.gaps[0].evidence_ids == []
    assert result.errors[0].trace_id == trace.trace_id


def test_S04_cross_scope_and_fabricated_citation(source):
    root, context, trace = source
    trace.filters = {"paper_ids": [OTHER]}
    result = adapter.build_readonly_evidence_contract("synthetic", context, trace, parsed_root=root)
    assert result.errors[0].code == "SOURCE_SCOPE_MISMATCH" and not result.evidence
    for citation in [
        GeneratedCitation(paper_id=PAPER, block_id="missing", page=1),
        GeneratedCitation(paper_id=PAPER, block_id="b1", page=2),
        GeneratedCitation(paper_id=OTHER, block_id="b1", page=1),
    ]:
        with pytest.raises(ClaimValidationError):
            ClaimEvidenceValidator().validate(
                [GeneratedClaim(claim_id="invalid", text="synthetic", citations=[citation])],
                context,
            )


def test_S06_opt_in_strict_frozen_schema(boundary):
    client, calls, context, trace = boundary
    before = [c.model_dump_json() for c in context], trace.model_dump_json()
    response = client.post(
        "/api/v1/retrieve", json={"query": "synthetic", "include_evidence_contract": True}
    )
    assert response.status_code == 200 and calls == {"retrieve": 1, "adapter": 1, "source_reads": 3}
    validate_frozen_schema(response.json())
    assert_mapping(response.json()["context"], response.json()["evidence_contract"])
    assert before == ([c.model_dump_json() for c in context], trace.model_dump_json())
    with pytest.raises(AssertionError):
        validate_frozen_schema({**response.json(), "unregistered": True})


@pytest.mark.parametrize("flag", [None, False])
def test_S07_default_exact_baseline_serialization_and_zero_adapter(boundary, flag):
    client, calls, context, trace = boundary
    payload = {"query": "synthetic"}
    if flag is not None:
        payload["include_evidence_contract"] = flag
    response = client.post("/api/v1/retrieve", json=payload)
    expected = rag.HybridRetrievalResponse(context=context, trace=trace).model_dump(mode="json")
    assert response.status_code == 200 and response.json() == expected
    assert calls == {"retrieve": 1, "adapter": 0, "source_reads": 0}
    assert "evidence_contract" not in response.json()


def test_S10_adapter_files_readonly(source):
    root, context, trace = source
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file()}
    adapter.build_readonly_evidence_contract("synthetic", context, trace, parsed_root=root)
    after = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file()}
    assert before == after


def test_S13_partial_all_invalid_and_unexpected_failure(source, monkeypatch):
    root, context, trace = source
    invalid = context[0].model_copy(update={"chunk_id": "missing"})
    partial = adapter.build_readonly_evidence_contract(
        "synthetic", [*context, invalid], trace, parsed_root=root
    )
    assert partial.status == "PARTIAL" and partial.coverage[0].availability == "PARTIAL"
    assert len(partial.evidence) == len(partial.errors) == len(partial.gaps) == 1
    failed = adapter.build_readonly_evidence_contract(
        "synthetic", [invalid], trace, parsed_root=root
    )
    assert failed.status == "ERROR" and failed.coverage[0].availability == "UNAVAILABLE"

    def broken(*args):
        raise RuntimeError("sensitive error must not be exposed")

    monkeypatch.setattr(adapter, "_read_paper", broken)
    unexpected = adapter.build_readonly_evidence_contract(
        "synthetic", context, trace, parsed_root=root
    )
    assert unexpected.status == "ERROR" and not unexpected.evidence
    assert unexpected.errors[0].code == "EVIDENCE_ADAPTER_ERROR"
    assert "sensitive" not in unexpected.model_dump_json()
    assert unexpected.coverage[0].reason_code == "ADAPTER_ERROR"


def test_S13_api_adapter_failure_preserves_original_result(boundary, monkeypatch):
    client, _, context, trace = boundary

    def broken(*args, **kwargs):
        raise RuntimeError("sensitive error")

    monkeypatch.setattr(adapter, "build_readonly_evidence_contract", broken)
    response = client.post(
        "/api/v1/retrieve", json={"query": "synthetic", "include_evidence_contract": True}
    )
    assert response.status_code == 200
    assert response.json()["context"] == [c.model_dump(mode="json") for c in context]
    assert response.json()["trace"] == trace.model_dump(mode="json")
    assert response.json()["evidence_contract"]["status"] == "ERROR"
    validate_frozen_schema(response.json())


@pytest.mark.parametrize(
    "payload",
    [
        {"query": "x" * 2001},
        {"query": "   "},
        {"query": "synthetic", "filters": {"paper_ids": [PAPER] * 39}},
        {"query": "synthetic", "filters": {"paper_ids": ["not-a-UUID"]}},
    ],
)
def test_S14_opt_in_limits_only(boundary, payload):
    client, calls, _, _ = boundary
    assert (
        client.post(
            "/api/v1/retrieve", json={**payload, "include_evidence_contract": True}
        ).status_code
        == 422
    )
    assert calls == {"retrieve": 0, "adapter": 0, "source_reads": 0}
    assert (
        client.post(
            "/api/v1/retrieve", json={**payload, "include_evidence_contract": False}
        ).status_code
        == 200
    )
    assert calls == {"retrieve": 1, "adapter": 0, "source_reads": 0}


@pytest.mark.parametrize("flag", [1, 0, "true", "false", None])
def test_S14_new_flag_strict_boolean(boundary, flag):
    client, calls, _, _ = boundary
    response = client.post(
        "/api/v1/retrieve", json={"query": "synthetic", "include_evidence_contract": flag}
    )
    assert response.status_code == 422 and calls["retrieve"] == 0


@pytest.mark.parametrize(
    "kind",
    [
        "prefix",
        "empty",
        "wrong",
        "legacy",
        "provider",
        "model",
        "revision",
        "index",
        "collection",
        "contextual",
        "neighbor",
    ],
)
def test_S15_text_and_artifact_contract(source, kind):
    root, context, trace = source
    candidate = context[0]
    if kind in {"prefix", "empty", "wrong", "neighbor"}:
        text = {
            "prefix": candidate.evidence[:5],
            "empty": "",
            "wrong": "fabricated",
            "neighbor": "prior neighbor\n\n" + candidate.evidence,
        }[kind]
        candidate = candidate.model_copy(update={"evidence": text})
    elif kind == "legacy":
        (root / PAPER / "paper_chunks.fixture-v1.jsonl").rename(root / PAPER / "paper_chunks.jsonl")
    elif kind == "contextual":
        trace.contextual_retrieval = True
    else:
        path = root / PAPER / "paper_chunks.fixture-v1_manifest.json"
        manifest = json.loads(path.read_text())
        field = {
            "provider": "embedding_provider",
            "model": "embedding_model",
            "revision": "embedding_revision",
            "index": "index_version",
            "collection": "collection",
        }[kind]
        manifest[field] = "wrong"
        path.write_text(json.dumps(manifest), encoding="utf-8")
    result = adapter.build_readonly_evidence_contract(
        "synthetic", [candidate], trace, parsed_root=root
    )
    assert result.status == ("OK" if kind == "prefix" else "ERROR")
    if kind == "prefix":
        assert result.evidence[0].item.text_or_reference == candidate.evidence
    else:
        assert result.evidence == [] and result.errors and result.gaps


def test_S15_duplicate_identity_is_not_silently_merged(source):
    root, context, trace = source
    same = adapter.build_readonly_evidence_contract(
        "synthetic", context * 2, trace, parsed_root=root
    )
    assert same.status == "OK" and len(same.evidence) == 1
    conflict = context[0].model_copy(update={"evidence": "different"})
    result = adapter.build_readonly_evidence_contract(
        "synthetic", [*context, conflict], trace, parsed_root=root
    )
    assert result.status == "ERROR" and result.evidence == [] and len(result.errors) == 2


def test_S16_response_union_schema_and_serialization(boundary):
    client, _, _, _ = boundary
    operation = client.get("/openapi.json").json()["paths"]["/api/v1/retrieve"]["post"]
    response = operation["responses"]["200"]["content"]["application/json"]["schema"]
    assert len(response["anyOf"]) == 2
    assert any("HybridRetrievalEvidenceResponse" in part["$ref"] for part in response["anyOf"])
    assert any(part["$ref"].endswith("/HybridRetrievalResponse") for part in response["anyOf"])


def test_S17_caller_cannot_supply_trusted_evidence(boundary):
    client, _, _, _ = boundary
    response = client.post(
        "/api/v1/retrieve",
        headers={"X-Embedding-Model": "wrong"},
        json={
            "query": "synthetic",
            "include_evidence_contract": True,
            "source_id": "invented",
            "evidence": "invented",
            "verification_status": "VERIFIED",
            "embedding_model": "wrong",
            "collection": "wrong",
            "evidence_contract": {"status": "OK"},
        },
    )
    data = response.json()["evidence_contract"]
    assert response.status_code == 200
    assert data["retrieval_identity"]["embedding_model"] == "hash-v1"
    assert data["retrieval_identity"]["collection"] == "fixture-collection"
    assert all(e["item"]["verification_status"] == "CANDIDATE" for e in data["evidence"])
    assert "invented" not in json.dumps(data)


def test_S18_frozen_retrieval_and_qa_functions_unchanged():
    original = (
        ROOT
        / "artifacts/paperresearch-v2/project-p1-d1-evidence-contract-minimal-integration"
        / "implementation/originals/rag.py"
    )
    before = ast.parse(original.read_text(encoding="utf-8"))
    after = ast.parse((ROOT / "src/paper_research/api/routes/rag.py").read_text(encoding="utf-8"))
    for name in [
        "_run_hybrid",
        "ask_question",
        "_load_chunks",
        "_load_block_pages",
        "_with_block_page_map",
    ]:
        first = next(
            node for node in before.body if isinstance(node, ast.FunctionDef) and node.name == name
        )
        second = next(
            node for node in after.body if isinstance(node, ast.FunctionDef) and node.name == name
        )
        assert ast.dump(first, include_attributes=False) == ast.dump(
            second, include_attributes=False
        )
