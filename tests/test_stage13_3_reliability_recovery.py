# ruff: noqa: E501
from __future__ import annotations

import json
import ssl
import zipfile
from pathlib import Path

import pytest

from paper_research.config import Settings
from paper_research.generation.citation_id_output import (
    CitationIdQA,
    resolve_citation_id_answer,
)
from paper_research.generation.citation_registry import (
    CitationRegistry,
    CitationRegistryError,
    reject_free_form_citation,
)
from paper_research.generation.output_adapter import (
    ClaimTextAdapterError,
    normalized_claim_text,
)
from paper_research.providers.response_envelope import ProviderResponseEnvelopeStore
from paper_research.retrieval.context_builder import ContextItem


def context() -> list[ContextItem]:
    return [
        ContextItem(
            chunk_id="ev1",
            paper_id="paper",
            block_ids=["b1", "b2"],
            block_page_map={"b1": 7, "b2": 8},
            section_path=["Methods"],
            page_start=7,
            page_end=8,
            evidence="evidence",
            score=1,
        )
    ]


@pytest.mark.parametrize(
    ("claim", "expected"),
    [
        ({"claim_text": "value"}, "value"),
        ({"text": "value"}, "value"),
        ({"claim_text": "value", "text": "value"}, "value"),
    ],
)
def test_q001_claim_text_compatibility(claim: dict, expected: str) -> None:
    assert normalized_claim_text(claim) == expected


def test_q001_claim_text_missing_or_conflicting_fails() -> None:
    with pytest.raises(ClaimTextAdapterError):
        normalized_claim_text({})
    with pytest.raises(ClaimTextAdapterError):
        normalized_claim_text({"text": "a", "claim_text": "b"})


def test_usage_is_persisted_before_parse_and_reservation_released(tmp_path: Path) -> None:
    ledger = tmp_path / "request-ledger.jsonl"
    ledger.write_text("", encoding="utf-8")
    raw = json.dumps(
        {
            "model": "fixture",
            "choices": [{"finish_reason": "stop", "message": {"content": "{}"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        }
    ).encode()
    store = ProviderResponseEnvelopeStore(tmp_path, ledger)
    envelope = store.record_received(
        request_id="r1", provider="siliconflow", model="fixture", raw_body=raw
    )
    envelope = store.parsing_started(envelope)
    envelope = store.post_processing_failed(envelope, ValueError("fixture"))
    events = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    assert [event["event"] for event in events] == [
        "raw_response_received",
        "provider_usage_recorded",
        "raw_response_persisted",
        "response_parsing_started",
        "post_processing_failed",
    ]
    assert events[1]["active_reserved_tokens"] == 0
    assert envelope.usage.total_tokens == 12
    assert envelope.parse_status == "post_processing_failed"
    assert (tmp_path / "raw-provider-response.json").exists()


def test_malformed_provider_body_is_preserved_before_decode(tmp_path: Path) -> None:
    ledger = tmp_path / "request-ledger.jsonl"
    ledger.write_text("", encoding="utf-8")
    store = ProviderResponseEnvelopeStore(tmp_path, ledger)
    with pytest.raises(json.JSONDecodeError):
        store.record_received(
            request_id="r-malformed",
            provider="siliconflow",
            model="fixture",
            raw_body=b"not-json",
        )
    assert (tmp_path / "raw-provider-response.json").read_bytes() == b"not-json"
    events = [json.loads(line)["event"] for line in ledger.read_text(encoding="utf-8").splitlines()]
    assert events == ["raw_response_received", "raw_response_persisted"]


def test_citation_registry_is_stable_unique_and_strict() -> None:
    first = CitationRegistry.from_context(context(), claim_allocations={"c1": ["ev1"]})
    second = CitationRegistry.from_context(context(), claim_allocations={"c1": ["ev1"]})
    assert first == second
    assert first.registry_hash == second.registry_hash
    assert [entry.citation_id for entry in first.entries] == ["E001", "E002"]
    assert first.prompt_entries()[0].keys() == {
        "citation_id", "evidence_id", "claim_ids", "context_position"
    }
    resolution = first.resolve(["E001", "E001"], claim_id="c1")
    assert resolution.entries[0].triple == ("paper", 7, "b1")
    assert resolution.duplicate_ids == ["E001"]
    with pytest.raises(CitationRegistryError, match="unknown"):
        first.resolve(["E999"])
    with pytest.raises(CitationRegistryError, match="free-form"):
        reject_free_form_citation({"paper_id": "paper", "page": 9, "block_id": "b1"})


def test_citation_registry_claim_constraint_and_outside_context() -> None:
    registry = CitationRegistry.from_context(context(), claim_allocations={"c1": ["ev1"]})
    with pytest.raises(CitationRegistryError, match="not allocated"):
        registry.resolve(["E001"], claim_id="c2")


def test_q019_free_page_fixture_is_blocked() -> None:
    registry = CitationRegistry.from_context(context())
    with pytest.raises(CitationRegistryError):
        reject_free_form_citation(
            {"paper_id": "paper", "page": 999, "block_id": "b1"}
        )
    assert registry.resolve(["E001"]).entries[0].page == 7


def test_citation_id_output_resolves_locally_and_unknown_fails() -> None:
    registry = CitationRegistry.from_context(context())
    parsed = CitationIdQA.model_validate(
        {
            "answerable": True,
            "answer": "answer",
            "claims": [
                {"claim_id": "c1", "claim_text": "claim", "citation_ids": ["E001"]}
            ],
            "refusal_reason": None,
        }
    )
    answer, duplicates = resolve_citation_id_answer(parsed, registry)
    assert answer["claims"][0]["citations"] == [
        {"paper_id": "paper", "page": 7, "block_id": "b1"}
    ]
    assert duplicates == []
    bad = parsed.model_copy(
        update={"claims": [parsed.claims[0].model_copy(update={"citation_ids": ["E999"]})]}
    )
    with pytest.raises(CitationRegistryError, match="unknown"):
        resolve_citation_id_answer(bad, registry)


def test_health_check_invalid_url_and_dns_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    import scripts.check_llm_provider_health_v1 as health

    invalid = health.check_health(Settings(llm_base_url="http://unsafe.example"))
    assert invalid["safe_to_start_batch"] is False
    monkeypatch.setattr(health.socket, "getaddrinfo", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("dns")))
    failed = health.check_health(Settings(llm_base_url="https://example.invalid"))
    assert failed["dns_status"] == "failed"
    assert failed["safe_to_start_batch"] is False


class _Socket:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _TLSContext:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail

    def wrap_socket(self, raw, server_hostname):
        del raw, server_hostname
        if self.fail:
            raise ssl.SSLError("tls fixture")
        return _Socket()


class _Response:
    def __init__(self, status_code: int, payload: dict | None = None) -> None:
        self.status_code = status_code
        self._payload = payload or {}

    def json(self) -> dict:
        return self._payload


def _healthy_transport(monkeypatch: pytest.MonkeyPatch, health, *, tls_fail=False) -> None:
    monkeypatch.setattr(
        health.socket,
        "getaddrinfo",
        lambda *args, **kwargs: [(None, None, None, None, None)],
    )
    monkeypatch.setattr(
        health.socket,
        "create_connection",
        lambda *args, **kwargs: _Socket(),
    )
    monkeypatch.setattr(
        health.ssl,
        "create_default_context",
        lambda *args, **kwargs: _TLSContext(fail=tls_fail),
    )


def test_health_check_tls_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    import scripts.check_llm_provider_health_v1 as health

    _healthy_transport(monkeypatch, health, tls_fail=True)
    result = health.check_health(Settings(llm_base_url="https://example.test"))
    assert result["tcp_status"] == "passed"
    assert result["tls_status"] == "failed"
    assert result["safe_to_start_batch"] is False


def test_health_check_models_success(monkeypatch: pytest.MonkeyPatch) -> None:
    import scripts.check_llm_provider_health_v1 as health

    _healthy_transport(monkeypatch, health)
    monkeypatch.setattr(health.httpx, "get", lambda *args, **kwargs: _Response(200))
    result = health.check_health(
        Settings(
            _env_file=None,
            llm_provider="siliconflow",
            llm_model="Qwen/Qwen3-8B",
            llm_base_url="https://example.test",
            llm_api_key="secret",
        )
    )
    assert result["models_endpoint_status"] == "passed"
    assert result["minimal_completion_status"] == "not_run"
    assert result["factory_provider"] == "siliconflow"
    assert result["factory_model"] == "Qwen/Qwen3-8B"
    assert result["template_fallback"] is False
    assert result["safe_to_start_batch"] is True
    assert "secret" not in json.dumps(result)


def test_health_check_requires_minimal_completion_after_models_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.check_llm_provider_health_v1 as health

    _healthy_transport(monkeypatch, health)
    monkeypatch.setattr(health.httpx, "get", lambda *args, **kwargs: _Response(200))
    monkeypatch.setattr(
        health.httpx,
        "post",
        lambda *args, **kwargs: _Response(
            200,
            {
                "model": "Qwen/Qwen3-8B",
                "choices": [
                    {"finish_reason": "stop", "message": {"content": "{\"ok\": true}"}}
                ],
                "usage": {
                    "prompt_tokens": 8,
                    "completion_tokens": 3,
                    "total_tokens": 11,
                },
            },
        ),
    )
    result = health.check_health(
        Settings(
            _env_file=None,
            llm_provider="siliconflow",
            llm_model="Qwen/Qwen3-8B",
            llm_base_url="https://example.test",
            llm_api_key="secret",
        ),
        require_minimal_completion=True,
    )
    assert result["models_endpoint_status"] == "passed"
    assert result["minimal_completion_status"] == "passed"
    assert result["minimal_completion_json_valid"] is True
    assert result["minimal_completion_model"] == "Qwen/Qwen3-8B"
    assert result["minimal_completion_usage"]["total_tokens"] == 11
    assert result["factory_provider"] == "siliconflow"
    assert result["template_fallback"] is False
    assert result["safe_to_start_batch"] is True
    assert "secret" not in json.dumps(result)


def test_health_endpoint_timeout_and_minimal_completion_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.check_llm_provider_health_v1 as health

    _healthy_transport(monkeypatch, health)
    monkeypatch.setattr(
        health.httpx,
        "get",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            health.httpx.ReadTimeout("fixture")
        ),
    )
    monkeypatch.setattr(
        health.httpx,
        "post",
        lambda *args, **kwargs: _Response(
            200,
            {
                "choices": [
                    {"finish_reason": "stop", "message": {"content": "{\"ok\": true}"}}
                ],
                "usage": {"total_tokens": 4},
            },
        ),
    )
    result = health.check_health(
        Settings(
            _env_file=None,
            llm_provider="siliconflow",
            llm_model="Qwen/Qwen3-8B",
            llm_base_url="https://example.test",
            llm_api_key="secret",
        ),
        allow_minimal_completion=True,
    )
    assert result["models_endpoint_status"] == "failed"
    assert result["minimal_completion_status"] == "passed"
    assert result["safe_to_start_batch"] is True


def test_review_rows_and_pack_are_safe_and_complete() -> None:
    import scripts.review_evidence_qa_dev_citations_v1 as review

    rows = review.read_jsonl(review.AUDIT)
    review.validate(rows)
    assert len(rows) == 24
    assert all(row["human_review_status"] == "approved" for row in rows)
    assert all(row["human_label"] in review.LABELS for row in rows)
    assert all(row["cited_evidence_text"] for row in rows)
    assert all("previous" in row["adjacent_evidence_context"] for row in rows)
    with zipfile.ZipFile(review.PACK) as archive:
        names = set(archive.namelist())
        packed_audit = [
            json.loads(line)
            for line in archive.read(
                "evidence-qa-dev-citation-audit-v1.jsonl"
            )
            .decode("utf-8")
            .splitlines()
        ]
    assert names == {
        "evidence-qa-dev-citation-audit-v1.jsonl",
        "evidence-corpus-v1.jsonl",
        "gold-set-v1.jsonl",
        "retrieval-gold-v2.jsonl",
        "claim-units-v1.jsonl",
        "evidence-qa-dev-v1.json",
        "evidence-qa-dev-citation-review-guide-v1.md",
    }
    assert not any(name.lower().startswith(".env") for name in names)
    assert all(row["human_review_status"] == "pending" for row in packed_audit)
    assert all(row["human_label"] is None for row in packed_audit)


def test_v1_and_v2_artifact_roots_are_isolated() -> None:
    assert Path("data/evaluation/evidence-qa-dev-v1") != Path(
        "data/evaluation/evidence-qa-dev-v2"
    )
