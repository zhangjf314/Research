from __future__ import annotations

import json

import scripts.compare_qwen_deepseek_canary_v1 as compare_canary
import scripts.prepare_deepseek_canary_v1 as prepare_canary
from paper_research.api.routes.capabilities import _api_key_fingerprint
from paper_research.config import Settings


def test_deepseek_canary_uses_same_qwen_v2_sample_ids() -> None:
    import scripts.run_full_qa_canary_v2 as canary

    assert canary.CANARY_IDS == [
        "q014",
        "q020",
        "q029",
        "q031",
        "q032",
        "q035",
        "q036",
        "q037",
        "q044",
        "q001",
        "q008",
        "q015",
        "q024",
        "q049",
        "q005",
    ]


def test_deepseek_canary_budget_missing_blocks_gate() -> None:
    import scripts.run_full_qa_canary_v2 as canary

    summary = canary._summarize([])
    assert summary["production_qa_canary_gate"] == "FAILED"
    source = canary.Path(canary.__file__).read_text(encoding="utf-8")
    assert "--require-budget" in source
    assert "Canary budget limits are required before running the batch" in source


def test_api_key_fingerprint_does_not_expose_secret() -> None:
    fingerprint = _api_key_fingerprint("sk-secret-value")
    assert fingerprint is not None
    assert "secret" not in fingerprint
    assert len(fingerprint) == 12


def test_prepare_deepseek_config_contains_no_key(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(prepare_canary, "OUT_JSON", tmp_path / "config.json")
    monkeypatch.setattr(prepare_canary, "OUT_DOC", tmp_path / "config.md")
    monkeypatch.setattr(prepare_canary, "_git_head", lambda: "abc123")
    monkeypatch.setattr(prepare_canary, "_sha256_path", lambda _path: "dataset-hash")
    monkeypatch.setattr(
        prepare_canary,
        "Settings",
        lambda: Settings(
            app_profile="production",
            embedding_provider="jina",
            embedding_model="jina-embeddings-v5-text-small",
            embedding_dimensions=1024,
            embedding_api_key="embedding",
            llm_provider="openai_compatible",
            llm_provider_name="deepseek",
            llm_base_url="https://api.deepseek.com",
            llm_api_key="sk-secret-value",
            llm_model="deepseek-v4-flash",
            llm_max_output_tokens=1024,
            deepseek_canary_max_input_tokens=200000,
            deepseek_canary_max_output_tokens=5000,
            deepseek_canary_max_total_tokens=205000,
            deepseek_canary_max_cost_usd="1",
            deepseek_canary_max_total_seconds=900,
            _env_file=None,
        ),
    )
    assert prepare_canary.main() == 0
    payload = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    rendered = json.dumps(payload, ensure_ascii=False)
    assert payload["llm"]["provider"] == "deepseek"
    assert payload["llm"]["model"] == "deepseek-v4-flash"
    assert payload["llm"]["api_key_present"] is True
    assert "sk-secret-value" not in rendered


def test_compare_canary_requires_identical_samples(monkeypatch, tmp_path) -> None:
    qwen = tmp_path / "qwen.json"
    deepseek = tmp_path / "deepseek.json"
    qwen.write_text(
        json.dumps({"canary_ids": ["q001"], "summary": {"completed": 1}}),
        encoding="utf-8",
    )
    deepseek.write_text(
        json.dumps({"canary_ids": ["q002"], "summary": {"completed": 1}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(compare_canary, "QWEN_JSON", qwen)
    monkeypatch.setattr(compare_canary, "DEEPSEEK_JSON", deepseek)
    monkeypatch.setattr(compare_canary, "OUT_JSON", tmp_path / "out.json")
    monkeypatch.setattr(compare_canary, "OUT_CSV", tmp_path / "out.csv")
    monkeypatch.setattr(compare_canary, "OUT_DOC", tmp_path / "out.md")
    assert compare_canary.main() == 2
