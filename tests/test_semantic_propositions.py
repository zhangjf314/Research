"""R1 contract controls, not model predictions or independent semantic gold."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from paper_research.agents.research_agent.semantic_review import (
    StructuredSemanticVerifier,
    normalize_propositional_judgments,
)
from paper_research.providers.llm import StructuredJSONResult, _deepseek_structured_tool


def sample():
    claim = "Model A uses 224, and Model A evaluated Moon rocks."
    return {
        "claims": [{"claim_id": "c", "text": claim, "evidence_ids": ["T"]}],
        "canonical_evidence": {
            "T": {"text": "Model | input resolution\nModel A | 224", "modality": "TABLE"},
            "X": {"text": "Moon rocks were evaluated.", "modality": "TEXT"},
        },
    }


def part(quote, verdict="SUPPORTED", scope="MATCHED", alias="T", source="Model A | 224"):
    return {
        "claim_quote": quote,
        "verdict": verdict,
        "scope_match": scope,
        "rationale": "Hand-authored contract control, not a model judgment.",
        "spans": [] if verdict == "INSUFFICIENT" else [{"evidence_id": alias, "quote": source}],
    }


def response(p, first="SUPPORTED", second="INSUFFICIENT", declared="PARTIAL", scope="MATCHED"):
    return {
        "judgments": [
            {
                "claim_id": "c",
                "verdict": declared,
                "rationale": "Control",
                "spans": [],
                "mismatches": [],
                "missing_evidence": [],
                "propositions": [
                    part("Model A uses 224,", first),
                    part("and Model A evaluated Moon rocks.", second, scope),
                ],
            }
        ],
    }


def normalize(p, r):
    output, audit = normalize_propositional_judgments(p, r)
    return output["judgments"][0], audit[0]


@pytest.mark.parametrize(
    "first,second,declared,expected",
    [
        ("SUPPORTED", "SUPPORTED", "SUPPORTED", "SUPPORTED"),
        ("SUPPORTED", "INSUFFICIENT", "SUPPORTED", "PARTIAL"),
        ("SUPPORTED", "CONTRADICTED", "SUPPORTED", "CONTRADICTED"),
        ("SUPPORTED", "CONTRADICTED", "PARTIAL", "CONTRADICTED"),
        ("CONTRADICTED", "INSUFFICIENT", "INSUFFICIENT", "CONTRADICTED"),
        ("INSUFFICIENT", "INSUFFICIENT", "SUPPORTED", "INSUFFICIENT"),
    ],
)
def test_aggregation_without_blanket_abstention(first, second, declared, expected):
    p = sample()
    row, audit = normalize(p, response(p, first, second, declared))
    assert row["verdict"] == expected
    assert audit["derived_verdict"] == expected
    if expected == "PARTIAL":
        assert row["supported_parts"] == ["Model A uses 224,"]
        assert row["unsupported_parts"] == ["and Model A evaluated Moon rocks."]


@pytest.mark.parametrize("scope", ["MISMATCHED", "UNRESOLVED"])
def test_scope_difference_is_missing_evidence_not_refutation(scope):
    p = sample()
    row, audit = normalize(p, response(p, second="CONTRADICTED", scope=scope))
    assert row["verdict"] == "PARTIAL"
    assert audit["propositions"][1]["effective_verdict"] == "INSUFFICIENT"
    p["claims"][0]["text"] = "Model A evaluated Moon rocks."
    r = response(p)
    r["judgments"][0]["propositions"] = [part(p["claims"][0]["text"], "CONTRADICTED", scope)]
    assert normalize(p, r)[0]["verdict"] == "INSUFFICIENT"


def test_omitted_negative_clause_blocks_support():
    p = sample()
    p["claims"][0]["text"] = "Model A uses 224, and Model A does not improve accuracy."
    r = response(p, declared="SUPPORTED")
    r["judgments"][0]["propositions"] = [part("Model A uses 224,")]
    row, _ = normalize(p, r)
    assert row["unable_reason"] == "PROPOSITION_CLAIM_COVERAGE_INCOMPLETE"


def test_legacy_output_is_not_filled_with_guessed_propositions():
    p = sample()
    r = response(p)
    del r["judgments"][0]["propositions"]
    row, _ = normalize(p, r)
    assert row["unable_reason"] == "PROPOSITION_CHECKS_REQUIRED"


@pytest.mark.parametrize("declared", ["PARTIAL", "CONTRADICTED", "INSUFFICIENT"])
def test_inconsistent_overall_record_is_never_promoted_to_support(declared):
    p = sample()
    row, _ = normalize(p, response(p, second="SUPPORTED", declared=declared))
    assert row["unable_reason"] == "SUPPORTED_PROPOSITION_RECORD_INCONSISTENT"


@pytest.mark.parametrize("field", ["mismatches", "missing_evidence"])
def test_all_supported_with_unresolved_problem_is_not_supported(field):
    p = sample()
    r = response(p, second="SUPPORTED", declared="SUPPORTED")
    r["judgments"][0][field] = ["Unresolved condition"]
    assert normalize(p, r)[0]["verdict"] == "INSUFFICIENT"


@pytest.mark.parametrize(
    "alias,quote,expected",
    [
        ("X", "Moon rocks were evaluated.", "EVIDENCE_OUTSIDE_CLAIM_SCOPE"),
        ("T", "Model A|224", "SOURCE_QUOTE_NOT_EXACT_OR_UNIQUE"),
    ],
)
def test_proposition_quotes_keep_exact_source_and_scope_guards(alias, quote, expected):
    p = sample()
    r = response(p)
    r["judgments"][0]["propositions"][0]["spans"] = [{"evidence_id": alias, "quote": quote}]
    assert normalize(p, r)[0]["unable_reason"] == expected


def test_unused_top_level_bad_quote_is_not_silently_removed():
    p = sample()
    r = response(p)
    r["judgments"][0]["spans"] = [{"evidence_id": "X", "quote": "Moon rocks were evaluated."}]
    assert normalize(p, r)[0]["unable_reason"] == "EVIDENCE_OUTSIDE_CLAIM_SCOPE"


def test_formula_candidate_requires_separately_cited_readable_context():
    p = sample()
    p["claims"][0]["text"] = "a = 2 U V"
    p["canonical_evidence"]["T"] = {
        "text": "a = U",
        "modality": "EQUATION",
        "readable_kind": "native_equation_candidate_text",
    }
    r = response(p, declared="SUPPORTED")
    r["judgments"][0]["propositions"] = [part("a = 2 U V", source="a = U")]
    assert normalize(p, r)[0]["unable_reason"] == "INCOMPLETE_FORMULA_CONTEXT"
    p["canonical_evidence"]["X"]["text"] = "a = 2 U V; U and V are vectors."
    p["claims"][0]["evidence_ids"].append("X")
    r["judgments"][0]["propositions"] = [part("a = 2 U V", alias="X", source="a = 2 U V")]
    assert normalize(p, r)[0]["verdict"] == "SUPPORTED"


def test_adverse_control_does_not_claim_to_prove_typed_part_semantics():
    p = sample()
    p["claims"][0]["text"] = "Model A uses 336."
    r = response(p, declared="SUPPORTED")
    r["judgments"][0]["propositions"] = [part("Model A uses 336.")]
    # Exact quotations and typed verdicts are NOT an entailment proof.
    assert normalize(p, r)[0]["verdict"] == "SUPPORTED"


def test_v4_schema_is_closed_and_server_derives_partial_breakdown():
    _, schema = _deepseek_structured_tool("research-semantic-review-v4", {})
    row = schema["properties"]["judgments"]["items"]
    props = row["properties"]["propositions"]
    assert "propositions" in row["required"]
    assert props["minItems"] == 1 and props["maxItems"] == 24
    assert not props["items"]["additionalProperties"]
    assert "supported_parts" not in row["properties"]
    _, old = _deepseek_structured_tool("research-semantic-review-v3", {})
    assert "supported_parts" in old["properties"]["judgments"]["items"]["properties"]
    assert "propositions" not in old["properties"]["judgments"]["items"]["properties"]


def test_default_adapter_calls_once_and_retains_raw_and_part_audit():
    p = sample()
    p["gold"] = "DO_NOT_SEND"
    p["claims"][0]["old_ai_verdict"] = "DO_NOT_SEND"
    raw = response(p)
    before = deepcopy(raw)
    calls = []
    provider = SimpleNamespace(
        max_retries=0,
        generate_structured_json=lambda **kw: (
            calls.append(kw)
            or StructuredJSONResult(provider="offline", model="fixture", payload=raw)
        ),
    )
    verifier = StructuredSemanticVerifier(provider, schema_version="v4")
    out = verifier(p)
    assert len(calls) == 1 and calls[0]["schema_name"] == "research-semantic-review-v4"
    assert "DO_NOT_SEND" not in calls[0]["user_prompt"]
    assert out["judgments"][0]["verdict"] == "PARTIAL"
    assert verifier.raw_payload == raw == before
    audit = verifier.context["proposition_audit"][0]
    assert audit["final_verdict"] == "PARTIAL" and len(audit["propositions"]) == 2
    assert json.loads(calls[0]["user_prompt"])["evidence_scopes"]
