"""Known-case engineering regressions, never independent semantic gold."""

from copy import deepcopy

import pytest

from paper_research.agents.research_agent.semantic_review import (
    normalize_semantic_judgments,
    semantic_verification_input,
)


def payload(claim="The model uses 224 and evaluated Moon rocks.", ids=None):
    return {
        "claims": [{"claim_id": "c", "text": claim, "evidence_ids": ids or ["T"]}],
        "canonical_evidence": {
            "T": {
                "source_summary": "The model evaluated Moon rocks.",
                "text": "Model | input resolution\nViT-B/32 | 224",
                "modality": "TABLE",
            },
            "F": {
                "text": "a = U",
                "modality": "EQUATION",
                "readable_kind": "native_equation_candidate_text",
            },
            "G": {
                "text": "a = 2 U V; U and V are defined vectors.",
                "modality": "TEXT",
                "readable_kind": "canonical_text",
            },
        },
    }


def judgment(quote="ViT-B/32 | 224", alias="T", verdict="SUPPORTED", **extra):
    return {
        "judgments": [
            {
                "claim_id": "c",
                "verdict": verdict,
                "rationale": "Test fixture",
                "spans": [{"evidence_id": alias, "quote": quote}],
                **extra,
            }
        ]
    }


def normalize(p, response):
    return normalize_semantic_judgments(p, response)["judgments"][0]


def test_title_not_sent_or_citable_but_literal_body_remains_citable():
    p = payload()
    prompt = semantic_verification_input(p)
    assert "source_summary" not in str(prompt)
    assert "The model evaluated Moon rocks." not in str(prompt)
    bad = normalize(p, judgment(quote=p["canonical_evidence"]["T"]["source_summary"]))
    assert bad["unable_reason"] == "SOURCE_QUOTE_NOT_EXACT_OR_UNIQUE"
    assert normalize(p, judgment())["verdict"] == "SUPPORTED"


@pytest.mark.parametrize("quote", ["ViT-B/32|224", "vit-b/32 | 224", "ViT-B/32 | 336"])
def test_exact_quotes_are_not_fuzzy_matched(quote):
    assert normalize(payload(), judgment(quote=quote))["verdict"] == "INSUFFICIENT"


def test_scope_packaging_and_server_reject_cross_claim_sources():
    p = payload(ids=["F"])
    p["claims"].append({"claim_id": "c2", "text": "Another claim", "evidence_ids": ["G"]})
    scopes = semantic_verification_input(p)["evidence_scopes"]
    assert set(scopes[0]["evidence"]) == {"F"}
    assert set(scopes[1]["evidence"]) == {"G"}
    row = normalize(payload(ids=["T"]), judgment(quote="a = 2 U V", alias="G"))
    assert row["unable_reason"] == "EVIDENCE_OUTSIDE_CLAIM_SCOPE"


@pytest.mark.parametrize("fragment", ["a = U", "z =", "r = 2 x +", "f(x) = w"])
def test_line_candidates_need_context_not_a_specific_formula(fragment):
    p = payload(ids=["F"])
    p["canonical_evidence"]["F"]["text"] = fragment
    row = normalize(p, judgment(quote=fragment, alias="F", verdict="CONTRADICTED"))
    assert row["verdict"] == "INSUFFICIENT"
    assert row["unable_reason"] == "INCOMPLETE_FORMULA_CONTEXT"


def test_explicitly_cited_readable_formula_context_can_support():
    p = payload(ids=["F", "G"])
    row = normalize(p, judgment(quote="a = 2 U V", alias="G"))
    assert row["verdict"] == "SUPPORTED" and row["spans"][0]["start"] == 0
    assert normalize(p, judgment(quote="a = U", alias="F"))["verdict"] == "INSUFFICIENT"


def test_partial_records_literal_supported_and_missing_parts():
    p = payload()
    row = normalize(
        p,
        judgment(
            verdict="PARTIAL",
            supported_parts=["The model uses 224"],
            unsupported_parts=["evaluated Moon rocks"],
        ),
    )
    assert row["verdict"] == "PARTIAL" and row["unsupported_parts"]
    # Conjunction does not mechanically turn an all-supported judgment into PARTIAL.
    assert normalize(p, judgment())["verdict"] == "SUPPORTED"
    assert (
        normalize(
            p,
            {
                "judgments": [
                    {
                        "claim_id": "c",
                        "verdict": "INSUFFICIENT",
                        "rationale": "No supporting part",
                        "spans": [],
                    }
                ]
            },
        )["verdict"]
        == "INSUFFICIENT"
    )


@pytest.mark.parametrize(
    "parts", [({},), ({"supported_parts": ["fabricated"], "unsupported_parts": ["Moon rocks"]},)]
)
def test_partial_cannot_be_asserted_without_valid_parts(parts):
    assert (
        normalize(payload(), judgment(verdict="PARTIAL", **parts[0]))["verdict"] == "INSUFFICIENT"
    )


def test_canonical_body_is_not_modified_by_scope_packaging():
    p = payload()
    before = deepcopy(p)
    semantic_verification_input(p)
    normalize(p, judgment())
    assert p == before
