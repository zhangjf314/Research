from __future__ import annotations

from paper_research.generation.citation_selection import (
    CitationCandidate,
    FallbackAction,
    analyze_claim_obligations,
    select_citations,
    validate_comparison_evidence,
    validate_numeric_evidence,
)


def candidate(
    citation_id: str,
    text: str,
    *,
    original: bool = True,
    block: str | None = None,
) -> CitationCandidate:
    return CitationCandidate(
        citation_id=citation_id,
        paper_id="paper",
        page=1,
        block_id=block or citation_id,
        text=text,
        original_selected=original,
        adjacent_completion=not original,
        retrieval_origin="original_selected" if original else "adjacent_completion",
    )


def test_claim_obligation_decomposition_is_generic() -> None:
    assert len(analyze_claim_obligations("A single atomic claim.").obligations) == 1
    assert (
        len(
            analyze_claim_obligations(
                "The model is fast and uses less memory."
            ).obligations
        )
        == 2
    )
    training = analyze_claim_obligations(
        "Training uses eight GPUs and an Adam schedule with warmup."
    )
    assert len(training.obligations) >= 3
    assert analyze_claim_obligations("Dataset size ranges from 22M to 23B tokens.").obligations
    comparison = analyze_claim_obligations("A uses recurrence, while B uses attention.")
    assert {item.comparison_side for item in comparison.obligations} == {"side_a", "side_b"}
    assert len(analyze_claim_obligations("Research and development is discussed.").obligations) == 1
    assert analyze_claim_obligations("").decomposition_confidence == 0


def test_numeric_validator_handles_equivalence_and_missing_values() -> None:
    assert validate_numeric_evidence(
        "Training uses eight GPUs.", [candidate("E1", "Training ran on 8 GPUs.")]
    ).complete
    assert validate_numeric_evidence(
        "The learning rate is 3e-4.", [candidate("E1", "The learning rate is 3e-4.")]
    ).complete
    assert validate_numeric_evidence(
        "The range is 22M to 23B.", [candidate("E1", "We vary from 22M to 23B.")]
    ).complete
    assert not validate_numeric_evidence(
        "The range is 22M to 23B.", [candidate("E1", "We use 22M tokens.")]
    ).complete
    assert not validate_numeric_evidence(
        "Training uses 8 GPUs.", [candidate("E1", "Training details are reported.")]
    ).complete


def test_comparison_validator_requires_both_sides() -> None:
    claim = "A uses recurrence, while B uses attention."
    assert validate_comparison_evidence(
        claim, [candidate("E1", "A uses recurrence, while B uses attention.")]
    ).complete
    assert not validate_comparison_evidence(
        claim, [candidate("E1", "A uses recurrence.")]
    ).complete


def test_primary_origin_cap_and_determinism() -> None:
    claim = "Training uses eight GPUs and an Adam schedule with warmup."
    candidates = [
        candidate("E2", "The optimizer is Adam.", original=False),
        candidate("E1", "Training uses eight GPUs.", original=True),
        candidate("E3", "Warmup is applied to Adam.", original=True),
        candidate("E4", "Training is described.", original=True),
    ]
    first = select_citations(claim, candidates)
    second = select_citations(claim, reversed(candidates))
    assert first == second
    assert first.selected_count <= 3
    assert first.primary_citation_ids == ("E3",)
    assert "E4" in first.rejected_citation_ids


def test_duplicate_triples_do_not_increase_budget() -> None:
    claim = "The method uses attention."
    result = select_citations(
        claim,
        [
            candidate("E2", "The method uses attention.", original=False, block="same"),
            candidate("E1", "The method uses attention.", original=True, block="same"),
        ],
    )
    assert result.selected_count == 1
    assert result.primary_citation_ids == ("E1",)


def test_incomplete_evidence_narrows_or_refuses() -> None:
    narrowed = select_citations(
        "A uses recurrence and B uses attention.",
        [candidate("E1", "A uses recurrence.")],
    )
    assert narrowed.fallback_action in {
        FallbackAction.ANSWERED_NARROWED,
        FallbackAction.UNSUPPORTED,
    }
    unsupported = select_citations("The value is 42.", [candidate("E1", "No value is given.")])
    assert unsupported.fallback_action == FallbackAction.UNSUPPORTED
