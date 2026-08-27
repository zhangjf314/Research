from paper_research.evaluation.ragq3_attribution import (
    EvidenceProvenance,
    SourceSpan,
    covered_claims,
    covered_gold_blocks,
)


def test_provenance_maps_multi_block_gold_without_text_guessing() -> None:
    unit = EvidenceProvenance("u", (SourceSpan("b1", 0, 3), SourceSpan("b2", 2, 6)))
    assert covered_gold_blocks(unit, {"b1", "b3"}) == {"b1"}
    assert covered_claims(unit, [{"b1"}, {"b2"}, {"b3"}]) == ({0, 1}, "COMPUTABLE")


def test_missing_claim_mapping_is_not_zero() -> None:
    unit = EvidenceProvenance("u", (SourceSpan("b1", 0, 1),))
    assert covered_claims(unit, None) == (None, "METRIC_NOT_COMPUTABLE")
