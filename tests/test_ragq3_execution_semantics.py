from paper_research.evaluation.ragq3_execution import (
    build_representation,
    execute_synthetic_candidate,
    synthetic_source_document,
    validate_ragq3_end_to_end_executability,
)


def test_all_q3x_candidates_complete_the_synthetic_source_gold_chain() -> None:
    for candidate_id in ("Q3X-R0", "Q3X-R1", "Q3X-R2", "Q3X-R3", "Q3X-R4"):
        result = execute_synthetic_candidate(candidate_id)
        assert result["covered_claim_ids"] == [0, 1]
        assert result["covered_gold_blocks"] == [synthetic_source_document()[1].source_block_id]
        assert result["claim_status"] == "COMPUTABLE"
        assert result["loss"] == {
            "required_claims_total": 2,
            "claims_available_in_pool": 2,
            "claims_selected_top5": 2,
            "claims_retained_after_packing": 2,
            "candidate_loss": 0,
            "ranking_loss": 0,
            "packing_loss": 0,
        }


def test_sentence_windows_are_structural_not_text_only() -> None:
    units = build_representation("Q3X-R3")
    assert units[0].text == units[1].text
    assert units[0].return_unit_id != units[1].return_unit_id
    assert units[0].source_spans == units[1].source_spans


def test_parent_children_share_parent_but_keep_retrieval_identities() -> None:
    units = build_representation("Q3X-R4")
    assert units[0].parent_id == units[1].parent_id
    assert units[0].retrieval_unit_id != units[1].retrieval_unit_id
    assert len(units[0].source_spans) == 2


def test_executability_validator_requires_all_five_q3x_arms() -> None:
    result = validate_ragq3_end_to_end_executability()
    assert list(result) == ["Q3X-R0", "Q3X-R1", "Q3X-R2", "Q3X-R3", "Q3X-R4"]
    assert {item["status"] for item in result.values()} == {"PASS"}
