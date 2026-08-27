from paper_research.evaluation.ragq3 import (
    EvidenceUnit,
    failure_labels,
    fragmentation_summary,
    gate_decision,
    loss_decomposition,
    parent_child_windows,
    sentence_windows,
)


def test_sentence_window_is_deterministic_and_preserves_block_identity() -> None:
    first = sentence_windows("b1", "One. Two. Three.")
    assert first == sentence_windows("b1", "One. Two. Three.")
    assert first[1].text == "One. Two. Three."
    assert first[1].block_ids == {"b1"}


def test_parent_child_mapping_is_deterministic_and_has_no_gold_input() -> None:
    children = parent_child_windows("p1", ["b1", "b2"], " ".join(["x"] * 300))
    assert len(children) == 3
    assert all(child.parent_id == "p1" and child.block_ids == {"b1", "b2"} for child in children)


def test_loss_and_failure_decomposition_separate_all_three_losses() -> None:
    pool = [EvidenceUnit("a", frozenset({"a"}), "a"), EvidenceUnit("b", frozenset({"b"}), "b")]
    loss = loss_decomposition(pool, pool[:1], pool[:1], [{"a"}, {"b"}, {"c"}])
    assert loss == {
        "required_claims_total": 3,
        "claims_available_in_pool": 2,
        "claims_selected_top5": 1,
        "claims_retained_after_packing": 1,
        "candidate_loss": 1,
        "ranking_loss": 1,
        "packing_loss": 0,
    }
    assert failure_labels(pool, pool[:1], pool[:1], [{"a"}, {"b"}, {"c"}]) == [
        "CANDIDATE_MISSING_REQUIRED_EVIDENCE",
        "CORRECT_EVIDENCE_IN_POOL_NOT_TOP5",
        "MULTI_EVIDENCE_INCOMPLETE",
    ]


def test_fragmentation_and_gate_are_explicit() -> None:
    audit = fragmentation_summary(
        [{"a", "b"}], ["a", "b"], {"a": "s", "b": "s"}, {"a": "p", "b": "p"}
    )
    assert audit["counts"] == {
        "adjacent": 1,
        "same_section": 1,
        "same_parent": 1,
        "cross_chunk_required": 1,
    }
    assert all(
        gate_decision(
            {"multi_evidence_all_claims_present@pool": 0.1},
            paper_tie_or_better_rate=0.75,
            gain_paper_count=3,
        ).values()
    )
