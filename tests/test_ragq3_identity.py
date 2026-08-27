from paper_research.evaluation.ragq3_identity import sentence_boundaries, stable_id


def test_identity_is_stable_across_mapping_order_and_newlines() -> None:
    left = stable_id("sentence", {"document": "p", "text": "A\r\nB", "index": 1})
    right = stable_id("sentence", {"index": 1, "text": "A\nB", "document": "p"})
    assert left == right


def test_frozen_sentence_boundaries_cover_edge_fixtures() -> None:
    assert sentence_boundaries("3.14 remains. Next?") == ["3.14 remains.", "Next?"]
    assert sentence_boundaries("A; B: C\nD!") == ["A; B: C D!"]
    assert sentence_boundaries("[12] Formula x.y is inline.") == ["[12] Formula x.y is inline."]
