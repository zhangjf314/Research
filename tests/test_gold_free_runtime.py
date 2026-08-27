import hashlib
import inspect

from paper_research.evaluation.gold_free_attribution import (
    GoldFreeEvaluationAttributionResolver,
)
from paper_research.indexing.gold_free_runtime import (
    CanonicalPaperIngestor,
    NeutralSourceBlock,
)


def fabricated_units():
    ingestor = CanonicalPaperIngestor()
    return ingestor.build_units(
        source_sha256=hashlib.sha256(b"fabricated-paper-v1").hexdigest(),
        blocks=[
            NeutralSourceBlock("p1-b1", "Methods establish the first result.", ("Methods",), 1, 1),
            NeutralSourceBlock("p1-b2", "Results compare the second result.", ("Results",), 2, 2),
        ],
    )


def test_gold_mutation_cannot_change_runtime_units_or_payload() -> None:
    first, second = fabricated_units(), fabricated_units()
    assert first == second
    assert [unit.index_payload() for unit in first] == [unit.index_payload() for unit in second]
    payload = first[0].index_payload()
    assert "gold" not in repr(payload).lower()
    assert "claim" not in repr(payload).lower()


def test_gold_changes_evaluation_attribution_only() -> None:
    units = fabricated_units()
    resolver = GoldFreeEvaluationAttributionResolver()
    neutral = units[0].neutral_source_block_ids[0]
    first = resolver.attribute(
        units, gold_by_neutral_source_id={neutral: "gold-a"}, claims=[{"gold-a"}]
    )
    second = resolver.attribute(
        units, gold_by_neutral_source_id={neutral: "gold-b"}, claims=[{"gold-b"}]
    )
    assert first.covered_gold_blocks == {"gold-a"}
    assert second.covered_gold_blocks == {"gold-b"}
    assert units == fabricated_units()


def test_runtime_module_has_no_gold_aware_import() -> None:
    assert CanonicalPaperIngestor.__module__ == "paper_research.indexing.gold_free_runtime"
    source = inspect.getsource(CanonicalPaperIngestor)
    assert "evaluation" not in source
