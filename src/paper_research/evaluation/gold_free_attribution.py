"""Evaluation-only Gold attribution over neutral runtime provenance."""
from __future__ import annotations

from dataclasses import dataclass

from paper_research.indexing.gold_free_runtime import CanonicalRuntimeUnit


@dataclass(frozen=True)
class Attribution:
    covered_gold_blocks: frozenset[str]
    covered_claim_indexes: frozenset[int]


class GoldFreeEvaluationAttributionResolver:
    """Maps neutral source IDs to frozen Gold only after retrieval is complete."""

    def attribute(
        self,
        units: list[CanonicalRuntimeUnit],
        *,
        gold_by_neutral_source_id: dict[str, str],
        claims: list[set[str]],
    ) -> Attribution:
        neutral_ids = {
            source_id for unit in units for source_id in unit.neutral_source_block_ids
        }
        covered = frozenset(
            gold_by_neutral_source_id[source_id]
            for source_id in neutral_ids & gold_by_neutral_source_id.keys()
        )
        claim_indexes = frozenset(
            index for index, claim in enumerate(claims) if covered & claim
        )
        return Attribution(covered, claim_indexes)
