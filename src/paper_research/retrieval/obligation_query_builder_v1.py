"""Build deterministic supplemental retrieval queries from claim obligations."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from enum import StrEnum

from paper_research.generation.claim_obligations import ClaimObligationSet

OBLIGATION_QUERY_BUILDER_VERSION = "obligation-query-builder-v1"


class ObligationQueryType(StrEnum):
    BASE_TEXT = "base_textual_query"
    NUMERIC_EXACT = "numeric_exact_query"
    UNIT_NORMALIZED = "unit_normalized_query"
    RANGE_ENDPOINT = "range_endpoint_query"
    VARIABLE_ANCHOR = "variable_anchor_query"
    COMPARISON_SIDE = "comparison_side_query"
    LIMITATION_POLARITY = "limitation_polarity_query"


@dataclass(frozen=True)
class ObligationQuery:
    query_text: str
    query_type: ObligationQueryType
    source_obligations: tuple[str, ...]
    normalized_terms: tuple[str, ...]
    expected_contribution: str
    deterministic_hash: str
    version: str = OBLIGATION_QUERY_BUILDER_VERSION


def _hash_parts(parts: tuple[str, ...]) -> str:
    payload = "\n".join(parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalized_terms(text: str) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            token
            for token in re.findall(r"[a-z0-9]+(?:[.-][a-z0-9]+)*", text.lower())
            if len(token) > 2
        )
    )


def _numbers(text: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(re.findall(r"\b\d+(?:\.\d+)?(?:[mbk]|%)?\b", text.lower())))


def _make(
    query_text: str,
    query_type: ObligationQueryType,
    source_obligations: tuple[str, ...],
    expected: str,
) -> ObligationQuery:
    terms = _normalized_terms(query_text)
    return ObligationQuery(
        query_text=query_text,
        query_type=query_type,
        source_obligations=source_obligations,
        normalized_terms=terms,
        expected_contribution=expected,
        deterministic_hash=_hash_parts((query_type.value, query_text, *source_obligations)),
    )


def build_obligation_queries(obligation_set: ClaimObligationSet) -> tuple[ObligationQuery, ...]:
    queries: list[ObligationQuery] = []
    for obligation in obligation_set.obligations:
        source = (obligation.obligation_id,)
        text = obligation.obligation_text
        queries.append(
            _make(
                text,
                ObligationQueryType.BASE_TEXT,
                source,
                "cover_required_obligation",
            )
        )
        for number in obligation.numeric_anchors or _numbers(text):
            queries.append(
                _make(
                    f"{number} {text}",
                    ObligationQueryType.NUMERIC_EXACT,
                    source,
                    "recover_exact_numeric_anchor",
                )
            )
            queries.append(
                _make(
                    f"{number.replace(',', '')} {' '.join(obligation.lexical_anchors)}",
                    ObligationQueryType.UNIT_NORMALIZED,
                    source,
                    "recover_normalized_numeric_anchor",
                )
            )
        if len(obligation.numeric_anchors) >= 2:
            for endpoint in obligation.numeric_anchors:
                queries.append(
                    _make(
                        f"{endpoint} {' '.join(obligation.lexical_anchors)}",
                        ObligationQueryType.RANGE_ENDPOINT,
                        source,
                        "recover_range_endpoint",
                    )
                )
        if obligation.comparison_side:
            queries.append(
                _make(
                    text,
                    ObligationQueryType.COMPARISON_SIDE,
                    source,
                    f"recover_{obligation.comparison_side}",
                )
            )
        if re.search(r"\b(limit|limitation|not|without|fail|cannot)\b", text, re.I):
            queries.append(
                _make(
                    text,
                    ObligationQueryType.LIMITATION_POLARITY,
                    source,
                    "preserve_limitation_or_polarity",
                )
            )
        if obligation.lexical_anchors:
            queries.append(
                _make(
                    " ".join(obligation.lexical_anchors),
                    ObligationQueryType.VARIABLE_ANCHOR,
                    source,
                    "recover_entity_variable_anchor",
                )
            )
    deduped = {query.deterministic_hash: query for query in queries}
    return tuple(sorted(deduped.values(), key=lambda query: query.deterministic_hash))
