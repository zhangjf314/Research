"""Frozen B0 copied-phrase rule: eight consecutive, identical whole tokens.

Null/empty applicability remains the caller's responsibility. This module does
not change metadata, semantic, provider, or gold checks.
"""

import unicodedata
from collections.abc import Sequence


def frozen_tokens(text: str) -> tuple[str, ...]:
    """Apply only B0's NFKC, casefold, and Unicode whitespace tokenization."""
    return tuple(unicodedata.normalize("NFKC", text).casefold().split())


def has_copied_phrase(
    query_tokens: Sequence[str], source_tokens: Sequence[str]
) -> bool:
    """Return whether any complete eight-token window appears in both sequences.

    A match of nine or more consecutive tokens necessarily contains a matching
    eight-token window. Punctuation remains part of each token's identity.
    """
    source_windows = {
        tuple(source_tokens[start : start + 8])
        for start in range(len(source_tokens) - 7)
    }
    return any(
        tuple(query_tokens[start : start + 8]) in source_windows
        for start in range(len(query_tokens) - 7)
    )
