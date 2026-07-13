import re

TOKEN_PATTERN = re.compile(r"[\w]+|[^\w\s]", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text)


def count_tokens(text: str) -> int:
    return len(tokenize(text))


def token_windows(text: str, max_tokens: int, overlap: int) -> list[str]:
    if max_tokens < 1 or overlap < 0 or overlap >= max_tokens:
        raise ValueError("require max_tokens > overlap >= 0")
    tokens = tokenize(text)
    if not tokens:
        return []
    step = max_tokens - overlap
    return [" ".join(tokens[start : start + max_tokens]) for start in range(0, len(tokens), step)]
