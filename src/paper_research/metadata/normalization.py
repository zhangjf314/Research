from __future__ import annotations

import re
import unicodedata

ARXIV_ID_RE = re.compile(
    r"(?i)(?:arxiv[:\s]*)?((?:\d{4}\.\d{4,5})(?:v\d+)?|[a-z-]+(?:\.[A-Z]{2})?/\d{7})(?:\.pdf)?"
)

LIGATURE_REPLACEMENTS = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\u95be\u5938\u606d": "ffi",
    "\u95be\u5938\u8eac": "ffl",
    "\u95be\u5938\u653b": "fi",
    "\u95be\u5938\u5f13": "fl",
    "\u94ff\u4e67": "ffi",
    "\u94ff\u4e6a": "ffl",
    "\u94ff\u4e65": "fi",
    "\u94ff\u4e6d": "fl",
}

MOJIBAKE_WORD_REPLACEMENTS = {
    "Efffiient": "Efficient",
    "efffiient": "efficient",
}


def normalize_title(value: str | None) -> str:
    if not value:
        return ""
    text = unicodedata.normalize("NFKC", value)
    for source, target in LIGATURE_REPLACEMENTS.items():
        text = text.replace(source, target)
    for source, target in MOJIBAKE_WORD_REPLACEMENTS.items():
        text = text.replace(source, target)
    text = re.sub(r"\s+", " ", text.replace("\n", " ")).strip()
    text = re.sub(r"\s+([,:;.!?])", r"\1", text)
    return text


def comparable_title(value: str | None) -> str:
    text = normalize_title(value).casefold()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def extract_arxiv_id(*values: str | None) -> str | None:
    for value in values:
        if not value:
            continue
        match = ARXIV_ID_RE.search(value)
        if not match:
            continue
        identifier = match.group(1)
        if re.match(r"\d{4}\.\d{4,5}", identifier):
            identifier = re.sub(r"v\d+$", "", identifier, flags=re.I)
        return identifier
    return None


def identifier_as_title(title: str | None) -> bool:
    normalized = normalize_title(title)
    return bool(normalized and extract_arxiv_id(normalized) == normalized.removesuffix(".pdf"))
