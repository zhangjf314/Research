"""Isolated external-document selection; no retriever or provider dependencies."""

import hashlib
import re
import unicodedata
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse
from xml.etree import ElementTree as ET

SEED = "phase2c-r6d-20261004"
CATEGORIES = ("cs.AI", "cs.CL", "cs.LG", "cs.SE", "stat.ML")
START, END = "2024-01-01", "2026-06-30"


def arxiv_id(value):
    match = re.search(r"(?<!\d)(\d{4}\.\d{4,5})(?:v\d+)?(?!\d)", value)
    return match.group(1) if match else None


def title_key(value):
    return re.sub(r"[^a-z0-9]", "", unicodedata.normalize("NFKC", value).casefold())


def rank(identifier):
    return hashlib.sha256((SEED + identifier).encode()).hexdigest()


def isolated_path(root, relative):
    root = Path(root).resolve()
    target = (root / relative).resolve()
    if not target.is_relative_to(root):
        raise ValueError("EXTERNAL_CORPUS_ISOLATION")
    return target


def arxiv_url(url):
    if urlparse(url).scheme != "https" or urlparse(url).hostname not in (
        "arxiv.org", "export.arxiv.org",
    ):
        raise ValueError("ARXIV_ONLY_NO_PRODUCTION_OR_PROVIDER_ACCESS")
    return url


def atom_candidates(raw, stratum):
    ns = {"a": "http://www.w3.org/2005/Atom", "x": "http://arxiv.org/schemas/atom"}
    result = []
    for entry in ET.fromstring(raw).findall("a:entry", ns):
        identifier = entry.findtext("a:id", "", ns).rsplit("/", 1)[-1]
        base = arxiv_id(identifier)
        if not base:
            raise ValueError("ARXIV_METADATA_ERROR")
        published = entry.findtext("a:published", "", ns)
        categories = [c.attrib["term"] for c in entry.findall("a:category", ns)]
        result.append({
            "arxiv_id": base, "versioned_id": identifier,
            "title": " ".join(entry.findtext("a:title", "", ns).split()),
            "abstract": " ".join(entry.findtext("a:summary", "", ns).split()),
            "published": published, "updated": entry.findtext("a:updated", "", ns),
            "doi": entry.findtext("x:doi", None, ns),
            "authors": [a.findtext("a:name", "", ns) for a in entry.findall("a:author", ns)],
            "categories": categories, "stratum": stratum,
            "pdf_url": "https://arxiv.org/pdf/" + identifier,
            "abs_url": "https://arxiv.org/abs/" + identifier, "ranking_hash": rank(base),
            "metadata_eligible": START <= published[:10] <= END and stratum in categories,
        })
    return result


def document_eligibility(candidate, parsed, exclusions, selected, pdf_hash):
    """Document availability/identity only. Never calls a scientific clause selector."""
    why = []
    if not candidate.get("metadata_eligible"):
        why.append("METADATA_DATE_OR_CATEGORY")
    for key, value in (("arxiv_ids", candidate["arxiv_id"]),
                       ("titles", title_key(candidate["title"])),
                       ("pdf_hashes", pdf_hash), ("dois", (candidate.get("doi") or "").lower())):
        if value and (value in exclusions[key] or value in selected[key]):
            why.append("DOCUMENT_DUPLICATE_" + key.upper())
    body = "\n".join(b.text for b in parsed.blocks if b.block_type == "paragraph")
    letters = [c for c in body if c.isalpha()]
    ascii_fraction = sum(c.isascii() for c in letters) / max(1, len(letters))
    words = re.findall(r"[A-Za-z]+", body.lower())
    english = sum(w in {"the", "and", "of", "to", "in", "for", "that", "with", "we"}
                  for w in words) / max(1, len(words))
    if parsed.metadata.page_count < 4 or len(body) < 8000 or len(words) < 1000:
        why.append("INSUFFICIENT_BODY_TEXT")
    if ascii_fraction < 0.90 or english < 0.03:
        why.append("NOT_PREDOMINANTLY_ENGLISH")
    article = (len(candidate.get("abstract", "")) >= 100
               and re.search(r"\bintroduction\b", body, re.I)
               and re.search(r"\breferences\b", body, re.I)
               and not re.search(r"^(editorial|erratum|corrigendum|retraction|book review)\b|"
                                 r"\b(thesis|tutorial)\b", candidate["title"], re.I))
    if not article:
        why.append("NOT_STANDARD_RESEARCH_ARTICLE")
    return {"eligible": not why, "reasons": why, "body_chars": len(body),
            "ascii_words": len(words), "ascii_letter_fraction": ascii_fraction,
            "english_function_word_fraction": english, "pages": parsed.metadata.page_count}


def next_candidate(candidates, examined, selected):
    for c in sorted(candidates, key=lambda c: (rank(c["arxiv_id"]), c["arxiv_id"])):
        if c["arxiv_id"] not in examined and c["arxiv_id"] not in selected:
            return c
    return None


def corpus_gate(selection):
    counts = Counter(c["stratum"] for c in selection)
    if len(selection) != 50 or counts != dict.fromkeys(CATEGORIES, 10):
        raise ValueError("EXTERNAL_CORPUS_REQUIRES_FROZEN_50")
    for key in ("arxiv_id", "pdf_sha256"):
        if len({c[key] for c in selection}) != 50:
            raise ValueError("EXTERNAL_DOCUMENT_DUPLICATE")
    if len({title_key(c["title"]) for c in selection}) != 50:
        raise ValueError("EXTERNAL_TITLE_DUPLICATE")


def benchmark_gate(items):
    if len(items) != 60:
        raise ValueError("PHASE2C_R6D_BLOCKED_EXTERNAL_DATA_INSUFFICIENT")
    papers = Counter(i["source_clauses"][0]["paper_id"] for i in items)
    if len(papers) < 30 or max(papers.values()) > 2:
        raise ValueError("PHASE2C_R6D_BLOCKED_INSUFFICIENT_SOURCE_DIVERSITY")


def provider_gate(corpus_frozen, benchmark_frozen, items, arm, gold=None):
    if not corpus_frozen or not benchmark_frozen:
        raise ValueError("NO_PROVIDER_BEFORE_CORPUS_AND_BENCHMARK_FREEZE")
    benchmark_gate(items)
    if arm not in ("gold", "v1_2", "v1_3"):
        raise ValueError("UNKNOWN_FORMAL_ARM")
    if arm != "gold" and (not gold or not gold.get("frozen") or gold["availability"] < .90):
        raise ValueError("GOLD_BEFORE_VERIFIER_GUARD")


def balanced_order(items, seed=20261007):
    import random
    ordered = list(items)
    random.Random(seed).shuffle(ordered)
    return [{"item_id": i["item_id"], "arms": ["v1_2", "v1_3"] if k % 2 == 0
             else ["v1_3", "v1_2"]} for k, i in enumerate(ordered)]
