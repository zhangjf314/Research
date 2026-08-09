from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"


def readme_text() -> str:
    return README.read_text(encoding="utf-8")


def markdown_links(text: str) -> list[str]:
    pattern = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
    return [match.group(1).split("#", 1)[0] for match in pattern.finditer(text)]


def markdown_images(text: str) -> list[str]:
    pattern = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
    return [match.group(1).split("#", 1)[0] for match in pattern.finditer(text)]


def is_external(target: str) -> bool:
    parsed = urlparse(target)
    return bool(parsed.scheme or parsed.netloc)


def test_readme_version_and_release_badges() -> None:
    text = readme_text()
    assert "v1.1.0-portfolio" in text
    assert "1.1.0+portfolio" in text
    assert "actions/workflows/ci.yml/badge.svg?branch=main" in text
    assert "releases/tag/v1.1.0-portfolio" in text


def test_readme_removes_stale_release_status() -> None:
    text = readme_text()
    forbidden = [
        "awaiting explicit user authorization",
        "merge, tag, push",
        "highest published version is v0.9.0-rc2",
        "Stage 10",
        "Stage 11",
        "Stage 12",
        "Stage 13",
        "Release Candidate",
        "个人求职 Portfolio 项目",
        "job-seeking Portfolio project",
    ]
    for phrase in forbidden:
        assert phrase not in text


def test_readme_relative_links_and_images_exist() -> None:
    text = readme_text()
    for target in markdown_links(text):
        if not target or is_external(target):
            continue
        assert (ROOT / target).exists(), target
    for target in markdown_images(text):
        if not target or is_external(target):
            continue
        assert (ROOT / target).exists(), target


def test_readme_has_no_secret_or_local_path() -> None:
    text = readme_text()
    forbidden = [
        "LLM_API_KEY",
        "DEEPSEEK_API_KEY",
        "JINA_API_KEY",
        "Authorization: Bearer",
        "sk-",
        "D:\\",
        "C:\\Users\\",
        "postgresql://",
        "postgresql+psycopg://",
    ]
    for phrase in forbidden:
        assert phrase not in text


def test_readme_truth_boundaries_are_explicit() -> None:
    text = readme_text()
    assert "internally authored and reviewed" in text
    assert "not a strict equal-budget causal ablation" in text
    assert "structural proxies" in text
    assert "semantic_judge_complete = false" in text
    assert "LIVE_EFFECTIVE_REPLAN_NOT_OBSERVED" in text
    assert "STRONG_GENERALIZATION_CLAIM_ALLOWED=false" in text

    forbidden_claims = [
        "strict blind benchmark",
        "production-grade grounding",
        "strong generalization benchmark",
        "commercial production-ready",
        "fully validated semantic benchmark",
        "proved long-term stability",
        "eliminates hallucination",
    ]
    lowered = text.lower()
    for claim in forbidden_claims:
        assert claim.lower() not in lowered


def test_readme_line_count_is_reasonable() -> None:
    line_count = len(readme_text().splitlines())
    assert 180 <= line_count <= 320
