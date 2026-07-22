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
    assert "v1.0.0-portfolio" in text
    assert "1.0.0+portfolio" in text
    assert "actions/workflows/ci.yml/badge.svg?branch=main" in text
    assert "releases/tag/v1.0.0-portfolio" in text


def test_readme_removes_stale_release_status() -> None:
    text = readme_text()
    forbidden = [
        "等待",
        "awaiting explicit user authorization",
        "merge, tag, push",
        "highest published version is v0.9.0-rc2",
        "当前版本为 v0.9.0-rc1 候选",
        "Stage 10",
        "Stage 11",
        "Stage 12",
        "Stage 13",
        "Release Candidate 状态",
        "当前候选版本",
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
    assert "内部开发评测集" in text
    assert "不是独立 blind benchmark" in text
    assert "SEMANTIC_CLAIM_SUPPORT_AUDIT=NOT_FORMALLY_VALIDATED" in text
    assert "STRONG_GROUNDING_CLAIM_ALLOWED=false" in text
    assert "STRONG_GENERALIZATION_CLAIM_ALLOWED=false" in text
    assert "RETRIEVAL_GENERALIZATION_EVIDENCE=DIAGNOSTIC_ONLY" in text

    forbidden_claims = [
        "严格证明泛化能力",
        "生产级强 Grounding",
        "商业生产就绪",
        "长期稳定性已证明",
        "消除幻觉",
        "production-grade grounding",
        "strong generalization benchmark",
    ]
    lowered = text.lower()
    for claim in forbidden_claims:
        assert claim.lower() not in lowered


def test_readme_line_count_is_reasonable() -> None:
    line_count = len(readme_text().splitlines())
    assert 150 <= line_count <= 230
