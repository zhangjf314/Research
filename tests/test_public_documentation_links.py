from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_readme_local_documentation_links_exist() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    local_doc_links = [
        match.group(1).split("#", 1)[0]
        for match in re.finditer(r"\[[^\]]+\]\((docs/[^)]+\.md(?:#[^)]+)?)\)", readme)
    ]

    assert local_doc_links
    missing = [link for link in local_doc_links if not (ROOT / link).exists()]

    assert missing == []


def test_v1_2_readiness_keeps_release_actions_unauthorized() -> None:
    readiness = (ROOT / "docs/releases/v1.2.0-portfolio-readiness.md").read_text(
        encoding="utf-8"
    )
    version_table = (ROOT / "docs/releases/v1.2.0-version-truth-table.md").read_text(
        encoding="utf-8"
    )

    assert "version_bumped = false" in readiness
    assert "tag_created = false" in readiness
    assert "github_release_created = false" in readiness
    assert "1.1.0+portfolio" in version_table
    assert "1.2.0+portfolio" in version_table
