"""Single runtime version source for the application.

The source tree uses ``pyproject.toml`` as the authoritative version while local
editable/install metadata can lag behind after RC bumps.  Packaged deployments can
fall back to installed distribution metadata when the source tree is unavailable.
"""

from __future__ import annotations

import re
import tomllib
from importlib import metadata
from pathlib import Path

PACKAGE_NAME = "paper-research-agent"


def _source_tree_version() -> str | None:
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    if not pyproject.exists():
        return None
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    version = data.get("project", {}).get("version")
    return str(version) if version else None


def package_version() -> str:
    """Return the canonical internal package version."""

    source_version = _source_tree_version()
    if source_version:
        return source_version
    return metadata.version(PACKAGE_NAME)


def display_version(version: str | None = None) -> str:
    """Return a human-facing version such as ``0.9.0-rc3`` or ``1.0.0-portfolio``."""

    raw = version or package_version()
    display = re.sub(r"(?<=\d)rc(?=\d)", "-rc", raw)
    return display.replace("+portfolio", "-portfolio")


__version__ = package_version()
__display_version__ = display_version(__version__)
