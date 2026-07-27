from __future__ import annotations

from pathlib import Path

from scripts.check_ui_javascript_syntax_v1 import DEFAULT_ROUTES, check_routes


def test_ui_inline_javascript_is_executable() -> None:
    summary = check_routes(DEFAULT_ROUTES, Path(".runtime/test-ui-js-check"))

    assert summary["failure_count"] == 0
    assert any(item["script_count"] > 0 for item in summary["results"])
