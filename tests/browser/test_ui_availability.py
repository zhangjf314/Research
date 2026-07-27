from __future__ import annotations

import importlib.util

import pytest
from fastapi.testclient import TestClient

from paper_research.main import app


def _record_console_error(console_errors: list[str]):
    def _handler(msg) -> None:
        if msg.type == "error":
            console_errors.append(msg.text)

    return _handler


def test_ui_pages_render_without_browser_console_errors() -> None:
    if importlib.util.find_spec("playwright") is None:
        pytest.skip("Playwright is not installed; browser smoke is not faked.")
    from playwright.sync_api import sync_playwright

    client = TestClient(app)
    routes = [
        "/api/v1/ui",
        "/api/v1/ui/library",
        "/api/v1/ui/search",
        "/api/v1/ui/research",
    ]
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        try:
            for route in routes:
                html = client.get(route).text
                page = browser.new_page()
                console_errors: list[str] = []
                page.on("console", _record_console_error(console_errors))
                page.set_content(html, wait_until="domcontentloaded")
                assert console_errors == []
                page.close()
        finally:
            browser.close()
