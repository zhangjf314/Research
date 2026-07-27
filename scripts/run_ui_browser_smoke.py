from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _record_console_error(console_errors: list[str]):
    def _handler(msg) -> None:
        if msg.type == "error":
            console_errors.append(msg.text)

    return _handler


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost")
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path(".runtime/ui-browser-smoke-v1.json"),
    )
    args = parser.parse_args()
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        summary = {
            "status": "BLOCKED_NOT_FAKED",
            "tool": "playwright",
            "reason": "Playwright is not installed in this workspace.",
            "base_url": args.base_url,
        }
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return 2

    routes = [
        "/api/v1/ui",
        "/api/v1/ui/library",
        "/api/v1/ui/search",
        "/api/v1/ui/research",
    ]
    failures: list[dict] = []
    results: list[dict] = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        try:
            for route in routes:
                page = browser.new_page()
                console_errors: list[str] = []
                page.on("console", _record_console_error(console_errors))
                url = args.base_url.rstrip("/") + route
                response = page.goto(url, wait_until="networkidle")
                result = {
                    "route": route,
                    "url": url,
                    "status_code": response.status if response else None,
                    "console_errors": console_errors,
                }
                if not response or response.status != 200 or console_errors:
                    failures.append(result)
                results.append(result)
                page.close()
        finally:
            browser.close()
    summary = {
        "status": "PASSED" if not failures else "FAILED",
        "tool": "playwright",
        "base_url": args.base_url,
        "results": results,
        "failure_count": len(failures),
        "failures": failures,
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
