from __future__ import annotations

# ruff: noqa: E402,I001
import argparse
import json
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient

from paper_research.main import app


DEFAULT_ROUTES = [
    "/api/v1/ui",
    "/api/v1/ui/library",
    "/api/v1/ui/search",
    "/api/v1/ui/research",
    "/api/v1/ui/evaluation",
    "/api/v1/ui/gold-review",
]


class ScriptExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.in_script = False
        self.scripts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "script":
            self.in_script = True
            self.scripts.append("")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "script":
            self.in_script = False

    def handle_data(self, data: str) -> None:
        if self.in_script and self.scripts:
            self.scripts[-1] += data


def extract_scripts(html: str) -> list[str]:
    parser = ScriptExtractor()
    parser.feed(html)
    return [script for script in parser.scripts if script.strip()]


def check_routes(routes: list[str], output_dir: Path) -> dict:
    client = TestClient(app)
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    failures: list[dict] = []
    for route in routes:
        response = client.get(route)
        route_result = {
            "route": route,
            "status_code": response.status_code,
            "script_count": 0,
            "scripts": [],
        }
        if response.status_code != 200:
            failure = {**route_result, "error": f"HTTP {response.status_code}"}
            failures.append(failure)
            results.append(failure)
            continue
        scripts = extract_scripts(response.text)
        route_result["script_count"] = len(scripts)
        page_name = route.rstrip("/").rsplit("/", 1)[-1] or "dashboard"
        for index, script in enumerate(scripts):
            script_path = output_dir / f"{page_name}-{index}.js"
            script_path.write_text(script, encoding="utf-8")
            completed = subprocess.run(
                ["node", "--check", str(script_path)],
                check=False,
                capture_output=True,
                text=True,
            )
            item = {
                "path": str(script_path),
                "returncode": completed.returncode,
                "stderr": completed.stderr.strip(),
            }
            route_result["scripts"].append(item)
            if completed.returncode != 0:
                failures.append({"route": route, **item})
        results.append(route_result)
    return {
        "tool": "node --check",
        "routes": routes,
        "results": results,
        "failure_count": len(failures),
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".runtime/ui-js-check"),
    )
    parser.add_argument("--route", action="append", dest="routes")
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    summary = check_routes(args.routes or DEFAULT_ROUTES, args.output_dir)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 1 if summary["failure_count"] else 0


if __name__ == "__main__":
    sys.exit(main())
