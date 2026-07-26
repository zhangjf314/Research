"""Offline replay for the post-smoke Deep Research report renderer hotfix."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from paper_research.agents.deep_research_graph import DeepResearchGraph
from paper_research.agents.research_synthesis_provider import ResearchSynthesis
from paper_research.api.markdown import render_markdown

PYTHON_DICT_LITERAL_RE = re.compile(r"\{['\"]text['\"]\s*:", flags=re.I)


class _EmptyLocalProvider:
    def search(
        self, query: str, paper_ids: list[str] | None = None, limit: int = 8
    ) -> list[dict[str, Any]]:
        del query, paper_ids, limit
        return []


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def replay(
    *,
    task_id: str,
    result_path: Path,
    raw_response_path: Path,
    output_json: Path,
    output_md: Path,
) -> dict[str, Any]:
    result = _load_json(result_path)
    raw_response = _load_json(raw_response_path)
    raw_content = raw_response.get("raw_content") or raw_response.get("content")
    if not isinstance(raw_content, str):
        raise ValueError("raw response does not contain string raw_content")

    synthesis = ResearchSynthesis.model_validate_json(raw_content)
    graph = DeepResearchGraph(local_provider=_EmptyLocalProvider())
    state: dict[str, Any] = {
        **result,
        "synthesis": synthesis.model_dump(),
        "node_history": result.get("node_history", []),
    }
    rendered = graph._report(state)  # noqa: SLF001 - deterministic replay of current renderer.
    validate_state = {**state, **rendered}
    validated = graph._validate(validate_state)  # noqa: SLF001 - deterministic replay.
    report = rendered["draft_report"]
    html = render_markdown(report)

    python_dict_literal_leakage = bool(PYTHON_DICT_LITERAL_RE.search(report))
    html_script_leakage = "<script" in html.lower() or "javascript:" in html.lower()
    citation_validation_passed = (
        validated["status"] == "COMPLETED"
        and all(item.get("valid") for item in validated.get("citation_results", []))
    )
    quality = rendered.get("report_quality") or {}
    summary = {
        "schema_version": "research-smoke-post-renderer-replay-v1",
        "task_id": task_id,
        "source_result_path": str(result_path.as_posix()),
        "source_raw_response_path": str(raw_response_path.as_posix()),
        "raw_provider_response_reused": True,
        "llm_called": False,
        "schema_validation": "PASSED",
        "current_renderer_replayed": True,
        "python_dict_literal_leakage": int(python_dict_literal_leakage),
        "duplicate_paragraph_count": quality.get("exact_duplicate_paragraph_count", 0),
        "duplicate_reference_count": quality.get("duplicate_reference_count"),
        "citation_validation": "PASSED" if citation_validation_passed else "FAILED",
        "report_quality_gate": "PASSED" if quality.get("passed") else "FAILED",
        "markdown_rendering": "PASSED" if html and not html_script_leakage else "FAILED",
        "script_leakage": html_script_leakage,
        "report_length": len(report),
        "report_sha256": _sha256_text(report),
        "html_length": len(html),
        "html_sha256": _sha256_text(html),
        "status": "PASSED"
        if (
            not python_dict_literal_leakage
            and citation_validation_passed
            and quality.get("passed")
            and html
            and not html_script_leakage
        )
        else "FAILED",
        "quality": quality,
        "citation_result_count": len(validated.get("citation_results", [])),
        "used_citation_count": sum(
            1 for item in validated.get("citation_results", []) if item.get("used")
        ),
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    md = [
        "# Research Smoke Post-Renderer Replay v1",
        "",
        f"- Task ID: `{task_id}`",
        f"- LLM called: `{summary['llm_called']}`",
        f"- Schema validation: `{summary['schema_validation']}`",
        f"- Python dict literal leakage: `{summary['python_dict_literal_leakage']}`",
        f"- Citation validation: `{summary['citation_validation']}`",
        f"- Report quality gate: `{summary['report_quality_gate']}`",
        f"- Markdown rendering: `{summary['markdown_rendering']}`",
        f"- Script leakage: `{str(summary['script_leakage']).lower()}`",
        f"- Report SHA-256: `{summary['report_sha256']}`",
        "",
        "This replay reused the saved raw provider response and did not call DeepSeek.",
    ]
    output_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    if summary["status"] != "PASSED":
        raise SystemExit(1)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task-id",
        default="3ae49190-2c9d-471f-8b4d-1b26544ccca3",
    )
    parser.add_argument(
        "--result-path",
        type=Path,
        default=Path(
            "artifacts/research-smoke-hotfix-v1/"
            "3ae49190-2c9d-471f-8b4d-1b26544ccca3.json"
        ),
    )
    parser.add_argument(
        "--raw-response-path",
        type=Path,
        default=Path(
            ".runtime/research-synthesis-provider/"
            "hotfix-deep-research-20260725220952/attempt-01.json"
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("data/evaluation/research-smoke-post-renderer-replay-v1.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("docs/research-smoke-post-renderer-replay-v1.md"),
    )
    args = parser.parse_args()
    replay(
        task_id=args.task_id,
        result_path=args.result_path,
        raw_response_path=args.raw_response_path,
        output_json=args.output_json,
        output_md=args.output_md,
    )


if __name__ == "__main__":
    main()
