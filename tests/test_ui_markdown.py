from __future__ import annotations

from fastapi.testclient import TestClient

from paper_research.api.markdown import render_markdown
from paper_research.main import create_app


def test_render_markdown_headings_and_lists() -> None:
    rendered = render_markdown("# Title\n\n- A\n- B")

    assert "<h1>Title</h1>" in rendered
    assert "<ul>" in rendered
    assert "<li>A</li>" in rendered
    assert "<li>B</li>" in rendered


def test_render_markdown_preserves_chinese() -> None:
    rendered = render_markdown("# 深度研究报告")

    assert "深度研究报告" in rendered


def test_render_markdown_strips_unsafe_raw_html() -> None:
    rendered = render_markdown("<script>alert(1)</script>\n<img src=x onerror=alert(1)>")

    assert "<script" not in rendered
    assert "</script" not in rendered
    assert "<img" not in rendered
    assert "onerror" not in rendered


def test_render_markdown_blocks_javascript_urls() -> None:
    rendered = render_markdown("[click](javascript:alert(1))")

    assert "javascript:" not in rendered.lower()
    assert "<a" not in rendered


def test_render_markdown_safe_links_are_hardened() -> None:
    rendered = render_markdown("[paper](https://example.org/paper)")

    assert 'href="https://example.org/paper"' in rendered
    assert 'target="_blank"' in rendered
    assert 'rel="nofollow noopener noreferrer"' in rendered


def test_render_markdown_tables_and_code_blocks_are_safe() -> None:
    rendered = render_markdown(
        "| A | B |\n|---|---|\n| 1 | 2 |\n\n"
        "```html\n<script>alert(1)</script>\n```"
    )

    assert "<table>" in rendered
    assert "<td>1</td>" in rendered
    assert "<pre><code" in rendered
    assert "script" in rendered
    assert "&lt;" in rendered or "&amp;lt;" in rendered
    assert "<script>" not in rendered


def test_render_markdown_rejects_non_string_input() -> None:
    try:
        render_markdown(None)  # type: ignore[arg-type]
    except TypeError as exc:
        assert "must be a string" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("non-string Markdown input was accepted")


def test_render_markdown_empty_string_is_empty_html() -> None:
    assert render_markdown("") == ""


def test_render_markdown_endpoint_length_limit() -> None:
    client = TestClient(create_app())
    response = client.post("/api/v1/ui/render-markdown", json={"markdown": "x" * 500_001})

    assert response.status_code == 422


def test_render_markdown_endpoint_returns_fragment() -> None:
    client = TestClient(create_app())
    response = client.post(
        "/api/v1/ui/render-markdown",
        json={"markdown": "# 深度研究报告\n\n- A\n\n<script>alert('xss')</script>"},
    )

    assert response.status_code == 200
    assert "<h1>深度研究报告</h1>" in response.text
    assert "<ul>" in response.text
    assert "<script" not in response.text


def test_research_page_uses_markdown_renderer_and_raw_controls() -> None:
    html = TestClient(create_app()).get("/api/v1/ui/research").text

    assert 'class="markdown-body"' in html or "class='markdown-body'" in html
    assert "id='report'" in html
    assert "id='report-raw'" in html
    assert "POST" in html
    assert "/api/v1/ui/render-markdown" in html
    assert "copyReport()" in html
    assert "toggleRaw()" in html
    assert "<pre id='report'>" not in html
    assert "textContent=d.report" not in html
    assert "data.report||JSON.stringify" not in html
