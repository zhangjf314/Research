"""Safe Markdown rendering for UI-only HTML fragments."""

from __future__ import annotations

import re
from collections.abc import MutableMapping
from typing import Any

import bleach
import mistune

_MARKDOWN = mistune.create_markdown(
    escape=True,
    plugins=[
        "table",
        "strikethrough",
        "task_lists",
    ],
)

_ALLOWED_TAGS = [
    "a",
    "blockquote",
    "br",
    "code",
    "del",
    "em",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "hr",
    "li",
    "ol",
    "p",
    "pre",
    "strong",
    "table",
    "tbody",
    "td",
    "th",
    "thead",
    "tr",
    "ul",
]

_ALLOWED_ATTRIBUTES = {
    "a": ["href", "title", "rel", "target"],
    "td": ["align"],
    "th": ["align"],
}

_ALLOWED_PROTOCOLS = {"http", "https", "mailto"}
_REL = "nofollow noopener noreferrer"
_ESCAPED_UNTRUSTED_HTML = re.compile(
    r"&lt;(?:script|iframe|object|embed|img)\b.*?(?:&lt;/(?:script|iframe|object|embed)&gt;|&gt;)",
    flags=re.IGNORECASE | re.DOTALL,
)


def _force_safe_link_attrs(
    attrs: MutableMapping[Any, str], new: bool = False
) -> MutableMapping[Any, str] | None:
    del new
    href_key = (None, "href")
    href = attrs.get(href_key, "")
    if not href or href == "#harmful-link":
        return None
    attrs[(None, "rel")] = _REL
    attrs[(None, "target")] = "_blank"
    return attrs


def render_markdown(markdown_text: str) -> str:
    """Render Markdown to a sanitized HTML fragment for UI display."""
    if not isinstance(markdown_text, str):
        raise TypeError("markdown_text must be a string")
    if not markdown_text:
        return ""

    raw_html = _MARKDOWN(markdown_text)
    raw_html = _ESCAPED_UNTRUSTED_HTML.sub("", raw_html)
    cleaned = bleach.clean(
        raw_html,
        tags=_ALLOWED_TAGS,
        attributes=_ALLOWED_ATTRIBUTES,
        protocols=_ALLOWED_PROTOCOLS,
        strip=True,
    )
    return bleach.linkify(
        cleaned,
        callbacks=[_force_safe_link_attrs],
        skip_tags={"pre", "code"},
    )
