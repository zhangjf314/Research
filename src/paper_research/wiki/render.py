"""Markdown is a byte-stable derived view; original strings remain intact."""

import re

from paper_research.evidence_graph.schema import canonical

from .schema import uri


def literal(text):
    runs = re.findall(r"`+", text)
    fence = "`" * max(3, max((len(r) + 1 for r in runs), default=3))
    return fence + "text\n" + text + "\n" + fence + "\n"


def render_markdown(page):
    label = {
        "PaperPage": "Source",
        "QuestionPage": "Original question",
        "RequirementPage": "Linked requirement",
        "ClaimPage": "Frozen claim",
    }[page.type]
    parts = [
        "# " + page.type,
        "",
        "Page ID: " + page.id,
        "",
        "Evidence status: FROZEN_MODEL_GOLD_NOT_HUMAN_TRUTH",
        "",
        "## " + label,
        "",
        literal(page.title),
        "## Original record",
        "",
        literal(canonical(page.source_record).decode()),
    ]
    for kind, links in page.links.items():
        parts += ["## " + kind, ""]
        parts += ["- [" + identity + "](" + uri(identity) + ")" for identity in links]
        parts.append("")
    parts += ["## Backlinks", ""]
    parts += ["- [" + identity + "](" + uri(identity) + ")" for identity in page.backlinks]
    for block in page.evidence:
        parts += [
            "",
            "## Grounded evidence",
            "",
            "Span: " + block["span_id"],
            "Chunk: " + block["chunk_id"],
            "Paper: " + block["paper_id"],
            "Pages: " + str(block["page_start"]) + "–" + str(block["page_end"]),
            "Source artifact: " + block["source_artifact"],
            "",
            literal(block["text"]),
        ]
    parts += ["", "## Provenance", "", literal(canonical(page.provenance.model_dump()).decode())]
    return "\n".join(parts) + "\n"


def render_root(entries):
    lines = ["# Agent-native Wiki Foundation v1", "", "FROZEN_MODEL_GOLD_NOT_HUMAN_TRUTH", ""]
    for kind in ("PaperPage", "QuestionPage", "RequirementPage", "ClaimPage"):
        lines += ["## " + kind, ""]
        lines += [
            "- [" + e["page_id"] + "](" + e["markdown_path"] + ")"
            for e in entries
            if e["page_type"] == kind
        ]
        lines.append("")
    return "\n".join(lines) + "\n"
