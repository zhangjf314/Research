import json
from pathlib import Path

import fitz

from paper_research.ingestion.artifacts import write_parse_artifacts
from paper_research.parsing.pymupdf_parser import PyMuPDFParser


def make_pdf(path: Path) -> None:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "A Structured Research Paper", fontsize=20)
    page.insert_text((72, 120), "1 Introduction", fontsize=15)
    page.insert_text(
        (72, 155),
        "This paper presents a deterministic fixture for parser verification.",
        fontsize=11,
    )
    document.set_metadata({"title": "Fixture Paper", "author": "Ada; Turing"})
    document.save(path)
    document.close()


def test_pymupdf_parser_preserves_structure_and_location(tmp_path: Path) -> None:
    source = tmp_path / "paper.pdf"
    make_pdf(source)

    parsed = PyMuPDFParser().parse(source)

    assert parsed.metadata.title == "Fixture Paper"
    assert parsed.metadata.authors == ["Ada", "Turing"]
    assert parsed.metadata.page_count == 1
    assert [block.block_type for block in parsed.blocks] == ["title", "heading", "paragraph"]
    paragraph = parsed.blocks[-1]
    assert paragraph.page_start == 1
    assert paragraph.section_path == ["1 Introduction"]
    assert paragraph.parent_block_id == parsed.blocks[1].block_id
    assert paragraph.bbox.x1 > paragraph.bbox.x0


def test_parse_artifacts_are_valid_json_and_jsonl(tmp_path: Path) -> None:
    source = tmp_path / "paper.pdf"
    make_pdf(source)
    parsed = PyMuPDFParser().parse(source)

    paths = write_parse_artifacts(parsed, tmp_path / "parsed")

    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    lines = paths["blocks"].read_text(encoding="utf-8").splitlines()
    assert metadata["page_count"] == 1
    assert len(lines) == 3
    assert json.loads(lines[0])["page_start"] == 1
    assert "Blocks: 3" in paths["report"].read_text(encoding="utf-8")


def test_blank_page_emits_ocr_warning(tmp_path: Path) -> None:
    source = tmp_path / "scan.pdf"
    document = fitz.open()
    document.new_page()
    document.save(source)
    document.close()

    parsed = PyMuPDFParser().parse(source)

    assert parsed.warnings[0].code == "LOW_TEXT_PAGE"
    assert parsed.warnings[0].page == 1
