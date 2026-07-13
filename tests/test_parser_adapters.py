from pathlib import Path

import fitz

from paper_research.parsing.docling_parser import DoclingParser
from paper_research.parsing.grobid_parser import GrobidParser
from paper_research.parsing.page_assets import render_page_assets

TEI = """<?xml version="1.0"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader><fileDesc><titleStmt><title>Test Paper</title>
    <author>Ada Lovelace</author></titleStmt></fileDesc></teiHeader>
  <text><body><div><head>Introduction</head><p>Evidence paragraph.</p></div></body>
  <back><listBibl><biblStruct><analytic><title>Reference A</title></analytic></biblStruct>
  </listBibl></back></text>
</TEI>"""


def test_grobid_tei_is_normalized() -> None:
    parsed = GrobidParser("http://grobid.invalid")._from_tei(TEI)

    assert parsed.metadata.title == "Test Paper"
    assert parsed.metadata.authors == ["Ada Lovelace"]
    assert [block.block_type for block in parsed.blocks] == [
        "heading",
        "paragraph",
        "reference",
    ]
    assert parsed.blocks[1].section_path == ["Introduction"]


def test_page_assets_are_rendered(tmp_path: Path) -> None:
    source = tmp_path / "two-pages.pdf"
    document = fitz.open()
    document.new_page().insert_text((72, 72), "Page one")
    document.new_page().insert_text((72, 72), "Page two")
    document.save(source)
    document.close()

    assets = render_page_assets(source, tmp_path / "pages", dpi=96)

    assert [path.name for path in assets] == ["page-0001.png", "page-0002.png"]
    assert all(path.read_bytes().startswith(b"\x89PNG") for path in assets)


def test_docling_reports_unavailable_without_optional_extra(tmp_path: Path) -> None:
    parser = DoclingParser()
    if not parser.supports(tmp_path / "paper.pdf"):
        assert parser.supports(tmp_path / "paper.txt") is False
