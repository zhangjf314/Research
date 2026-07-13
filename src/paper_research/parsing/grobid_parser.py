from pathlib import Path
from xml.etree import ElementTree

import httpx

from paper_research.parsing.interface import PaperParser
from paper_research.parsing.types import BoundingBox, PaperBlock, PaperMetadata, ParsedPaper


class GrobidParser(PaperParser):
    def __init__(self, base_url: str, timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def supports(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == ".pdf"

    def parse(self, file_path: Path) -> ParsedPaper:
        with file_path.open("rb") as stream:
            response = httpx.post(
                f"{self.base_url}/api/processFulltextDocument",
                files={"input": (file_path.name, stream, "application/pdf")},
                data={"consolidateHeader": "1", "consolidateCitations": "0"},
                timeout=self.timeout,
            )
        response.raise_for_status()
        return self._from_tei(response.text)

    def _from_tei(self, tei: str) -> ParsedPaper:
        root = ElementTree.fromstring(tei)
        namespace = {"tei": "http://www.tei-c.org/ns/1.0"}
        title = self._text(root.find(".//tei:titleStmt/tei:title", namespace))
        authors = [
            self._text(author)
            for author in root.findall(".//tei:titleStmt/tei:author", namespace)
            if self._text(author)
        ]
        blocks: list[PaperBlock] = []
        for division in root.findall(".//tei:text/tei:body/tei:div", namespace):
            heading_text = self._text(division.find("tei:head", namespace))
            if heading_text:
                blocks.append(self._block(heading_text, "heading", len(blocks)))
            for paragraph in division.findall("tei:p", namespace):
                text = self._text(paragraph)
                if text:
                    block = self._block(text, "paragraph", len(blocks))
                    if heading_text:
                        block.section_path = [heading_text]
                        block.parent_block_id = blocks[-1].block_id if blocks else None
                    blocks.append(block)
        for reference in root.findall(".//tei:listBibl/tei:biblStruct", namespace):
            text = self._text(reference)
            if text:
                blocks.append(self._block(text, "reference", len(blocks)))
        for index, block in enumerate(blocks):
            block.previous_block_id = blocks[index - 1].block_id if index else None
            block.next_block_id = blocks[index + 1].block_id if index + 1 < len(blocks) else None
        return ParsedPaper(
            parser="grobid",
            metadata=PaperMetadata(title=title or None, authors=authors, page_count=0),
            blocks=blocks,
        )

    @staticmethod
    def _text(element: ElementTree.Element | None) -> str:
        return " ".join("".join(element.itertext()).split()) if element is not None else ""

    @staticmethod
    def _block(text: str, block_type: str, index: int) -> PaperBlock:
        return PaperBlock(
            block_id=f"b{index + 1:06d}",
            block_type=block_type,
            page_start=1,
            page_end=1,
            block_index=index,
            text=text,
            bbox=BoundingBox(x0=0, y0=0, x1=0, y1=0),
        )
