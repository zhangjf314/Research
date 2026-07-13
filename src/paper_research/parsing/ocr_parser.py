import os
from pathlib import Path

import fitz

from paper_research.parsing.pymupdf_parser import PyMuPDFParser
from paper_research.parsing.types import (
    BoundingBox,
    PaperBlock,
    PaperMetadata,
    ParsedPaper,
    ParseWarning,
)


class OCRParser(PyMuPDFParser):
    def __init__(self, language: str = "eng", dpi: int = 300) -> None:
        self.language = language
        self.dpi = dpi

    def parse(self, file_path: Path) -> ParsedPaper:
        blocks: list[PaperBlock] = []
        warnings: list[ParseWarning] = []
        with fitz.open(file_path) as document:
            for page_index, page in enumerate(document):
                try:
                    text_page = page.get_textpage_ocr(
                        language=self.language, dpi=self.dpi, full=True
                    )
                except RuntimeError as exc:
                    tessdata = os.environ.get("TESSDATA_PREFIX", "not configured")
                    raise RuntimeError(
                        f"OCR unavailable; install Tesseract and set TESSDATA_PREFIX ({tessdata})"
                    ) from exc
                for raw in page.get_text("blocks", textpage=text_page, sort=True):
                    text = str(raw[4]).strip()
                    if not text:
                        continue
                    blocks.append(
                        PaperBlock(
                            block_id=f"b{len(blocks) + 1:06d}",
                            block_type="paragraph",
                            page_start=page_index + 1,
                            page_end=page_index + 1,
                            source_page=page_index + 1,
                            is_ocr=True,
                            ocr_confidence=None,
                            block_index=len(blocks),
                            text=text,
                            bbox=BoundingBox(x0=raw[0], y0=raw[1], x1=raw[2], y1=raw[3]),
                        )
                    )
            self._link_blocks(blocks)
            metadata = {str(key): str(value) for key, value in document.metadata.items() if value}
            return ParsedPaper(
                parser="ocr",
                parser_name="ocr",
                is_ocr=True,
                ocr_confidence=None,
                metadata=PaperMetadata(
                    title=metadata.get("title"),
                    page_count=document.page_count,
                    pdf_metadata=metadata,
                ),
                blocks=blocks,
                warnings=[
                    *warnings,
                    ParseWarning(
                        code="OCR_CONFIDENCE_UNAVAILABLE",
                        message="PyMuPDF OCR output does not expose Tesseract confidence values.",
                    ),
                ],
            )
