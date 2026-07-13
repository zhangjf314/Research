from pathlib import Path

import fitz

from paper_research.parsing.interface import PaperParser
from paper_research.parsing.types import (
    BoundingBox,
    PaperBlock,
    PaperMetadata,
    ParsedPaper,
    ParseWarning,
)


class PyMuPDFParser(PaperParser):
    def supports(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == ".pdf"

    def parse(self, file_path: Path) -> ParsedPaper:
        blocks: list[PaperBlock] = []
        warnings: list[ParseWarning] = []
        section_path: list[str] = []
        with fitz.open(file_path) as document:
            if document.needs_pass:
                raise ValueError("password-protected PDFs are not supported")
            body_size = self._estimate_body_size(document)
            for page_index, page in enumerate(document):
                page_blocks = page.get_text("dict", sort=True).get("blocks", [])
                page_text_count = 0
                for raw_block in page_blocks:
                    if raw_block.get("type") != 0:
                        continue
                    text, max_size = self._block_text(raw_block)
                    if not text:
                        continue
                    page_text_count += len(text)
                    block_type = self._classify(text, max_size, body_size, page_index, len(blocks))
                    if block_type == "heading":
                        section_path = [text]
                    block_id = f"b{len(blocks) + 1:06d}"
                    bbox = raw_block["bbox"]
                    blocks.append(
                        PaperBlock(
                            block_id=block_id,
                            block_type=block_type,
                            section_path=list(section_path),
                            page_start=page_index + 1,
                            page_end=page_index + 1,
                            source_page=page_index + 1,
                            block_index=len(blocks),
                            text=text,
                            bbox=BoundingBox(x0=bbox[0], y0=bbox[1], x1=bbox[2], y1=bbox[3]),
                        )
                    )
                if page_text_count < 20:
                    warnings.append(
                        ParseWarning(
                            code="LOW_TEXT_PAGE",
                            message="Page has little extractable text; OCR may be required.",
                            page=page_index + 1,
                        )
                    )
            self._link_blocks(blocks)
            metadata = {str(key): str(value) for key, value in document.metadata.items() if value}
            title = metadata.get("title") or next(
                (block.text for block in blocks if block.block_type == "title"), None
            )
            authors = [
                item.strip() for item in metadata.get("author", "").split(";") if item.strip()
            ]
            return ParsedPaper(
                parser="pymupdf",
                parser_name="pymupdf",
                is_ocr=False,
                metadata=PaperMetadata(
                    title=title,
                    authors=authors,
                    page_count=document.page_count,
                    pdf_metadata=metadata,
                ),
                blocks=blocks,
                warnings=warnings,
            )

    @staticmethod
    def _block_text(raw_block: dict) -> tuple[str, float]:
        lines: list[str] = []
        sizes: list[float] = []
        for line in raw_block.get("lines", []):
            spans = line.get("spans", [])
            line_text = "".join(span.get("text", "") for span in spans).strip()
            if line_text:
                lines.append(line_text)
                sizes.extend(float(span.get("size", 0)) for span in spans)
        return "\n".join(lines).strip(), max(sizes, default=0)

    @staticmethod
    def _estimate_body_size(document: fitz.Document) -> float:
        sizes: list[float] = []
        for page in document:
            for block in page.get_text("dict").get("blocks", []):
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        if span.get("text", "").strip():
                            sizes.append(round(float(span.get("size", 0)), 1))
        return max(set(sizes), key=sizes.count) if sizes else 10.0

    @staticmethod
    def _classify(
        text: str, max_size: float, body_size: float, page_index: int, block_count: int
    ) -> str:
        if page_index == 0 and block_count == 0 and max_size >= body_size * 1.25:
            return "title"
        short = len(text) <= 180 and text.count("\n") <= 2
        if short and max_size >= body_size * 1.15:
            return "heading"
        return "paragraph"

    @staticmethod
    def _link_blocks(blocks: list[PaperBlock]) -> None:
        current_heading: str | None = None
        for index, block in enumerate(blocks):
            block.previous_block_id = blocks[index - 1].block_id if index else None
            block.next_block_id = blocks[index + 1].block_id if index + 1 < len(blocks) else None
            if block.block_type == "heading":
                current_heading = block.block_id
            elif block.block_type == "paragraph":
                block.parent_block_id = current_heading
