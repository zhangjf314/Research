import importlib.util
from pathlib import Path

from paper_research.parsing.interface import PaperParser
from paper_research.parsing.types import BoundingBox, PaperBlock, PaperMetadata, ParsedPaper


class DoclingParser(PaperParser):
    def supports(self, file_path: Path) -> bool:
        installed = importlib.util.find_spec("docling") is not None
        return file_path.suffix.lower() == ".pdf" and installed

    def parse(self, file_path: Path) -> ParsedPaper:
        if not self.supports(file_path):
            raise RuntimeError("Docling is not installed; install the 'parsing' extra")
        from docling.document_converter import DocumentConverter

        result = DocumentConverter().convert(str(file_path))
        document = result.document
        blocks: list[PaperBlock] = []
        for item in document.iterate_items():
            element = item[0] if isinstance(item, tuple) else item
            text = str(getattr(element, "text", "") or "").strip()
            if not text:
                continue
            label = str(getattr(element, "label", "paragraph")).lower()
            block_type = self._map_label(label)
            provenance = (getattr(element, "prov", None) or [None])[0]
            page = int(getattr(provenance, "page_no", 1) or 1)
            bbox_value = getattr(provenance, "bbox", None)
            bbox = BoundingBox(
                x0=float(getattr(bbox_value, "l", 0)),
                y0=float(getattr(bbox_value, "t", 0)),
                x1=float(getattr(bbox_value, "r", 0)),
                y1=float(getattr(bbox_value, "b", 0)),
            )
            blocks.append(
                PaperBlock(
                    block_id=f"b{len(blocks) + 1:06d}",
                    block_type=block_type,
                    page_start=page,
                    page_end=page,
                    block_index=len(blocks),
                    text=text,
                    bbox=bbox,
                )
            )
        self._link(blocks)
        title = next((block.text for block in blocks if block.block_type == "title"), None)
        page_count = max((block.page_end for block in blocks), default=0)
        return ParsedPaper(
            parser="docling",
            metadata=PaperMetadata(title=title, page_count=page_count),
            blocks=blocks,
        )

    @staticmethod
    def _map_label(label: str) -> str:
        if "title" in label:
            return "title"
        if "section" in label or "heading" in label:
            return "heading"
        if "table" in label:
            return "table"
        if "formula" in label:
            return "formula"
        if "reference" in label:
            return "reference"
        return "paragraph"

    @staticmethod
    def _link(blocks: list[PaperBlock]) -> None:
        heading: PaperBlock | None = None
        for index, block in enumerate(blocks):
            block.previous_block_id = blocks[index - 1].block_id if index else None
            block.next_block_id = blocks[index + 1].block_id if index + 1 < len(blocks) else None
            if block.block_type == "heading":
                heading = block
            elif heading is not None:
                block.parent_block_id = heading.block_id
                block.section_path = [heading.text]
