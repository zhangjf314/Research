from pathlib import Path

from paper_research.config import Settings, get_settings
from paper_research.parsing.docling_parser import DoclingParser
from paper_research.parsing.grobid_parser import GrobidParser
from paper_research.parsing.interface import PaperParser
from paper_research.parsing.ocr_parser import OCRParser
from paper_research.parsing.pymupdf_parser import PyMuPDFParser
from paper_research.parsing.types import ParsedPaper


class ParserRouter:
    def __init__(
        self, parsers: list[PaperParser] | None = None, settings: Settings | None = None
    ) -> None:
        self.settings = settings or get_settings()
        self.parsers = parsers

    def parse(self, file_path: Path) -> ParsedPaper:
        if self.parsers is not None:
            parser = next((item for item in self.parsers if item.supports(file_path)), None)
            if parser is None:
                raise ValueError(f"no parser supports {file_path.suffix or 'this file'}")
            return parser.parse(file_path)

        backend = self.settings.parser_backend.lower()
        if backend == "grobid":
            if not self.settings.grobid_url:
                raise ValueError("GROBID_URL is required when PARSER_BACKEND=grobid")
            return GrobidParser(self.settings.grobid_url).parse(file_path)
        if backend == "docling":
            return DoclingParser().parse(file_path)
        if backend not in {"auto", "pymupdf", "ocr"}:
            raise ValueError(f"unknown parser backend: {backend}")
        if backend == "ocr":
            return OCRParser(self.settings.ocr_language).parse(file_path)

        if backend == "auto" and DoclingParser().supports(file_path):
            return DoclingParser().parse(file_path)
        parsed = PyMuPDFParser().parse(file_path)
        low_text_pages = {
            warning.page for warning in parsed.warnings if warning.code == "LOW_TEXT_PAGE"
        }
        if low_text_pages and len(low_text_pages) >= max(1, parsed.metadata.page_count // 2):
            return OCRParser(self.settings.ocr_language).parse(file_path)
        return parsed
