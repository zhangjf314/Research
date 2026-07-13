from abc import ABC, abstractmethod
from pathlib import Path

from paper_research.parsing.types import ParsedPaper


class PaperParser(ABC):
    @abstractmethod
    def supports(self, file_path: Path) -> bool:
        """Return whether this parser can handle the file."""

    @abstractmethod
    def parse(self, file_path: Path) -> ParsedPaper:
        """Parse a paper into the normalized representation."""
