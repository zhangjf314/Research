import pytest
from pydantic import ValidationError

from paper_research.schemas.paper import PaperCreate


def test_paper_create_defaults() -> None:
    paper = PaperCreate(title="A paper")
    assert paper.authors == []
    assert paper.source_type == "upload"


def test_paper_title_must_not_be_empty() -> None:
    with pytest.raises(ValidationError):
        PaperCreate(title="")
