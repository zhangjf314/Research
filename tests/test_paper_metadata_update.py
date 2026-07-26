from paper_research.schemas.paper import PaperMetadataUpdate


def test_metadata_update_schema_allows_year_and_rejects_empty_title() -> None:
    update = PaperMetadataUpdate(title="Paper", year=2024)
    assert update.year == 2024
    assert update.title == "Paper"


def test_metadata_update_schema_does_not_include_vector_fields() -> None:
    fields = set(PaperMetadataUpdate.model_fields)
    assert "title" in fields
    assert "year" in fields
    assert "embedding" not in fields
    assert "chunks" not in fields
