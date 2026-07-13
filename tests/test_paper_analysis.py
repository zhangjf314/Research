from paper_research.analysis.paper_analyzer import PaperAnalyzer
from paper_research.parsing.types import BoundingBox, PaperBlock, PaperMetadata, ParsedPaper


def paragraph(block_id: str, section: str, text: str, page: int) -> PaperBlock:
    return PaperBlock(
        block_id=block_id,
        block_type="paragraph",
        section_path=[section],
        page_start=page,
        page_end=page,
        block_index=int(block_id[1:]),
        text=text,
        bbox=BoundingBox(x0=1, y0=1, x1=100, y1=20),
    )


def test_analysis_fields_are_bound_to_page_evidence() -> None:
    parsed = ParsedPaper(
        parser="fixture",
        metadata=PaperMetadata(title="Evidence Paper", page_count=4),
        blocks=[
            paragraph(
                "b1",
                "Introduction",
                "The central problem is inefficient adaptation. We propose a low rank method.",
                1,
            ),
            paragraph(
                "b2", "Method", "Our method freezes weights and trains low rank matrices.", 2
            ),
            paragraph(
                "b3",
                "Experiments",
                "We train on the Example dataset with Adam and report accuracy against baselines.",
                3,
            ),
            paragraph(
                "b4",
                "Conclusion and Limitations",
                "However, the method is limited to tested language tasks. Future work remains.",
                4,
            ),
        ],
    )

    analysis = PaperAnalyzer().analyze("paper-1", parsed)

    assert analysis.research_problem.evidence[0].page_start == 1
    assert analysis.main_contributions.evidence[0].block_id == "b1"
    assert analysis.method_summary.evidence[0].page_start == 2
    assert analysis.experiment_card.datasets.evidence[0].page_start == 3
    assert analysis.limitations.evidence[0].page_start == 4
    assert analysis.future_work.evidence[0].quote.startswith("Future work")


def test_missing_field_does_not_invent_content() -> None:
    parsed = ParsedPaper(
        parser="fixture",
        metadata=PaperMetadata(page_count=1),
        blocks=[paragraph("b1", "Body", "A neutral sentence with no claims.", 1)],
    )

    analysis = PaperAnalyzer().analyze("paper-1", parsed)

    assert analysis.main_contributions.value is None
    assert analysis.main_contributions.evidence == []
    assert analysis.main_contributions.confidence == 0
