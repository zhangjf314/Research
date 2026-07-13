import json
from pathlib import Path

from paper_research.analysis.paper_analyzer import PaperAnalyzer
from paper_research.analysis.types import PaperAnalysis
from paper_research.parsing.types import PaperBlock, PaperMetadata, ParsedPaper, ParseWarning


class AnalysisService:
    def __init__(self, analyzer: PaperAnalyzer | None = None) -> None:
        self.analyzer = analyzer or PaperAnalyzer()

    def analyze_artifacts(self, paper_id: str, parsed_dir: Path) -> PaperAnalysis:
        metadata = PaperMetadata.model_validate(
            json.loads((parsed_dir / "paper_metadata.json").read_text(encoding="utf-8"))
        )
        blocks = [
            PaperBlock.model_validate(json.loads(line))
            for line in (parsed_dir / "paper_blocks.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        parsed = ParsedPaper(
            parser="artifact",
            metadata=metadata,
            blocks=blocks,
            warnings=[ParseWarning(code="ARTIFACT_RELOAD", message="Loaded from parse artifacts")],
        )
        analysis = self.analyzer.analyze(paper_id, parsed)
        (parsed_dir / "paper_analysis.json").write_text(
            json.dumps(analysis.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return analysis
