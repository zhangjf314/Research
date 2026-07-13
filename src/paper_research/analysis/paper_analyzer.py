import re

from paper_research.analysis.types import (
    Evidence,
    EvidenceBackedField,
    ExperimentCard,
    PaperAnalysis,
)
from paper_research.parsing.types import PaperBlock, ParsedPaper

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
URL_PATTERN = re.compile(r"https?://[^\s)]+")


class PaperAnalyzer:
    def analyze(self, paper_id: str, parsed: ParsedPaper) -> PaperAnalysis:
        blocks = [block for block in parsed.blocks if block.block_type == "paragraph"]
        introduction = self._sections(blocks, "abstract", "introduction", "background")
        method = self._sections(blocks, "method", "approach", "model", "architecture")
        experiments = self._sections(
            blocks, "experiment", "evaluation", "result", "implementation", "training"
        )
        conclusion = self._sections(blocks, "conclusion", "discussion", "limitation")
        return PaperAnalysis(
            paper_id=paper_id,
            title=parsed.metadata.title,
            research_background=self._field(introduction, limit=2),
            research_problem=self._keyword_field(
                introduction or blocks, ("problem", "challenge", "goal", "aim"), limit=2
            ),
            main_contributions=self._keyword_field(
                introduction or conclusion or blocks,
                ("contribution", "we propose", "we present", "we introduce", "our work"),
                limit=4,
            ),
            method_summary=self._field(method, limit=4),
            experiment_summary=self._field(experiments, limit=4),
            main_results=self._keyword_field(
                experiments or conclusion, ("result", "outperform", "improve", "achieve"), limit=4
            ),
            limitations=self._keyword_field(
                conclusion or blocks,
                ("limitation", "limited", "however", "future work", "cannot"),
                limit=3,
            ),
            future_work=self._keyword_field(
                conclusion or blocks, ("future work", "future research", "remain"), limit=2
            ),
            experiment_card=self._experiment_card(experiments or blocks),
        )

    @staticmethod
    def _sections(blocks: list[PaperBlock], *keywords: str) -> list[PaperBlock]:
        return [
            block
            for block in blocks
            if any(keyword in " > ".join(block.section_path).lower() for keyword in keywords)
        ]

    def _field(self, blocks: list[PaperBlock], limit: int) -> EvidenceBackedField:
        selected = blocks[:limit]
        sentences = [self._first_sentence(block.text) for block in selected]
        return self._make_field(selected, sentences)

    def _keyword_field(
        self, blocks: list[PaperBlock], keywords: tuple[str, ...], limit: int
    ) -> EvidenceBackedField:
        selections: list[tuple[PaperBlock, str]] = []
        for block in blocks:
            for sentence in SENTENCE_SPLIT.split(block.text.replace("\n", " ")):
                if any(keyword in sentence.lower() for keyword in keywords):
                    selections.append((block, sentence.strip()))
                    break
            if len(selections) >= limit:
                break
        return self._make_field(
            [block for block, _ in selections], [sentence for _, sentence in selections]
        )

    def _experiment_card(self, blocks: list[PaperBlock]) -> ExperimentCard:
        return ExperimentCard(
            datasets=self._keyword_field(blocks, ("dataset", "benchmark", "corpus"), 5),
            baselines=self._keyword_field(blocks, ("baseline", "compared with", "comparison"), 5),
            metrics=self._keyword_field(
                blocks, ("accuracy", "bleu", "f1", "perplexity", "metric"), 5
            ),
            hyperparameters=self._keyword_field(
                blocks, ("learning rate", "batch size", "epoch", "dropout"), 5
            ),
            hardware=self._keyword_field(blocks, ("gpu", "tpu", "hardware", "v100", "a100"), 3),
            software=self._keyword_field(blocks, ("pytorch", "tensorflow", "jax"), 3),
            training_details=self._keyword_field(blocks, ("train", "optimization", "optimizer"), 5),
            ablation_settings=self._keyword_field(blocks, ("ablation", "without", "remove"), 4),
            code_url=self._url_field(blocks, "code"),
            data_url=self._url_field(blocks, "data"),
        )

    def _url_field(self, blocks: list[PaperBlock], keyword: str) -> EvidenceBackedField:
        for block in blocks:
            if keyword not in block.text.lower():
                continue
            urls = URL_PATTERN.findall(block.text)
            if urls:
                return self._make_field([block], urls)
        return EvidenceBackedField()

    @staticmethod
    def _make_field(
        blocks: list[PaperBlock], values: list[str]
    ) -> EvidenceBackedField:
        cleaned = [value.strip() for value in values if value.strip()]
        evidence = [
            Evidence(
                block_id=block.block_id,
                section_path=block.section_path,
                page_start=block.page_start,
                page_end=block.page_end,
                quote=value[:500],
            )
            for block, value in zip(blocks, cleaned, strict=False)
        ]
        return EvidenceBackedField(
            value=cleaned if cleaned else None,
            evidence=evidence,
            confidence=min(0.9, 0.45 + 0.1 * len(evidence)) if evidence else 0.0,
        )

    @staticmethod
    def _first_sentence(text: str) -> str:
        return SENTENCE_SPLIT.split(text.replace("\n", " "))[0].strip()
