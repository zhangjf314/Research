import re

from paper_research.chunking.tokenizer import tokenize


def evaluate_answer(
    answer: str,
    contexts: list[str],
    citations: list[dict],
    expected_points: list[str] | None = None,
) -> dict[str, float]:
    expected_points = expected_points or []
    answer_terms = _terms(answer)
    context_terms = _terms(" ".join(contexts))
    expected_terms = _terms(" ".join(expected_points))
    sentences = [part.strip() for part in re.split(r"(?<=[.!?。！？])\s+", answer) if part.strip()]
    supported = [
        len(_terms(sentence) & context_terms) / max(1, len(_terms(sentence))) >= 0.4
        for sentence in sentences
    ]
    citation_marked = [sentence for sentence in sentences if re.search(r"\[[^\]]+\]", sentence)]
    valid_citations = [citation for citation in citations if citation.get("valid", True)]
    return {
        "answer_relevancy": round(_overlap(answer_terms, expected_terms), 6),
        "faithfulness": round(sum(supported) / len(supported), 6) if supported else 0.0,
        "context_precision": round(_overlap(answer_terms, context_terms), 6),
        "context_recall": round(_overlap(expected_terms, context_terms), 6),
        "citation_coverage": round(len(citation_marked) / len(sentences), 6) if sentences else 0.0,
        "citation_correctness": round(len(valid_citations) / len(citations), 6)
        if citations
        else 0.0,
        "unsupported_claim_rate": round(1 - sum(supported) / len(supported), 6)
        if supported
        else 1.0,
    }


def _terms(text: str) -> set[str]:
    return {token.lower() for token in tokenize(text) if token.isalnum()}


def _overlap(first: set[str], second: set[str]) -> float:
    return len(first & second) / max(1, len(first))
