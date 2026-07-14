from typing import Any

__all__ = ["Answer", "Citation", "QAService"]


def __getattr__(name: str) -> Any:
    """Avoid importing qa_service while providers.llm imports prompt helpers."""
    if name in __all__:
        from paper_research.generation.qa_service import Answer, Citation, QAService

        return {"Answer": Answer, "Citation": Citation, "QAService": QAService}[name]
    raise AttributeError(name)
