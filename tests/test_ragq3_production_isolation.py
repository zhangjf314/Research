from pathlib import Path


def test_production_rag_route_does_not_import_gold_aware_evaluator() -> None:
    route = Path("src/paper_research/api/routes/rag.py").read_text(encoding="utf-8")
    assert "ragq3_attribution" not in route
    assert "ragq3_execution" not in route
