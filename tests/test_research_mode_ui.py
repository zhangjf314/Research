from __future__ import annotations

from fastapi.testclient import TestClient

from paper_research.main import create_app


def _research_html() -> str:
    return TestClient(create_app()).get("/api/v1/ui/research").text


def test_research_ui_exposes_two_explicit_modes() -> None:
    html = _research_html()

    assert "Deep Research Workflow" in html
    assert "Research Agent" in html
    assert "Predefined research orchestration" in html
    assert "State/observation-driven research execution" in html
    assert "Frozen Current Hybrid" in html
    assert "Use Workflow" in html
    assert "Use Agent" in html
    assert "Current mode:" in html
    assert "[ WORKFLOW ]" in html
    assert "default" not in html.lower() or "workflow" in html.lower()


def test_research_ui_routes_workflow_and_agent_to_separate_endpoints() -> None:
    html = _research_html()

    assert "fetch('/api/v1/research/deep'" in html
    assert "fetch('/api/v1/research/agent'" in html
    assert html.count("fetch('/api/v1/research/deep'") == 1
    assert html.count("fetch('/api/v1/research/agent'") == 1
    assert "allow_external_search:false" in html
    assert "researchMode" in html
    assert "selectResearchMode(\"workflow\")" in html
    assert "selectResearchMode(\"agent\")" in html


def test_research_ui_shows_mode_specific_status_and_result_fields() -> None:
    html = _research_html()

    assert "Deep Research Workflow status" in html
    assert "Research Agent status" in html
    assert "Mode</th><td id='status-mode'" in html
    assert "Provider requests" in html
    assert "Estimated cost" in html
    assert "Agent trace" in html
    assert "agent-plan-version" in html
    assert "agent-selected-tool" in html
    assert "agent-evidence-count" in html
    assert "agent-replan-count" in html
    assert "Execution Mode" in html


def test_research_ui_does_not_expose_private_reasoning_terms() -> None:
    html = _research_html().lower()

    assert "chain-of-thought" not in html
    assert "reasoning_content" not in html
    assert "system prompt" not in html


def test_research_ui_supports_mode_switching_and_running_guard() -> None:
    html = _research_html()

    assert "A task is running. Reset or wait for a terminal state before switching mode." in html
    assert "activeTaskRunning" in html
    assert "resetResearchState()" in html
    assert "Fill example" in html
    assert "Compare the main technical routes" in html
