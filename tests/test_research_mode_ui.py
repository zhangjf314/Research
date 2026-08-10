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


def test_dashboard_exposes_dual_research_mode_entries() -> None:
    html = TestClient(create_app()).get("/api/v1/ui").text

    assert "Research execution modes" in html
    assert "Deep Research Workflow" in html
    assert "Research Agent" in html
    assert "/api/v1/ui/research?mode=workflow" in html
    assert "/api/v1/ui/research?mode=agent" in html


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


def test_research_ui_gates_report_controls_on_real_report_body() -> None:
    html = _research_html()

    assert "id='report-title'>Research Output" in html
    assert "id='report-actions' hidden" in html
    assert "setReportControls(hasReportBody, title)" in html
    assert "hasReportBody: typeof data.report === 'string' && data.report.trim().length > 0" in html
    assert "setReportControls(true, 'Research Report')" in html
    assert "setReportControls(false, `${adapter.label} Failure Details`)" in html


def test_research_ui_agent_completed_without_report_is_not_labeled_report() -> None:
    html = _research_html()

    assert "Research Agent Execution Result" in html
    assert "does not include a final narrative research report" in html
    assert "if (normalized.hasReportBody)" in html
    assert "setReportControls(false, 'Research Agent Execution Result')" in html


def test_research_ui_uses_submitted_mode_as_single_source_of_truth() -> None:
    html = _research_html()

    assert "let taskExecutionMode = researchMode;" in html
    assert "taskExecutionMode = researchMode;" in html
    assert "const adapter = modeAdapters[taskExecutionMode];" in html
    assert "execution_mode: taskExecutionMode" in html
    assert "assertModeConsistency(researchMode, adapter)" in html


def test_research_ui_workflow_stage_history_is_visited_not_success() -> None:
    html = _research_html()

    assert "Visited stages are not success indicators." in html
    assert "Terminal status:" in html
    assert (
        "const visitedStages = (data.node_history || []).map((item, idx) => "
        "`- ${idx + 1}. ${item}`)"
    ) in html


def test_research_ui_agent_trace_distinguishes_decision_events_from_tools() -> None:
    html = _research_html()

    assert "Decision event: ${lastTool.phase}" in html
    assert "Decision event: ${phase}" in html
    assert (
        "const selectedTool = lastTool.tool || lastTool.tool_name || "
        "lastTool.action || ''"
    ) in html


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
