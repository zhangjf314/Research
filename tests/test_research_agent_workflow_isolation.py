from fastapi.testclient import TestClient

from paper_research.main import create_app


def test_agent_api_is_parallel_to_existing_deep_research_api() -> None:
    client = TestClient(create_app())
    routes = set(client.get("/openapi.json").json()["paths"])
    assert "/api/v1/research/deep" in routes
    assert "/api/v1/research/deep/{task_id}/resume" in routes
    assert "/api/v1/research/agent" in routes
    assert "/api/v1/research/agent/{task_id}/resume" in routes


def test_research_agent_schema_uses_explicit_agent_mode() -> None:
    client = TestClient(create_app())
    schema = client.get("/openapi.json").json()
    response_schema = schema["components"]["schemas"]["ResearchAgentResponse"]
    assert response_schema["properties"]["research_mode"]["default"] == "agent"


def test_research_agent_schema_adds_report_fields_without_removing_execution_fields() -> None:
    client = TestClient(create_app())
    schema = client.get("/openapi.json").json()
    properties = schema["components"]["schemas"]["ResearchAgentResponse"]["properties"]

    for existing in [
        "status",
        "verification_state",
        "tool_history",
        "provider_call_count",
        "token_usage",
    ]:
        assert existing in properties
    for added in [
        "report_status",
        "report_markdown",
        "report_usage",
        "report_provider_requests",
        "agent_execution_provider_requests",
        "agent_report_tokens",
        "total_agent_user_request_tokens",
    ]:
        assert added in properties
