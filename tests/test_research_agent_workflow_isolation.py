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
