import json

from tests.research_agent_helpers import MockEvidenceProvider, evidence, runner


def test_trace_event_ids_are_unique_and_sanitized(tmp_path) -> None:
    agent = runner(tmp_path, MockEvidenceProvider([[evidence("a")], [evidence("b")]]))
    state = agent.run("alpha and beta")
    events = [
        json.loads(line)
        for line in agent.trace.path_for(state.task_id).read_text(encoding="utf-8").splitlines()
    ]
    ids = [event["event_id"] for event in events]
    assert len(ids) == len(set(ids))
    assert all("Authorization" not in json.dumps(event) for event in events)
    assert events[-1]["stop_reason"] is not None

