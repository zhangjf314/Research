import json

from tests.research_agent_helpers import MockEvidenceProvider, evidence, runner


def test_agent_trace_does_not_include_gold_fields(tmp_path) -> None:
    agent = runner(tmp_path, MockEvidenceProvider([[evidence("a")], [evidence("b")]]))
    state = agent.run("alpha and beta")
    trace = agent.trace.path_for(state.task_id).read_text(encoding="utf-8")
    lowered = trace.lower()
    forbidden = ["gold_answer", "gold_block_ids", "gold required", "benchmark category"]
    assert not any(item in lowered for item in forbidden)
    assert "chain-of-thought" not in lowered
    assert all(
        "raw_provider_response" not in json.loads(line)
        for line in trace.splitlines()
    )

