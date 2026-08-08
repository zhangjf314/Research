from __future__ import annotations

import json
from pathlib import Path

from paper_research.agents.research_agent.state import AgentState


class JsonResearchAgentCheckpointStore:
    def __init__(self, root: Path = Path(".runtime/research-agent/checkpoints")) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, state: AgentState, phase: str) -> str:
        checkpoint_id = f"{state.task_id}-{len(state.checkpoint_chain) + 1:04d}-{phase}"
        state.checkpoint_id = checkpoint_id
        state.checkpoint_chain.append(checkpoint_id)
        path = self.path_for(state.task_id)
        path.write_text(state.model_dump_json(indent=2), encoding="utf-8")
        return checkpoint_id

    def load(self, task_id: str) -> AgentState:
        path = self.path_for(task_id)
        if not path.exists():
            raise KeyError(f"checkpoint not found: {task_id}")
        return AgentState.model_validate(json.loads(path.read_text(encoding="utf-8")))

    def path_for(self, task_id: str) -> Path:
        return self.root / f"{task_id}.json"

