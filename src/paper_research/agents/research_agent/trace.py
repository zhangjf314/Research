from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from paper_research.agents.research_agent.state import AgentState


class ResearchAgentTraceWriter:
    def __init__(self, root: Path = Path(".runtime/research-agent/traces")) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def append(
        self,
        state: AgentState,
        *,
        phase: str,
        latency_ms: float = 0.0,
        extra: dict[str, Any] | None = None,
    ) -> str:
        event_id = f"{state.task_id}:{len(state.trace_event_ids) + 1:04d}:{phase}"
        event = {
            "event_id": event_id,
            "task_id": state.task_id,
            "step_id": state.step_count,
            "timestamp": datetime.now(UTC).isoformat(),
            "phase": phase,
            "plan_version": state.plan_version,
            "action": state.last_action.action if state.last_action else None,
            "tool_name": state.last_action.action if state.last_action else None,
            "sanitized_tool_args": _sanitize_args(
                state.last_action.arguments if state.last_action else {}
            ),
            "target_subquestions": state.last_action.target_subquestion_ids
            if state.last_action
            else [],
            "observation_status": state.observations[-1].status if state.observations else None,
            "evidence_added_count": len(state.observations[-1].evidence_added)
            if state.observations
            else 0,
            "evidence_state_count": len(state.evidence_state.items),
            "verification_status": state.verification_state.status
            if state.verification_state
            else None,
            "remaining_budget": {
                "steps": state.remaining_step_budget,
                "tools": state.remaining_tool_budget,
                "tokens": state.remaining_token_budget,
                "cost_usd": state.remaining_cost_budget,
            },
            "latency_ms": latency_ms,
            "provider_usage": state.token_usage.model_dump(),
            "tool_usage": {"tool_call_count": state.tool_call_count},
            "checkpoint_id": state.checkpoint_id,
            "stop_reason": state.stop_reason,
        }
        if extra:
            event.update(extra)
        path = self.path_for(state.task_id)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        state.trace_event_ids.append(event_id)
        return event_id

    def path_for(self, task_id: str) -> Path:
        return self.root / f"{task_id}.jsonl"


def _sanitize_args(arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        key: ("<redacted>" if "key" in key.lower() or "authorization" in key.lower() else value)
        for key, value in arguments.items()
    }
