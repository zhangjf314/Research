from __future__ import annotations

import json
from typing import Any, Protocol

from paper_research.agents.research_agent.models import ResearchPlan, Subquestion, ToolAction
from paper_research.agents.research_agent.state import AgentState
from paper_research.providers.llm import StructuredJSONResult


class StructuredJSONProvider(Protocol):
    provider_name: str
    model_name: str

    def generate_structured_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema_name: str,
        request_context: dict[str, Any] | None = None,
        max_output_tokens: int | None = None,
    ) -> StructuredJSONResult:
        """Existing provider method used for strict JSON decisions."""


class AgentDecisionProviderError(RuntimeError):
    def __init__(self, message: str, *, usage_result: StructuredJSONResult | None = None) -> None:
        super().__init__(message)
        self.usage_result = usage_result


class LLMResearchAgentDecisionProvider:
    """Provider-backed Planner/Policy adapter for Stage 3 live validation.

    The adapter asks the configured LLM for bounded JSON decisions only. It does
    not expose gold labels, raw evidence text dumps, hidden reasoning, API keys,
    or additional tools.
    """

    schema_version = "research-agent-decision-v1"

    def __init__(self, provider: StructuredJSONProvider) -> None:
        self.provider = provider

    def plan(self, question: str, fallback_plan: ResearchPlan, task_id: str) -> tuple[
        ResearchPlan,
        StructuredJSONResult,
    ]:
        result = self.provider.generate_structured_json(
            system_prompt=(
                "You are a research planning controller. Return strict JSON only. "
                "Do not include hidden reasoning. Create at most 6 subquestions."
            ),
            user_prompt=json.dumps(
                {
                    "research_question": question,
                    "fallback_plan": fallback_plan.model_dump(),
                    "allowed_status": "OPEN",
                    "output_schema": {
                        "objective": "string",
                        "subquestions": [
                            {"id": "SQ1", "question": "string", "status": "OPEN"}
                        ],
                        "completion_criteria": ["string"],
                    },
                },
                ensure_ascii=False,
            ),
            schema_name=self.schema_version,
            request_context={"task_id": task_id, "agent_phase": "PLAN"},
            max_output_tokens=900,
        )
        payload = result.payload
        try:
            plan = ResearchPlan(
                objective=str(payload["objective"]),
                subquestions=[
                    Subquestion(
                        id=str(item["id"]),
                        question=str(item["question"]),
                        status="OPEN",
                    )
                    for item in payload["subquestions"][:6]
                ],
                completion_criteria=[
                    str(item) for item in payload.get("completion_criteria", [])[:6]
                ]
                or fallback_plan.completion_criteria,
            )
        except Exception as exc:
            raise AgentDecisionProviderError(
                f"invalid provider plan payload: {type(exc).__name__}",
                usage_result=result,
            ) from exc
        if not plan.subquestions:
            raise AgentDecisionProviderError(
                "provider plan contained no subquestions",
                usage_result=result,
            )
        return plan, result

    def decide(
        self,
        state: AgentState,
        deterministic_action: ToolAction,
    ) -> tuple[ToolAction, StructuredJSONResult]:
        allowed_actions = [
            "retrieve_evidence",
            "inspect_evidence",
            "inspect_paper",
            "verify_evidence",
            "finish",
        ]
        result = self.provider.generate_structured_json(
            system_prompt=(
                "You are a research agent policy controller. Return strict JSON only. "
                "Select exactly one allowed action from current state and observations. "
                "Decision reason must be short and audit-friendly, not chain-of-thought."
            ),
            user_prompt=json.dumps(
                {
                    "research_question": state.research_question,
                    "plan_version": state.plan_version,
                    "subquestions": [item.model_dump() for item in state.subquestions],
                    "resolved_subquestions": state.resolved_subquestions,
                    "unresolved_subquestions": state.unresolved_subquestions,
                    "evidence_count": len(state.evidence_state.items),
                    "last_observation": state.observations[-1].model_dump()
                    if state.observations
                    else None,
                    "verification_state": state.verification_state.model_dump()
                    if state.verification_state
                    else None,
                    "remaining_budget": {
                        "steps": state.remaining_step_budget,
                        "tools": state.remaining_tool_budget,
                        "tokens": state.remaining_token_budget,
                        "cost_usd": state.remaining_cost_budget,
                    },
                    "deterministic_recommendation": deterministic_action.model_dump(),
                    "allowed_actions": allowed_actions,
                    "output_schema": {
                        "action": (
                            "retrieve_evidence|inspect_evidence|inspect_paper|"
                            "verify_evidence|finish"
                        ),
                        "arguments": {},
                        "target_subquestion_ids": ["SQ1"],
                        "decision_reason": "short reason",
                    },
                },
                ensure_ascii=False,
            ),
            schema_name=self.schema_version,
            request_context={"task_id": state.task_id, "agent_phase": "DECIDE"},
            max_output_tokens=500,
        )
        payload = result.payload
        try:
            action = ToolAction(
                action=str(payload["action"]),
                arguments=dict(payload.get("arguments") or {}),
                target_subquestion_ids=[
                    str(item) for item in payload.get("target_subquestion_ids", [])
                ],
                decision_reason=str(payload["decision_reason"])[:240],
            )
        except Exception as exc:
            raise AgentDecisionProviderError(
                f"invalid provider action payload: {type(exc).__name__}",
                usage_result=result,
            ) from exc
        if action.action not in allowed_actions:
            raise AgentDecisionProviderError(
                f"provider selected unsupported action: {action.action}",
                usage_result=result,
            )
        if action.action == "retrieve_evidence" and not action.target_subquestion_ids:
            action.target_subquestion_ids = deterministic_action.target_subquestion_ids
        if action.action == "retrieve_evidence" and "query" not in action.arguments:
            action.arguments["query"] = deterministic_action.arguments.get(
                "query",
                state.research_question,
            )
        return action, result


def provider_usage_delta(result: StructuredJSONResult) -> dict[str, Any]:
    usage = result.usage
    return {
        "provider": result.provider,
        "model": result.model,
        "provider_request_id": result.provider_request_id,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
        "estimated_cost_usd": usage.estimated_cost_usd or 0.0,
        "usage_source": usage.usage_source,
        "request_attempt_count": result.request_attempt_count,
        "retry_count": result.retry_count,
        "latency_ms": result.total_latency_ms,
    }
