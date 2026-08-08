"""Research Agent v1 runtime.

This package is intentionally parallel to the existing Deep Research workflow.
It wraps the frozen Stage 2 RAG backend and adds dynamic action selection,
evidence-state updates, verification, checkpointing, retry, budget and trace.
"""

from paper_research.agents.research_agent.backend_lock import (
    EXPECTED_STAGE2_FINAL_CONFIG_HASH,
    RAGBackendLockError,
    validate_rag_backend_lock,
)
from paper_research.agents.research_agent.runner import ResearchAgentRunner
from paper_research.agents.research_agent.state import AgentBudget, AgentState

__all__ = [
    "EXPECTED_STAGE2_FINAL_CONFIG_HASH",
    "AgentBudget",
    "AgentState",
    "RAGBackendLockError",
    "ResearchAgentRunner",
    "validate_rag_backend_lock",
]
