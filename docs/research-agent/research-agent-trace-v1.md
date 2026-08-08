# Research Agent Trace v1

Each step records task id, step id, timestamp, phase, plan version, action,
tool name, sanitized tool args, target subquestions, observation status,
evidence added count, evidence state count, verification status, remaining
budget, latency, provider usage, tool usage, checkpoint id and stop reason.

Trace must not include API keys, Authorization headers, hidden reasoning,
chain-of-thought or raw provider responses.
