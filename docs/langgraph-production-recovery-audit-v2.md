# LangGraph Production Recovery Audit v2

- Gate: `PASSED`
- Run ID / thread ID: `pg-recovery-q003-8a8bf07e6a4c`
- Provider/model: `deepseek/deepseek-v4-flash`
- Checkpointer: `postgresql`
- Pause node: `synthesize`
- Interrupted status: `interrupted`
- Container recreated: `True`
- Resume completed: `True`
- Duplicate completed nodes: `0`
- Duplicate provider requests: `0`
- Duplicate usage settlements: `0`
- Active reserved tokens: `0`
- Citation validation: `passed`

## Scope note

The API LangGraph route already uses PostgresSaver but is deterministic and does not call LLM; this release gate uses the existing bounded Deep Research smoke runner with a PostgreSQL checkpoint adapter to verify real-provider stop/resume accounting.
