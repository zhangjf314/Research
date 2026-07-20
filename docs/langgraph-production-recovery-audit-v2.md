# LangGraph Production Recovery Audit v2

Status: `NOT_EXECUTED`

`/api/v1/capabilities` reports `langgraph_checkpoint=postgres`, so the Docker API
runtime is configured for PostgreSQL checkpoints. The successful final Deep
Research evidence, however, is the isolated smoke run
`live-q003-ed900ef2e202`, whose config records `checkpointer: sqlite smoke
checkpoint`.

The Stage 13.39 production recovery gate therefore remains open. The following
sequence still needs a dedicated authorized run:

1. Start a production Deep Research run.
2. Stop after a persisted intermediate node.
3. Force recreate the API container.
4. Resume the same thread/run id.
5. Verify completed nodes, retrieval calls, request ledgers, reservations, and
   side effects are not duplicated.
6. Complete with strict citation validation.

No successful Deep Research run was rerun for this audit.
