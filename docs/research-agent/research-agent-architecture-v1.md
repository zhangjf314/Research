# Research Agent Architecture v1

LangGraph remains the existing workflow substrate for the control-group Deep
Research path. Research Agent v1 is introduced as a parallel runtime path whose
next actions are selected dynamically from current state and observations rather
than following a fixed research sequence.

```mermaid
flowchart TD
    START --> PLAN
    PLAN --> DECIDE
    DECIDE --> EXECUTE["EXECUTE TOOL"]
    EXECUTE --> OBSERVE
    OBSERVE --> UPDATE["UPDATE STATE"]
    UPDATE --> VERIFY
    VERIFY -->|PASS| FINISH
    VERIFY -->|FAIL/PARTIAL| REPLAN
    REPLAN --> DECIDE
    DECIDE -->|budget exhausted| STOP_BUDGET["STOP: budget exhausted"]
    DECIDE -->|no progress| STOP_PROGRESS["STOP: no progress"]
    EXECUTE -->|fatal provider/tool failure| STOP_FAILURE["STOP: provider/tool failure"]
```

The existing `CONTROL_GROUP_WORKFLOW` is not rewritten and remains the default.
Agent mode is exposed separately as `research_mode=agent`.
