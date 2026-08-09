# Stage 4B Attempt 3 Root Cause

- root_cause: `The API container used by Attempt 3 was built before the Stage 4B.3 failure-materialization fix; the loaded deployed research route did not contain the AgentDecisionProviderError materialization branch.`
- category: `STALE_DEPLOYED_API_RUNTIME`
- behavior_change_required: `False`
- deployment_fix_applied: `True`

Controlled replay did not prove deployed source parity. Attempt 3 used a stale API image whose loaded `research.py` lacked the failure materialization branch.
