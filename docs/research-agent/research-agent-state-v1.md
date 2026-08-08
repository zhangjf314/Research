# Research Agent State v1

State contains task identity, research question, current plan, subquestions,
resolved/unresolved subquestions, evidence state, observations, tool history,
candidate/verified/unsupported claims, contradictions, budgets, retry state,
verification state, status, stop reason and checkpoint metadata.

Evidence state is keyed by stable `paper_id:block_id:page` identifiers and
deduplicates repeated observations deterministically.
