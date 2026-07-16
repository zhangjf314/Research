# Claim-level Gold Dev v1

Stage 13.10 Phase A created a human-only adjudication layer for the frozen 27
answerable Dev required claims. It does not modify question-level Gold or any historical metric.

- Schema: `claim-evidence-gold-dev-schema-v1`
- Gold version: `claim-evidence-gold-dev-v1`
- Required claims: 27
- Candidate relations: 313
- Candidate count distribution: {8: 1, 9: 1, 10: 2, 12: 23}
- Historical question-level Gold candidates retained: 99
- Pending adjudications: 27
- Candidate cap exceeded before truncation:
  22

Candidate provenance is diagnostic only. Retrieval hits, model citations, and prior human citation
support labels are never converted automatically into claim-level Gold. Multi-block claims may be
approved as a `minimum_complete_set` containing multiple relation IDs. Equivalent valid evidence
remains separate from historical exact Gold.

`READY_FOR_HUMAN_CLAIM_GOLD_ADJUDICATION=true`

`WAITING_FOR_EXTERNAL_CLAIM_GOLD_REVIEW`

`READY_FOR_DEV_V3_2=false`
