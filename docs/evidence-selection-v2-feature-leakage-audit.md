# Evidence Selection v2 Feature Leakage Audit

- Source: `src\paper_research\generation\evidence_selection_v2.py`
- Gold online leakage: `0`
- Human-label online leakage: `0`
- Fixed-ID special cases: `0`
- Gate: `PASSED`

The selector is allowed to be used in offline replay and tests. Production selection must not read Gold relations, human labels, fixed question IDs, fixed claim IDs, or failure taxonomy labels.
