# Dev v2 Claim Coverage Counterfactual

- Historical automatic coverage: **14/27 = 0.518519**
- Human-adjudicated diagnostic coverage: **16/27 = 0.592593**
- Historical metric modified: **False**

Prompt v2 does not enumerate required claims, permits silent omission and merged claims, and relies on a matcher with both false negatives and false positives. q050 remains a separate malformed-JSON engineering failure. Adjacent completion cannot simply be removed because the offline replay loses Gold evidence. Required-claim slots remain the selected general repair direction.
