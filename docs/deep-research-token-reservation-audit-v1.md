# Deep Research Token Reservation Audit v1

Historical failure `live-q003-cbc99df5b041` ended with `reserved_total_tokens=8081` and `budget_accounting_status=indeterminate_conservative_reserved`. That was an engineering accounting defect: terminal provider failure should not leave active reservations.

Stage 13.38 changes:

- `TOKEN_BUDGET_RESERVED` is emitted after successful reservation.
- `TOKEN_BUDGET_SETTLED` is emitted only after provider-reported usage is committed.
- `TOKEN_BUDGET_RELEASED` is emitted when provider failure or missing usage prevents settlement.
- Provider failure now releases reserved tokens and leaves `active_reserved_tokens=0`.
- Missing provider usage also releases reserved tokens and leaves actual token/cost totals at zero.
- Cost remains based only on settled usage, never on released reservation.

The historical run is not rewritten; this audit explains the defect and fixes future terminal states.
