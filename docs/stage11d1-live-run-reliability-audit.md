# Stage 11D.1 Live Run Reliability Audit

## Evidence and chronology

- q003 `live-q003-798ac68288e0`: started `2026-07-14T13:42:38.300778+00:00`, completed, requests=1, tokens=9350.
- q049 `live-q049-1d47dc6a1ab8`: started `2026-07-14T13:45:03.515534+00:00`, completed first, requests=1, tokens=10026, citation=passed.
- q049 `live-q049-24a797337315`: started `2026-07-14T13:46:50.957854+00:00`, ran later, one provider attempt ended with ConnectError and unavailable usage.

The later failed q049 invocation wrote the top-level JSON, CSV, trace and Markdown because the old runner called `write_outputs` after every run. The old selection semantics were therefore implicit latest-attempt-wins, with no question/run isolation.

## Ledgers

The completed q049 ledger contains one retained request ID, input=8959, output=1067, total=10026, cost=0, and six unique events.

The failed q049 checkpoint records one attempted request, zero settled tokens, no usage record, three completed graph events, ConnectError and the old `usage_unavailable` budget label. Its request ID was not retained. Migration marks that field `not_retained`; it does not invent an identifier or retroactively alter SQLite.

A generic ConnectError cannot prove whether failure occurred before bytes were sent, while connecting, or before a response arrived. It is conservatively classified as `failed_after_send_unknown` with `unavailable_after_send_attempt`.

## Root cause and risk

The old synthesize node generated request_id only in a local variable immediately before the provider call and appended it only on successful usage settlement. Its exception path converted LLMProviderError into BudgetBlocked, producing the inaccurate budget_blocked status. Unknown usage was represented by absent usage plus numeric zero totals.

Without isolated run directories, a later failure could overwrite successful evidence. Separate CLI invocations also initialized global accounting independently, creating a cross-run budget visibility risk. Stage 11D.1 isolates attempts, preserves conservative reservations, and makes summary selection explicit and auditable.
