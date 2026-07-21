# Public Demo Security

Before recording or publishing a demo:

- Use `.env.example` or a redacted `.env`; never show live API keys.
- Do not open raw provider response files or long trace JSON in the recording.
- Do not publish human review ZIPs, imported review packages, database dumps,
  Qdrant snapshots, or private PDF collections.
- If showing `/api/v1/capabilities`, only show key presence/fingerprint, never a
  full key.
- State that the Compose credentials are local development defaults and must not
  be exposed to an untrusted network.
- State that the evaluation set is a 50-record human-reviewed internal
  development set, not a blind public benchmark.
