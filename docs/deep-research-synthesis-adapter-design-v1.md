# Deep Research Synthesis Adapter Design v1

Date: 2026-07-23

## Existing LLM layers audited

- OpenAI-compatible transport: `SiliconFlowLLMProvider` and
  `OpenAICompatibleLLMProvider` in `src/paper_research/providers/llm.py`.
- JSON response mode: chat-completions payload uses
  `response_format={"type": "json_object"}`.
- Timeout and retry: provider-level `timeout_seconds` / `max_retries` are reused.
- Provider error taxonomy: `LLMProviderError`, timeout/network classification, HTTP
  429/5xx retry handling, malformed response classification.
- Usage/cost: provider `_usage()` extracts prompt/completion/total tokens and uses
  configured per-million prices.
- API key redaction/audit: provider audit helpers redact sensitive text and never
  persist Authorization headers.
- Existing request/reservation utilities: `evaluation/request_accounting.py` and
  bounded smoke ledgers provide idempotent settlement patterns. This hotfix records
  synthesis usage in the Deep Research state and keeps active reservation at zero
  for the bounded smoke evidence.

## Design principles

- Transport is shared.
- Business protocols are isolated.
- Accounting remains unified.
- Research synthesis has its own schema and prompt.
- QA `generate_claim_answer()` is not reused for Deep Research synthesis.
- No second HTTP client, API-key reader, DeepSeek URL builder, or pricing calculator
  is introduced in the agent layer.
- Evidence dump is not a successful production fallback.

## Adapter architecture

```text
DeepResearchGraph
  -> HybridLocalResearchProvider
  -> DeepSeekResearchSynthesisProvider
       -> shared Structured JSON transport on OpenAICompatibleLLMProvider
       -> ResearchSynthesis Pydantic validation
       -> citation allowlist validation
  -> deterministic Markdown renderer
  -> report quality gate
```

## Failure semantics

- Missing production retrieval: `FAILED_RETRIEVAL`
- Missing production synthesis configuration: `FAILED_PROVIDER_CONFIGURATION`
- Provider transport failure: `FAILED_PROVIDER` / provider-specific error code
- Schema or citation allowlist failure after bounded repair: `FAILED_PROVIDER_SCHEMA`
- Report duplicate/citation quality failure: `FAILED_REPORT_QUALITY`

None of these states may return the old artifact-local evidence dump as
`COMPLETED`.
