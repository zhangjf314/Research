# UI pages unavailable forensics v1

This hotfix investigated Library, Search, and Deep Research from Nginx, FastAPI,
runtime capability, JavaScript syntax, and rate-limit layers.

## Reproduction summary

The UI routes were not missing at the HTTP layer. Through Nginx/FastAPI, the
following routes returned HTTP 200:

- `/api/v1/ui`
- `/api/v1/ui/library`
- `/api/v1/ui/search`
- `/api/v1/ui/research`
- `/api/v1/health`
- `/api/v1/capabilities`
- `/api/v1/papers?limit=5`

The failures were therefore browser/runtime failures rather than Nginx 404 or
FastAPI route-registration failures.

## Root causes

### Library

Library had an executable JavaScript parse failure in the inline table renderer:
the authors fallback string was broken. Metadata enrichment had the same class
of broken fallback literal for old/new values. A route-level 200 could still
render a dead page in the browser.

### Search

Search was route-available, but it did not have deterministic HTTP/429 error
rendering. Failed API responses could leave the user with a poor or ambiguous
state instead of a visible retry message.

### Deep Research

Deep Research had two independent problems:

1. Docker Compose defaulted `DEEP_RESEARCH_ENABLED` to `false`, so a production
   container could show a real DeepSeek LLM provider while Deep Research itself
   remained disabled.
2. The inline Deep Research JavaScript emitted a Python-decoded newline inside a
   quoted JavaScript string and had a broken localized validation message.

### Shared infrastructure

The project did not have an executable JavaScript syntax gate. Existing route
availability checks could pass while browser JavaScript failed.

The Redis rate limiter also used a single bucket for almost every non-health
route and used `request.client.host` directly. Behind Nginx this collapses
clients to the proxy IP. UI page GETs also consumed business quota.

## Fixes

- Added `scripts/check_ui_javascript_syntax_v1.py`, which extracts FastAPI UI
  `<script>` blocks and validates them with `node --check`.
- Fixed Library and Deep Research JavaScript syntax.
- Added visible UI handling for 429 responses with retry timing.
- Exempted UI GET pages, health, capabilities, `/docs`, and `/openapi.json` from
  business API rate buckets.
- Split rate-limit buckets into `read_api`, `search`, `upload`,
  `metadata_enrichment`, and `deep_research`.
- Added `Retry-After`, `error_code=RATE_LIMITED`, and `request_id` to 429
  responses.
- Trusted `X-Forwarded-For` and `X-Real-IP` only when the direct client address
  is inside `TRUSTED_PROXY_CIDRS`.
- Added an explicit `deep_research` capability entry showing provider, model,
  response format, thinking mode, and template fallback status.
- Changed the Docker Compose default for `DEEP_RESEARCH_ENABLED` to `true`, while
  keeping provider and live-call configuration fail-closed if they are missing.

## Browser smoke status

Playwright is not installed in the current workspace, so real browser smoke is
recorded as `BLOCKED_NOT_FAKED`. Node syntax validation is executable and is now
part of the test suite.

## Remaining limitations

This hotfix restores route availability, executable UI JavaScript, safer
rate-limit behavior, and runtime capability visibility. It does not rerun Full
QA, Deep Research batches, or paid model evaluation.
