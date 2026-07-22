# Repository homepage audit v1

## Starting point

- Repository: `zhangjf314/Research`
- Branch: `main`
- Release commit: `a8e7b89f29cd6127cf96e83ab314139235410be2`
- Tag: `v1.0.0-portfolio`
- Runtime version: `1.0.0+portfolio`
- Display version: `1.0.0-portfolio`

## Issues found in the old README

1. It still said merge, tag, push, and remote release were awaiting authorization.
2. It mixed final release status with old development-phase status.
3. It contained mutually conflicting version conclusions.
4. It placed long development logs on the recruiter-facing homepage.
5. It lacked a concise product value statement.
6. It lacked a compact final metrics table.
7. It lacked a clean architecture overview.
8. It did not provide a focused demo section.
9. It did not clearly separate internal development evaluation from blind benchmark claims.
10. The root directory contained a binary planning document that was not suitable as a homepage entry.

## Changes made

- Archived the previous README at `docs/history/README-pre-v1-homepage-refresh.md`.
- Rewrote `README.md` from scratch as the public repository homepage.
- Added a Mermaid architecture visual instead of fabricating UI screenshots.
- Added `docs/quickstart.md` and `docs/api-examples.md`.
- Added `docs/releases/v1.0.0-portfolio.md` for GitHub Release notes.
- Moved `PaperResearch Agent 项目规划.docx` to `docs/archive/` using Git tracking.
- Added automated README homepage tests.

## Verification

- README old lines: 304.
- README new lines: 230.
- README relative links: checked by `tests/test_readme_homepage.py`.
- README relative images: checked by `tests/test_readme_homepage.py`; no local image assets are referenced.
- Hero visual: Mermaid architecture diagram.
- Full release test: `605 passed, 1 warning`.
- Ruff: passed.
- compileall: passed.
- `git diff --check`: passed.
- `docker compose config --quiet`: passed.

## GitHub repository metadata

`gh` is not installed in the current execution environment, so GitHub About, Topics, and the public Release page could not be updated from this run.

Prepared release notes are available at `docs/releases/v1.0.0-portfolio.md`.

## Truth boundary retained

- `SEMANTIC_CLAIM_SUPPORT_AUDIT=NOT_FORMALLY_VALIDATED`
- `STRONG_GROUNDING_CLAIM_ALLOWED=false`
- `RETRIEVAL_GENERALIZATION_EVIDENCE=DIAGNOSTIC_ONLY`

No QA prompt, retrieval logic, context selector, Gold data, DeepSeek configuration, Reranker, formal evaluation result, Full QA run, Deep Research run, Docker service architecture, tag, or Git history was modified.
