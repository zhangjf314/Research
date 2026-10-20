# Publication checks

Date: 2026-10-20 (Asia/Shanghai). This is a source publication check, not a new research experiment or science qualification.

## Observed checks

| Check | Result | Boundary |
|---|---|---|
| Curated offline contracts documented in README | **120 passed, 1 skipped**, 11.53 seconds | Skip: private frozen FNO candidate replay absent. One existing Starlette/httpx deprecation warning. No model inference. |
| Core imports and API schema | PASS | Eight core modules, existing native geometry/fusion modules, 37 OpenAPI paths; HTTP/socket calls explicitly denied during smoke. No live DB/service/model qualification. |
| Python syntax | PASS | Compiled distributed Python source in memory; no global bytecode/cache writes. |
| Docker Compose configuration | PASS | `docker compose config --quiet`. Docker image build, dependency installation and service startup were **not run**. |
| Git whitespace | PASS | One redundant blank line removed from the release copy of indexing/service.py; executable AST unchanged. |
| Relative README/new v2 documentation links | PASS | Checked inside the released tree, including referenced source paths. Historical documents may refer to intentionally excluded local experiments. |
| Source/asset selection | PASS | No private .env, model weights, PDFs, images, DB/index files or full provider payloads added; no released-tree file over 1 MB. |
| Credential pattern review | No findings | High-confidence provider/GitHub/AWS/private-key patterns plus literal credential assignment review; not a mathematical guarantee that every secret format is detectable. Local example passwords and synthetic fixtures are not real credentials. |

## Failures kept visible

The initial selected run had 102 passes, 1 skip, 3 failures and 12 setup errors. One failure was a missing test helper in the initial package; its existing helper and small interface schema were subsequently included. It passes in the final curated suite.

The old `test_runtime_index_identity.py` fixture attempts index mutation without the newer D6 capability; it produces twelve setup errors and one failed test. The old `test_research_agent_verifier.py` expects FAIL where the current verifier returns PARTIAL. The two representative failures were reproduced against the original workspace (2 failed), proving they are pre-existing contract drift, not introduced by publication. Their tests and production gates remain unchanged. They are not counted as passing or hidden with xfail.

The README's bounded suite uses current, relevant contract tests; it is not a statement that `pytest` across every historical test passes. Full historical regression, large matrices, new model runs, source-index builds and independent scientific acceptance were not performed for publication.

## Git and history scope

Confirmed remote: `https://github.com/zhangjf314/Research.git`. Public base: `7843ad41eaac9b2bf26cbbdf7e21aea2e84ce784`. Original development branch: `codex/rag-quality-v4-a`, HEAD `0e9f65c092299c29a9c503dc2aea6a2a79df3d72`; its index was initially empty and its local changes were preserved.

The new release branch starts on public main and adds curated publication commits, including a final Docker source-path packaging fix. It does **not** include the extra 150 private development commits or their 1120 unique blobs (358,526,359 bytes). No history rewrite, force push, main merge or new public repository/Release was performed.

The already-public base ancestry was inspected: 126 commits, 1859 blobs, 121,633,533 aggregate blob bytes, no detected credential pattern and no real .env path. It already contains four objects over 10 MB: two historical evaluation JSON/JSONL files and two OCR PDF fixtures (roughly 13.5–25.3 MB each). They are not in the current published tree and are not new objects exposed by this branch. No claim is made to have purged or relicensed existing public history. Removing existing public history would require a separate, explicit decision; this task did not rewrite it.

## Packaging and unresolved environment boundaries

The [reproducibility note](reproducibility.md) lists all packaging-only adaptations. Core scientific results and strict human/publication requirements are unchanged. A fresh clone needs its own authorized PDFs, source admission and compatible local assets for full native/visual operation. The original end-to-end scientific artifacts are not distributed as fixtures.

The GitHub CLI was not available in the publication environment; a Draft PR is optional and not required for this branch delivery. Actual push success must be checked against the remote branch SHA in the publication receipt, not inferred from this pre-push document.
