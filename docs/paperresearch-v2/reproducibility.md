# Reproducibility and publication boundary

The public release is a curated engineering snapshot, not a redistribution of the private experimental workspace. The existing public main is the parent. No private development commits are merged, rebased, rewritten or pushed; original working changes and historical artifacts remain local.

## Included

Core application source, existing Provider/Runner/tools/semantic-review/D6 implementation, selected automatic contracts, minimal existing Phase4 runtime dependencies, Docker/configuration files, MIT license and concise v2 documentation. Core source bytes match the source at packaging time except an extra blank line removed at the end of indexing/service.py; its AST remains identical. The file inventory binds released paths and SHA-256.

## Packaging-only adaptations

1. The Phase4 helper's ROOT is derived from `__file__` instead of the original workstation path. Retrieval math, ranking, source gates and budgets are unchanged.
2. The distributed frozen model manifest removes cached workstation paths and a workstation-specific missing-dependency wheel observation. Model repository/revision, input contracts and execution settings remain. This changes its file SHA and is **not** presented as the historical manifest or a way to reuse mismatched indexes.
3. Dockerfile no longer copies missing migrations or historical soak/evaluation artifacts. It copies only the distributed Phase4 runtime closure. Existing Docker architecture/services are retained; no new CI or deployment service is introduced.
4. `.gitignore`/`.dockerignore` protect local assets and explicitly allow the few runtime dependency files. Example flags default to no paid/visual generation; optional native dependencies are declared.
5. The generic assembly tests retain their assertions. Only the historical FNO replay skips when its excluded raw candidate files are absent. Synthetic fixtures are never represented as human gold.
6. indexing/service.py has one redundant end-of-file blank line removed so the release diff passes whitespace checks; no executable code changes.

Historical experiment contracts/manifests/code copies were not edited in the original workspace. These distribution adaptations live solely on the release branch. They do not improve or rerun scientific results.

## Not included

Original PDF/media copies, complete provider requests/responses, headers, credentials, human review records, raw traces, Checkpoints, corpus/gold/benchmark records, models, model caches, DB/Qdrant stores and generated indexes. Third-party papers/models must be acquired separately under their licenses. Selected public-safe historical summaries already in main are retained; the private bulk history is not ancestry of this release.

## What can be reproduced

Generic deterministic source/claim/coverage/Provider contract tests can run without research model calls. Base API/hash retrieval can be configured using one's own authorized data. Full native/visual runs require compatible local assets and separate cost authorization. The public source alone cannot reproduce every historical result because private inputs/assets are excluded. Remote model weights may change behind a service alias even with recorded parameters and fingerprint.

Cases, audit qualification failures and cost scope are summarized in [cases.md](cases.md) and [final_report.md](final_report.md). No new paper evaluation, training, prompt search or model call was performed for publication.
