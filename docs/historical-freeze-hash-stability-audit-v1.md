# Historical freeze hash stability audit v1

Scope: post-merge verification for the local merge commit `87ac5aa91d9238563bb51bbb82b9d9a9cf4baaaf` against the release branch head `3920aa96c26218493d3e14f4d07aa95cd8cae4ac`.

## Finding

The failing historical freeze checks were caused by Windows line-ending materialization, not by semantic changes to Gold, retrieval data, QA results, or freeze manifests.

- Git attributes: `* text=auto eol=lf`.
- Git blob hashes for the checked files match `HEAD` and the release branch.
- Canonical JSON/JSONL hashes remain stable.
- Historical expected raw hashes match the CRLF text variant of the current LF working-tree files.

## Fix

Two small changes were made:

1. `scripts/import_stage13_10_claim_gold_review_v1.py` now validates source hashes by the portable canonical fields: `algorithm`, `mode`, `value`, `schema_version`, and `canonicalization_version`. `raw_value_at_review` drift is reported as audit metadata instead of being treated as a cross-platform semantic failure.
2. `tests/test_stage13_14_dev_v3_3.py` now uses `verify_legacy_raw_hash()` for legacy raw hashes. This still fails if the expected raw hash cannot be explained by an allowed text normalization variant.
3. Stage 13.16/17/18 historical freeze reconstruction now uses the same legacy CRLF text-hash convention for frozen text artifacts, while tests use `verify_legacy_raw_hash()` for text JSON evidence and retain raw exact checks for provider raw responses.

No expected hash was refreshed. No Gold file, retrieval protocol, freeze file, QA result, prompt, retrieval logic, or context selector was modified.

## Affected raw-hash evidence

| Path | Expected raw hash | Current raw hash | Explanation |
| --- | --- | --- | --- |
| `data/evaluation/stage13-12-dev-v3-2-failure-freeze-v1.json` | `26d8a661132d627b44ac035f503dbb16879f4724356d8e910737b39d088eae13` | `5484323b6b5d62354b220e41a4ef681bc45bbcb89b9c6af535b5ecc85bc75281` | expected hash matches CRLF variant |
| `data/evaluation/stage13-12-reservation-reconciliation-v1.json` | `53b1af45259f33ffac38f735dcb816749cb0359a3d03800ab2daa71384af30c1` | `b851fdfe2b0b06e422a3829db1fb6e5b985475731b6f9d5092328cd91cbc5a91` | expected hash matches CRLF variant |

## Gate

`semantic_integrity_gate=PASSED`

`pytest_historical_hash_failures_after_fix=0`

Historical protection was not weakened: real content changes still fail canonical hash, immutable-record hash, Git blob, and legacy raw-hash explainability checks.
