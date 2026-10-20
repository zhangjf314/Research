# Usage guide

Use the [README](../../README.md) installation/startup commands with your own local configuration and authorized PDFs. The default hash/template setup avoids paid model inference. A fresh database needs the existing SQLAlchemy schema initialized; no new migration system is introduced.

## Source admission

The upload endpoint accepts intake but does not grant trust. `paper_research.admission status` reads current state. `admit --approve --authorization-ref ... <pdf...>` is the trusted local operator's closed-batch admission: it validates bytes, records an admission, builds and seals a generation and publishes ACTIVE. Use a new DATA_DIR/collection for a new corpus. Duplicate or existing admission identities fail closed. Avoid direct mutation of published indexes.

The API, CLI and source files must resolve the same DATA_DIR and database/collection. In Docker, commands run inside the API container; an input must exist in its data volume. Local absolute paths are not portable between the host and container. Do not publish the generated data volume.

## Agent and model use

`POST /api/v1/research/agent` takes query, optional paper_ids, budget, task_id, pause_after_phase, generate_draft and multimodal_enabled. Default offline generation is disabled. Inspect task IDs and checkpoints before resuming; do not launch duplicate paid tasks. Multimodal and the read-only Evidence Contract opt-in cannot be combined in one request.

Actual visual generation requires a compatible Provider and explicit `LLM_VISION_ENABLED`, `RESEARCH_DRAFT_ENABLED` and `LIVE_MODEL_CALLS_ENABLED`, plus a local `RESEARCH_DRAFT_TOKEN` sent as `X-Research-Draft-Token`. Keys/tokens remain in private local environment variables. Failure to decode media or bind sources must not silently become claimed visual success. Provider compatibility is checked by the implementation; do not assume every OpenAI-compatible model is visual.

Model input/output limits and prices must be assessed for the actual request. Paid operation is not authorized by installing this repository or enabling a flag. The formal one-attempt experiment controllers had additional exclusive-attempt/budget rules; the normal API does not impose a universal dollar cap. Its current output clamp is 4096 and may truncate lengthy JSON. Preserve failed responses and do not automatically retry as if the first run succeeded.

## Optional native/CLIP assets

The runtime imports a small existing Phase4 dependency closure under `artifacts/paperresearch-v2/`; only its code and model-input contract are distributed. No benchmark, model weight, SQLite/Qdrant index or paper is included. Native extraction is pinned to PyMuPDF 1.28.0. Optional dependencies are in `.[multimodal]`.

`RESEARCH_NATIVE_BUNDLE` identifies your validated bundle. `RESEARCH_NATIVE_FULL_INDEX` selects the full N1/M1 backend and requires a compatible full index and local pinned model files. The backend expects locally provisioned embedding, reranker and visual assets below the Phase4 implementation's `models/` directories; use the repository/revision identifiers in its distributed model manifest and obey model licenses. There is no automatic download on request.

The historical full-index build path also expects a compatible previously frozen source-index manifest/vector store; this public release does not package those inputs or claim that a fresh clone can rebuild the original experiments by one command. Do not fabricate missing source indexes or bypass identity guards. Use a compatible authorized local environment for full N1/M1, or keep the base text path. Source-only availability is distinct from turnkey historical asset reproduction.

The public model manifest omits cached machine paths/observations. Its SHA therefore differs from the historical manifest: old indexes bound to that original SHA must not be silently relabeled to match. Historical artifacts remain in the original local project; independently prepared assets need their own truthful identity.

## Report statuses

- **UNVERIFIED**: generated or deterministically derived draft with unresolved semantic/source audit obligations.
- **SOURCE_AUDITED_RELEASE**: explicitly AI-origin source auditing and report delivery, not strict human review.
- **AI_REVIEWED_USER_ACCEPTED**: user accepts AI review; not a claim that the user personally checked every original source.
- Strict **HUMAN_REVIEWED/TRUSTED** publication: requires actual complete HumanApproval and every existing `publish_reviewed` gate. No automatic promotion from the above statuses.

Review the full report, not just its claim array. Bind numeric results to models, columns and conditions; check complete formulas, image pixels/ROI, negation and inference scope. Exact matching citations are not semantic proof. Do not fill human identity/approval fields from model suggestions.

## Portability and tests

Reports should use relative image/PDF links and retain SHA/page/unit/bbox in separate receipts. Copy report plus assets to a fresh directory and verify links. Local task and source records can contain private paths; do not publish those records indiscriminately.

The [README test command](../../README.md#tests) runs generic offline contracts. Large historical replay suites and raw evidence are intentionally absent. A fixture-missing skip does not imply science PASS. This release did not rebuild model indexes or restart a live deployment for validation.
