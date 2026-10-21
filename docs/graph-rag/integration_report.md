# Graph/Wiki runtime integration report

Status: **PAPERRESEARCH_GRAPHRAG_RUNTIME_INTEGRATED** (bounded structural runtime).
This is an engineering result, not scientific multi-hop reasoning qualification.
The v2 COCO-LM/FNO/U-NO conclusions and strict publication rules are unchanged.

## Implementation and source audit

The previous graph builder loaded a frozen corpus plus Phase2B gold; its Wiki
constructor expected frozen-gold pages. Those are offline research interfaces,
not user-corpus production data sources. They remain unchanged. A small runtime
adapter now supplies only D6-authorized canonical Paper/Chunk records to the
existing graph builder, validator and GraphQuery. A structural Wiki subclass
derives source-backed PaperPages and reuses WikiQuery's query/context APIs.
The [dependency diagram](architecture.md) identifies the exact boundary.

GraphRAG follows the existing Hybrid search: bounded source containment expansion,
local BM25 ranking, RRF fusion, deduplication and the existing ContextBuilder.
It returns source text rather than node names. `/retrieve`, direct graph/Wiki
queries and the original Agent retrieve_evidence tool share the adapter. Defaults
remain Hybrid; explicit opt-in, versioned receipts and checkpoint persistence
make graph use/fallback visible. No retrieval model, semantic verifier, Planner,
HumanApproval or publish_reviewed algorithm was replaced.

Every query verifies the current D6/physical index, reconstructs sealed block-page
maps and re-derives the expected graph/Wiki. Generation, source text, topology,
quotes, PDF identity and Wiki provenance cannot be authorized by rehashing a
modified graph file. Existing canonical evidence schemas were not relaxed; graph
path/usage receipts are separate from scientific support and dense/sparse origin.

## Real local end-to-end result

This was not mock acceptance. The user's existing authorized COCO-LM and FNO PDFs
were freshly parsed/admitted into isolated PostgreSQL and a new real HTTP Qdrant
collection. Their SHA-256 identities are in [acceptance_summary.json](acceptance_summary.json).
Neither benchmark corpus nor gold data was used. An I/O audit hook rejected
access to the frozen research corpus/gold paths in both builder and API process.
The same engineering question asked for Fourier-layer construction linked to
Burgers/Darcy experiments. This reuses known papers for integration testing; it
is not a new independent scientific experiment.

| Check | Actual observation |
| --- | --- |
| D6 admission and runtime index | ACTIVE; 2 PDFs, 113 chunks, 115 nodes, 113 structural edges, 2 Wiki pages; index 748,648 bytes |
| Production API | Real Uvicorn/HTTP `/retrieve`, `/graph/query`, `/wiki/query` returned 200 and original canonical evidence |
| Graph relation lookup | Two-hop source expansion; 115 nodes/113 edges visited within the authorized corpus for fused retrieval |
| Direct graph/Wiki query | 16 source units / 4 Wiki evidence entries returned under the configured context caps |
| Existing Agent | HTTP task paused and resumed through persisted Checkpoints; 3 retrieve_evidence calls, 3 graph receipts, 11 canonical items, 17 checkpoints |
| Agent terminal/publication status | PARTIAL / UNRESOLVABLE_GAP; report suppressed with SEMANTIC_SUPPORT_NOT_ASSESSED; no trusted report, draft approval or paid generation |
| Citation recheck | 36 returned evidence occurrences from graph retrieval, direct graph, Wiki and Agent checked against exact canonical prefixes and original physical PDF-page block text |
| Cost and execution | 26.38 s total acceptance; 0 paid requests, 0 neural forwards, $0; local hash embedding and BM25 operations executed |

The text comparison used identical question, paper scope and top_k=5. Hybrid
took **938.71 ms** wall time, GraphRAG **768.09 ms**, of which **77.65 ms** was
recorded graph-stage time. Both returned the same five chunk IDs; the last two
swapped order. Added/lost evidence: **0 / 0**. Context: **5,646 characters**, cap
16,000. These sequential single local observations include cache/warm-up effects;
the lower second wall time is not evidence of a GraphRAG speed advantage.
Peak RAM/CPU were not instrumented. There is **no measured recall, scientific
answer-quality gain or proof of superiority over Hybrid**. A separate generic
fixture verifies that source expansion can add eligible units when budget/rank
permit, but that fixture is not counted as real-paper effectiveness evidence.

The raw local acceptance is retained under `.runtime/gw21-6/`, including actual
HTTP bodies, trace, checkpoints, parser output and graph index. Public summaries
contain identities, counts and hashes, not raw paper copies or API credentials.
The acceptance API process stopped automatically. The task's dedicated
PostgreSQL container is stopped at completion; existing application/Qdrant
services are left untouched. Isolated databases/collections remain for audit.
Local retained runtime data occupy approximately 139 MB (principally the isolated
PostgreSQL cluster and reproducible parser/index outputs); none are committed.
System policy rejected recursive cleanup of task-created test temporary folders,
so those folders are also retained without a deletion workaround.

## Negative source and budget tests

All twelve HTTP negative scenarios passed. Unauthorized papers returned 403;
stale generation returned 409; altered edges, orphan structures, sourceless Wiki,
foreign benchmark schema and added gold nodes were refused; node and hop budget
violations returned 422; canonical text tampering hit the existing D6 gate;
corrupt runtime JSON was refused. Missing index returned 503 with strict options
and returned an explicit FALLBACK_HYBRID only under dual server/request permission.
Mutations affected only the test-created index/canonical files and were restored
to their exact original bytes. The stale-generation control changed the runtime
binding; it did not modify an old scientific generation.

Additional unit controls cover scope separation, rehashed forgery rejection,
empty paths, edge/node caps, deduplication/fusion, context limits, two-sided
fallback, default API options, incompatible Agent modes, unsupported non-paper
filters and graph checkpoint restrictions. Unit fixtures remain labeled synthetic.

## Verification and preserved failures

- Bounded current-contract suite: **135 passed, 1 skipped**. It includes 12 new
  GraphRAG tests, C1 safety, product/source identity, goal assembly, semantic
  contracts, state/policy and gold-free runtime controls. The skip is the existing
  FNO historical replay without its excluded raw fixture; it is not a PASS.
- A wider run including the old checkpoint test produced a pre-existing failure:
  it expects a CURRENT_C1 fixture to resume, which current C1 intentionally forbids.
  Replaying the HEAD checkpoint implementation reproduces the same refusal.
  It was neither fixed nor relabeled as passing. Real CURRENT_C2 graph pause/resume
  succeeds in the local HTTP acceptance.
- New modules/tests/acceptance script pass Ruff. Changed production lines are
  checked separately; pre-existing file-wide lint debt is not silently reformatted.
- The 635 files in the v2 closeout input-identity freeze still match. Existing
  Graph/Wiki research implementations, strict publisher and semantic verifier are
  unchanged; no historical scientific result was rewritten.

Earlier integration attempts are retained, not overwritten: SQLite could not
represent the existing PostgreSQL ARRAY model, so the harness moved to actual
PostgreSQL. The first actual graph retrieve hit the exact citation guard because
multi-page chunk maps were missing; production now recovers them from sealed
parser blocks instead of assigning all blocks to the first page. A subsequent
isolated Agent test lacked the repository-relative original backend lock; the
harness now copies its exact bytes. One setup attempt encountered an already
created data directory; no query or admission occurred in that attempt. These
are engineering debugging runs, not repeated scientific qualification trials.

## Remaining limits and release scope

- Relations are deterministic containment within papers, not inferred inter-paper
  semantic relations, reviewed claims or scientific support labels. Source-backed
  Wiki is available, but there is no new Wiki LLM synthesis or semantic judge.
- Graph opt-in currently supports text ACTIVE Agent and paper filters. Combining
  native multimodal Agent/readonly mode or page/section filters is explicitly
  refused. No claim is made that all modality combinations are integrated.
- Full canonical graph/Wiki reconstruction on every query prioritizes identity
  verification over speed; the 5,000-chunk/32 MiB caps are deliberate. No load or
  cross-domain retrieval-quality benchmark was run.
- The local client 1.18.0/server 1.12.5 Qdrant version check emitted a compatibility
  warning. Actual admission, physical checks and retrieval succeeded; the warning
  was not suppressed, and this is not qualification of that version pairing.
  Use a compatible pair for deployment. No global upgrade/service replacement
  was performed.
- The existing single-operator authority is retained; no new multi-tenant ACL is
  implied. Generation/scope validation is not user authentication.
- Automatic semantic verification, visual/math correctness and automatic trusted
  publication remain unqualified. FNO and U-NO scientific FAIL are unchanged.

The feature branch publishes source, targeted tests, configuration switches and
these concise safe documents. Raw PDFs, generated indexes, SQL data, request
logs, model assets and local worktree/runtime directories are excluded. Publication
targets the existing PR #1; there is no main/release rewrite or automatic merge.
No GitHub Actions workflow is configured in the current repository; CI success
is not claimed. The existing PR remains Draft for final review of this release
and the documented compatibility/test limitations.
