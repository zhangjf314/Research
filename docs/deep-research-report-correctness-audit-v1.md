# Deep Research Report Correctness Audit v1

Date: 2026-07-23

This post-release hotfix addresses two confirmed issues in the Deep Research UI
and report generation path.

## Root cause

1. `/api/v1/ui/research` hard-coded a real default query inside the textarea:
   `RAG methods, results, and limitations`. This could create accidental tasks
   and made the UI look pre-filled instead of user-driven.
2. `/api/v1/research/deep` was still wired to an old artifact-local research
   provider. That provider scanned parsed JSONL files and rebuilt BM25 on each
   request. The graph then stored evidence with `(evidence_id, sub_question)`,
   allowing the same evidence to appear once per section.
3. `_synthesize()` did not synthesize a structured answer. It only detected a
   narrow contradiction pattern.
4. `_report()` copied chunks into each section and listed every evidence record,
   which could duplicate the same evidence and produce identical sections.
5. Citation validation only checked whether a marker string existed in the final
   Markdown report. It did not verify that the citation ID existed in a unique
   evidence catalog or that references were deduplicated.

## Fix summary

- The Research UI textarea is now empty by default and uses a placeholder plus a
  manual "fill example" button.
- Empty or too-short UI queries are rejected client-side before calling
  `/api/v1/research/deep`.
- Production API routing now prefers the existing hybrid retrieval stack through
  `HybridLocalResearchProvider`.
- If production hybrid retrieval cannot be constructed, the API returns
  `FAILED_RETRIEVAL` instead of silently falling back to the legacy evidence dump
  path as a successful report.
- Evidence is normalized into a `ResearchEvidence` model.
- The global evidence key is `paper_id + evidence_id`.
- Section-to-evidence relationships are stored separately in
  `section_evidence_ids`.
- Reports are generated from structured synthesis data and deterministic Markdown
  rendering, not by dumping raw chunks into four sections.
- References are globally deduplicated by citation ID.
- Report quality metrics now check duplicate paragraphs, duplicate bullets,
  duplicate references, cross-section similarity, unique evidence count, section
  evidence counts, citation ID validity, citation context/page validity, and raw
  quote copy ratio.

## Production route before

```text
POST /api/v1/research/deep
-> ArtifactLocalResearchProvider(settings.parsed_papers_dir)
-> scan */paper_chunks.jsonl
-> rebuild BM25 per request
-> evidence key = evidence_id + sub_question
-> _report copies evidence chunks into Markdown
-> marker-string validation
```

## Production route after

```text
POST /api/v1/research/deep
-> HybridLocalResearchProvider(settings)
-> Dense retrieval + lexical retrieval + RRF + Qdrant
-> section-specific retrieval queries
-> global ResearchEvidence catalog
-> section_evidence_ids relationship layer
-> structured deterministic synthesis
-> deterministic Markdown report
-> citation/catalog validation
-> report quality gate
```

## Remaining boundaries

- This hotfix does not rerun the frozen 50-item Full QA evaluation.
- This hotfix does not rerun frozen Deep Research release evidence.
- This hotfix does not move or recreate `v1.0.0-portfolio`.
- `SEMANTIC_CLAIM_SUPPORT_AUDIT=NOT_FORMALLY_VALIDATED` remains in force.
- `STRONG_GROUNDING_CLAIM_ALLOWED=false` remains in force.
- The deterministic citation checks should not be overstated as full semantic
  claim-support proof.
