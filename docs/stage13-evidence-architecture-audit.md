# Stage 13 Evidence Architecture Audit

Audit date: 2026-07-14. This audit was completed before changing the retrieval protocol.

## 1. Corpus block and chunk schema

`PaperBlock` is the persisted parser unit. It stores `block_id`, one of six block types
(`title`, `heading`, `paragraph`, `table`, `formula`, `reference`), `section_path`, one-based
`page_start/page_end`, `block_index`, text, bounding box, parent/previous/next block IDs,
`source_page`, and OCR fields. PyMuPDF, Docling and GROBID populate previous/next links within a
paper. The schema has no explicit caption, figure, algorithm or theorem types; those structures
are therefore represented as headings, paragraphs, tables or formulas depending on parser output.

The schema does not persist sentence boundaries. Some analysis code splits text into sentences at
runtime, but neither the block nor chunk artifact records spans. Formula and table blocks are
retained. Captions, algorithms and theorems are not reliably distinguishable. OCR confidence is
available, but parse confidence is not generalized beyond OCR.

`StructuralChunker` drops headings as standalone retrieval content, groups consecutive blocks only
when section path and block type match, caps chunks at 400 estimated tokens, and creates overlapping
windows for oversized single blocks. A chunk stores several `block_ids` but only the first block's
type and section path. Its UUID is not derived from content. Neighbor context stores text, not
neighbor block identities. Thus a block may appear in one chunk or multiple overlapping windows,
and a chunk can mix several independently citable blocks.

## 2. Gold granularity and claim obligations

The 50-item Gold Set contains 48 answerable questions and two unanswerable questions. Every
answerable question has three verbatim `required_claims`: 144 claims total. These questions list
164 Gold block IDs, or 3.42 blocks per question; 45/48 require more than one Gold block and 14 have
multiple Gold blocks on one Gold page. The current schema assigns Gold blocks to the question, not
to individual required claims. It therefore cannot determine without new annotation whether:

- one block supports several claims;
- one claim requires several blocks jointly;
- blocks are alternatives rather than a joint set;
- a page-level match supports a particular claim.

All four patterns are possible in the data, but claiming a specific mapping would be Gold leakage.
The planned claim-evidence file must remain pending until a reviewer establishes these relations.
Gold is block-oriented, while retrieval is chunk-oriented. Exact block availability is recovered by
expanding each retrieved chunk's `block_ids`; the chunk itself is not a Gold unit.

The audit corpus visibly contains same-page blocks with different semantics, overly short heading
fragments, author/title metadata, reference blocks, and long paragraph blocks. Parser boundaries can
split one logical statement across adjacent blocks. OCR fixtures are excluded from the 34-document
Production corpus, so OCR boundary defects are not allowed to influence formal Stage 13 metrics.

## 3. Existing retrieval and score semantics

Dense retrieval uses Jina query vectors against stored chunk vectors. Sparse retrieval is an
in-memory BM25 index over the same chunks. `HybridRetriever` executes both in parallel and applies
rank-based reciprocal-rank fusion. A fused row retains final RRF score plus dense and sparse ranks,
but not calibrated dense/lexical score components or evidence-role features. Structural information
only enters through chunk construction and optional context expansion/caps; it is not an explicit
support score.

Stage 11C.6 uses the fixed 34-document, 2,062-point Jina collection and has already shown that
larger Top-N, neighbor expansion and same-page expansion do not reliably improve answer quality.
The current default context builder consumes fused rank order, optionally prepends/appends neighbor
text, deduplicates only by chunk UUID, and truncates a rank-order prefix by character/token budget.
It therefore optimizes query relevance rather than direct claim support. It can admit metadata,
reference-only material and several chunks from one section unless a separate experimental strategy
adds caps.

## 4. Citation timing and validation

The Production prompt receives all allowed context triples. The LLM generates claims and citations
together. `QAService` then validates every `(paper_id, page, block_id)` against the supplied context
and rejects invalid triples. It may retry malformed/invalid output, but it does not repair an illegal
triple. This is strict identifier validity, not evidence support.

Citation allocation is therefore currently generation-time selection from the whole context,
followed by post-generation triple validation. There is no pre-generation claim-specific evidence
allocation. A claim can choose any allowed context citation, and the existing deterministic semantic
diagnostic has been shown to overestimate human support.

## 5. Redesign boundary

Stage 13 must add a derived Evidence Unit without changing `PaperBlock`, derive Claim Units without
changing required claims, and allocate evidence before generation. Metadata/reference filtering,
score decomposition, multi-block sets and per-claim context traces must be explicit. Page proximity,
adjacency and token overlap remain diagnostic features only and never become citation correctness.
The existing Jina collection, Gold files, prompt baseline, strict triple validator and all historical
artifacts remain unchanged.
