# ruff: noqa: E501
"""Build the Stage 11A.5 production corpus manifest and retrieval protocol.

This script never edits gold-set-v1.jsonl.  It preserves the approved answer
annotations while adding a separate, auditable retrieval task definition.
"""

import json
from pathlib import Path

from sqlalchemy import create_engine, text

from paper_research.config import Settings

GOLD_V1 = Path("data/evaluation/gold-set-v1.jsonl")
CORPUS_V1 = Path("data/evaluation/production-corpus-v1.json")
RETRIEVAL_GOLD_V2 = Path("data/evaluation/retrieval-gold-v2.jsonl")
AUDIT_REPORT = Path("docs/retrieval-gold-v2-audit.md")

MIXED_OCR_FIXTURE = "9a9b40a4-1725-418d-8c82-8cad235a34c2"
SCANNED_OCR_FIXTURE = "fbd2556c-fb5f-4088-8c8a-52223a32a1bd"
TEXT_ACCEPTANCE_FIXTURE = "0228c3e3-8630-4f5c-b8dc-b83c13eabe5a"
EXCLUDED_FIXTURES = {MIXED_OCR_FIXTURE, SCANNED_OCR_FIXTURE}

LEGACY_TITLE_TO_ARXIV = {
    "1706.03762": "1706.03762",
    "BERT: Pre-training of Deep Bidirectional Transformers for": "1810.04805",
    "1910.10683": "1910.10683",
    "Scaling Laws for Neural Language Models": "2001.08361",
    "Language Models are Few-Shot Learners": "2005.14165",
    "The Power of Scale for Parameter-Efﬁcient Prompt Tuning": "2104.08691",
    "LORA: LOW-RANK ADAPTATION OF LARGE LAN-\nGUAGE MODELS": "2106.09685",
    "Training language models to follow instructions": "2203.02155",
    "Scaling Instruction-Finetuned Language Models": "2210.11416",
    "LLaMA: Open and Efﬁcient Foundation Language Models": "2302.13971",
}

UNANSWERABLE_TARGETS = {
    "q005": "1706.03762",
    "q030": "2104.08691",
}
UNANSWERABLE_QUERIES = {
    "q005": (
        "Within the attention-only Transformer sequence transduction paper, what exact "
        "total energy consumption is reported for all experiments?"
    ),
    "q030": (
        "Within the parameter-efficient prompt tuning paper, what exact total energy "
        "consumption is reported for all experiments?"
    ),
}
REVIEW_IDENTITY_FIELDS = (
    "question_id",
    "original_question",
    "retrieval_query",
    "retrieval_scope",
    "retrieval_filter",
    "query_revision_version",
)
REVIEW_METADATA_FIELDS = (
    "query_revision_reviewer",
    "query_revision_reviewed_at",
    "query_revision_review_notes",
)


def canonical_paper_id(database_id: str, title: str, arxiv_id: str | None) -> str:
    return str(arxiv_id or LEGACY_TITLE_TO_ARXIV.get(title.strip()) or database_id)


def build_corpus(settings: Settings) -> dict:
    engine = create_engine(settings.database_url)
    with engine.connect() as connection:
        rows = connection.execute(
            text(
                "SELECT id, title, arxiv_id, pdf_path, source_url, file_hash "
                "FROM papers ORDER BY created_at"
            )
        ).mappings()
        papers = list(rows)
    entries = []
    for paper in papers:
        database_id = str(paper["id"])
        excluded = database_id in EXCLUDED_FIXTURES
        if excluded:
            role = "ocr_fixture"
            reason = "OCR fallback fixture retained for OCR tests; excluded from retrieval evaluation"
        elif database_id == TEXT_ACCEPTANCE_FIXTURE:
            role = "release_acceptance_text_fixture"
            reason = None
        else:
            role = "research_paper"
            reason = None
        entries.append(
            {
                "paper_id": canonical_paper_id(
                    database_id, str(paper["title"]), paper["arxiv_id"]
                ),
                "database_id": database_id,
                "title": str(paper["title"]),
                "source_path": paper["pdf_path"],
                "source_url": paper["source_url"],
                "file_hash": paper["file_hash"],
                "corpus_role": role,
                "included_in_production": not excluded,
                "exclusion_reason": reason,
            }
        )
    included = [entry for entry in entries if entry["included_in_production"]]
    excluded = [entry for entry in entries if not entry["included_in_production"]]
    research_count = sum(entry["corpus_role"] == "research_paper" for entry in included)
    if len(entries) != 36 or len(included) != 34 or len(excluded) != 2:
        raise RuntimeError(
            f"unexpected corpus boundary: total={len(entries)}, included={len(included)}, "
            f"excluded={len(excluded)}"
        )
    if {entry["database_id"] for entry in excluded} != EXCLUDED_FIXTURES:
        raise RuntimeError("OCR fixture exclusion set does not match the audited database IDs")
    return {
        "manifest_version": "production-corpus-v1",
        "boundary_basis": (
            "36 indexed documents minus the mixed-page and fully-scanned OCR fixtures; "
            "the text-native release acceptance document remains included"
        ),
        "total_indexed_documents": len(entries),
        "included_documents": len(included),
        "included_research_papers": research_count,
        "included_release_acceptance_documents": len(included) - research_count,
        "excluded_ocr_fixtures": len(excluded),
        "papers": entries,
    }


def load_gold() -> list[dict]:
    return [
        json.loads(line)
        for line in GOLD_V1.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def retrieval_record(item: dict) -> dict:
    question_id = item["question_id"]
    if not item["answerable"]:
        scope = "unanswerable"
        target_ids = [UNANSWERABLE_TARGETS[question_id]]
        query = UNANSWERABLE_QUERIES[question_id]
        reason = (
            "The original unanswerable item omitted its target paper and was duplicated. "
            "The retrieval query adds a natural-language paper identity inferred from its "
            "five-question source group; gold evidence remains empty."
        )
        query_review_status = "pending_human_review"
    elif item["scope"] == "multi_paper":
        scope = "multi_paper"
        target_ids = list(item["gold_paper_ids"])
        query = item["question"]
        reason = (
            "Specified-paper comparison: retained the approved natural question and added "
            "the two-paper retrieval filter. q049 evaluates contributions while q050 "
            "evaluates architectural usage, so their evidence intents remain distinct."
        )
        query_review_status = "not_required_scope_only"
    else:
        scope = "paper"
        target_ids = list(item["gold_paper_ids"])
        query = item["question"]
        reason = (
            "Known-paper reading comprehension: retained the approved question verbatim "
            "and added a paper_id filter instead of treating it as paper discovery."
        )
        query_review_status = "not_required_scope_only"
    return {
        "question_id": question_id,
        "original_question": item["question"],
        "retrieval_query": query,
        "retrieval_scope": scope,
        "retrieval_filter": {"paper_ids": target_ids},
        "gold_paper_ids": list(item["gold_paper_ids"]),
        "gold_pages": list(item["gold_pages"]),
        "gold_block_ids": list(item["gold_block_ids"]),
        "category": item["category"],
        "difficulty": item["difficulty"],
        "review_status": item["review_status"],
        "query_revision_reason": reason,
        "query_revision_version": "retrieval-query-v2",
        "query_revision_author": "Codex Stage 11A.5 protocol correction",
        "query_revision_review_status": query_review_status,
    }


def preserve_query_review(record: dict, previous: dict | None) -> dict:
    """Preserve human review only while every review identity field is unchanged."""
    if previous is None:
        return record
    unchanged = all(previous.get(field) == record.get(field) for field in REVIEW_IDENTITY_FIELDS)
    if unchanged:
        record["query_revision_review_status"] = previous.get(
            "query_revision_review_status", record["query_revision_review_status"]
        )
        for field in REVIEW_METADATA_FIELDS:
            if field in previous:
                record[field] = previous[field]
    else:
        record["query_revision_review_status"] = "pending_human_review"
        for field in REVIEW_METADATA_FIELDS:
            record.pop(field, None)
    return record


def validate_protocol(records: list[dict], corpus: dict) -> None:
    included_ids = {
        entry["paper_id"] for entry in corpus["papers"] if entry["included_in_production"]
    }
    scopes = {"global", "paper", "multi_paper", "unanswerable"}
    if len(records) != 50 or any(record["retrieval_scope"] not in scopes for record in records):
        raise RuntimeError("retrieval protocol must contain 50 records with valid scopes")
    for record in records:
        if record["review_status"] != "approved":
            raise RuntimeError(f"non-approved source record: {record['question_id']}")
        if not record["query_revision_reason"]:
            raise RuntimeError(f"missing revision reason: {record['question_id']}")
        if record["query_revision_review_status"] == "approved" and not all(
            record.get(field) for field in REVIEW_METADATA_FIELDS
        ):
            raise RuntimeError(
                f"approved query revision lacks review metadata: {record['question_id']}"
            )
        if record["retrieval_scope"] == "global":
            lowered = record["retrieval_query"].lower()
            if "the target paper" in lowered or "target papers" in lowered:
                raise RuntimeError(f"unresolved global reference: {record['question_id']}")
        if record["retrieval_scope"] == "paper" and len(
            record["retrieval_filter"]["paper_ids"]
        ) != 1:
            raise RuntimeError(f"paper scope requires one filter: {record['question_id']}")
        if record["retrieval_scope"] == "multi_paper" and len(
            record["retrieval_filter"]["paper_ids"]
        ) < 2:
            raise RuntimeError(f"multi-paper scope requires two papers: {record['question_id']}")
        if record["retrieval_scope"] == "unanswerable" and (
            record["gold_paper_ids"] or record["gold_block_ids"] or record["gold_pages"]
        ):
            raise RuntimeError(f"unanswerable gold evidence must be empty: {record['question_id']}")
        unknown = set(record["retrieval_filter"]["paper_ids"]) - included_ids
        if unknown:
            raise RuntimeError(f"filter references papers outside production corpus: {unknown}")


def write_audit(records: list[dict]) -> None:
    counts = {
        scope: sum(record["retrieval_scope"] == scope for record in records)
        for scope in ("global", "paper", "multi_paper", "unanswerable")
    }
    rewritten = [
        record for record in records if record["retrieval_query"] != record["original_question"]
    ]
    filtered = [record for record in records if record["retrieval_filter"]["paper_ids"]]
    pending = [
        record
        for record in records
        if record["query_revision_review_status"] == "pending_human_review"
    ]
    approved_revisions = [
        record
        for record in records
        if record["query_revision_review_status"] == "approved"
    ]
    lines = [
        "# Retrieval Gold v2 Protocol Audit",
        "",
        "## Summary",
        "",
        f"- Global: {counts['global']}",
        f"- Paper: {counts['paper']}",
        f"- Multi-paper: {counts['multi_paper']}",
        f"- Unanswerable: {counts['unanswerable']}",
        f"- Retrieval query text changed: {len(rewritten)}",
        f"- Queries with a paper filter: {len(filtered)}",
        f"- Human-approved query revisions: {len(approved_revisions)}",
        f"- Query revisions pending human review: {len(pending)}",
        "- Original `gold-set-v1.jsonl` modified: no",
        "",
        "All 46 answerable single-paper records are known-paper reading-comprehension tasks, not paper-discovery tasks. Their approved question text is retained and evaluation applies a one-paper filter. The two comparison records apply the same explicit two-paper filter. No item is naturally global, so Global metrics are reported as N/A rather than creating answer-derived discovery queries.",
        "",
        "## Duplicate and near-duplicate audit",
        "",
        "`q005` and `q030` had identical text and no stored target paper. Their v2 queries add different natural-language paper identities inferred from the surrounding five-question source groups. Both inferences were human-approved by `zjf` on 2026-07-13; no gold paper, block, or page was fabricated. `q049` and `q050` share the same two-paper filter but remain distinct: q049 asks for contribution comparison, while q050 asks how the architectures are used.",
        "",
        "## Filter and gold-evidence semantics",
        "",
        "For an `unanswerable` record, `retrieval_filter.paper_ids` identifies the paper being searched. Empty `gold_paper_ids`, `gold_block_ids`, and `gold_pages` mean there is no evidence supporting an answer; the filter is target scope, not positive gold evidence. A future schema version should add an explicit `target_paper_ids` field so this distinction is structural rather than documented only by convention.",
        "",
        "## Before and after",
        "",
        "| ID | Scope | Before | Retrieval query | Filter | Revision reason | Query review | Reviewer | Reviewed at |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for record in records:
        before = record["original_question"].replace("|", "\\|")
        after = record["retrieval_query"].replace("|", "\\|")
        reason = record["query_revision_reason"].replace("|", "\\|")
        paper_filter = ", ".join(record["retrieval_filter"]["paper_ids"])
        lines.append(
            f"| {record['question_id']} | {record['retrieval_scope']} | {before} | {after} | {paper_filter} | {reason} | {record['query_revision_review_status']} | {record.get('query_revision_reviewer', '')} | {record.get('query_revision_reviewed_at', '')} |"
        )
    lines.extend(
        [
            "",
            "## Why v1 was not interpretable",
            "",
            "v1 treated phrases such as `the target paper` as unrestricted corpus queries. The question contained no observable variable identifying the intended paper, so paper-ID Hit/MRR largely measured accidental lexical or embedding preference. It also mixed known-paper block retrieval, multi-paper evidence coverage, and unanswerable behavior into one answerable paper-retrieval aggregate.",
            "",
            "## What v2 fixes",
            "",
            "v2 separates scope semantics, applies explicit filters before both Dense and Sparse retrieval, evaluates paper-scoped items against gold blocks, evaluates comparisons using target-paper coverage plus evidence recall, and records unanswerable retrieval scores without including them in answerable recall. The original approved questions remain intact in a separate field and file.",
            "",
            "## Remaining limitations",
            "",
            "- The dataset contains no genuine global paper-discovery questions, so it cannot validate corpus-wide semantic paper identification.",
            "- The current schema relies on `retrieval_filter.paper_ids` to carry the target for unanswerable items; a future version should add `target_paper_ids`.",
            "- Most gold evidence covers ten legacy papers, not the full 34-document Production corpus.",
            "- Block IDs are parser/chunker-version dependent; future re-parsing requires a gold evidence migration audit.",
            "- Retrieval scores alone cannot establish answerability or refusal thresholds.",
            "",
        ]
    )
    AUDIT_REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    settings = Settings()
    corpus = build_corpus(settings)
    source = load_gold()
    previous_records = (
        {record["question_id"]: record for record in load_jsonl(RETRIEVAL_GOLD_V2)}
        if RETRIEVAL_GOLD_V2.exists()
        else {}
    )
    records = [
        preserve_query_review(retrieval_record(item), previous_records.get(item["question_id"]))
        for item in source
    ]
    validate_protocol(records, corpus)
    CORPUS_V1.write_text(json.dumps(corpus, ensure_ascii=False, indent=2), encoding="utf-8")
    RETRIEVAL_GOLD_V2.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    write_audit(records)
    counts = {
        scope: sum(record["retrieval_scope"] == scope for record in records)
        for scope in ("global", "paper", "multi_paper", "unanswerable")
    }
    print(json.dumps({"corpus": {"included": 34, "excluded": 2}, "scopes": counts}))


if __name__ == "__main__":
    main()
