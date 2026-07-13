# ruff: noqa: E501
"""Generate three deterministic PDF classes and validate the OCR fallback end to end."""

import json
import os
import time
from pathlib import Path

import fitz
from qdrant_client import QdrantClient

from paper_research.chunking.structural_chunker import StructuralChunker
from paper_research.config import Settings
from paper_research.indexing.embedding import HashEmbeddingProvider
from paper_research.indexing.vector_store import QdrantVectorStore
from paper_research.parsing.router import ParserRouter
from paper_research.retrieval.dense import DenseRetriever

ROOT = Path("data/ocr-audit-v1")
REPORT = Path("docs/ocr-audit-v1.md")
SENTENCES = {
    "text": "Text native evidence says attention enables parallel sequence modeling.",
    "mixed": "Scanned mixed-page evidence says OCR fallback preserves the source page.",
    "scanned": "Fully scanned evidence says optical recognition recovers searchable content.",
}


def make_text_pdf(path: Path, sentence: str) -> None:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 100), "OCR Release Candidate Audit", fontsize=18)
    page.insert_text((72, 150), sentence, fontsize=11)
    document.save(path)
    document.close()


def image_page(sentence: str) -> bytes:
    source = fitz.open()
    page = source.new_page()
    page.insert_text((72, 100), "Scanned Page", fontsize=18)
    page.insert_text((72, 150), sentence, fontsize=11)
    payload = page.get_pixmap(matrix=fitz.Matrix(3, 3), alpha=False).tobytes("png")
    source.close()
    return payload


def make_scanned_pdf(path: Path, sentence: str) -> None:
    document = fitz.open()
    page = document.new_page()
    page.insert_image(page.rect, stream=image_page(sentence))
    document.save(path)
    document.close()


def make_mixed_pdf(path: Path, sentence: str) -> None:
    document = fitz.open()
    first = document.new_page()
    first.insert_text((72, 100), "Native first page for mixed document.", fontsize=11)
    second = document.new_page()
    second.insert_image(second.rect, stream=image_page(sentence))
    document.save(path)
    document.close()


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    files = {
        "text": ROOT / "text-native.pdf",
        "mixed": ROOT / "mixed-native-scanned.pdf",
        "scanned": ROOT / "fully-scanned.pdf",
    }
    make_text_pdf(files["text"], SENTENCES["text"])
    make_mixed_pdf(files["mixed"], SENTENCES["mixed"])
    make_scanned_pdf(files["scanned"], SENTENCES["scanned"])
    settings = Settings(parser_backend="auto", ocr_language="eng")
    router = ParserRouter(settings=settings)
    embedding = HashEmbeddingProvider(384)
    results = []
    for kind, path in files.items():
        started = time.perf_counter()
        parsed = router.parse(path)
        parse_ms = round((time.perf_counter() - started) * 1000, 3)
        chunks = StructuralChunker().chunk(kind, parsed.blocks)
        store = QdrantVectorStore(QdrantClient(":memory:"), f"ocr-{kind}", 384)
        store.upsert(chunks, embedding.embed([chunk.chunk_text for chunk in chunks]))
        answer_results = DenseRetriever(embedding, store).retrieve(SENTENCES[kind], top_k=3)
        combined = " ".join(block.text for block in parsed.blocks).lower()
        expected_page = 2 if kind == "mixed" else 1
        row = {
            "pdf_type": kind,
            "path": str(path),
            "parser_name": parsed.parser_name or parsed.parser,
            "is_ocr": parsed.is_ocr,
            "ocr_confidence": parsed.ocr_confidence,
            "source_pages": sorted({block.source_page or block.page_start for block in parsed.blocks}),
            "parse_warnings": [warning.model_dump() for warning in parsed.warnings],
            "block_count": len(parsed.blocks),
            "chunk_count": len(chunks),
            "parse_latency_ms": parse_ms,
            "text_recovered": " ".join(SENTENCES[kind].lower().split()) in " ".join(combined.split()),
            "reading_order_preserved": all(
                block.block_index == index for index, block in enumerate(parsed.blocks)
            ),
            "page_preserved": expected_page in {
                block.source_page or block.page_start for block in parsed.blocks
            },
            "normalized_blocks": all(
                block.block_id and block.text and block.bbox and block.source_page
                for block in parsed.blocks
            ),
            "indexed": bool(chunks),
            "qa_returned": bool(answer_results),
            "citation_page": answer_results[0].chunk.page_start if answer_results else None,
            "citation_page_end": answer_results[0].chunk.page_end if answer_results else None,
            "citation_contains_expected_page": bool(answer_results)
            and answer_results[0].chunk.page_start <= expected_page
            <= answer_results[0].chunk.page_end,
        }
        results.append(row)
    output = {
        "tesseract_executable": os.environ.get("TESSERACT_EXE"),
        "tessdata_prefix": os.environ.get("TESSDATA_PREFIX"),
        "results": results,
    }
    (ROOT / "ocr-audit-results.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(output)
    print(json.dumps(output, ensure_ascii=False, indent=2))


def write_report(output: dict) -> None:
    lines = [
        "# OCR End-to-End Audit v1",
        "",
        f"- Tesseract: `{output['tesseract_executable']}`",
        f"- TESSDATA_PREFIX: `{output['tessdata_prefix']}`",
        "- OCR is an optional fallback, not the default text-PDF path.",
        "- OCR confidence is recorded as `null`: PyMuPDF's OCR text-page API does not expose word confidence.",
        "",
        "| PDF type | Parser | OCR | Blocks | Chunks | Text recovered | Order | Page | Indexed | QA | Citation range | Contains evidence page | Parse ms |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in output["results"]:
        lines.append(
            f"| {row['pdf_type']} | {row['parser_name']} | {row['is_ocr']} | "
            f"{row['block_count']} | {row['chunk_count']} | {row['text_recovered']} | "
            f"{row['reading_order_preserved']} | {row['page_preserved']} | "
            f"{row['indexed']} | {row['qa_returned']} | "
            f"{row['citation_page']}-{row['citation_page_end']} | "
            f"{row['citation_contains_expected_page']} | "
            f"{row['parse_latency_ms']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Installation boundary",
            "",
            "Host OCR tests require `D:\\Program Files\\Tesseract-OCR\\tesseract.exe` and `D:\\Program Files\\Tesseract-OCR\\tessdata\\eng.traineddata`. The current API Docker image does not install Tesseract, so OCR inside the deployed container remains unverified/unsupported until the image adds the package and language data.",
            "",
        ]
    )
    REPORT.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
