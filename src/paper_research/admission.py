"""Local trusted-operator PDF admission; deliberately has no HTTP authority route."""

import argparse
import getpass
import hashlib
import json
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pymupdf as fitz
from fastapi import UploadFile

from paper_research.config import get_settings
from paper_research.indexing.registry import IndexIdentityError, IndexRegistry, runtime_index_identity


def status(settings) -> dict:
    registry = IndexRegistry(settings.data_dir / "index_registry.json")
    if not (registry.trust_root / "authorization.json").exists():
        return {"status": "EMPTY", "data_dir": str(settings.data_dir)}
    auth = registry.authorization()
    journal = registry.journal(auth["generation_id"])
    binding = registry.read().get("indexes", {}).get(auth["collection_name"], {}).get("trust_binding")
    result = {"status": journal["state"], "admission_id": auth["admission_id"],
              "generation_id": auth["generation_id"], "collection": auth["collection_name"],
              "failure": journal["failure"], "inventory_digest": journal["inventory_digest"],
              "seal_digest": journal["seal_digest"]}
    if binding and binding["generation_id"] == auth["generation_id"]:
        result["status"] = binding["publication_state"]
    commitment = registry.admission_path(auth["admission_id"], "commitment.json")
    if commitment.exists():
        documents = registry.read_record(commitment, "admission_commitment", "admission_digest")
        result["paper_ids"] = [d["paper_id"] for d in documents["documents"]]
    return result


def admit(settings, paths: list[Path], authorization_ref: str) -> dict:
    from paper_research.db import SessionLocal
    from paper_research.indexing.rebuild import IndexRebuildService
    from paper_research.repositories.paper import PaperRepository

    registry = IndexRegistry(settings.data_dir / "index_registry.json")
    if (registry.trust_root / "authorization.json").exists():
        raise ValueError("This DATA_DIR already has an admission; inspect status. Do not overwrite it.")
    # Validate the entire closed batch before creating any authorization or source files.
    inputs, hashes = [], set()
    with SessionLocal() as db:
        for path in paths:
            if path.suffix.lower() != ".pdf" or not path.is_file():
                raise ValueError(f"Not a PDF file: {path}")
            if not 0 < path.stat().st_size <= settings.max_upload_bytes:
                raise ValueError(f"PDF is empty or exceeds the configured size limit: {path}")
            raw = path.read_bytes()
            if not raw.startswith(b"%PDF-"):
                raise ValueError(f"Invalid PDF header: {path}")
            with fitz.open(stream=raw, filetype="pdf") as document:
                if document.is_encrypted or document.is_repaired or document.page_count < 1:
                    raise ValueError(f"PDF is encrypted, damaged, or empty: {path}")
            digest = hashlib.sha256(raw).hexdigest()
            if digest in hashes or PaperRepository(db).get_by_hash(digest) is not None:
                raise ValueError(f"Duplicate PDF bytes: {path}")
            hashes.add(digest)
            inputs.append((path, raw, digest))
    admission, generation = str(uuid.uuid4()), str(uuid.uuid4())
    sources = []
    for index, (_, raw, digest) in enumerate(inputs):
        relative = f"inputs/{admission}/{digest}.pdf"
        registry.durable_bytes(registry.protected_path(relative), raw, immutable=True)
        sources.append({"source_ref": f"source-{index}", "pdf_sha256": digest,
                        "byte_count": len(raw), "input_relative_path": relative})
    collection = settings.active_collection
    authority = f"local-operator:{getpass.getuser()}"
    intent = {"schema_version": "corpus-admission-intent-v1", "admission_id": admission,
              "authorization_ref": authorization_ref, "authority_id": authority,
              "closed_input_scope": "EXPLICIT_OPERATOR_AUTHORIZED_PDFS", "sources": sources,
              "created_at": datetime.now(UTC).isoformat(), "intent_digest": "0" * 64}
    authorization = {"schema_version": "controlled-corpus-authorization-v1",
                     "authority_id": authority, "authorization_ref": authorization_ref,
                     "admission_id": admission, "generation_id": generation,
                     "logical_collection": collection, "collection_name": collection,
                     "index_identity": runtime_index_identity(settings.provider_metadata, collection),
                     "runtime_metadata": settings.provider_metadata}
    registry.initialize_admission(intent, authorization)
    print(json.dumps({"status": "ADMITTING", "admission_id": admission,
                      "generation_id": generation}), file=sys.stderr, flush=True)
    with SessionLocal() as db:
        for source, (original, _, _) in zip(sources, inputs, strict=True):
            with registry.protected_path(source["input_relative_path"]).open("rb") as stream:
                registry.intake_source(admission, source["source_ref"],
                                       UploadFile(stream, filename=original.name), db, settings)
            print(json.dumps({"status": "ADMITTED", "source_ref": source["source_ref"]}),
                  file=sys.stderr, flush=True)
    print(json.dumps({"status": "BUILDING", "generation_id": generation}), file=sys.stderr, flush=True)
    IndexRebuildService(settings).rebuild(admission)
    return status(settings)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status", help="Read existing admission/build/publication status")
    command = commands.add_parser("admit", help="Authorize and build one closed PDF batch")
    command.add_argument("--approve", action="store_true", required=True,
                         help="Explicitly authorize these exact PDF bytes as a trusted local operator")
    command.add_argument("--authorization-ref", required=True, help="Operator's approval reference/reason")
    command.add_argument("pdf", nargs="+", type=Path)
    args = parser.parse_args()
    settings = get_settings()
    try:
        if args.command == "admit":
            if not args.authorization_ref.strip():
                raise ValueError("authorization-ref must not be blank")
            result = admit(settings, args.pdf, args.authorization_ref.strip())
        else:
            result = status(settings)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "ERROR", "error_code": exc.code if isinstance(exc, IndexIdentityError)
                          else type(exc).__name__, "message": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
