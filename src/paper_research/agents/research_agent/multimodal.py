"""Opt-in bridge to Phase4 native records. No changes to frozen experiments."""

import base64
import importlib.util
import json
import math
import re
import sys
from functools import lru_cache
from pathlib import Path

from paper_research.agents.research_agent.models import NativeEvidenceItem
from paper_research.agents.research_agent.verifier import ResearchSafetyError
from paper_research.chunking.types import Chunk
from paper_research.indexing.registry import CorpusTrustError
from paper_research.pdf_multimodal.identity import digest, file_sha, native
from paper_research.retrieval.sparse import BM25Retriever

ROOT = Path(__file__).resolve().parents[4]
PHASE4 = ROOT / "artifacts/paperresearch-v2"


@lru_cache
def phase4_modules():
    """Read-only imports of the qualified geometry pipeline and pure RRF adapter."""

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    folder = PHASE4 / "phase4a-r6-source-id-repair/engineering"
    geometry = load("geometry_v2", folder / "geometry_v2.py")
    pipeline = load("agent_phase4_pipeline", folder / "pipeline_v2.py")
    fusion = load(
        "agent_phase4_fusion",
        PHASE4 / "phase4b-v3-b2-retrieval-preregistration/engineering/contract.py",
    )
    return geometry, pipeline, fusion


def extract_active_pdf(path, native_paper_id, expected_sha, page_indices=None):
    """Small-scope transport into unchanged Phase4 detector/constructor functions.

    The caller must obtain expected_sha from D6 source admission. Historical CLI
    fixture/formal authorization guards are left intact and are never called.
    """
    import pymupdf as fitz

    from paper_research.pdf_multimodal import extraction as old
    from paper_research.pdf_multimodal.geometry import q

    g, pipeline, _ = phase4_modules()
    c = g.make_protocol()
    if file_sha(path) != expected_sha:
        raise CorpusTrustError("NATIVE_PDF_IDENTITY_INVALID")
    components, pages = [], []
    with fitz.open(path) as doc:
        if len(page_indices if page_indices is not None else doc) > 80:
            raise ResearchSafetyError("NATIVE_SMALL_SCOPE_LIMIT", 422)
        for pi, page in enumerate(doc):
            if page_indices is not None and pi not in page_indices:
                continue
            rect = page.rect * page.derotation_matrix
            w, h = rect.width, rect.height
            pages.append(
                {
                    "paper_id": native_paper_id,
                    "pdf_sha256": expected_sha,
                    "page_index": pi,
                    "page_number": pi + 1,
                    "width_pt": q(w),
                    "height_pt": q(h),
                    "width_native": w,
                    "height_native": h,
                    "page_bounds_native": [0, 0, w, h],
                    "rotation": page.rotation,
                    "mediabox": native(page.mediabox),
                    "cropbox": native(page.cropbox),
                }
            )
            calls = c["calls"]

            def source(kind, raw, box, refs, pi=pi, w=w, h=h):
                return g.source(kind, raw, box, native_paper_id, expected_sha, pi, w, h, refs)

            for bi, block in enumerate(page.get_text(**calls["text"]["kwargs"])["blocks"]):
                if block["type"] != 0:
                    continue
                s = source("TEXT_BLOCK", block, block["bbox"], {"block_index": bi})
                components.append(s)
                cap = old.caption_anchor(s, c) if s["evidence_visible"] else None
                for li, line in enumerate(block["lines"]):
                    if old.equation_branch(line, q(w), cap is not None, c):
                        components.append(
                            source(
                                "TEXT_LINE",
                                line,
                                line["bbox"],
                                {
                                    "block_component_id": s["component_id"],
                                    "block_index": bi,
                                    "line_index": li,
                                    "span_indices": list(range(len(line["spans"]))),
                                },
                            )
                        )
            for raw in page.get_image_info(**calls["images"]["kwargs"]):
                components.append(
                    source(
                        "RASTER_OBJECT",
                        raw,
                        raw["bbox"],
                        {
                            "xref": raw.get("xref"),
                            "digest": native(raw.get("digest")),
                            "raw_placement_bbox": [q(v) for v in raw["bbox"]],
                        },
                    )
                )
            for di, raw in enumerate(page.get_drawings(**calls["vectors"]["kwargs"])):
                components.append(source("VECTOR_OBJECT", raw, raw["rect"], {"drawing_index": di}))
            for ti, table in enumerate(
                old.find_tables_unrotated(page, calls["tables"]["kwargs"]).tables
            ):
                raw = pipeline.table_dump(table, ti, c)
                components.append(
                    source(
                        "TABLE_PRIMITIVE",
                        raw,
                        table.bbox,
                        {"table_index": ti, "config_digest": digest(calls["tables"])},
                    )
                )
        page_count = len(doc)
    return pipeline.construct(
        components,
        pages,
        {native_paper_id: {"pdf_sha256": expected_sha, "page_count": page_count}},
        c,
    )


class NativeCorpus:
    """A small, independently source-roundtripped subset of existing native units."""

    def __init__(self, path: Path, *, visual=None):
        from decimal import Decimal

        self.path = path.resolve()
        self.sha = file_sha(self.path)
        # Geometry stays Decimal; opaque parser primitives retain native floats.
        raw = self.path.read_text(encoding="utf8")
        self.bundle = json.loads(raw, parse_float=Decimal)
        native = json.loads(raw)
        for a, b in zip(self.bundle["components"], native["components"], strict=True):
            a["raw"] = b["raw"]
        self.paths = {key: Path(value) for key, value in self.bundle.pop("source_paths").items()}
        g, p, self.fusion = phase4_modules()
        self.contract = g.make_protocol()
        # Re-read once per PDF rather than reopening it once per primitive. The
        # unchanged Phase4 constructor rebuilds every unit and its source identity.
        rebuilt = [
            extract_active_pdf(
                self.paths[key],
                key,
                paper["pdf_sha256"],
                [p["page_index"] for p in self.bundle["pages"] if p["paper_id"] == key],
            )
            for key, paper in self.bundle["papers"].items()
        ]
        combined = {"papers": {}, "protocol_digest": g.protocol_digest(self.contract)}
        for key in ("pages", "components", "anchors", "units"):
            combined[key] = [row for part in rebuilt for row in part[key]]
        for part in rebuilt:
            combined["papers"].update(part["papers"])
        # JSON number normalization avoids Decimal-vs-binary-float comparison
        # artifacts without changing native raw primitives or source unit IDs.
        rebuilt_json = json.loads(json.dumps(combined, default=float))
        expected_json = {key: value for key, value in native.items() if key != "source_paths"}
        if rebuilt_json != expected_json:
            raise CorpusTrustError("NATIVE_SOURCE_ROUNDTRIP_INVALID")
        self.validation = {"all_native_records_roundtripped": True, "papers": len(rebuilt)}
        # JSON-native copies preserve their own digest across Pydantic checkpoints.
        self.units = {u["unit_id"]: u for u in native["units"]}
        self.anchors = {a["caption_id"]: a for a in self.bundle["anchors"]}
        self.visual = visual
        self.full_backend = None
        self.rows = []
        for u in self.units.values():
            if u["unit_type"] == "TEXT_BLOCK":
                continue
            text = self.readable(u)
            if text:
                self.rows.append(
                    Chunk(
                        chunk_id=u["unit_id"],
                        paper_id=u["paper_id"],
                        block_ids=u["component_ids"],
                        block_type="paragraph",
                        page_start=u["page_number"],
                        page_end=u["page_number"],
                        chunk_text=text,
                        token_count=0,
                    )
                )
        self.sparse = BM25Retriever(self.rows)

    def readable(self, unit):
        if unit["unit_type"] == "FIGURE_CANDIDATE":
            return self.anchors[unit["caption_id"]]["exact_block_text"]
        return unit["raw_text"] or ""

    def guard(self):
        if file_sha(self.path) != self.sha or any(
            file_sha(self.paths[key]) != paper["pdf_sha256"]
            for key, paper in self.bundle["papers"].items()
        ):
            raise CorpusTrustError("NATIVE_SOURCE_STALE", status_code=409)

    def search(self, query, modality, arm, pdf_hashes, limit):
        self.guard()
        self.context_parents = {}
        if modality not in {"FIGURE", "TABLE", "EQUATION"} or arm not in {"N1", "M1"}:
            raise ResearchSafetyError("NATIVE_QUERY_INVALID", 422)
        eligible = {
            u["unit_id"]
            for u in self.units.values()
            if u["unit_type"] == modality + "_CANDIDATE" and u["pdf_sha256"] in pdf_hashes
            # Native detection also labels arrows, URLs and isolated model names
            # as equations. Apply the draft's existing readable-math prerequisite
            # before spending the bounded tool result slots, not after retrieval.
            and (modality != "EQUATION" or (
                not re.search(r"https?://", self.readable(u))
                and any(symbol in self.readable(u) for symbol in "=∑∈∝≤≥")))
        }
        if self.full_backend is not None:
            hits, channels = self.full_backend.search(query, arm, eligible, limit)
            # Phase4 returns representative units of selected pages. Expose
            # bounded same-page native context for Agent citations as well;
            # the frozen Retriever, encoders, vectors and page ranks stay intact.
            if modality in {"TABLE", "EQUATION"} and hits:
                pages = {(self.units[uid]["paper_id"], self.units[uid]["page_index"]): (uid, score)
                         for uid, score in hits}
                original = {uid for uid, _ in hits}
                context = [self.units[uid] for uid in eligible - original
                           if (self.units[uid]["paper_id"], self.units[uid]["page_index"]) in pages]
                context.sort(key=lambda u: (-len(self.readable(u)), u["unit_id"]))
                added = [(u["unit_id"], pages[(u["paper_id"], u["page_index"])][1])
                         for u in context[:max(0, limit - len(hits))]]
                self.context_parents = {uid: pages[(self.units[uid]["paper_id"],
                    self.units[uid]["page_index"])][0] for uid, _ in added}
                hits = hits + added
            return hits, channels
        lexical = [
            r.chunk.chunk_id
            for r in self.sparse.retrieve(query, top_k=len(self.rows))
            if r.chunk.chunk_id in eligible
        ][:20]
        channels = {"sparse": lexical}
        if arm == "M1":
            if self.visual is None:
                raise ResearchSafetyError("CLIP_INDEX_UNAVAILABLE", 422)
            channels["image"] = self.visual.rank(query, sorted(eligible))[:20]
        fused = self.fusion.reciprocal_rank_fusion(channels, {key: 1 for key in channels})
        # Native-unit citations need units on the same page, unlike page-recall evaluation.
        return fused[:limit], channels


class FullPhase4NativeBackend:
    """Run the existing N1/M1 Retriever on a small, source-compatible subset.

    No scientific defaults are rewritten: encoders, surrogates, sparse retrieval,
    fusion, reranker, tie ordering and page selection are the Phase4 functions.
    Scope/modality filtering happens before retrieval, as an Agent tool argument.
    """

    def __init__(self, corpus, path, *, build=False):
        from collections import Counter
        from types import SimpleNamespace

        import numpy as np
        from qdrant_client import QdrantClient

        self.corpus, self.path = corpus, Path(path).resolve()
        folder = PHASE4 / "phase4b-v3-b2-d1-dev-retrieval-implementation/engineering"

        # These modules only define functions/classes; never run their orchestration.
        def load(name):
            old = sys.modules.get(name)
            if (
                old is not None
                and Path(old.__file__).resolve() != (folder / (name + ".py")).resolve()
            ):
                raise ResearchSafetyError("PHASE4_MODULE_IDENTITY_CONFLICT")
            spec = importlib.util.spec_from_file_location(name, folder / (name + ".py"))
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            return module

        self.common = load("common")
        self.adapter = load("retrieval_adapter")
        builder = load("index_builder")
        runtime = load("model_runtime")
        self.counts = Counter()

        def charge(name, amount=1):
            self.counts[name] += amount
            if self.counts["max_model_forward_input_records"] > 600:
                raise ResearchSafetyError("NATIVE_LOCAL_FORWARD_BUDGET_EXCEEDED")

        def check():
            if getattr(self, "budget", None) is not None and self.budget.violation is not None:
                raise ResearchSafetyError(self.budget.violation)

        self.budget = SimpleNamespace(
            limit={"max_per_query_seconds": 600},
            charge=charge,
            check=check,
            checkpoint=lambda _: None,
            violation=None,
        )
        manifest_path = self.common.B2 / "model/frozen_model_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf8"))
        self.models = None
        self.runtime, self.model_manifest = runtime, manifest
        self.np, self.client = np, QdrantClient(":memory:")
        self.substrates = {}
        if build:
            if self.path.exists():
                raise ResearchSafetyError("NATIVE_FULL_INDEX_ALREADY_EXISTS", 409)
            import pymupdf as fitz
            from PIL import Image

            rows, dense, images = [], {}, {}
            # Prefer compatible frozen vectors. Open SQLite in mode=ro, without
            # starting a Qdrant client on the historical directory (which locks it).
            import pickle
            import sqlite3

            from qdrant_client.local.persistence import CollectionPersistence

            existing_root = self.common.OUT / "indexes"
            frozen = json.loads((existing_root / "index_manifest.json").read_text(encoding="utf8"))
            needed = {chunk.chunk_id for chunk in corpus.rows}
            existing = {}
            for arm_name in ("n1", "m1"):
                sqlite_path = existing_root / "qdrant/collection" / arm_name / "storage.sqlite"
                frozen_file = frozen["files"][sqlite_path.relative_to(self.common.OUT).as_posix()]
                if file_sha(sqlite_path) != frozen_file["SHA256"]:
                    raise CorpusTrustError("PHASE4_FROZEN_INDEX_CHANGED")
                with sqlite3.connect(sqlite_path.as_uri() + "?mode=ro", uri=True) as storage:
                    for ordinal, line in enumerate(
                        (existing_root / arm_name / "payload.jsonl").open(encoding="utf8")
                    ):
                        record = json.loads(line)
                        if record["unit_id"] not in needed:
                            continue
                        encoded_key = CollectionPersistence.encode_key(ordinal)
                        point = pickle.loads(
                            storage.execute(
                                "SELECT point FROM points WHERE id=?", (encoded_key,)
                            ).fetchone()[0]
                        )
                        if point.payload != {"evidence_id": record["unit_id"]}:
                            raise CorpusTrustError("PHASE4_VECTOR_SOURCE_MISMATCH")
                        existing[(arm_name, record["unit_id"])] = record, point.vector
            for chunk in corpus.rows:
                u = corpus.units[chunk.chunk_id]
                local = [
                    v
                    for v in corpus.units.values()
                    if (v["paper_id"], v["page_index"]) == (u["paper_id"], u["page_index"])
                ]
                row = {
                    "evidence_id": u["unit_id"],
                    "unit_id": u["unit_id"],
                    "document_id": u["paper_id"],
                    "page_id": u["paper_id"] + "::page::" + str(u["page_index"]),
                    "modality": builder.TYPE[u["unit_type"]],
                    "text": builder.surrogate(u, local, corpus.anchors),
                    "source_provenance": {
                        "page_start": u["page_number"],
                        "page_end": u["page_number"],
                        "PDF_SHA256": u["pdf_sha256"],
                        "bbox": u["bbox"],
                    },
                    "payload_reference": str(corpus.path) + "#" + u["unit_id"],
                }
                rows.append(row)
                reusable = existing.get(("n1", u["unit_id"]))
                visual = existing.get(("m1", u["unit_id"]))
                if reusable is not None and visual is not None:
                    if (
                        reusable[0]["text"] != row["text"]
                        or reusable[0]["source_provenance"]["PDF_SHA256"] != u["pdf_sha256"]
                    ):
                        raise CorpusTrustError("PHASE4_VECTOR_SOURCE_MISMATCH")
                    dense[u["unit_id"]] = reusable[1]
                    images[u["unit_id"]] = visual[1]
                    print(f"reuse native/CLIP vectors {len(rows)}/{len(corpus.rows)}", flush=True)
                    continue
                dense[u["unit_id"]] = (
                    self._get_models().dense(row["text"], purpose="index_dense").tolist()
                )
                with fitz.open(corpus.paths[u["paper_id"]]) as pdf:
                    page = pdf[u["page_index"]]
                    pix = page.get_pixmap(
                        matrix=fitz.Matrix(2, 2),
                        clip=fitz.Rect(u["bbox"]),
                        colorspace=fitz.csRGB,
                        alpha=False,
                    )
                    image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                images[u["unit_id"]] = (
                    self._get_models().clip_image(image, purpose="index_image").tolist()
                )
                print(f"native full index {len(rows)}/{len(corpus.rows)}", flush=True)
            identity = {
                "bundle_sha256": corpus.sha,
                "profile": "phase4-full-v1",
                "model_manifest_sha256": file_sha(manifest_path),
                "reused_compatible_frozen_vectors": len(
                    [r for r in rows if ("n1", r["unit_id"]) in existing]
                ),
                "source_index_manifest_sha256": file_sha(existing_root / "index_manifest.json"),
                "implementation_sha256": {
                    p.name: file_sha(p)
                    for p in (
                        folder / "retrieval_adapter.py",
                        folder / "model_runtime.py",
                        folder / "index_builder.py",
                    )
                },
            }
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps({"identity": identity, "rows": rows, "dense": dense, "images": images}),
                encoding="utf8",
            )
        self.data = json.loads(self.path.read_text(encoding="utf8"))
        identity = self.data["identity"]
        if (
            identity["bundle_sha256"] != corpus.sha
            or identity["model_manifest_sha256"] != file_sha(manifest_path)
            or any(
                file_sha(folder / name) != sha
                for name, sha in identity["implementation_sha256"].items()
            )
        ):
            raise CorpusTrustError("NATIVE_FULL_INDEX_IDENTITY_INVALID")
        self.sha = file_sha(self.path)
        self.trace = None

    def _get_models(self):
        if self.models is None:
            integrity = json.loads(
                (self.common.OUT / "preflight/model_integrity.json").read_text(encoding="utf8")
            )
            for role in integrity["roles"].values():
                for item in role["files"].values():
                    if file_sha(self.common.OUT / item["path"]) != item["SHA256"]:
                        raise CorpusTrustError("PHASE4_MODEL_WEIGHTS_CHANGED")
            self.models = self.runtime.Models(self.budget, self.model_manifest)
        return self.models

    def _substrate(self, records, image):
        from qdrant_client import models

        name = ("image-" if image else "native-") + digest([r["unit_id"] for r in records])[:16]
        if name not in self.substrates:
            self.client.create_collection(
                name,
                vectors_config=models.VectorParams(
                    size=512 if image else 1024, distance=models.Distance.COSINE
                ),
            )
            vectors = self.data["images" if image else "dense"]
            self.client.upsert(
                name,
                [
                    models.PointStruct(id=i, vector=vectors[r["unit_id"]])
                    for i, r in enumerate(records)
                ],
            )
            self.substrates[name] = self.adapter.Substrate(
                self.client, name, records, None if image else self.adapter.sparse(records)
            )
        return self.substrates[name]

    def search(self, query, arm, eligible, limit):
        self.guard()
        rows = [r for r in self.data["rows"] if r["unit_id"] in eligible]
        if not rows:
            return [], {"dense": [], "sparse": [], **({"image": []} if arm == "M1" else {})}
        retriever = self.adapter.Retriever(
            arm,
            self._substrate(rows, False),
            self._get_models(),
            self.budget,
            self._substrate(rows, True) if arm == "M1" else None,
        )
        hits = retriever.retrieve(query, top_k=5)
        self.trace = self.adapter.trace_json(retriever.last_trace)
        if self.budget.violation is not None:
            raise ResearchSafetyError(self.budget.violation)
        return [(h.unit_id, h.score) for h in hits[:limit]], retriever.last_trace["channels"]

    def guard(self):
        if file_sha(self.path) != self.sha:
            raise CorpusTrustError("NATIVE_FULL_INDEX_STALE")

    def close(self):
        self.client.close()


class LocalClipIndex:
    """Small isolated index using the existing Phase4 CLIP model and crop settings."""

    def __init__(self, corpus, model_path: Path, index_path: Path):
        self.corpus, self.model_path, self.index_path = (
            corpus,
            model_path.resolve(),
            index_path.resolve(),
        )
        self.data = json.loads(self.index_path.read_text(encoding="utf8"))
        self.sha = file_sha(self.index_path)
        self.model = None
        self.guard()

    def guard(self):
        if self.data["bundle_sha256"] != self.corpus.sha or file_sha(self.index_path) != self.sha:
            raise CorpusTrustError("CLIP_INDEX_STALE")
        if any(
            file_sha(self.model_path / path) != sha
            for path, sha in self.data["model_files"].items()
        ):
            raise CorpusTrustError("CLIP_MODEL_STALE")

    def _load(self):
        if self.model is None:
            import torch
            from transformers import CLIPModel, CLIPTokenizerFast

            torch.set_num_threads(4)
            self.torch = torch
            self.model = CLIPModel.from_pretrained(self.model_path, local_files_only=True).eval()
            self.tokenizer = CLIPTokenizerFast.from_pretrained(
                self.model_path, local_files_only=True
            )

    def rank(self, query, eligible):
        self.guard()
        self._load()
        with self.torch.inference_mode():
            encoded = self.tokenizer([query], return_tensors="pt", truncation=True, max_length=77)
            vector = self.model.get_text_features(**encoded, return_dict=True).pooler_output
            vector = vector / vector.norm(dim=-1, keepdim=True)
            scores = [
                (key, float((vector[0] * self.torch.tensor(self.data["vectors"][key])).sum()))
                for key in eligible
            ]
        if any(not math.isfinite(score) for _, score in scores):
            raise CorpusTrustError("CLIP_SCORE_INVALID")
        return [key for key, _ in sorted(scores, key=lambda x: (-x[1], x[0]))]


class MultimodalResearchProvider:
    """Keep Hybrid RAG unchanged and add native retrieval under the same D6 lease."""

    def __init__(self, text_provider, corpus):
        self.text = text_provider
        self.corpus = corpus

    def __getattr__(self, key):
        return getattr(self.text, key)

    def _text_state(self, state):
        clone = state.model_copy(deep=True)
        clone.evidence_state.items = {
            key: item
            for key, item in clone.evidence_state.items.items()
            if not isinstance(item, NativeEvidenceItem)
        }
        return clone

    def _sources(self, binding):
        authority = self.text._registry.authority_bundle(binding["generation_id"])
        if authority["inventory"]["inventory_digest"] != binding["inventory_digest"]:
            raise CorpusTrustError("NATIVE_GENERATION_INVALID")
        return {doc["paper_id"]: doc["pdf_sha256"] for doc in authority["commitment"]["documents"]}

    def validate_active_snapshot(self, state):
        self.text.validate_active_snapshot(self._text_state(state))
        with self.text._admission():
            binding, binding_sha = self.text._active_binding(state.paper_ids)
            source_hashes = self._sources(binding)
            self.corpus.guard()
            for key, item in state.evidence_state.items.items():
                if not isinstance(item, NativeEvidenceItem):
                    continue
                try:
                    NativeEvidenceItem.model_validate(item.model_dump())
                except ValueError as exc:
                    raise CorpusTrustError("NATIVE_PROVENANCE_INVALID") from exc
                unit = self.corpus.units.get(item.unit_id)
                if (
                    not state.multimodal_enabled
                    or key != item.stable_key
                    or unit is None
                    or digest(unit) != item.native_unit_sha256
                    or item.bundle_sha256 != self.corpus.sha
                    or item.generation_binding_digest != binding_sha
                    or source_hashes.get(item.paper_id) != item.pdf_sha256
                    or item.native_paper_id != unit["paper_id"]
                    or item.source_pdf != str(self.corpus.paths[unit["paper_id"]].resolve())
                    or item.readable_text != self.corpus.readable(unit)
                    or item.retrieval_identity
                    != state.evidence_provenance["p1_active_canonical_provenance_v2"][
                        "retrieval_identity"
                    ]
                ):
                    raise CorpusTrustError("NATIVE_PROVENANCE_INVALID")
                for occurrence in item.occurrences:
                    if (
                        occurrence.task_id != state.task_id
                        or occurrence.paper_scope != (state.paper_ids or [])
                        or occurrence.generation_binding_digest != binding_sha
                    ):
                        raise CorpusTrustError("NATIVE_OCCURRENCE_INVALID")
                    if occurrence.native_profile == "phase4-full-v1" and (
                        self.corpus.full_backend is None
                        or occurrence.native_index_sha256 != self.corpus.full_backend.sha
                    ):
                        raise CorpusTrustError("NATIVE_FULL_INDEX_IDENTITY_INVALID")
                    if occurrence.native_profile == "phase4-full-v1":
                        self.corpus.full_backend.guard()
                    if ":same-page-context:" in occurrence.trace_id:
                        prefix, parent_id = occurrence.trace_id.split(":same-page-context:", 1)
                        parent = self.corpus.units.get(parent_id)
                        if (prefix != occurrence.tool_call_id or parent is None
                            or parent_id == item.unit_id
                            or any(parent[k] != unit[k] for k in
                                   ("paper_id", "pdf_sha256", "page_index", "unit_type"))
                            or parent_id not in {uid for ids in occurrence.channels.values() for uid in ids}):
                            raise CorpusTrustError("NATIVE_CONTEXT_SOURCE_INVALID")

    def retrieve_active_observation(self, query, paper_ids, limit, tool_call_id, state, **options):
        self.validate_active_snapshot(state)
        return self.text.retrieve_active_observation(
            query, paper_ids, limit, tool_call_id, self._text_state(state), **options
        )

    def retrieve_native(self, state, action, tool_call_id):
        if not state.multimodal_enabled:
            raise ResearchSafetyError("MULTIMODAL_OPT_IN_REQUIRED", 422)
        self.validate_active_snapshot(state)
        with self.text._admission():
            binding, binding_sha = self.text._active_binding(state.paper_ids)
            if (
                binding_sha
                != state.evidence_provenance["p1_active_canonical_provenance_v2"][
                    "generation_binding_digest"
                ]
            ):
                raise CorpusTrustError("NATIVE_GENERATION_STALE")
            sources = self._sources(binding)
            scope = action.arguments.get("paper_ids")
            if scope is None:
                scope = state.paper_ids or list(sources)
            if (
                not isinstance(scope, list)
                or not scope
                or any(not isinstance(paper, str) for paper in scope)
                or len(set(scope)) != len(scope)
                or not set(scope) <= set(state.paper_ids or sources)
            ):
                raise ResearchSafetyError("NATIVE_SCOPE_INVALID", 422)
            hashes = {sources[paper]: paper for paper in scope}
            query = str(action.arguments.get("query") or state.research_question)
            arm = action.arguments.get("arm", "N1")
            hits, channels = self.corpus.search(
                query,
                action.arguments.get("modality"),
                arm,
                hashes,
                min(5, max(1, int(action.arguments.get("top_k", 3)))),
            )
            items = []
            for rank, (uid, score) in enumerate(hits, 1):
                u = self.corpus.units[uid]
                paper = hashes[u["pdf_sha256"]]
                text = self.corpus.readable(u)
                kind = {
                    "FIGURE": "caption_only",
                    "TABLE": "table_structure",
                    "EQUATION": "native_equation_candidate_text",
                }[action.arguments["modality"]]
                items.append(
                    NativeEvidenceItem(
                        evidence_id=f"{paper}:{uid}",
                        paper_id=paper,
                        block_id=uid,
                        page=u["page_number"],
                        page_end=u["page_number"],
                        text_or_reference=text[:800],
                        source_type="pdf_native",
                        discovered_by_tool=tool_call_id,
                        discovered_at_step=state.step_count,
                        relevance=score,
                        unit_id=uid,
                        modality=action.arguments["modality"],
                        source_pdf=str(self.corpus.paths[u["paper_id"]].resolve()),
                        pdf_sha256=u["pdf_sha256"],
                        native_paper_id=u["paper_id"],
                        native_unit=u,
                        native_unit_sha256=digest(u),
                        readable_kind=kind,
                        readable_text=text,
                        bundle_sha256=self.corpus.sha,
                        retrieval_identity=state.evidence_provenance[
                            "p1_active_canonical_provenance_v2"
                        ]["retrieval_identity"],
                        generation_binding_digest=binding_sha,
                        occurrences=[
                            {
                                "task_id": state.task_id,
                                "tool_call_id": tool_call_id,
                                "trace_id": (tool_call_id + ":same-page-context:"
                                    + self.corpus.context_parents[uid]
                                    if uid in self.corpus.context_parents else tool_call_id),
                                "step": state.step_count,
                                "query": query,
                                "paper_scope": state.paper_ids or [],
                                "target_subquestion_ids": action.target_subquestion_ids,
                                "generation_binding_digest": binding_sha,
                                "arm": arm,
                                "channels": channels,
                                "rank": rank,
                                "native_profile": "phase4-full-v1"
                                if self.corpus.full_backend is not None
                                else "agent-sparse-adapter-v1",
                                "native_index_sha256": self.corpus.full_backend.sha
                                if self.corpus.full_backend is not None
                                else None,
                            }
                        ],
                    )
                )
            return items

    def draft_evidence(self, state):
        self.validate_active_snapshot(state)
        result = self.text.draft_evidence(self._text_state(state))
        result.extend(
            item.model_dump(mode="json")
            for item in state.evidence_state.items.values()
            if isinstance(item, NativeEvidenceItem)
        )
        return result

    def visual_evidence(self, state, sources, asset_dir):
        """Render only admitted FIGURE units; no caller-supplied paths/bboxes."""
        import pymupdf as fitz

        self.validate_active_snapshot(state)
        images = []
        with self.text._admission():
            _, binding_sha = self.text._active_binding(state.paper_ids)
            if binding_sha != state.evidence_provenance[
                "p1_active_canonical_provenance_v2"]["generation_binding_digest"]:
                raise CorpusTrustError("VISION_GENERATION_STALE")
            for source in sources:
                if source.get("modality") != "FIGURE":
                    continue
                item = next((e for e in state.evidence_state.items.values()
                             if e.evidence_id == source["canonical_evidence_id"]), None)
                if (not isinstance(item, NativeEvidenceItem) or item.modality != "FIGURE"
                    or not re.fullmatch(r"E\d+", source["evidence_id"]) or any(
                    source.get(k) != getattr(item, k) for k in
                    ("paper_id", "page", "unit_id", "pdf_sha256", "native_unit_sha256",
                     "generation_binding_digest", "source_pdf")
                ) or digest(source["native_unit"]) != item.native_unit_sha256):
                    raise CorpusTrustError("VISION_SOURCE_IDENTITY_INVALID")
                path = Path(item.source_pdf)
                if file_sha(path) != item.pdf_sha256:
                    raise CorpusTrustError("VISION_PDF_CHANGED")
                with fitz.open(path) as pdf:
                    page = pdf[item.page - 1]
                    box = [float(v) for v in item.native_unit["bbox"]]
                    rect = fitz.Rect(box)
                    if page.rotation != 0 or rect.is_empty or not page.rect.contains(rect):
                        raise ResearchSafetyError("VISION_GEOMETRY_UNSUPPORTED", 422)
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=rect,
                                          colorspace=fitz.csRGB, alpha=False)
                    raw = pix.tobytes("png")
                    width, height = pix.width, pix.height
                    render_bbox = [pix.x / 2, pix.y / 2,
                                   (pix.x + width) / 2, (pix.y + height) / 2]
                if file_sha(path) != item.pdf_sha256:
                    raise CorpusTrustError("VISION_PDF_CHANGED")
                import hashlib
                image_sha = hashlib.sha256(raw).hexdigest()
                asset = Path(asset_dir) / (source["evidence_id"] + "-" + image_sha[:16] + ".png")
                asset.parent.mkdir(parents=True, exist_ok=True)
                if asset.exists() and asset.read_bytes() != raw:
                    raise ResearchSafetyError("VISION_ASSET_CHANGED", 409)
                if not asset.exists():
                    asset.write_bytes(raw)
                images.append({"evidence_id": source["evidence_id"],
                    "canonical_evidence_id": item.evidence_id, "paper_id": item.paper_id,
                    "pdf_sha256": item.pdf_sha256, "source_pdf": str(path.resolve()),
                    "page": item.page, "bbox": box, "render_bbox": render_bbox,
                    "unit_id": item.unit_id,
                    "native_unit_sha256": item.native_unit_sha256,
                    "generation_binding_digest": binding_sha,
                    "retrieval_identity": item.retrieval_identity,
                    "asset_path": str(asset.resolve()), "sha256": image_sha,
                    "width": width, "height": height,
                    "data_url": "data:image/png;base64," + base64.b64encode(raw).decode("ascii")})
        if not images:
            raise ResearchSafetyError("VISION_NO_NATIVE_FIGURE", 422)
        return images


def configured_multimodal_provider(provider, settings):
    if settings.research_native_bundle is None or settings.research_native_full_index is None:
        raise ResearchSafetyError("NATIVE_FULL_INDEX_UNCONFIGURED", 422)
    corpus = NativeCorpus(settings.research_native_bundle)
    corpus.full_backend = FullPhase4NativeBackend(corpus, settings.research_native_full_index)
    return MultimodalResearchProvider(provider, corpus)
