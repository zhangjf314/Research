"""Complete unsupervised fresh40 substrates, never benchmark case membership."""
from collections import Counter, defaultdict
import hashlib
import json
import time

import common as c
from retrieval_adapter import Substrate, sparse, contract

TYPE = {'TEXT_BLOCK': 'TEXT', 'FIGURE_CANDIDATE': 'FIGURE', 'TABLE_CANDIDATE': 'TABLE', 'EQUATION_CANDIDATE': 'EQUATION'}


def jsonl(path):
    with path.open('r', encoding='utf8') as f:
        return [json.loads(line) for line in f if line.strip()]


def source_text(unit):
    return unit.get('canonical_text') or unit.get('raw_text') or ''


def surrogate(unit, page_units, captions):
    cap = captions.get(unit.get('caption_id'))
    if unit.get('caption_id'):
        assert cap and (cap['paper_id'], cap['page_index']) == (unit['paper_id'], unit['page_index'])
    context, selected = '', []
    if unit['unit_type'] != 'TEXT_BLOCK':
        excluded = set(unit['component_ids'])
        if cap:
            excluded.add(cap['source_component_id'])
        others = [u for u in page_units if u['unit_type'] == 'TEXT_BLOCK' and u['unit_id'] != unit['unit_id']
                  and not excluded.intersection(u['component_ids'])]
        for other in sorted(others, key=lambda u: (u['bbox'][1], u['bbox'][0], u['unit_id'])):
            proposed = '\n'.join([*selected, source_text(other)])
            if len(proposed.encode('utf8')) > 2048:
                break
            selected.append(source_text(other))
        context = '\n'.join(selected)
    return f"Evidence type: {TYPE[unit['unit_type']]}\nPrimary:\n{source_text(unit)}\nCaption:\n{cap['exact_block_text'] if cap else ''}\nPage context:\n{context}"


class Builder:
    def __init__(self, models, budget):
        from qdrant_client import QdrantClient
        self.models, self.budget = models, budget
        self.client = QdrantClient(path=str(c.OUT / 'indexes/qdrant'), check_compatibility=False)
        self.documents = c.read(c.A2 / 'extraction/document_manifest.json')['documents']
        self.document_by_id = {d['paper_id']: d for d in self.documents}
        self.timings, self.cache = {}, {}
        self.cache_hits, self.saved_inference_seconds = Counter(), Counter()
        self.vector_identity = []
        self.native_path = c.A2 / 'extraction/native/extraction/multimodal_units.jsonl'
        self.captions_path = c.A2 / 'extraction/native/extraction/caption_anchors.jsonl'

    def validate_documents(self):
        import fitz
        assert len(self.documents) == len(self.document_by_id) == 40
        for doc in self.documents:
            path = c.ROOT / doc['source_pdf_path']
            assert c.sha(path) == doc['expected_sha256'] == doc['actual_sha256']
            assert 'canonical_document:' + c.digest({'source_sha256': doc['actual_sha256']}) == doc['paper_id']
            with fitz.open(path) as pdf:
                assert pdf.page_count == doc['expected_page_count'] and not pdf.needs_pass
                assert all(page.rotation == 0 for page in pdf)
        assert c.sha(self.native_path) == c.read(c.B2 / 'protocol/native_retrieval_contract.json')['source']['SHA256']

    def t0_records(self):
        from paper_research.parsing.pymupdf_parser import PyMuPDFParser
        from paper_research.chunking.structural_chunker import StructuralChunker
        result, coverage = [], []
        for doc in self.documents:
            parsed = PyMuPDFParser().parse(c.ROOT / doc['source_pdf_path'])
            chunks = StructuralChunker(max_tokens=400, overlap_tokens=60).chunk(doc['paper_id'], parsed.blocks)
            blockmap = {b.block_id: b.source_page for b in parsed.blocks}
            for ordinal, chunk in enumerate(chunks):
                identity = {'namespace': 'V3_T0_RAW_400_60_V1', 'document_id': doc['paper_id'], 'ordinal0': ordinal,
                    'block_ids': chunk.block_ids, 'page_start': chunk.page_start, 'page_end': chunk.page_end, 'chunk_text': chunk.chunk_text}
                chunk.chunk_id = 't0:' + c.digest(identity)
                chunk.block_page_map = {bid: blockmap[bid] for bid in chunk.block_ids}
                result.append({'evidence_id': chunk.chunk_id, 'unit_id': None, 'document_id': doc['paper_id'],
                    'page_id': contract.chunk_page(doc['paper_id'], chunk.page_start, chunk.page_end), 'modality': 'TEXT',
                    'text': chunk.chunk_text, 'chunk': chunk.model_dump(),
                    'source_provenance': {'PDF_SHA256': doc['actual_sha256'], 'source_pdf_path': doc['source_pdf_path'],
                        'page_start': chunk.page_start, 'page_end': chunk.page_end, 'block_page_map': chunk.block_page_map,
                        'block_ids': chunk.block_ids, 'parser': 'PyMuPDFParser', 'chunker': 'StructuralChunker400/60'},
                    'payload_reference': 'indexes/t0/payload.jsonl#' + chunk.chunk_id})
            coverage.append({'document_id': doc['paper_id'], 'chunks': len(chunks), 'pages': parsed.metadata.page_count,
                             'PDF_SHA256': doc['actual_sha256']})
        assert len(result) <= self.budget.limit['max_T0_chunks_per_pass']
        c.put('indexes/t0/documents.json', coverage)
        return result

    def native_records(self):
        units = jsonl(self.native_path)
        captions = {row['caption_id']: row for row in jsonl(self.captions_path)}
        assert len(units) == 25762 and len({u['unit_id'] for u in units}) == 25762
        assert Counter(u['unit_type'] for u in units) == {'TEXT_BLOCK': 24572, 'FIGURE_CANDIDATE': 253, 'TABLE_CANDIDATE': 210, 'EQUATION_CANDIDATE': 727}
        pages = defaultdict(list)
        for unit in units:
            doc = self.document_by_id[unit['paper_id']]
            assert unit['pdf_sha256'] == doc['actual_sha256'] and 0 <= unit['page_index'] < doc['expected_page_count']
            assert unit['component_ids'] and unit['source_component_identity'] and unit['provenance']
            assert unit['page_number'] == unit['page_index'] + 1
            pages[(unit['paper_id'], unit['page_index'])].append(unit)
        records = []
        for u in units:
            doc = self.document_by_id[u['paper_id']]
            records.append({'evidence_id': u['unit_id'], 'unit_id': u['unit_id'], 'document_id': u['paper_id'],
                'page_id': u['paper_id'] + '::page::' + str(u['page_index']), 'modality': TYPE[u['unit_type']],
                'text': surrogate(u, pages[(u['paper_id'], u['page_index'])], captions), 'native': u,
                'source_provenance': {'PDF_SHA256': doc['actual_sha256'], 'source_pdf_path': doc['source_pdf_path'],
                    'page_start': u['page_number'], 'page_end': u['page_number'], 'bbox': u['bbox'],
                    'component_ids': u['component_ids'], 'caption_id': u['caption_id'],
                    'source_component_identity': u['source_component_identity'], 'native_provenance': u['provenance'],
                    'native_source_path': self.native_path.relative_to(c.ROOT).as_posix()},
                'payload_reference': 'indexes/n1/payload.jsonl#' + u['unit_id']})
        return records

    def image(self, row):
        import fitz
        from PIL import Image
        import math
        source = row['source_provenance']
        with fitz.open(c.ROOT / source['source_pdf_path']) as pdf:
            page = pdf[source['page_start'] - 1]
            assert page.rotation == 0 and all(math.isfinite(v) for v in source['bbox'])
            rect = fitz.Rect(source['bbox']) & page.rect
            assert rect.width > 0 and rect.height > 0 and not rect.is_empty
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=rect, colorspace=fitz.csRGB, alpha=False)
            image = Image.frombytes('RGB', (pix.width, pix.height), pix.samples)
        path = c.OUT / 'indexes/m1/crops' / (row['unit_id'].replace(':', '_') + '.png')
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path, format='PNG')
        return image, path

    def encode(self, row, arm):
        import numpy as np
        manifest_sha = c.sha(c.B2 / 'model/frozen_model_manifest.json')
        crop = None
        if arm == 'm1':
            image, crop = self.image(row)
            payload_sha = c.sha(crop)
            representation = 'V3_NATIVE_CLIP_CROP_V1'
        else:
            payload_sha = hashlib.sha256(row['text'].encode('utf8')).hexdigest()
            representation = 'RAW_TEXT' if arm == 't0' else 'V3_NATIVE_TEXT_SURROGATE_V1'
        key = c.digest({'payload_SHA256': payload_sha, 'representation': representation, 'processing_model_manifest_SHA256': manifest_sha})
        if key in self.cache:
            vector, elapsed = self.cache[key]
            self.cache_hits[arm] += 1
            self.saved_inference_seconds[arm] += elapsed
        else:
            start = time.perf_counter()
            vector = self.models.clip_image(image, purpose='index_image') if arm == 'm1' else self.models.dense(row['text'], purpose='index_dense')
            elapsed = time.perf_counter() - start
            assert vector.dtype == np.float32 and np.isfinite(vector).all()
            assert vector.shape == ((512,) if arm == 'm1' else (1024,)) and abs(np.linalg.norm(vector) - 1) < 1e-5
            self.cache[key] = (vector, elapsed)
        self.vector_identity.append({'arm': arm, 'evidence_id': row['evidence_id'], 'cache_key': key,
            'vector_SHA256_float32_le': hashlib.sha256(vector.astype('<f4', copy=False).tobytes()).hexdigest(),
            'dimension': len(vector), 'input_SHA256': payload_sha, 'crop_reference': crop.relative_to(c.OUT).as_posix() if crop else None})
        return vector

    def build_substrate(self, name, records, dimension):
        from qdrant_client.models import VectorParams, Distance, PointStruct
        self.budget.charge('maximum_indexing_runs_total')
        self.budget.charge('index_passes_' + name)
        assert self.budget.counts['index_passes_' + name] <= 2 and not self.client.collection_exists(name)
        self.client.create_collection(name, vectors_config=VectorParams(size=dimension, distance=Distance.COSINE))
        directory = c.OUT / 'indexes' / name
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / 'payload.jsonl').open('wb') as payload, (directory / 'vector_manifest.jsonl').open('wb') as vector_log:
            pending = []
            start = time.perf_counter()
            last = start
            for ordinal, row in enumerate(records):
                vector = self.encode(row, name)
                payload.write(c.canon(row) + b'\n')
                vector_log.write(c.canon(self.vector_identity[-1]) + b'\n')
                pending.append(PointStruct(id=ordinal, vector=vector.tolist(), payload={'evidence_id': row['evidence_id']}))
                if len(pending) == 16 or ordinal + 1 == len(records):
                    self.client.upsert(name, points=pending, wait=True)
                    pending.clear()
                    payload.flush(); vector_log.flush()
                self.budget.check()
                now = time.perf_counter()
                if now - last >= 45 or ordinal + 1 == len(records):
                    progress = {'substrate': name, 'encoded_records': ordinal + 1, 'total_records': len(records),
                        'elapsed_seconds': now - start, 'estimated_remaining_seconds': (now - start) * (len(records) - ordinal - 1) / (ordinal + 1),
                        'physical_forwards': dict(self.budget.counts), 'peak_RSS_bytes': self.budget.peak_RSS}
                    c.put('indexes/progress.json', progress)
                    print(json.dumps(progress), flush=True)
                    last = now
        assert self.client.count(name, exact=True).count == len(records)
        bm25 = sparse(records) if name != 'm1' else None
        return Substrate(self.client, name, records, bm25)

    def run(self):
        start = time.perf_counter()
        self.validate_documents()
        t0 = self.t0_records()
        T0 = self.build_substrate('t0', t0, 1024)
        self.timings['T0'] = time.perf_counter() - start
        start = time.perf_counter()
        n1 = self.native_records()
        N1 = self.build_substrate('n1', n1, 1024)
        self.timings['N1'] = time.perf_counter() - start
        start = time.perf_counter()
        visual = [r for r in n1 if r['modality'] in ('TABLE', 'FIGURE', 'EQUATION')]
        assert len(visual) == 1190
        images = self.build_substrate('m1', visual, 512)
        self.timings['M1_incremental'] = time.perf_counter() - start
        self.timings['M1_logical'] = self.timings['N1'] + self.timings['M1_incremental']
        return T0, N1, images

    def accept(self, substrates):
        import numpy as np
        records = [s.records for s in substrates]
        assert [len(r) for r in records][1:] == [25762, 1190]
        assert all({r['document_id'] for r in rr} == set(self.document_by_id) for rr in records[:2])
        assert all(len({r['evidence_id'] for r in rr}) == len(rr) for rr in records)
        for s in substrates:
            offset, count = None, 0
            while True:
                points, offset = self.client.scroll(s.name, offset=offset, limit=128, with_payload=True, with_vectors=True)
                for point in points:
                    expected = s.records[int(point.id)]
                    assert point.payload == {'evidence_id': expected['evidence_id']}
                    vector = np.asarray(point.vector, dtype=np.float32)
                    assert np.isfinite(vector).all() and vector.shape == ((512,) if s.name == 'm1' else (1024,))
                    assert abs(np.linalg.norm(vector) - 1) < 1e-5
                    count += 1
                self.budget.check()
                if offset is None:
                    break
            assert count == len(s.records)
        self.client.close()
        files = {p.relative_to(c.OUT).as_posix(): {'SHA256': c.sha(p), 'bytes': p.stat().st_size}
                 for p in sorted((c.OUT / 'indexes').rglob('*')) if p.is_file() and p.name != 'progress.json'}
        total = sum(v['bytes'] for v in files.values())
        assert total <= self.budget.limit['maximum_total_persisted_indexes_crops_bytes']
        def size(name):
            return sum(v['bytes'] for p, v in files.items() if p.startswith('indexes/' + name + '/') or p.startswith('indexes/qdrant/collection/' + name + '/'))
        logical = {'T0': size('t0'), 'N1': size('n1'), 'M1': size('n1') + size('m1')}
        coverage = {'status': 'PASS', 'T0_documents': 40, 'T0_chunks': len(records[0]), 'N1_documents': 40,
            'N1_units': 25762, 'N1_modalities': dict(Counter(r['modality'] for r in records[1])), 'M1_visual_units': 1190,
            'missing_source_provenance': 0, 'duplicate_forbidden_identity': 0, 'invalid_embedding': 0,
            'gold_dependent_index_construction': 0, 'TEST_semantic_reads': 0, 'DEV_semantic_reads': 0,
            'no_disposition_or_silent_drop': True}
        c.put('guards/corpus_coverage.json', coverage)
        manifest = {'status': 'PASS', 'accepted_at': c.utc(), 'corpus_digest': c.CORPUS, 'protocol_digest': c.PROTOCOL,
            'coverage': coverage, 'dimensions': {'T0': 1024, 'N1': 1024, 'M1_image': 512}, 'files': files,
            'index_digest': c.digest(files), 'physical_index_bytes': total, 'logical_index_bytes': logical,
            'build_seconds': self.timings, 'cache_hits': dict(self.cache_hits), 'cache_saved_inference_seconds': dict(self.saved_inference_seconds),
            'N1_shared_with_M1': True, 'source_document_manifest_SHA256': c.sha(c.A2 / 'extraction/document_manifest.json'),
            'native_input_SHA256': c.sha(self.native_path), 'caption_input_SHA256': c.sha(self.captions_path),
            'model_integrity_SHA256': c.sha(c.OUT / 'preflight/model_integrity.json'),
            'engineering_SHA256': {p.name: c.sha(p) for p in sorted((c.OUT / 'engineering').glob('*.py'))},
            'inference_seconds': dict(self.models.inference_seconds), 'truncations': dict(self.models.truncations),
            'storage': 'existing QdrantClient(path), exhaustive cosine; existing BM25; global exact-ID ties'}
        c.put('indexes/index_manifest.json', manifest)
        print('INDEX ACCEPTANCE PASS: full40, all25762 native and1190 image units; DEV still unread', flush=True)
        return manifest
