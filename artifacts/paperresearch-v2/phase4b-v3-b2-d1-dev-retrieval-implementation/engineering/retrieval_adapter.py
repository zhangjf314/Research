"""One gold-free adapter for frozen T0/N1/M1; exact local Qdrant and BM25."""
from dataclasses import asdict
import math
import time

import common as c

contract = c.inherited_contract()


class Substrate:
    def __init__(self, client, name, records, sparse=None):
        self.client, self.name, self.records, self.sparse = client, name, records, sparse
        self.by_id = {r['evidence_id']: r for r in records}
        assert len(self.by_id) == len(records)

    def dense(self, vector):
        # Local Qdrant is exhaustive. Read all scores to enforce global boundary
        # ties by exact identity, rather than accepting its point-ID tie order.
        points = self.client.query_points(self.name, query=vector.tolist(), limit=len(self.records),
            with_payload=False, with_vectors=False).points
        assert len(points) == len(self.records) and all(math.isfinite(p.score) for p in points)
        ranked = [(self.records[int(p.id)]['evidence_id'], float(p.score)) for p in points]
        return [key for key, score in sorted(ranked, key=lambda v: (-v[1], v[0]))[:20]]

    def lexical(self, query):
        # Existing BM25 formulation/tokenizer; ask for all positive scores so
        # channel-depth boundary ties can be resolved by the frozen exact ID.
        items = self.sparse.retrieve(query, top_k=len(self.records))
        return [x.chunk.chunk_id for x in sorted(items, key=lambda x: (-x.score, x.chunk.chunk_id))[:20]]


class Retriever:
    def __init__(self, arm, text, models, budget, images=None):
        assert arm in ('T0', 'N1', 'M1') and (arm == 'M1') == (images is not None)
        self.arm, self.text, self.models, self.budget, self.images = arm, text, models, budget, images
        self.last_trace = None

    def retrieve(self, query: str, top_k: int = 5) -> list[contract.EvidenceHit]:
        assert isinstance(query, str) and query and top_k == 5
        start = time.perf_counter()
        self.budget.charge('maximum_query_requests')
        channels = {'dense': self.text.dense(self.models.dense(query, purpose='query_dense')),
                    'sparse': self.text.lexical(query)}
        if self.arm == 'M1':
            channels['image'] = self.images.dense(self.models.clip_text(query, purpose='query_image_text'))
        fused = contract.reciprocal_rank_fusion(channels, {name: 1 for name in channels})
        reranked = []
        for pre_rank, (identifier, fusion_score) in enumerate(fused, 1):
            row = self.text.by_id[identifier]
            score = self.models.rerank(query, row['text'], purpose='rerank')
            assert math.isfinite(score)
            reranked.append((score, pre_rank, identifier, fusion_score))
            if time.perf_counter() - start > self.budget.limit['max_per_query_seconds']:
                self.budget.violation = 'PER_QUERY_SECONDS_EXCEEDED'
                self.budget.check()
        reranked.sort(key=lambda v: (-v[0], v[1], v[2]))
        pool = [contract.EvidenceHit(rank=i, score=score, document_id=self.text.by_id[key]['document_id'],
            page_id=self.text.by_id[key]['page_id'], unit_id=self.text.by_id[key]['unit_id'],
            modality=self.text.by_id[key]['modality'], source_provenance=self.text.by_id[key]['source_provenance'],
            payload_reference=self.text.by_id[key]['payload_reference'], retriever_arm=self.arm)
            for i, (score, pre_rank, key, fusion_score) in enumerate(reranked, 1)]
        output = contract.distinct_pages(pool, 5)
        elapsed = time.perf_counter() - start
        if elapsed > self.budget.limit['max_per_query_seconds']:
            self.budget.violation = 'PER_QUERY_SECONDS_EXCEEDED'
        self.budget.check()
        self.last_trace = {'channels': channels, 'fused': fused, 'rerank_order': reranked,
            'pool': pool, 'final': output, 'seconds': elapsed, 'score_cache_hits': 0}
        return output


def sparse(records):
    from paper_research.chunking.types import Chunk
    from paper_research.retrieval.sparse import BM25Retriever
    chunks = [Chunk(chunk_id=r['evidence_id'], paper_id=r['document_id'], block_ids=[], block_type='paragraph',
        page_start=r['source_provenance']['page_start'], page_end=r['source_provenance']['page_end'],
        chunk_text=r['text'], token_count=0) for r in records]
    return BM25Retriever(chunks, k1=1.5, b=.75, contextual=False)


def trace_json(trace):
    return {**trace, 'pool': [asdict(h) for h in trace['pool']], 'final': [asdict(h) for h in trace['final']]}
