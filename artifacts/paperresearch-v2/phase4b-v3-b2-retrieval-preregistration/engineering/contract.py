"""Frozen pure B2 adapters/statistics. No index, models, providers or benchmark IO."""
from dataclasses import dataclass, replace
import math
import random
from statistics import mean
from typing import Protocol

MM = ('TABLE', 'FIGURE', 'EQUATION')
SEED = 2026100801
REPETITIONS = 10000
TEXT_TOLERANCE = 0.10


@dataclass(frozen=True)
class EvidenceHit:
    rank: int
    score: float
    document_id: str
    page_id: str
    unit_id: str | None
    modality: str
    source_provenance: dict
    payload_reference: str
    retriever_arm: str


class Retriever(Protocol):
    def retrieve(self, query: str, top_k: int) -> list[EvidenceHit]: ...


def chunk_page(document_id, page_start, page_end):
    """A cross-page chunk earns just its earliest source page, before gold exists."""
    if not isinstance(page_start, int) or not isinstance(page_end, int) or not 1 <= page_start <= page_end:
        raise ValueError('invalid 1-based source pages')
    return document_id + '::page::' + str(page_start - 1)


def distinct_pages(ranked, k=5):
    if k not in (5, 10):
        raise ValueError('only preregistered depths 5/10')
    seen, output = set(), []
    for hit in ranked[:30]:
        if not hit.page_id.startswith(hit.document_id + '::page::') or not math.isfinite(hit.score):
            raise ValueError('invalid hit identity/score')
        page_number = hit.page_id.removeprefix(hit.document_id + '::page::')
        if not page_number.isdecimal() or str(int(page_number)) != page_number:
            raise ValueError('noncanonical page ID')
        if hit.page_id not in seen:
            seen.add(hit.page_id)
            output.append(replace(hit, rank=len(output) + 1))
        if len(output) == k:
            break
    return output


def reciprocal_rank_fusion(channels, weights):
    if set(channels) != set(weights) or any(v <= 0 for v in weights.values()):
        raise ValueError('invalid channel weights')
    scores = {}
    for name in sorted(channels):
        identifiers = channels[name]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError('duplicate within channel')
        for rank, identifier in enumerate(identifiers[:20], 1):
            scores[identifier] = scores.get(identifier, 0.0) + weights[name] / (60 + rank)
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:30]


def page_metrics(ranked, gold_primary_page):
    pages = [h.page_id for h in distinct_pages(ranked, 10)]
    rank = pages.index(gold_primary_page) + 1 if gold_primary_page in pages else None
    within5 = rank is not None and rank <= 5
    return {'gold_source_page_recall_at_5': float(within5),
            'page_mrr_at_5': 1.0 / rank if within5 else 0.0,
            'page_ndcg_at_5': 1.0 / math.log2(rank + 1) if within5 else 0.0,
            'gold_source_page_recall_at_10': float(rank is not None)}


def unit_recall(ranked, gold_primary_unit, arm):
    # T0's V3 chunk construction deliberately has no exact native-unit mapping.
    if arm == 'T0':
        return 'NOT_APPLICABLE'
    if arm not in ('N1', 'M1') or not gold_primary_unit:
        raise ValueError('invalid native metric inputs')
    hits = distinct_pages(ranked, 5)
    if any(h.retriever_arm != arm or not h.unit_id for h in hits):
        raise ValueError('native hits must retain exact provenance')
    return float(any(h.unit_id == gold_primary_unit for h in hits))


def legacy_evidence_recall(ranked, primary_unit, acceptable_alternates, arm):
    """Original Phase4B any-G native metric, before distinct-page adaptation."""
    if arm == 'T0':
        return 'NOT_APPLICABLE'
    if arm not in ('N1', 'M1') or not primary_unit:
        raise ValueError('legacy native scoring requires declared primary')
    gold = {primary_unit, *acceptable_alternates}
    seen, first = set(), []
    for position, hit in enumerate(ranked[:30]):
        key = hit.unit_id if hit.unit_id else ('unmappable', position)
        if key in seen:
            continue
        seen.add(key)
        first.append(hit.unit_id)
        if len(first) == 5:
            break
    return float(any(unit in gold for unit in first))


def percentile(values, probability):
    """Linear interpolated order statistic, equivalent to NumPy method=linear."""
    values = sorted(values)
    position = (len(values) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def paired_statistics(rows, modalities=MM):
    """Rows are paired outcomes, never query strings: modality,paper_id,T0,M1."""
    groups = {m: [r for r in rows if r['modality'] == m] for m in modalities}
    if any(not g for g in groups.values()):
        raise ValueError('every preregistered stratum required')
    selected = [r for m in modalities for r in groups[m]]
    if any(r['T0'] not in (0, 1) or r['M1'] not in (0, 1) or not r['paper_id'] for r in selected):
        raise ValueError('paired binary recall/paper identity required')
    deltas = {m: [r['M1'] - r['T0'] for r in groups[m]] for m in modalities}
    observed = mean(mean(d) for d in deltas.values())
    rng = random.Random(SEED)
    bootstrap = [mean(mean(rng.choice(d) for _ in d) for d in deltas.values()) for _ in range(REPETITIONS)]
    rng = random.Random(SEED + 1)
    permutation = [mean(mean(v * rng.choice((-1, 1)) for v in d) for d in deltas.values()) for _ in range(REPETITIONS)]
    papers = sorted({r['paper_id'] for r in selected})
    cluster_rows = {p: [r for r in selected if r['paper_id'] == p] for p in papers}
    rng = random.Random(SEED + 2)
    cluster, missing = [], 0
    while len(cluster) < REPETITIONS:
        sampled = [r for _ in papers for r in cluster_rows[rng.choice(papers)]]
        grouped = {m: [r['M1'] - r['T0'] for r in sampled if r['modality'] == m] for m in modalities}
        if any(not d for d in grouped.values()):
            missing += 1
            if missing > REPETITIONS * 100:
                raise ValueError('cluster sampling cannot support all strata')
            continue
        cluster.append(mean(mean(d) for d in grouped.values()))
    rng = random.Random(SEED + 3)
    cluster_permutation = []
    for _ in range(REPETITIONS):
        signs = {p: rng.choice((-1, 1)) for p in papers}
        cluster_permutation.append(mean(mean((r['M1'] - r['T0']) * signs[r['paper_id']] for r in groups[m]) for m in modalities))
    interval = lambda v: [percentile(v, .025), percentile(v, .975)]
    p = lambda v: (1 + sum(abs(x) >= abs(observed) - 1e-12 for x in v)) / (REPETITIONS + 1)
    return {'delta': observed, 'per_modality_delta': {m: mean(d) for m, d in deltas.items()},
            'paired_stratified_CI95': interval(bootstrap), 'two_sided_permutation_p': p(permutation),
            'paper_cluster_CI95': interval(cluster), 'paper_cluster_permutation_p': p(cluster_permutation),
            'cluster_missing_strata_draws_rejected': missing, 'paper_count': len(papers),
            'counts': {m: len(g) for m, g in groups.items()}, 'seed': SEED, 'repetitions': REPETITIONS}


def benefit_decision(primary, text_control, technical_valid=True):
    if not technical_valid:
        return 'NOT_ASSESSED'
    credible = primary['paired_stratified_CI95'][0] > 0 and primary['paper_cluster_CI95'][0] > 0
    control_ok = text_control['paired_stratified_CI95'][0] >= -TEXT_TOLERANCE and text_control['paper_cluster_CI95'][0] >= -TEXT_TOLERANCE
    if primary['delta'] > 0 and credible and control_ok:
        return 'SUPPORTED_BENEFIT'
    mixed_modalities = any(d > 0 for d in primary['per_modality_delta'].values()) and any(d < 0 for d in primary['per_modality_delta'].values())
    if primary['delta'] > 0 or mixed_modalities:
        return 'MIXED'
    return 'NO_DEMONSTRATED_BENEFIT'


def semantic_access_allowed(path, b1_root):
    """B2 denies all B1 semantic files, including DEV. Resolve aliases first."""
    from pathlib import Path
    resolved, b1 = Path(path).resolve(), Path(b1_root).resolve()
    if not resolved.is_relative_to(b1):
        return True
    metadata = {'decision.json', 'completion_freeze.json', 'entry_audit.json',
                'accepted/manifest.json', 'accepted/manifest_digest.json',
                'split/dev_manifest.json', 'split/test_manifest.json',
                'split/test_seal.json', 'sealed/construction_seal.json'}
    return resolved.relative_to(b1).as_posix() in metadata


def guarded_read(path, b1_root):
    if not semantic_access_allowed(path, b1_root):
        raise PermissionError('B2 semantic surface sealed; opaque SHA256 hashing only')
    return path.read_text(encoding='utf8')
