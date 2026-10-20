"""Small offline selection contracts. Fixtures are not scientific gold."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from paper_research.agents.research_agent import final_report as reports
from paper_research.agents.research_agent.models import NativeEvidenceItem
from paper_research.agents.research_agent.state import AgentState
from paper_research.agents.research_agent.verifier import ResearchSafetyError
from paper_research.pdf_multimodal.identity import digest
from paper_research.providers.llm import StructuredJSONResult


def native(key, modality, text, page=1):
    uid = 'mmu:' + key
    unit = {'unit_id': uid, 'paper_id': 'fixture-paper', 'pdf_sha256': 'a' * 64,
            'page_number': page, 'bbox': [0, 0, 32, 24],
            'unit_type': {'FIGURE': 'FIGURE_CANDIDATE', 'TABLE': 'TABLE_CANDIDATE',
                          'EQUATION': 'EQUATION_CANDIDATE'}[modality],
            'canonical_text': text, 'raw_text': text}
    return NativeEvidenceItem(
        evidence_id='fixture:' + uid, paper_id='fixture', block_id=uid,
        page=page, page_end=page, text_or_reference=text[:800],
        discovered_by_tool='fixture', discovered_at_step=1,
        unit_id=uid, modality=modality, source_pdf='fixture-only.pdf', pdf_sha256='a' * 64,
        native_paper_id='fixture-paper', native_unit=unit, native_unit_sha256=digest(unit),
        readable_kind={'FIGURE': 'caption_only', 'TABLE': 'table_structure',
                       'EQUATION': 'native_equation_candidate_text'}[modality],
        readable_text=text, bundle_sha256='b' * 64, retrieval_identity={'fixture': True},
        generation_binding_digest='c' * 64,
        occurrences=[{'task_id': 'fixture', 'tool_call_id': 'fixture', 'trace_id': 'fixture',
            'step': 1, 'query': 'fixture', 'paper_scope': ['fixture'], 'target_subquestion_ids': [],
            'generation_binding_digest': 'c' * 64, 'arm': 'N1', 'channels': {'sparse': [uid]},
            'rank': 1}],
    ).model_dump(mode='json')


def text(key, content, page=1):
    sha = hashlib.sha256(content.encode()).hexdigest()
    return {'evidence_id': 'fixture:' + key, 'paper_id': 'fixture', 'chunk_id': key,
        'page': page, 'page_end': page, 'block_ids': [key], 'block_page_map': {key: page},
        'section': 'Methods', 'text_or_reference': content, 'model_text': content,
        'verification_status': 'CANDIDATE', 'generation_binding_digest': 'c' * 64,
        'retrieval_identity': {'fixture': True}, 'text_proof': {
            'display_text_sha256': sha, 'canonical_text_sha256': sha,
            'canonical_text_length': len(content), 'canonical_source_sha256': 'd' * 64}}


def table(name, value='0.90'):
    return native(name, 'TABLE', f'Table 1: {name} results\nModel\taccuracy\nMethod\t{value}')


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(httpx.Client, 'send', lambda *a, **k: pytest.fail('Network forbidden'))


@pytest.mark.parametrize('reverse', [False, True])
def test_architecture_role_not_fixed_position(reverse):
    pool = [native('curve', 'FIGURE', 'Accuracy curves across epochs'),
            native('diagram', 'FIGURE', 'The model architecture and pipeline')]
    if reverse:
        pool.reverse()
    result = reports.assemble_draft_input('Explain the model architecture.', pool)
    assert result['sources'][0]['unit_id'] == 'mmu:diagram'
    assert result['evidence_coverage']['requirements'][0]['selected_aliases']


def test_irrelevant_large_numeric_table_does_not_win():
    other = table('Gamma', '0.80 ' * 90)
    result = reports.assemble_draft_input('Compare experiments on Alpha.', [other, table('Alpha')])
    selected = {s['unit_id'] for s in result['sources']}
    assert 'mmu:Alpha' in selected
    assert 'mmu:Gamma' not in selected


def test_short_core_formula_beats_long_unrelated_formula():
    pool = [native('long', 'EQUATION', 'x = ' + 'a+b+' * 100),
            native('short', 'EQUATION', 'Loss = a + b; objective')]
    result = reports.assemble_draft_input('Explain the core objective formula.', pool)
    assert [s['unit_id'] for s in result['sources']] == ['mmu:short']
    assert 'FORMULA_COMPLETENESS_UNVERIFIED' in result['sources'][0]['context_notes'][0]


def test_experiment_diversity_and_budget_gap():
    pool = [table('Alpha'), table('Alpha'), table('Beta'), table('Gamma')]
    result = reports.assemble_draft_input('Compare Alpha, Beta and Gamma experiments.', pool)
    rows = {r['id']: r for r in result['evidence_coverage']['requirements']}
    assert len([s for s in result['sources'] if s.get('modality') == 'TABLE']) == 2
    assert rows['experiment:Alpha']['selected_aliases']
    assert rows['experiment:Beta']['selected_aliases']
    assert rows['experiment:Gamma']['status'] == 'COVERAGE_LIMITED_BY_BUDGET'


def test_chinese_question_preserves_each_named_experiment():
    rows = reports._research_evidence_requirements('结合Alpha、Beta、Gamma实验比较结果')
    assert [r['entity'] for r in rows if r['entity']] == ['Alpha', 'Beta', 'Gamma']
    unparsed = reports.assemble_draft_input('比较甲任务、乙任务实验结果', [table('Alpha')])
    row = next(r for r in unparsed['evidence_coverage']['requirements'] if r['id'] == 'experiment:UNKNOWN')
    assert row['status'] == 'UNKNOWN' and not row['selected_aliases']


def test_anonymous_table_does_not_inherit_same_page_experiment():
    fragment = native('fragment', 'TABLE', 'A\t0.90\nB\t0.80')
    context = text('context', 'Alpha results have relative error 0.1. Beta results have relative error 0.2.')
    result = reports.assemble_draft_input('Compare Alpha and Beta experiments.', [fragment, context])
    source = next(s for s in result['sources'] if s.get('modality') == 'TABLE')
    assert any(n.startswith('TABLE_HEADERS_UNCONFIRMED') for n in source['context_notes'])
    for row in result['evidence_coverage']['requirements']:
        if row['id'].startswith('experiment:'):
            assert row['status'] != 'CANDIDATE_CONTEXT_PRESENT'
    assert 'Alpha' not in source['text'] and 'Beta' not in source['text']


def test_missing_requested_evidence_and_unknown_goal_are_not_covered():
    curve = native('curve', 'FIGURE', 'Results curve')
    result = reports.assemble_draft_input('Explain architecture and computation cost.', [curve])
    assert all(r['status'] == 'NO_RELIABLE_CANDIDATE'
               for r in result['evidence_coverage']['requirements'])
    unknown = reports.assemble_draft_input('Tell me something.', [curve])
    assert unknown['evidence_coverage']['requirements'][0]['status'] == 'UNKNOWN'


def test_byte_budget_rejects_without_rewriting_chosen_context(monkeypatch):
    monkeypatch.setattr(reports, 'DRAFT_INPUT_BYTES', 5000)
    pool = [native('arch', 'FIGURE', 'Architecture. ' + 'z' * 2600),
            native('cost', 'TABLE', 'Table 1: Costs\nModel\truntime\nA\t0.1 ' + 'z' * 2600)]
    result = reports.assemble_draft_input('Explain architecture and runtime cost.', pool)
    assert any(r['status'] == 'COVERAGE_LIMITED_BY_BUDGET'
               for r in result['evidence_coverage']['requirements'])
    original = {s['evidence_id']: s['readable_text'] for s in pool}
    assert all(s['text'] == original[s['canonical_evidence_id']] for s in result['sources'])


def test_mutated_canonical_text_is_rejected_even_if_not_selected():
    damaged = text('damaged', 'An irrelevant source')
    damaged['model_text'] = 'fabricated'
    with pytest.raises(ResearchSafetyError, match='CANONICAL_PROVENANCE_INVALID'):
        reports.assemble_draft_input('Explain architecture.', [native('fig', 'FIGURE', 'Architecture'), damaged])


def test_coverage_matches_literal_prompt_identity_and_source_input_unchanged():
    pool = [native('fig', 'FIGURE', 'Architecture diagram'), table('Alpha'),
            text('math', 'Definition 1: the kernel operator K is defined as K(v) = Wv.')]
    before = copy.deepcopy(pool)
    result = reports.assemble_draft_input('Explain architecture, core formula and Alpha experiments.', pool)
    body = json.loads(result['user_prompt'])
    selected = [s for s in result['evidence_coverage']['candidates'] if s['selected']]
    assert {s['alias'] for s in selected} == {s['evidence_id'] for s in body['candidate_evidence']}
    assert all(s['assembled_in_user_prompt'] == s['selected'] for s in result['evidence_coverage']['candidates'])
    assert result['evidence_coverage']['transport_status'] == 'NOT_SENT'
    assert pool == before


def test_planner_maps_only_question_roles_and_default_text_keeps_prompt():
    plans = [{'id': 'architecture-step', 'question': 'Explain architecture.'},
             {'id': 'unrequested', 'question': 'Compare experiment results.'}]
    result = reports.assemble_draft_input('Explain architecture.', [native('fig', 'FIGURE', 'Architecture')],
                                         planner_context=plans)
    rows = result['evidence_coverage']['requirements']
    assert [r['id'] for r in rows] == ['architecture']
    assert rows[0]['planner_subquestion_ids'] == ['architecture-step']
    plain = reports.assemble_draft_input('Explain the method.', [text('method', 'Our method uses a kernel.')])
    assert 'evidence_coverage' in plain
    assert 'evidence_coverage' not in json.loads(plain['user_prompt'])


@pytest.mark.parametrize('provider_fails', [False, True])
def test_existing_generation_persists_coverage_without_forwarding_server_metadata(tmp_path, monkeypatch, provider_fails):
    monkeypatch.chdir(tmp_path)
    state = AgentState(task_id='offline-goal-fixture', research_question='Explain architecture.',
                       multimodal_enabled=True, status='PARTIAL',
                       evidence_provenance={'p1_active_canonical_provenance_v2': {}})
    local = SimpleNamespace(draft_evidence=lambda _: [native('fig', 'FIGURE', 'Architecture')])

    class Provider:
        def generate_structured_json(self, **request):
            assert 'evidence_coverage' not in request
            if provider_fails:
                raise ValueError('Offline provider failure fixture')
            return StructuredJSONResult(provider='OFFLINE_FIXTURE', model='not-a-model-call', payload={
                'draft_markdown': 'UNVERIFIED fixture [E01]',
                'claims': [{'text': 'Fixture only', 'evidence_ids': ['E01']}], 'evidence_gaps': []})

    if provider_fails:
        with pytest.raises(ValueError, match='Offline provider failure'):
            reports.generate_unverified_draft(state, local, Provider())
        files = list(tmp_path.glob('.runtime/research-synthesis-provider/*/evidence_coverage.json'))
        record = json.loads(files[0].read_text(encoding='utf8'))
        assert record['transport_status'] == 'PROVIDER_CALL_ENTERED_HTTP_OUTCOME_UNKNOWN'
        return
    result = reports.generate_unverified_draft(state, local, Provider())
    path = tmp_path / '.runtime/research-synthesis-provider' / result['request_id'] / 'evidence_coverage.json'
    record = json.loads(path.read_text(encoding='utf8'))
    assert record == result['evidence_coverage']
    assert record['provider'] == 'OFFLINE_FIXTURE'
    assert record['transport_status'] == 'PROVIDER_RETURNED_NOT_HTTP_ATTESTATION'
    assert result['status'] == 'UNVERIFIED'


def test_frozen_p3_replay_is_known_case_not_new_research():
    root = Path(__file__).resolve().parents[1] / 'artifacts/paperresearch-v2/p3-r-fno-first-run'
    if not (root / 'retrieved_evidence.json').exists():
        pytest.skip('Private frozen replay data is not distributed; generic contracts run offline')
    pool = json.loads((root / 'retrieved_evidence.json').read_text(encoding='utf8'))
    state = json.loads((root / 'agent_state.json').read_text(encoding='utf8'))
    result = reports.assemble_draft_input(state['research_question'], pool)
    sources = result['sources']
    assert len(pool) == 35
    assert next(s for s in sources if s.get('modality') == 'FIGURE')['page'] == 4
    assert any('convolution theorem' in s['text'] for s in sources)
    assert any('64 × 64 × 20' in s['text'] for s in sources)
    assert any('evaluate a single instance' in s['text'] for s in sources)
    assert all(sum(s.get('modality', 'TEXT') == k for s in sources) <= v
               for k, v in reports._MULTIMODAL_LIMITS.items())
    assert result['evidence_coverage']['transport_status'] == 'NOT_SENT'
