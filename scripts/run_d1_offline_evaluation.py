"""D1 offline hybrid/section-selection evaluation from frozen C1/C2/B1 artifacts."""
# ruff: noqa
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
from run_c1_gold_free_execution import questions_and_gold
from run_c2_snapshot_native_reranking import metrics, score_question

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'artifacts/rag-quality-v3/d1/execution'; ARMS=('D1-R0','D1-R1','D1-R2')
def save(p,v): p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf8')
def load_json(p): return json.loads(p.read_text(encoding='utf8'))
def reorder_hybrid(cands, order):
 ranks={x:i for i,x in enumerate(order,1)}
 return sorted(cands,key=lambda x:(-(1/(60+x['fused_rank'])+1/(60+ranks[x['candidate_unit_id']])),x['candidate_unit_id']))
def selector(cands):
 if not cands:return cands
 picked=[cands[0]]; seen={tuple(cands[0]['source_spans'][0])}
 for x in cands[1:]:
  if len(picked)<5 and len(seen)<3 and tuple(x['source_spans'][0]) not in seen: picked.append(x);seen.add(tuple(x['source_spans'][0]))
 for x in cands:
  if x not in picked:picked.append(x)
 return picked
def cohort_c1():
 s=load_json(ROOT/'artifacts/rag-quality-v3/c1/execution/snapshots/c1-r0-candidate-snapshot-v1.json');t=load_json(ROOT/'artifacts/rag-quality-v3/c2/execution/traces/c2-r1-ordering-v1.json');qs,gmap=questions_and_gold();return 'DEV176',s,t,{q['id']:q for q in qs},gmap
def cohort_b1():
 s=load_json(ROOT/'artifacts/rag-quality-v3/b1/execution/snapshots/b1-candidate-snapshot-v1.json');t=load_json(ROOT/'artifacts/rag-quality-v3/b1/execution/traces/b1-r1-ordering-v1.json');run=load_json(ROOT/'artifacts/rag-quality-v3/b1/pre-freeze/b1-blind-runtime-questions-v1.json')['questions'];gold=load_json(ROOT/'artifacts/rag-quality-v3/b1/pre-freeze/b1-blind-evaluation-gold-v1.json')['gold'];gm={x['id']:x for x in gold};qs={x['id']:{**x,'dataset':'POSTBLIND','gold':set(gm[x['id']]['neutral_gold_unit_ids']),'claims':[set(c) for c in gm[x['id']]['required_claims']]} for x in run};return 'POSTBLIND60',s,t,qs,{n:n for x in gold for n in x['neutral_gold_unit_ids']}
def evaluate(name,snap,trace,qs,gmap):
 orders={x['question_id']:x['candidate_unit_ids'] for x in trace['records']}; rows={a:[] for a in ARMS}; identities=0
 for rec in snap['records']:
  q=qs[rec['question_id']];base=rec['candidates'];by={x['candidate_unit_id']:x for x in base};r0=[by[x] for x in orders[q['id']]];r1=reorder_hybrid(base,orders[q['id']]);r2=selector(r1)
  if not (set(x['candidate_unit_id'] for x in r0)==set(x['candidate_unit_id'] for x in r1)==set(x['candidate_unit_id'] for x in r2)):raise RuntimeError('D1_PAIRED_IDENTITY_VIOLATION:'+q['id'])
  identities+=1
  for arm,order in zip(ARMS,(r0,r1,r2),strict=True):rows[arm].append(score_question(q,order,gmap))
 return rows,identities
def mean(rows,key):return round(sum(x[key] for x in rows)/len(rows),6)
def safety(base,cand):
 subs={};b={x['id']:x for x in base}
 for cat in ('single_evidence','multi_evidence','semantic','formula','comparison'):
  rs=[x for x in cand if (x['category'].startswith('single') if cat=='single_evidence' else ('paraphrase' in x['category'] if cat=='semantic' else (x['category'] in ('comparison','compare') if cat=='comparison' else x['category']==cat)))];d=mean(rs,'gold_recall_5')-mean([b[x['id']] for x in rs],'gold_recall_5') if rs else 0.0;subs[cat]=round(d,6)
 paper=defaultdict(list)
 for x in cand:
  if x['required_claim_coverage@5'] is not None:paper[x['doc']].append(x['required_claim_coverage@5']-b[x['id']]['required_claim_coverage@5'])
 rate=sum(sum(v)/len(v)>=0 for v in paper.values())/len(paper)
 return {'subsets':subs,'paper_rate':round(rate,6),'pass':all(v>=-.02 for v in subs.values()) and rate>=.75}
def main():
 cohorts={};all_rows={a:[] for a in ARMS};ident={};cohort_rows={}
 for data in (cohort_c1(),cohort_b1()):
  name,s,t,q,g=data;rows,n=evaluate(name,s,t,q,g);cohort_rows[name]=rows;cohorts[name]={a:{'metrics':metrics(rows[a])[0],'losses':metrics(rows[a])[1]} for a in ARMS};ident[name]=n
  for a in ARMS:all_rows[a]+=rows[a]
 cohorts['COMBINED236']={a:{'metrics':metrics(all_rows[a])[0],'losses':metrics(all_rows[a])[1]} for a in ARMS}
 # use stored cohort row collections for gates
 d0=load_json(ROOT/'artifacts/rag-quality-v3/d0/d0-post-blind-failure-attribution-v1.json')['residual_details'];bs=load_json(ROOT/'artifacts/rag-quality-v3/b1/execution/snapshots/b1-candidate-snapshot-v1.json');bt=load_json(ROOT/'artifacts/rag-quality-v3/b1/execution/traces/b1-r1-ordering-v1.json');borders={x['question_id']:x['candidate_unit_ids'] for x in bt['records']}; recovery={a:defaultdict(lambda:{'fixed':0,'unchanged':0,'newly_regressed':0}) for a in ('D1-R1','D1-R2')}
 for case in d0:
  rec=next(x for x in bs['records'] if x['question_id']==case['question_id']);by={x['candidate_unit_id']:x for x in rec['candidates']};base=[by[x] for x in borders[case['question_id']]];hy=reorder_hybrid(rec['candidates'],borders[case['question_id']]);orders={'D1-R1':hy,'D1-R2':selector(hy)};claim=set(case['missing_claim']);base_hit=any(set(x['neutral_source_block_ids'])&claim for x in base[:5])
  for arm,order in orders.items():
   hit=any(set(x['neutral_source_block_ids'])&claim for x in order[:5]);bucket=recovery[arm][case['taxonomy']];bucket['fixed' if hit and not base_hit else 'newly_regressed' if base_hit and not hit else 'unchanged']+=1
 result={'schema_version':'ragq3-d1-offline-evaluation-v1','freeze_commit':'dd6bd9783de6ea869c3a9fb4fa7eae8e90feb94f','paired_identity':ident,'cohorts':cohorts,'d0_residual_recovery':{arm:dict(values) for arm,values in recovery.items()},'provider_calls':0,'invariants':{'retrieval':0,'embedding':0,'reranker':0,'full_qa':'NOT_RUN','production_change':'no'}}
 # Explicit gates from metrics only; selector gate safety is evaluated from cohort aggregate.
 for arm in ('D1-R1','D1-R2'):
  verdict=True
  for cohort in ('DEV176','POSTBLIND60'):
   a,b=cohorts[cohort]['D1-R0']['metrics'],cohorts[cohort][arm]['metrics']; verdict &= all(b[k]-a[k]>=-.02 for k in ('GoldR@5','MRR','NDCG@10','context_precision','context_recall','required_claim_coverage@5','multi_evidence_complete_rate@5')); verdict &= safety(cohort_rows[cohort]['D1-R0'],cohort_rows[cohort][arm])['pass']
  result.setdefault('frozen_gate',{})[arm]={'PASS':bool(verdict)}
 selected='D1-R1' if result['frozen_gate']['D1-R1']['PASS'] else 'NONE'
 if result['frozen_gate']['D1-R2']['PASS']:
  c=cohorts['COMBINED236'];gain=c['D1-R2']['metrics']['multi_evidence_complete_rate@5']-c['D1-R1']['metrics']['multi_evidence_complete_rate@5'];noninf=c['D1-R2']['metrics']['MRR']>=c['D1-R1']['metrics']['MRR']-.02 and c['D1-R2']['metrics']['NDCG@10']>=c['D1-R1']['metrics']['NDCG@10']-.02
  if gain>=.05 and noninf:selected='D1-R2'
 result.update({'selected_candidate':selected,'decision':'MIXED_POST_RETRIEVAL_DEVELOPMENT_VALIDATED' if selected!='NONE' else 'MIXED_POST_RETRIEVAL_INTERVENTION_NOT_SUFFICIENT','B2_FRESH_BLIND_ELIGIBLE':'yes' if selected!='NONE' else 'no','tuning':'no'})
 save(OUT/'d1-final-decision-v1.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
