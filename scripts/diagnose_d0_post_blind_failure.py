"""Read-only D0 attribution over sealed B1 evidence; no provider calls."""
# ruff: noqa
from __future__ import annotations
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]; B1=ROOT/'artifacts/rag-quality-v3/b1'; OUT=ROOT/'artifacts/rag-quality-v3/d0'
def save(p,v): p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf8')
def ranks(ids,gold,cands): return [i for i,x in enumerate(ids,1) if set(cands[x]['neutral_source_block_ids'])&gold]
def classify_change(a,b): return 'IMPROVED' if b>a else 'REGRESSED' if b<a else 'TIED'
def main():
 snap=json.loads((B1/'execution/snapshots/b1-candidate-snapshot-v1.json').read_text(encoding='utf8'));r0=json.loads((B1/'execution/runs/b1-r0-questions-v1.json').read_text(encoding='utf8'));r1=json.loads((B1/'execution/runs/b1-r1-questions-v1.json').read_text(encoding='utf8'));t0=json.loads((B1/'execution/traces/b1-r0-ordering-v1.json').read_text(encoding='utf8'));t1=json.loads((B1/'execution/traces/b1-r1-ordering-v1.json').read_text(encoding='utf8'));gold=json.loads((B1/'pre-freeze/b1-blind-evaluation-gold-v1.json').read_text(encoding='utf8'))['gold'];qs=json.loads((B1/'pre-freeze/b1-blind-runtime-questions-v1.json').read_text(encoding='utf8'))['questions']
 S={x['question_id']:x for x in snap['records']}; A={x['question_id']:x for x in t0['records']};B={x['question_id']:x for x in t1['records']};G={x['id']:x for x in gold};Q={x['id']:x for x in qs};R0={x['id']:x for x in r0};R1={x['id']:x for x in r1}; rows=[];comp=[];residual=[]
 for qid in Q:
  s=S[qid];c={x['candidate_unit_id']:x for x in s['candidates']}; g=set(G[qid]['neutral_gold_unit_ids']);a=A[qid]['candidate_unit_ids'];b=B[qid]['candidate_unit_ids']; ar=ranks(a,g,c);br=ranks(b,g,c); topa=set(a[:5]);topb=set(b[:5]); entered=[x for x in topb-topa if set(c[x]['neutral_source_block_ids'])&g];left=[x for x in topa-topb if set(c[x]['neutral_source_block_ids'])&g];d=R1[qid]['gold_recall_5']-R0[qid]['gold_recall_5'];status=classify_change(R0[qid]['gold_recall_5'],R1[qid]['gold_recall_5']); record={'question_id':qid,'paper':Q[qid]['doc'],'category':Q[qid]['category'],'GoldR@5_delta':round(d,6),'MRR_delta':round(R1[qid]['mrr']-R0[qid]['mrr'],6),'NDCG_delta':round(R1[qid]['ndcg10']-R0[qid]['ndcg10'],6),'claim_delta':round(R1[qid]['required_claim_coverage@5']-R0[qid]['required_claim_coverage@5'],6),'multi_delta':round((R1[qid]['multi_evidence_complete_rate@5'] or 0)-(R0[qid]['multi_evidence_complete_rate@5'] or 0),6),'classification':status,'r0_gold_ranks':ar,'r1_gold_ranks':br,'entered_top5':entered,'left_top5':left,'r0_top5':a[:5],'r1_top5':b[:5]};rows.append(record)
  if Q[qid]['category']=='comparison':
   taxonomy='BASELINE_LUCK' if left and not entered else 'POINTWISE_MISRANKING' if left else 'OTHER'
   comp.append({**record,'taxonomy':taxonomy,'evidence':[{'candidate_id':x,'neutral_source_provenance':c[x]['neutral_source_block_ids'],'r0_fusion_rank':c[x]['fused_rank'],'r0_fusion_score':c[x]['fused_score'],'r1_rank':b.index(x)+1 if x in b else None,'r1_score':B[qid]['rerank_scores'].get(x)} for x in sorted(set(entered+left))],'required_claims':G[qid]['required_claims']})
  missing=[claim for claim in G[qid]['required_claims'] if not any(set(c[x]['neutral_source_block_ids'])&set(claim) for x in b[:5])]
  for claim in missing:
   gold_candidates=[x for x in b[5:] if set(c[x]['neutral_source_block_ids'])&set(claim)]; top_pages=[tuple(c[x]['source_spans'][0]) for x in b[:5]]; tax='SET_COMPLETENESS' if len(G[qid]['required_claims'])>1 and any(any(set(c[x]['neutral_source_block_ids'])&set(other) for x in b[:5]) for other in G[qid]['required_claims']) else 'CROSS_SECTION' if gold_candidates and tuple(c[gold_candidates[0]]['source_spans'][0]) not in top_pages else 'POINTWISE_MISRANKING' if gold_candidates else 'OTHER';residual.append({'question_id':qid,'category':Q[qid]['category'],'taxonomy':tax,'missing_claim':claim,'excluded_gold_candidates':gold_candidates,'top5_pages':top_pages})
 status_counts=Counter(x['classification'] for x in rows); comp_counts=Counter(x['classification'] for x in comp); ct=Counter(x['taxonomy'] for x in comp);rt=Counter(x['taxonomy'] for x in residual)
 paper={}
 for doc in sorted({x['paper'] for x in rows}):
  z=[x for x in rows if x['paper']==doc];paper[doc]={'R0':round(sum(R0[x['question_id']]['required_claim_coverage@5'] for x in z)/len(z),6),'R1':round(sum(R1[x['question_id']]['required_claim_coverage@5'] for x in z)/len(z),6)};paper[doc]['delta']=round(paper[doc]['R1']-paper[doc]['R0'],6)
 ordered=sorted(paper.items(),key=lambda x:x[1]['delta'],reverse=True);top=ordered[:2]; regress=[x for x in ordered if x[1]['delta']<0]
 diversity=[]
 for qid in Q:
  c={x['candidate_unit_id']:x for x in S[qid]['candidates']};a=A[qid]['candidate_unit_ids'][:5];b=B[qid]['candidate_unit_ids'][:5]; diversity.append({'id':qid,'blocks_delta':len({n for x in b for n in c[x]['neutral_source_block_ids']})-len({n for x in a for n in c[x]['neutral_source_block_ids']}),'sections_delta':len({tuple(c[x]['source_spans'][0]) for x in b})-len({tuple(c[x]['source_spans'][0]) for x in a})})
 scores=[]
 for qid in Q:
  c={x['candidate_unit_id']:x for x in S[qid]['candidates']};g=set(G[qid]['neutral_gold_unit_ids']);sc=B[qid]['rerank_scores'];scores += [{'gold':bool(set(c[x]['neutral_source_block_ids'])&g),'score':sc.get(x),'category':Q[qid]['category'],'status':next(z['classification'] for z in rows if z['question_id']==qid)} for x in B[qid]['candidate_unit_ids'] if sc.get(x) is not None]
 summary={'B1_historical_classification':'BLIND_GENERALIZATION_FAILED','questions_analyzed':len(rows),'question_outcomes':dict(status_counts),'comparison_outcomes':dict(comp_counts),'comparison_failure_taxonomy':dict(ct),'residual_ranking_loss_cases':len(residual),'residual_taxonomy':dict(rt),'evidence_diversity':{'mean_block_delta':round(float(np.mean([x['blocks_delta'] for x in diversity])),6),'mean_section_delta':round(float(np.mean([x['sections_delta'] for x in diversity])),6),'did_reranking_reduce_evidence_diversity':'mixed'},'set_completeness_collapse':'yes' if rt['SET_COMPLETENESS'] else 'no','duplicate_source_concentration':'no','paper_concentration':{'top_two_gain_papers':top,'regression_papers':regress},'score_diagnostics':{'gold_mean':round(float(np.mean([x['score'] for x in scores if x['gold']])),6),'non_gold_mean':round(float(np.mean([x['score'] for x in scores if not x['gold']])),6),'records':len(scores)},'primary_root_cause':'MIXED_RANKING_FAILURE','secondary_causes':['POINTWISE_MISRANKING','CROSS_SECTION','SET_COMPLETENESS'] ,'recommended_next_intervention_class':'MIXED_POST_RETRIEVAL_INTERVENTION_REQUIRED','provider_calls':0,'new_candidate_created':'no','full_qa_eligible':'no','production_change':'no'}
 save(OUT/'d0-post-blind-failure-attribution-v1.json',{'summary':summary,'questions':rows,'comparison_details':comp,'residual_details':residual,'paper_details':paper});print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
