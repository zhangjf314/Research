"""Frozen C1 Gold-free representation development execution."""
# ruff: noqa
from __future__ import annotations
import argparse, hashlib, json, math, uuid
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
import run_q3c_local_live as common
import run_q3d_siliconflow_live as q3d
from paper_research.chunking.types import Chunk
from paper_research.config import Settings
from paper_research.evaluation.ragq3_identity import stable_id
from paper_research.retrieval.context_builder import ContextBuilder
from paper_research.retrieval.dense import RetrievalResult
from paper_research.retrieval.filters import RetrievalFilter
from paper_research.retrieval.fusion import FusedResult, reciprocal_rank_fusion
from paper_research.retrieval.hybrid import HybridRetriever
from paper_research.retrieval.sparse import BM25Retriever

ROOT=Path(__file__).resolve().parents[1]; CANONICAL_ROOT=ROOT
OUT=ROOT/'artifacts/rag-quality-v3/c1/execution'; FREEZE='37fdf30884950bee5a4b5d58eff90da0f09c287b'
ARMS=('C1-R0','C1-R1','C1-R2','C1-R3','C1-R4')
def save(p,v): p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(v,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf8')
def h(v): return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def collection(a): return 'ragq3_c1_'+a.lower().replace('-','_')+'_37fdf30'
def point_id(x): return str(uuid.uuid5(uuid.NAMESPACE_URL,'ragq3-c1:'+x))
def norm(v):
 import numpy as np
 a=np.asarray(v,dtype=np.float32); n=np.linalg.norm(a,axis=1,keepdims=True)
 if np.any(n==0): raise RuntimeError('GOLD_FREE_RUNTIME_VIOLATION:zero_vector')
 return (a/n).tolist()
def docid(ext): return stable_id('c1_document',{'external_source_id':ext})
def neutral(doc,raw): return stable_id('c1_neutral_block',{'document':doc,'parser_local_block_id':raw})
def build(arm):
 src,rawmeta=common.build(arm.replace('C1','Q3C')); units=[]; meta={}
 for u in src:
  d=docid(u.paper_id); nids=[neutral(d,b) for b in u.block_ids]
  uid=stable_id('c1_unit',{'source_unit':u.chunk_id,'arm':arm,'freeze':FREEZE})
  x=Chunk(chunk_id=uid,paper_id=d,block_ids=nids,section_path=u.section_path,block_type=u.block_type,page_start=u.page_start,page_end=u.page_end,chunk_text=u.chunk_text,token_count=u.token_count)
  units.append(x); meta[uid]={'neutral_source_block_ids':nids,'source_spans':[[u.page_start,u.page_end]],'text_sha256':hashlib.sha256(u.chunk_text.encode()).hexdigest()}
 if len({u.chunk_id for u in units})!=len(units): raise RuntimeError('GOLD_FREE_RUNTIME_VIOLATION:duplicate_unit')
 return units,meta
def questions_and_gold():
 docs,_=common.registry(); historical={v['canonical_document_id']:k for k,v in docs.items()}; result=[]; goldmap={}
 for q in q3d.canonical_questions():
  ext=historical[q['doc']]; d=docid(ext); blockmap=docs[ext]['canonical_gold_blocks']
  # evaluation-only lineage: neutral parser IDs to frozen canonical Gold IDs
  for raw,gold in blockmap.items(): goldmap[neutral(d,raw)]=gold
  z=dict(q); z['doc']=d; result.append(z)
 return result,goldmap
def materialize(arm,client,provider):
 units,meta=build(arm); name=collection(arm); names={x.name for x in client.get_collections().collections}
 if name not in names: client.create_collection(name,vectors_config=models.VectorParams(size=1024,distance=models.Distance.COSINE))
 known=set(); off=None
 while True:
  rows,off=client.scroll(name,limit=512,offset=off,with_payload=True,with_vectors=False); known|={str(x.payload.get('unit_id')) for x in rows}
  if off is None: break
 pending=[x for x in units if x.chunk_id not in known]
 for i in range(0,len(pending),32):
  batch=pending[i:i+32]; vec=norm(provider.embed_documents([x.chunk_text for x in batch])); client.upsert(name,[models.PointStruct(id=point_id(x.chunk_id),vector=v,payload={'unit_id':x.chunk_id,'canonical_document_id':x.paper_id,'neutral_source_block_ids':meta[x.chunk_id]['neutral_source_block_ids'],'source_spans':meta[x.chunk_id]['source_spans'],'text_sha256':meta[x.chunk_id]['text_sha256']}) for x,v in zip(batch,vec,strict=True)],wait=True); print(f'{arm} indexed {i+len(batch)}/{len(pending)}',flush=True)
 count=client.get_collection(name).points_count; save(OUT/'indexes'/f'{arm.lower()}-index-v1.json',{'arm':arm,'status':'PASS' if count==len(units) else 'FAIL','collection':name,'points':count,'expected_points':len(units),'gold_free_payload':True,'runtime_contract':FREEZE,'provider':provider.stats})
 if count!=len(units): raise RuntimeError(f'index incomplete {arm}')
 return units,meta
def mean(rows,k): return round(sum(x[k] for x in rows)/len(rows),6) if rows else 0.0
def covered(items,gmap): return {gmap[n] for ids in items for n in ids if n in gmap}
def eval_arm(arm,client,provider,units,meta):
 qs,gmap=questions_and_gold(); by={x.chunk_id:x for x in units}; sparse=BM25Retriever(units,1.5,.75); rows=[]; snapshots=[]
 for i,q in enumerate(qs,1):
  query=q['query']; exp=HybridRetriever._is_experiment_design_query(query); con=HybridRetriever._is_contribution_query(query); depth=80 if exp else 60 if con else 20; routed,signals=HybridRetriever._route_query(query,retrieval_scope='paper',experiment_design_query=exp)
  f=models.Filter(must=[models.FieldCondition(key='canonical_document_id',match=models.MatchValue(value=q['doc']))]); pts=client.query_points(collection(arm),query=norm([provider.embed_query(routed)])[0],query_filter=f,limit=depth,with_payload=True).points
  dense=[RetrievalResult(by[str(x.payload['unit_id'])],float(x.score)) for x in pts]; lex=sparse.retrieve(routed,top_k=depth,retrieval_filter=RetrievalFilter(paper_ids=[q['doc']])); ds={x.chunk.chunk_id:x.score for x in dense}; bs={x.chunk.chunk_id:x.score for x in lex}; fused=reciprocal_rank_fusion(dense,lex)[:depth]
  if not fused or len({x.chunk.chunk_id for x in fused})!=len(fused): raise RuntimeError(f'GOLD_FREE_RUNTIME_VIOLATION:candidates:{q["id"]}')
  top=HybridRetriever._context_candidates(query,fused,top_k=5,retrieval_scope='paper'); context=ContextBuilder(include_neighbors=False,max_characters=10**9,max_tokens=12000).build(top)
  poolids=[meta[x.chunk.chunk_id]['neutral_source_block_ids'] for x in fused]; topids=[meta[x.chunk.chunk_id]['neutral_source_block_ids'] for x in top]; packids=[meta[x.chunk_id]['neutral_source_block_ids'] for x in context]; gold=q['gold']; pg=covered(poolids,gmap); tg=covered(topids,gmap); cg=covered(packids,gmap); ideal=sum(1/math.log2(j+1) for j in range(1,min(len(gold),10)+1))
  row={'id':q['id'],'dataset':q['dataset'],'category':q['category'],'doc':q['doc'],'pool_gold_recall':len(pg&gold)/len(gold),'gold_recall_5':len(tg&gold)/len(gold),'mrr':next((1/j for j,x in enumerate(topids,1) if covered([x],gmap)&gold),0.0),'ndcg10':sum((1 if covered([x],gmap)&gold else 0)/math.log2(j+1) for j,x in enumerate(poolids[:10],1))/ideal,'context_precision':sum(bool(covered([x],gmap)&gold) for x in packids)/len(packids) if packids else 0.0,'context_recall':len(cg&gold)/len(gold)}
  if q['claims'] is None: row.update({'claim_status':'METRIC_NOT_COMPUTABLE','required_claim_coverage@pool':None,'required_claim_coverage@5':None,'all_claims_candidate_covered_rate@pool':None,'all_claims_covered_rate@5':None,'multi_evidence_all_claims_present@pool':None,'multi_evidence_complete_rate@5':None})
  else:
   claims=q['claims']; a=sum(bool(pg&c) for c in claims); b=sum(bool(tg&c) for c in claims); z=sum(bool(cg&c) for c in claims); row.update({'claim_status':'COMPUTABLE','required_claim_coverage@pool':a/len(claims),'required_claim_coverage@5':b/len(claims),'all_claims_candidate_covered_rate@pool':float(a==len(claims)),'all_claims_covered_rate@5':float(b==len(claims)),'multi_evidence_all_claims_present@pool':float(len(claims)>1 and a==len(claims)),'multi_evidence_complete_rate@5':float(len(claims)>1 and b==len(claims)),'candidate_loss':len(claims)-a,'ranking_loss':a-b,'packing_loss':b-z})
  rows.append(row); snapshots.append({'question_id':q['id'],'query_sha256':hashlib.sha256(query.encode()).hexdigest(),'candidate_depth_requested':depth,'candidate_count_actual':len(fused),'routing_signals':signals,'candidates':[{'candidate_unit_id':x.chunk.chunk_id,'canonical_document_id':x.chunk.paper_id,'neutral_source_block_ids':meta[x.chunk.chunk_id]['neutral_source_block_ids'],'source_spans':meta[x.chunk.chunk_id]['source_spans'],'text_sha256':meta[x.chunk.chunk_id]['text_sha256'],'dense_rank':x.dense_rank,'dense_score':ds.get(x.chunk.chunk_id),'bm25_rank':x.sparse_rank,'bm25_score':bs.get(x.chunk.chunk_id),'fused_rank':j,'fused_score':x.score,'text':x.chunk.chunk_text} for j,x in enumerate(fused,1)]}); print(f'{arm} evaluated {i}/176',flush=True)
 comp=[x for x in rows if x['claim_status']=='COMPUTABLE']; multi=[x for x in comp if x['dataset']=='C']; metrics={'candidate_pool_gold_recall':mean(rows,'pool_gold_recall'),'gold_block_recall@5':mean(rows,'gold_recall_5'),'MRR':mean(rows,'mrr'),'NDCG@10':mean(rows,'ndcg10'),'context_gold_precision':mean(rows,'context_precision'),'context_gold_recall':mean(rows,'context_recall'),'required_claim_coverage@pool':mean(comp,'required_claim_coverage@pool'),'required_claim_coverage@5':mean(comp,'required_claim_coverage@5'),'all_claims_candidate_covered_rate@pool':mean(comp,'all_claims_candidate_covered_rate@pool'),'all_claims_covered_rate@5':mean(comp,'all_claims_covered_rate@5'),'multi_evidence_all_claims_present@pool':mean(multi,'multi_evidence_all_claims_present@pool'),'multi_evidence_complete_rate@5':mean(multi,'multi_evidence_complete_rate@5'),'B_D_claim_metrics':'METRIC_NOT_COMPUTABLE'}; losses={k:sum(x.get(k,0) for x in comp) for k in ('candidate_loss','ranking_loss','packing_loss')}; save(OUT/'runs'/f'{arm.lower()}-questions-v1.json',rows); save(OUT/'snapshots'/f'{arm.lower()}-candidate-snapshot-v1.json',{'arm':arm,'questions':len(snapshots),'runtime_contract':FREEZE,'records':snapshots,'global_sha256':h(snapshots)}); save(OUT/'summaries'/f'{arm.lower()}-summary-v1.json',{'arm':arm,'status':'PASS','attempted_questions':len(rows),'metrics':metrics,'losses':losses,'provider':provider.stats})
def main():
 p=argparse.ArgumentParser();p.add_argument('arm',choices=ARMS);p.add_argument('--evaluate',action='store_true');a=p.parse_args();load_dotenv(CANONICAL_ROOT/'.env',override=True);s=Settings();c=QdrantClient(url=s.qdrant_url,api_key=s.qdrant_api_key,check_compatibility=False);pr=q3d.HostedProvider(s);u,m=materialize(a.arm,c,pr);eval_arm(a.arm,c,pr,u,m) if a.evaluate else None
if __name__=='__main__': main()
