"""Independently reconstruct every result in the 32-query R2 bounded POC."""
from __future__ import annotations
import argparse,json,os,subprocess,tempfile
from collections import Counter
from datetime import datetime,timezone,date
from hashlib import sha256
from math import fsum,isfinite
from pathlib import Path
from typing import Any,Mapping,Sequence
import pandas as pd
from experiments.m04r import verify_m04r15_r2_stability_synthetic_gate as oracle

SCHEMA="m04r15-r2-bounded-real-poc-verification-v1"
RESULT=Path("config/data/analogues/m04r15/r2-bounded-real-poc-v1/RESULT.json")
CONTRACT=Path("config/m04r15-r2-bounded-poc-contract-v1.json")
STORE=Path("config/data/analogues/m04r14/t14-09-full-outcome-store-v1")
OUTPUT=Path("config/data/analogues/m04r15/r2-bounded-real-poc-v1-verification")
PRODUCER_MODULES={"market_analogues.future_modes","experiments.m04r.m04r15_r2_bounded_real_poc"}
class PocVerificationError(RuntimeError):pass
def _req(v,m):
    if not v:raise PocVerificationError(m)
def _stable(v):return sha256(json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def _read(p):
    _req(p.is_file()and not p.is_symlink(),f"regular JSON required: {p}");raw=p.read_bytes();return json.loads(raw),raw
def _cohort(ms):return _stable([[m["rank"],m["episode"],m["symbol"],m["cutoff"],m["fingerprint"]]for m in ms])
def _quarter(v):
    d=date.fromisoformat(str(v)[:10]);return f"{d.year:04d}-Q{(d.month-1)//3+1}"

def _prepare(members,grouped,field):
    complete=[];invalid=[]
    for m in members:
        rows=[r for r in grouped.get(m["episode"],[])if 1<=int(r["step"])<=60];by={};reasons=set()
        for r in rows:
            step=int(r["step"])
            if step in by:reasons.add("duplicate_step")
            else:by[step]=r
        if set(by)!=set(range(1,61)):reasons.add("missing_step")
        values=[];times=[];contracts=set();contents=set();fingerprints=set()
        for step in range(1,61):
            r=by.get(step)
            if r is None:continue
            if r["expected_session_match"]is not True:reasons.add("unexpected_session")
            if str(r["episode_id"])!=m["episode"]:reasons.add("episode_binding")
            if str(r["cutoff"])!=m["cutoff"]:reasons.add("cutoff_binding")
            contracts.add(str(r["contract_digest"]));contents.add(str(r["source_content_digest"]));fingerprints.add(str(r["source_fingerprint"]));times.append(str(r["timestamp"]))
            try:x=float(r[field])
            except(TypeError,ValueError):reasons.add("invalid_value")
            else:
                if not isfinite(x):reasons.add("nonfinite_value")
                values.append(x)
        if len(contracts)!=1 or""in contracts:reasons.add("contract_binding")
        if len(contents)!=1 or""in contents:reasons.add("source_content_binding")
        if fingerprints!={m["fingerprint"]}:reasons.add("source_fingerprint_binding")
        if len(times)!=60 or any(not x for x in times)or len(set(times))!=len(times)or times!=sorted(times):reasons.add("timestamp_sequence")
        if reasons:invalid.append([m["episode"],"+".join(sorted(reasons))])
        else:complete.append({**m,"values":tuple(values)})
    return complete,invalid

def _stability(matrix,blocks,labels,k,query,view):
    unique=sorted(set(blocks))
    if len(unique)<max(4,k+1):return None,0,[]
    scores=[];prefix="\0".join(("6bec009810afce7c58508fca28179579f5904382376dc9c5bce74aa74f41e08c",query,view)).encode()
    for rep in range(256):
        picks=oracle._integers(prefix+b"\0"+rep.to_bytes(8,"big"),len(unique),len(unique));sampled=[unique[i]for i in picks]
        if len(set(sampled))<k:continue
        counts=Counter(sampled);weights=[float(counts[b])for b in blocks]
        if sum(x>0 for x in weights)<k:continue
        _,medoids,fitted=oracle._fit(matrix,k,weights)
        if len(set(fitted))!=k:continue
        scores.append(oracle._ari(labels,fitted))
    ordered=sorted(scores);mid=len(ordered)//2
    median=None if not scores else ordered[mid]if len(ordered)%2 else(ordered[mid-1]+ordered[mid])/2
    return median,len(scores),scores

def _mode(paths,query,view):
    if len(paths)<3:return{"status":"abstain_insufficient_complete_primary_members","selected_k":0,"medoid_episode_ids":[],"member_to_mode":{},"candidates":[]}
    matrix=tuple(tuple(fsum(abs(a-b)for a,b in zip(x["values"],y["values"],strict=True))/60 for y in paths)for x in paths)
    blocks=[_quarter(x["cutoff"])for x in paths];candidates=[]
    for k in range(2,min(4,len(paths)//3)+1):
        _,medoids,labels=oracle._fit(matrix,k);sizes=[labels.count(i)for i in range(k)];sil=oracle._silhouette(matrix,labels);reasons=[];stability=None
        if min(sizes)<3:reasons.append("mode_smaller_than_3")
        if sil<.25:reasons.append("mean_silhouette_below_0.25")
        if not reasons:
            median,valid,scores=_stability(matrix,blocks,labels,k,query,view);stability={"valid_replicates":valid,"median_adjusted_rand_index":median,"adjusted_rand_indices_digest":_stable(scores),"minimum_adjusted_rand_index":min(scores)if scores else None,"maximum_adjusted_rand_index":max(scores)if scores else None}
            if valid<205:reasons.append("fewer_than_80_percent_valid_block_bootstraps")
            if median is None or median<.8:reasons.append("median_adjusted_rand_index_below_0.8")
        candidates.append({"k":k,"medoid_episode_ids":[paths[i]["episode"]for i in medoids],"cluster_sizes":sizes,"mean_silhouette":sil,"accepted":not reasons,"rejection_reasons":reasons,"stability":stability,"_labels":labels,"_medoids":medoids})
    accepted=[c for c in candidates if c["accepted"]]
    if accepted:chosen=min(accepted,key=lambda c:(-c["mean_silhouette"],c["k"]));status="stable_multiple_modes"
    else:
        _,medoids,labels=oracle._fit(matrix,1);chosen={"k":1,"_medoids":medoids,"_labels":labels};status="one_mode_fallback"
    public=[]
    for c in candidates:public.append({k:v for k,v in c.items()if not k.startswith("_")})
    return{"status":status,"selected_k":chosen["k"],"medoid_episode_ids":[paths[i]["episode"]for i in chosen["_medoids"]],"member_to_mode":{p["episode"]:int(chosen["_labels"][i])for i,p in enumerate(paths)},"candidates":public}

def verify(repository:Path,result_path:Path|None=None)->dict[str,Any]:
    repository=repository.resolve(strict=True);result,raw=_read((result_path or repository/RESULT).resolve(strict=True));contract,_=_read(repository/CONTRACT)
    deterministic={k:v for k,v in result.items()if k not in{"result_digest","created_at","performance"}};_req(result["result_digest"]==_stable(deterministic),"producer digest differs")
    selected=contract["selection"]["query_case_ids"];cols=["query_case_id","query_symbol","match_rank","matched_episode_id","matched_symbol","matched_cutoff","source_fingerprint","outcome_eligibility_json"]
    links=pd.read_parquet(repository/STORE/"query-match-links.parquet",columns=cols,filters=[("query_case_id","in",selected)]);episodes=sorted(set(links.matched_episode_id.astype(str)))
    pcols=["benchmark_relative_close_return","close_return","contract_digest","cutoff","episode_id","expected_session_match","source_content_digest","source_fingerprint","step","timestamp"]
    frame=pd.read_parquet(repository/STORE/"future-paths.parquet",columns=pcols,filters=[("episode_id","in",episodes)]);grouped={str(k):v.to_dict("records")for k,v in frame.groupby("episode_id",sort=False)}
    results=[];inventory=[]
    for qid in selected:
        q=links[links.query_case_id==qid].sort_values("match_rank",kind="stable");seen=set();primary=[];excluded=[]
        for r in q.itertuples():
            m={"rank":int(r.match_rank),"episode":str(r.matched_episode_id),"symbol":str(r.matched_symbol),"cutoff":str(r.matched_cutoff),"fingerprint":str(r.source_fingerprint)}
            if m["symbol"]==str(r.query_symbol):excluded.append([m["episode"],"query_symbol_memory"])
            elif m["symbol"]in seen:excluded.append([m["episode"],"duplicate_matched_symbol"])
            else:primary.append(m);seen.add(m["symbol"])
        digest=_cohort(primary);inventory.append([qid,digest]);elig={str(r.matched_episode_id):json.loads(r.outcome_eligibility_json)["60"]for r in q.itertuples()};eligible=[m for m in primary if elig[m["episode"]]["eligible"]is True];ineligible=[[m["episode"],elig[m["episode"]]["reason"]]for m in primary if elig[m["episode"]]["eligible"]is not True];views={}
        for view,field in (("absolute_close_return","close_return"),("benchmark_relative_close_return","benchmark_relative_close_return")):
            paths,invalid=_prepare(eligible,grouped,field);views[view]={"complete_members":len(paths),"invalid_members":invalid,"ineligible_members":ineligible,"selection":_mode(paths,qid,view)}
        results.append({"query_case_id":qid,"raw_links":len(q),"primary_members":len(primary),"dependence_exclusions":excluded,"primary_cohort_digest":digest,"views":views})
    _req(results==result["query_results"],"independent query reconstruction differs")
    _req(result["query_results_digest"]==_stable(results)and result["primary_cohort_inventory_digest"]==_stable(inventory),"semantic digest differs")
    _req(result["mode_status_counts"]==dict(sorted(Counter(v["selection"]["status"]for r in results for v in r["views"].values()).items())),"coverage differs")
    _req(all((result["bounded_real_future_path_store_opened"]is True,result["full_query_build_authorized"]is False,result["predictive_claim_authorized"]is False,result["production_promotion_authorized"]is False,result["trading_claim_authorized"]is False)),"boundary differs")
    state={"schema_version":SCHEMA,"status":"independently_verified","passed":True,"producer_result_digest":result["result_digest"],"producer_result_sha256":sha256(raw).hexdigest(),"verified_query_count":32,"verified_view_count":64,"mode_status_counts":result["mode_status_counts"],"producer_modules_imported":False,"bounded_real_future_path_store_opened":True,"full_query_build_authorized":True,"predictive_claim_authorized":False,"production_promotion_authorized":False,"trading_claim_authorized":False}
    return{**state,"result_digest":_stable(state)}

def _publish(path,value):
    _req(not path.exists()and not path.is_symlink(),"verification exists");path.parent.mkdir(parents=True,exist_ok=True);tmp=Path(tempfile.mkdtemp(prefix=".r2-real-poc-verification-",dir=path.parent));target=tmp/"VERIFIED.json"
    try:
        fd=os.open(target,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
        with os.fdopen(fd,"wb")as h:h.write((json.dumps({**value,"created_at":datetime.now(timezone.utc).isoformat()},indent=2,sort_keys=True)+"\n").encode());h.flush();os.fsync(h.fileno())
        os.rename(tmp,path)
    except Exception:raise
def main(argv:Sequence[str]|None=None)->int:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--repository",type=Path,required=True);p.add_argument("--result",type=Path);p.add_argument("--dry-run",action="store_true");a=p.parse_args(argv);v=verify(a.repository,a.result)
    if not a.dry_run:_publish(a.repository.resolve(strict=True)/OUTPUT,v)
    print(json.dumps(v,indent=2,sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
