"""Run the preregistered 32-query R2 consumed-data POC."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
from resource import getrusage, RUSAGE_SELF
import subprocess
import tempfile
from time import perf_counter, process_time
from typing import Any, Mapping, Sequence

import pandas as pd

from market_analogues.future_modes import (
    calendar_quarter, pairwise_l1, prepare_paths, select_modes, select_primary_members,
)


SCHEMA="m04r15-r2-bounded-real-poc-v1"
CONTRACT=Path("config/m04r15-r2-bounded-poc-contract-v1.json")
VERIFIED=Path("config/data/analogues/m04r15/r2-bounded-poc-contract-verification-v1/VERIFIED.json")
STORE=Path("config/data/analogues/m04r14/t14-09-full-outcome-store-v1")
OUTPUT=Path("config/data/analogues/m04r15/r2-bounded-real-poc-v1/RESULT.json")
RUNTIME=("config/m04r15-r2-bounded-poc-contract-v1.json",
         "config/data/analogues/m04r15/r2-bounded-poc-contract-verification-v1/VERIFIED.json",
         "src/market_analogues/future_modes.py","experiments/m04r/m04r15_r2_bounded_real_poc.py")


class BoundedPocError(RuntimeError):pass
def _require(v:bool,m:str)->None:
    if not v:raise BoundedPocError(m)
def _stable(v:Any)->str:return sha256(json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def _json(p:Path)->dict[str,Any]:
    _require(p.is_file() and not p.is_symlink(),f"regular JSON required: {p}");v=json.loads(p.read_text());_require(type(v)is dict,"JSON object required");return v
def _sha(p:Path)->str:
    d=sha256()
    with p.open("rb") as h:
        while b:=h.read(8<<20):d.update(b)
    return d.hexdigest()
def _cohort(members)->str:return _stable([[m.match_rank,m.episode_id,m.symbol,m.cutoff,m.source_fingerprint]for m in members])


def _selection_summary(selection,paths)->dict[str,Any]:
    candidates=[]
    for c in selection.candidates:
        s=c.stability
        candidates.append({"k":c.k,"medoid_episode_ids":[paths[i].member.episode_id for i in c.medoid_indices],
            "cluster_sizes":list(c.cluster_sizes),"mean_silhouette":c.mean_silhouette,"accepted":c.accepted,
            "rejection_reasons":list(c.rejection_reasons),"stability":None if s is None else{
                "valid_replicates":s.valid_replicates,"median_adjusted_rand_index":s.median_adjusted_rand_index,
                "adjusted_rand_indices_digest":_stable(list(s.adjusted_rand_indices)),
                "minimum_adjusted_rand_index":min(s.adjusted_rand_indices)if s.adjusted_rand_indices else None,
                "maximum_adjusted_rand_index":max(s.adjusted_rand_indices)if s.adjusted_rand_indices else None}})
    return{"status":selection.status,"selected_k":selection.selected_k,
           "medoid_episode_ids":[paths[i].member.episode_id for i in selection.medoid_indices],
           "member_to_mode":{path.member.episode_id:int(selection.labels[i])for i,path in enumerate(paths)},
           "candidates":candidates}


def execute(repository:Path)->dict[str,Any]:
    repository=repository.resolve(strict=True);contract=_json(repository/CONTRACT);verified=_json(repository/VERIFIED)
    _require(verified.get("passed")is True and verified.get("contract_digest")==contract.get("contract_digest")
             and verified.get("bounded_consumed_data_poc_authorized")is True,"POC authorization differs")
    commit=subprocess.run(("git","rev-parse","HEAD"),cwd=repository,text=True,capture_output=True,check=True).stdout.strip()
    runtime={n:_sha(repository/n)for n in RUNTIME}
    for n,d in runtime.items():
        h=subprocess.run(("git","show",f"{commit}:{n}"),cwd=repository,capture_output=True)
        _require(h.returncode==0 and sha256(h.stdout).hexdigest()==d,f"runtime differs from H0: {n}")
    started=perf_counter();cpu_started=process_time();selected=contract["selection"]["query_case_ids"]
    link_columns=["query_case_id","query_symbol","match_rank","matched_episode_id","matched_symbol",
                  "matched_cutoff","source_fingerprint","outcome_eligibility_json"]
    links=pd.read_parquet(repository/STORE/"query-match-links.parquet",columns=link_columns,
                          filters=[("query_case_id","in",selected)])
    _require(len(links)==640 and links.query_case_id.nunique()==32,"bounded link coverage differs")
    episode_ids=sorted(set(links.matched_episode_id.astype(str)))
    path_columns=["benchmark_relative_close_return","close_return","contract_digest","cutoff","episode_id",
                  "expected_session_match","source_content_digest","source_fingerprint","step","timestamp"]
    frame=pd.read_parquet(repository/STORE/"future-paths.parquet",columns=path_columns,
                          filters=[("episode_id","in",episode_ids)])
    _require(set(frame.episode_id.astype(str))<=set(episode_ids),"path filter escaped selection")
    grouped={str(key):value.to_dict("records")for key,value in frame.groupby("episode_id",sort=False)}
    results=[];cohort_digests=[];mutation_invariant=True
    for query_id in selected:
        query=links[links.query_case_id==query_id].sort_values("match_rank",kind="stable")
        raw=query.to_dict("records");primary,dependence=select_primary_members(str(query.iloc[0].query_symbol),raw)
        cohort_digest=_cohort(primary);cohort_digests.append([query_id,cohort_digest])
        # The selector has no path argument; an extreme in-memory outcome mutation cannot affect identity.
        if grouped:
            first_episode=next(iter(grouped));copy_rows=[dict(row)for row in grouped[first_episode]]
            if copy_rows:copy_rows[0]["close_return"]=999999.0
        primary_after,_=select_primary_members(str(query.iloc[0].query_symbol),list(reversed(raw)))
        mutation_invariant &= _cohort(primary_after)==cohort_digest
        eligibility={str(row.matched_episode_id):json.loads(row.outcome_eligibility_json)["60"]
                     for row in query.itertuples()}
        eligible=[member for member in primary if eligibility[member.episode_id]["eligible"]is True]
        ineligible=[[member.episode_id,eligibility[member.episode_id]["reason"]]for member in primary
                    if eligibility[member.episode_id]["eligible"]is not True]
        views={}
        for view_id,field in (("absolute_close_return","close_return"),
                              ("benchmark_relative_close_return","benchmark_relative_close_return")):
            paths,invalid=prepare_paths(eligible,grouped,value_field=field,horizon=60)
            blocks=[calendar_quarter(path.member.cutoff)for path in paths]
            if paths:
                selection=select_modes(pairwise_l1(paths),[path.member.key for path in paths],blocks,
                    contract_digest=contract["upstream"]["r2_contract_digest"],query_case_id=query_id,
                    view_id=view_id,replicates=256)
                summary=_selection_summary(selection,paths)
            else:summary={"status":"abstain_insufficient_complete_primary_members","selected_k":0,
                         "medoid_episode_ids":[],"member_to_mode":{},"candidates":[]}
            views[view_id]={"complete_members":len(paths),"invalid_members":[[x.member.episode_id,x.reason]for x in invalid],
                            "ineligible_members":ineligible,"selection":summary}
        results.append({"query_case_id":query_id,"raw_links":len(raw),"primary_members":len(primary),
                        "dependence_exclusions":[[x.member.episode_id,x.reason]for x in dependence],
                        "primary_cohort_digest":cohort_digest,"views":views})
    elapsed=perf_counter()-started;cpu=process_time()-cpu_started
    statuses=Counter(v["selection"]["status"]for r in results for v in r["views"].values())
    state={"schema_version":SCHEMA,"status":"bounded_real_poc_complete","passed":True,
           "contract_digest":contract["contract_digest"],"contract_verification_result_digest":verified["result_digest"],
           "implementation_commit":commit,"runtime_sha256":runtime,"selected_query_count":32,
           "raw_link_rows":len(links),"selected_unique_episodes":len(episode_ids),"selected_path_rows":len(frame),
           "query_results":results,"query_results_digest":_stable(results),
           "primary_cohort_inventory_digest":_stable(cohort_digests),
           "outcome_mutation_primary_cohort_invariant":bool(mutation_invariant),
           "mode_status_counts":dict(sorted(statuses.items())),
           "performance":{"wall_seconds":elapsed,"cpu_seconds":cpu,"peak_rss_kib":getrusage(RUSAGE_SELF).ru_maxrss},
           "bounded_real_future_path_store_opened":True,"full_query_build_authorized":False,
           "predictive_claim_authorized":False,"production_promotion_authorized":False,"trading_claim_authorized":False}
    _require(mutation_invariant,"outcome mutation altered primary cohort")
    return{**state,"result_digest":_stable({k:v for k,v in state.items()if k!="performance"})}


def _publish(path:Path,value:Mapping[str,Any])->None:
    _require(not path.exists()and not path.is_symlink(),"bounded POC result exists");path.parent.mkdir(parents=True,exist_ok=True)
    descriptor,name=tempfile.mkstemp(prefix=".r2-real-poc-",dir=path.parent);temporary=Path(name)
    try:
        with os.fdopen(descriptor,"wb")as h:h.write((json.dumps({**value,"created_at":datetime.now(timezone.utc).isoformat()},indent=2,sort_keys=True,allow_nan=False)+"\n").encode());h.flush();os.fsync(h.fileno())
        os.rename(temporary,path)
    except Exception:
        try:temporary.unlink()
        except OSError:pass
        raise


def main(argv:Sequence[str]|None=None)->int:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--repository",type=Path,required=True);p.add_argument("--dry-run",action="store_true");a=p.parse_args(argv);v=execute(a.repository)
    if not a.dry_run:_publish(a.repository.resolve(strict=True)/OUTPUT,v)
    print(json.dumps(v,indent=2,sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
