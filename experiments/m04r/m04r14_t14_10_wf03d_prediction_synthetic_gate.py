"""Run D4 prediction/access mutation gates without real query outcomes."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Sequence

import numpy as np
import pandas as pd

from market_analogues.types import stable_hash
from market_analogues.walk_forward_predictions import (
    pointwise_path_prediction,
    rank_weights,
    route_prediction,
)

from experiments.m04r import m04r14_t14_10_wf03d_prediction_store as producer


SCHEMA="m04r14-t14-10-wf03d-prediction-synthetic-verification-v2"
OUTPUT=Path("config/data/analogues/m04r14/t14-10-wf03d-prediction-synthetic-v2")


class SyntheticPredictionError(RuntimeError): pass


def _sha(path: Path)->str:
    digest=sha256();
    with path.open("rb") as handle:
        while block:=handle.read(8<<20): digest.update(block)
    return digest.hexdigest()


def _fixtures(future_label: str, future_value: float)->tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    registry=pd.DataFrame([{"query_id":"q","case_id":"case","cutoff":"2020-03-31T00:00:00","month":"2020-03","fold_id":"development","scored":True}])
    links=[]; outcomes=[]; eligibility=[]
    for method in ("composite","price_only","deterministic_random","recent_return_volatility"):
        for rank in range(1,21):
            links.append({"query_id":"q","method":method,"rank":rank,"matched_episode_id":f"e{rank}","distance_hex":float(rank/10).hex()})
            for horizon in producer.outcome_store.HORIZONS:
                eligibility.append({"query_id":"q","method":method,"rank":rank,"horizon_sessions":horizon,"eligible":rank!=20,"reason":"eligible" if rank!=20 else "outcome_not_yet_observable"})
    for rank in range(1,21):
        for horizon in producer.outcome_store.HORIZONS:
            outcomes.append({"episode_id":f"e{rank}","horizon_sessions":horizon,"barrier_label":future_label if rank==20 else ("favorable_first" if rank%2 else "adverse_first"),"close_return":future_value if rank==20 else rank/100,"benchmark_relative_return":rank/200,"maximum_favorable_excursion":rank/50,"maximum_adverse_excursion":-rank/70})
    return registry,pd.DataFrame(links),pd.DataFrame(outcomes),pd.DataFrame(eligibility)


def execute(repository: Path, output: Path)->dict[str,Any]:
    repository=repository.resolve(strict=True)
    if subprocess.run(["git","status","--porcelain","--untracked-files=all"],cwd=repository,text=True,capture_output=True,check=True).stdout:
        raise SyntheticPredictionError("clean worktree required")
    checks=[]
    def check(name: str, condition: bool)->None:
        if not condition: raise SyntheticPredictionError(f"synthetic check failed: {name}")
        checks.append(name)
    check("locked_rank_weights",np.array_equal(rank_weights([1,11,21],weighted=True),[1,.5,.25]))
    prediction=route_prediction(["favorable_first","censored",None],[1,2,3],weighted=True)
    check("route_exclusion_accounting",prediction.eligible_rows==1 and prediction.censored_rows==1 and prediction.unavailable_rows==1)
    first=_fixtures("favorable_first",999.0); second=_fixtures("no_touch",-999.0)
    p1=producer._primary_predictions(*first,[]); p2=producer._primary_predictions(*second,[])
    check("ineligible_future_primary_mutation",p1.equals(p2))
    c1=producer._continuous_predictions(*first); c2=producer._continuous_predictions(*second)
    check("ineligible_future_continuous_mutation",c1.equals(c2))
    prior=pd.DataFrame([
        {"query_id":"old","completion_timestamp":"2020-03-31T00:00:00","barrier_label":"favorable_first","regime":"r"},
        {"query_id":"future","completion_timestamp":"2020-04-01T00:00:00","barrier_label":"adverse_first","regime":"r"},
    ])
    baseline=producer._baseline_predictions(first[0],prior,"r")
    check("completion_equality_included_future_excluded",set(baseline.prior_eligible_rows)=={1})
    matrix=np.full((3,126),np.nan); matrix[:,0]=[3,1,2]; ranks=[1,3,7]
    median,count,ess=producer._path_matrix_summary(matrix,ranks,True)
    scalar=pointwise_path_prediction([3,1,2],ranks,weighted=True)
    check("vector_path_scalar_formula",median[0]==scalar[0] and count[0]==scalar[1] and abs(ess[0]-scalar[2])<=4e-15)
    check("missing_path_abstains_step",np.isnan(median[1]) and count[1]==0 and ess[1]==0)
    with tempfile.TemporaryDirectory() as raw:
        root=Path(raw)/"month-2024-01"; root.mkdir()
        for name in (*producer.PREDICTION_FILES,"PREDICTIONS_SEALED.json","query-outcomes.parquet"): (root/name).touch()
        try: producer._validate_existing_month(root,"2024-01",True)
        except producer.WalkForwardPredictionError: rejected=True
        else: rejected=False
    check("final_outcome_layout_rejected",rejected)
    head=subprocess.run(["git","rev-parse","HEAD"],cwd=repository,text=True,capture_output=True,check=True).stdout.strip()
    state={"schema_version":SCHEMA,"status":"verified","passed":True,"implementation_h0":head,
           "producer_sha256":_sha(repository/"experiments/m04r/m04r14_t14_10_wf03d_prediction_store.py"),
           "kernel_sha256":_sha(repository/"src/market_analogues/walk_forward_predictions.py"),
           "checks":checks,"check_count":len(checks),"real_query_outcomes_accessed":False,"final_period_result_opened":False,
           "production_promotion_authorized":False}
    value={**state,"result_digest":stable_hash(state),"created_at":datetime.now(timezone.utc).isoformat()}
    if output.exists(): raise SyntheticPredictionError("synthetic output exists")
    output.parent.mkdir(parents=True,exist_ok=True); temporary=Path(tempfile.mkdtemp(prefix=f".{output.name}.",dir=output.parent))
    descriptor=os.open(temporary/"VERIFIED.json",os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
    with os.fdopen(descriptor,"w") as handle: json.dump(value,handle,indent=2,sort_keys=True,allow_nan=False); handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary,output); return value


def main(argv: Sequence[str]|None=None)->int:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--repository",required=True,type=Path); parser.add_argument("--output",type=Path); args=parser.parse_args(argv)
    result=execute(args.repository,(args.output or args.repository/OUTPUT).resolve()); print(json.dumps(result,indent=2,sort_keys=True)); return 0


if __name__=="__main__": raise SystemExit(main())
