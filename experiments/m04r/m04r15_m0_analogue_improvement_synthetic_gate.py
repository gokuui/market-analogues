"""Publish the outcome-free M0 analogue-improvement metric synthetic receipt."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import stat
import subprocess
from typing import Any, Mapping, Sequence
from uuid import uuid4

import numpy as np
import pandas as pd

from market_analogues.analogue_improvement import (
    PRIMARY_CLASSES,
    brier_reliability_resolution,
    classwise_calibration_gate,
    coverage_gate,
    evaluate_primary_improvement,
    require_causal_prediction_order,
    require_outcome_mutation_invariance,
    validate_chronological_folds,
    weighted_empirical_crps,
)


SCHEMA = "m04r15-m0-analogue-improvement-synthetic-v1"
CONTRACT = Path("config/analogue-improvement-metric-contract-v1.json")
OUTPUT = Path("config/data/analogues/m04r15/m0-analogue-improvement-synthetic-v1/RESULT.json")
RUNTIME = (
    "config/analogue-improvement-metric-contract-v1.json",
    "src/market_analogues/analogue_improvement.py",
    "experiments/m04r/m04r15_m0_analogue_improvement_synthetic_gate.py",
    "tests/test_analogue_improvement.py",
)
CONTRACT_KEYS = {
    "schema_version", "status", "purpose", "population", "primary_target", "candidate",
    "mandatory_comparators", "primary_formulas", "inference", "calibration",
    "secondary_continuous", "causality", "acceptance", "performance", "claims",
    "contract_digest",
}


class SyntheticGateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SyntheticGateError(message)


def stable(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def snapshot(path: Path) -> bytes:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as error:
        raise SyntheticGateError(f"unsafe file: {path}") from error
    try:
        info = os.fstat(descriptor); require(stat.S_ISREG(info.st_mode), f"regular file required: {path}")
        with os.fdopen(descriptor, "rb", closefd=False) as handle: return handle.read()
    finally: os.close(descriptor)


def git(root: Path, *args: str, binary: bool = False) -> str | bytes:
    result = subprocess.run(("git", *args), cwd=root, capture_output=True, text=not binary, check=False)
    require(result.returncode == 0, f"git {' '.join(args)} failed")
    return result.stdout if binary else result.stdout.strip()


def decode_json(content: bytes, path: Path) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            require(key not in result, f"duplicate JSON key: {path}/{key}")
            result[key] = value
        return result
    try:
        value = json.loads(content, object_pairs_hook=pairs,
                           parse_constant=lambda token: require(False, f"nonfinite JSON: {path}/{token}"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SyntheticGateError(f"invalid JSON: {path}") from error
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def contract(root: Path) -> dict[str, Any]:
    value = decode_json(snapshot(root / CONTRACT), root / CONTRACT)
    require(set(value) == CONTRACT_KEYS, "metric contract field closure differs")
    require(value["contract_digest"] == stable({k:v for k,v in value.items() if k != "contract_digest"}),
            "metric contract digest differs")
    require(value["status"] == "frozen_before_new_untouched_outcomes"
            and value["claims"]["real_forward_outcomes_opened_by_M0"] is False,
            "metric contract boundary differs")
    return value


def fixture() -> tuple[list[str], list[pd.Timestamp], list[str], list[list[float]], list[list[float]], list[list[float]]]:
    labels=[]; cutoffs=[]; folds=[]; candidate=[]; matched=[]; locked=[]
    for month_index, month in enumerate(pd.period_range("2026-01", periods=48, freq="M")):
        for row in range(3):
            position=(month_index+row)%3; labels.append(PRIMARY_CLASSES[position]); cutoffs.append(month.to_timestamp("M")); folds.append(f"fold-{month_index//12}")
            candidate.append([.075,.075,.075]); candidate[-1][position]=.85
            matched.append([1/3,1/3,1/3]); locked.append([.275,.275,.275]); locked[-1][position]=.45
    return labels,cutoffs,folds,candidate,matched,locked


def publish(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); require(not path.exists() and not path.is_symlink(), "create-only result exists")
    temporary=path.parent/f".{path.name}.tmp-{os.getpid()}-{uuid4().hex}"; content=(json.dumps(payload,indent=2,sort_keys=True,allow_nan=False)+"\n").encode()
    descriptor=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    try:
        with os.fdopen(descriptor,"wb",closefd=True) as handle: handle.write(content); handle.flush(); os.fsync(handle.fileno())
        os.link(temporary,path); directory=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY); os.fsync(directory); os.close(directory)
    finally: temporary.unlink(missing_ok=True)


def execute(root: Path) -> dict[str, Any]:
    root=root.resolve(); require(not str(git(root,"status","--porcelain","--untracked-files=all")),"clean committed tree required")
    head=str(git(root,"rev-parse","HEAD")); runtime={name:snapshot(root/name) for name in RUNTIME}
    for name,content in runtime.items(): require(git(root,"show",f"{head}:{name}",binary=True)==content,f"runtime not committed: {name}")
    frozen=contract(root); labels,cutoffs,folds,candidate,matched,locked=fixture(); checks=[]
    decision=evaluate_primary_improvement(labels=labels,candidate_probabilities=candidate,matched_probabilities=matched,locked_probabilities=locked,cutoffs=cutoffs,fold_ids=folds,coverage_passed=True,calibration_passed=True,leakage_passed=True,determinism_passed=True,performance_passed=True,resamples=1000,block_length=6)
    require(decision.passed and all(decision.gates.values()),"synthetic positive decision differs"); checks.append("positive_two_comparator_conjunctive_decision")
    negative=evaluate_primary_improvement(labels=labels,candidate_probabilities=locked,matched_probabilities=matched,locked_probabilities=locked,cutoffs=cutoffs,fold_ids=folds,coverage_passed=True,calibration_passed=True,leakage_passed=True,determinism_passed=True,performance_passed=True,resamples=1000,block_length=6)
    require(not negative.passed and "positive_skill_both_comparators" in negative.reasons,"synthetic negative decision differs"); checks.append("equal_incumbent_failure")
    reversed_decision=evaluate_primary_improvement(labels=list(reversed(labels)),candidate_probabilities=list(reversed(candidate)),matched_probabilities=list(reversed(matched)),locked_probabilities=list(reversed(locked)),cutoffs=list(reversed(cutoffs)),fold_ids=list(reversed(folds)),coverage_passed=True,calibration_passed=True,leakage_passed=True,determinism_passed=True,performance_passed=True,resamples=1000,block_length=6)
    require(decision.brier_skill==reversed_decision.brier_skill and decision.fold_brier_skill==reversed_decision.fold_brier_skill,"row-order invariance differs"); checks.append("row_order_invariance")
    calibration=classwise_calibration_gate(labels,candidate,cutoffs,resamples=1000,block_length=6); require(calibration.passed,"classwise calibration fixture differs"); checks.append("classwise_calibration")
    disclosure=brier_reliability_resolution(labels,candidate); require(all(np.isfinite(list(disclosure.values()))),"Brier decomposition differs"); checks.append("brier_decomposition")
    coverage=coverage_gate(folds,[True]*len(folds),[True]*len(folds)); require(coverage.passed,"coverage gate differs"); checks.append("coverage")
    sessions=pd.bdate_range("2020-01-01","2025-12-31"); fold_contract=[{"fold_id":f"f{i}","start":f"{2020+i}-01-01","end":f"{2020+i}-12-31"} for i in range(4)]
    require(len(validate_chronological_folds(sessions,fold_contract))==4,"fold purge differs"); checks.append("four_fold_sixty_session_purge")
    require_causal_prediction_order(["2026-01-31"],["2026-01-31"],["2026-02-01"],["2026-02-02"]); checks.append("causal_maturity_and_seal_order")
    require_outcome_mutation_invariance([["e1","e2"]],[["e1","e2"]],[[.6,.3,.1]],[[.6,.3,.1]]); checks.append("future_outcome_mutation_invariance")
    values=np.asarray([-1.,0.,2.]); weights=np.asarray([1.,2.,4.]); observed=.5
    naive=float(np.sum(weights*np.abs(values-observed))/weights.sum()-np.sum(weights[:,None]*weights[None,:]*np.abs(values[:,None]-values[None,:]))/(2*weights.sum()**2))
    require(abs(weighted_empirical_crps(values,weights,observed)-naive)<=1e-15,"CRPS oracle differs"); checks.append("weighted_empirical_CRPS")
    state={"schema_version":SCHEMA,"status":"synthetic_metric_gate_passed","passed":True,"implementation_commit":head,"runtime_sha256":{name:sha256(content).hexdigest() for name,content in runtime.items()},"contract_digest":frozen["contract_digest"],"checks":checks,"check_count":len(checks),"positive_decision":asdict(decision),"negative_decision":asdict(negative),"real_forward_outcomes_opened":False,"predictive_claim_authorized":False,"production_promotion_authorized":False,"trading_claim_authorized":False}
    deterministic={**state,"result_digest":stable(state)}; result={**deterministic,"created_at":datetime.now(timezone.utc).isoformat()}; publish(root/OUTPUT,result)
    require({name:snapshot(root/name) for name in RUNTIME}==runtime,"runtime changed during publication"); return result


def main(argv: Sequence[str]|None=None)->int:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--repository",type=Path,default=Path.cwd())
    try: result=execute(parser.parse_args(argv).repository)
    except SyntheticGateError as error: print(f"M0 synthetic gate refused: {error}",file=os.sys.stderr); return 2
    print(json.dumps({"passed":result["passed"],"result_digest":result["result_digest"]},sort_keys=True)); return 0


if __name__=="__main__": raise SystemExit(main())
