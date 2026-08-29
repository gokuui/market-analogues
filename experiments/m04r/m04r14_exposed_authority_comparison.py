"""Marker-first comparison of qualified P8 results with exposed authorities."""
from __future__ import annotations

import argparse, json, os, subprocess
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.m04r import m04r14_all60_contract as contract
from experiments.m04r import verify_m04r14_throughput_poc as throughput_verifier

SCHEMA = "m04r14-exposed-authority-comparison-v1"
CANDIDATE = Path("config/data/analogues/m04r14/throughput-development-poc-v1")
OUTPUT = Path("config/data/analogues/m04r14/exposed-authority-comparison-v1")
AUTHORITY = Path("config/data/analogues/m04r11/authorities-sealed-v4")
AUTHORITY_VERIFICATION = Path("config/data/analogues/m04r11/authority-verification-v4/m04r11-authority-verification.json")
STABLE_CERTIFICATE_FIELDS = (
    "contract_digest", "eligible_candidates", "exact_evaluated", "generation_id",
    "input_digest", "maximum_quantized_bound_excess", "minimum_native_pruned_bound",
    "native_bound_accounting", "query_episode_id", "safely_pruned", "schema_version",
    "stop_threshold", "stopped_early", "threshold_closure_passes",
)

class ComparisonError(RuntimeError): pass

def _read(path: Path) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file(): raise ComparisonError(f"regular file required: {path}")
    raw=path.read_bytes()
    def pairs(values):
        out={}
        for k,v in values:
            if k in out: raise ComparisonError(f"duplicate JSON key: {k}")
            out[k]=v
        return out
    value=json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda x: (_ for _ in ()).throw(ComparisonError(f"non-finite JSON: {x}")))
    if type(value) is not dict: raise ComparisonError("JSON object required")
    return value,raw

def _sha(raw: bytes) -> str: return sha256(raw).hexdigest()
def _digest(value: Any) -> str: return contract.stable_digest(value)

def _atomic(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink(): raise ComparisonError(f"create-only target exists: {path}")
    fd=os.open(path, os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o644)
    with os.fdopen(fd,"wb") as h:
        h.write(contract.canonical_bytes(value)+b"\n"); h.flush(); os.fsync(h.fileno())

def compare_case(candidate: Mapping[str, Any], authority: Mapping[str, Any]) -> dict[str,bool]:
    cc,ac=candidate.get("certificate",{}),authority.get("certificate",{})
    return {
        "case_id_equal": candidate.get("registry_case_id")==authority.get("registry_case_id"),
        "query_id_equal": candidate.get("query_episode_id")==authority.get("query_episode_id"),
        "ordered_matches_equal": candidate.get("matches")==authority.get("matches"),
        "stock_prefix_equal": candidate.get("query_stock_prefix")==authority.get("query_stock_prefix"),
        "benchmark_prefix_equal": candidate.get("query_benchmark_prefix")==authority.get("query_benchmark_prefix"),
        "stable_certificate_fields_equal": all(cc.get(k)==ac.get(k) for k in STABLE_CERTIFICATE_FIELDS),
        "candidate_gate_passed": candidate.get("gate_passed") is True,
        "authority_gate_passed": authority.get("gate_passed") is True,
    }

def execute(repository: Path) -> dict[str,Any]:
    repository=repository.resolve(strict=True); output=repository/OUTPUT
    if output.exists() or output.is_symlink(): raise ComparisonError("comparison root exists")
    if subprocess.run(["git","status","--porcelain"],cwd=repository,text=True,capture_output=True,check=True).stdout:
        raise ComparisonError("comparison launch requires clean Git")
    head=subprocess.run(["git","rev-parse","HEAD"],cwd=repository,text=True,capture_output=True,check=True).stdout.strip()
    candidate_state=throughput_verifier.verify(repository/CANDIDATE,repository=repository)
    output.mkdir(parents=True)
    marker_state={"schema_version":SCHEMA,"status":"results_opened",
        "git_head":head,"candidate_result_digest":candidate_state["candidate_result_digest"],
        "candidate_verification_digest":candidate_state["result_digest"],
        "authority_root":str((repository/AUTHORITY).resolve()),
        "authority_or_outcome_read_before_marker":False}
    marker_state["marker_digest"]=_digest(marker_state)
    marker={**marker_state,"created_at":datetime.now(timezone.utc).isoformat()}
    _atomic(output/"RESULTS_OPENED.json",marker)
    verification,vraw=_read(repository/AUTHORITY_VERIFICATION)
    if verification.get("failures")!=[] or verification.get("result_digest")!="99d11756ed7542714635eca3fb75a43faed93881121d1f22d8d87426f4d6b190":
        raise ComparisonError("authority verification differs")
    candidates={}
    for p in (repository/CANDIDATE/"cases").glob("*.json"):
        d,raw=_read(p); candidates[d["registry_case_id"]]=(d,_sha(raw))
    authorities={}
    for p in (repository/AUTHORITY/"cases").glob("*.json"):
        d,raw=_read(p); authorities[d["registry_case_id"]]=(d,_sha(raw))
    if len(candidates)!=60 or set(candidates)!=set(authorities): raise ComparisonError("case sets differ")
    rows=[]
    for case_id in sorted(candidates):
        c,csha=candidates[case_id]; a,asha=authorities[case_id]; gates=compare_case(c,a)
        rows.append({"case_id":case_id,"query_id":c["query_episode_id"],
            "matches":len(c["matches"]),"candidate_sha256":csha,"authority_sha256":asha,
            "gates":gates,"passed":all(gates.values())})
    state={"schema_version":SCHEMA,"status":"complete","marker_digest":marker["marker_digest"],
        "git_head":head,"candidate_result_digest":candidate_state["candidate_result_digest"],
        "candidate_verification_digest":candidate_state["result_digest"],
        "authority_verification_digest":verification["result_digest"],
        "authority_verification_sha256":_sha(vraw),"cases":60,"ordered_matches":1200,
        "all_passed":all(r["passed"] for r in rows),"rows":rows,
        "production_promotion_authorized":False}
    state["result_digest"]=_digest(state)
    result={**state,"created_at":datetime.now(timezone.utc).isoformat()}
    _atomic(output/"COMPARISON.json",result)
    if not state["all_passed"]: raise ComparisonError("authority comparison failed")
    return result

def main(argv: Sequence[str]|None=None)->int:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--repository",type=Path,required=True)
    result=execute(p.parse_args(argv).repository); print(json.dumps(result,indent=2,sort_keys=True)); return 0
if __name__=="__main__": raise SystemExit(main())
