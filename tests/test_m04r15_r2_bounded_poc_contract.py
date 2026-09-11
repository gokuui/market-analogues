from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.m04r import verify_m04r15_r2_bounded_poc_contract as verifier
CONTRACT=ROOT/"config/m04r15-r2-bounded-poc-contract-v1.json"


def _write(path:Path,value:dict)->None:
    value["contract_digest"]=verifier._stable({k:v for k,v in value.items() if k!="contract_digest"})
    path.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")


def test_real_poc_contract_reconstructs_selection_without_futures()->None:
    result=verifier.validate(ROOT)
    assert result["passed"] is True and result["selected_query_count"]==32
    assert result["columns_opened"]==["query_case_id"]
    assert result["future_path_store_opened"] is False
    assert result["full_query_build_authorized"] is False


@pytest.mark.parametrize("mutation,match",[
    (lambda c:c["selection"].update(sample_size=33),"selection"),
    (lambda c:c["computation"].update(threshold_changes_authorized=True),"computation"),
    (lambda c:c["verification"].update(repeat_identity=False),"verification"),
    (lambda c:c["claim_boundary"].update(predictive_claim_authorized=True),"claim"),
])
def test_poc_contract_rejects_self_consistent_weakening(tmp_path:Path,mutation,match:str)->None:
    value=deepcopy(json.loads(CONTRACT.read_text()));mutation(value);path=tmp_path/"contract.json";_write(path,value)
    with pytest.raises(verifier.PocContractError,match=match):verifier.validate(ROOT,path)


def test_poc_contract_rejects_upstream_rewrite(tmp_path:Path)->None:
    value=deepcopy(json.loads(CONTRACT.read_text()));value["upstream"]["r202_verification_result_digest"]="0"*64
    path=tmp_path/"contract.json";_write(path,value)
    with pytest.raises(verifier.PocContractError,match="upstream"):verifier.validate(ROOT,path)
