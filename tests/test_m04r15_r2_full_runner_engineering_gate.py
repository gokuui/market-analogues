from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from experiments.m04r import m04r15_r2_full_runner_engineering_gate as gate


ROOT = Path(__file__).resolve().parents[1]


def _inputs() -> tuple[dict, dict, dict, dict]:
    return tuple(json.loads((ROOT / path).read_text()) for path in (
        gate.CONTRACT, gate.BOUNDED_CONTRACT, gate.POC_RESULT,
        gate.POC_VERIFICATION,
    ))


def test_r203_lineage_accepts_exact_two_contract_chain() -> None:
    gate._validate_lineage(*_inputs())


@pytest.mark.parametrize("target,path", [
    (0, ("contract_digest",)),
    (1, ("upstream", "r2_contract_digest")),
    (2, ("contract_digest",)),
    (3, ("producer_result_digest",)),
    (3, ("full_query_build_authorized",)),
])
def test_r203_lineage_rejects_mutation(target: int, path: tuple[str, ...]) -> None:
    values = list(_inputs())
    values[target] = deepcopy(values[target])
    owner = values[target]
    for key in path[:-1]:
        owner = owner[key]
    owner[path[-1]] = False if owner[path[-1]] is True else "mutated"
    with pytest.raises(gate.EngineeringGateError, match="chain differs"):
        gate._validate_lineage(*values)
