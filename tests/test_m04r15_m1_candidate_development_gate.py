from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.m04r import m04r15_m1_candidate_development_gate as gate
from market_analogues.analogue_candidate import MIXTURE_WEIGHTS


ROOT = Path(__file__).resolve().parents[1]


def test_strict_json_decoder_rejects_duplicate_nonfinite_and_array(tmp_path: Path) -> None:
    with pytest.raises(gate.CandidateGateError, match="duplicate"):
        gate.decode_json(b'{"a":1,"a":2}', tmp_path / "x")
    with pytest.raises(gate.CandidateGateError, match="nonfinite"):
        gate.decode_json(b'{"a":NaN}', tmp_path / "x")
    with pytest.raises(gate.CandidateGateError, match="object"):
        gate.decode_json(b'[]', tmp_path / "x")


def test_all_consumed_inputs_are_manifest_bound_and_keep_claims_false() -> None:
    content, decoded = gate.validate_inputs(ROOT)
    assert set(content) == set(gate.INPUTS)
    assert decoded["m0_verification"]["predictive_claim_authorized"] is False
    assert decoded["r1b_lock"]["claims"]["predictive_claim_authorized"] is False
    assert decoded["final_verification"]["production_promotion_authorized"] is False


def test_fixed_candidate_is_unique_coarse_simplex_robust_optimum() -> None:
    content, _ = gate.validate_inputs(ROOT)
    data = gate.load_development(content)
    queries, _, _ = data
    candidates, leave_one_out = gate.grid(data)
    selected = candidates[0]
    assert len(queries) == 3936 and len(candidates) == 286
    assert selected["weights"] == dict(MIXTURE_WEIGHTS)
    assert selected["minimum_fold_two_comparator_skill"] > .014
    assert len(selected["metrics"]) == 6
    assert sum(row["rows"] for row in selected["metrics"][:-1]) == 2595
    assert all(row["skill_vs_matched"] > 0 and row["skill_vs_locked"] > 0
               for row in selected["metrics"][:-1])
    assert all(row["log_loss_difference_vs_matched"] <= 0
               and row["log_loss_difference_vs_locked"] <= 0
               for row in selected["metrics"])
    assert all(row["held_skill_vs_matched"] > 0 and row["held_skill_vs_locked"] > 0
               for row in leave_one_out)
    assert candidates[0]["minimum_fold_two_comparator_skill"] \
        > candidates[1]["minimum_fold_two_comparator_skill"]


def test_publication_is_create_only(tmp_path: Path) -> None:
    target = tmp_path / "FROZEN.json"
    gate.publish(target, {"passed": True})
    assert json.loads(target.read_text()) == {"passed": True}
    with pytest.raises(gate.CandidateGateError, match="create-only"):
        gate.publish(target, {"passed": True})
