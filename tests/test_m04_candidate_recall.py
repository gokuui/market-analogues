from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from market_analogues.m04_candidate_recall import (
    M04CandidateRecallError,
    M04CaseResult,
    aggregate_m04_cases,
    load_completed_m04_case,
    load_m04_candidate_recall_spec,
    write_m04_case,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "config" / "m04-candidate-recall-contract.yaml"


def test_official_m04_contract_binds_one_development_and_23_holdout_cases() -> None:
    spec = load_m04_candidate_recall_spec(CONTRACT)
    assert spec.digest == "02cd4b2acd5c5f945c847a66edc3da89ddd34c51c8e9fcda456d8e753063d494"
    assert len(spec.payload["holdout_episode_ids"]) == 23
    assert spec.payload["development_case"]["episode_id"] not in spec.payload["holdout_episode_ids"]
    assert spec.payload["retrieval"]["candidate_pool"] == 20_000
    assert spec.payload["acceptance"]["minimum_authority_top20_recall_per_case"] == .95
    assert spec.payload["scope"]["real_forward_outcomes_accessed"] is False


def test_m04_contract_rejects_a_lower_recall_floor(tmp_path: Path) -> None:
    payload = yaml.safe_load(CONTRACT.read_text())
    payload["acceptance"]["minimum_authority_top20_recall_per_case"] = .94
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(payload))
    with pytest.raises(M04CandidateRecallError, match="cannot be below 0.95"):
        load_m04_candidate_recall_spec(path)


def _case(spec, episode_id: str, recall: float = .95) -> M04CaseResult:
    return M04CaseResult(True, {
        "schema_version": "m04-candidate-recall-result-v1",
        "contract_id": spec.payload["contract_id"],
        "contract_digest": spec.digest,
        "query_episode_id": episode_id,
        "dataset_id": "nse",
        "symbol": "DEMO",
        "cutoff": "2020-01-01T00:00:00",
        "authority_top20_recall": recall,
        "candidate_pool_returned": 20_000,
        "elapsed_seconds": 1.0,
        "peak_rss_mb": 10.0,
        "rows_considered": 100,
    }, (), ())


def test_completed_case_is_digest_checked_and_resumable(tmp_path: Path) -> None:
    spec = load_m04_candidate_recall_spec(CONTRACT)
    episode_id = spec.payload["holdout_episode_ids"][0]
    machine = tmp_path / f"{episode_id}.json"
    write_m04_case(_case(spec, episode_id), machine, tmp_path / "case.html")
    loaded = load_completed_m04_case(machine, spec, episode_id)
    assert loaded is not None and loaded.passed
    payload = json.loads(machine.read_text())
    payload["metrics"]["authority_top20_recall"] = 0.0
    machine.write_text(json.dumps(payload))
    with pytest.raises(M04CandidateRecallError, match="digest mismatch"):
        load_completed_m04_case(machine, spec, episode_id)


def test_matrix_requires_every_locked_case(tmp_path: Path) -> None:
    spec = load_m04_candidate_recall_spec(CONTRACT)
    for episode_id in spec.payload["holdout_episode_ids"]:
        write_m04_case(_case(spec, episode_id), tmp_path / f"{episode_id}.json", tmp_path / f"{episode_id}.html")
    passed, metrics, failures = aggregate_m04_cases(spec, tmp_path)
    assert passed and not failures
    assert metrics["completed_cases"] == 23
    missing = spec.payload["holdout_episode_ids"][-1]
    (tmp_path / f"{missing}.json").unlink()
    passed, metrics, failures = aggregate_m04_cases(spec, tmp_path)
    assert not passed
    assert metrics["completed_cases"] == 22
    assert failures == (f"missing case {missing}",)
