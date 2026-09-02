from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.m04r import m04r14_t14_10_wf03_combined_batch as subject
from experiments.m04r import m04r14_t14_10_wf03_feasibility as base
from experiments.m04r import m04r14_t14_10_wf03b_dtw_component_ladder as ladder


def preregistration() -> dict:
    return {
        "preregistration_digest": "pre",
        "contract": {"digest": "contract", "schema_version": "staged"},
    }


def row() -> dict:
    return {
        "episode_id": "a" * 24, "case_id": "case-a", "symbol": "AAA",
        "cutoff": "2020-01-31T00:00:00", "fold_id": "development",
        "fold_role": "development", "scored": True,
    }


def valid_case() -> dict:
    certificate = {
        "schema_version": "staged",
        "query_episode_id": "a" * 24,
        "packed_generation_id": base.GENERATION_ID,
        "dtw_generation_id": ladder.DTW_GENERATION_ID,
        "contract_digest": "contract",
        "input_digest": "input",
        "eligible_candidates": 100,
        "seed_rows": 20,
        "rigid_bound_evaluated": 100,
        "rigid_bound_admitted": 80,
        "dtw_bound_evaluated": 75,
        "combined_bound_admitted": 30,
        "exact_evaluated": 40,
        "native_bound_pruned": 0,
        "maximum_bound_excess": 0.0,
        "seed_threshold": 1.0,
        "final_threshold": 0.19,
        "minimum_rigid_pruned": 1.1,
        "minimum_combined_pruned": 1.2,
    }
    value = {
        "schema_version": "m04r14-wf03-combined-batch-case-v1",
        "status": "complete", "query_id": "a" * 24, "case_id": "case-a",
        "symbol": "AAA", "cutoff": "2020-01-31T00:00:00",
        "fold_id": "development", "fold_role": "development", "scored": True,
        "preregistration_digest": "pre",
        "packed_generation_id": base.GENERATION_ID,
        "dtw_generation_id": ladder.DTW_GENERATION_ID,
        "contract_digest": "contract", "certificate": certificate,
        "matches": [{
            "episode_id": f"{number:024x}", "symbol": f"S{number:02d}",
            "cutoff": "2019-01-01T00:00:00",
            "distance_hex": float(number / 100).hex(), "quality_tier": "A",
        } for number in range(20)],
        "outcomes_or_labels_used": False,
        "historical_walk_forward_query_outcomes_opened": False,
        "final_period_result_opened": False,
    }
    certificate["result_digest"] = subject._certificate_result_digest(
        certificate, value["matches"]
    )
    value["semantic_digest"] = base.stable_hash(
        subject._case_semantic_state(value)
    )
    return base._sealed(value, "case_digest")


def test_case_path_accepts_only_episode_ids(tmp_path: Path) -> None:
    assert subject._case_path(tmp_path, "b" * 24) == tmp_path / f"{'b' * 24}.json"
    with pytest.raises(subject.CombinedBatchError, match="query ID"):
        subject._case_path(tmp_path, "../unsafe")


def test_valid_case_closes_all_structural_and_semantic_gates() -> None:
    value = valid_case()
    assert subject._validate_case(value, row(), preregistration()) == value


@pytest.mark.parametrize("mutation", (
    lambda value: value["matches"].append(dict(value["matches"][0])),
    lambda value: value["matches"][1].update(symbol="S00"),
    lambda value: value["certificate"].update(final_threshold=1.1),
    lambda value: value["certificate"].update(minimum_rigid_pruned=1.0),
    lambda value: value.update(outcomes_or_labels_used=True),
    lambda value: value.update(preregistration_digest="changed"),
))
def test_case_validation_rejects_semantic_drift(mutation) -> None:
    value = valid_case()
    value.pop("case_digest")
    mutation(value)
    value["semantic_digest"] = base.stable_hash(
        subject._case_semantic_state(value)
    )
    value = base._sealed(value, "case_digest")
    with pytest.raises(subject.CombinedBatchError, match="case differs"):
        subject._validate_case(value, row(), preregistration())


def test_existing_invalid_receipt_is_refused_not_replaced(tmp_path: Path) -> None:
    path = subject._case_path(tmp_path, row()["episode_id"])
    path.write_text("{}")
    with pytest.raises(Exception):
        subject._existing_case(path, row(), preregistration())


def test_dtw_file_identity_detects_metadata_change(tmp_path: Path, monkeypatch) -> None:
    generation = tmp_path / "generations" / ladder.DTW_GENERATION_ID
    generation.mkdir(parents=True)
    for name in ("manifest.json", "dtw-samples.bin", "dtw-overflow-samples.bin"):
        (generation / name).write_bytes(name.encode())
    monkeypatch.setattr(ladder, "DTW_ROOT_RELATIVE", Path("."))
    first = subject._dtw_identity(tmp_path)
    (generation / "dtw-samples.bin").write_bytes(b"changed")
    second = subject._dtw_identity(tmp_path)
    assert first["digest"] != second["digest"]
