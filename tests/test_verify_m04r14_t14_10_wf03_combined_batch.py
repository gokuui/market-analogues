from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.m04r import verify_m04r14_t14_10_wf03_combined_batch as subject


def test_rerun_sample_is_deterministic_and_fold_stratified() -> None:
    rows = [{
        "fold_id": fold, "episode_id": f"{fold * 10 + value:024x}"
    } for fold in range(6) for value in range(5)]
    first = subject.select_rerun_sample(rows)
    assert first == subject.select_rerun_sample(list(reversed(rows)))
    assert len(first) == 6
    assert len(set(first)) == 6


def test_lookup_positions_resolves_ids_independently() -> None:
    import numpy as np
    values = np.asarray([
        np.void(bytes.fromhex("02" * 12)), np.void(bytes.fromhex("01" * 12)),
    ], dtype="V12")
    order = np.argsort(values, kind="stable")
    positions = subject._lookup_positions(
        values[order], order, ["02" * 12, "01" * 12],
    )
    assert positions.tolist() == [0, 1]
    with pytest.raises(subject.CombinedBatchVerificationError, match="absent"):
        subject._lookup_positions(values[order], order, ["03" * 12])


def test_inventory_normalizes_distinct_physical_layouts() -> None:
    import numpy as np
    main = np.zeros(2, dtype=np.dtype(subject.INVENTORY_DTYPE.descr + [
        ("samples", "<f2", (4,)),
    ]))
    overflow = np.zeros(1, dtype=np.dtype(subject.INVENTORY_DTYPE.descr + [
        ("padding", "V7"),
    ]))
    main["symbol_id"] = [1, 2]
    overflow["symbol_id"] = [3]
    result = subject._inventory(main, overflow)
    assert result.dtype == subject.INVENTORY_DTYPE
    assert result["symbol_id"].tolist() == [1, 2, 3]


def test_seal_validation_detects_mutation() -> None:
    from market_analogues.types import stable_hash
    value = {"status": "complete"}
    value["case_digest"] = stable_hash(value)
    assert subject._seal_valid(value, "case_digest")
    value["status"] = "changed"
    assert not subject._seal_valid(value, "case_digest")


def test_certificate_digest_is_sensitive_to_matches_and_bounds() -> None:
    certificate = {
        "schema_version": "v", "contract_digest": "c",
        "packed_generation_id": "p", "dtw_generation_id": "d",
        "query_episode_id": "q", "input_digest": "i",
        "eligible_candidates": 100, "seed_rows": 20,
        "rigid_bound_evaluated": 100, "rigid_bound_admitted": 80,
        "dtw_bound_evaluated": 75, "combined_bound_admitted": 30,
        "exact_evaluated": 40, "native_bound_pruned": 0,
        "seed_threshold": 1.0, "final_threshold": 0.5,
        "minimum_rigid_pruned": 1.1, "minimum_combined_pruned": 1.2,
        "maximum_bound_excess": 0.0,
    }
    matches = [{"episode_id": "a" * 24, "distance_hex": (0.5).hex()}]
    first = subject._certificate_digest(certificate, matches)
    changed_matches = [{"episode_id": "b" * 24, "distance_hex": (0.5).hex()}]
    assert subject._certificate_digest(certificate, changed_matches) != first
    certificate["rigid_bound_admitted"] = 79
    assert subject._certificate_digest(certificate, matches) != first
