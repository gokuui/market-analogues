import json
from pathlib import Path

import numpy as np
import pytest

from experiments.m04r import m04r14_r1b_support_pilot as pilot
from market_analogues.adequacy_support import (
    AdequacySupportError, aligned_value, capped_matched_set_count,
    causally_eligible, deterministic_terciles, farthest_first_partition,
    matched_support,
)


def test_matched_support_is_exact_and_capped() -> None:
    cells = ("a", "a", "a", "b", "b")
    assert matched_support(range(5), (0, 3), cells, cap=100) == 6
    assert matched_support(range(5), (0, 1, 3), cells, cap=5) == 5
    assert capped_matched_set_count({"a": 1}, {"a": 2}, cap=10) == 0


def test_matched_support_rejects_invalid_membership() -> None:
    with pytest.raises(AdequacySupportError, match="subset"):
        matched_support((0,), (1,), ("a", "a"), cap=10)
    with pytest.raises(AdequacySupportError, match="duplicates"):
        matched_support((0, 0), (0,), ("a",), cap=10)


def test_deterministic_terciles_tie_break_by_id_and_keep_missing() -> None:
    result = deterministic_terciles([1.0, np.nan, 1.0, 3.0], ["b", "z", "a", "c"])
    assert result.tolist() == [1, -1, 0, 2]


def test_farthest_first_partition_is_repeatable_and_merges_small_cells() -> None:
    values = np.asarray([[0.0], [0.1], [10.0], [10.1], [30.0]])
    first = farthest_first_partition(values, ["e", "d", "c", "b", "a"], clusters=3, minimum_size=2)
    second = farthest_first_partition(values, ["e", "d", "c", "b", "a"], clusters=3, minimum_size=2)
    assert np.array_equal(first[0], second[0])
    assert first[1] == second[1]
    labels, medoids = first
    assert len(medoids) == 2
    assert sorted(np.bincount(labels).tolist()) == [2, 3]


def test_farthest_first_partition_merges_empty_tied_cells() -> None:
    labels, medoids = farthest_first_partition(
        np.zeros((4, 2)), ["d", "c", "b", "a"], clusters=3, minimum_size=1,
    )
    assert len(medoids) == 1
    assert labels.tolist() == [0, 0, 0, 0]


def test_causal_eligibility_includes_latest_but_excludes_same_symbol_start() -> None:
    assert causally_eligible(
        candidate_symbol="A", candidate_cutoff_ns=20,
        query_symbol="B", query_start_ns=10, latest_eligible_ns=20,
    )
    assert not causally_eligible(
        candidate_symbol="A", candidate_cutoff_ns=10,
        query_symbol="A", query_start_ns=10, latest_eligible_ns=20,
    )
    assert causally_eligible(
        candidate_symbol="A", candidate_cutoff_ns=9,
        query_symbol="A", query_start_ns=10, latest_eligible_ns=20,
    )


def test_aligned_value_crosses_main_overflow_boundary() -> None:
    main = np.asarray([[1.0], [2.0]])
    overflow = np.asarray([[3.0]])
    assert aligned_value(main, overflow, 1).tolist() == [2.0]
    assert aligned_value(main, overflow, 2).tolist() == [3.0]
    with pytest.raises(AdequacySupportError, match="outside"):
        aligned_value(main, overflow, 3)


def test_atomic_create_json_never_overwrites_existing_path(tmp_path: Path) -> None:
    target = tmp_path / "frozen.json"
    pilot._atomic_create_json(target, {"value": 1})
    assert json.loads(target.read_text()) == {"value": 1}
    with pytest.raises(pilot.SupportPilotError, match="create-only"):
        pilot._atomic_create_json(target, {"value": 2})
    assert json.loads(target.read_text()) == {"value": 1}


def test_atomic_directory_publish_never_replaces_existing_result(tmp_path: Path) -> None:
    first = tmp_path / "first"
    first.mkdir(); (first / "value.txt").write_text("one")
    destination = tmp_path / "published"
    pilot._atomic_publish_directory(first, destination)
    assert (destination / "value.txt").read_text() == "one"

    second = tmp_path / "second"
    second.mkdir(); (second / "value.txt").write_text("two")
    with pytest.raises(pilot.SupportPilotError, match="create-only output"):
        pilot._atomic_publish_directory(second, destination)
    assert (destination / "value.txt").read_text() == "one"
    assert (second / "value.txt").read_text() == "two"


def test_packed_store_selection_binds_durable_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    resident = tmp_path / "resident"
    durable_relative = Path("durable")
    durable_manifest = repository / durable_relative / "generations" / "g1" / "manifest.json"
    resident_manifest = resident / "generations" / "g1" / "manifest.json"
    durable_manifest.parent.mkdir(parents=True)
    resident_manifest.parent.mkdir(parents=True)
    durable_manifest.write_text('{"generation":"g1"}\n')
    resident_manifest.write_text(durable_manifest.read_text())
    monkeypatch.setattr(pilot.r1a, "PACKED_DURABLE", durable_relative)
    monkeypatch.setattr(pilot.r1a, "PACKED_RESIDENT", resident)

    assert pilot._select_packed_store(repository, "g1") == resident
    resident_manifest.write_text('{"generation":"different"}\n')
    with pytest.raises(pilot.SupportPilotError, match="manifests differ"):
        pilot._select_packed_store(repository, "g1")
    resident_manifest.unlink()
    assert pilot._select_packed_store(repository, "g1") == repository / durable_relative
