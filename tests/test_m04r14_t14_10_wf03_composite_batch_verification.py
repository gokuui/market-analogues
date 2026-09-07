from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.m04r import verify_m04r14_t14_10_wf03_composite_batch as subject


def test_alignment_accepts_only_complete_monotone_dtw_path() -> None:
    assert subject._alignment_valid([[0, 0], [1, 0], [2, 1], [2, 2]], 3, 3)
    assert not subject._alignment_valid([[0, 0], [2, 1], [2, 2]], 3, 3)
    assert not subject._alignment_valid([[0, 0], [1, 1]], 3, 3)
    assert not subject._alignment_valid([[0.0, 0], [1, 1]], 2, 2)


def test_lookup_preserves_requested_order() -> None:
    records = np.asarray([
        np.void(bytes.fromhex("02" * 12)),
        np.void(bytes.fromhex("01" * 12)),
    ], dtype="V12")
    order = np.argsort(records, kind="stable")
    positions = subject._lookup_positions(
        records[order], order, ["02" * 12, "01" * 12],
    )
    assert positions.tolist() == [0, 1]


def test_lookup_rejects_absent_identifier() -> None:
    records = np.asarray([np.void(bytes.fromhex("01" * 12))], dtype="V12")
    with pytest.raises(subject.CompositeBatchVerificationError):
        subject._lookup_positions(records, np.asarray([0]), ["02" * 12])


def test_rerun_sample_is_two_per_fold_and_order_independent() -> None:
    rows = [
        {"fold_id": fold, "episode_id": f"{value:024x}"}
        for fold in ("b", "a") for value in range(4)
    ]
    forward = subject.select_rerun_sample(rows)
    reverse = subject.select_rerun_sample(list(reversed(rows)))
    assert forward == reverse
    assert len(forward) == 4


def test_rerun_sample_refuses_small_fold() -> None:
    with pytest.raises(subject.CompositeBatchVerificationError):
        subject.select_rerun_sample([{"fold_id": "a", "episode_id": "0" * 24}])


def test_inventory_normalizes_main_and_overflow_layouts() -> None:
    main = np.zeros(1, dtype=np.dtype([
        ("episode_id", "V12"), ("cutoff_ns", "<i8"),
        ("symbol_id", "<u4"), ("quality_tier", "u1"), ("extra", "u1"),
    ]))
    overflow = np.zeros(1, dtype=np.dtype([
        ("episode_id", "V12"), ("cutoff_ns", "<i8"),
        ("symbol_id", "<u4"), ("quality_tier", "u1"), ("other", "<u2"),
    ]))
    combined = subject._inventory(main, overflow)
    assert combined.dtype == subject.INVENTORY_DTYPE
    assert len(combined) == 2
