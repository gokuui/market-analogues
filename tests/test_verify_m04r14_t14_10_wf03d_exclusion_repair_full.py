from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.m04r import (  # noqa: E402
    verify_m04r14_t14_10_wf03d_exclusion_repair_full as subject,
)


def _rows(symbols: list[str]) -> list[dict[str, str]]:
    return [
        {"symbol": symbol, "episode_id": f"{index:024x}"}
        for index, symbol in enumerate(symbols)
    ]


def test_independent_full_subset_reconstructs_replacement() -> None:
    rows = _rows(["A", "Q", *[f"S{index}" for index in range(19)]])
    selected, proof = subject._independent_subset(rows, "Q")
    assert selected == [rows[0], *rows[2:]]
    assert proof["proof_kind"] == "exact_top_k_plus_one_drop_single_excluded_symbol"
    assert proof["excluded_rows"] == 1


@pytest.mark.parametrize("symbols", [
    [f"S{index}" for index in range(21)],
    ["Q", "Q", *[f"S{index}" for index in range(19)]],
    ["Q", "S0", "S0", *[f"S{index}" for index in range(1, 19)]],
])
def test_independent_full_subset_rejects_invalid_prefix(symbols: list[str]) -> None:
    with pytest.raises(subject.ExclusionRepairFullVerificationError):
        subject._independent_subset(_rows(symbols), "Q")
