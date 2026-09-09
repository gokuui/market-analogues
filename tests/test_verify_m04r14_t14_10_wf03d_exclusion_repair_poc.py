from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.m04r import (  # noqa: E402
    verify_m04r14_t14_10_wf03d_exclusion_repair_poc as subject,
)


def _rows(symbols: list[str]) -> list[dict[str, str]]:
    return [
        {"symbol": symbol, "episode_id": f"{index:024x}"}
        for index, symbol in enumerate(symbols)
    ]


def test_independent_subset_drops_exactly_one_query_symbol() -> None:
    rows = _rows(["Q", *[f"S{index}" for index in range(20)]])
    selected, proof = subject._independent_subset(rows, "Q")
    assert len(selected) == 20
    assert all(row["symbol"] != "Q" for row in selected)
    assert proof == {
        "input_rows": 21, "selected_rows": 20, "excluded_rows": 1,
        "query_symbol": "Q", "top_k": 20,
        "proof_kind": "exact_top_k_plus_one_drop_single_excluded_symbol",
    }


@pytest.mark.parametrize("symbols", [
    [f"S{index}" for index in range(21)],
    ["Q", "Q", *[f"S{index}" for index in range(19)]],
    ["Q", "S0", "S0", *[f"S{index}" for index in range(1, 19)]],
])
def test_independent_subset_rejects_invalid_prefix(symbols: list[str]) -> None:
    with pytest.raises(subject.ExclusionRepairPocVerificationError):
        subject._independent_subset(_rows(symbols), "Q")
