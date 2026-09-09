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


def test_manifest_query_binding_maps_query_id_to_inventory_episode_id() -> None:
    methods = ("composite", "price_only")
    row = {
        "episode_id": "a" * 24,
        "case_id": "nasdaq-TEST-2020-01-01-252",
        "symbol": "TEST",
        "cutoff": "2020-01-01T00:00:00",
    }
    entry = {
        "query_id": row["episode_id"],
        "case_id": row["case_id"],
        "symbol": row["symbol"],
        "cutoff": row["cutoff"],
        "methods": [{"method": method} for method in methods],
    }
    assert subject._query_binding_matches(entry, row, methods)
    assert not subject._query_binding_matches(
        {**entry, "query_id": "b" * 24}, row, methods,
    )


@pytest.mark.parametrize("symbols", [
    [f"S{index}" for index in range(21)],
    ["Q", "Q", *[f"S{index}" for index in range(19)]],
    ["Q", "S0", "S0", *[f"S{index}" for index in range(1, 19)]],
])
def test_independent_full_subset_rejects_invalid_prefix(symbols: list[str]) -> None:
    with pytest.raises(subject.ExclusionRepairFullVerificationError):
        subject._independent_subset(_rows(symbols), "Q")
