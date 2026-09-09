from __future__ import annotations

import pytest

from market_analogues.subset_selection import (
    SubsetSelectionError,
    certified_prefix_without_query_symbol,
)


def _rows(*symbols: str) -> list[dict[str, object]]:
    return [
        {"symbol": symbol, "rank_value": index}
        for index, symbol in enumerate(symbols, 1)
    ]


def test_unaffected_certified_top_k_is_unchanged_over_subset() -> None:
    selected, proof = certified_prefix_without_query_symbol(
        _rows("A", "B", "C"), "Q", top_k=3,
    )
    assert [row["symbol"] for row in selected] == ["A", "B", "C"]
    assert proof.excluded_rows == 0
    assert proof.proof_kind == "exact_top_k_unchanged_over_subset"


@pytest.mark.parametrize("position", range(4))
def test_top_k_plus_one_drops_query_symbol_at_any_rank(position: int) -> None:
    symbols = ["A", "B", "C"]
    symbols.insert(position, "Q")
    selected, proof = certified_prefix_without_query_symbol(
        _rows(*symbols), "Q", top_k=3,
    )
    assert [row["symbol"] for row in selected] == ["A", "B", "C"]
    assert proof.excluded_rows == 1
    assert proof.proof_kind == "exact_top_k_plus_one_drop_single_excluded_symbol"


def test_affected_top_k_without_extra_row_fails_closed() -> None:
    with pytest.raises(SubsetSelectionError, match="too short"):
        certified_prefix_without_query_symbol(
            _rows("A", "Q", "B"), "Q", top_k=3,
        )


@pytest.mark.parametrize(
    ("rows", "symbol", "top_k", "message"),
    [
        (_rows("A", "A", "B"), "Q", 2, "distinct symbols"),
        ([{"rank_value": 1}], "Q", 1, "has no symbol"),
        (_rows("A"), "", 1, "non-empty"),
        (_rows("A"), "Q", True, "positive integer"),
    ],
)
def test_malformed_subset_proofs_fail_closed(
    rows: list[dict[str, object]], symbol: str, top_k: int, message: str,
) -> None:
    with pytest.raises(SubsetSelectionError, match=message):
        certified_prefix_without_query_symbol(rows, symbol, top_k=top_k)
