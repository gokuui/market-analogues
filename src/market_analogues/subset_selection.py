from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


class SubsetSelectionError(ValueError):
    """Raised when a certified superset prefix cannot prove a subset top-k."""


@dataclass(frozen=True)
class SubsetSelectionProof:
    input_rows: int
    selected_rows: int
    excluded_rows: int
    query_symbol: str
    top_k: int
    proof_kind: str


def certified_prefix_without_query_symbol(
    rows: Sequence[Mapping[str, Any]], query_symbol: str, *, top_k: int,
) -> tuple[tuple[dict[str, Any], ...], SubsetSelectionProof]:
    """Derive an exact subset top-k from an exact, symbol-unique prefix.

    ``rows`` must already be ordered by the certified source ranking and contain
    no more than one row per symbol.  If the query symbol occurs, a prefix of at
    least ``top_k + 1`` is necessary and sufficient: removing that one row and
    retaining the first ``top_k`` proves the top-k over the excluded-symbol
    subset.  If it does not occur, the first ``top_k`` is unchanged.
    """
    if type(query_symbol) is not str or not query_symbol:
        raise SubsetSelectionError("query symbol must be non-empty")
    if type(top_k) is not int or isinstance(top_k, bool) or top_k < 1:
        raise SubsetSelectionError("top-k must be a positive integer")
    normalized: list[dict[str, Any]] = []
    symbols: list[str] = []
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise SubsetSelectionError(f"prefix row {index} must be a mapping")
        symbol = raw.get("symbol")
        if type(symbol) is not str or not symbol:
            raise SubsetSelectionError(f"prefix row {index} has no symbol")
        normalized.append(dict(raw))
        symbols.append(symbol)
    if len(symbols) != len(set(symbols)):
        raise SubsetSelectionError("certified prefix must contain distinct symbols")
    excluded = symbols.count(query_symbol)
    required = top_k + excluded
    if excluded > 1:
        raise SubsetSelectionError("query symbol occurs more than once")
    if len(normalized) < required:
        raise SubsetSelectionError(
            "certified prefix is too short after query-symbol exclusion"
        )
    selected = tuple(
        row for row in normalized if row["symbol"] != query_symbol
    )[:top_k]
    if len(selected) != top_k:
        raise SubsetSelectionError("excluded prefix cannot fill requested top-k")
    proof = SubsetSelectionProof(
        input_rows=len(normalized), selected_rows=len(selected),
        excluded_rows=excluded, query_symbol=query_symbol, top_k=top_k,
        proof_kind=(
            "exact_top_k_unchanged_over_subset"
            if excluded == 0 else "exact_top_k_plus_one_drop_single_excluded_symbol"
        ),
    )
    return selected, proof
