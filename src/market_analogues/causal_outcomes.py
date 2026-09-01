"""Frozen T14-09 causal forward-outcome semantics.

This module computes descriptive outcomes only.  It has no retrieval imports and
cannot alter candidate generation, distance, ranking, or match weights.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import pandas as pd


HORIZONS = (5, 10, 20, 40, 60, 126)
PATH_STEPS = 126
ATR_LOOKBACK = 20
FAVORABLE_ATR = 2.0
ADVERSE_ATR = 1.0


class CausalOutcomeError(ValueError):
    pass


@dataclass(frozen=True)
class OutcomeBundle:
    outcomes: pd.DataFrame
    paths: pd.DataFrame


def deduplicate_episode_requests(
    requests: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Deduplicate exact episode identities and reject conflicting reuse."""
    by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for raw in requests:
        row = dict(raw)
        key = (
            str(row.get("dataset", "")), str(row.get("episode_id", "")),
            str(row.get("source_fingerprint", "")),
        )
        if not all(key):
            raise CausalOutcomeError("episode request identity is incomplete")
        prior = by_key.get(key)
        if prior is not None and prior != row:
            raise CausalOutcomeError("duplicate episode identity has conflicting metadata")
        by_key[key] = row
    return [by_key[key] for key in sorted(by_key)]


def _sessions(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    required = {"timestamp", "open", "high", "low", "close"}
    if not required.issubset(frame.columns):
        raise CausalOutcomeError(f"{name} missing columns: {sorted(required - set(frame.columns))}")
    result = frame.loc[:, ["timestamp", "open", "high", "low", "close"]].copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"])
    if result["timestamp"].dt.tz is not None:
        result["timestamp"] = result["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)
    if result["timestamp"].duplicated().any():
        raise CausalOutcomeError(f"{name} contains duplicate timestamps")
    if not result["timestamp"].is_monotonic_increasing:
        raise CausalOutcomeError(f"{name} timestamps are not increasing")
    result = result.reset_index(drop=True)
    numeric = result[["open", "high", "low", "close"]]
    if not numeric.apply(lambda column: column.map(math.isfinite)).all().all():
        raise CausalOutcomeError(f"{name} contains non-finite OHLC")
    if (numeric <= 0).any().any():
        raise CausalOutcomeError(f"{name} contains non-positive OHLC")
    if (result["high"] < result[["open", "close", "low"]].max(axis=1)).any() \
            or (result["low"] > result[["open", "close", "high"]].min(axis=1)).any():
        raise CausalOutcomeError(f"{name} contains inconsistent OHLC")
    return result


def _origin_index(stock: pd.DataFrame, cutoff: pd.Timestamp | str) -> tuple[int, pd.Timestamp]:
    value = pd.Timestamp(cutoff)
    if value.tzinfo is not None:
        value = value.tz_convert("UTC").tz_localize(None)
    positions = stock.index[stock["timestamp"] == value]
    if len(positions) != 1:
        raise CausalOutcomeError("cutoff must identify exactly one stored stock session")
    return int(positions[0]), value


def _origin_atr(stock: pd.DataFrame, origin: int) -> float | None:
    if origin < ATR_LOOKBACK:
        return None
    values: list[float] = []
    for index in range(origin - ATR_LOOKBACK + 1, origin + 1):
        high = float(stock.at[index, "high"])
        low = float(stock.at[index, "low"])
        previous = float(stock.at[index - 1, "close"])
        values.append(max(high - low, abs(high - previous), abs(low - previous)))
    atr = math.fsum(values) / ATR_LOOKBACK
    return atr if math.isfinite(atr) and atr > 0 else None


def _barrier(
    future: pd.DataFrame, origin_close: float, atr: float | None, complete: bool,
) -> tuple[str, int | None, str | None, float | None, float | None]:
    favorable = origin_close + FAVORABLE_ATR * atr if atr is not None else None
    adverse = origin_close - ADVERSE_ATR * atr if atr is not None else None
    if not complete or atr is None:
        return "censored", None, None, favorable, adverse
    for offset, row in enumerate(future.iloc[:20].itertuples(index=False), start=1):
        up = float(row.high) >= float(favorable)
        down = float(row.low) <= float(adverse)
        if up and down:
            return "ambiguous_same_first_touch_bar", offset, str(row.timestamp.isoformat()), favorable, adverse
        if up:
            return "favorable_first", offset, str(row.timestamp.isoformat()), favorable, adverse
        if down:
            return "adverse_first", offset, str(row.timestamp.isoformat()), favorable, adverse
    return "no_touch", None, None, favorable, adverse


def _benchmark_map(benchmark: pd.DataFrame) -> dict[pd.Timestamp, float]:
    return {
        pd.Timestamp(row.timestamp): float(row.close)
        for row in benchmark.itertuples(index=False)
    }


def _continuity(
    stock_future: pd.DataFrame, benchmark: pd.DataFrame, cutoff: pd.Timestamp, horizon: int,
) -> tuple[bool, str]:
    observed = stock_future["timestamp"].iloc[:horizon]
    if len(observed) < horizon:
        return False, "source_end_before_horizon"
    endpoint = pd.Timestamp(observed.iloc[-1])
    expected = set(benchmark.loc[
        (benchmark["timestamp"] > cutoff) & (benchmark["timestamp"] <= endpoint),
        "timestamp",
    ])
    if not expected.issubset(set(observed)):
        return False, "suspension_or_missing_session"
    return True, "complete"


def compute_episode_outcomes(
    stock_bars: pd.DataFrame,
    benchmark_bars: pd.DataFrame,
    *,
    episode_id: str,
    cutoff: pd.Timestamp | str,
    source_fingerprint: str,
    contract_digest: str,
    source_content_digest: str,
    horizons: Sequence[int] = HORIZONS,
) -> OutcomeBundle:
    """Compute immutable episode outcomes and the observed future path."""
    if tuple(horizons) != HORIZONS:
        raise CausalOutcomeError("horizons differ from the frozen contract")
    if not episode_id or not source_fingerprint or not contract_digest or not source_content_digest:
        raise CausalOutcomeError("non-empty identity bindings are required")
    stock = _sessions(stock_bars, "stock")
    benchmark = _sessions(benchmark_bars, "benchmark")
    origin, cutoff_value = _origin_index(stock, cutoff)
    origin_close = float(stock.at[origin, "close"])
    atr = _origin_atr(stock, origin)
    stock_future = stock.iloc[origin + 1:origin + 1 + PATH_STEPS].reset_index(drop=True)
    benchmark_close = _benchmark_map(benchmark)
    benchmark_origin = benchmark_close.get(cutoff_value)
    path_rows: list[dict[str, Any]] = []
    expected = list(benchmark.loc[
        benchmark["timestamp"] > cutoff_value, "timestamp"
    ].iloc[:PATH_STEPS])
    for offset, row in enumerate(stock_future.itertuples(index=False), start=1):
        timestamp = pd.Timestamp(row.timestamp)
        stock_gross = float(row.close) / origin_close
        relative: float | None = None
        benchmark_endpoint = benchmark_close.get(timestamp)
        if benchmark_origin is not None and benchmark_endpoint is not None:
            relative = stock_gross / (benchmark_endpoint / benchmark_origin) - 1.0
        path_rows.append({
            "contract_digest": contract_digest,
            "source_content_digest": source_content_digest,
            "source_fingerprint": source_fingerprint,
            "episode_id": episode_id,
            "cutoff": cutoff_value.isoformat(),
            "step": offset,
            "timestamp": timestamp.isoformat(),
            "close_return": stock_gross - 1.0,
            "high_return": float(row.high) / origin_close - 1.0,
            "low_return": float(row.low) / origin_close - 1.0,
            "benchmark_relative_close_return": relative,
            "atr_normalized_close_move": (
                (float(row.close) - origin_close) / atr if atr is not None else None
            ),
            "expected_session_match": (
                offset <= len(expected) and timestamp == expected[offset - 1]
            ),
        })

    outcome_rows: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        available = min(len(stock_future), horizon)
        continuity, continuity_reason = _continuity(
            stock_future, benchmark, cutoff_value, horizon,
        )
        enough = len(stock_future) >= horizon
        complete = enough and continuity
        if not enough:
            status = "source_end_before_horizon"
        elif not continuity:
            status = continuity_reason
        else:
            status = "complete"
        window = stock_future.iloc[:available]
        endpoint = pd.Timestamp(window.iloc[-1]["timestamp"]) if available else None
        close_return = mfe = mae = mfe_atr = mae_atr = None
        time_to_mfe = time_to_mae = None
        relative = None
        benchmark_status = "not_applicable"
        if complete:
            highs = [float(value) for value in window["high"]]
            lows = [float(value) for value in window["low"]]
            best = max(highs)
            worst = min(lows)
            close_return = float(window.iloc[-1]["close"]) / origin_close - 1.0
            mfe = best / origin_close - 1.0
            mae = worst / origin_close - 1.0
            time_to_mfe = highs.index(best) + 1
            time_to_mae = lows.index(worst) + 1
            if atr is not None:
                mfe_atr = (best - origin_close) / atr
                mae_atr = (worst - origin_close) / atr
            benchmark_endpoint = benchmark_close.get(endpoint)
            if benchmark_origin is None or benchmark_endpoint is None:
                benchmark_status = "benchmark_missing_relative_only"
            else:
                relative = (1.0 + close_return) / (benchmark_endpoint / benchmark_origin) - 1.0
                benchmark_status = "complete"
        barrier_label = barrier_offset = barrier_timestamp = favorable = adverse = None
        barrier_status = "not_primary_horizon"
        if horizon == 20:
            barrier_label, barrier_offset, barrier_timestamp, favorable, adverse = _barrier(
                stock_future.iloc[:20], origin_close, atr, complete,
            )
            if not complete:
                barrier_status = status
            elif atr is None:
                barrier_status = "insufficient_atr_history"
            else:
                barrier_status = "complete"
        outcome_rows.append({
            "contract_digest": contract_digest,
            "source_content_digest": source_content_digest,
            "source_fingerprint": source_fingerprint,
            "episode_id": episode_id,
            "cutoff": cutoff_value.isoformat(),
            "horizon_sessions": horizon,
            "available_sessions": available,
            "completion_timestamp": endpoint.isoformat() if complete and endpoint is not None else None,
            "complete": complete,
            "status": status,
            "origin_close": origin_close,
            "origin_atr": atr,
            "close_return": close_return,
            "benchmark_relative_return": relative,
            "benchmark_status": benchmark_status,
            "maximum_favorable_excursion": mfe,
            "maximum_adverse_excursion": mae,
            "mfe_atr": mfe_atr,
            "mae_atr": mae_atr,
            "time_to_mfe": time_to_mfe,
            "time_to_mae": time_to_mae,
            "barrier_label": barrier_label,
            "barrier_status": barrier_status,
            "barrier_touch_offset": barrier_offset,
            "barrier_touch_timestamp": barrier_timestamp,
            "favorable_barrier_price": favorable,
            "adverse_barrier_price": adverse,
            "corporate_action_adjustment_warning": "unknown_provenance",
        })
    outcomes = pd.DataFrame(outcome_rows)
    outcomes["horizon_sessions"] = outcomes["horizon_sessions"].astype("int64")
    outcomes["available_sessions"] = outcomes["available_sessions"].astype("int64")
    for column in ("time_to_mfe", "time_to_mae", "barrier_touch_offset"):
        outcomes[column] = outcomes[column].astype("Int64")
    outcomes["complete"] = outcomes["complete"].astype("bool")
    paths = pd.DataFrame(path_rows)
    if not paths.empty:
        paths["step"] = paths["step"].astype("int64")
        paths["expected_session_match"] = paths["expected_session_match"].astype("bool")
    return OutcomeBundle(outcomes, paths)


def outcome_embargo(
    completion_timestamp: pd.Timestamp | str | None,
    query_cutoff: pd.Timestamp | str,
    *,
    complete: bool,
) -> tuple[bool, str]:
    if not complete or completion_timestamp is None:
        return False, "incomplete_horizon"
    completion = pd.Timestamp(completion_timestamp)
    query = pd.Timestamp(query_cutoff)
    if completion.tzinfo is not None:
        completion = completion.tz_convert("UTC").tz_localize(None)
    if query.tzinfo is not None:
        query = query.tz_convert("UTC").tz_localize(None)
    if completion > query:
        return False, "outcome_not_yet_observable"
    return True, "eligible"
