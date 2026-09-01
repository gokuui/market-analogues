"""Deliberately separate scalar oracle for frozen T14-09 outcome semantics.

This module does not import ``market_analogues.causal_outcomes``.  It is used
only with synthetic fixtures before any real forward path is opened.
"""
from __future__ import annotations

import math
from typing import Any

import pandas as pd


HORIZONS = (5, 10, 20, 40, 60, 126)


def _ordered(frame: pd.DataFrame) -> list[dict[str, Any]]:
    result = frame.sort_values("timestamp", kind="stable").to_dict("records")
    for row in result:
        value = pd.Timestamp(row["timestamp"])
        if value.tzinfo is not None:
            value = value.tz_convert("UTC").tz_localize(None)
        row["timestamp"] = value
        for key in ("open", "high", "low", "close"):
            row[key] = float(row[key])
    return result


def reference_episode(
    stock_bars: pd.DataFrame,
    benchmark_bars: pd.DataFrame,
    *,
    episode_id: str,
    cutoff: pd.Timestamp | str,
    source_fingerprint: str,
    contract_digest: str,
    source_content_digest: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    stock = _ordered(stock_bars)
    benchmark = _ordered(benchmark_bars)
    cutoff_value = pd.Timestamp(cutoff)
    if cutoff_value.tzinfo is not None:
        cutoff_value = cutoff_value.tz_convert("UTC").tz_localize(None)
    origin_positions = [i for i, row in enumerate(stock) if row["timestamp"] == cutoff_value]
    if len(origin_positions) != 1:
        raise ValueError("oracle cutoff differs")
    origin = origin_positions[0]
    origin_close = stock[origin]["close"]
    atr = None
    if origin >= 20:
        true_ranges = []
        for index in range(origin - 19, origin + 1):
            row = stock[index]
            previous = stock[index - 1]["close"]
            true_ranges.append(max(
                row["high"] - row["low"],
                abs(row["high"] - previous),
                abs(row["low"] - previous),
            ))
        candidate = math.fsum(true_ranges) / 20
        if math.isfinite(candidate) and candidate > 0:
            atr = candidate
    future = stock[origin + 1:origin + 127]
    benchmark_close = {row["timestamp"]: row["close"] for row in benchmark}
    benchmark_origin = benchmark_close.get(cutoff_value)
    expected = [row["timestamp"] for row in benchmark if row["timestamp"] > cutoff_value][:126]
    paths: list[dict[str, Any]] = []
    for position, row in enumerate(future, 1):
        stock_gross = row["close"] / origin_close
        benchmark_endpoint = benchmark_close.get(row["timestamp"])
        relative = None
        if benchmark_origin is not None and benchmark_endpoint is not None:
            relative = stock_gross / (benchmark_endpoint / benchmark_origin) - 1
        paths.append({
            "contract_digest": contract_digest,
            "source_content_digest": source_content_digest,
            "source_fingerprint": source_fingerprint,
            "episode_id": episode_id,
            "cutoff": cutoff_value.isoformat(),
            "step": position,
            "timestamp": row["timestamp"].isoformat(),
            "close_return": stock_gross - 1,
            "high_return": row["high"] / origin_close - 1,
            "low_return": row["low"] / origin_close - 1,
            "benchmark_relative_close_return": relative,
            "atr_normalized_close_move": (
                (row["close"] - origin_close) / atr if atr is not None else None
            ),
            "expected_session_match": (
                position <= len(expected) and row["timestamp"] == expected[position - 1]
            ),
        })
    outcomes: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        window = future[:horizon]
        available = len(window)
        observed = {row["timestamp"] for row in window}
        endpoint_for_continuity = window[-1]["timestamp"] if window else None
        benchmark_expected = {
            row["timestamp"] for row in benchmark
            if endpoint_for_continuity is not None
            and cutoff_value < row["timestamp"] <= endpoint_for_continuity
        }
        continuous = benchmark_expected.issubset(observed)
        complete = available == horizon and continuous
        if available < horizon:
            status = "source_end_before_horizon"
        elif not continuous:
            status = "suspension_or_missing_session"
        else:
            status = "complete"
        endpoint = window[-1]["timestamp"] if window else None
        close_return = relative = mfe = mae = mfe_atr = mae_atr = None
        time_to_mfe = time_to_mae = None
        benchmark_status = "not_applicable"
        if complete:
            highs = [row["high"] for row in window]
            lows = [row["low"] for row in window]
            best = max(highs)
            worst = min(lows)
            close_return = window[-1]["close"] / origin_close - 1
            mfe = best / origin_close - 1
            mae = worst / origin_close - 1
            time_to_mfe = highs.index(best) + 1
            time_to_mae = lows.index(worst) + 1
            if atr is not None:
                mfe_atr = (best - origin_close) / atr
                mae_atr = (worst - origin_close) / atr
            benchmark_endpoint = benchmark_close.get(endpoint)
            if benchmark_origin is None or benchmark_endpoint is None:
                benchmark_status = "benchmark_missing_relative_only"
            else:
                relative = (1 + close_return) / (benchmark_endpoint / benchmark_origin) - 1
                benchmark_status = "complete"
        barrier_label = barrier_offset = barrier_timestamp = favorable = adverse = None
        barrier_status = "not_primary_horizon"
        if horizon == 20:
            favorable = origin_close + 2 * atr if atr is not None else None
            adverse = origin_close - atr if atr is not None else None
            if not complete:
                barrier_label = "censored"
                barrier_status = status
            elif atr is None:
                barrier_label = "censored"
                barrier_status = "insufficient_atr_history"
            else:
                barrier_status = "complete"
                for offset, row in enumerate(window, 1):
                    up = row["high"] >= favorable
                    down = row["low"] <= adverse
                    if up or down:
                        barrier_offset = offset
                        barrier_timestamp = row["timestamp"].isoformat()
                        if up and down:
                            barrier_label = "ambiguous_same_first_touch_bar"
                        elif up:
                            barrier_label = "favorable_first"
                        else:
                            barrier_label = "adverse_first"
                        break
                if barrier_label is None:
                    barrier_label = "no_touch"
        outcomes.append({
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
    return outcomes, paths
