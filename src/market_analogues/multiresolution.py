from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any

import numpy as np
import pandas as pd

from .representation import dense_channels
from .structural import directional_events
from .types import Episode, stable_hash


STATE_SCHEMA_VERSION = "multi-resolution-state-v1"
REQUIRED_HORIZONS = (252, 126, 63, 21, 10, 5)
SAMPLE_POINTS = 32


@dataclass(frozen=True)
class FieldObservation:
    name: str
    group: str
    source_inputs: tuple[str, ...]
    earliest_observation: str
    missingness: str


CHANNEL_FIELD_OBSERVATIONS: tuple[FieldObservation, ...] = (
    FieldObservation("close_path", "price", ("close",), "official_session_close", "not_expected"),
    FieldObservation("return", "price", ("close", "prior_close"), "official_session_close", "first_session"),
    FieldObservation("overnight", "price", ("open", "prior_close"), "official_session_open", "first_session"),
    FieldObservation("intraday", "price", ("open", "close"), "official_session_close", "invalid_price_only"),
    FieldObservation("range_pct", "candle", ("high", "low"), "official_session_close", "invalid_price_only"),
    FieldObservation("body_atr", "candle", ("open", "close", "high", "low"), "official_session_close", "rolling_warmup_or_zero_atr"),
    FieldObservation("upper_wick_atr", "candle", ("open", "close", "high", "low"), "official_session_close", "rolling_warmup_or_zero_atr"),
    FieldObservation("lower_wick_atr", "candle", ("open", "close", "high", "low"), "official_session_close", "rolling_warmup_or_zero_atr"),
    FieldObservation("close_location", "candle", ("close", "high", "low"), "official_session_close", "zero_range"),
    FieldObservation("atr_pct", "volatility", ("close", "high", "low", "prior_close"), "official_session_close", "rolling_warmup"),
    FieldObservation("volume_robust_z", "volume", ("volume",), "official_session_close", "rolling_warmup_or_constant_volume"),
    FieldObservation("return_shock_z", "volatility", ("close", "prior_close"), "official_session_close", "rolling_warmup_or_constant_return"),
    FieldObservation("distance_high_63", "structure", ("close",), "official_session_close", "rolling_warmup"),
    FieldObservation("distance_ma_20", "structure", ("close",), "official_session_close", "rolling_warmup"),
    FieldObservation("distance_ma_50", "structure", ("close",), "official_session_close", "rolling_warmup"),
    FieldObservation("compression_ratio", "structure", ("high", "low"), "official_session_close", "rolling_warmup_or_zero_long_range"),
    FieldObservation("benchmark_close", "market_context", ("benchmark_close",), "official_session_close", "missing_exact_session_or_context"),
    FieldObservation("benchmark_return", "market_context", ("benchmark_close", "prior_benchmark_close"), "official_session_close", "missing_exact_session_or_context"),
    FieldObservation("benchmark_path", "market_context", ("benchmark_close",), "official_session_close", "missing_exact_session_or_context"),
    FieldObservation("benchmark_volatility", "market_context", ("benchmark_close",), "official_session_close", "rolling_warmup_or_context"),
    FieldObservation("benchmark_drawdown", "market_context", ("benchmark_close",), "official_session_close", "missing_exact_session_or_context"),
    FieldObservation("relative_return", "relative_strength", ("close", "benchmark_close"), "official_session_close", "missing_exact_session_or_context"),
    FieldObservation("relative_path", "relative_strength", ("close", "benchmark_close"), "official_session_close", "missing_exact_session_or_context"),
)

SUMMARY_FIELD_OBSERVATIONS: tuple[FieldObservation, ...] = (
    FieldObservation("net_log_return", "price", ("close",), "official_session_close", "not_expected"),
    FieldObservation("maximum_drawdown", "price", ("close",), "official_session_close", "not_expected"),
    FieldObservation("distance_from_window_high", "price", ("close",), "official_session_close", "not_expected"),
    FieldObservation("realized_volatility", "volatility", ("close",), "official_session_close", "single_session"),
    FieldObservation("median_range_pct", "candle", ("high", "low"), "official_session_close", "invalid_price_only"),
    FieldObservation("range_contraction", "structure", ("high", "low"), "official_session_close", "zero_or_missing_early_range"),
    FieldObservation("latest_overnight", "price", ("open", "prior_close"), "official_session_open", "first_session"),
    FieldObservation("latest_close_location", "candle", ("close", "high", "low"), "official_session_close", "zero_range"),
    FieldObservation("latest_atr_pct", "volatility", ("close", "high", "low"), "official_session_close", "rolling_warmup"),
    FieldObservation("latest_compression_ratio", "structure", ("high", "low"), "official_session_close", "rolling_warmup"),
    FieldObservation("mean_volume_robust_z", "volume", ("volume",), "official_session_close", "rolling_warmup_or_constant_volume"),
    FieldObservation("latest_volume_robust_z", "volume", ("volume",), "official_session_close", "rolling_warmup_or_constant_volume"),
    FieldObservation("up_down_volume_z_spread", "volume", ("volume", "close"), "official_session_close", "missing_up_or_down_sessions"),
    FieldObservation("relative_log_return", "relative_strength", ("close", "benchmark_close"), "official_session_close", "missing_context"),
    FieldObservation("benchmark_log_return", "market_context", ("benchmark_close",), "official_session_close", "missing_context"),
    FieldObservation("benchmark_context_fraction", "market_context", ("benchmark_close",), "official_session_close", "not_expected"),
    FieldObservation("directional_events_3pct", "structure", ("close", "volume"), "official_session_close", "not_expected"),
    FieldObservation("directional_events_6pct", "structure", ("close", "volume"), "official_session_close", "not_expected"),
    FieldObservation("directional_events_12pct", "structure", ("close", "volume"), "official_session_close", "not_expected"),
)


@dataclass(frozen=True)
class MaskedSample:
    values: np.ndarray
    observed: np.ndarray


@dataclass(frozen=True)
class ResolutionView:
    horizon_sessions: int
    start_timestamp: str
    end_timestamp: str
    observed_sessions: int
    samples: dict[str, MaskedSample]
    summary: dict[str, float | None]


@dataclass
class MultiResolutionState:
    schema_version: str
    query_episode_id: str
    cutoff: str
    observation_mode: str
    quality_tier: str
    quality_issues: tuple[str, ...]
    field_contract_digest: str
    channels: pd.DataFrame
    views: dict[int, ResolutionView]
    state_digest: str


def field_contract() -> dict[str, dict[str, Any]]:
    fields = CHANNEL_FIELD_OBSERVATIONS + SUMMARY_FIELD_OBSERVATIONS
    if len({field.name for field in CHANNEL_FIELD_OBSERVATIONS}) != len(CHANNEL_FIELD_OBSERVATIONS):
        raise RuntimeError("duplicate channel field contract")
    if len({field.name for field in SUMMARY_FIELD_OBSERVATIONS}) != len(SUMMARY_FIELD_OBSERVATIONS):
        raise RuntimeError("duplicate summary field contract")
    return {
        field.name: {
            "group": field.group,
            "source_inputs": list(field.source_inputs),
            "earliest_observation": field.earliest_observation,
            "available_by_primary_decision": True,
            "missingness": field.missingness,
            "future_inputs_allowed": False,
        }
        for field in fields
    }


def _masked_resample(series: pd.Series, points: int = SAMPLE_POINTS) -> MaskedSample:
    raw = series.astype(float).to_numpy()
    finite = np.isfinite(raw)
    targets = np.linspace(0, max(len(raw) - 1, 0), points)
    if not len(raw) or not finite.any():
        return MaskedSample(np.zeros(points, dtype=np.float32), np.zeros(points, dtype=bool))
    indices = np.arange(len(raw))
    values = np.interp(targets, indices[finite], raw[finite]).astype(np.float32)
    nearest = np.rint(targets).astype(int)
    observed = finite[nearest]
    values[~observed] = 0.0
    return MaskedSample(values, observed)


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _sum_if_any(series: pd.Series) -> float | None:
    values = series.astype(float)
    return _finite(values.sum(min_count=1))


def _resolution_summary(bars: pd.DataFrame, channels: pd.DataFrame) -> dict[str, float | None]:
    close = bars["close"].astype(float)
    returns = np.log(close / close.shift(1))
    drawdown = close / close.cummax() - 1
    split = max(len(channels) // 3, 1)
    early_range = channels["range_pct"].iloc[:split].median(skipna=True)
    late_range = channels["range_pct"].iloc[-split:].median(skipna=True)
    contraction = late_range / early_range if pd.notna(early_range) and early_range != 0 else np.nan
    up_volume = channels.loc[channels["return"] > 0, "volume_robust_z"].mean(skipna=True)
    down_volume = channels.loc[channels["return"] < 0, "volume_robust_z"].mean(skipna=True)
    events = directional_events(bars[["close", "volume"]])
    event_counts = {
        threshold: int((events["threshold"] == threshold).sum()) if len(events) else 0
        for threshold in (0.03, 0.06, 0.12)
    }
    context_fraction = float(channels["benchmark_close"].notna().mean())
    return {
        "net_log_return": _finite(np.log(close.iloc[-1] / close.iloc[0])),
        "maximum_drawdown": _finite(drawdown.min()),
        "distance_from_window_high": _finite(close.iloc[-1] / close.max() - 1),
        "realized_volatility": _finite(returns.std(skipna=True)),
        "median_range_pct": _finite(channels["range_pct"].median(skipna=True)),
        "range_contraction": _finite(contraction),
        "latest_overnight": _finite(channels["overnight"].iloc[-1]),
        "latest_close_location": _finite(channels["close_location"].iloc[-1]),
        "latest_atr_pct": _finite(channels["atr_pct"].iloc[-1]),
        "latest_compression_ratio": _finite(channels["compression_ratio"].iloc[-1]),
        "mean_volume_robust_z": _finite(channels["volume_robust_z"].mean(skipna=True)),
        "latest_volume_robust_z": _finite(channels["volume_robust_z"].iloc[-1]),
        "up_down_volume_z_spread": _finite(up_volume - down_volume),
        "relative_log_return": _sum_if_any(channels["relative_return"]),
        "benchmark_log_return": _sum_if_any(channels["benchmark_return"]),
        "benchmark_context_fraction": context_fraction,
        "directional_events_3pct": float(event_counts[0.03]),
        "directional_events_6pct": float(event_counts[0.06]),
        "directional_events_12pct": float(event_counts[0.12]),
    }


def _frame_digest(frame: pd.DataFrame) -> str:
    digest = sha256()
    digest.update("\0".join(str(column) for column in frame.columns).encode())
    digest.update(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
    return digest.hexdigest()


def _state_manifest(
    episode: Episode,
    cutoff: pd.Timestamp,
    channels: pd.DataFrame,
    views: dict[int, ResolutionView],
    field_digest: str,
) -> dict[str, Any]:
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "query_episode_id": episode.key.id,
        "cutoff": cutoff.isoformat(),
        "observation_mode": "after_close_daily",
        "quality_tier": episode.quality_tier,
        "quality_issues": list(episode.quality_issues),
        "field_contract_digest": field_digest,
        "channels_digest": _frame_digest(channels),
        "views": {
            str(horizon): {
                "start_timestamp": view.start_timestamp,
                "end_timestamp": view.end_timestamp,
                "observed_sessions": view.observed_sessions,
                "summary": view.summary,
                "samples": {
                    name: {
                        "values": sample.values.tolist(),
                        "observed": sample.observed.astype(int).tolist(),
                    }
                    for name, sample in sorted(view.samples.items())
                },
            }
            for horizon, view in sorted(views.items(), reverse=True)
        },
    }


def build_multiresolution_state(
    episode: Episode,
    horizons: tuple[int, ...] = REQUIRED_HORIZONS,
) -> MultiResolutionState:
    if tuple(horizons) != REQUIRED_HORIZONS:
        raise ValueError(f"horizons must be exactly {REQUIRED_HORIZONS}")
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    if not required <= set(episode.bars.columns):
        raise ValueError(f"episode bars missing columns: {sorted(required - set(episode.bars.columns))}")
    bars = episode.bars.copy()
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], errors="raise")
    if bars["timestamp"].duplicated().any():
        raise ValueError("episode bars contain duplicate timestamps")
    if not bars["timestamp"].is_monotonic_increasing:
        raise ValueError("episode bars must be sorted by timestamp")
    cutoff = pd.Timestamp(episode.key.cutoff)
    observed = bars[bars["timestamp"] <= cutoff].tail(max(horizons)).reset_index(drop=True)
    if len(observed) < max(horizons):
        raise ValueError(f"need {max(horizons)} observed sessions, found {len(observed)}")
    benchmark = episode.benchmark
    if benchmark is not None:
        benchmark = benchmark.copy()
        benchmark["timestamp"] = pd.to_datetime(benchmark["timestamp"], errors="raise")
        benchmark = benchmark[benchmark["timestamp"] <= cutoff].sort_values("timestamp").reset_index(drop=True)
    causal_episode = Episode(
        episode.key, observed, benchmark, episode.quality_tier, episode.quality_issues,
    )
    channels = dense_channels(causal_episode)
    contract = field_contract()
    channel_fields = {field.name for field in CHANNEL_FIELD_OBSERVATIONS}
    actual_channels = set(channels.columns) - {"timestamp"}
    if actual_channels != channel_fields:
        raise RuntimeError(
            f"dense channel contract mismatch; missing={sorted(actual_channels - channel_fields)}, "
            f"stale={sorted(channel_fields - actual_channels)}"
        )
    views: dict[int, ResolutionView] = {}
    for horizon in horizons:
        channel_slice = channels.tail(horizon).reset_index(drop=True)
        bar_slice = observed.tail(horizon).reset_index(drop=True)
        summary = _resolution_summary(bar_slice, channel_slice)
        expected_summary = {field.name for field in SUMMARY_FIELD_OBSERVATIONS}
        if set(summary) != expected_summary:
            raise RuntimeError("summary field contract mismatch")
        views[horizon] = ResolutionView(
            horizon,
            pd.Timestamp(bar_slice["timestamp"].iloc[0]).isoformat(),
            pd.Timestamp(bar_slice["timestamp"].iloc[-1]).isoformat(),
            len(bar_slice),
            {name: _masked_resample(channel_slice[name]) for name in sorted(actual_channels)},
            summary,
        )
    field_digest = stable_hash(contract)
    manifest = _state_manifest(episode, cutoff, channels, views, field_digest)
    return MultiResolutionState(
        STATE_SCHEMA_VERSION, episode.key.id, cutoff.isoformat(), "after_close_daily",
        episode.quality_tier, episode.quality_issues, field_digest, channels, views,
        stable_hash(manifest),
    )


def state_manifest(state: MultiResolutionState) -> dict[str, Any]:
    payload = {
        "schema_version": state.schema_version,
        "query_episode_id": state.query_episode_id,
        "cutoff": state.cutoff,
        "observation_mode": state.observation_mode,
        "quality_tier": state.quality_tier,
        "quality_issues": list(state.quality_issues),
        "field_contract_digest": state.field_contract_digest,
        "state_digest": state.state_digest,
        "channels_digest": _frame_digest(state.channels),
        "views": {
            str(horizon): {
                "start_timestamp": view.start_timestamp,
                "end_timestamp": view.end_timestamp,
                "observed_sessions": view.observed_sessions,
                "summary": view.summary,
                "sample_availability": {
                    name: int(sample.observed.sum()) for name, sample in view.samples.items()
                },
            }
            for horizon, view in sorted(state.views.items(), reverse=True)
        },
    }
    payload["manifest_digest"] = stable_hash(payload)
    return payload
