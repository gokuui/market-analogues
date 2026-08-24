from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .context import align_benchmark, relative_channels
from .structural import directional_events
from .types import Episode

EPS = 1e-12


def _safe_log_ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    return np.log(a.clip(lower=EPS) / b.clip(lower=EPS))


def _rolling_robust_z(series: pd.Series, window: int = 20) -> pd.Series:
    median = series.rolling(window, min_periods=max(5, window // 2)).median()
    mad = (series - median).abs().rolling(window, min_periods=max(5, window // 2)).median()
    return (series - median) / (1.4826 * mad.replace(0, np.nan))


def dense_channels(episode: Episode) -> pd.DataFrame:
    b = episode.bars.reset_index(drop=True)
    prev_close = b["close"].shift(1)
    true_range = pd.concat([
        b["high"] - b["low"], (b["high"] - prev_close).abs(),
        (b["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = true_range.rolling(14, min_periods=5).mean()
    span = (b["high"] - b["low"]).replace(0, np.nan)
    body = b["close"] - b["open"]
    out = pd.DataFrame({
        "timestamp": b["timestamp"],
        "close_path": _safe_log_ratio(b["close"], pd.Series(b["close"].iloc[0], index=b.index)),
        "return": _safe_log_ratio(b["close"], prev_close),
        "overnight": _safe_log_ratio(b["open"], prev_close),
        "intraday": _safe_log_ratio(b["close"], b["open"]),
        "range_pct": _safe_log_ratio(b["high"], b["low"]),
        "body_atr": body / atr.replace(0, np.nan),
        "upper_wick_atr": (b["high"] - b[["open", "close"]].max(axis=1)) / atr.replace(0, np.nan),
        "lower_wick_atr": (b[["open", "close"]].min(axis=1) - b["low"]) / atr.replace(0, np.nan),
        "close_location": (b["close"] - b["low"]) / span,
        "atr_pct": atr / b["close"],
        # log(volume), rather than log1p(volume), makes a change of volume units
        # an exact additive shift which the rolling robust z-score removes.
        "volume_robust_z": _rolling_robust_z(
            np.log(b["volume"].astype(float).clip(lower=EPS)), 20,
        ),
        "return_shock_z": _rolling_robust_z(_safe_log_ratio(b["close"], prev_close), 63),
        "distance_high_63": b["close"] / b["close"].rolling(63, min_periods=20).max() - 1,
        "distance_ma_20": b["close"] / b["close"].rolling(20, min_periods=10).mean() - 1,
        "distance_ma_50": b["close"] / b["close"].rolling(50, min_periods=20).mean() - 1,
    })
    range20 = b["high"].rolling(20, min_periods=10).max() / b["low"].rolling(20, min_periods=10).min() - 1
    range63 = b["high"].rolling(63, min_periods=20).max() / b["low"].rolling(63, min_periods=20).min() - 1
    out["compression_ratio"] = range20 / range63.replace(0, np.nan)
    context = align_benchmark(b, episode.benchmark)
    relative = relative_channels(b, context)
    for col in context.columns:
        if col != "timestamp":
            out[col] = context[col].to_numpy()
    out["relative_return"] = relative["relative_return"].to_numpy()
    out["relative_path"] = relative["relative_path"].to_numpy()
    return out.replace([np.inf, -np.inf], np.nan)


def _resample(series: pd.Series, n: int) -> np.ndarray:
    values = series.astype(float).to_numpy()
    valid = np.isfinite(values)
    if valid.sum() == 0:
        return np.zeros(n, dtype=np.float32)
    indices = np.arange(len(values))
    values = np.interp(indices, indices[valid], values[valid])
    if len(values) == 1:
        return np.full(n, values[0], dtype=np.float32)
    x = np.linspace(0, len(values) - 1, n)
    return np.interp(x, indices, values).astype(np.float32)


COARSE_LAYOUT = {
    "close_path": 64,
    "atr_pct": 16,
    "volume_robust_z": 16,
    "relative_path": 16,
    "benchmark_path": 16,
}


def coarse_vector(channels: pd.DataFrame) -> np.ndarray:
    parts = [_resample(channels[name], size) for name, size in COARSE_LAYOUT.items()]
    vec = np.concatenate(parts).astype(np.float32)
    return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)


def resample_optional(values: pd.Series, n: int) -> np.ndarray | None:
    raw = values.astype(float).to_numpy()
    valid = np.isfinite(raw)
    if valid.sum() < max(3, len(raw) // 5):
        return None
    idx = np.arange(len(raw))
    filled = np.interp(idx, idx[valid], raw[valid])
    return np.interp(np.linspace(0, len(raw) - 1, n), idx, filled)


def stage_signature(frame: pd.DataFrame, stages: int = 12) -> np.ndarray:
    boundaries = np.linspace(0, len(frame), stages + 1, dtype=int)
    values: list[float] = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        block = frame.iloc[start:max(end, start + 1)]
        path = block.close_path.dropna()
        net = float(path.iloc[-1] - path.iloc[0]) if len(path) > 1 else 0.0
        values.extend([
            net / .08,
            float(block["return"].std(skipna=True) or 0.0) / .02,
            float(block["volume_robust_z"].mean(skipna=True) or 0.0),
            float(block["relative_return"].sum(skipna=True) or 0.0) / .08,
        ])
    return np.nan_to_num(np.asarray(values), nan=0.0, posinf=0.0, neginf=0.0)


def structural_signature(channels: pd.DataFrame) -> np.ndarray:
    pseudo = pd.DataFrame({
        "close": np.exp(channels.close_path.fillna(0).to_numpy()),
        "volume": np.exp(channels.volume_robust_z.fillna(0).clip(-5, 5).to_numpy()),
    })
    events = directional_events(pseudo)
    if events.empty:
        return np.zeros(9)
    values: list[float] = []
    for threshold in (.03, .06, .12):
        selected = events[events.threshold == threshold]
        values.extend([
            len(selected) / max(len(channels), 1),
            selected.amplitude.abs().mean() if len(selected) else 0,
            selected.duration.mean() / max(len(channels), 1) if len(selected) else 0,
        ])
    return np.asarray(values)


@dataclass(frozen=True)
class Representation:
    channels: pd.DataFrame
    coarse: np.ndarray
    samples_48: dict[str, np.ndarray | None]
    samples_64: dict[str, np.ndarray | None]
    stage: np.ndarray
    structural: np.ndarray


def represent(episode: Episode) -> Representation:
    """Build distance-v1 through the shared scalar/sliding exact kernel."""
    channels = dense_channels(episode)
    # Local import prevents a module cycle: exact_batch owns the vectorized
    # channel/materialization kernel and imports this module's data contract.
    from .exact_batch import exact_channel_rows, materialize_exact_representations

    positions, channel_rows = exact_channel_rows(
        episode.bars.reset_index(drop=True), episode.benchmark,
        lookback=len(episode.bars), stride=1,
    )
    if len(positions) != 1:
        raise ValueError("distance-v1 requires a non-empty fixed episode")
    exact = materialize_exact_representations(channel_rows)[0]
    return Representation(
        channels, exact.coarse, exact.samples_48, exact.samples_64,
        exact.stage, exact.structural,
    )
