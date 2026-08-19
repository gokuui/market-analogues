from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .types import Episode, EpisodeKey, InstrumentKey


FAMILIES = (
    "trend_contraction_breakout",
    "rounded_base",
    "volatile_reversal",
    "failed_breakout",
    "steady_trend",
)


@dataclass(frozen=True)
class SyntheticCase:
    episode: Episode
    family: str
    seed: int


def _latent_returns(family: str, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(0, 1, n)
    noise = rng.normal(0, 1, n)
    if family == "trend_contraction_breakout":
        drift = np.where(x < .48, .0024, np.where(x < .88, .0001, .0060))
        sigma = np.where(x < .48, .012, np.where(x < .88, .010 * (1.1 - x), .018))
        volume = 1.0 + .45 * (x < .48) + 2.0 * (x >= .88)
    elif family == "rounded_base":
        drift = .0002 + .0025 * (2 * x - 1)
        sigma = .009 + .010 * np.abs(2 * x - 1)
        volume = .8 + 1.4 * np.abs(2 * x - 1)
    elif family == "volatile_reversal":
        drift = np.where(x < .55, -.0028, .0038)
        sigma = np.where(x < .55, .022, .015)
        volume = np.where(x < .55, 1.8, 1.25) + .5 * (np.abs(x - .55) < .08)
    elif family == "failed_breakout":
        drift = np.where(x < .70, .0010, np.where(x < .82, .007, -.0045))
        sigma = np.where(x < .70, .010, .019)
        volume = 1.0 + 1.6 * (x >= .70)
    elif family == "steady_trend":
        drift = np.full(n, .0018)
        sigma = np.full(n, .007)
        volume = 1.0 + .15 * np.sin(6 * np.pi * x)
    else:
        raise ValueError(f"unknown synthetic family: {family}")
    cycle = .0015 * np.sin(8 * np.pi * x + rng.uniform(-.5, .5))
    returns = drift + cycle + sigma * noise
    return returns, volume


def generate_case(family: str, seed: int, n: int = 252, dataset_id: str = "synthetic") -> SyntheticCase:
    """Generate deterministic, valid daily OHLCV with a known latent morphology."""
    if n < 63:
        raise ValueError("synthetic episodes need at least 63 bars")
    if family not in FAMILIES:
        raise ValueError(f"unknown synthetic family: {family}")
    # Salt by family so equal seed numbers do not accidentally share their
    # random innovations across labels and inflate cross-family similarity.
    rng = np.random.default_rng(seed + 100_003 * FAMILIES.index(family))
    returns, volume_shape = _latent_returns(family, n, rng)
    close = 100 * np.exp(np.cumsum(returns))
    gap = rng.normal(0, .0025, n)
    open_ = np.r_[close[0] / np.exp(returns[0]), close[:-1]] * np.exp(gap)
    intraday_scale = np.maximum(np.abs(returns), .004)
    high = np.maximum(open_, close) * (1 + rng.uniform(.15, .75, n) * intraday_scale)
    low = np.minimum(open_, close) * (1 - rng.uniform(.15, .75, n) * intraday_scale)
    volume = np.maximum(1, 1_000_000 * volume_shape * np.exp(rng.normal(0, .22, n))).astype(np.int64)
    timestamp = pd.date_range("2010-01-04", periods=n, freq="B")
    bars = pd.DataFrame({
        "timestamp": timestamp, "open": open_, "high": high, "low": low,
        "close": close, "volume": volume,
    })
    market_returns = .00035 + .30 * returns + rng.normal(0, .005, n)
    benchmark = pd.DataFrame({
        "timestamp": timestamp,
        "close": 1000 * np.exp(np.cumsum(market_returns)),
    })
    instrument = InstrumentKey(dataset_id, f"{family}-{seed}")
    key = EpisodeKey(instrument, timestamp[-1], n, "dense-v1")
    return SyntheticCase(Episode(key, bars, benchmark), family, seed)


def transform_case(
    case: SyntheticCase,
    *,
    name: str,
    price_scale: float = 1.0,
    volume_scale: float = 1.0,
    noise: float = 0.0,
    reverse_returns: bool = False,
    context_flip: bool = False,
    time_shift_days: int = 0,
    seed: int = 0,
) -> SyntheticCase:
    """Create labeled metamorphic positives and hard negatives."""
    rng = np.random.default_rng(seed)
    bars = case.episode.bars.copy()
    if reverse_returns or noise:
        base_returns = np.diff(np.log(bars.close.to_numpy()), prepend=np.log(bars.close.iloc[0]))
        if reverse_returns:
            base_returns = -base_returns
        base_returns += rng.normal(0, noise, len(base_returns))
        new_close = bars.close.iloc[0] * np.exp(np.cumsum(base_returns))
        ratio = new_close / bars.close.to_numpy()
        for col in ["open", "high", "low", "close"]:
            bars[col] = bars[col].to_numpy() * ratio
    bars[["open", "high", "low", "close"]] *= price_scale
    bars["volume"] = np.maximum(1, bars.volume.to_numpy() * volume_scale).astype(np.int64)
    if time_shift_days:
        bars["timestamp"] = bars["timestamp"] + pd.Timedelta(days=time_shift_days)
    benchmark = case.episode.benchmark.copy() if case.episode.benchmark is not None else None
    if benchmark is not None and context_flip:
        ret = np.diff(np.log(benchmark.close.to_numpy()), prepend=np.log(benchmark.close.iloc[0]))
        benchmark["close"] = benchmark.close.iloc[0] * np.exp(np.cumsum(-ret))
    if benchmark is not None and time_shift_days:
        benchmark["timestamp"] = benchmark["timestamp"] + pd.Timedelta(days=time_shift_days)
    instrument = InstrumentKey(case.episode.key.instrument.dataset_id, f"{case.family}-{case.seed}-{name}")
    key = EpisodeKey(instrument, bars.timestamp.iloc[-1], len(bars), "dense-v1")
    return SyntheticCase(Episode(key, bars, benchmark), case.family, case.seed)


def verification_corpus(seeds_per_family: int = 8) -> list[SyntheticCase]:
    return [generate_case(family, seed) for family in FAMILIES for seed in range(seeds_per_family)]
