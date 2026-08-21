from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .types import Episode, EpisodeKey, InstrumentKey


@dataclass(frozen=True)
class LatentStructureSpec:
    latent_id: str
    control_points: tuple[tuple[float, float], ...]
    volatility_start: float
    volatility_end: float
    tail_volatility: float
    volume_dryness: float
    tail_participation: float
    benchmark_slope: float
    benchmark_volatility: float


@dataclass(frozen=True)
class LatentStructureCase:
    episode: Episode
    latent_id: str
    seed: int
    role: str


LATENT_SPECS: tuple[LatentStructureSpec, ...] = (
    LatentStructureSpec("latent-00", ((0, 0), (.28, .24), (.48, .16), (.68, .23), (.88, .20), (1, .34)), .016, .005, .014, .55, 2.2, .0005, .005),
    LatentStructureSpec("latent-01", ((0, .18), (.30, -.14), (.52, -.20), (.72, -.06), (.90, .06), (1, .18)), .021, .008, .012, .75, 1.5, -.0002, .008),
    LatentStructureSpec("latent-02", ((0, 0), (.18, .13), (.34, .08), (.50, .21), (.67, .16), (.84, .29), (1, .36)), .011, .007, .010, .90, 1.4, .0004, .004),
    LatentStructureSpec("latent-03", ((0, 0), (.22, .04), (.43, -.025), (.64, .022), (.84, -.008), (.94, .004), (1, .13)), .014, .0035, .012, .45, 2.5, .0001, .004),
    LatentStructureSpec("latent-04", ((0, .16), (.24, .04), (.43, -.27), (.61, -.10), (.78, .02), (.91, .08), (1, .17)), .024, .010, .013, .85, 1.7, -.0004, .009),
    LatentStructureSpec("latent-05", ((0, 0), (.25, .17), (.50, .34), (.68, .43), (.80, .36), (.94, .39), (1, .45)), .012, .005, .009, .60, 1.6, .0007, .005),
    LatentStructureSpec("latent-06", ((0, .20), (.19, .08), (.36, .16), (.55, .03), (.72, .10), (.88, -.01), (1, -.08)), .014, .011, .016, 1.10, 1.8, -.0003, .006),
    LatentStructureSpec("latent-07", ((0, 0), (.18, .19), (.38, -.07), (.57, .16), (.76, .04), (.91, .12), (1, .27)), .022, .012, .015, .95, 2.0, .0002, .010),
)


def _spec(latent_id: str) -> LatentStructureSpec:
    for value in LATENT_SPECS:
        if value.latent_id == latent_id:
            return value
    raise ValueError(f"unknown latent structure: {latent_id}")


def generate_latent_structure(
    latent_id: str,
    seed: int,
    *,
    n: int = 300,
    role: str = "candidate",
    tempo: float = 1.0,
    noise_scale: float = 1.0,
    price_scale: float = 1.0,
    volume_scale: float = 1.0,
    critical_negative: bool = False,
) -> LatentStructureCase:
    if n < 252:
        raise ValueError("latent structures require at least 252 sessions")
    if not .8 <= tempo <= 1.2:
        raise ValueError("tempo must be between 0.8 and 1.2")
    if noise_scale < 0 or price_scale <= 0 or volume_scale <= 0:
        raise ValueError("noise and unit scales are invalid")
    spec = _spec(latent_id)
    rng = np.random.default_rng(seed + 1_000_003 * int(latent_id.rsplit("-", 1)[-1]))
    x = np.linspace(0, 1, n)
    warped = np.clip(x**tempo, 0, 1)
    control_x = np.asarray([point[0] for point in spec.control_points])
    control_y = np.asarray([point[1] for point in spec.control_points])
    latent_path = np.interp(warped, control_x, control_y)
    sigma = spec.volatility_start + (spec.volatility_end - spec.volatility_start) * x
    tail = x >= .88
    sigma[tail] = spec.tail_volatility
    innovations = rng.normal(0, sigma * noise_scale)
    innovations = innovations - pd.Series(innovations).rolling(7, min_periods=1).mean().to_numpy() * .55
    log_close = latent_path + np.cumsum(innovations) * .28
    if critical_negative:
        tail_start = int(n * .78)
        tail_returns = np.diff(log_close[tail_start - 1:], prepend=log_close[tail_start - 1])
        log_close[tail_start:] = log_close[tail_start - 1] + np.cumsum(-1.15 * tail_returns[1:])
        log_close[tail_start:] += np.linspace(0, -.16, n - tail_start)
    close = 100 * price_scale * np.exp(log_close)
    returns = np.diff(np.log(close), prepend=np.log(close[0]))
    gaps = rng.normal(0, np.maximum(sigma * .18, .0008))
    open_ = np.r_[close[0] / np.exp(returns[0]), close[:-1]] * np.exp(gaps)
    spread = np.maximum(np.abs(returns), sigma * .45)
    high = np.maximum(open_, close) * (1 + rng.uniform(.20, .70, n) * spread)
    low = np.minimum(open_, close) * (1 - rng.uniform(.20, .70, n) * spread)
    dry_curve = 1 - (1 - spec.volume_dryness) * np.clip((x - .45) / .43, 0, 1)
    tail_curve = np.where(tail, spec.tail_participation, 1.0)
    directional_participation = np.exp(np.clip(returns, -.04, .04) * 8)
    if critical_negative:
        directional_participation = 1 / directional_participation
        tail_curve = np.where(tail, 1 / max(spec.tail_participation, .1), 1.0)
    volume = np.maximum(
        1,
        1_000_000 * volume_scale * dry_curve * tail_curve
        * directional_participation * np.exp(rng.normal(0, .16 * noise_scale, n)),
    ).astype(np.int64)
    timestamps = pd.date_range("2005-01-03", periods=n, freq="B")
    bars = pd.DataFrame({
        "timestamp": timestamps, "open": open_, "high": high, "low": low,
        "close": close, "volume": volume,
    })
    benchmark_returns = (
        spec.benchmark_slope
        + .18 * returns
        + rng.normal(0, spec.benchmark_volatility, n)
    )
    if critical_negative:
        benchmark_returns[int(n * .78):] -= .0025
    benchmark = pd.DataFrame({
        "timestamp": timestamps,
        "close": 1000 * np.exp(np.cumsum(benchmark_returns)),
    })
    suffix = "critical-negative" if critical_negative else role
    instrument = InstrumentKey("latent-synthetic", f"{latent_id}-{seed}-{suffix}")
    key = EpisodeKey(instrument, timestamps[-1], 252, "multiresolution-v1")
    return LatentStructureCase(Episode(key, bars, benchmark), latent_id, seed, suffix)
