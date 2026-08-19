from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ForwardOutcome:
    cutoff: str
    horizon_bars: int
    available_bars: int
    forward_return: float | None
    max_favorable_excursion: float | None
    max_adverse_excursion: float | None
    censored: bool


def compute_outcomes(
    full_bars: pd.DataFrame,
    cutoff: pd.Timestamp | str,
    horizons: tuple[int, ...] = (5, 10, 20, 60),
) -> pd.DataFrame:
    """Compute display-only future outcomes, never representation inputs."""
    bars = full_bars.sort_values("timestamp").reset_index(drop=True)
    cutoff = pd.Timestamp(cutoff)
    eligible = bars.index[bars.timestamp <= cutoff]
    if not len(eligible):
        raise ValueError("cutoff precedes available bars")
    origin = int(eligible[-1])
    origin_close = float(bars.close.iloc[origin])
    results: list[ForwardOutcome] = []
    for horizon in horizons:
        future = bars.iloc[origin + 1:origin + horizon + 1]
        available = len(future)
        complete = available == horizon
        if available:
            final = float(future.close.iloc[-1] / origin_close - 1)
            mfe = float(future.high.max() / origin_close - 1)
            mae = float(future.low.min() / origin_close - 1)
        else:
            final = mfe = mae = None
        results.append(ForwardOutcome(
            cutoff.isoformat(), horizon, available,
            final if complete else None, mfe if complete else None,
            mae if complete else None, not complete,
        ))
    return pd.DataFrame([asdict(item) for item in results])


def summarize_match_outcomes(outcomes: list[pd.DataFrame]) -> pd.DataFrame:
    rows = [row._asdict() for frame in outcomes for row in frame.itertuples(index=False)]
    complete = pd.DataFrame(rows)
    if complete.empty:
        return complete
    usable = complete[~complete.censored]
    return usable.groupby("horizon_bars", as_index=False).agg(
        sample_size=("forward_return", "count"),
        median_return=("forward_return", "median"),
        return_q25=("forward_return", lambda x: float(x.quantile(.25))),
        return_q75=("forward_return", lambda x: float(x.quantile(.75))),
        positive_rate=("forward_return", lambda x: float(np.mean(x > 0))),
        median_mfe=("max_favorable_excursion", "median"),
        median_mae=("max_adverse_excursion", "median"),
    )
