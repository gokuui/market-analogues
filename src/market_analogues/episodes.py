from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from .adapters import OHLCVSource
from .types import Episode, EpisodeKey, InstrumentKey


@dataclass(frozen=True)
class EpisodeManifestRecord:
    episode_id: str
    dataset_id: str
    symbol: str
    cutoff: str
    lookback: int
    available_bars: int
    quality_tier: str
    quality_issues: str
    source_hash: str
    representation_version: str


def candidate_positions(n_bars: int, minimum: int = 126, stride: int = 5) -> list[int]:
    if n_bars < minimum:
        return []
    return list(range(minimum - 1, n_bars, stride))


def build_episode(
    source: OHLCVSource,
    instrument: InstrumentKey,
    cutoff: pd.Timestamp | str,
    lookback: int,
    representation_version: str,
    quality_tier: str = "A",
    quality_issues: tuple[str, ...] = (),
) -> Episode:
    bars = source.load(instrument)
    cutoff = pd.Timestamp(cutoff)
    eligible = bars[bars["timestamp"] <= cutoff]
    if eligible.empty:
        raise ValueError(f"no bars at or before cutoff {cutoff}")
    if len(eligible) < min(lookback, 126):
        raise ValueError(f"insufficient bars: {len(eligible)}")
    window = eligible.tail(lookback).reset_index(drop=True)
    actual_cutoff = pd.Timestamp(window["timestamp"].iloc[-1])
    benchmark = source.load_benchmark()
    if benchmark is not None:
        benchmark = benchmark[benchmark["timestamp"] <= actual_cutoff].copy()
    key = EpisodeKey(instrument, actual_cutoff, lookback, representation_version)
    return Episode(key, window, benchmark, quality_tier, quality_issues)


def build_manifest(
    source: OHLCVSource,
    representation_version: str,
    quality: pd.DataFrame | None = None,
    stride: int = 5,
    lookbacks: tuple[int, ...] = (21, 63, 126, 252),
    instrument_limit: int | None = None,
    output: Path | None = None,
) -> pd.DataFrame:
    qmap: dict[str, tuple[str, str]] = {}
    if quality is not None and len(quality):
        qmap = {str(r.symbol): (str(r.tier), str(r.issues or "")) for r in quality.itertuples()}
    records: list[EpisodeManifestRecord] = []
    instruments = source.instruments()
    if instrument_limit is not None:
        instruments = instruments[:instrument_limit]
    for instrument in instruments:
        tier, issues = qmap.get(instrument.source_symbol, ("A", ""))
        if tier == "QUARANTINED":
            continue
        bars = source.load(instrument)
        source_hash = source.fingerprint(instrument)
        for pos in candidate_positions(len(bars), min(lookbacks), stride):
            cutoff = pd.Timestamp(bars["timestamp"].iloc[pos])
            for lookback in lookbacks:
                if pos + 1 >= lookback:
                    key = EpisodeKey(instrument, cutoff, lookback, representation_version)
                    records.append(EpisodeManifestRecord(
                        key.id, instrument.dataset_id, instrument.source_symbol,
                        cutoff.isoformat(), lookback, lookback, tier, issues,
                        source_hash, representation_version,
                    ))
    frame = pd.DataFrame([asdict(x) for x in records])
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(output, index=False)
    return frame
