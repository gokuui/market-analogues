from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
from typing import Any

import pandas as pd


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return sha256(payload.encode()).hexdigest()


@dataclass(frozen=True, order=True)
class InstrumentKey:
    dataset_id: str
    source_symbol: str

    def __str__(self) -> str:
        return f"{self.dataset_id}:{self.source_symbol}"


@dataclass(frozen=True, order=True)
class EpisodeKey:
    instrument: InstrumentKey
    cutoff: pd.Timestamp
    lookback: int
    representation_version: str

    @property
    def id(self) -> str:
        return stable_hash({
            "instrument": str(self.instrument), "cutoff": self.cutoff.isoformat(),
            "lookback": self.lookback, "version": self.representation_version,
        })[:24]


@dataclass
class Episode:
    key: EpisodeKey
    bars: pd.DataFrame
    benchmark: pd.DataFrame | None = None
    quality_tier: str = "A"
    quality_issues: tuple[str, ...] = ()


@dataclass(frozen=True)
class SearchQuery:
    episode_key: EpisodeKey
    search_datasets: tuple[str, ...] = ()
    quality_tiers: tuple[str, ...] = ("A",)
    top_k: int = 20
    cross_dataset: bool = False
    deduplicate_overlaps: bool = True
    max_per_instrument: int = 3
    minimum_history_gap_bars: int = 60


@dataclass
class AnalogueMatch:
    episode_key: EpisodeKey
    total_distance: float
    component_distances: dict[str, float]
    alignment: list[tuple[int, int]] = field(default_factory=list)
    quality_tier: str = "A"
    quality_issues: tuple[str, ...] = ()


@dataclass
class VerificationReport:
    suite: str
    passed: bool
    metrics: dict[str, float | int | str]
    failures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
