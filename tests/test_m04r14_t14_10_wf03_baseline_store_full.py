from dataclasses import asdict
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.m04r import m04r14_t14_10_wf03_baseline_store_full as subject
from experiments.m04r import verify_m04r14_t14_10_wf03_baseline_store_full as verifier
from market_analogues.causal_prefix import causal_prefix_digest
from market_analogues.packed_bound_store import OVERFLOW_DTYPE, PACK_DTYPE, TIER_CODES


def _packed(symbol_id: int, cutoff: pd.Timestamp) -> np.ndarray:
    rows = np.zeros(1, dtype=PACK_DTYPE)
    rows["episode_id"][0] = np.void((symbol_id + 1).to_bytes(12, "big"))
    rows["symbol_id"][0] = symbol_id
    rows["cutoff_ns"][0] = int(cutoff.value)
    rows["quality_tier"][0] = TIER_CODES["A"]
    return rows


def test_full_selection_covers_every_physical_row() -> None:
    cutoffs = pd.to_datetime(["2020-01-01", "2020-01-02"])
    rows = np.concatenate((_packed(0, cutoffs[0]), _packed(1, cutoffs[1])))
    overflow = np.zeros(1, dtype=OVERFLOW_DTYPE)
    overflow["episode_id"][0] = np.void((3).to_bytes(12, "big"))
    overflow["symbol_id"][0] = 1
    overflow["cutoff_ns"][0] = int(cutoffs[1].value)
    overflow["quality_tier"][0] = TIER_CODES["A"]
    loaded = SimpleNamespace(
        symbols=("A", "B"), rows=rows, overflow=overflow,
        manifest={"provenance": {"source_prefixes": {
            "A": {"digest": "a"}, "B": {"digest": "b"},
        }}},
    )
    selection = subject.full_selection(loaded)
    assert [(
        row["symbol_id"], row["main_start"], row["overflow_start"],
        row["rows"], row["overflow_rows"],
    ) for row in selection] == [(0, 0, 0, 1, 0), (1, 1, 0, 1, 1)]


def test_full_verifier_boundary_probes_cover_empty_one_and_many() -> None:
    assert verifier._boundary_probes(0) == []
    assert verifier._boundary_probes(1) == [0]
    assert verifier._boundary_probes(5) == [0, 4]


def test_full_worker_builds_and_revalidates_feature_shard(
    tmp_path: Path, monkeypatch,
) -> None:
    timestamps = pd.date_range("2020-01-01", periods=80, freq="D")
    frame = pd.DataFrame({
        "timestamp": timestamps,
        "open": np.ones(80), "high": np.ones(80), "low": np.ones(80),
        "close": np.exp(np.arange(80) * .01), "volume": np.ones(80),
    })
    source = SimpleNamespace(load=lambda _key: frame.copy())
    packed = SimpleNamespace(
        rows=_packed(0, timestamps[-1]),
        overflow=np.empty(0, dtype=OVERFLOW_DTYPE),
    )
    specification = {
        "symbol": "XYZ", "symbol_id": 0,
        "main_start": 0, "overflow_start": 0,
        "rows": 1, "overflow_rows": 0,
        "source_prefix": asdict(causal_prefix_digest(frame, timestamps[-1])),
    }
    monkeypatch.setattr(subject, "_SOURCE", source)
    monkeypatch.setattr(subject, "_PACKED", packed)
    monkeypatch.setattr(subject, "_MAXIMUM", timestamps[-1])
    first = subject._build_symbol((specification, str(tmp_path)))
    second = subject._build_symbol((specification, str(tmp_path)))
    assert first == second
    assert first["rows"] == 1 and first["missing_rows"] == 0
    assert subject._valid_shard(tmp_path, specification) == first
