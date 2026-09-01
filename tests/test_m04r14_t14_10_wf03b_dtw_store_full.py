from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.m04r.m04r14_t14_10_wf03b_dtw_store_full import (
    _build_symbol,
    _full_selection,
    _scan_generation,
    _validated_shard,
    _write_progress,
)
from market_analogues.causal_prefix import causal_prefix_digest
from market_analogues.dtw_sample_store import DTW_SAMPLE_DTYPE, make_dtw_sample_record
from market_analogues.packed_bound_store import OVERFLOW_DTYPE, make_packed_record
from market_analogues.quantized_bound import quantize_bound_row
from market_analogues.representation import represent
from market_analogues.synthetic import generate_case
from market_analogues.types import EpisodeKey, InstrumentKey


def test_full_selection_is_exact_packed_symbol_order() -> None:
    loaded = SimpleNamespace(
        symbols=("B", "A"),
        manifest={"provenance": {"source_prefixes": {
            "B": {"digest": "b"}, "A": {"digest": "a"},
        }}},
    )
    assert _full_selection(loaded) == [
        {"symbol": "B", "symbol_id": 0, "source_prefix": {"digest": "b"}},
        {"symbol": "A", "symbol_id": 1, "source_prefix": {"digest": "a"}},
    ]


def test_full_worker_builds_and_resumes_one_exact_symbol(
    tmp_path: Path, monkeypatch,
) -> None:
    import experiments.m04r.m04r14_t14_10_wf03b_dtw_store_full as subject

    candidate = generate_case("rounded_base", 150_000).episode
    bars = candidate.bars.copy()
    cutoff = pd.Timestamp(bars.timestamp.iloc[-1])
    key = InstrumentKey("nasdaq", "XYZ")
    identity = EpisodeKey(key, cutoff, 252, "dense-v1")
    representation = represent(candidate)
    packed = make_packed_record(
        identity.id, int(cutoff.value), 0, "A", quantize_bound_row(representation),
    )

    class Source:
        def load(self, _key):
            return bars.copy()

    specification = {
        "symbol": "XYZ", "symbol_id": 0,
        "source_prefix": vars(causal_prefix_digest(bars, cutoff)),
    }
    monkeypatch.setattr(subject, "_SOURCE", Source())
    monkeypatch.setattr(subject, "_BENCHMARK", candidate.benchmark)
    monkeypatch.setattr(subject, "_MAXIMUM_CUTOFF", cutoff)
    monkeypatch.setattr(subject, "_PACKED", SimpleNamespace(
        rows=packed, overflow=np.empty(0, dtype=OVERFLOW_DTYPE),
    ))
    first = _build_symbol((specification, str(tmp_path)))
    second = _build_symbol((specification, str(tmp_path)))
    assert first == second
    assert first["rows"] == 1
    assert first["overflow_rows"] == 0
    assert _validated_shard(tmp_path, specification) == first


def test_complete_scan_accounts_for_main_and_overflow() -> None:
    query = represent(generate_case("trend_contraction_breakout", 150_001).episode)
    main = make_dtw_sample_record(
        represent(generate_case("rounded_base", 150_002).episode)
    )
    overflow = np.zeros(1, dtype=DTW_SAMPLE_DTYPE)
    overflow["orders"] = np.arange(64, dtype=np.uint8)
    elapsed, digest, count = _scan_generation(
        query, SimpleNamespace(rows=main, overflow=overflow),
    )
    assert elapsed >= 0
    assert len(digest) == 64
    assert count == 2


def test_progress_is_atomically_replaceable_not_create_only(tmp_path: Path) -> None:
    path = tmp_path / "work" / "PROGRESS.json"
    _write_progress(
        path, status="building", completed=0, total=2, reused=0,
        rows=0, overflow_rows=0,
    )
    _write_progress(
        path, status="shards_complete", completed=2, total=2, reused=2,
        rows=7, overflow_rows=1,
    )
    import json

    value = json.loads(path.read_text())
    assert value["status"] == "shards_complete"
    assert value["completed_symbols"] == value["reused_symbols"] == 2
    assert not list(path.parent.glob("*.tmp"))
