from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.m04r.verify_m04r14_t14_10_wf03b_dtw_store_full import (
    FullStoreVerificationError,
    _independent_validate,
    _probe_query_ids,
    _raw_symbol,
    _scan,
    _select,
)
from market_analogues.causal_prefix import causal_prefix_digest
from market_analogues.dtw_sample_store import (
    DTW_SAMPLE_DTYPE,
    make_dtw_sample_record,
    make_zero_dtw_sample_record,
)
from market_analogues.packed_bound_store import OVERFLOW_DTYPE, make_packed_record
from market_analogues.quantized_bound import quantize_bound_row
from market_analogues.representation import represent
from market_analogues.synthetic import generate_case
from market_analogues.types import EpisodeKey, InstrumentKey


def test_frozen_probe_query_ids_use_four_field_probe_contract() -> None:
    assert _probe_query_ids() == [
        "a69def453340e01048a52284",
        "4ebfb91d87892612d998e71e",
        "b923b64e7f3b36fe2a8d0643",
    ]


def test_selection_forces_edges_and_computes_physical_offsets() -> None:
    symbols = ("A", "B", "C", "D")
    metadata = [
        {"rows": 2, "overflow_rows": 0, "zero_bound_rows": 0,
         "source_prefix": {"digest": "a"}},
        {"rows": 3, "overflow_rows": 1, "zero_bound_rows": 0,
         "source_prefix": {"digest": "b"}},
        {"rows": 4, "overflow_rows": 0, "zero_bound_rows": 2,
         "source_prefix": {"digest": "c"}},
        {"rows": 5, "overflow_rows": 0, "zero_bound_rows": 0,
         "source_prefix": {"digest": "d"}},
    ]
    packed = SimpleNamespace(overflow=np.asarray(
        [(1,)], dtype=[("symbol_id", "<u4")],
    ))
    selected = _select(symbols, metadata, packed)
    by_id = {row["symbol_id"]: row for row in selected}
    assert {1, 2}.issubset(by_id)
    assert by_id[2]["main_offset"] == 5
    assert by_id[2]["overflow_offset"] == 1


def test_independent_validator_handles_strided_memmap_and_rejects_values(
    tmp_path: Path,
) -> None:
    record = make_dtw_sample_record(
        represent(generate_case("rounded_base", 160_000).episode)
    )
    pair = np.concatenate((record, record))
    path = tmp_path / "records.bin"
    pair.tofile(path)
    mapped = np.memmap(path, dtype=DTW_SAMPLE_DTYPE, mode="r")
    _independent_validate(mapped)
    high_presence = record.copy()
    high_presence["presence"][0] |= np.uint8(16)
    with pytest.raises(FullStoreVerificationError, match="values"):
        _independent_validate(high_presence)
    padding = record.copy()
    padding["padding"].view(np.uint8)[0] = 1
    with pytest.raises(FullStoreVerificationError, match="values"):
        _independent_validate(padding)
    order = record.copy()
    order["orders"][0, 0, 0] = order["orders"][0, 0, 1]
    with pytest.raises(FullStoreVerificationError, match="permutation"):
        _independent_validate(order)


def test_forward_reverse_scans_are_exact_for_both_lanes() -> None:
    query = represent(generate_case("trend_contraction_breakout", 160_001).episode)
    candidate = represent(generate_case("rounded_base", 160_002).episode)
    samples = SimpleNamespace(
        rows=make_dtw_sample_record(candidate),
        overflow=make_zero_dtw_sample_record(),
    )
    forward, _seconds, forward_digest = _scan(
        query, samples, block_rows=1, reverse=False,
    )
    reverse, _seconds, reverse_digest = _scan(
        query, samples, block_rows=2, reverse=True,
    )
    assert np.array_equal(forward, reverse)
    assert forward_digest == reverse_digest


def test_raw_worker_reconstructs_store_by_frozen_offsets(monkeypatch) -> None:
    import experiments.m04r.verify_m04r14_t14_10_wf03b_dtw_store_full as subject

    candidate = generate_case("rounded_base", 160_003).episode
    bars = candidate.bars.copy()
    cutoff = pd.Timestamp(bars.timestamp.iloc[-1])
    key = InstrumentKey("nasdaq", "XYZ")
    episode = EpisodeKey(key, cutoff, 252, "dense-v1")
    representation = represent(candidate)
    packed_row = make_packed_record(
        episode.id, int(cutoff.value), 0, "A", quantize_bound_row(representation),
    )

    class Source:
        def load(self, _key):
            return bars.copy()

    monkeypatch.setattr(subject, "_SOURCE", Source())
    monkeypatch.setattr(subject, "_BENCHMARK", candidate.benchmark)
    monkeypatch.setattr(subject, "_PACKED", SimpleNamespace(
        rows=packed_row, overflow=np.empty(0, dtype=OVERFLOW_DTYPE),
    ))
    monkeypatch.setattr(subject, "_SAMPLES", SimpleNamespace(
        rows=make_dtw_sample_record(representation),
        overflow=np.empty(0, dtype=DTW_SAMPLE_DTYPE),
    ))
    monkeypatch.setattr(subject, "_MAXIMUM", cutoff)
    result = _raw_symbol({
        "symbol": "XYZ", "symbol_id": 0, "rows": 1,
        "overflow_rows": 0, "main_offset": 0, "overflow_offset": 0,
        "zero_bound_rows": 0,
        "source_prefix": vars(causal_prefix_digest(bars, cutoff)),
    })
    assert result["rows"] == 1
    assert result["overflow_rows"] == 0
