from pathlib import Path
import sys
from types import SimpleNamespace
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.m04r.m04r14_t14_10_wf03b_dtw_store_poc import (
    _build_symbol, _packed_rows_state, _slice, select_symbols,
)
from market_analogues.causal_prefix import causal_prefix_digest
from market_analogues.packed_bound_store import (
    OVERFLOW_DTYPE, make_packed_record,
)
from market_analogues.quantized_bound import quantize_bound_row
from market_analogues.representation import represent
from market_analogues.synthetic import generate_case
from market_analogues.types import EpisodeKey, InstrumentKey, stable_hash


class _Loaded:
    symbols = ("C", "A", "B", "D")
    rows = np.asarray([
        (0, 1), (0, 2), (1, 3), (2, 4), (3, 5),
    ], dtype=[("symbol_id", "<u4"), ("cutoff_ns", "<i8")])
    overflow = np.asarray([(2, 6)], dtype=rows.dtype)
    manifest = {"provenance": {"source_prefixes": {
        symbol: {"symbol": symbol} for symbol in symbols
    }}}


def test_slice_finds_exact_sorted_symbol_block() -> None:
    assert _slice(_Loaded.rows, 0)["cutoff_ns"].tolist() == [1, 2]
    assert len(_slice(_Loaded.rows, 9)) == 0


def test_selection_forces_overflow_and_is_symbol_ordered(monkeypatch) -> None:
    import experiments.m04r.m04r14_t14_10_wf03b_dtw_store_poc as subject

    monkeypatch.setattr(subject, "_packed_rows_state", lambda rows: [len(rows)])
    rows = select_symbols(_Loaded, count=3)
    assert [row["symbol_id"] for row in rows] == sorted(
        row["symbol_id"] for row in rows
    )
    assert any(row["symbol_id"] == 2 and row["forced_overflow"] for row in rows)


def test_real_worker_flow_on_one_complete_synthetic_symbol(
    tmp_path: Path, monkeypatch,
) -> None:
    import experiments.m04r.m04r14_t14_10_wf03b_dtw_store_poc as subject

    candidate = generate_case("rounded_base", 140_000).episode
    query = generate_case("trend_contraction_breakout", 140_001).episode
    bars = candidate.bars.copy()
    cutoff = pd.Timestamp(bars.timestamp.iloc[-1])
    identity = EpisodeKey(InstrumentKey("nasdaq", "XYZ"), cutoff, 252, "dense-v1")
    representation = represent(candidate)
    packed = make_packed_record(
        identity.id, int(cutoff.value), 0, "A", quantize_bound_row(representation),
    )

    class Source:
        def load(self, _key):
            return bars.copy()

    loaded = SimpleNamespace(
        rows=packed, overflow=np.empty(0, dtype=OVERFLOW_DTYPE),
    )
    specification = {
        "symbol": "XYZ", "symbol_id": 0, "rows": 1, "overflow_rows": 0,
        "source_prefix": vars(causal_prefix_digest(bars, cutoff)),
        "main_slice_digest": stable_hash(_packed_rows_state(packed)),
        "overflow_slice_digest": stable_hash([]),
    }
    monkeypatch.setattr(subject, "_SOURCE", Source())
    monkeypatch.setattr(subject, "_BENCHMARK", candidate.benchmark)
    monkeypatch.setattr(subject, "_QUERY", represent(query))
    monkeypatch.setattr(subject, "_QUERY_ID", query.key.id)
    monkeypatch.setattr(subject, "_MAXIMUM_CUTOFF", cutoff)
    monkeypatch.setattr(subject, "_PACKED", loaded)
    metadata = _build_symbol((specification, str(tmp_path)))
    assert metadata["rows"] == 1
    assert metadata["overflow_rows"] == 0
    assert len(metadata["scalar_probes"]) == 1


def test_gate_evidence_uses_strict_json_native_scalars() -> None:
    payload = {
        "maximum_scalar_delta": float(np.float64(0.0)),
        "gates": {
            "scalar_batch_passed": bool(np.float64(0.0) <= 1e-12),
            "capacity_passed": bool(np.int64(10) < 20),
        },
        "scan_seconds": [float(np.float64(0.1))],
    }
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload
