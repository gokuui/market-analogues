from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from experiments.m04r import m04r14_r1b_b2_geometry as geometry
from market_analogues.causal_prefix import causal_prefix_digest
from market_analogues.config import BenchmarkSpec, DatasetSpec
from market_analogues.types import EpisodeKey, InstrumentKey, stable_hash


def _charts(rows: int, *, offset: float = 0.0) -> np.ndarray:
    values = np.arange(rows * 141, dtype=np.float64).reshape(rows, 141)
    return np.sin(values / 19.0) + offset


def _bars(rows: int = 252) -> pd.DataFrame:
    timestamps = pd.bdate_range("2020-01-02", periods=rows)
    close = 100.0 + np.arange(rows, dtype=np.float64) / 10.0
    return pd.DataFrame({
        "timestamp": timestamps, "open": close - 0.2, "high": close + 0.5,
        "low": close - 0.7, "close": close, "volume": 1_000_000.0 + np.arange(rows),
    })


def test_array_digest_is_little_endian_shape_bound_and_positive_zero() -> None:
    value = geometry._canonical_float(np.asarray([[-0.0, 1.0]]), (1, 2), "x")
    assert value.dtype.str == "<f8"
    assert not np.signbit(value[0, 0])
    assert geometry._array_semantic_digest(value) != geometry._array_semantic_digest(value.T)
    assert geometry._array_semantic_digest(value) == sha256(
        b"[1,2]" + value.tobytes(order="C")
    ).hexdigest()


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_canonical_float_refuses_nonfinite(bad: float) -> None:
    with pytest.raises(geometry.GeometryError, match="finiteness"):
        geometry._canonical_float([[bad] * 141], (1, 141), "bad")


def test_distance_serial_parallel_byte_identity() -> None:
    queries = _charts(9)
    candidates = _charts(5, offset=0.15)
    serial = geometry.distance_matrices(queries, candidates, workers=1, chunk_rows=2)
    parallel = geometry.distance_matrices(queries, candidates, workers=12, chunk_rows=2)
    assert serial[0].tobytes() == parallel[0].tobytes()
    assert serial[1].tobytes() == parallel[1].tobytes()
    assert np.array_equal(serial[0], serial[0].T)
    assert np.all(serial[0].diagonal() == 0.0)


def test_distance_resume_uses_exact_shards(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    queries = _charts(7); candidates = _charts(3, offset=0.1)
    expected = geometry.distance_matrices(
        queries, candidates, workers=1, work=tmp_path, binding="a", chunk_rows=2,
    )
    monkeypatch.setattr(geometry, "_distance_worker", lambda bounds: pytest.fail("recomputed"))
    resumed = geometry.distance_matrices(
        queries, candidates, workers=1, work=tmp_path, binding="a", chunk_rows=2,
    )
    assert expected[0].tobytes() == resumed[0].tobytes()
    assert expected[1].tobytes() == resumed[1].tobytes()


def test_distance_resume_refuses_corruption_and_binding_drift(tmp_path: Path) -> None:
    queries = _charts(5); candidates = _charts(2)
    geometry.distance_matrices(
        queries, candidates, workers=1, work=tmp_path, binding="a", chunk_rows=2,
    )
    with pytest.raises(geometry.GeometryError, match="metadata differs"):
        geometry.distance_matrices(
            queries, candidates, workers=1, work=tmp_path, binding="b", chunk_rows=2,
        )
    _, qq, _ = geometry._shard_paths(tmp_path, 0, 2)
    data = bytearray(qq.read_bytes()); data[-1] ^= 1; qq.write_bytes(data)
    with pytest.raises(geometry.GeometryError, match="source bytes differ"):
        geometry.distance_matrices(
            queries, candidates, workers=1, work=tmp_path, binding="a", chunk_rows=2,
        )


def test_distance_resume_refuses_unbound_artifacts(tmp_path: Path) -> None:
    (tmp_path / "foreign.npy").write_bytes(b"foreign")
    with pytest.raises(geometry.GeometryError, match="unexpected geometry work artifacts"):
        geometry.distance_matrices(
            _charts(3), _charts(2), workers=1, work=tmp_path,
            binding="a", chunk_rows=2,
        )


def test_distance_resume_discards_only_unpublished_staging(tmp_path: Path) -> None:
    staging = tmp_path / ".rows-00000-00002.tmp-999-deadbeef"
    staging.mkdir()
    (staging / "query_pair.npy").write_bytes(b"interrupted")
    result = geometry.distance_matrices(
        _charts(3), _charts(2), workers=1, work=tmp_path,
        binding="a", chunk_rows=2,
    )
    assert result[0].shape == (3, 3)
    assert not staging.exists()


def test_distance_resume_refuses_partial_published_directory(tmp_path: Path) -> None:
    meta, _, _ = geometry._shard_paths(tmp_path, 0, 2)
    meta.parent.mkdir()
    meta.write_text("{}")
    with pytest.raises(geometry.GeometryError, match="partial or unexpected"):
        geometry.distance_matrices(
            _charts(3), _charts(2), workers=1, work=tmp_path,
            binding="a", chunk_rows=2,
        )


def test_external_manifest_rehashes_every_exact_file(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"; benchmark = tmp_path / "IXIC.parquet"
    stock = tmp_path / "ABC.parquet"
    config.write_bytes(b"config"); benchmark.write_bytes(b"benchmark"); stock.write_bytes(b"stock")
    def record(path: Path) -> dict[str, object]:
        return {"path": str(path), "bytes": path.stat().st_size, "sha256": geometry._file_sha(path)}
    state = {
        "schema_version": "m04r14-r1b-external-source-manifest-v1", "dataset_id": "nasdaq",
        "resolver": "test", "config": record(config), "dataset_spec": {},
        "source_lock_digest": "x", "query_ids_digest": "q", "cohort_ids_digest": "c",
        "query_symbols": ["ABC"], "cohort_symbols": ["ABC"], "required_symbols": ["ABC"],
        "stock_files": [{"dataset_id": "nasdaq", "symbol": "ABC", **record(stock)}],
        "benchmark": record(benchmark), "stock_files_count": 1,
        "ohlcv_decoded": False, "geometry_materialized": False, "immutability": "test",
    }
    manifest = {**state, "manifest_digest": stable_hash(state)}
    prereg = {"authorities": {"external_source_manifest": manifest}}
    geometry.verify_external_source_manifest(prereg)
    stock.write_bytes(b"changed")
    with pytest.raises(geometry.GeometryError, match="source bytes differ"):
        geometry.verify_external_source_manifest(prereg)


def test_bound_frame_decodes_authenticated_buffer_not_replaced_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "ABC.csv"
    original = b"timestamp,open,high,low,close,volume\n2020-01-02,1,2,0.5,1.5,10\n"
    replacement = b"timestamp,open,high,low,close,volume\n2020-01-02,9,9,9,9,99\n"
    path.write_bytes(original)
    record = {"path": str(path), "bytes": len(original), "sha256": sha256(original).hexdigest()}
    spec = DatasetSpec(dataset_id="nasdaq", adapter="directory", path=tmp_path, format="csv")
    read_csv = geometry.pd.read_csv

    def replace_then_decode(buffer):
        path.write_bytes(replacement)
        return read_csv(buffer)

    monkeypatch.setattr(geometry.pd, "read_csv", replace_then_decode)
    frame = geometry._bound_frame(record, spec, symbol="ABC")
    assert frame.loc[0, "close"] == 1.5
    assert path.read_bytes() == replacement


@pytest.mark.parametrize("field,value", [("bytes", 1), ("sha256", "0" * 64)])
def test_bound_bytes_refuses_wrong_size_or_digest(tmp_path: Path, field: str, value: object) -> None:
    path = tmp_path / "source"; path.write_bytes(b"frozen")
    record: dict[str, object] = {
        "path": str(path), "bytes": 6, "sha256": sha256(b"frozen").hexdigest(),
    }
    record[field] = value
    with pytest.raises(geometry.GeometryError, match="source bytes differ"):
        geometry._bound_bytes(record)


def test_frozen_dataset_spec_drift_is_refused(tmp_path: Path) -> None:
    spec = DatasetSpec(dataset_id="nasdaq", adapter="directory", path=tmp_path, format="parquet")
    frozen = json.loads(json.dumps(asdict(spec), default=str))
    geometry._validate_dataset_spec(spec, {"dataset_spec": frozen})
    with pytest.raises(geometry.GeometryError, match="specification differs"):
        geometry._validate_dataset_spec(spec, {"dataset_spec": {**frozen, "interval": "1h"}})


def test_query_prefix_drift_refused_before_representation(monkeypatch: pytest.MonkeyPatch) -> None:
    stock = _bars(); benchmark = _bars()
    cutoff = stock.timestamp.iloc[-1]
    key = EpisodeKey(InstrumentKey("nasdaq", "ABC"), cutoff, 252, "dense-v1")
    row = {
        "cutoff": cutoff.isoformat(), "lookback": 252, "representation_version": "dense-v1",
        "symbol": "ABC", "quality_tier": "A",
        "stock_prefix": asdict(causal_prefix_digest(stock, cutoff)),
        "benchmark_prefix": asdict(causal_prefix_digest(benchmark, cutoff)),
    }
    monkeypatch.setattr(geometry, "_vector", lambda episode: (np.zeros(141), "digest"))
    vector, audit = geometry.reconstruct_query(key.id, row, stock, benchmark)
    assert vector.shape == (141,) and audit["query_episode_id"] == key.id
    changed = stock.copy(); changed.loc[0, "close"] += 1.0
    with pytest.raises(geometry.GeometryError, match="stock prefix differs"):
        geometry.reconstruct_query(key.id, row, changed, benchmark)


def test_candidate_episode_identity_and_prefix_audit(monkeypatch: pytest.MonkeyPatch) -> None:
    stock = _bars(); benchmark = _bars(); cutoff = stock.timestamp.iloc[-1]
    key = EpisodeKey(InstrumentKey("nasdaq", "ABC"), cutoff, 252, "dense-v1")
    row = {"episode_id": key.id, "symbol": "ABC", "cutoff_ns": cutoff.value}
    monkeypatch.setattr(geometry, "_vector", lambda episode: (np.ones(141), "representation"))
    vector, audit = geometry.reconstruct_candidate(row, stock, benchmark)
    assert vector.shape == (141,) and audit["representation_digest"] == "representation"
    with pytest.raises(geometry.GeometryError, match="EpisodeKey differs"):
        geometry.reconstruct_candidate({**row, "episode_id": "0" * 24}, stock, benchmark)


def test_reconstruction_serial_threaded_byte_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    query_ids = ["q0", "q1", "q2"]
    candidate_ids = ["e0", "e1"]
    rows = [{"episode_id": value, "symbol": "A" if index < 2 else "B"}
            for index, value in enumerate(query_ids)]
    (tmp_path / geometry.joint.REGISTRY).parent.mkdir(parents=True)
    (tmp_path / geometry.joint.REGISTRY).write_text(json.dumps({"cases_data": rows}))
    registry_sha = geometry._file_sha(tmp_path / geometry.joint.REGISTRY)
    files = {"A": tmp_path / "A.parquet", "B": tmp_path / "B.parquet"}
    records = [{"symbol": symbol, "path": str(path)} for symbol, path in files.items()]
    spec = DatasetSpec(dataset_id="nasdaq", adapter="directory", path=tmp_path,
                       format="parquet", benchmark=BenchmarkSpec(path=tmp_path / "benchmark"))
    manifest_state = {
        "stock_files": records, "stock_files_count": 2, "required_symbols": ["A", "B"],
        "config": {"path": str(tmp_path / "config")},
        "benchmark": {"path": str(tmp_path / "benchmark")},
        "dataset_spec": json.loads(json.dumps(asdict(spec), default=str)),
    }
    manifest = {**manifest_state, "manifest_digest": stable_hash(manifest_state)}
    audits = [{"query_episode_id": value} for value in query_ids]
    prereg = {"authorities": {
        "external_source_manifest": manifest,
        "authority_file_sha256": {str(geometry.joint.REGISTRY): registry_sha},
        "sealed_query_transform": {"query_audits": audits},
        "population": {"query_ids": query_ids, "cohort_ids": candidate_ids,
                       "episodes": [{"episode_id": "e0", "symbol": "A"},
                                    {"episode_id": "e1", "symbol": "B"}]},
    }}

    class FakeSource:
        def __init__(self, _spec=None) -> None:
            self._files = files

    monkeypatch.setattr(geometry, "DirectorySource", FakeSource)
    monkeypatch.setattr(geometry, "_bound_bytes", lambda record: b"frozen")
    monkeypatch.setattr(geometry, "load_config_bytes",
                        lambda content, path: SimpleNamespace(datasets={"nasdaq": spec}))
    monkeypatch.setattr(geometry, "_bound_frame", lambda record, value, symbol, benchmark=False:
                        pd.DataFrame({"timestamp": [pd.Timestamp("2020-01-01")],
                                      "symbol": [symbol]}))
    monkeypatch.setattr(geometry, "reconstruct_query", lambda query_id, row, stock, benchmark:
                        (np.full(141, float(query_id[-1])), {"query_episode_id": query_id}))
    monkeypatch.setattr(geometry, "reconstruct_candidate", lambda row, stock, benchmark:
                        (np.full(141, 10.0 + float(row["episode_id"][-1])),
                         {"episode_id": row["episode_id"]}))
    serial = geometry.reconstruct_vectors(tmp_path, prereg, workers=1)
    threaded = geometry.reconstruct_vectors(tmp_path, prereg, workers=12)
    assert serial[0].tobytes() == threaded[0].tobytes()
    assert serial[1].tobytes() == threaded[1].tobytes()
    assert serial[2:] == threaded[2:]


def test_causal_eligibility_full_denominator() -> None:
    prereg = {"authorities": {"population": {
        "query_ids": ["q0", "q1", "q2"], "cohort_ids": ["e0", "e1"],
        "episodes": [
            {"episode_id": "e0", "eligible_query_ids": ["q0", "q2"]},
            {"episode_id": "e1", "eligible_query_ids": ["q1", "q2"]},
        ],
    }}}
    result = geometry.causal_eligibility(prereg)
    assert result.dtype == np.bool_
    assert result.tolist() == [[True, False], [False, True], [True, True]]


def test_causal_eligibility_refuses_unknown_or_duplicate_ids() -> None:
    base = {"authorities": {"population": {
        "query_ids": ["q0"], "cohort_ids": ["e0"],
        "episodes": [{"episode_id": "e0", "eligible_query_ids": ["q0", "q0"]}],
    }}}
    with pytest.raises(geometry.GeometryError, match="eligibility IDs differ"):
        geometry.causal_eligibility(base)


def test_atomic_json_is_create_only_and_rejects_nonfinite(tmp_path: Path) -> None:
    target = tmp_path / "IDS.json"
    geometry._atomic_json(target, {"x": 1})
    with pytest.raises(geometry.GeometryError, match="create-only"):
        geometry._atomic_json(target, {"x": 2})
    with pytest.raises(ValueError):
        geometry._atomic_json(tmp_path / "bad.json", {"x": np.nan})


def test_run_publishes_exact_closed_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prereg_path = tmp_path / "prereg.json"
    prereg = {
        "preregistration_digest": "p", "implementation_commit": "h0",
        "authorities": {"external_source_manifest": {
            "manifest_digest": "m", "source_lock_digest": "s", "stock_files_count": 1,
            "config": {"sha256": "c"}, "benchmark": {"sha256": "b"},
            "stock_files": [{"symbol": "ABC"}],
        }},
    }
    prereg_path.write_text(json.dumps(prereg))
    monkeypatch.setattr(geometry.joint, "PREREGISTRATION", Path("prereg.json"))
    monkeypatch.setattr(geometry.joint, "validate_h1", lambda repository, value: "h1")
    monkeypatch.setattr(geometry, "OUTPUT", Path("output"))
    monkeypatch.setattr(geometry, "WORK", Path("work"))
    eligibility = np.asarray([[True], [False]], dtype=np.bool_)
    arrays = {
        "raw_queries": np.zeros((2, 141)), "raw_candidates": np.zeros((1, 141)),
        "transformed_queries": np.zeros((2, 141)), "transformed_candidates": np.zeros((1, 141)),
        "query_pair_distances": np.zeros((2, 2)), "query_candidate_distances": np.zeros((2, 1)),
        "specificity_ranks": np.asarray([[0.5], [np.nan]]), "causal_eligibility": eligibility,
    }
    ids = {"query_ids_digest": "q", "candidate_ids_digest": "e", "population_digest": "pop"}
    monkeypatch.setattr(geometry, "compute", lambda *args, **kwargs: (arrays, ids, {"ok": True}))
    result = geometry.run(tmp_path, workers=12)
    assert result["passed"] is True
    output = tmp_path / "output"
    assert sorted(path.name for path in output.iterdir()) == [
        "IDS.json", "RESULT.json", *(f"{name}.npy" for name in sorted(geometry.ARRAY_NAMES))
    ]
    published = json.loads((output / "RESULT.json").read_text())
    assert published["geometry_digest"] == stable_hash({
        key: value for key, value in published.items() if key != "geometry_digest"
    })
    with pytest.raises(geometry.GeometryError, match="already exists"):
        geometry.run(tmp_path, workers=12)


def test_output_and_work_paths_are_separate() -> None:
    assert geometry.OUTPUT != geometry.WORK
    assert geometry.OUTPUT.name not in {"", ".", ".."}
    assert geometry.WORK.name.startswith(".")
