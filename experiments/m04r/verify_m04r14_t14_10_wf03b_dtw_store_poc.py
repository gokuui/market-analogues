"""Independent raw reconstruction verifier for the bounded DTW store POC."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from hashlib import sha256
import json
import multiprocessing
from pathlib import Path
import subprocess
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from numba import set_num_threads

from market_analogues.causal_prefix import causal_prefix_digest
from market_analogues.dtw_interval_bound import (
    DtwIntervalBoundError, quantize_dtw_samples, quantized_dtw_lower_bound,
)
from market_analogues.dtw_sample_store import (
    DTW_SAMPLE_DTYPE, DTW_SAMPLE_ROW_BYTES, dtw_sample_lower_bounds,
    make_dtw_sample_record_from_quantized, make_zero_dtw_sample_record,
    validate_dtw_sample_records,
)
from market_analogues.exact_batch import sliding_exact_representations
from market_analogues.packed_bound_store import decode_episode_id, load_packed_generation
from market_analogues.types import EpisodeKey, InstrumentKey, stable_hash

from experiments.m04r import m04r14_t14_10_wf03_feasibility as base


PRODUCER_ROOT = Path("config/data/analogues/m04r14/t14-10-wf03b-dtw-store-poc-v2")
PREREGISTRATION = Path(
    "experiments/m04r/m04r14_t14_10_wf03b_dtw_store_poc_v2_preregistered.json"
)
VERIFICATION_ROOT = Path(
    "config/data/analogues/m04r14/t14-10-wf03b-dtw-store-poc-v2-verification"
)
TOLERANCE = 1e-12
MAX_PROJECTED_SCAN_SECONDS = 90.0

_SOURCE: Any = None
_BENCHMARK: pd.DataFrame | None = None
_QUERY: Any = None
_PACKED: Any = None
_MAXIMUM: pd.Timestamp | None = None
_PRODUCER_ROOT: Path | None = None


class StoreVerificationError(RuntimeError):
    pass


def _stem(symbol: str) -> str:
    return sha256(symbol.encode()).hexdigest()


def _paths(root: Path, symbol: str) -> tuple[Path, Path, Path]:
    stem = root / _stem(symbol)
    return (stem.with_suffix(".rows.bin"), stem.with_suffix(".overflow.bin"),
            stem.with_suffix(".json"))


def _slice(records: np.ndarray, symbol_id: int) -> np.ndarray:
    ids = records["symbol_id"]
    first = int(np.searchsorted(ids, symbol_id, side="left"))
    last = int(np.searchsorted(ids, symbol_id, side="right"))
    return records[first:last]


def _episode_ids(records: np.ndarray) -> list[str]:
    return [decode_episode_id(row["episode_id"]) for row in records]


def _verify_symbol(specification: Mapping[str, Any]) -> dict[str, Any]:
    set_num_threads(1)
    if _SOURCE is None or _BENCHMARK is None or _QUERY is None \
            or _PACKED is None or _MAXIMUM is None or _PRODUCER_ROOT is None:
        raise StoreVerificationError("independent worker is not initialized")
    symbol = specification["symbol"]; symbol_id = int(specification["symbol_id"])
    full = _SOURCE.load(InstrumentKey("nasdaq", symbol))
    if asdict(causal_prefix_digest(full, _MAXIMUM)) != specification["source_prefix"]:
        raise StoreVerificationError("independent source prefix differs")
    main_packed = _slice(_PACKED.rows, symbol_id)
    overflow_packed = _slice(_PACKED.overflow, symbol_id)
    main_ids = {value: index for index, value in enumerate(_episode_ids(main_packed))}
    overflow_ids = {value: index for index, value in enumerate(_episode_ids(overflow_packed))}
    main = np.zeros(len(main_packed), dtype=DTW_SAMPLE_DTYPE)
    overflow = np.zeros(len(overflow_packed), dtype=DTW_SAMPLE_DTYPE)
    producer_metadata = base._read(_paths(
        _PRODUCER_ROOT / "build-a", symbol,
    )[2])
    base._validate_seal(producer_metadata, "shard_digest")
    probe_map = {row["episode_id"]: row for row in producer_metadata["scalar_probes"]}
    probe_deltas = []
    zero_rows = 0
    seen = set()
    frame = full[full.timestamp <= _MAXIMUM].reset_index(drop=True)
    batch = sliding_exact_representations(
        frame, _BENCHMARK, lookback=252, stride=5, batch_size=512,
    )
    for position, representation in zip(batch.positions, batch.representations, strict=True):
        cutoff = pd.Timestamp(frame.timestamp.iloc[int(position)])
        episode_id = EpisodeKey(
            InstrumentKey("nasdaq", symbol), cutoff, 252, "dense-v1",
        ).id
        if episode_id in main_ids:
            lane = "main"; index = main_ids[episode_id]
        elif episode_id in overflow_ids:
            lane = "overflow"; index = overflow_ids[episode_id]
        else:
            raise StoreVerificationError("independent episode is outside packed slice")
        if episode_id in seen:
            raise StoreVerificationError("independent episode repeats")
        seen.add(episode_id)
        try:
            quantized = quantize_dtw_samples(representation)
            record = make_dtw_sample_record_from_quantized(quantized)
        except DtwIntervalBoundError:
            quantized = None
            record = make_zero_dtw_sample_record(); zero_rows += 1
        if lane == "main":
            main[index] = record[0]
        else:
            overflow[index] = record[0]
        if episode_id in probe_map:
            scalar = (quantized_dtw_lower_bound(_QUERY, quantized)
                      if quantized is not None else 0.0)
            expected = probe_map[episode_id]
            if expected["lane"] != lane or expected["lane_index"] != index:
                raise StoreVerificationError("independent scalar probe alignment differs")
            probe_deltas.append(abs(float.fromhex(expected["scalar_bound_hex"]) - scalar))
    if seen != set(main_ids) | set(overflow_ids):
        raise StoreVerificationError("independent episode inventory differs")
    validate_dtw_sample_records(main); validate_dtw_sample_records(overflow)
    for build in ("build-a", "build-b"):
        row_path, overflow_path, _metadata = _paths(_PRODUCER_ROOT / build, symbol)
        if not np.array_equal(main, np.fromfile(row_path, dtype=DTW_SAMPLE_DTYPE)) \
                or not np.array_equal(
                    overflow, np.fromfile(overflow_path, dtype=DTW_SAMPLE_DTYPE)):
            raise StoreVerificationError("independent raw reconstruction differs")
    return {
        "symbol": symbol, "symbol_id": symbol_id,
        "rows": len(main), "overflow_rows": len(overflow),
        "zero_bound_rows": zero_rows, "scalar_probes": len(probe_deltas),
        "maximum_scalar_delta": max(probe_deltas, default=0.0),
        "main_digest": stable_hash(main.tobytes().hex()),
        "overflow_digest": stable_hash(overflow.tobytes().hex()),
    }


def _load_all(root: Path, selection: Sequence[Mapping[str, Any]]) -> np.ndarray:
    main = []; overflow = []
    for row in selection:
        main_path, overflow_path, _metadata = _paths(root, row["symbol"])
        main.append(np.fromfile(main_path, dtype=DTW_SAMPLE_DTYPE))
        overflow.append(np.fromfile(overflow_path, dtype=DTW_SAMPLE_DTYPE))
    return np.concatenate((*main, *overflow))


def verify(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    prereg = base._read(repository / PREREGISTRATION)
    base._validate_seal(prereg, "preregistration_digest")
    root = repository / PRODUCER_ROOT
    contract = base._read(root / "CONTRACT.json")
    result = base._read(root / "RESULT.json"); base._validate_seal(result)
    if contract != prereg or result.get("passed") is not True \
            or result.get("full_store_build_authorized") is not True:
        raise StoreVerificationError("producer evidence differs")
    head = prereg["implementation_commit"]
    subprocess.run(["git", "merge-base", "--is-ancestor", head, "HEAD"],
                   cwd=repository, check=True)
    for path, digest in prereg["runtime_files"].items():
        blob = subprocess.run(["git", "show", f"{head}:{path}"], cwd=repository,
                              capture_output=True, check=False)
        if blob.returncode or sha256(blob.stdout).hexdigest() != digest \
                or base._sha(repository / path) != digest:
            raise StoreVerificationError("frozen producer runtime differs")
    resident = base._resident()
    packed = load_packed_generation(
        Path(resident["store_root"]), base.GENERATION_ID,
        expected_provenance_digest=base.PROVENANCE_DIGEST,
        verify_content=False, validate_records=False,
    )
    _registry, by_id = base._registry(repository)
    source, _episode, _request, query = base._context(
        repository, by_id[prereg["inputs"]["query_id"]],
    )
    benchmark = source.load_benchmark()
    if benchmark is None:
        raise StoreVerificationError("independent benchmark is absent")
    maximum = pd.Timestamp(packed.manifest["provenance"]["benchmark_prefix"][
        "requested_cutoff"
    ])
    global _SOURCE, _BENCHMARK, _QUERY, _PACKED, _MAXIMUM, _PRODUCER_ROOT
    _SOURCE, _BENCHMARK, _QUERY, _PACKED = source, benchmark, query.representation, packed
    _MAXIMUM, _PRODUCER_ROOT = maximum, root
    with ProcessPoolExecutor(
        max_workers=8, mp_context=multiprocessing.get_context("fork"),
    ) as executor:
        cases = list(executor.map(_verify_symbol, reversed(prereg["selection"])))
    cases.sort(key=lambda row: row["symbol_id"])
    records = _load_all(root / "build-b", prereg["selection"])
    dtw_sample_lower_bounds(query.representation, records[:1])
    timings = []
    digests = []
    for _ in range(3):
        started = perf_counter()
        values = dtw_sample_lower_bounds(query.representation, records)
        timings.append(perf_counter() - started)
        digests.append(stable_hash([float(value).hex() for value in values]))
    if len(set(digests)) != 1:
        raise StoreVerificationError("independent batch scan differs")
    eligible = int(packed.manifest["row_count"]) + int(packed.manifest["overflow_count"])
    projected = float(np.median(timings)) * eligible / len(records)
    maximum_delta = float(max(row["maximum_scalar_delta"] for row in cases))
    gates = {
        "all_raw_rows_equal": sum(row["rows"] for row in cases) == result["rows"]
            and sum(row["overflow_rows"] for row in cases) == result["overflow_rows"],
        "zero_rows_equal": sum(row["zero_bound_rows"] for row in cases)
            == result["zero_bound_rows"],
        "scalar_probes_equal": sum(row["scalar_probes"] for row in cases)
            == result["scalar_probes"] and maximum_delta <= TOLERANCE,
        "scan_deterministic": True,
        "scan_performance_passed": projected <= MAX_PROJECTED_SCAN_SECONDS,
        "capacity_recomputed": eligible * DTW_SAMPLE_ROW_BYTES
            == result["projected_store_bytes"],
    }
    if not all(gates.values()):
        raise StoreVerificationError("independent store gate differs")
    state = {
        "schema_version": "m04r14-t14-10-wf03b-dtw-store-verification-v1",
        "status": "verified", "passed": True, "gates": gates,
        "producer_result_digest": result["result_digest"],
        "producer_result_sha256": base._sha(root / "RESULT.json"),
        "cases": cases, "case_digest": stable_hash(cases),
        "maximum_scalar_delta": maximum_delta,
        "scan_seconds": [float(value) for value in timings],
        "projected_full_scan_seconds": projected,
        "scan_digest": digests[0], "full_store_build_authorized": True,
        "historical_walk_forward_query_outcomes_opened": False,
        "final_period_result_opened": False,
        "production_promotion_authorized": False,
    }
    return base._sealed(state, "verification_digest")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    result = verify(args.repository)
    if not args.dry_run:
        root = args.repository.resolve() / VERIFICATION_ROOT
        if root.exists() or root.is_symlink():
            raise StoreVerificationError("verification root exists")
        root.mkdir(parents=True)
        base._atomic(root / "VERIFIED.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
