"""Independently verify the full aligned WF-03 baseline feature generation."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
import multiprocessing
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import numpy as np
import pandas as pd

from market_analogues.baseline_feature_store import (
    FEATURE_DTYPE,
    load_feature_generation,
    validate_feature_records,
)
from market_analogues.causal_prefix import causal_prefix_digest
from market_analogues.packed_bound_store import load_packed_generation
from market_analogues.types import InstrumentKey, stable_hash

from experiments.m04r import m04r14_t14_10_wf03_baseline_poc as bounded
from experiments.m04r import m04r14_t14_10_wf03_baseline_store_full as producer
from experiments.m04r import m04r14_t14_10_wf03_feasibility as base
from experiments.m04r import verify_m04r14_t14_10_wf03_baseline_poc as bounded_verifier


OUTPUT_RELATIVE = Path(
    "config/data/analogues/m04r14/"
    "t14-10-wf03-baseline-store-full-v1-verification"
)
WORKERS = 8

_SOURCE: Any = None
_PACKED: Any = None
_FEATURES: Any = None
_MAXIMUM: pd.Timestamp | None = None
_SHARD_ROOT: Path | None = None


class FullBaselineVerificationError(RuntimeError):
    pass


def _verify_symbol(specification: dict[str, Any]) -> dict[str, Any]:
    if _SOURCE is None or _PACKED is None or _FEATURES is None \
            or _MAXIMUM is None or _SHARD_ROOT is None:
        raise FullBaselineVerificationError("full baseline verifier is uninitialized")
    symbol = str(specification["symbol"])
    symbol_id = int(specification["symbol_id"])
    frame = _SOURCE.load(InstrumentKey("nasdaq", symbol))
    if asdict(causal_prefix_digest(frame, _MAXIMUM)) != specification["source_prefix"]:
        raise FullBaselineVerificationError(f"full baseline source differs: {symbol}")
    frame = frame[frame.timestamp <= _MAXIMUM].reset_index(drop=True)
    main_records = bounded._slice(_PACKED.rows, symbol_id)
    overflow_records = bounded._slice(_PACKED.overflow, symbol_id)
    rows_path, overflow_path, metadata_path = bounded._paths(_SHARD_ROOT, symbol)
    metadata = base._read(metadata_path)
    base._validate_seal(metadata, "shard_digest")
    shard_main = np.fromfile(rows_path, dtype=FEATURE_DTYPE)
    shard_overflow = np.fromfile(overflow_path, dtype=FEATURE_DTYPE)
    validate_feature_records(shard_main)
    validate_feature_records(shard_overflow)
    main_first = int(np.searchsorted(
        _PACKED.rows["symbol_id"], symbol_id, side="left",
    ))
    overflow_first = int(np.searchsorted(
        _PACKED.overflow["symbol_id"], symbol_id, side="left",
    ))
    if not np.array_equal(
        shard_main, _FEATURES.rows[main_first:main_first + len(main_records)],
    ) or not np.array_equal(
        shard_overflow,
        _FEATURES.overflow[overflow_first:overflow_first + len(overflow_records)],
    ):
        raise FullBaselineVerificationError("full generation/shard alignment differs")
    if metadata["rows_sha256"] != base._sha(rows_path) \
            or metadata["overflow_sha256"] != base._sha(overflow_path) \
            or not (metadata["rows"] == specification["rows"] == len(main_records)) \
            or not (metadata["overflow_rows"] == specification["overflow_rows"]
                    == len(overflow_records)):
        raise FullBaselineVerificationError("full baseline shard receipt differs")
    main_probes = sorted(set(
        value for value in (0, len(main_records) - 1) if value >= 0
    ))
    overflow_probes = list(range(len(overflow_records)))
    maximum_delta = 0.0
    non_bitwise = 0
    values_checked = 0
    for records, observed, probes in (
        (main_records, shard_main["values"], main_probes),
        (overflow_records, shard_overflow["values"], overflow_probes),
    ):
        if not probes:
            continue
        selected = records[np.asarray(probes, dtype=np.int64)]
        expected = bounded_verifier._independent_feature_rows(frame, selected)
        actual = observed[np.asarray(probes, dtype=np.int64)]
        if not np.array_equal(np.isnan(actual), np.isnan(expected)):
            raise FullBaselineVerificationError("full baseline sample missingness differs")
        finite = np.isfinite(actual) & np.isfinite(expected)
        if np.any(finite):
            maximum_delta = max(maximum_delta, float(np.max(
                np.abs(actual[finite] - expected[finite])
            )))
            non_bitwise += int(np.sum(
                actual[finite].view(np.uint64) != expected[finite].view(np.uint64)
            ))
        values_checked += int(actual.size)
    return {
        "symbol_id": symbol_id,
        "rows": len(main_records),
        "overflow_rows": len(overflow_records),
        "feature_values_checked": values_checked,
        "non_bitwise_values": non_bitwise,
        "maximum_delta": maximum_delta,
        "shard_digest": metadata["shard_digest"],
    }


def verify(repository: Path) -> dict[str, Any]:
    started = perf_counter()
    repository = repository.resolve(strict=True)
    root = repository / producer.OUTPUT_RELATIVE
    preregistration = base._read(repository / producer.PREREGISTRATION_RELATIVE)
    base._validate_seal(preregistration, "preregistration_digest")
    result = base._read(root / "RESULT.json")
    base._validate_seal(result)
    if result.get("passed") is not True \
            or result.get("independent_verification_authorized") is not True \
            or result.get("outcomes_or_labels_used") is not False:
        raise FullBaselineVerificationError("full baseline producer result differs")
    loaded, source, verification = producer._prerequisites(repository)
    if verification["verification_digest"] != preregistration["inputs"][
        "bounded_verification_digest"
    ]:
        raise FullBaselineVerificationError("bounded verification binding differs")
    generation = load_feature_generation(
        root / "store", result["generation_id"],
        packed_manifest=loaded.manifest, verify_content=True,
    )
    validate_feature_records(generation.rows)
    validate_feature_records(generation.overflow)
    maximum = pd.Timestamp(loaded.manifest["provenance"]["benchmark_prefix"][
        "requested_cutoff"
    ])
    global _SOURCE, _PACKED, _FEATURES, _MAXIMUM, _SHARD_ROOT
    _SOURCE, _PACKED, _FEATURES = source, loaded, generation
    _MAXIMUM, _SHARD_ROOT = maximum, root / "shards"
    with ProcessPoolExecutor(
        max_workers=WORKERS, mp_context=multiprocessing.get_context("fork"),
    ) as executor:
        observations = list(executor.map(
            _verify_symbol, preregistration["selection"], chunksize=8,
        ))
    observations.sort(key=lambda row: row["symbol_id"])
    maximum_delta = max(row["maximum_delta"] for row in observations)
    non_bitwise = sum(row["non_bitwise_values"] for row in observations)
    rows = sum(row["rows"] for row in observations)
    overflow_rows = sum(row["overflow_rows"] for row in observations)
    gates = {
        "producer_and_preregistration_seals_valid": True,
        "content_hashes_valid": True,
        "all_symbol_source_prefixes_valid": len(observations) == len(loaded.symbols),
        "every_shard_matches_published_generation": True,
        "packed_row_counts_equal": rows == len(loaded.rows),
        "packed_overflow_counts_equal": overflow_rows == len(loaded.overflow),
        "raw_feature_samples_at_most_1e_12": maximum_delta <= 1e-12,
        "all_published_features_finite": bool(
            np.isfinite(generation.rows["values"]).all()
            and np.isfinite(generation.overflow["values"]).all()
        ),
        "outcomes_or_labels_excluded": True,
    }
    if not all(gates.values()):
        raise FullBaselineVerificationError("full baseline verification gate differs")
    state = {
        "schema_version": "m04r14-t14-10-wf03-baseline-store-full-verification-v1",
        "status": "complete",
        "passed": True,
        "gates": gates,
        "producer_result_digest": result["result_digest"],
        "preregistration_digest": preregistration["preregistration_digest"],
        "generation_id": generation.generation_id,
        "symbols_verified": len(observations),
        "rows_verified": rows,
        "overflow_rows_verified": overflow_rows,
        "feature_values_reconstructed": sum(
            row["feature_values_checked"] for row in observations
        ),
        "non_bitwise_feature_values": non_bitwise,
        "maximum_feature_delta": maximum_delta,
        "shard_receipt_digest": stable_hash(observations),
        "elapsed_seconds": perf_counter() - started,
        "outcomes_or_labels_used": False,
        "historical_walk_forward_query_outcomes_opened": False,
        "final_period_result_opened": False,
        "monthly_committee_batch_retrieval_authorized": True,
    }
    return base._sealed(state, "verification_digest")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    result = verify(args.repository)
    if not args.dry_run:
        root = args.repository.resolve() / OUTPUT_RELATIVE
        if root.exists() or root.is_symlink():
            raise FullBaselineVerificationError("full baseline verification output exists")
        root.mkdir(parents=True)
        base._atomic(root / "VERIFIED.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
