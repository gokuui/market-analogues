"""Build the preregistered full-universe aligned WF-03 baseline feature store."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from hashlib import sha256
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from market_analogues.baseline_feature_store import (
    FEATURE_DTYPE,
    baseline_feature_store_contract,
    features_for_packed_records,
    load_feature_generation,
    validate_feature_records,
    write_feature_generation_from_shards,
)
from market_analogues.causal_prefix import causal_prefix_digest
from market_analogues.packed_bound_store import load_packed_generation
from market_analogues.resident_store import resident_file_identity_lease
from market_analogues.types import InstrumentKey, stable_hash

from experiments.m04r import m04r14_t14_10_wf03_baseline_poc as bounded
from experiments.m04r import m04r14_t14_10_wf03_feasibility as base


SCHEMA = "m04r14-t14-10-wf03-baseline-store-full-preregistration-v1"
OUTPUT_RELATIVE = Path(
    "config/data/analogues/m04r14/t14-10-wf03-baseline-store-full-v1"
)
PREREGISTRATION_RELATIVE = Path(
    "experiments/m04r/m04r14_t14_10_wf03_baseline_store_full_preregistered.json"
)
BOUNDED_VERIFICATION_RELATIVE = Path(
    "config/data/analogues/m04r14/"
    "t14-10-wf03-baseline-poc-v3-verification/VERIFIED.json"
)
WORKERS = 8
RUNTIME_FILES = (
    "experiments/m04r/m04r14_t14_10_wf03_baseline_store_full.py",
    "experiments/m04r/m04r14_t14_10_wf03_baseline_poc.py",
    "experiments/m04r/m04r14_t14_10_wf03_feasibility.py",
    "src/market_analogues/baseline_feature_store.py",
    "src/market_analogues/baseline_neighbors.py",
    "src/market_analogues/causal_prefix.py",
    "src/market_analogues/packed_bound_store.py",
    "src/market_analogues/resident_store.py",
)

_SOURCE: Any = None
_PACKED: Any = None
_MAXIMUM: pd.Timestamp | None = None


class FullBaselineStoreError(RuntimeError):
    pass


def _git(repository: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repository, text=True, capture_output=True, check=False,
    )
    if result.returncode:
        raise FullBaselineStoreError(f"git {' '.join(args)} failed")
    return result.stdout.strip()


def _replace_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w") as handle:
        json.dump(dict(value), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _prerequisites(repository: Path) -> tuple[Any, Any, dict[str, Any]]:
    verification = base._read(repository / BOUNDED_VERIFICATION_RELATIVE)
    base._validate_seal(verification, "verification_digest")
    if verification.get("passed") is not True \
            or verification.get("full_feature_store_build_authorized") is not True \
            or verification.get("outcomes_or_labels_used") is not False:
        raise FullBaselineStoreError("bounded baseline verification does not authorize build")
    resident = base._resident()
    loaded = load_packed_generation(
        Path(resident["store_root"]), base.GENERATION_ID,
        expected_provenance_digest=base.PROVENANCE_DIGEST,
        verify_content=False, validate_records=False,
    )
    _registry, by_id = base._registry(repository)
    source, _episode, _request, _query = base._context(
        repository, by_id[base.PROBES[0][1]],
    )
    return loaded, source, verification


def full_selection(loaded: Any) -> list[dict[str, Any]]:
    output = []
    for symbol_id, symbol in enumerate(loaded.symbols):
        main = bounded._slice(loaded.rows, symbol_id)
        overflow = bounded._slice(loaded.overflow, symbol_id)
        output.append({
            "symbol": symbol,
            "symbol_id": symbol_id,
            "rows": len(main),
            "overflow_rows": len(overflow),
            "source_prefix": loaded.manifest["provenance"]["source_prefixes"][symbol],
        })
    if sum(row["rows"] for row in output) != len(loaded.rows) \
            or sum(row["overflow_rows"] for row in output) != len(loaded.overflow):
        raise FullBaselineStoreError("full baseline selection count differs")
    return output


def build_preregistration(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    if _git(repository, "status", "--porcelain"):
        raise FullBaselineStoreError("full baseline preregistration requires clean commit")
    loaded, _source, verification = _prerequisites(repository)
    selection = full_selection(loaded)
    output = repository / OUTPUT_RELATIVE
    if output.exists() or output.is_symlink():
        raise FullBaselineStoreError("full baseline output must be absent before freeze")
    state = {
        "schema_version": SCHEMA,
        "status": "frozen_before_full_baseline_feature_build",
        "implementation_commit": _git(repository, "rev-parse", "HEAD"),
        "runtime_files": {
            path: base._sha(repository / path) for path in RUNTIME_FILES
        },
        "inputs": {
            "bounded_verification_digest": verification["verification_digest"],
            "bounded_verification_sha256": base._sha(
                repository / BOUNDED_VERIFICATION_RELATIVE
            ),
            "packed_generation_id": loaded.generation_id,
            "packed_manifest_digest": loaded.manifest["manifest_digest"],
            "packed_rows_sha256": loaded.manifest["rows_sha256"],
            "packed_overflow_sha256": loaded.manifest["overflow_sha256"],
            "packed_provenance_digest": loaded.manifest["provenance_digest"],
        },
        "selection": selection,
        "selection_digest": stable_hash(selection),
        "feature_store_contract": baseline_feature_store_contract(),
        "execution": {
            "symbols": len(selection),
            "workers": WORKERS,
            "rows": len(loaded.rows),
            "overflow_rows": len(loaded.overflow),
            "expected_bytes": (
                len(loaded.rows) + len(loaded.overflow)
            ) * FEATURE_DTYPE.itemsize,
            "resume": "reuse only sealed shards whose counts and hashes revalidate",
            "output_root": str(output.resolve()),
        },
        "claims": {
            "outcomes_or_labels_used": False,
            "historical_walk_forward_query_outcomes_opened": False,
            "final_period_result_opened": False,
            "production_promotion_authorized": False,
        },
    }
    return base._sealed(state, "preregistration_digest")


def validate_preregistration(
    repository: Path, value: Mapping[str, Any],
) -> tuple[Any, Any, list[dict[str, Any]]]:
    base._validate_seal(value, "preregistration_digest")
    loaded, source, verification = _prerequisites(repository)
    selection = full_selection(loaded)
    if value.get("schema_version") != SCHEMA \
            or value.get("selection") != selection \
            or value.get("selection_digest") != stable_hash(selection) \
            or value.get("feature_store_contract") != baseline_feature_store_contract() \
            or value.get("inputs", {}).get("bounded_verification_digest") \
            != verification["verification_digest"]:
        raise FullBaselineStoreError("full baseline preregistration differs")
    head = value.get("implementation_commit")
    if type(head) is not str:
        raise FullBaselineStoreError("full baseline implementation differs")
    _git(repository, "merge-base", "--is-ancestor", head, "HEAD")
    for path, digest in value["runtime_files"].items():
        blob = subprocess.run(
            ["git", "show", f"{head}:{path}"], cwd=repository,
            capture_output=True, check=False,
        )
        if blob.returncode or sha256(blob.stdout).hexdigest() != digest \
                or base._sha(repository / path) != digest:
            raise FullBaselineStoreError(f"full baseline runtime differs: {path}")
    return loaded, source, selection


def _valid_shard(root: Path, specification: Mapping[str, Any]) -> dict[str, Any] | None:
    rows_path, overflow_path, metadata_path = bounded._paths(
        root, str(specification["symbol"]),
    )
    if not all(path.is_file() and not path.is_symlink()
               for path in (rows_path, overflow_path, metadata_path)):
        return None
    try:
        metadata = base._read(metadata_path)
        base._validate_seal(metadata, "shard_digest")
        main = np.fromfile(rows_path, dtype=FEATURE_DTYPE)
        overflow = np.fromfile(overflow_path, dtype=FEATURE_DTYPE)
        validate_feature_records(main)
        validate_feature_records(overflow)
        valid = all((
            metadata["schema_version"] == "m04r14-wf03-baseline-full-shard-v1",
            metadata["symbol"] == specification["symbol"],
            metadata["symbol_id"] == specification["symbol_id"],
            metadata["source_prefix"] == specification["source_prefix"],
            metadata["rows"] == specification["rows"] == len(main),
            metadata["overflow_rows"] == specification["overflow_rows"] == len(overflow),
            metadata["rows_sha256"] == base._sha(rows_path),
            metadata["overflow_sha256"] == base._sha(overflow_path),
        ))
    except Exception:
        return None
    return metadata if valid else None


def _build_symbol(task: tuple[dict[str, Any], str]) -> dict[str, Any]:
    specification, root_value = task
    if _SOURCE is None or _PACKED is None or _MAXIMUM is None:
        raise FullBaselineStoreError("full baseline worker is not initialized")
    root = Path(root_value)
    existing = _valid_shard(root, specification)
    if existing is not None:
        return existing
    symbol = str(specification["symbol"])
    symbol_id = int(specification["symbol_id"])
    frame = _SOURCE.load(InstrumentKey("nasdaq", symbol))
    prefix = asdict(causal_prefix_digest(frame, _MAXIMUM))
    if prefix != specification["source_prefix"]:
        raise FullBaselineStoreError(f"full baseline source changed: {symbol}")
    frame = frame[frame.timestamp <= _MAXIMUM].reset_index(drop=True)
    main = features_for_packed_records(
        frame, bounded._slice(_PACKED.rows, symbol_id),
    )
    overflow = features_for_packed_records(
        frame, bounded._slice(_PACKED.overflow, symbol_id),
    )
    rows_path, overflow_path, metadata_path = bounded._paths(root, symbol)
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    bounded._write_array(rows_path, main)
    bounded._write_array(overflow_path, overflow)
    state = {
        "schema_version": "m04r14-wf03-baseline-full-shard-v1",
        "symbol": symbol,
        "symbol_id": symbol_id,
        "source_prefix": prefix,
        "rows": len(main),
        "overflow_rows": len(overflow),
        "missing_rows": int(np.isnan(main["values"]).all(axis=1).sum()),
        "missing_overflow_rows": int(np.isnan(overflow["values"]).all(axis=1).sum()),
        "rows_sha256": base._sha(rows_path),
        "overflow_sha256": base._sha(overflow_path),
    }
    metadata = base._sealed(state, "shard_digest")
    base._atomic(metadata_path, metadata)
    return metadata


def execute(repository: Path, preregistration: Mapping[str, Any]) -> dict[str, Any]:
    started = perf_counter()
    loaded, source, selection = validate_preregistration(repository, preregistration)
    root = repository / OUTPUT_RELATIVE
    if root.is_symlink() or root.exists() and not root.is_dir():
        raise FullBaselineStoreError("full baseline output path differs")
    if root.exists():
        if base._read(root / "CONTRACT.json") != preregistration:
            raise FullBaselineStoreError("full baseline resume contract differs")
        result_path = root / "RESULT.json"
        if result_path.exists():
            result = base._read(result_path)
            base._validate_seal(result)
            return result
    else:
        root.mkdir(parents=True)
        base._atomic(root / "CONTRACT.json", preregistration)
    work = root / "shards"
    work.mkdir(exist_ok=True)
    maximum = pd.Timestamp(loaded.manifest["provenance"]["benchmark_prefix"][
        "requested_cutoff"
    ])
    lease_before = resident_file_identity_lease(base.RESIDENT_ROOT / "READY.json")
    global _SOURCE, _PACKED, _MAXIMUM
    _SOURCE, _PACKED, _MAXIMUM = source, loaded, maximum
    tasks = [(dict(row), str(work)) for row in selection]
    metadata = []
    with ProcessPoolExecutor(
        max_workers=WORKERS, mp_context=multiprocessing.get_context("fork"),
    ) as executor:
        for completed, row in enumerate(
            executor.map(_build_symbol, tasks, chunksize=8), start=1,
        ):
            metadata.append(row)
            if completed % 128 == 0 or completed == len(tasks):
                _replace_json(root / "PROGRESS.json", {
                    "schema_version": "m04r14-wf03-baseline-full-progress-v1",
                    "status": "building" if completed < len(tasks) else "publishing",
                    "completed_symbols": completed,
                    "total_symbols": len(tasks),
                    "rows": sum(int(value["rows"]) for value in metadata),
                    "overflow_rows": sum(
                        int(value["overflow_rows"]) for value in metadata
                    ),
                })
    metadata.sort(key=lambda row: row["symbol_id"])
    row_paths = [bounded._paths(work, row["symbol"])[0] for row in selection]
    overflow_paths = [bounded._paths(work, row["symbol"])[1] for row in selection]
    generation_id = write_feature_generation_from_shards(
        root / "store", row_paths, overflow_paths,
        packed_manifest=loaded.manifest,
        provenance={
            "preregistration_digest": preregistration["preregistration_digest"],
            "selection_digest": preregistration["selection_digest"],
            "bounded_verification_digest": preregistration["inputs"][
                "bounded_verification_digest"
            ],
            "workers": WORKERS,
            "outcomes_or_labels_used": False,
        },
    )
    generation = load_feature_generation(
        root / "store", generation_id, packed_manifest=loaded.manifest,
    )
    lease_after = resident_file_identity_lease(base.RESIDENT_ROOT / "READY.json")
    missing_main = int(np.isnan(generation.rows["values"]).all(axis=1).sum())
    missing_overflow = int(np.isnan(generation.overflow["values"]).all(axis=1).sum())
    gates = {
        "all_symbols_built": len(metadata) == len(selection),
        "packed_alignment": (
            len(generation.rows) == len(loaded.rows)
            and len(generation.overflow) == len(loaded.overflow)
        ),
        "all_features_finite": missing_main == 0 and missing_overflow == 0,
        "content_addressed_generation_valid": True,
        "resident_lease_unchanged": (
            lease_before["lease_digest"] == lease_after["lease_digest"]
        ),
    }
    if not all(gates.values()):
        raise FullBaselineStoreError("full baseline terminal gate differs")
    state = {
        "schema_version": "m04r14-t14-10-wf03-baseline-store-full-result-v1",
        "status": "complete",
        "passed": True,
        "gates": gates,
        "generation_id": generation_id,
        "generation_manifest_digest": generation.manifest["manifest_digest"],
        "selection_digest": preregistration["selection_digest"],
        "symbols": len(metadata),
        "rows": len(generation.rows),
        "overflow_rows": len(generation.overflow),
        "store_bytes": (len(generation.rows) + len(generation.overflow))
            * FEATURE_DTYPE.itemsize,
        "missing_rows": missing_main,
        "missing_overflow_rows": missing_overflow,
        "elapsed_seconds": perf_counter() - started,
        "resident_lease_digest": lease_after["lease_digest"],
        "outcomes_or_labels_used": False,
        "historical_walk_forward_query_outcomes_opened": False,
        "final_period_result_opened": False,
        "production_promotion_authorized": False,
        "independent_verification_authorized": True,
    }
    result = base._sealed(state)
    base._atomic(root / "RESULT.json", result)
    _replace_json(root / "PROGRESS.json", {
        "schema_version": "m04r14-wf03-baseline-full-progress-v1",
        "status": "complete",
        "completed_symbols": len(metadata),
        "total_symbols": len(metadata),
        "rows": len(generation.rows),
        "overflow_rows": len(generation.overflow),
        "result_digest": result["result_digest"],
    })
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preregister", "run"))
    parser.add_argument("--repository", required=True, type=Path)
    args = parser.parse_args(argv)
    repository = args.repository.resolve(strict=True)
    path = repository / PREREGISTRATION_RELATIVE
    if args.mode == "preregister":
        base._atomic(path, build_preregistration(repository))
        return 0
    result = execute(repository, base._read(path))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
