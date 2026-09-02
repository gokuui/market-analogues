"""Independently verify the bounded real WF-03 neighbor-baseline gate."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import subprocess
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from market_analogues.baseline_feature_store import FEATURE_DTYPE
from market_analogues.causal_prefix import causal_prefix_digest
from market_analogues.packed_bound_store import load_packed_generation
from market_analogues.types import InstrumentKey, stable_hash

from experiments.m04r import m04r14_t14_10_wf03_baseline_poc as producer
from experiments.m04r import m04r14_t14_10_wf03_feasibility as base


OUTPUT_RELATIVE = Path(
    "config/data/analogues/m04r14/t14-10-wf03-baseline-poc-v3-verification"
)


class BaselinePocVerificationError(RuntimeError):
    pass


def _independent_feature_rows(frame: pd.DataFrame, records: np.ndarray) -> np.ndarray:
    timestamps = frame["timestamp"].to_numpy(dtype="datetime64[ns]").view(np.int64)
    positions = {int(value): position for position, value in enumerate(timestamps)}
    close = frame["close"].to_numpy(dtype=np.float64)
    output = np.full((len(records), 3), np.nan, dtype=np.float64)
    for row_number, cutoff in enumerate(records["cutoff_ns"]):
        position = positions.get(int(cutoff))
        if position is None:
            raise BaselinePocVerificationError("independent cutoff reconstruction failed")
        if position < 63:
            continue
        window = close[position - 63:position + 1]
        if len(window) != 64 or not np.isfinite(window).all() or np.any(window <= 0):
            continue
        output[row_number] = (
            window[-1] / window[-21] - 1.0,
            window[-1] / window[0] - 1.0,
            np.std(np.diff(np.log(window[-21:])), ddof=1),
        )
    return output


def _independent_eligible(
    records: np.ndarray, query: Any, query_symbol_id: int,
) -> np.ndarray:
    tier_codes = {1, 2}
    return (
        (records["cutoff_ns"] <= query.latest_eligible_ns)
        & (records["episode_id"] != np.void(bytes.fromhex(query.episode_id)))
        & np.isin(records["quality_tier"], list(tier_codes))
        & ~(
            (records["symbol_id"] == query_symbol_id)
            & (records["cutoff_ns"] >= query.query_start_ns)
        )
    )


def _digest(domain: bytes, query_id: str, value: bytes) -> bytes:
    digest = sha256()
    digest.update(domain)
    digest.update(b"\0")
    digest.update(bytes.fromhex(query_id))
    digest.update(b"\0")
    digest.update(value)
    return digest.digest()


def _independent_random(
    records: np.ndarray, eligible: np.ndarray, symbols: tuple[str, ...], query_id: str,
) -> list[dict[str, Any]]:
    best: dict[int, tuple[bytes, bytes]] = {}
    for position in np.flatnonzero(eligible):
        episode_id = bytes(records["episode_id"][position])
        symbol_id = int(records["symbol_id"][position])
        key = _digest(b"wf03-random-episode-v1", query_id, episode_id)
        if symbol_id not in best or (key, episode_id) < best[symbol_id]:
            best[symbol_id] = (key, episode_id)
    ordered = sorted(best, key=lambda value: (
        _digest(b"wf03-random-symbol-v1", query_id, symbols[value].encode()),
        symbols[value],
    ))[:20]
    return [{
        "episode_id": best[symbol_id][1].hex(),
        "symbol": symbols[symbol_id],
        "distance_hex": None,
        "order_key": best[symbol_id][0].hex(),
    } for symbol_id in ordered]


def _independent_rank(
    records: np.ndarray, features: np.ndarray, eligible: np.ndarray,
    symbols: tuple[str, ...], query_features: np.ndarray, query_id: str,
) -> list[dict[str, Any]]:
    positions = [
        int(value) for value in np.flatnonzero(eligible)
        if np.isfinite(features[int(value)]).all()
    ]
    count = len(positions)
    query_bytes = bytes.fromhex(query_id)
    distance = {position: 0.0 for position in positions}
    for column in range(3):
        ordered: list[tuple[float, bytes, int | None]] = [
            (float(features[position, column]), bytes(records["episode_id"][position]), position)
            for position in positions
        ]
        ordered.append((float(query_features[column]), query_bytes, None))
        ordered.sort(key=lambda row: (row[0], row[1]))
        query_rank = next(rank / max(count, 1) for rank, row in enumerate(ordered)
                          if row[2] is None)
        for rank, row in enumerate(ordered):
            if row[2] is not None:
                distance[row[2]] += abs(rank / max(count, 1) - query_rank)
    ranked = sorted(positions, key=lambda position: (
        distance[position], bytes(records["episode_id"][position]),
    ))
    output: list[dict[str, Any]] = []
    seen: set[int] = set()
    for position in ranked:
        symbol_id = int(records["symbol_id"][position])
        if symbol_id in seen:
            continue
        seen.add(symbol_id)
        output.append({
            "episode_id": bytes(records["episode_id"][position]).hex(),
            "symbol": symbols[symbol_id],
            "distance_hex": distance[position].hex(),
            "order_key": "",
        })
        if len(output) == 20:
            break
    return output


def verify(repository: Path) -> dict[str, Any]:
    started = perf_counter()
    repository = repository.resolve(strict=True)
    root = repository / producer.OUTPUT_RELATIVE
    preregistration = base._read(repository / producer.PREREGISTRATION_RELATIVE)
    base._validate_seal(preregistration, "preregistration_digest")
    result = base._read(root / "RESULT.json")
    base._validate_seal(result)
    if result.get("passed") is not True \
            or result.get("selection_digest") != preregistration.get("selection_digest") \
            or result.get("outcomes_or_labels_used") is not False:
        raise BaselinePocVerificationError("producer baseline result differs")
    implementation = preregistration.get("implementation_commit")
    if type(implementation) is not str:
        raise BaselinePocVerificationError("producer implementation binding differs")
    for path, digest in preregistration["runtime_files"].items():
        blob = subprocess.run(
            ["git", "show", f"{implementation}:{path}"], cwd=repository,
            capture_output=True, check=False,
        )
        if blob.returncode or sha256(blob.stdout).hexdigest() != digest:
            raise BaselinePocVerificationError("frozen producer blob differs")
    resident = base._resident()
    loaded = load_packed_generation(
        Path(resident["store_root"]), base.GENERATION_ID,
        expected_provenance_digest=base.PROVENANCE_DIGEST,
        verify_content=False, validate_records=False,
    )
    _registry, by_id = base._registry(repository)
    source = None
    contexts = []
    for label, query_id, _symbol, _cutoff in base.PROBES:
        current_source, episode, _request, packed_query = base._context(
            repository, by_id[query_id],
        )
        source = source or current_source
        contexts.append((label, episode, packed_query))
    maximum = pd.Timestamp(loaded.manifest["provenance"]["benchmark_prefix"][
        "requested_cutoff"
    ])
    record_main: list[np.ndarray] = []
    record_overflow: list[np.ndarray] = []
    feature_main: list[np.ndarray] = []
    feature_overflow: list[np.ndarray] = []
    maximum_delta = 0.0
    non_bitwise = 0
    reconstructed = 0
    for specification in preregistration["selection"]:
        symbol = specification["symbol"]
        symbol_id = int(specification["symbol_id"])
        frame = source.load(InstrumentKey("nasdaq", symbol))
        prefix = causal_prefix_digest(frame, maximum)
        if prefix.__dict__ != specification["source_prefix"]:
            raise BaselinePocVerificationError("independent source prefix differs")
        frame = frame[frame.timestamp <= maximum].reset_index(drop=True)
        main = producer._slice(loaded.rows, symbol_id)
        overflow = producer._slice(loaded.overflow, symbol_id)
        rows_path, overflow_path, metadata_path = producer._paths(
            root / "shards", symbol,
        )
        metadata = base._read(metadata_path)
        base._validate_seal(metadata, "shard_digest")
        observed_main = np.fromfile(rows_path, dtype=FEATURE_DTYPE)["values"]
        observed_overflow = np.fromfile(overflow_path, dtype=FEATURE_DTYPE)["values"]
        expected_main = _independent_feature_rows(frame, main)
        expected_overflow = _independent_feature_rows(frame, overflow)
        for observed, expected in (
            (observed_main, expected_main), (observed_overflow, expected_overflow),
        ):
            finite = np.isfinite(observed) & np.isfinite(expected)
            if not np.array_equal(np.isnan(observed), np.isnan(expected)):
                raise BaselinePocVerificationError("independent feature missingness differs")
            if np.any(finite):
                delta = np.abs(observed[finite] - expected[finite])
                maximum_delta = max(maximum_delta, float(np.max(delta)))
                non_bitwise += int(np.sum(observed[finite].view(np.uint64)
                                          != expected[finite].view(np.uint64)))
            reconstructed += len(observed)
        record_main.append(producer._neighbor_records(main))
        record_overflow.append(producer._neighbor_records(overflow))
        feature_main.append(observed_main)
        feature_overflow.append(observed_overflow)
    records = np.concatenate((np.concatenate(record_main), np.concatenate(record_overflow)))
    features = np.concatenate((np.concatenate(feature_main), np.concatenate(feature_overflow)))
    independent_probes = []
    producer_by_id = {
        row["query_id"]: row for row in result["probe_results"]
    }
    for label, episode, query in contexts:
        query_symbol_id = loaded.symbols.index(episode.key.instrument.source_symbol)
        eligible = _independent_eligible(records, query, query_symbol_id)
        query_features = _independent_feature_rows(
            episode.bars,
            np.asarray([(int(episode.bars.timestamp.iloc[-1].value),)],
                       dtype=[("cutoff_ns", "<i8")]),
        )[0]
        random_rows = _independent_random(records, eligible, loaded.symbols, episode.key.id)
        rank_rows = _independent_rank(
            records, features, eligible, loaded.symbols, query_features, episode.key.id,
        )
        published = producer_by_id.get(episode.key.id)
        if published is None \
                or published["eligible_rows"] != int(eligible.sum()) \
                or published["query_features_hex"] != [value.hex() for value in query_features] \
                or published["random_neighbors"] != random_rows \
                or published["rank_neighbors"] != rank_rows:
            raise BaselinePocVerificationError(
                f"independent {label} baseline retrieval differs"
            )
        independent_probes.append({
            "label": label,
            "query_id": episode.key.id,
            "eligible_rows": int(eligible.sum()),
            "random_neighbor_digest": stable_hash(random_rows),
            "rank_neighbor_digest": stable_hash(rank_rows),
        })
    gates = {
        "producer_seal_valid": True,
        "frozen_runtime_blobs_valid": True,
        "source_prefixes_valid": True,
        "all_feature_rows_reconstructed": reconstructed == len(records),
        "maximum_feature_delta_at_most_1e_12": maximum_delta <= 1e-12,
        "independent_eligibility_equal": True,
        "independent_random_neighbors_equal": True,
        "independent_rank_neighbors_equal": True,
        "outcomes_or_labels_excluded": True,
    }
    if not all(gates.values()) or not math.isfinite(maximum_delta):
        raise BaselinePocVerificationError("independent baseline gates differ")
    state = {
        "schema_version": "m04r14-t14-10-wf03-baseline-poc-verification-v3",
        "status": "complete",
        "passed": True,
        "gates": gates,
        "producer_result_digest": result["result_digest"],
        "preregistration_digest": preregistration["preregistration_digest"],
        "rows_reconstructed": reconstructed,
        "non_bitwise_feature_values": non_bitwise,
        "maximum_feature_delta": maximum_delta,
        "independent_probes": independent_probes,
        "elapsed_seconds": perf_counter() - started,
        "outcomes_or_labels_used": False,
        "historical_walk_forward_query_outcomes_opened": False,
        "final_period_result_opened": False,
        "full_feature_store_build_authorized": True,
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
            raise BaselinePocVerificationError("baseline verification output exists")
        root.mkdir(parents=True)
        base._atomic(root / "VERIFIED.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
