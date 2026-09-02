"""Independently verify all WF-03 baseline query receipts and a stratified oracle."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
import multiprocessing
from pathlib import Path
import subprocess
from typing import Any, Sequence
from time import perf_counter

import numpy as np

from market_analogues.adapters import source_from_spec
from market_analogues.baseline_feature_store import load_feature_generation
from market_analogues.baseline_neighbors import recent_return_volatility_neighbors
from market_analogues.config import load_config
from market_analogues.episodes import build_episode
from market_analogues.packed_bound_store import load_packed_generation
from market_analogues.search import latest_eligible_cutoff
from market_analogues.types import InstrumentKey, stable_hash

from experiments.m04r import m04r14_t14_10_wf03_baseline_batch as producer
from experiments.m04r import m04r14_t14_10_wf03_baseline_poc as bounded
from experiments.m04r import m04r14_t14_10_wf03_baseline_store_full as feature_store
from experiments.m04r import m04r14_t14_10_wf03_feasibility as base
from experiments.m04r import verify_m04r14_t14_10_wf03_baseline_poc as oracle


OUTPUT_RELATIVE = Path(
    "config/data/analogues/m04r14/t14-10-wf03-baseline-batch-v2-verification"
)
WORKERS = 8
SAMPLE_PER_FOLD = 4

_SOURCE: Any = None
_RECORDS: np.ndarray | None = None
_FEATURES: np.ndarray | None = None
_SYMBOLS: tuple[str, ...] = ()
_CASES_ROOT: Path | None = None
_GENERATION_ID = ""


class BaselineBatchVerificationError(RuntimeError):
    pass


def select_oracle_sample(rows: Sequence[dict[str, Any]]) -> list[str]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["fold_id"]), []).append(row)
    selected = []
    for fold in sorted(grouped):
        ordered = sorted(grouped[fold], key=lambda row: (
            stable_hash({
                "purpose": "wf03-baseline-batch-oracle-v1",
                "fold": fold,
                "query_id": row["episode_id"],
            }),
            row["episode_id"],
        ))
        if len(ordered) < SAMPLE_PER_FOLD:
            raise BaselineBatchVerificationError("baseline oracle fold is undersized")
        selected.extend(row["episode_id"] for row in ordered[:SAMPLE_PER_FOLD])
    return selected


def _lookup_positions(sorted_ids: np.ndarray, order: np.ndarray, ids: list[str]) -> np.ndarray:
    requested = np.asarray([np.void(bytes.fromhex(value)) for value in ids], dtype="V12")
    positions = np.searchsorted(sorted_ids, requested)
    if np.any(positions >= len(sorted_ids)) \
            or not np.array_equal(sorted_ids[positions], requested):
        raise BaselineBatchVerificationError("published neighbor episode is absent")
    return order[positions]


def _validate_case(
    value: dict[str, Any], row: dict[str, Any], records: np.ndarray,
    id_order: np.ndarray, sorted_ids: np.ndarray,
) -> dict[str, Any]:
    base._validate_seal(value, "case_digest")
    if not all((
        value.get("schema_version") == "m04r14-wf03-baseline-batch-case-v2",
        value.get("status") == "complete",
        value.get("query_id") == row["episode_id"],
        value.get("case_id") == row["case_id"],
        value.get("symbol") == row["symbol"],
        value.get("cutoff") == row["cutoff"],
        value.get("fold_id") == row["fold_id"],
        value.get("fold_role") == row["fold_role"],
        value.get("scored") is bool(row["scored"]),
        value.get("feature_generation_id") == _GENERATION_ID,
        value.get("outcomes_or_labels_used") is False,
        value.get("historical_walk_forward_query_outcomes_opened") is False,
        value.get("final_period_result_opened") is False,
    )):
        raise BaselineBatchVerificationError("baseline batch case binding differs")
    all_positions = []
    same_symbol_cutoffs: list[int] = []
    for method in ("random_neighbors", "rank_neighbors"):
        neighbors = value.get(method)
        if type(neighbors) is not list or len(neighbors) != 20 \
                or len({item.get("symbol") for item in neighbors}) != 20:
            raise BaselineBatchVerificationError("baseline neighbor inventory differs")
        positions = _lookup_positions(
            sorted_ids, id_order, [item["episode_id"] for item in neighbors],
        )
        if any(_SYMBOLS[int(records["symbol_id"][position])] != item["symbol"]
               for position, item in zip(positions, neighbors, strict=True)) \
                or np.any(records["cutoff_ns"][positions] > value["latest_eligible_ns"]):
            raise BaselineBatchVerificationError("published baseline eligibility differs")
        if method == "random_neighbors":
            for position, item in zip(positions, neighbors, strict=True):
                expected = oracle._digest(
                    b"wf03-random-episode-v1", value["query_id"],
                    bytes(records["episode_id"][position]),
                ).hex()
                if item.get("distance_hex") is not None or item.get("order_key") != expected:
                    raise BaselineBatchVerificationError("random neighbor key differs")
        else:
            try:
                distances = [float.fromhex(item["distance_hex"]) for item in neighbors]
            except (TypeError, ValueError) as exc:
                raise BaselineBatchVerificationError("rank distance differs") from exc
            if not np.isfinite(distances).all() or any(item.get("order_key") != ""
                                                       for item in neighbors):
                raise BaselineBatchVerificationError("rank neighbor value differs")
        all_positions.extend(int(position) for position in positions)
        same_symbol_cutoffs.extend(
            int(records["cutoff_ns"][position])
            for position, item in zip(positions, neighbors, strict=True)
            if item["symbol"] == row["symbol"]
        )
    return {
        "query_id": value["query_id"],
        "case_digest": value["case_digest"],
        "neighbor_position_digest": stable_hash(all_positions),
        "maximum_same_symbol_cutoff_ns": (
            max(same_symbol_cutoffs) if same_symbol_cutoffs else None
        ),
    }


def _oracle_query(row: dict[str, Any]) -> dict[str, Any]:
    if _SOURCE is None or _RECORDS is None or _FEATURES is None \
            or not _SYMBOLS or _CASES_ROOT is None:
        raise BaselineBatchVerificationError("baseline oracle worker is uninitialized")
    published = base._read(producer._case_path(_CASES_ROOT, row["episode_id"]))
    episode = build_episode(
        _SOURCE, InstrumentKey("nasdaq", row["symbol"]), row["cutoff"],
        int(row["lookback"]), row["representation_version"],
    )
    if episode.key.id != row["episode_id"]:
        raise BaselineBatchVerificationError("oracle query reconstruction differs")
    latest = latest_eligible_cutoff(episode, base.MINIMUM_HISTORY_GAP)
    query = type("Query", (), {
        "episode_id": episode.key.id,
        "query_start_ns": int(episode.bars.timestamp.iloc[0].value),
        "latest_eligible_ns": int(latest.value),
    })()
    symbol_id = producer._query_symbol_id(_SYMBOLS, row["symbol"])
    eligible = oracle._independent_eligible(
        _RECORDS, query, -1 if symbol_id is None else symbol_id,
    ) if symbol_id is not None else (
        (_RECORDS["cutoff_ns"] <= query.latest_eligible_ns)
        & (_RECORDS["episode_id"] != np.void(bytes.fromhex(query.episode_id)))
        & np.isin(_RECORDS["quality_tier"], [1, 2])
    )
    query_features = oracle._independent_feature_rows(
        episode.bars,
        np.asarray([(int(episode.bars.timestamp.iloc[-1].value),)],
                   dtype=[("cutoff_ns", "<i8")]),
    )[0]
    random_rows = oracle._independent_random(
        _RECORDS, eligible, _SYMBOLS, episode.key.id,
    )
    rank = recent_return_volatility_neighbors(
        _FEATURES, _RECORDS["episode_id"], _RECORDS["symbol_id"],
        eligible, _SYMBOLS, query_features, episode.key.id,
    )
    rank_rows = producer._neighbor_state(rank)
    if published["query_features_hex"] != [value.hex() for value in query_features] \
            or published["eligible_rows"] != int(eligible.sum()) \
            or published["random_neighbors"] != random_rows \
            or published["rank_neighbors"] != rank_rows:
        raise BaselineBatchVerificationError("baseline stratified oracle differs")
    return {
        "query_id": episode.key.id,
        "eligible_rows": int(eligible.sum()),
        "random_digest": stable_hash(random_rows),
        "rank_digest": stable_hash(rank_rows),
    }


def verify(repository: Path) -> dict[str, Any]:
    started = perf_counter()
    repository = repository.resolve(strict=True)
    if producer._git(repository, "status", "--porcelain"):
        raise BaselineBatchVerificationError("baseline batch verifier requires clean commit")
    verifier_commit = producer._git(repository, "rev-parse", "HEAD")
    verifier_relative = str(Path(__file__).resolve().relative_to(repository))
    verifier_sha256 = base._sha(repository / verifier_relative)
    blob = subprocess.run(
        ["git", "show", f"{verifier_commit}:{verifier_relative}"],
        cwd=repository, capture_output=True, check=False,
    )
    if blob.returncode or sha256(blob.stdout).hexdigest() != verifier_sha256:
        raise BaselineBatchVerificationError("baseline verifier Git binding differs")
    preregistration = base._read(repository / producer.PREREGISTRATION_RELATIVE)
    registry, _by_id, feature_result = producer.validate_preregistration(
        repository, preregistration,
    )
    root = repository / producer.OUTPUT_RELATIVE
    result = base._read(root / "RESULT.json")
    base._validate_seal(result)
    if result.get("passed") is not True \
            or result.get("queries") != len(registry["queries_data"]) \
            or result.get("independent_verification_authorized") is not True:
        raise BaselineBatchVerificationError("baseline producer terminal differs")
    resident = base._resident()
    packed = load_packed_generation(
        Path(resident["store_root"]), base.GENERATION_ID,
        expected_provenance_digest=base.PROVENANCE_DIGEST,
        verify_content=False, validate_records=False,
    )
    generation = load_feature_generation(
        repository / feature_store.OUTPUT_RELATIVE / "store",
        feature_result["generation_id"], packed_manifest=packed.manifest,
        verify_content=True,
    )
    records = np.concatenate((
        bounded._neighbor_records(packed.rows),
        bounded._neighbor_records(packed.overflow),
    ))
    features = np.concatenate((generation.rows, generation.overflow))["values"]
    id_order = np.argsort(records["episode_id"], kind="stable")
    sorted_ids = records["episode_id"][id_order]
    global _SOURCE, _RECORDS, _FEATURES, _SYMBOLS, _CASES_ROOT, _GENERATION_ID
    _SOURCE = source_from_spec(
        load_config(repository / base.CONFIG_RELATIVE).datasets["nasdaq"]
    )
    _RECORDS, _FEATURES, _SYMBOLS = records, features, packed.symbols
    _CASES_ROOT, _GENERATION_ID = root / "cases", generation.generation_id
    observations = []
    for row in registry["queries_data"]:
        path = producer._case_path(_CASES_ROOT, row["episode_id"])
        observations.append(_validate_case(
            base._read(path), row, records, id_order, sorted_ids,
        ))
    registry_by_id = {
        row["episode_id"]: dict(row) for row in registry["queries_data"]
    }
    same_symbol_cases = 0
    for observation in observations:
        maximum_cutoff = observation["maximum_same_symbol_cutoff_ns"]
        if maximum_cutoff is None:
            continue
        same_symbol_cases += 1
        row = registry_by_id[observation["query_id"]]
        episode = build_episode(
            _SOURCE, InstrumentKey("nasdaq", row["symbol"]), row["cutoff"],
            int(row["lookback"]), row["representation_version"],
        )
        if maximum_cutoff >= int(episode.bars.timestamp.iloc[0].value):
            raise BaselineBatchVerificationError("same-symbol overlap is not causal")
    sample_ids = select_oracle_sample(registry["queries_data"])
    with ProcessPoolExecutor(
        max_workers=WORKERS, mp_context=multiprocessing.get_context("fork"),
    ) as executor:
        oracle_results = list(executor.map(
            _oracle_query, [registry_by_id[value] for value in sample_ids], chunksize=1,
        ))
    gates = {
        "producer_and_preregistration_valid": True,
        "all_case_seals_and_bindings_valid": len(observations) == 3936,
        "all_157440_neighbor_references_valid": True,
        "all_neighbor_cutoffs_causally_eligible": True,
        "all_random_episode_order_keys_valid": True,
        "stratified_independent_random_equal": True,
        "stratified_exhaustive_rank_equal": True,
        "feature_generation_content_valid": True,
        "outcomes_or_labels_excluded": True,
    }
    if not all(gates.values()):
        raise BaselineBatchVerificationError("baseline batch verification gate differs")
    state = {
        "schema_version": "m04r14-t14-10-wf03-baseline-batch-verification-v1",
        "status": "complete", "passed": True, "gates": gates,
        "producer_result_digest": result["result_digest"],
        "preregistration_digest": preregistration["preregistration_digest"],
        "verifier_commit": verifier_commit,
        "verifier_runtime_sha256": verifier_sha256,
        "queries_verified": len(observations),
        "neighbor_references_verified": len(observations) * 40,
        "same_symbol_cases_verified": same_symbol_cases,
        "case_observation_digest": stable_hash(observations),
        "oracle_sample_ids": sample_ids,
        "oracle_sample_digest": stable_hash(oracle_results),
        "oracle_queries": len(oracle_results),
        "elapsed_seconds": perf_counter() - started,
        "outcomes_or_labels_used": False,
        "historical_walk_forward_query_outcomes_opened": False,
        "final_period_result_opened": False,
        "baseline_batch_complete": True,
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
            raise BaselineBatchVerificationError("baseline batch verification exists")
        root.mkdir(parents=True)
        base._atomic(root / "VERIFIED.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
