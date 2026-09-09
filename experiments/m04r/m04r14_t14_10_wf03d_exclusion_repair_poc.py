"""Preregister and run the bounded real WF-03D top-21 exclusion repair POC."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import resource
import subprocess
from time import perf_counter
from typing import Any, Mapping, Sequence

import numba
import numpy as np

from market_analogues.adapters import CachedOHLCVSource, source_from_spec
from market_analogues.baseline_feature_store import load_feature_generation
from market_analogues.baseline_neighbors import (
    build_baseline_rank_index, deterministic_random_neighbors,
    indexed_recent_return_volatility_neighbors, recent_return_volatility,
)
from market_analogues.config import load_config
from market_analogues.dtw_component_search import certified_staged_dtw_component_search
from market_analogues.episodes import build_episode
from market_analogues.packed_bound_search import PackedBoundQuery, _eligible_mask
from market_analogues.packed_bound_store import load_packed_generation
from market_analogues.search import latest_eligible_cutoff
from market_analogues.subset_selection import certified_prefix_without_query_symbol
from market_analogues.types import InstrumentKey, SearchQuery, stable_hash
from market_analogues.certified_packed_search import certified_packed_search

from experiments.m04r import m04r14_t14_10_wf03_baseline_batch as baseline_batch
from experiments.m04r import m04r14_t14_10_wf03_baseline_poc as baseline_poc
from experiments.m04r import m04r14_t14_10_wf03_baseline_store_full as feature_store
from experiments.m04r import m04r14_t14_10_wf03_combined_batch as price_batch
from experiments.m04r import m04r14_t14_10_wf03_composite_topology_poc as composite_kernel
from experiments.m04r import m04r14_t14_10_wf03_feasibility as base
from experiments.m04r import m04r14_t14_10_wf03b_dtw_component_ladder as dtw_ladder
from experiments.m04r import m04r14_t14_10_wf03d_exclusion_audit as audit_producer


SCHEMA = "m04r14-t14-10-wf03d-exclusion-repair-poc-preregistration-v1"
OUTPUT_RELATIVE = Path(
    "config/data/analogues/m04r14/t14-10-wf03d-exclusion-repair-poc-v1"
)
PREREGISTRATION_RELATIVE = Path(
    "experiments/m04r/m04r14_t14_10_wf03d_exclusion_repair_poc_v1_preregistered.json"
)
AUDIT_VERIFICATION = Path(
    "config/data/analogues/m04r14/"
    "t14-10-wf03d-exclusion-audit-v1-verification/VERIFIED.json"
)
FEATURE_RESULT = feature_store.OUTPUT_RELATIVE / "RESULT.json"
FEATURE_VERIFICATION = baseline_batch.FEATURE_VERIFICATION_RELATIVE
DTW_RESULT = dtw_ladder.DTW_RESULT_RELATIVE
DTW_VERIFICATION = dtw_ladder.DTW_VERIFICATION_RELATIVE
QUERY_IDS = (
    "023e7cecc419456895dd4f9e",  # composite, price, return/volatility
    "4b5bfd2d1a2adc26eacd43e0",  # composite, deterministic random
)
EXPECTED_AFFECTED = {
    QUERY_IDS[0]: ("composite", "price_only", "recent_return_volatility"),
    QUERY_IDS[1]: ("composite", "deterministic_random"),
}
TOP_K = 20
SUPERSET_K = TOP_K + 1
THREADS = 12
INITIAL_FRONTIER = 1_000
MAXIMUM_FRONTIER = 16_384
SEED_ROWS = 512
PRICE_SEED_ROWS = 2_048
BLOCK_ROWS = 4_096
TOLERANCE = 1e-12
RUNTIME_FILES = (
    "config/datasets.example.yaml",
    "experiments/m04r/m04r14_t14_10_wf03d_exclusion_repair_poc.py",
    "experiments/m04r/m04r14_t14_10_wf03d_exclusion_audit.py",
    "experiments/m04r/m04r14_t14_10_wf03_composite_topology_poc.py",
    "experiments/m04r/m04r14_t14_10_wf03_combined_batch.py",
    "experiments/m04r/m04r14_t14_10_wf03_baseline_batch.py",
    "experiments/m04r/m04r14_t14_10_wf03_baseline_poc.py",
    "experiments/m04r/m04r14_t14_10_wf03_baseline_store_full.py",
    "experiments/m04r/m04r14_t14_10_wf03_feasibility.py",
    "experiments/m04r/m04r14_t14_10_wf03b_dtw_component_ladder.py",
    "pyproject.toml",
) + tuple(
    str(path.relative_to(Path(__file__).resolve().parents[2]))
    for path in sorted(
        (Path(__file__).resolve().parents[2] / "src/market_analogues").glob("*.py")
    )
)


class ExclusionRepairPocError(RuntimeError):
    pass


def _git(repository: Path, *args: str, raw: bool = False) -> str | bytes:
    result = subprocess.run(
        ["git", *args], cwd=repository, capture_output=True,
        text=not raw, check=False,
    )
    if result.returncode:
        message = result.stderr if not raw else result.stderr.decode(errors="replace")
        raise ExclusionRepairPocError(message.strip() or "git command failed")
    return result.stdout if raw else result.stdout.strip()


def _sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _affected_for_query(audit: Mapping[str, Any], query_id: str) -> tuple[str, ...]:
    return tuple(
        method for method in (
            "composite", "price_only", "deterministic_random",
            "recent_return_volatility",
        )
        if query_id in audit["methods"][method]["affected_query_ids"]
    )


def build_preregistration(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    if _git(repository, "status", "--porcelain"):
        raise ExclusionRepairPocError("repair POC preregistration requires clean commit")
    if (repository / OUTPUT_RELATIVE).exists() \
            or (repository / PREREGISTRATION_RELATIVE).exists():
        raise ExclusionRepairPocError("repair POC paths must be absent before freeze")
    audit = base._read(repository / audit_producer.OUTPUT_RELATIVE / "AUDIT.json")
    base._validate_seal(audit, "audit_digest")
    verified = base._read(repository / AUDIT_VERIFICATION)
    base._validate_seal(verified, "verification_digest")
    if verified.get("passed") is not True \
            or verified.get("audit_digest") != audit["audit_digest"]:
        raise ExclusionRepairPocError("verified exclusion audit differs")
    registry, by_id = base._registry(repository)
    rows = [by_id[value] for value in QUERY_IDS]
    actual = {value: _affected_for_query(audit, value) for value in QUERY_IDS}
    if actual != EXPECTED_AFFECTED \
            or set().union(*(set(value) for value in actual.values())) != {
                "composite", "price_only", "deterministic_random",
                "recent_return_volatility",
            }:
        raise ExclusionRepairPocError("repair POC method coverage differs")
    resident = base._resident()
    feature_result = base._read(repository / FEATURE_RESULT)
    base._validate_seal(feature_result)
    feature_verified = base._read(repository / FEATURE_VERIFICATION)
    base._validate_seal(feature_verified, "verification_digest")
    if feature_result.get("passed") is not True \
            or feature_verified.get("passed") is not True \
            or feature_verified.get("producer_result_digest") != feature_result["result_digest"] \
            or feature_verified.get("generation_id") != feature_result["generation_id"]:
        raise ExclusionRepairPocError("verified feature generation differs")
    dtw_result = base._read(repository / DTW_RESULT)
    base._validate_seal(dtw_result)
    dtw_verified = base._read(repository / DTW_VERIFICATION)
    base._validate_seal(dtw_verified, "verification_digest")
    if dtw_result.get("passed") is not True \
            or dtw_verified.get("passed") is not True \
            or dtw_verified.get("producer_result_digest") != dtw_result["result_digest"] \
            or dtw_verified.get("generation_id") != dtw_result["generation_id"]:
        raise ExclusionRepairPocError("verified DTW generation differs")
    dtw_identity = price_batch._dtw_physical_identity(repository)
    source_cases = {}
    for row in rows:
        query_id = row["episode_id"]
        source_cases[query_id] = {
            "composite": _sha(repository / audit_producer.COMPOSITE_ROOT / "cases" / f"{query_id}.json"),
            "price_only": _sha(repository / audit_producer.PRICE_ROOT / "cases" / f"{query_id}.json"),
            "baselines": _sha(repository / audit_producer.BASELINE_ROOT / "cases" / f"{query_id}.json"),
        }
    head = str(_git(repository, "rev-parse", "HEAD"))
    state = {
        "schema_version": SCHEMA,
        "status": "frozen_before_real_top21_exclusion_repair_poc",
        "implementation_commit": head,
        "runtime_files": {path: _sha(repository / path) for path in RUNTIME_FILES},
        "inputs": {
            "registry_digest": registry["registry_digest"],
            "audit_digest": audit["audit_digest"],
            "audit_verification_digest": verified["verification_digest"],
            "audit_verification_sha256": _sha(repository / AUDIT_VERIFICATION),
            "resident_content_digest": resident["content_digest"],
            "packed_generation_id": base.GENERATION_ID,
            "dtw_generation_id": dtw_ladder.DTW_GENERATION_ID,
            "dtw_result_digest": dtw_result["result_digest"],
            "dtw_result_sha256": _sha(repository / DTW_RESULT),
            "dtw_verification_digest": dtw_verified["verification_digest"],
            "dtw_verification_sha256": _sha(repository / DTW_VERIFICATION),
            "dtw_physical_identity_digest": dtw_identity["digest"],
            "feature_generation_id": feature_result["generation_id"],
            "feature_result_digest": feature_result["result_digest"],
            "feature_result_sha256": _sha(repository / FEATURE_RESULT),
            "feature_verification_digest": feature_verified["verification_digest"],
            "feature_verification_sha256": _sha(repository / FEATURE_VERIFICATION),
            "source_case_sha256": source_cases,
        },
        "queries": rows,
        "expected_affected_methods": {
            key: list(value) for key, value in EXPECTED_AFFECTED.items()
        },
        "execution": {
            "top_k": TOP_K, "certified_superset_k": SUPERSET_K,
            "threads": THREADS, "initial_frontier": INITIAL_FRONTIER,
            "maximum_frontier": MAXIMUM_FRONTIER, "seed_rows": SEED_ROWS,
            "price_seed_rows": PRICE_SEED_ROWS, "block_rows": BLOCK_ROWS,
            "tolerance_hex": TOLERANCE.hex(),
            "output_root": str((repository / OUTPUT_RELATIVE).resolve()),
        },
        "claims": {
            "outcomes_or_labels_used": False,
            "historical_walk_forward_query_outcomes_opened": False,
            "final_period_result_opened": False,
            "production_promotion_authorized": False,
        },
    }
    return base._sealed(state, "preregistration_digest")


def _sole_child(repository: Path, preregistration: Mapping[str, Any]) -> str:
    h0 = str(preregistration["implementation_commit"])
    raw = (repository / PREREGISTRATION_RELATIVE).read_bytes()
    accepted = []
    for line in str(_git(repository, "rev-list", "--all", "--children")).splitlines():
        values = line.split()
        if values and values[0] == h0:
            for child in values[1:]:
                changed = str(_git(
                    repository, "diff-tree", "--no-commit-id", "--name-only", "-r", child,
                )).splitlines()
                parents = str(_git(repository, "rev-list", "--parents", "-n", "1", child)).split()
                if parents == [child, h0] and changed == [PREREGISTRATION_RELATIVE.as_posix()] \
                        and _git(repository, "show", f"{child}:{PREREGISTRATION_RELATIVE}", raw=True) == raw:
                    accepted.append(child)
    if len(set(accepted)) != 1:
        raise ExclusionRepairPocError("repair POC requires one preregistration-only child")
    return accepted[0]


def validate_preregistration(repository: Path, value: Mapping[str, Any]) -> list[dict[str, Any]]:
    if _git(repository, "status", "--porcelain"):
        raise ExclusionRepairPocError("repair POC execution requires clean commit")
    base._validate_seal(value, "preregistration_digest")
    if value.get("schema_version") != SCHEMA \
            or value.get("status") != "frozen_before_real_top21_exclusion_repair_poc" \
            or value.get("expected_affected_methods") != {
                key: list(methods) for key, methods in EXPECTED_AFFECTED.items()
            } \
            or value.get("execution") != {
                "top_k": TOP_K, "certified_superset_k": SUPERSET_K,
                "threads": THREADS, "initial_frontier": INITIAL_FRONTIER,
                "maximum_frontier": MAXIMUM_FRONTIER, "seed_rows": SEED_ROWS,
                "price_seed_rows": PRICE_SEED_ROWS, "block_rows": BLOCK_ROWS,
                "tolerance_hex": TOLERANCE.hex(),
                "output_root": str((repository / OUTPUT_RELATIVE).resolve()),
            } \
            or value.get("claims") != {
                "outcomes_or_labels_used": False,
                "historical_walk_forward_query_outcomes_opened": False,
                "final_period_result_opened": False,
                "production_promotion_authorized": False,
            }:
        raise ExclusionRepairPocError("repair POC frozen contract differs")
    child = _sole_child(repository, value)
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", child, "HEAD"], cwd=repository,
    ).returncode:
        raise ExclusionRepairPocError("repair POC lineage differs")
    for path, digest in value["runtime_files"].items():
        blob = _git(repository, "show", f"{value['implementation_commit']}:{path}", raw=True)
        if sha256(blob).hexdigest() != digest or _sha(repository / path) != digest:
            raise ExclusionRepairPocError(f"repair POC runtime differs: {path}")
    verified = base._read(repository / AUDIT_VERIFICATION)
    base._validate_seal(verified, "verification_digest")
    if verified.get("verification_digest") != value["inputs"]["audit_verification_digest"]:
        raise ExclusionRepairPocError("repair POC audit verification differs")
    if _sha(repository / AUDIT_VERIFICATION) != value["inputs"]["audit_verification_sha256"]:
        raise ExclusionRepairPocError("repair POC audit verification bytes differ")
    feature_result = base._read(repository / FEATURE_RESULT)
    feature_verified = base._read(repository / FEATURE_VERIFICATION)
    dtw_result = base._read(repository / DTW_RESULT)
    dtw_verified = base._read(repository / DTW_VERIFICATION)
    base._validate_seal(feature_result)
    base._validate_seal(feature_verified, "verification_digest")
    base._validate_seal(dtw_result)
    base._validate_seal(dtw_verified, "verification_digest")
    inputs = value["inputs"]
    observed_bindings = {
        "feature_generation_id": feature_result.get("generation_id"),
        "feature_result_digest": feature_result.get("result_digest"),
        "feature_result_sha256": _sha(repository / FEATURE_RESULT),
        "feature_verification_digest": feature_verified.get("verification_digest"),
        "feature_verification_sha256": _sha(repository / FEATURE_VERIFICATION),
        "dtw_generation_id": dtw_result.get("generation_id"),
        "dtw_result_digest": dtw_result.get("result_digest"),
        "dtw_result_sha256": _sha(repository / DTW_RESULT),
        "dtw_verification_digest": dtw_verified.get("verification_digest"),
        "dtw_verification_sha256": _sha(repository / DTW_VERIFICATION),
        "dtw_physical_identity_digest": price_batch._dtw_physical_identity(repository)["digest"],
    }
    if any(inputs.get(key) != observed for key, observed in observed_bindings.items()) \
            or feature_verified.get("passed") is not True \
            or feature_verified.get("producer_result_digest") != feature_result.get("result_digest") \
            or dtw_verified.get("passed") is not True \
            or dtw_verified.get("producer_result_digest") != dtw_result.get("result_digest"):
        raise ExclusionRepairPocError("repair POC verified substrate differs")
    registry, by_id = base._registry(repository)
    rows = [by_id[query_id] for query_id in QUERY_IDS]
    if value.get("queries") != rows or value.get("inputs", {}).get("registry_digest") != registry["registry_digest"]:
        raise ExclusionRepairPocError("repair POC frozen queries differ")
    for row in rows:
        query_id = row["episode_id"]
        paths = {
            "composite": repository / audit_producer.COMPOSITE_ROOT / "cases" / f"{query_id}.json",
            "price_only": repository / audit_producer.PRICE_ROOT / "cases" / f"{query_id}.json",
            "baselines": repository / audit_producer.BASELINE_ROOT / "cases" / f"{query_id}.json",
        }
        if any(_sha(path) != inputs["source_case_sha256"][query_id][method]
               for method, path in paths.items()):
            raise ExclusionRepairPocError("repair POC source case bytes differ")
    return rows


def _proof(rows: list[dict[str, Any]], query_symbol: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected, proof = certified_prefix_without_query_symbol(
        rows, query_symbol, top_k=TOP_K,
    )
    return list(selected), asdict(proof)


def _query_episode(source: Any, row: Mapping[str, Any]) -> Any:
    episode = build_episode(
        source, InstrumentKey("nasdaq", str(row["symbol"])), str(row["cutoff"]),
        int(row["lookback"]), str(row["representation_version"]),
    )
    if episode.key.id != row["episode_id"]:
        raise ExclusionRepairPocError("repair POC query reconstruction differs")
    return episode


def _request(episode: Any) -> SearchQuery:
    return SearchQuery(
        episode.key, ("nasdaq",), ("A", "B"), SUPERSET_K,
        False, True, base.MAX_PER_INSTRUMENT, base.MINIMUM_HISTORY_GAP,
    )


def _composite(
    row: Mapping[str, Any], source: Any, packed_root: Path,
) -> dict[str, Any]:
    started = perf_counter(); episode = _query_episode(source, row)
    result = certified_packed_search(
        episode, source, _request(episode), packed_root, base.GENERATION_ID,
        store_dataset_id="nasdaq", initial_frontier_rows=INITIAL_FRONTIER,
        maximum_frontier_rows=MAXIMUM_FRONTIER, seed_rows=SEED_ROWS,
        block_rows=BLOCK_ROWS, workers=THREADS, tolerance=TOLERANCE,
        verify_content=False, requested_positions=True,
        vector_lower_bounds=True, deferred_alignments=True,
        compact_scored=True, native_bound_deferral=True,
        streaming_threshold_closure=True, branch_aware_packed_bounds=True,
        proposal_threads=THREADS,
    )
    matches = [composite_kernel._match(value) for value in result.matches]
    old = base._read(
        row["_repository"] / audit_producer.COMPOSITE_ROOT / "cases" /
        f"{row['episode_id']}.json"
    )["retrieval"]["matches"]
    if matches[:TOP_K] != old:
        raise ExclusionRepairPocError("composite top-21 prefix differs from sealed top-20")
    selected, proof = _proof(matches, str(row["symbol"]))
    return {
        "method": "composite", "superset_matches": matches,
        "corrected_matches": selected, "subset_proof": proof,
        "certificate": composite_kernel._certificate_json_value(result.certificate),
        "elapsed_seconds": perf_counter() - started,
    }


def _price(
    row: Mapping[str, Any], source: Any, packed_root: Path,
    prepared: dict[str, Any], repository: Path,
) -> dict[str, Any]:
    started = perf_counter(); episode = _query_episode(source, row)
    result = certified_staged_dtw_component_search(
        episode, source, _request(episode), packed_root, base.GENERATION_ID,
        repository / dtw_ladder.DTW_ROOT_RELATIVE, dtw_ladder.DTW_GENERATION_ID,
        store_dataset_id="nasdaq", seed_rows=PRICE_SEED_ROWS,
        block_rows=BLOCK_ROWS, rigid_threads=THREADS, dtw_threads=THREADS,
        exact_workers=THREADS, tolerance=TOLERANCE, verify_content=False,
        prepared_symbol_cache=prepared, adaptive_seed=True,
    )
    matches = price_batch._matches(result)
    old = base._read(
        repository / audit_producer.PRICE_ROOT / "cases" /
        f"{row['episode_id']}.json"
    )["matches"]
    if matches[:TOP_K] != old:
        raise ExclusionRepairPocError("price top-21 prefix differs from sealed top-20")
    selected, proof = _proof(matches, str(row["symbol"]))
    return {
        "method": "price_only", "superset_matches": matches,
        "corrected_matches": selected, "subset_proof": proof,
        "certificate": asdict(result.certificate),
        "elapsed_seconds": perf_counter() - started,
    }


def execute(repository: Path, preregistration: Mapping[str, Any]) -> dict[str, Any]:
    started = perf_counter(); repository = repository.resolve(strict=True)
    rows = validate_preregistration(repository, preregistration)
    root = repository / OUTPUT_RELATIVE
    if root.exists() or root.is_symlink():
        raise ExclusionRepairPocError("repair POC output already exists")
    resident = base._resident()
    if resident["content_digest"] != preregistration["inputs"]["resident_content_digest"]:
        raise ExclusionRepairPocError("repair POC resident content differs")
    numba.set_num_threads(THREADS)
    packed_root = Path(resident["store_root"])
    packed = load_packed_generation(
        packed_root, base.GENERATION_ID, verify_content=False,
        validate_records=False, expected_provenance_digest=base.PROVENANCE_DIGEST,
    )
    feature_result = base._read(repository / feature_store.OUTPUT_RELATIVE / "RESULT.json")
    features_loaded = load_feature_generation(
        repository / feature_store.OUTPUT_RELATIVE / "store",
        feature_result["generation_id"], packed_manifest=packed.manifest,
        verify_content=True,
    )
    records = np.concatenate((
        baseline_poc._neighbor_records(packed.rows),
        baseline_poc._neighbor_records(packed.overflow),
    ))
    feature_values = np.concatenate((
        features_loaded.rows, features_loaded.overflow,
    ))["values"]
    rank_index = build_baseline_rank_index(feature_values, records["episode_id"])
    raw_source = source_from_spec(
        load_config(repository / base.CONFIG_RELATIVE).datasets["nasdaq"]
    )
    source = CachedOHLCVSource(raw_source, max_entries=None)
    prepared: dict[str, Any] = {}
    cases = []
    for raw_row in rows:
        row = {**raw_row, "_repository": repository}
        methods = []
        affected = EXPECTED_AFFECTED[row["episode_id"]]
        if "composite" in affected:
            methods.append(_composite(row, source, packed_root))
        if "price_only" in affected:
            methods.append(_price(row, source, packed_root, prepared, repository))
        episode = _query_episode(source, row)
        packed_query = PackedBoundQuery(
            episode.key.id, episode.key.instrument.source_symbol,
            int(episode.bars.timestamp.iloc[0].value),
            int(latest_eligible_cutoff(episode, base.MINIMUM_HISTORY_GAP).value),
            composite_kernel.represent(episode), ("A", "B"),
        )
        symbol_id = packed.symbols.index(row["symbol"]) if row["symbol"] in packed.symbols else None
        eligible = _eligible_mask(records, packed_query, symbol_id)
        old_baseline = base._read(
            repository / audit_producer.BASELINE_ROOT / "cases" /
            f"{row['episode_id']}.json"
        )
        if "deterministic_random" in affected:
            values = baseline_batch._neighbor_state(deterministic_random_neighbors(
                records["episode_id"], records["symbol_id"], eligible,
                packed.symbols, row["episode_id"], top_k=SUPERSET_K,
            ))
            if values[:TOP_K] != old_baseline["random_neighbors"]:
                raise ExclusionRepairPocError("random top-21 prefix differs")
            selected, proof = _proof(values, row["symbol"])
            methods.append({"method": "deterministic_random", "superset_matches": values,
                            "corrected_matches": selected, "subset_proof": proof})
        if "recent_return_volatility" in affected:
            query_features = recent_return_volatility(
                episode.bars["close"].to_numpy(dtype=np.float64)
            )
            values = baseline_batch._neighbor_state(
                indexed_recent_return_volatility_neighbors(
                    rank_index, feature_values, records["episode_id"],
                    records["symbol_id"], eligible, packed.symbols,
                    query_features, row["episode_id"], top_k=SUPERSET_K,
                )
            )
            if values[:TOP_K] != old_baseline["rank_neighbors"]:
                raise ExclusionRepairPocError("return/volatility top-21 prefix differs")
            selected, proof = _proof(values, row["symbol"])
            methods.append({"method": "recent_return_volatility", "superset_matches": values,
                            "corrected_matches": selected, "subset_proof": proof})
        if tuple(value["method"] for value in methods) != affected \
                or any(len(value["corrected_matches"]) != TOP_K for value in methods) \
                or any(any(match["symbol"] == row["symbol"] for match in value["corrected_matches"])
                       for value in methods):
            raise ExclusionRepairPocError("repair POC corrected method result differs")
        cases.append({
            "query_id": row["episode_id"], "symbol": row["symbol"],
            "cutoff": row["cutoff"], "methods": methods,
        })
    state = {
        "schema_version": "m04r14-t14-10-wf03d-exclusion-repair-poc-result-v1",
        "status": "complete", "passed": True,
        "preregistration_digest": preregistration["preregistration_digest"],
        "queries": cases,
        "query_count": len(cases),
        "method_runs": sum(len(value["methods"]) for value in cases),
        "all_original_top20_prefixes_exact": True,
        "all_corrected_top20_exclude_query_symbol": True,
        "elapsed_seconds": perf_counter() - started,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "process_swap_kib": composite_kernel._swap_kib(),
        "outcomes_or_labels_used": False,
        "historical_walk_forward_query_outcomes_opened": False,
        "final_period_result_opened": False,
        "production_promotion_authorized": False,
        "independent_verification_authorized": True,
    }
    if price_batch._dtw_physical_identity(repository)["digest"] \
            != preregistration["inputs"]["dtw_physical_identity_digest"]:
        raise ExclusionRepairPocError("repair POC DTW substrate changed during execution")
    sealed = base._sealed(state)
    root.mkdir(parents=True)
    base._atomic(root / "CONTRACT.json", preregistration)
    base._atomic(root / "RESULT.json", sealed)
    return sealed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preregister", "run"))
    parser.add_argument("--repository", required=True, type=Path)
    args = parser.parse_args(argv)
    repository = args.repository.resolve(strict=True)
    path = repository / PREREGISTRATION_RELATIVE
    if args.mode == "preregister":
        base._atomic(path, build_preregistration(repository)); return 0
    result = execute(repository, base._read(path))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
