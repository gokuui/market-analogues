"""Matched 12-core topology POC for the true WF-03 multichannel authority."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from hashlib import sha256
import json
import multiprocessing
import math
import os
from pathlib import Path
import resource
import subprocess
from time import perf_counter
from typing import Any, Mapping, Sequence

import numba
import pandas as pd

from market_analogues.adapters import source_from_spec
from market_analogues.certified_packed_search import (
    certified_packed_search, certified_packed_search_contract,
)
from market_analogues.config import load_config
from market_analogues.distance import DistanceConfig
from market_analogues.episodes import build_episode
from market_analogues.packed_bound_search import (
    PackedBoundQuery, packed_bound_search_contract,
    scan_packed_bound_proposals_many,
)
from market_analogues.representation import represent
from market_analogues.search import latest_eligible_cutoff
from market_analogues.types import InstrumentKey, SearchQuery, stable_hash

from experiments.m04r import m04r14_t14_10_wf03_feasibility as base


SCHEMA = "m04r14-t14-10-wf03-composite-topology-preregistration-v1"
OUTPUT_RELATIVE = Path(
    "config/data/analogues/m04r14/t14-10-wf03-composite-topology-poc-v1"
)
PREREGISTRATION_RELATIVE = Path(
    "experiments/m04r/"
    "m04r14_t14_10_wf03_composite_topology_poc_v1_preregistered.json"
)
AUTHORITY_RELATIVE = Path(
    "config/data/analogues/m04r14/t14-10-wf03-feasibility-v1/cases/"
    "000-early-a69def453340e01048a52284/attempts/attempt-0001/EXACT.json"
)
QUERY_IDS = (
    "cb0774b91343c225c486d973",
    "fb6669b28c515fcd8a38d6b6",
    "a69def453340e01048a52284",
    "14db3379bd20a4ad5d937186",
)
AUTHORITY_QUERY_ID = "a69def453340e01048a52284"
INITIAL_FRONTIER = 1_000
MAXIMUM_FRONTIER = 16_384
PROPOSAL_QUOTA = MAXIMUM_FRONTIER + 1
SEED_ROWS = 512
BLOCK_ROWS = 4_096
TOLERANCE = 1e-12
TOTAL_THREADS = 12
PARALLEL_GROUPS = 4
SEQUENTIAL_THREADS = TOTAL_THREADS
PARALLEL_THREADS = TOTAL_THREADS // PARALLEL_GROUPS
_REPOSITORY = Path(__file__).resolve().parents[2]
RUNTIME_FILES = (
    "config/datasets.example.yaml",
    "experiments/m04r/m04r14_t14_10_wf03_composite_topology_poc.py",
    "experiments/m04r/m04r14_t14_10_wf03_feasibility.py",
    "pyproject.toml",
) + tuple(
    str(path.relative_to(_REPOSITORY))
    for path in sorted((_REPOSITORY / "src/market_analogues").rglob("*.py"))
)


class CompositeTopologyError(RuntimeError):
    pass


POSITIVE_INFINITY_SENTINEL = {"__nonfinite_float__": "positive_infinity"}


def _certificate_json_value(value: Any) -> dict[str, Any]:
    """Encode only contract-valid intermediate +inf values as strict JSON."""
    certificate = asdict(value)
    for item in certificate.get("rounds", []):
        threshold = item.get("constrained_threshold")
        if type(threshold) is float and math.isinf(threshold) and threshold > 0:
            if item.get("selected_rows", base.TOP_K) >= base.TOP_K \
                    or item.get("certified") is not False:
                raise CompositeTopologyError("invalid infinite round threshold")
            item["constrained_threshold"] = dict(POSITIVE_INFINITY_SENTINEL)
    for item in certificate.get("threshold_closure_passes", []):
        upper = item.get("upper_inclusive")
        if type(upper) is float and math.isinf(upper) and upper > 0:
            if item.get("certified") is not True \
                    or item.get("selected_rows", 0) < base.TOP_K:
                raise CompositeTopologyError("invalid infinite closure threshold")
            item["upper_inclusive"] = dict(POSITIVE_INFINITY_SENTINEL)

    def reject_nonfinite(item: Any) -> None:
        if type(item) is float and not math.isfinite(item):
            raise CompositeTopologyError("unexpected non-finite certificate value")
        if type(item) is dict:
            for nested in item.values():
                reject_nonfinite(nested)
        elif type(item) in (list, tuple):
            for nested in item:
                reject_nonfinite(nested)

    reject_nonfinite(certificate)
    return certificate


def decode_certificate_json_value(value: Mapping[str, Any]) -> dict[str, Any]:
    """Restore explicit intermediate sentinels for digest reconstruction."""
    def decode(item: Any) -> Any:
        if item == POSITIVE_INFINITY_SENTINEL:
            return float("inf")
        if type(item) is dict:
            return {key: decode(nested) for key, nested in item.items()}
        if type(item) is list:
            return [decode(nested) for nested in item]
        return item

    return decode(dict(value))


def _git(repository: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repository, text=True,
        capture_output=True, check=False,
    )
    if result.returncode:
        raise CompositeTopologyError(result.stderr.strip() or "git command failed")
    return result.stdout.strip()


def _sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _swap_kib() -> int:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmSwap:"):
            return int(line.split()[1])
    raise CompositeTopologyError("process swap observation differs")


def _contract() -> dict[str, Any]:
    return certified_packed_search_contract(
        requested_positions=True, vector_lower_bounds=True,
        deferred_alignments=True, compact_scored=True,
        native_bound_deferral=True, streaming_threshold_closure=True,
        branch_aware_packed_bounds=True,
    )


def _inputs(repository: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    registry, by_id = base._registry(repository)
    try:
        rows = [by_id[query_id] for query_id in QUERY_IDS]
    except KeyError as exc:
        raise CompositeTopologyError("frozen topology query is absent") from exc
    if len({row["cutoff"] for row in rows}) != 1 \
            or any(row.get("scored") is not True for row in rows):
        raise CompositeTopologyError("topology queries are not one scored month")
    return registry, rows


def build_preregistration(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    if _git(repository, "status", "--porcelain"):
        raise CompositeTopologyError("topology preregistration requires clean commit")
    if (repository / OUTPUT_RELATIVE).exists() \
            or (repository / OUTPUT_RELATIVE).is_symlink():
        raise CompositeTopologyError("topology output must be absent before freeze")
    registry, rows = _inputs(repository)
    resident = base._resident()
    state = {
        "schema_version": SCHEMA,
        "status": "frozen_before_true_composite_topology_measurement",
        "implementation_commit": _git(repository, "rev-parse", "HEAD"),
        "runtime_files": {path: _sha(repository / path) for path in RUNTIME_FILES},
        "inputs": {
            "registry_digest": registry["registry_digest"],
            "query_ids_digest": stable_hash(list(QUERY_IDS)),
            "packed_generation_id": base.GENERATION_ID,
            "packed_provenance_digest": base.PROVENANCE_DIGEST,
            "resident_content_digest": resident["content_digest"],
            "authority_sha256": _sha(repository / AUTHORITY_RELATIVE),
        },
        "queries": rows,
        "contracts": {
            "proposal": packed_bound_search_contract(branch_aware=True),
            "authority": _contract(),
            "distance_weights": DistanceConfig().weights,
        },
        "execution": {
            "topology_order": ["sequential-p1t12", "parallel-p4t3"],
            "total_threads_each": TOTAL_THREADS,
            "parallel_groups": PARALLEL_GROUPS,
            "initial_frontier_rows": INITIAL_FRONTIER,
            "maximum_frontier_rows": MAXIMUM_FRONTIER,
            "proposal_quota": PROPOSAL_QUOTA,
            "seed_rows": SEED_ROWS,
            "block_rows": BLOCK_ROWS,
            "tolerance_hex": TOLERANCE.hex(),
            "output_root": str((repository / OUTPUT_RELATIVE).resolve()),
            "resume": "reuse only sealed topology receipts bound to this preregistration",
        },
        "gates": {
            "sequential_parallel_semantics_equal": True,
            "frozen_multichannel_authority_equal": True,
            "all_certificates_close": True,
            "zero_process_swap": True,
            "outcomes_or_labels_excluded": True,
        },
        "claims": {
            "historical_query_retrieval_opened": True,
            "historical_walk_forward_query_outcomes_opened": False,
            "final_period_result_opened": False,
            "production_promotion_authorized": False,
        },
    }
    return base._sealed(state, "preregistration_digest")


def validate_preregistration(
    repository: Path, value: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    base._validate_seal(value, "preregistration_digest")
    registry, rows = _inputs(repository)
    resident = base._resident()
    expected = value.get("inputs", {})
    execution = value.get("execution", {})
    if not all((
        value.get("schema_version") == SCHEMA,
        value.get("status") == "frozen_before_true_composite_topology_measurement",
        value.get("queries") == rows,
        value.get("contracts", {}).get("authority") == _contract(),
        value.get("contracts", {}).get("distance_weights")
            == DistanceConfig().weights,
        expected.get("registry_digest") == registry["registry_digest"],
        expected.get("query_ids_digest") == stable_hash(list(QUERY_IDS)),
        expected.get("packed_generation_id") == base.GENERATION_ID,
        expected.get("packed_provenance_digest") == base.PROVENANCE_DIGEST,
        expected.get("resident_content_digest") == resident["content_digest"],
        expected.get("authority_sha256") == _sha(repository / AUTHORITY_RELATIVE),
        set(value.get("runtime_files", {})) == set(RUNTIME_FILES),
        execution.get("topology_order")
            == ["sequential-p1t12", "parallel-p4t3"],
        execution.get("total_threads_each") == TOTAL_THREADS,
        execution.get("parallel_groups") == PARALLEL_GROUPS,
        execution.get("initial_frontier_rows") == INITIAL_FRONTIER,
        execution.get("maximum_frontier_rows") == MAXIMUM_FRONTIER,
        execution.get("proposal_quota") == PROPOSAL_QUOTA,
        execution.get("seed_rows") == SEED_ROWS,
        execution.get("block_rows") == BLOCK_ROWS,
        execution.get("tolerance_hex") == TOLERANCE.hex(),
        value.get("claims") == {
            "historical_query_retrieval_opened": True,
            "historical_walk_forward_query_outcomes_opened": False,
            "final_period_result_opened": False,
            "production_promotion_authorized": False,
        },
    )):
        raise CompositeTopologyError("topology preregistration differs")
    commit = value.get("implementation_commit")
    if type(commit) is not str:
        raise CompositeTopologyError("topology implementation commit differs")
    _git(repository, "merge-base", "--is-ancestor", commit, "HEAD")
    for path, digest in value.get("runtime_files", {}).items():
        blob = subprocess.run(
            ["git", "show", f"{commit}:{path}"], cwd=repository,
            capture_output=True, check=False,
        )
        if blob.returncode or sha256(blob.stdout).hexdigest() != digest \
                or _sha(repository / path) != digest:
            raise CompositeTopologyError(f"topology runtime differs: {path}")
    return registry, rows, resident


def _match(value: Any) -> dict[str, Any]:
    return {
        "episode_id": value.episode_key.id,
        "symbol": value.episode_key.instrument.source_symbol,
        "cutoff": value.episode_key.cutoff.isoformat(),
        "total_distance": value.total_distance,
        "component_distances": dict(value.component_distances),
        "alignment": [list(item) for item in value.alignment],
        "quality_tier": value.quality_tier,
    }


def _certificate_semantics(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != "elapsed_seconds"}


def _case_semantics(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "query_id": value["query_id"],
        "proposal_result_digest": value["proposal_result_digest"],
        "certificate": _certificate_semantics(value["certificate"]),
        "matches": value["matches"],
    }


def compare_topologies(
    sequential: Mapping[str, Any], parallel: Mapping[str, Any],
) -> bool:
    left = {row["query_id"]: _case_semantics(row) for row in sequential["cases"]}
    right = {row["query_id"]: _case_semantics(row) for row in parallel["cases"]}
    return left == right and set(left) == set(QUERY_IDS)


def _certificate_closed(value: Mapping[str, Any], matches: Sequence[Any]) -> bool:
    decoded = decode_certificate_json_value(value)
    accounting = decoded.get("native_bound_accounting", {})
    try:
        accounted = (
            accounting["exact_dtw_evaluated"]
            + accounting["native_bound_pruned"]
            + accounting["packed_bound_pruned"]
        )
        return all((
            decoded["schema_version"] == _contract()["schema_version"],
            decoded["eligible_candidates"] == accounted,
            decoded["exact_evaluated"] == accounting["exact_dtw_evaluated"],
            decoded["safely_pruned"]
                == accounting["native_bound_pruned"] + accounting["packed_bound_pruned"],
            decoded["maximum_quantized_bound_excess"] <= TOLERANCE,
            type(decoded["result_digest"]) is str and len(decoded["result_digest"]) == 64,
            len(matches) == base.TOP_K,
            len({row["symbol"] for row in matches}) == base.TOP_K,
        ))
    except (KeyError, TypeError):
        return False


def _run_group(
    repository_string: str, store_root_string: str,
    rows: tuple[dict[str, Any], ...], threads: int,
) -> dict[str, Any]:
    started = perf_counter()
    numba.set_num_threads(threads)
    repository = Path(repository_string)
    lease_before = base._resident()
    if lease_before["store_root"] != str(Path(store_root_string).resolve()):
        raise CompositeTopologyError("topology resident root differs")
    source = source_from_spec(
        load_config(repository / base.CONFIG_RELATIVE).datasets["nasdaq"]
    )
    contexts = []
    packed_queries = []
    for row in rows:
        episode = build_episode(
            source, InstrumentKey("nasdaq", row["symbol"]), row["cutoff"],
            int(row["lookback"]), row["representation_version"],
        )
        if episode.key.id != row["episode_id"]:
            raise CompositeTopologyError("topology query reconstruction differs")
        request = SearchQuery(
            episode.key, ("nasdaq",), ("A", "B"), base.TOP_K,
            False, True, base.MAX_PER_INSTRUMENT, base.MINIMUM_HISTORY_GAP,
        )
        packed_queries.append(PackedBoundQuery(
            episode.key.id, episode.key.instrument.source_symbol,
            int(pd.Timestamp(episode.bars.timestamp.iloc[0]).value),
            int(latest_eligible_cutoff(
                episode, base.MINIMUM_HISTORY_GAP,
            ).value),
            represent(episode), request.quality_tiers,
        ))
        contexts.append((row, episode, request))
    proposal = scan_packed_bound_proposals_many(
        Path(store_root_string), base.GENERATION_ID, packed_queries,
        route_quotas={"composite": PROPOSAL_QUOTA},
        block_rows=BLOCK_ROWS, branch_aware=True, verify_content=False,
        expected_provenance_digest=base.PROVENANCE_DIGEST,
    )
    cases = []
    for (row, episode, request), report in zip(
        contexts, proposal.reports, strict=True,
    ):
        exact_started = perf_counter()
        result = certified_packed_search(
            episode, source, request, Path(store_root_string), base.GENERATION_ID,
            store_dataset_id="nasdaq", initial_frontier_rows=INITIAL_FRONTIER,
            maximum_frontier_rows=MAXIMUM_FRONTIER, seed_rows=SEED_ROWS,
            block_rows=BLOCK_ROWS, workers=threads, tolerance=TOLERANCE,
            verify_content=False, requested_positions=True,
            vector_lower_bounds=True, deferred_alignments=True,
            compact_scored=True, native_bound_deferral=True,
            streaming_threshold_closure=True, branch_aware_packed_bounds=True,
            precomputed_proposal=report,
        )
        certificate = json.loads(json.dumps(
            _certificate_json_value(result.certificate), allow_nan=False,
        ))
        matches = json.loads(json.dumps([_match(item) for item in result.matches], allow_nan=False))
        semantic = {
            "query_id": row["episode_id"],
            "proposal_result_digest": report.result_digest,
            "certificate": _certificate_semantics(certificate),
            "matches": matches,
        }
        cases.append({
            **semantic, "certificate": certificate,
            "exact_seconds": perf_counter() - exact_started,
            "semantic_digest": stable_hash(semantic),
        })
    lease_after = base._resident()
    if lease_after["identity_digest"] != lease_before["identity_digest"]:
        raise CompositeTopologyError("topology resident identity drifted")
    return {
        "queries": len(rows), "threads": threads,
        "proposal_seconds": proposal.elapsed_seconds,
        "elapsed_seconds": perf_counter() - started,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "swap_kib": _swap_kib(),
        "resident_identity_digest": lease_before["identity_digest"],
        "cases": cases,
    }


def _run_fresh_groups(
    repository: Path, store_root: Path, rows: list[dict[str, Any]],
    *, groups: int, threads: int,
) -> dict[str, Any]:
    if len(rows) % groups:
        raise CompositeTopologyError("topology groups do not divide queries")
    grouped = tuple(
        tuple(rows[index::groups]) for index in range(groups)
    )
    started = perf_counter()
    results = []
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=groups, mp_context=context) as executor:
        futures = [executor.submit(
            _run_group, str(repository), str(store_root), group, threads,
        ) for group in grouped]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"[composite-topology] group complete queries={result['queries']} "
                f"threads={result['threads']} seconds={result['elapsed_seconds']:.2f}",
                flush=True,
            )
    by_id = {
        row["query_id"]: row for result in results for row in result["cases"]
    }
    ordered = [by_id[query_id] for query_id in QUERY_IDS]
    return {
        "schema_version": "m04r14-wf03-composite-topology-run-v1",
        "status": "complete", "groups": groups,
        "threads_per_group": threads,
        "total_threads": groups * threads,
        "wall_seconds": perf_counter() - started,
        "maximum_group_seconds": max(row["elapsed_seconds"] for row in results),
        "sum_group_peak_rss_mb": sum(row["peak_rss_mb"] for row in results),
        "maximum_group_peak_rss_mb": max(row["peak_rss_mb"] for row in results),
        "all_zero_swap": all(row["swap_kib"] == 0 for row in results),
        "resident_identity_digests": sorted({
            row["resident_identity_digest"] for row in results
        }),
        "group_measurements": results,
        "cases": ordered,
    }


def _valid_topology(
    value: Mapping[str, Any], preregistration_digest: str,
    *, expected_groups: int, expected_threads: int,
) -> bool:
    try:
        base._validate_seal(value)
        return all((
            value["status"] == "complete",
            value["preregistration_digest"] == preregistration_digest,
            value["groups"] == expected_groups,
            value["threads_per_group"] == expected_threads,
            value["total_threads"] == TOTAL_THREADS,
            value["all_zero_swap"] is True,
            value["outcomes_or_labels_used"] is False,
            value["historical_walk_forward_query_outcomes_opened"] is False,
            value["final_period_result_opened"] is False,
            [row["query_id"] for row in value["cases"]] == list(QUERY_IDS),
            all(row["semantic_digest"] == stable_hash(_case_semantics(row))
                for row in value["cases"]),
            all(_certificate_closed(row["certificate"], row["matches"])
                for row in value["cases"]),
        ))
    except (KeyError, TypeError, ValueError, base.FeasibilityError):
        return False


def execute(repository: Path, preregistration: Mapping[str, Any]) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    if _git(repository, "status", "--porcelain"):
        raise CompositeTopologyError("topology execution requires clean commit")
    _registry, rows, resident = validate_preregistration(repository, preregistration)
    root = repository / OUTPUT_RELATIVE
    if root.is_symlink():
        raise CompositeTopologyError("topology output root is symlinked")
    root.mkdir(parents=True, exist_ok=True)
    contract_path = root / "CONTRACT.json"
    if contract_path.exists():
        if base._read(contract_path) != preregistration:
            raise CompositeTopologyError("topology output contract differs")
    else:
        base._atomic(contract_path, preregistration)
    if (root / "RESULT.json").exists():
        result = base._read(root / "RESULT.json")
        base._validate_seal(result)
        if result.get("schema_version") \
                != "m04r14-t14-10-wf03-composite-topology-result-v1" \
                or result.get("status") != "complete" \
                or result.get("passed") is not True \
                or result.get("preregistration_digest") \
                != preregistration["preregistration_digest"]:
            raise CompositeTopologyError("topology terminal result differs")
        return result

    topology_specs = (
        ("SEQUENTIAL.json", 1, SEQUENTIAL_THREADS),
        ("PARALLEL.json", PARALLEL_GROUPS, PARALLEL_THREADS),
    )
    outputs = []
    for filename, groups, threads in topology_specs:
        print(
            f"[composite-topology] starting {filename} groups={groups} "
            f"threads-per-group={threads}", flush=True,
        )
        path = root / filename
        if path.exists():
            value = base._read(path)
            if not _valid_topology(
                value, preregistration["preregistration_digest"],
                expected_groups=groups, expected_threads=threads,
            ):
                raise CompositeTopologyError(f"invalid retained topology: {filename}")
        else:
            measurement = _run_fresh_groups(
                repository, Path(resident["store_root"]), rows,
                groups=groups, threads=threads,
            )
            value = base._sealed({
                **measurement,
                "preregistration_digest": preregistration["preregistration_digest"],
                "outcomes_or_labels_used": False,
                "historical_walk_forward_query_outcomes_opened": False,
                "final_period_result_opened": False,
            })
            base._atomic(path, value)
        outputs.append(value)

    sequential, parallel = outputs
    authority = base._read(repository / AUTHORITY_RELATIVE)
    current_authority = next(
        row for row in sequential["cases"] if row["query_id"] == AUTHORITY_QUERY_ID
    )
    topology_equal = compare_topologies(sequential, parallel)
    authority_equal = current_authority["matches"] == authority["matches"] \
        and current_authority["certificate"]["result_digest"] \
        == authority["certificate"]["result_digest"]
    gates = {
        "sequential_parallel_semantics_equal": topology_equal,
        "frozen_multichannel_authority_equal": authority_equal,
        "all_certificates_close": all(
            _certificate_closed(row["certificate"], row["matches"])
            for value in outputs for row in value["cases"]
        ),
        "zero_process_swap": all(value["all_zero_swap"] for value in outputs),
        "outcomes_or_labels_excluded": True,
    }
    state = {
        "schema_version": "m04r14-t14-10-wf03-composite-topology-result-v1",
        "status": "complete", "passed": all(gates.values()), "gates": gates,
        "preregistration_digest": preregistration["preregistration_digest"],
        "sequential_digest": sequential["result_digest"],
        "parallel_digest": parallel["result_digest"],
        "sequential_wall_seconds": sequential["wall_seconds"],
        "parallel_wall_seconds": parallel["wall_seconds"],
        "parallel_speedup": sequential["wall_seconds"] / parallel["wall_seconds"],
        "selected_topology": (
            "parallel-p4t3" if parallel["wall_seconds"] < sequential["wall_seconds"]
            else "sequential-p1t12"
        ),
        "outcomes_or_labels_used": False,
        "historical_walk_forward_query_outcomes_opened": False,
        "final_period_result_opened": False,
        "production_promotion_authorized": False,
    }
    result = base._sealed(state)
    base._atomic(root / "RESULT.json", result)
    if not result["passed"]:
        raise CompositeTopologyError("composite topology gates failed")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preregister", "run"))
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    repository = args.repository.resolve(strict=True)
    if args.action == "preregister":
        result = build_preregistration(repository)
        base._atomic(repository / PREREGISTRATION_RELATIVE, result)
    else:
        result = execute(
            repository, base._read(repository / PREREGISTRATION_RELATIVE),
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
