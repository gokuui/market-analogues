"""Compare p4t3 and p12t1 true-composite width on twelve frozen queries."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from hashlib import sha256
import json
import multiprocessing
from pathlib import Path
import subprocess
from time import perf_counter
from typing import Any, Mapping, Sequence

from market_analogues.types import stable_hash

from experiments.m04r import m04r14_t14_10_wf03_composite_topology_poc as topology
from experiments.m04r import m04r14_t14_10_wf03_feasibility as base


SCHEMA = "m04r14-t14-10-wf03-composite-width-preregistration-v1"
OUTPUT_RELATIVE = Path(
    "config/data/analogues/m04r14/t14-10-wf03-composite-width-poc-v1"
)
PREREGISTRATION_RELATIVE = Path(
    "experiments/m04r/m04r14_t14_10_wf03_composite_width_poc_v1_preregistered.json"
)
TOPOLOGY_VERIFICATION_RELATIVE = Path(
    "config/data/analogues/m04r14/"
    "t14-10-wf03-composite-topology-poc-v1-verification/VERIFIED.json"
)
WIDTH = 12
MONTH = "2014-03-31T00:00:00"
TOPOLOGIES = (("P4T3.json", 4, 3), ("P12T1.json", 12, 1))
_REPOSITORY = Path(__file__).resolve().parents[2]
RUNTIME_FILES = (
    "config/datasets.example.yaml",
    "experiments/m04r/m04r14_t14_10_wf03_composite_width_poc.py",
    "experiments/m04r/m04r14_t14_10_wf03_composite_topology_poc.py",
    "experiments/m04r/m04r14_t14_10_wf03_feasibility.py",
    "pyproject.toml",
) + tuple(
    str(path.relative_to(_REPOSITORY))
    for path in sorted((_REPOSITORY / "src/market_analogues").rglob("*.py"))
)


class CompositeWidthError(RuntimeError):
    pass


def _git(repository: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repository, text=True,
        capture_output=True, check=False,
    )
    if result.returncode:
        raise CompositeWidthError(result.stderr.strip() or "git command failed")
    return result.stdout.strip()


def _sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _inputs(repository: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    registry, _by_id = base._registry(repository)
    month = [row for row in registry["queries_data"] if row["cutoff"] == MONTH]
    rows = sorted(month, key=lambda row: row["episode_id"])[:WIDTH]
    if len(rows) != WIDTH or any(row.get("scored") is not True for row in rows):
        raise CompositeWidthError("deterministic width sample differs")
    verification = base._read(repository / TOPOLOGY_VERIFICATION_RELATIVE)
    base._validate_seal(verification, "verification_digest")
    if verification.get("passed") is not True \
            or verification.get("producer_result_digest") \
            != "19a98489842ffeaf5141771f4e664ac10af3616cd3a5a9689335cd0c0eadab33":
        raise CompositeWidthError("topology verification prerequisite differs")
    return registry, rows, verification


def build_preregistration(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    if _git(repository, "status", "--porcelain"):
        raise CompositeWidthError("width preregistration requires clean commit")
    if (repository / OUTPUT_RELATIVE).exists() \
            or (repository / OUTPUT_RELATIVE).is_symlink():
        raise CompositeWidthError("width output must be absent before freeze")
    registry, rows, verification = _inputs(repository)
    resident = base._resident()
    state = {
        "schema_version": SCHEMA,
        "status": "frozen_before_composite_width_measurement",
        "implementation_commit": _git(repository, "rev-parse", "HEAD"),
        "runtime_files": {path: _sha(repository / path) for path in RUNTIME_FILES},
        "inputs": {
            "registry_digest": registry["registry_digest"],
            "topology_verification_digest": verification["verification_digest"],
            "packed_generation_id": base.GENERATION_ID,
            "packed_provenance_digest": base.PROVENANCE_DIGEST,
            "resident_content_digest": resident["content_digest"],
        },
        "selection": {
            "month": MONTH, "available_rows": len([
                row for row in registry["queries_data"] if row["cutoff"] == MONTH
            ]),
            "rule": "ascending episode ID, first twelve",
            "query_ids": [row["episode_id"] for row in rows],
        },
        "queries": rows,
        "contract": topology._contract(),
        "distance_weights": topology.DistanceConfig().weights,
        "execution": {
            "topology_order": [name.removesuffix(".json") for name, _, _ in TOPOLOGIES],
            "topologies": [{"groups": groups, "threads_per_group": threads}
                           for _, groups, threads in TOPOLOGIES],
            "total_threads_each": 12,
            "output_root": str((repository / OUTPUT_RELATIVE).resolve()),
            "resume": "reuse only sealed complete topology receipts",
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
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base._validate_seal(value, "preregistration_digest")
    registry, rows, verification = _inputs(repository)
    resident = base._resident()
    if not all((
        value.get("schema_version") == SCHEMA,
        value.get("status") == "frozen_before_composite_width_measurement",
        value.get("queries") == rows,
        value.get("selection", {}).get("query_ids")
            == [row["episode_id"] for row in rows],
        value.get("contract") == topology._contract(),
        value.get("inputs", {}).get("registry_digest") == registry["registry_digest"],
        value.get("inputs", {}).get("topology_verification_digest")
            == verification["verification_digest"],
        value.get("inputs", {}).get("packed_generation_id") == base.GENERATION_ID,
        value.get("inputs", {}).get("packed_provenance_digest")
            == base.PROVENANCE_DIGEST,
        value.get("inputs", {}).get("resident_content_digest") == resident["content_digest"],
        set(value.get("runtime_files", {})) == set(RUNTIME_FILES),
        value.get("execution", {}).get("topology_order") == ["P4T3", "P12T1"],
        value.get("execution", {}).get("total_threads_each") == 12,
        value.get("claims") == {
            "outcomes_or_labels_used": False,
            "historical_walk_forward_query_outcomes_opened": False,
            "final_period_result_opened": False,
            "production_promotion_authorized": False,
        },
    )):
        raise CompositeWidthError("width preregistration differs")
    commit = value.get("implementation_commit")
    if type(commit) is not str:
        raise CompositeWidthError("width implementation commit differs")
    _git(repository, "merge-base", "--is-ancestor", commit, "HEAD")
    for path, digest in value["runtime_files"].items():
        blob = subprocess.run(
            ["git", "show", f"{commit}:{path}"], cwd=repository,
            capture_output=True, check=False,
        )
        if blob.returncode or sha256(blob.stdout).hexdigest() != digest \
                or _sha(repository / path) != digest:
            raise CompositeWidthError(f"width runtime differs: {path}")
    return rows, resident


def _run(
    repository: Path, store_root: Path, rows: list[dict[str, Any]],
    groups: int, threads: int,
) -> dict[str, Any]:
    grouped = [tuple(rows[index::groups]) for index in range(groups)]
    started = perf_counter()
    results = []
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=groups, mp_context=context) as executor:
        futures = [executor.submit(
            topology._run_group, str(repository), str(store_root), group, threads,
        ) for group in grouped]
        for future in as_completed(futures):
            result = future.result(); results.append(result)
            print(
                f"[composite-width] group queries={result['queries']} "
                f"threads={threads} seconds={result['elapsed_seconds']:.2f}",
                flush=True,
            )
    by_id = {case["query_id"]: case for result in results for case in result["cases"]}
    ordered = [by_id[row["episode_id"]] for row in rows]
    state = {
        "schema_version": "m04r14-wf03-composite-width-run-v1",
        "status": "complete", "groups": groups, "threads_per_group": threads,
        "total_threads": groups * threads,
        "wall_seconds": perf_counter() - started,
        "maximum_group_seconds": max(result["elapsed_seconds"] for result in results),
        "sum_group_peak_rss_mb": sum(result["peak_rss_mb"] for result in results),
        "all_zero_swap": all(result["swap_kib"] == 0 for result in results),
        "cases": ordered,
        "outcomes_or_labels_used": False,
        "historical_walk_forward_query_outcomes_opened": False,
        "final_period_result_opened": False,
    }
    return state


def _semantic_map(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        row["query_id"]: topology._case_semantics(row) for row in value["cases"]
    }


def execute(repository: Path, preregistration: Mapping[str, Any]) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    if _git(repository, "status", "--porcelain"):
        raise CompositeWidthError("width execution requires clean commit")
    rows, resident = validate_preregistration(repository, preregistration)
    root = repository / OUTPUT_RELATIVE
    if root.is_symlink():
        raise CompositeWidthError("width root is symlinked")
    root.mkdir(parents=True, exist_ok=True)
    contract = root / "CONTRACT.json"
    if contract.exists():
        if base._read(contract) != preregistration:
            raise CompositeWidthError("width output contract differs")
    else:
        base._atomic(contract, preregistration)
    outputs = []
    for filename, groups, threads in TOPOLOGIES:
        path = root / filename
        print(f"[composite-width] starting {filename}", flush=True)
        if path.exists():
            value = base._read(path); base._validate_seal(value)
        else:
            value = base._sealed({
                **_run(repository, Path(resident["store_root"]), rows, groups, threads),
                "preregistration_digest": preregistration["preregistration_digest"],
            })
            base._atomic(path, value)
        if value.get("groups") != groups or value.get("threads_per_group") != threads \
                or value.get("total_threads") != 12 or value.get("all_zero_swap") is not True \
                or value.get("preregistration_digest") \
                != preregistration["preregistration_digest"] \
                or value.get("outcomes_or_labels_used") is not False \
                or value.get("historical_walk_forward_query_outcomes_opened") is not False \
                or value.get("final_period_result_opened") is not False \
                or [case["query_id"] for case in value["cases"]] \
                != [row["episode_id"] for row in rows] \
                or any(case["semantic_digest"]
                       != stable_hash(topology._case_semantics(case))
                       for case in value["cases"]) \
                or any(not topology._certificate_closed(case["certificate"], case["matches"])
                       for case in value["cases"]):
            raise CompositeWidthError(f"width topology receipt differs: {filename}")
        outputs.append(value)
    p4, p12 = outputs
    semantics_equal = _semantic_map(p4) == _semantic_map(p12)
    gates = {
        "all_twelve_queries_equal": semantics_equal,
        "all_twenty_four_certificates_close": True,
        "zero_process_swap": True,
        "outcomes_or_labels_excluded": True,
    }
    state = {
        "schema_version": "m04r14-t14-10-wf03-composite-width-result-v1",
        "status": "complete", "passed": all(gates.values()), "gates": gates,
        "preregistration_digest": preregistration["preregistration_digest"],
        "p4t3_digest": p4["result_digest"], "p12t1_digest": p12["result_digest"],
        "p4t3_wall_seconds": p4["wall_seconds"],
        "p12t1_wall_seconds": p12["wall_seconds"],
        "p12_over_p4_speedup": p4["wall_seconds"] / p12["wall_seconds"],
        "selected_topology": "p12t1" if p12["wall_seconds"] < p4["wall_seconds"] else "p4t3",
        "outcomes_or_labels_used": False,
        "historical_walk_forward_query_outcomes_opened": False,
        "final_period_result_opened": False,
        "production_promotion_authorized": False,
    }
    result = base._sealed(state)
    if not (root / "RESULT.json").exists():
        base._atomic(root / "RESULT.json", result)
    elif base._read(root / "RESULT.json") != result:
        raise CompositeWidthError("width terminal result differs")
    if not result["passed"]:
        raise CompositeWidthError("width gates failed")
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
        result = execute(repository, base._read(repository / PREREGISTRATION_RELATIVE))
    print(json.dumps(result, indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
