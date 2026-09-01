"""Outcome-blind real-data tightness gate for the quantized DTW interval bound."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from time import perf_counter
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from market_analogues.causal_prefix import causal_prefix_digest
from market_analogues.distance import complete_representation_distance
from market_analogues.dtw_interval_bound import (
    dtw_interval_bound_contract,
    quantize_dtw_samples,
    quantized_dtw_lower_bound,
)
from market_analogues.exact_batch import (
    batch_representation_lower_bounds,
    exact_representations_at_positions,
)
from market_analogues.packed_bound_search import (
    BoundProposal,
    scan_packed_bound_threshold,
)
from market_analogues.packed_bound_store import load_packed_generation
from market_analogues.types import EpisodeKey, InstrumentKey, stable_hash

from experiments.m04r import m04r14_t14_10_wf03_feasibility as base
from experiments.m04r import m04r14_t14_10_wf03b_price_threshold_diagnostic as diagnostic


SCHEMA = "m04r14-t14-10-wf03b-dtw-bound-poc-preregistration-v1"
OUTPUT_RELATIVE = Path(
    "config/data/analogues/m04r14/t14-10-wf03b-dtw-bound-poc-v1"
)
PREREGISTRATION_RELATIVE = Path(
    "experiments/m04r/m04r14_t14_10_wf03b_dtw_bound_poc_preregistered.json"
)
SAMPLE_ROWS = 4_096
WORKERS = 8
MINIMUM_CASE_PRUNED_FRACTION = 0.25
MINIMUM_AGGREGATE_PRUNED_FRACTION = 0.35
TOLERANCE = 1e-12
RUNTIME_FILES = (
    "experiments/m04r/m04r14_t14_10_wf03b_dtw_bound_poc.py",
    "experiments/m04r/m04r14_t14_10_wf03_feasibility.py",
    "experiments/m04r/m04r14_t14_10_wf03b_price_poc.py",
    "experiments/m04r/m04r14_t14_10_wf03b_price_threshold_diagnostic.py",
    "src/market_analogues/adapters.py",
    "src/market_analogues/causal_prefix.py",
    "src/market_analogues/config.py",
    "src/market_analogues/dtw_interval_bound.py",
    "src/market_analogues/distance.py",
    "src/market_analogues/episodes.py",
    "src/market_analogues/exact_batch.py",
    "src/market_analogues/packed_bound_search.py",
    "src/market_analogues/packed_bound_store.py",
    "src/market_analogues/quantized_bound.py",
    "src/market_analogues/representation.py",
    "src/market_analogues/resident_store.py",
    "src/market_analogues/search.py",
    "src/market_analogues/types.py",
)


class DtwBoundPocError(RuntimeError):
    pass


def _git(repository: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repository, text=True, capture_output=True, check=False,
    )
    if result.returncode:
        raise DtwBoundPocError(f"git {' '.join(args)} failed")
    return result.stdout.strip()


def selection_key(query_id: str, episode_id: str) -> tuple[str, str]:
    return stable_hash({
        "schema": "wf03b-dtw-bound-sample-v1",
        "query_id": query_id,
        "episode_id": episode_id,
    }), episode_id


def retain_sample(
    current: Iterable[BoundProposal], incoming: Iterable[BoundProposal],
    *, query_id: str, sample_rows: int,
) -> list[BoundProposal]:
    if sample_rows < 1:
        raise DtwBoundPocError("sample size must be positive")
    by_id: dict[str, BoundProposal] = {}
    for row in (*tuple(current), *tuple(incoming)):
        previous = by_id.setdefault(row.episode_id, row)
        if previous != row:
            raise DtwBoundPocError("sample contains a conflicting episode ID")
    return sorted(
        by_id.values(), key=lambda row: selection_key(query_id, row.episode_id),
    )[:sample_rows]


def promotion_gate(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total_rows = sum(int(row["sample_rows"]) for row in cases)
    total_pruned = sum(int(row["enhanced_pruned"]) for row in cases)
    aggregate = total_pruned / total_rows if total_rows else 0.0
    safety = all(
        int(row["false_prunes"]) == 0
        and float(row["maximum_bound_excess"]) <= TOLERANCE
        and float(row["maximum_packed_native_excess"]) <= TOLERANCE
        for row in cases
    )
    material = bool(cases) and all(
        float(row["enhanced_pruned_fraction"]) >= MINIMUM_CASE_PRUNED_FRACTION
        for row in cases
    ) and aggregate >= MINIMUM_AGGREGATE_PRUNED_FRACTION
    return {
        "safety_passed": safety,
        "material_pruning_passed": material,
        "minimum_case_pruned_fraction": MINIMUM_CASE_PRUNED_FRACTION,
        "minimum_aggregate_pruned_fraction": MINIMUM_AGGREGATE_PRUNED_FRACTION,
        "observed_aggregate_pruned_fraction": aggregate,
        "auxiliary_store_authorized": safety and material,
    }


def _inputs(repository: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    threshold_path = repository / diagnostic.OUTPUT_RELATIVE / "RESULT.json"
    threshold = base._read(threshold_path)
    base._validate_seal(threshold)
    if threshold.get("passed") is not True or len(threshold.get("cases", ())) != 3:
        raise DtwBoundPocError("threshold diagnostic prerequisite differs")
    failed = {row["query_id"]: row for row in diagnostic._failed_cases(repository)}
    cases = []
    for row in threshold["cases"]:
        prior = failed.get(row["query_id"])
        if prior is None or row["threshold"] != prior["threshold"] \
                or row["additional_admitted"] < SAMPLE_ROWS:
            raise DtwBoundPocError("price threshold case differs")
        cases.append({
            "ordinal": row["ordinal"], "label": row["label"],
            "query_id": row["query_id"], "threshold": row["threshold"],
            "additional_admitted": row["additional_admitted"],
            "eligible_candidates": row["eligible_candidates"],
            "admitted_set_digest": row["forward_report"]["admitted_set_digest"],
            "threshold_case_digest": stable_hash(row),
            "price_failure_sha256": prior["failed_sha256"],
            "price_proposal_sha256": prior["proposal_sha256"],
        })
    return cases, threshold


def build_preregistration(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    if _git(repository, "status", "--porcelain"):
        raise DtwBoundPocError("DTW POC preregistration requires a clean commit")
    cases, threshold = _inputs(repository)
    state = {
        "schema_version": SCHEMA,
        "status": "frozen_before_real_dtw_bound_measurements",
        "implementation_commit": _git(repository, "rev-parse", "HEAD"),
        "runtime_files": {path: base._sha(repository / path) for path in RUNTIME_FILES},
        "inputs": {
            "generation_id": base.GENERATION_ID,
            "provenance_digest": base.PROVENANCE_DIGEST,
            "registry_digest": base.REGISTRY_DIGEST,
            "threshold_result_sha256": base._sha(
                repository / diagnostic.OUTPUT_RELATIVE / "RESULT.json"
            ),
            "threshold_result_digest": threshold["result_digest"],
        },
        "cases": cases,
        "bound_contract": dtw_interval_bound_contract(),
        "execution": {
            "sample_rows_per_case": SAMPLE_ROWS,
            "sample": (
                "smallest SHA-256 keys among old-bound-admitted rows after excluding "
                "the already exact 16,384-row frontier"
            ),
            "workers": WORKERS, "tolerance": TOLERANCE,
            "forward_block_rows": 4_096,
            "minimum_case_pruned_fraction": MINIMUM_CASE_PRUNED_FRACTION,
            "minimum_aggregate_pruned_fraction": MINIMUM_AGGREGATE_PRUNED_FRACTION,
            "output_root": str((repository / OUTPUT_RELATIVE).resolve()),
        },
        "claims": {
            "historical_walk_forward_query_outcomes_opened": False,
            "final_period_result_opened": False,
            "authority_or_outcome_paths_accepted": False,
            "production_promotion_authorized": False,
        },
    }
    return base._sealed(state, "preregistration_digest")


def validate_preregistration(repository: Path, value: Mapping[str, Any]) -> None:
    base._validate_seal(value, "preregistration_digest")
    cases, threshold = _inputs(repository)
    if value.get("schema_version") != SCHEMA or value.get("cases") != cases \
            or value.get("bound_contract") != dtw_interval_bound_contract() \
            or value.get("inputs", {}).get("threshold_result_digest") \
            != threshold["result_digest"]:
        raise DtwBoundPocError("DTW POC preregistration differs")
    head = value.get("implementation_commit")
    if type(head) is not str:
        raise DtwBoundPocError("DTW POC implementation commit differs")
    _git(repository, "merge-base", "--is-ancestor", head, "HEAD")
    for path, digest in value["runtime_files"].items():
        if base._sha(repository / path) != digest:
            raise DtwBoundPocError(f"DTW POC runtime differs: {path}")
        blob = subprocess.run(
            ["git", "show", f"{head}:{path}"], cwd=repository,
            capture_output=True, check=False,
        )
        if blob.returncode or sha256(blob.stdout).hexdigest() != digest:
            raise DtwBoundPocError(f"DTW POC implementation binding differs: {path}")


def _quantiles(values: Sequence[float]) -> dict[str, float]:
    result = np.quantile(np.asarray(values, dtype=np.float64), (0, .25, .5, .75, 1))
    return dict(zip(("minimum", "q25", "median", "q75", "maximum"),
                    (float(value) for value in result), strict=True))


def _score_symbol(
    symbol: str, proposals: Sequence[BoundProposal], *, source: Any,
    benchmark: pd.DataFrame, query_representation: Any, manifest: Mapping[str, Any],
    query_id: str, lookback: int, representation_version: str,
) -> list[dict[str, Any]]:
    prefixes = manifest["provenance"]["source_prefixes"]
    expected = prefixes.get(symbol)
    if type(expected) is not dict:
        raise DtwBoundPocError(f"missing packed source prefix: {symbol}")
    bars = source.load(InstrumentKey("nasdaq", symbol))
    if asdict(causal_prefix_digest(bars, expected["requested_cutoff"])) != expected:
        raise DtwBoundPocError(f"packed source prefix changed: {symbol}")
    last_cutoff = max(row.cutoff_ns for row in proposals)
    frame = bars[bars.timestamp.astype("int64") <= last_cutoff].reset_index(drop=True)
    positions = {int(pd.Timestamp(value).value): index
                 for index, value in enumerate(frame.timestamp)}
    requested = []
    for row in proposals:
        position = positions.get(row.cutoff_ns)
        if position is None or position + 1 < lookback:
            raise DtwBoundPocError("cannot reconstruct sampled candidate")
        key = EpisodeKey(
            InstrumentKey("nasdaq", symbol), pd.Timestamp(row.cutoff_ns),
            lookback, representation_version,
        )
        if key.id != row.episode_id:
            raise DtwBoundPocError("sampled episode identity differs")
        requested.append(position)
    representations = exact_representations_at_positions(
        frame, benchmark, positions=np.asarray(requested, dtype=int),
        lookback=lookback,
    )
    bounded = batch_representation_lower_bounds(query_representation, representations)
    outputs = []
    for index, (proposal, candidate) in enumerate(
            zip(proposals, representations, strict=True)):
        native = float(bounded.components["price"][index])
        components = {
            name: float(values[index]) for name, values in bounded.components.items()
        }
        _total, exact_components, _path = complete_representation_distance(
            query_representation, candidate, float(bounded.totals[index]),
            components, float(bounded.rigid_price[index]), reconstruct_path=False,
        )
        exact = float(exact_components["price"])
        dtw_bound = quantized_dtw_lower_bound(
            query_representation, quantize_dtw_samples(candidate),
        )
        enhanced = proposal.lower_bound + .45 * dtw_bound
        if not np.isfinite((proposal.lower_bound, native, dtw_bound, enhanced, exact)).all() \
                or proposal.lower_bound > native + TOLERANCE \
                or native > exact + TOLERANCE \
                or enhanced > exact + TOLERANCE:
            raise DtwBoundPocError("sampled bound safety differs")
        outputs.append({
            "episode_id": proposal.episode_id, "symbol": proposal.symbol,
            "selection_key": selection_key(query_id, proposal.episode_id)[0],
            "packed_bound": proposal.lower_bound, "native_bound": native,
            "dtw_interval_bound": dtw_bound, "enhanced_bound": enhanced,
            "exact_price": exact,
        })
    return outputs


def _case(
    repository: Path, case: Mapping[str, Any], registry_row: Mapping[str, Any],
    resident: Mapping[str, Any],
) -> dict[str, Any]:
    started = perf_counter()
    source, episode, _request, packed = base._context(repository, registry_row)
    failed = {row["query_id"]: row for row in diagnostic._failed_cases(repository)}[
        case["query_id"]
    ]
    excluded = frozenset(failed["excluded_episode_ids"])
    sample: list[BoundProposal] = []

    def consume(rows: tuple[BoundProposal, ...]) -> None:
        nonlocal sample
        sample = retain_sample(
            sample, rows, query_id=case["query_id"], sample_rows=SAMPLE_ROWS,
        )

    scan = scan_packed_bound_threshold(
        Path(resident["store_root"]), base.GENERATION_ID, packed,
        component="price", branch_aware=True,
        upper_inclusive=float(case["threshold"]),
        excluded_episode_ids=excluded, block_rows=4_096,
        block_order="forward", verify_content=False,
        expected_provenance_digest=base.PROVENANCE_DIGEST, consume=consume,
    )
    if scan.admitted_rows != case["additional_admitted"] \
            or scan.admitted_set_digest != case["admitted_set_digest"] \
            or len(sample) != SAMPLE_ROWS:
        raise DtwBoundPocError("sample scan differs from frozen threshold diagnostic")
    loaded = load_packed_generation(
        Path(resident["store_root"]), base.GENERATION_ID,
        expected_provenance_digest=base.PROVENANCE_DIGEST,
        verify_content=False, validate_records=False,
    )
    benchmark = source.load_benchmark()
    if benchmark is None:
        raise DtwBoundPocError("DTW POC requires benchmark context")
    expected_benchmark = loaded.manifest["provenance"]["benchmark_prefix"]
    if asdict(causal_prefix_digest(
            benchmark, expected_benchmark["requested_cutoff"])) != expected_benchmark:
        raise DtwBoundPocError("packed benchmark prefix changed")
    grouped: dict[str, list[BoundProposal]] = {}
    for row in sample:
        grouped.setdefault(row.symbol, []).append(row)
    exact_started = perf_counter()
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = [executor.submit(
            _score_symbol, symbol, tuple(rows), source=source, benchmark=benchmark,
            query_representation=packed.representation, manifest=loaded.manifest,
            query_id=case["query_id"], lookback=episode.key.lookback,
            representation_version=episode.key.representation_version,
        ) for symbol, rows in sorted(grouped.items())]
        scored = [item for future in futures for item in future.result()]
    exact_seconds = perf_counter() - exact_started
    scored.sort(key=lambda row: (row["selection_key"], row["episode_id"]))
    if [row["episode_id"] for row in scored] != [row.episode_id for row in sample]:
        raise DtwBoundPocError("scored sample order differs")
    threshold = float(case["threshold"])
    enhanced_pruned = sum(row["enhanced_bound"] > threshold for row in scored)
    exact_competitors = sum(row["exact_price"] <= threshold for row in scored)
    false_prunes = sum(
        row["enhanced_bound"] > threshold and row["exact_price"] <= threshold
        for row in scored
    )
    maximum_excess = max(row["enhanced_bound"] - row["exact_price"] for row in scored)
    maximum_packed_native_excess = max(
        row["packed_bound"] - row["native_bound"] for row in scored
    )
    compact_scores = [{
        key: (value.hex() if isinstance(value, float) else value)
        for key, value in row.items()
    } for row in scored]
    gains = [row["enhanced_bound"] - row["packed_bound"] for row in scored]
    ratios = [row["enhanced_bound"] / row["exact_price"]
              if row["exact_price"] > 0 else 1.0 for row in scored]
    state = {
        "schema_version": "m04r14-t14-10-wf03b-dtw-bound-case-v1",
        "status": "complete", "ordinal": case["ordinal"],
        "label": case["label"], "query_id": case["query_id"],
        "threshold": threshold, "additional_admitted": scan.admitted_rows,
        "admitted_set_digest": scan.admitted_set_digest,
        "sample_rows": len(scored), "sample_distinct_symbols": len(grouped),
        "sample_digest": stable_hash([
            {"episode_id": row.episode_id,
             "selection_key": selection_key(case["query_id"], row.episode_id)[0]}
            for row in sample
        ]),
        "score_digest": stable_hash(compact_scores),
        "enhanced_pruned": enhanced_pruned,
        "enhanced_pruned_fraction": enhanced_pruned / len(scored),
        "exact_competitors": exact_competitors, "false_prunes": false_prunes,
        "maximum_bound_excess": maximum_excess,
        "maximum_packed_native_excess": maximum_packed_native_excess,
        "bound_gain_quantiles": _quantiles(gains),
        "enhanced_exact_ratio_quantiles": _quantiles(ratios),
        "scan_seconds": scan.elapsed_seconds, "exact_seconds": exact_seconds,
        "elapsed_seconds": perf_counter() - started,
        "historical_walk_forward_query_outcomes_opened": False,
        "final_period_result_opened": False,
    }
    return base._sealed(state, "case_digest")


def execute(repository: Path, preregistration: Mapping[str, Any]) -> dict[str, Any]:
    validate_preregistration(repository, preregistration)
    _registry, by_id = base._registry(repository)
    resident = base._resident()
    root = repository / OUTPUT_RELATIVE
    if root.exists() or root.is_symlink():
        raise DtwBoundPocError("DTW POC output already exists")
    root.mkdir(parents=True)
    base._atomic(root / "CONTRACT.json", preregistration)
    cases = []
    for row in preregistration["cases"]:
        result = _case(repository, row, by_id[row["query_id"]], resident)
        path = root / f"CASE-{int(row['ordinal']):03d}.json"
        base._atomic(path, result)
        cases.append(result)
    gate = promotion_gate(cases)
    state = {
        "schema_version": "m04r14-t14-10-wf03b-dtw-bound-result-v1",
        "status": "complete", "cases": cases, "case_digest": stable_hash(cases),
        "gate": gate, "passed": gate["safety_passed"],
        "historical_walk_forward_query_outcomes_opened": False,
        "final_period_result_opened": False,
        "production_promotion_authorized": False,
    }
    result = base._sealed(state)
    base._atomic(root / "RESULT.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preregister", "run"))
    parser.add_argument("--repository", type=Path, required=True)
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
