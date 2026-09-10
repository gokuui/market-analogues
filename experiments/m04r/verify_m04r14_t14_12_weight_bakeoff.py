"""Independently verify the T14-12 outcome-blind volatility-weight bake-off."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from market_analogues.types import stable_hash

from experiments.m04r import m04r14_t14_10_wf03_feasibility as io
from experiments.m04r import verify_m04r14_t14_12_balance as balance_oracle
from experiments.m04r import verify_m04r14_t14_12_rank_matches as rank_oracle
from experiments.m04r import m04r14_t14_12_weight_bakeoff as target


SCHEMA = "m04r14-t14-12-volatility-weight-bakeoff-verification-v2"
WEIGHTS = (1, 2, 4, 8, 16)
TOLERANCE = 5e-12


class WeightBakeoffVerificationError(RuntimeError): pass


def _now() -> str: return datetime.now(timezone.utc).isoformat()


def _sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20): digest.update(block)
    return digest.hexdigest()


def _valid(value: Mapping[str, Any], key: str = "result_digest", *, timing: bool = False) -> bool:
    omitted = {key, "created_at"} | ({"elapsed_seconds"} if timing else set())
    return value.get(key) == stable_hash({name: item for name, item in value.items() if name not in omitted})


def _weighted_batch(names: np.ndarray, ranks: np.ndarray, candidates: np.ndarray, events: np.ndarray,
                    contract: str, ids: Sequence[str], weight: int) -> list[tuple[np.ndarray, np.ndarray]]:
    if not len(candidates): return [(np.array([], dtype=int), np.array([], dtype=float)) for _ in events]
    scale = np.sqrt(np.array([1., float(weight), 1., 1.])); points = ranks[candidates] * scale
    queries = ranks[events] * scale; frontier = min(len(candidates), 16); tree = cKDTree(points)
    distances, indices = tree.query(queries, k=frontier)
    if frontier == 1: distances, indices = np.asarray(distances)[:, None], np.asarray(indices)[:, None]
    result = []
    for row, signal_id in enumerate(ids):
        local = np.atleast_1d(indices[row]).astype(int)
        if len(candidates) > frontier:
            kth, last = float(np.atleast_1d(distances[row])[4]), float(np.atleast_1d(distances[row])[-1])
            if abs(last - kth) <= 1e-14 * max(1., abs(last), abs(kth)):
                local = np.asarray(tree.query_ball_point(queries[row], kth * (1 + 1e-12) + 1e-15), dtype=int)
        result.append(rank_oracle._sort(names, candidates[local], points[local], queries[row], contract, signal_id))
    return result


def _year_statistics(year: int, frame: pd.DataFrame, contract: str,
                     exhaustive_dates: set[pd.Timestamp] | None = None) -> tuple[dict[str, Any], dict[str, list[int]], int]:
    states: dict[str, dict[str, Any]] = defaultdict(balance_oracle._empty); coverage: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    era = balance_oracle._era(year); exhaustive = 0
    for signal_date, group in frame.groupby("signal_date", sort=True):
        date_text = pd.Timestamp(signal_date).date().isoformat()
        for population in ("broad", "investable"):
            current = group if population == "broad" else group.loc[group.investable]
            if current.empty: continue
            current = current.reset_index(drop=True); names = current.symbol.astype(str).to_numpy(object)
            raw = current[list(rank_oracle.MATCH_COLUMNS)].to_numpy(float); ranks = rank_oracle._ranks(raw)
            deciles = np.clip(np.ceil(10 * ranks), 1, 10).astype(np.uint8)
            transformed = {name: balance_oracle._transform(name, current[name].to_numpy(float))
                           for name in balance_oracle.VERIFY_TRANSFORMS}
            for signal_name, (risk_column, raw_column, event_column) in rank_oracle.SIGNALS.items():
                candidates = np.flatnonzero(current[risk_column].to_numpy(bool) & ~current[raw_column].to_numpy(bool))
                events = np.flatnonzero(current[event_column].to_numpy(bool))
                ids = [rank_oracle._event_id(contract, signal_name, str(names[index]), date_text) for index in events]
                for weight in WEIGHTS:
                    selected = _weighted_batch(names, ranks, candidates, events, contract, ids, weight)
                    if exhaustive_dates is not None and pd.Timestamp(signal_date) in exhaustive_dates:
                        scale = np.sqrt(np.array([1., float(weight), 1., 1.]))
                        for event_index, signal_id, observed in zip(events, ids, selected):
                            expected = rank_oracle._sort(names, candidates, ranks[candidates] * scale,
                                                         ranks[event_index] * scale, contract, signal_id)
                            if not np.array_equal(observed[0], expected[0]) or not np.array_equal(observed[1], expected[1]):
                                raise WeightBakeoffVerificationError("weighted spatial/exhaustive selection differs")
                            exhaustive += 1
                    key = f"{weight}|{population}|{signal_name}"; coverage[key][0] += len(events)
                    coverage[key][1] += sum(len(indices) == 5 for indices, _ in selected)
                    full_positions = [position for position, (indices, _) in enumerate(selected) if len(indices) == 5]
                    if not full_positions: continue
                    full_events = events[full_positions]; control_matrix = np.vstack([selected[position][0] for position in full_positions])
                    for column_index, name in enumerate(balance_oracle.VERIFY_TRANSFORMS):
                        event_values = np.repeat(transformed[name][full_events], 5)
                        control_values = transformed[name][control_matrix.ravel()]
                        gaps = np.abs(np.repeat(deciles[full_events, column_index].astype(int), 5)
                                      - deciles[control_matrix.ravel(), column_index].astype(int))
                        for scope in ("overall", era):
                            balance_oracle._add(states[f"{weight}|{population}|{signal_name}|{scope}|{name}"],
                                                event_values, control_values, gaps)
    return dict(states), dict(coverage), exhaustive


def _assert_stats(expected: Mapping[str, Any], observed: Mapping[str, Any], label: str) -> None:
    if expected.keys() != observed.keys(): raise WeightBakeoffVerificationError(f"{label} statistic keys differ")
    for key in expected: balance_oracle._assert_nested(expected[key], observed[key], f"{label}/{key}")


def _summaries(states: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    rows = []
    for weight in WEIGHTS:
        subset = {}
        for key, value in states.items():
            if not key.startswith(f"{weight}|"): continue
            _, population, signal, scope, covariate = key.split("|")
            subset[f"{population}|{signal}|{scope}|all|{covariate}"] = value
        frame = balance_oracle._summaries(subset); frame.insert(0, "volatility_weight", weight); rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def execute(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    if subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=repository,
                      capture_output=True, text=True, check=True).stdout: raise WeightBakeoffVerificationError("clean worktree required")
    if target.WEIGHTS != WEIGHTS: raise WeightBakeoffVerificationError("candidate grid differs")
    started = perf_counter(); prereg, _ = target.validate_preregistration(repository); root = repository / target.OUTPUT_RELATIVE
    seal = io._read(root / "SEALED.json")
    if not target._valid(seal, timing=True) or seal.get("file_manifest") != target._manifest(root, target.OUTPUT_FILES):
        raise WeightBakeoffVerificationError("bake-off seal differs")
    frame = target.rank_matches.v1._read_causal_panel(repository); grouped = frame.groupby(frame.signal_date.dt.year, sort=True)
    combined: dict[str, dict[str, Any]] = defaultdict(balance_oracle._empty); coverage: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    digests = []; exhaustive = 0
    for year in prereg["years"]:
        year_frame = grouped.get_group(year); dates = sorted(pd.Timestamp(value) for value in year_frame.signal_date.unique())
        samples = set(sorted(dates, key=lambda value: stable_hash([year, str(value)]))[:1])
        expected, expected_coverage, checked = _year_statistics(year, year_frame, prereg["contract_digest"], samples)
        year_seal = io._read(repository / target.CACHE_RELATIVE / f"year-{year}" / "YEAR_SEALED.json")
        if not target._valid(year_seal, timing=True) or year_seal.get("outcome_columns_read") != []:
            raise WeightBakeoffVerificationError(f"bake-off year differs: {year}")
        _assert_stats(expected, year_seal["statistics"], f"year {year}")
        if expected_coverage != year_seal["coverage"]: raise WeightBakeoffVerificationError(f"coverage differs: {year}")
        for key, value in expected.items(): balance_oracle._merge(combined[key], value)
        for key, value in expected_coverage.items(): coverage[key] = [coverage[key][i] + value[i] for i in range(2)]
        digests.append(year_seal["result_digest"]); exhaustive += checked
    summary = _summaries(combined); balance_oracle._assert_frame(summary, pd.read_parquet(root / "balance-bakeoff.parquet"), "bake-off summary")
    candidate_results = []
    for weight in WEIGHTS:
        current = summary.loc[summary.volatility_weight.eq(weight)]; overall_mask = current.scope.eq("overall")
        overall = bool((current.loc[overall_mask, "standardized_mean_difference"].abs() <= .10).all())
        eras = bool((current.loc[~overall_mask, "standardized_mean_difference"].abs() <= .20).all())
        keys = [key for key in coverage if key.startswith(f"{weight}|")]; total = sum(coverage[key][0] for key in keys)
        full = sum(coverage[key][1] for key in keys); rate = full / total
        candidate_results.append({"volatility_weight": weight, "overall_balance_pass": overall,
                                  "era_balance_pass": eras, "full_match_coverage": rate,
                                  "candidate_pass": overall and eras and rate >= .90})
    passing = [row["volatility_weight"] for row in candidate_results if row["candidate_pass"]]; selected = min(passing) if passing else None
    decision = io._read(root / "selection-decision.json")
    if not target._valid(decision) or decision["candidate_results"] != candidate_results \
            or decision["selected_volatility_weight"] != selected or decision["selection_pass"] != (selected is not None):
        raise WeightBakeoffVerificationError("bake-off decision differs")
    if digests != seal["year_result_digests"] or seal.get("post_signal_results_accessed") is not False:
        raise WeightBakeoffVerificationError("bake-off aggregate binding differs")
    state = {"schema_version": SCHEMA, "status": "verified", "passed": True,
        "store_result_digest": seal["result_digest"], "store_result_sha256": _sha(root / "SEALED.json"),
        "verified_summary_rows": len(summary), "independent_exhaustive_cases": exhaustive,
        "selected_volatility_weight": selected,
        "gates": {"all_weighted_year_statistics_reconstructed": True, "all_candidate_decisions_reconstructed": True,
                  "stratified_exhaustive_oracle_passed": True, "outcome_boundary_remained_closed": True},
        "outcome_columns_read": [], "post_signal_results_accessed": False, "production_promotion_authorized": False,
        "elapsed_seconds": perf_counter() - started, "created_at": _now()}
    result = {**state, "verification_digest": stable_hash(state)}; output = repository / target.VERIFICATION_RELATIVE
    if output.exists(): return io._read(output / "VERIFIED.json")
    output.parent.mkdir(parents=True, exist_ok=True); temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try: target.smoke._atomic_json(temporary / "VERIFIED.json", result); os.replace(temporary, output)
    except BaseException: shutil.rmtree(temporary, ignore_errors=True); raise
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--repository", type=Path, required=True)
    args = parser.parse_args(argv); print(json.dumps(execute(args.repository), indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
