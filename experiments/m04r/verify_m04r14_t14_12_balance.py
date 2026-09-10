"""Independently verify the outcome-blind T14-12 balance audit."""
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

from market_analogues.types import stable_hash

from experiments.m04r import m04r14_t14_10_wf03_feasibility as base
from experiments.m04r import m04r14_t14_12_balance as target


SCHEMA = "m04r14-t14-12-post-signal-balance-verification-v1"
VERIFY_TRANSFORMS = {
    "prior_return_63": "identity", "prior_volatility_20": "log1p",
    "prior_close": "log", "prior_median_dollar_volume_20": "log1p",
}
TOLERANCE = 5e-12


class BalanceVerificationError(RuntimeError): pass


def _now() -> str: return datetime.now(timezone.utc).isoformat()


def _sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20): digest.update(block)
    return digest.hexdigest()


def _valid(value: Mapping[str, Any], key: str = "result_digest", *, timing: bool = False) -> bool:
    omitted = {key, "created_at"} | ({"elapsed_seconds"} if timing else set())
    return value.get(key) == stable_hash({name: item for name, item in value.items() if name not in omitted})


def _transform(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float); rule = VERIFY_TRANSFORMS[name]
    if not np.isfinite(array).all(): raise BalanceVerificationError(f"non-finite covariate: {name}")
    if rule == "identity": return array
    if rule == "log": return np.log(array)
    if rule == "log1p": return np.log1p(array)
    raise BalanceVerificationError("unknown transform")


def _empty() -> dict[str, Any]:
    return {"n": 0, "event_sum": 0., "event_sumsq": 0., "control_sum": 0.,
            "control_sumsq": 0., "difference_sum": 0., "absolute_difference_sum": 0.,
            "decile_gap_histogram": [0] * 10}


def _era(year: int) -> str:
    boundaries = ((2007, "era_2000_2007"), (2012, "era_2008_2012"),
                  (2017, "era_2013_2017"), (2022, "era_2018_2022"))
    for end, name in boundaries:
        if year <= end: return name
    return "era_2023_2026"


def _add(state: dict[str, Any], event: np.ndarray, control: np.ndarray, gap: np.ndarray) -> None:
    difference = event - control; state["n"] += event.size
    state["event_sum"] += float(np.sum(event)); state["event_sumsq"] += float(np.sum(event * event))
    state["control_sum"] += float(np.sum(control)); state["control_sumsq"] += float(np.sum(control * control))
    state["difference_sum"] += float(np.sum(difference)); state["absolute_difference_sum"] += float(np.sum(np.abs(difference)))
    counts = np.bincount(gap.astype(int), minlength=10)
    state["decile_gap_histogram"] = [old + int(new) for old, new in zip(state["decile_gap_histogram"], counts)]


def _reconstruct_year(year: int, panel: pd.DataFrame, controls: pd.DataFrame) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    features = panel[["signal_date", "symbol", *VERIFY_TRANSFORMS]]
    merged = controls.merge(features, left_on=["signal_date", "event_symbol"],
                            right_on=["signal_date", "symbol"], validate="many_to_one")
    merged = merged.rename(columns={name: f"event_{name}" for name in VERIFY_TRANSFORMS}).drop(columns="symbol")
    merged = merged.merge(features, left_on=["signal_date", "control_symbol"],
                          right_on=["signal_date", "symbol"], validate="many_to_one")
    merged = merged.rename(columns={name: f"control_{name}" for name in VERIFY_TRANSFORMS}).drop(columns="symbol")
    states: dict[str, dict[str, Any]] = defaultdict(_empty); era = _era(year)
    for (population, signal), base_group in merged.groupby(["population", "signal_name"], sort=True):
        for tier in ("all", "exact_all_four_deciles", "within_one_bucket_all_four", "same_date_unmatched"):
            group = base_group if tier == "all" else base_group.loc[base_group.match_tier.eq(tier)]
            if group.empty: continue
            for scope in ("overall", era):
                for name in VERIFY_TRANSFORMS:
                    event = _transform(name, group[f"event_{name}"].to_numpy(float))
                    control = _transform(name, group[f"control_{name}"].to_numpy(float))
                    gap = np.abs(group[f"event_{name}_decile"].to_numpy(int) - group[f"control_{name}_decile"].to_numpy(int))
                    _add(states[f"{population}|{signal}|{scope}|{tier}|{name}"], event, control, gap)
    reuse = []
    for (population, signal), group in controls.groupby(["population", "signal_name"], sort=True):
        usage = group.groupby(["signal_date", "control_symbol"], sort=False).size().to_numpy(float)
        reuse.append({"year": year, "population": population, "signal_name": signal,
                      "control_rows": len(group), "unique_date_control_identities": len(usage),
                      "maximum_same_date_reuse": int(usage.max()),
                      "reuse_effective_size": float(usage.sum() ** 2 / np.sum(usage * usage))})
    return dict(states), reuse


def _assert_nested(expected: Mapping[str, Any], observed: Mapping[str, Any], label: str) -> None:
    if expected.keys() != observed.keys(): raise BalanceVerificationError(f"{label} keys differ")
    for key, value in expected.items():
        actual = observed[key]
        if isinstance(value, list):
            if len(value) != len(actual) or any(not math.isclose(float(a), float(b), rel_tol=TOLERANCE, abs_tol=TOLERANCE)
                                                for a, b in zip(value, actual)):
                raise BalanceVerificationError(f"{label} list differs: {key}")
        elif isinstance(value, float):
            if not math.isclose(value, float(actual), rel_tol=TOLERANCE, abs_tol=TOLERANCE):
                raise BalanceVerificationError(f"{label} numeric differs: {key}")
        elif value != actual: raise BalanceVerificationError(f"{label} differs: {key}")


def _merge(destination: dict[str, Any], source: Mapping[str, Any]) -> None:
    for name in ("n", "event_sum", "event_sumsq", "control_sum", "control_sumsq",
                 "difference_sum", "absolute_difference_sum"):
        destination[name] += source[name]
    destination["decile_gap_histogram"] = [a + int(b) for a, b in zip(
        destination["decile_gap_histogram"], source["decile_gap_histogram"],
    )]


def _summaries(states: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    rows = []
    for key, state in sorted(states.items()):
        population, signal, scope, tier, covariate = key.split("|"); n = int(state["n"])
        event_mean, control_mean = state["event_sum"] / n, state["control_sum"] / n
        event_var = max(0., (state["event_sumsq"] - state["event_sum"] ** 2 / n) / max(1, n - 1))
        control_var = max(0., (state["control_sumsq"] - state["control_sum"] ** 2 / n) / max(1, n - 1))
        scale = math.sqrt(.5 * (event_var + control_var)); histogram = state["decile_gap_histogram"]
        cutoff, cumulative, p95 = math.ceil(.95 * n), 0, 0
        for gap, count in enumerate(histogram):
            cumulative += count
            if cumulative >= cutoff: p95 = gap; break
        rows.append({"population": population, "signal_name": signal, "scope": scope,
                     "match_tier": tier, "covariate": covariate, "paired_control_rows": n,
                     "event_mean": event_mean, "control_mean": control_mean,
                     "standardized_mean_difference": (event_mean - control_mean) / scale if scale else 0.,
                     "variance_ratio": event_var / control_var if control_var else np.nan,
                     "mean_paired_difference": state["difference_sum"] / n,
                     "mean_absolute_paired_difference": state["absolute_difference_sum"] / n,
                     "mean_absolute_decile_gap": sum(i * count for i, count in enumerate(histogram)) / n,
                     "p95_absolute_decile_gap": p95})
    return pd.DataFrame(rows)


def _assert_frame(expected: pd.DataFrame, observed: pd.DataFrame, label: str) -> None:
    if list(expected.columns) != list(observed.columns) or len(expected) != len(observed):
        raise BalanceVerificationError(f"{label} schema/count differs")
    for column in expected:
        left, right = expected[column], observed[column]
        if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            a, b = left.to_numpy(float), right.to_numpy(float)
            if not np.array_equal(np.isnan(a), np.isnan(b)): raise BalanceVerificationError(f"{label} missingness: {column}")
            finite = np.isfinite(a) & np.isfinite(b)
            if finite.any() and np.max(np.abs(a[finite] - b[finite])) > TOLERANCE:
                raise BalanceVerificationError(f"{label} numeric differs: {column}")
        elif not left.astype(str).equals(right.astype(str)): raise BalanceVerificationError(f"{label} differs: {column}")


def execute(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    if subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=repository,
                      capture_output=True, text=True, check=True).stdout:
        raise BalanceVerificationError("clean worktree required")
    if target.TRANSFORMS != VERIFY_TRANSFORMS: raise BalanceVerificationError("transform contract differs")
    started = perf_counter(); prereg, _ = target.validate_preregistration(repository)
    root = repository / target.OUTPUT_RELATIVE; seal = base._read(root / "SEALED.json")
    if not target._valid(seal, timing=True) or seal.get("file_manifest") != target._manifest(root, target.OUTPUT_FILES):
        raise BalanceVerificationError("balance seal differs")
    frame = target.match_stage._read_causal_panel(repository)
    grouped = frame.groupby(frame.signal_date.dt.year, sort=True)
    combined: dict[str, dict[str, Any]] = defaultdict(_empty); reuse = []; digests = []
    for year in prereg["years"]:
        year_seal = base._read(repository / target.CACHE_RELATIVE / f"year-{year}" / "YEAR_SEALED.json")
        if not target._valid(year_seal, timing=True): raise BalanceVerificationError(f"year seal differs: {year}")
        if year_seal.get("outcome_columns_read") != [] or year_seal.get("post_signal_results_accessed") is not False:
            raise BalanceVerificationError(f"year outcome boundary differs: {year}")
        controls = pd.read_parquet(repository / target.match_stage.CACHE_RELATIVE / f"year-{year}" / "control-identities.parquet")
        expected_stats, expected_reuse = _reconstruct_year(year, grouped.get_group(year), controls)
        if expected_stats.keys() != year_seal["statistics"].keys(): raise BalanceVerificationError("year statistic keys differ")
        for key in expected_stats: _assert_nested(expected_stats[key], year_seal["statistics"][key], f"year {year}/{key}")
        if len(expected_reuse) != len(year_seal["reuse"]): raise BalanceVerificationError("reuse count differs")
        for expected, observed in zip(expected_reuse, year_seal["reuse"]): _assert_nested(expected, observed, "reuse")
        for key, value in expected_stats.items(): _merge(combined[key], value)
        reuse.extend(expected_reuse); digests.append(year_seal["result_digest"])
    expected_summary = _summaries(combined)
    _assert_frame(expected_summary, pd.read_parquet(root / "balance-summary.parquet"), "balance summary")
    _assert_frame(pd.DataFrame(reuse), pd.read_parquet(root / "reuse-summary.parquet"), "reuse summary")
    primary = expected_summary.loc[expected_summary.match_tier.eq("all")]
    overall = bool((primary.loc[primary.scope.eq("overall"), "standardized_mean_difference"].abs() <= .10).all())
    eras = bool((primary.loc[~primary.scope.eq("overall"), "standardized_mean_difference"].abs() <= .20).all())
    decision = base._read(root / "balance-decision.json")
    if not target._valid(decision) or decision["overall_balance_pass"] != overall \
            or decision["era_balance_pass"] != eras or decision["balance_gate_pass"] != (overall and eras):
        raise BalanceVerificationError("balance decision differs")
    if digests != seal["year_result_digests"] or seal.get("outcome_columns_read") != [] \
            or seal.get("post_signal_results_accessed") is not False:
        raise BalanceVerificationError("aggregate binding differs")
    state = {"schema_version": SCHEMA, "status": "verified", "passed": True,
             "store_result_digest": seal["result_digest"], "store_result_sha256": _sha(root / "SEALED.json"),
             "verified_balance_summary_rows": len(expected_summary), "verified_reuse_summary_rows": len(reuse),
             "balance_gate_pass": overall and eras,
             "gates": {"all_year_statistics_reconstructed": True, "all_transforms_and_smds_reconstructed": True,
                       "all_decile_histograms_reconstructed": True, "all_reuse_diagnostics_reconstructed": True,
                       "balance_decision_reconstructed": True, "outcome_boundary_remained_closed": True},
             "outcome_columns_read": [], "post_signal_results_accessed": False,
             "production_promotion_authorized": False, "elapsed_seconds": perf_counter() - started,
             "created_at": _now()}
    result = {**state, "verification_digest": stable_hash(state)}
    output = repository / target.VERIFICATION_RELATIVE
    if output.exists(): return base._read(output / "VERIFIED.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try: target.smoke._atomic_json(temporary / "VERIFIED.json", result); os.replace(temporary, output)
    except BaseException: shutil.rmtree(temporary, ignore_errors=True); raise
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--repository", type=Path, required=True)
    args = parser.parse_args(argv); print(json.dumps(execute(args.repository), indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
