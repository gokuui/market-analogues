"""Audit T14-12 covariate balance without opening any post-signal outcome."""
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

from experiments.m04r import m04r14_t14_09_outcome_smoke as smoke
from experiments.m04r import m04r14_t14_10_wf03_feasibility as base
from experiments.m04r import m04r14_t14_12_matches as match_stage


SCHEMA = "m04r14-t14-12-post-signal-balance-v1"
PREREGISTRATION_RELATIVE = Path("experiments/m04r/m04r14_t14_12_balance_v1_preregistered.json")
CACHE_RELATIVE = Path("config/data/analogues/m04r14/t14-12-post-signal-balance-v1-cache")
OUTPUT_RELATIVE = Path("config/data/analogues/m04r14/t14-12-post-signal-balance-v1")
VERIFICATION_RELATIVE = Path("config/data/analogues/m04r14/t14-12-post-signal-balance-v1-verification")
OUTPUT_FILES = ("balance-summary.parquet", "reuse-summary.parquet", "balance-decision.json")
TRANSFORMS = {
    "prior_return_63": "identity",
    "prior_volatility_20": "log1p",
    "prior_close": "log",
    "prior_median_dollar_volume_20": "log1p",
}
TIERS = ("exact_all_four_deciles", "within_one_bucket_all_four", "same_date_unmatched")
RUNTIME_FILES = (
    "experiments/m04r/m04r14_t14_12_balance.py",
    "experiments/m04r/verify_m04r14_t14_12_balance.py",
    "experiments/m04r/m04r14_t14_12_matches.py",
    "config/m04r14-t14-12-post-signal-contract.json", "pyproject.toml",
)


class BalanceStudyError(RuntimeError): pass


def _now() -> str: return datetime.now(timezone.utc).isoformat()


def _git(repository: Path, *args: str, raw: bool = False) -> str | bytes:
    result = subprocess.run(["git", *args], cwd=repository, capture_output=True, text=not raw, check=False)
    if result.returncode:
        error = result.stderr if not raw else result.stderr.decode(errors="replace")
        raise BalanceStudyError(error.strip())
    return result.stdout if raw else result.stdout.strip()


def _sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20): digest.update(block)
    return digest.hexdigest()


def _seal(value: Mapping[str, Any], key: str = "result_digest", *, timing: bool = False) -> dict[str, Any]:
    omitted = {"elapsed_seconds"} if timing else set(); result = dict(value)
    result[key] = stable_hash({name: item for name, item in result.items() if name not in omitted})
    result["created_at"] = _now(); return result


def _valid(value: Mapping[str, Any], key: str = "result_digest", *, timing: bool = False) -> bool:
    omitted = {key, "created_at"} | ({"elapsed_seconds"} if timing else set())
    return value.get(key) == stable_hash({name: item for name, item in value.items() if name not in omitted})


def _manifest(root: Path, names: Sequence[str]) -> list[dict[str, Any]]:
    return [{"path": name, "bytes": (root / name).stat().st_size, "sha256": _sha(root / name)} for name in names]


def _verified_matches(repository: Path) -> dict[str, Any]:
    store_path = repository / match_stage.OUTPUT_RELATIVE / "SEALED.json"
    verify_path = repository / match_stage.VERIFICATION_RELATIVE / "VERIFIED.json"
    store, receipt = base._read(store_path), base._read(verify_path)
    if not match_stage._valid(store, timing=True) or receipt.get("passed") is not True \
            or receipt.get("store_result_digest") != store.get("result_digest") \
            or receipt.get("outcome_columns_read") != [] or receipt.get("inference_accessed") is not False:
        raise BalanceStudyError("verified match boundary differs")
    return {
        "match_store_result_digest": store["result_digest"], "match_store_sha256": _sha(store_path),
        "match_verification_digest": receipt["verification_digest"], "match_verification_sha256": _sha(verify_path),
        "event_match_rows": int(store["event_match_rows"]),
        "control_identity_rows": int(store["control_identity_rows"]),
    }


def _runtime_manifest(repository: Path, head: str) -> dict[str, str]:
    tracked = set(str(_git(repository, "ls-tree", "-r", "--name-only", head)).splitlines())
    if any(name not in tracked for name in RUNTIME_FILES): raise BalanceStudyError("runtime file is uncommitted")
    return {name: sha256(_git(repository, "show", f"{head}:{name}", raw=True)).hexdigest() for name in RUNTIME_FILES}


def build_preregistration(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    if _git(repository, "status", "--porcelain", "--untracked-files=all"): raise BalanceStudyError("clean worktree required")
    if (repository / CACHE_RELATIVE).exists() or (repository / OUTPUT_RELATIVE).exists():
        raise BalanceStudyError("balance namespaces must be absent")
    matches = _verified_matches(repository); contract = match_stage.panel_stage._contract(repository)
    h0 = str(_git(repository, "rev-parse", "HEAD"))
    return _seal({
        "schema_version": SCHEMA, "status": "frozen_before_real_covariate_balance_access",
        "implementation_h0": h0, "runtime_files": _runtime_manifest(repository, h0),
        "verified_matches": matches, "contract_digest": contract["contract_digest"],
        "years": base._read(repository / match_stage.PREREGISTRATION_RELATIVE)["years"],
        "covariate_transforms": TRANSFORMS,
        "standardized_difference": "transformed_event_minus_control_mean_divided_by_sqrt_half_sample_variance_sum",
        "paired_diagnostics": "mean_absolute_transformed_difference_and_decile_gap_histogram",
        "overall_abs_smd_limit": 0.10, "each_era_abs_smd_limit": 0.20,
        "balance_gate": "all_four_covariates_pass_overall_and_in_each_frozen_era_for_each_population_and_signal",
        "tier_and_control_reuse_diagnostics_are_descriptive": True,
        "outcome_columns_read": [], "post_signal_results_accessed": False,
        "failed_gate_requires_outcome_blind_rematching_before_inference": True,
        "production_promotion_authorized": False,
    }, "preregistration_digest")


def _sole_child(repository: Path, raw: bytes, h0: str) -> str:
    found = []
    for line in str(_git(repository, "rev-list", "--all", "--children")).splitlines():
        values = line.split()
        if not values or values[0] != h0: continue
        for child in values[1:]:
            parents = str(_git(repository, "rev-list", "--parents", "-n", "1", child)).split()
            changed = str(_git(repository, "diff-tree", "--no-commit-id", "--name-only", "-r", child)).splitlines()
            if parents == [child, h0] and changed == [PREREGISTRATION_RELATIVE.as_posix()] \
                    and _git(repository, "show", f"{child}:{PREREGISTRATION_RELATIVE}", raw=True) == raw:
                found.append(child)
    if len(set(found)) != 1: raise BalanceStudyError("expected exact preregistration-only child")
    return found[0]


def validate_preregistration(repository: Path) -> tuple[dict[str, Any], str]:
    if _git(repository, "status", "--porcelain", "--untracked-files=all"): raise BalanceStudyError("clean worktree required")
    path = repository / PREREGISTRATION_RELATIVE; raw = path.read_bytes(); prereg = base._read(path)
    if prereg.get("schema_version") != SCHEMA or not _valid(prereg, "preregistration_digest"):
        raise BalanceStudyError("balance preregistration differs")
    h1 = _sole_child(repository, raw, str(prereg["implementation_h0"]))
    if subprocess.run(["git", "merge-base", "--is-ancestor", h1, "HEAD"], cwd=repository).returncode:
        raise BalanceStudyError("HEAD does not descend from preregistration")
    if prereg["verified_matches"] != _verified_matches(repository): raise BalanceStudyError("verified matches drifted")
    for name, expected in prereg["runtime_files"].items():
        if _sha(repository / name) != expected: raise BalanceStudyError(f"runtime drifted: {name}")
    return prereg, h1


def _transform(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all(): raise BalanceStudyError(f"non-finite covariate: {name}")
    rule = TRANSFORMS[name]
    if rule == "identity": return array
    if rule == "log":
        if (array <= 0).any(): raise BalanceStudyError(f"nonpositive log covariate: {name}")
        return np.log(array)
    if rule == "log1p":
        if (array < 0).any(): raise BalanceStudyError(f"negative log1p covariate: {name}")
        return np.log1p(array)
    raise BalanceStudyError(f"unknown transform: {rule}")


def _empty_stats() -> dict[str, Any]:
    return {"n": 0, "event_sum": 0., "event_sumsq": 0., "control_sum": 0.,
            "control_sumsq": 0., "difference_sum": 0., "absolute_difference_sum": 0.,
            "decile_gap_histogram": [0] * 10}


def _add(stats: dict[str, Any], event: np.ndarray, control: np.ndarray, gaps: np.ndarray) -> None:
    difference = event - control; stats["n"] += len(event)
    stats["event_sum"] += float(event.sum()); stats["event_sumsq"] += float(np.dot(event, event))
    stats["control_sum"] += float(control.sum()); stats["control_sumsq"] += float(np.dot(control, control))
    stats["difference_sum"] += float(difference.sum())
    stats["absolute_difference_sum"] += float(np.abs(difference).sum())
    histogram = np.bincount(gaps.astype(np.int64), minlength=10)
    stats["decile_gap_histogram"] = [a + int(b) for a, b in zip(stats["decile_gap_histogram"], histogram)]


def _merge_stats(target: dict[str, Any], source: Mapping[str, Any]) -> None:
    for name in ("n", "event_sum", "event_sumsq", "control_sum", "control_sumsq",
                 "difference_sum", "absolute_difference_sum"):
        target[name] += source[name]
    target["decile_gap_histogram"] = [a + int(b) for a, b in zip(
        target["decile_gap_histogram"], source["decile_gap_histogram"],
    )]


def _era(year: int) -> str:
    if year <= 2007: return "era_2000_2007"
    if year <= 2012: return "era_2008_2012"
    if year <= 2017: return "era_2013_2017"
    if year <= 2022: return "era_2018_2022"
    return "era_2023_2026"


def _year_stats(year: int, panel: pd.DataFrame, controls: pd.DataFrame) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    features = panel[["signal_date", "symbol", *TRANSFORMS]].copy()
    if features.duplicated(["signal_date", "symbol"]).any(): raise BalanceStudyError("duplicate feature identity")
    joined = controls.merge(features, left_on=["signal_date", "event_symbol"],
                            right_on=["signal_date", "symbol"], how="left", validate="many_to_one")
    joined = joined.rename(columns={name: f"event_{name}" for name in TRANSFORMS}).drop(columns="symbol")
    joined = joined.merge(features, left_on=["signal_date", "control_symbol"],
                          right_on=["signal_date", "symbol"], how="left", validate="many_to_one")
    joined = joined.rename(columns={name: f"control_{name}" for name in TRANSFORMS}).drop(columns="symbol")
    if joined[[f"event_{name}" for name in TRANSFORMS] + [f"control_{name}" for name in TRANSFORMS]].isna().any().any():
        raise BalanceStudyError(f"control identity absent from causal panel: {year}")
    stats: dict[str, dict[str, Any]] = defaultdict(_empty_stats); era = _era(year)
    for (population, signal), base_group in joined.groupby(["population", "signal_name"], sort=True):
        for tier in ("all", *TIERS):
            group = base_group if tier == "all" else base_group.loc[base_group.match_tier == tier]
            if group.empty: continue
            for scope in ("overall", era):
                for name in TRANSFORMS:
                    event = _transform(name, group[f"event_{name}"].to_numpy(float))
                    control = _transform(name, group[f"control_{name}"].to_numpy(float))
                    gaps = np.abs(group[f"event_{name}_decile"].to_numpy(int) - group[f"control_{name}_decile"].to_numpy(int))
                    _add(stats[f"{population}|{signal}|{scope}|{tier}|{name}"], event, control, gaps)
    reuse_rows = []
    for (population, signal), group in controls.groupby(["population", "signal_name"], sort=True):
        usage = group.groupby(["signal_date", "control_symbol"], sort=False).size().to_numpy(float)
        reuse_rows.append({
            "year": year, "population": population, "signal_name": signal,
            "control_rows": len(group), "unique_date_control_identities": len(usage),
            "maximum_same_date_reuse": int(usage.max()),
            "reuse_effective_size": float(usage.sum() ** 2 / np.dot(usage, usage)),
        })
    return dict(stats), reuse_rows


def _write_year(year: int, panel: pd.DataFrame, cache: Path) -> dict[str, Any]:
    final = cache / f"year-{year}"; path = final / "YEAR_SEALED.json"
    if path.exists():
        value = base._read(path)
        if not _valid(value, timing=True): raise BalanceStudyError(f"year checkpoint differs: {year}")
        return value
    started = perf_counter()
    # Resolve from the repository-independent sibling cache layout.
    controls_path = cache.parent / match_stage.CACHE_RELATIVE.name / f"year-{year}" / "control-identities.parquet"
    if not controls_path.exists(): raise BalanceStudyError(f"matched controls absent: {year}")
    controls = pd.read_parquet(controls_path); stats, reuse = _year_stats(year, panel, controls)
    state = {
        "schema_version": SCHEMA, "status": "year_sealed", "passed": True, "year": year,
        "panel_rows": len(panel), "control_identity_rows": len(controls), "statistics": stats,
        "reuse": reuse, "outcome_columns_read": [], "post_signal_results_accessed": False,
        "elapsed_seconds": perf_counter() - started,
    }
    seal = _seal(state, timing=True); final.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".year-{year}.", dir=final.parent))
    try:
        smoke._atomic_json(temporary / "YEAR_SEALED.json", seal); os.replace(temporary, final)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True); raise
    return seal


def _summary_rows(stats: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for key, value in sorted(stats.items()):
        population, signal, scope, tier, covariate = key.split("|"); n = int(value["n"])
        event_mean = value["event_sum"] / n; control_mean = value["control_sum"] / n
        event_var = max(0., (value["event_sumsq"] - value["event_sum"] ** 2 / n) / max(1, n - 1))
        control_var = max(0., (value["control_sumsq"] - value["control_sum"] ** 2 / n) / max(1, n - 1))
        pooled = math.sqrt((event_var + control_var) / 2)
        histogram = value["decile_gap_histogram"]; threshold = math.ceil(.95 * n); cumulative = 0; p95 = 0
        for gap, count in enumerate(histogram):
            cumulative += count
            if cumulative >= threshold: p95 = gap; break
        rows.append({
            "population": population, "signal_name": signal, "scope": scope, "match_tier": tier,
            "covariate": covariate, "paired_control_rows": n, "event_mean": event_mean,
            "control_mean": control_mean, "standardized_mean_difference": (event_mean - control_mean) / pooled if pooled else 0.,
            "variance_ratio": event_var / control_var if control_var else np.nan,
            "mean_paired_difference": value["difference_sum"] / n,
            "mean_absolute_paired_difference": value["absolute_difference_sum"] / n,
            "mean_absolute_decile_gap": sum(i * count for i, count in enumerate(histogram)) / n,
            "p95_absolute_decile_gap": p95,
        })
    return rows


def execute(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True); prereg, h1 = validate_preregistration(repository)
    root = repository / OUTPUT_RELATIVE
    if root.exists():
        seal = base._read(root / "SEALED.json")
        if not _valid(seal, timing=True) or seal.get("file_manifest") != _manifest(root, OUTPUT_FILES):
            raise BalanceStudyError("existing balance output differs")
        return seal
    started = perf_counter(); frame = match_stage._read_causal_panel(repository)
    grouped_years = frame.groupby(frame.signal_date.dt.year, sort=True)
    cache = repository / CACHE_RELATIVE; cache.mkdir(parents=True, exist_ok=True)
    seals = [_write_year(year, grouped_years.get_group(year), cache) for year in prereg["years"]]
    combined: dict[str, dict[str, Any]] = defaultdict(_empty_stats); reuse = []
    for seal in seals:
        for key, value in seal["statistics"].items(): _merge_stats(combined[key], value)
        reuse.extend(seal["reuse"])
    summary = pd.DataFrame(_summary_rows(combined))
    primary = summary.loc[summary.match_tier.eq("all")]
    overall_pass = bool((primary.loc[primary.scope.eq("overall"), "standardized_mean_difference"].abs() <= .10).all())
    era_pass = bool((primary.loc[~primary.scope.eq("overall"), "standardized_mean_difference"].abs() <= .20).all())
    decision = _seal({
        "schema_version": SCHEMA, "status": "balance_decision", "overall_abs_smd_limit": .10,
        "each_era_abs_smd_limit": .20, "overall_balance_pass": overall_pass,
        "era_balance_pass": era_pass, "balance_gate_pass": overall_pass and era_pass,
        "outcome_blind_rematching_required": not (overall_pass and era_pass),
        "post_signal_inference_authorized": overall_pass and era_pass,
        "outcome_columns_read": [], "production_promotion_authorized": False,
    })
    temporary = Path(tempfile.mkdtemp(prefix=f".{root.name}.", dir=root.parent))
    smoke._atomic_parquet(temporary / OUTPUT_FILES[0], summary)
    smoke._atomic_parquet(temporary / OUTPUT_FILES[1], pd.DataFrame(reuse))
    smoke._atomic_json(temporary / OUTPUT_FILES[2], decision)
    state = {
        "schema_version": SCHEMA, "status": "sealed", "passed": True,
        "preregistration_h1": h1, "preregistration_digest": prereg["preregistration_digest"],
        "match_store_result_digest": prereg["verified_matches"]["match_store_result_digest"],
        "year_result_digests": [value["result_digest"] for value in seals],
        "balance_summary_rows": len(summary), "reuse_summary_rows": len(reuse),
        "balance_decision_result_digest": decision["result_digest"],
        "file_manifest": _manifest(temporary, OUTPUT_FILES), "outcome_columns_read": [],
        "post_signal_results_accessed": False, "elapsed_seconds": perf_counter() - started,
        "independent_verification_authorized": True, "production_promotion_authorized": False,
    }
    seal = _seal(state, timing=True); smoke._atomic_json(temporary / "SEALED.json", seal)
    try: os.replace(temporary, root)
    except BaseException: shutil.rmtree(temporary, ignore_errors=True); raise
    return seal


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="command", required=True)
    for command in ("preregister", "run"):
        child = sub.add_parser(command); child.add_argument("--repository", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "preregister":
        value = build_preregistration(args.repository); smoke._atomic_json(args.repository / PREREGISTRATION_RELATIVE, value)
    else: value = execute(args.repository)
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)); return 0


if __name__ == "__main__": raise SystemExit(main())
