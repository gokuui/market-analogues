"""Seal final T14-12 weight-4 balance and reproduce the selected bake-off rows."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
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
from experiments.m04r import m04r14_t14_10_wf03_feasibility as io
from experiments.m04r import m04r14_t14_12_balance as base_balance
from experiments.m04r import m04r14_t14_12_final_matches as final_matches
from experiments.m04r import m04r14_t14_12_weight_bakeoff as bakeoff


SCHEMA = "m04r14-t14-12-post-signal-final-balance-v4"
PREREGISTRATION_RELATIVE = Path("experiments/m04r/m04r14_t14_12_final_balance_v4_preregistered.json")
CACHE_RELATIVE = Path("config/data/analogues/m04r14/t14-12-post-signal-final-balance-v4-cache")
OUTPUT_RELATIVE = Path("config/data/analogues/m04r14/t14-12-post-signal-final-balance-v4")
VERIFICATION_RELATIVE = Path("config/data/analogues/m04r14/t14-12-post-signal-final-balance-v4-verification")
OUTPUT_FILES = base_balance.OUTPUT_FILES
RUNTIME_FILES = (
    "experiments/m04r/m04r14_t14_12_final_balance.py",
    "experiments/m04r/verify_m04r14_t14_12_final_balance.py",
    "experiments/m04r/m04r14_t14_12_balance.py",
    "experiments/m04r/verify_m04r14_t14_12_balance.py",
    "experiments/m04r/m04r14_t14_12_final_matches.py",
    "config/m04r14-t14-12-post-signal-contract.json", "pyproject.toml",
)


class FinalBalanceError(RuntimeError): pass


def _now() -> str: return datetime.now(timezone.utc).isoformat()


def _git(repository: Path, *args: str, raw: bool = False) -> str | bytes:
    result = subprocess.run(["git", *args], cwd=repository, capture_output=True, text=not raw, check=False)
    if result.returncode:
        error = result.stderr if not raw else result.stderr.decode(errors="replace")
        raise FinalBalanceError(error.strip())
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


def _verified_final_matches(repository: Path) -> dict[str, Any]:
    store_path = repository / final_matches.OUTPUT_RELATIVE / "SEALED.json"
    verify_path = repository / final_matches.VERIFICATION_RELATIVE / "VERIFIED.json"
    store, receipt = io._read(store_path), io._read(verify_path)
    receipt_state = {name: item for name, item in receipt.items() if name != "verification_digest"}
    if not final_matches._valid(store, timing=True) or receipt.get("verification_digest") != stable_hash(receipt_state) \
            or receipt.get("store_result_digest") != store.get("result_digest") \
            or receipt.get("passed") is not True or receipt.get("selected_volatility_weight") != 4 \
            or receipt.get("outcome_columns_read") != []:
        raise FinalBalanceError("verified final matches differ")
    return {"match_store_result_digest": store["result_digest"], "match_store_sha256": _sha(store_path),
            "match_verification_digest": receipt["verification_digest"],
            "match_verification_sha256": _sha(verify_path), "event_match_rows": store["event_match_rows"],
            "control_identity_rows": store["control_identity_rows"]}


def _selected_bakeoff(repository: Path) -> dict[str, Any]:
    bound = final_matches._verified_selection(repository)
    path = repository / bakeoff.OUTPUT_RELATIVE / "balance-bakeoff.parquet"
    return {**bound, "balance_bakeoff_sha256": _sha(path)}


def _runtime_manifest(repository: Path, head: str) -> dict[str, str]:
    tracked = set(str(_git(repository, "ls-tree", "-r", "--name-only", head)).splitlines())
    if any(name not in tracked for name in RUNTIME_FILES): raise FinalBalanceError("runtime file is uncommitted")
    return {name: sha256(_git(repository, "show", f"{head}:{name}", raw=True)).hexdigest() for name in RUNTIME_FILES}


def build_preregistration(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    if _git(repository, "status", "--porcelain", "--untracked-files=all"): raise FinalBalanceError("clean worktree required")
    if (repository / CACHE_RELATIVE).exists() or (repository / OUTPUT_RELATIVE).exists(): raise FinalBalanceError("final balance namespaces exist")
    prior = io._read(repository / base_balance.PREREGISTRATION_RELATIVE); h0 = str(_git(repository, "rev-parse", "HEAD"))
    return _seal({"schema_version": SCHEMA, "status": "frozen_before_final_balance_access",
        "implementation_h0": h0, "runtime_files": _runtime_manifest(repository, h0),
        "verified_final_matches": _verified_final_matches(repository), "verified_selected_bakeoff": _selected_bakeoff(repository),
        "years": io._read(repository / final_matches.PREREGISTRATION_RELATIVE)["years"],
        "covariate_transforms": prior["covariate_transforms"],
        "overall_abs_smd_limit": prior["overall_abs_smd_limit"], "each_era_abs_smd_limit": prior["each_era_abs_smd_limit"],
        "selected_bakeoff_rows_must_reproduce_exactly": True, "decision_rule_identical_to_V1": True,
        "outcome_columns_read": [], "post_signal_results_accessed": False,
        "production_promotion_authorized": False}, "preregistration_digest")


def _sole_child(repository: Path, raw: bytes, h0: str) -> str:
    found = []
    for line in str(_git(repository, "rev-list", "--all", "--children")).splitlines():
        values = line.split()
        if not values or values[0] != h0: continue
        for child in values[1:]:
            parents = str(_git(repository, "rev-list", "--parents", "-n", "1", child)).split()
            changed = str(_git(repository, "diff-tree", "--no-commit-id", "--name-only", "-r", child)).splitlines()
            if parents == [child, h0] and changed == [PREREGISTRATION_RELATIVE.as_posix()] \
                    and _git(repository, "show", f"{child}:{PREREGISTRATION_RELATIVE}", raw=True) == raw: found.append(child)
    if len(set(found)) != 1: raise FinalBalanceError("expected exact preregistration-only child")
    return found[0]


def validate_preregistration(repository: Path) -> tuple[dict[str, Any], str]:
    if _git(repository, "status", "--porcelain", "--untracked-files=all"): raise FinalBalanceError("clean worktree required")
    path = repository / PREREGISTRATION_RELATIVE; raw = path.read_bytes(); prereg = io._read(path)
    if prereg.get("schema_version") != SCHEMA or not _valid(prereg, "preregistration_digest"): raise FinalBalanceError("final balance prereg differs")
    h1 = _sole_child(repository, raw, str(prereg["implementation_h0"]))
    if subprocess.run(["git", "merge-base", "--is-ancestor", h1, "HEAD"], cwd=repository).returncode: raise FinalBalanceError("H1 is not ancestor")
    if prereg["verified_final_matches"] != _verified_final_matches(repository) \
            or prereg["verified_selected_bakeoff"] != _selected_bakeoff(repository): raise FinalBalanceError("final balance inputs drifted")
    for name, expected in prereg["runtime_files"].items():
        if _sha(repository / name) != expected: raise FinalBalanceError(f"runtime drifted: {name}")
    return prereg, h1


def _write_year(year: int, panel: pd.DataFrame, cache: Path, repository: Path) -> dict[str, Any]:
    final = cache / f"year-{year}"; path = final / "YEAR_SEALED.json"
    if path.exists():
        value = io._read(path)
        if not _valid(value, timing=True): raise FinalBalanceError(f"final balance year differs: {year}")
        return value
    started = perf_counter(); controls = pd.read_parquet(repository / final_matches.CACHE_RELATIVE / f"year-{year}" / "control-identities.parquet")
    stats, reuse = _year_stats_in_bakeoff_order(year, panel, controls)
    state = {"schema_version": SCHEMA, "status": "year_sealed", "passed": True, "year": year,
        "panel_rows": len(panel), "control_identity_rows": len(controls), "statistics": stats, "reuse": reuse,
        "outcome_columns_read": [], "post_signal_results_accessed": False, "elapsed_seconds": perf_counter() - started}
    seal = _seal(state, timing=True); final.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".year-{year}.", dir=final.parent))
    try: smoke._atomic_json(temporary / "YEAR_SEALED.json", seal); os.replace(temporary, final)
    except BaseException: shutil.rmtree(temporary, ignore_errors=True); raise
    return seal


def _year_stats_in_bakeoff_order(
    year: int, panel: pd.DataFrame, controls: pd.DataFrame,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Replay the date/population/signal accumulation order used by the frozen bake-off."""
    stats: dict[str, dict[str, Any]] = defaultdict(base_balance._empty_stats); era = base_balance._era(year)
    grouped_controls = {key: group for key, group in controls.groupby(
        ["signal_date", "population", "signal_name"], sort=False,
    )}
    for signal_date, group in panel.groupby("signal_date", sort=True):
        for population in ("broad", "investable"):
            current = group if population == "broad" else group.loc[group.investable]
            if current.empty: continue
            current = current.reset_index(drop=True); names = current.symbol.astype(str).to_numpy(object)
            positions = {str(name): index for index, name in enumerate(names)}
            transformed = {name: base_balance._transform(name, current[name].to_numpy(float))
                           for name in base_balance.TRANSFORMS}
            for signal_name in final_matches.v1.SIGNALS:
                selected = grouped_controls.get((pd.Timestamp(signal_date), population, signal_name))
                if selected is None or selected.empty: continue
                event_positions = np.fromiter((positions[str(value)] for value in selected.event_symbol), dtype=np.int64)
                control_positions = np.fromiter((positions[str(value)] for value in selected.control_symbol), dtype=np.int64)
                for name in base_balance.TRANSFORMS:
                    event_values = transformed[name][event_positions]; control_values = transformed[name][control_positions]
                    gaps = np.abs(selected[f"event_{name}_decile"].to_numpy(int)
                                  - selected[f"control_{name}_decile"].to_numpy(int))
                    for scope in ("overall", era):
                        base_balance._add(stats[f"{population}|{signal_name}|{scope}|all|{name}"],
                                          event_values, control_values, gaps)
    reuse_rows = []
    for (population, signal), group in controls.groupby(["population", "signal_name"], sort=True):
        usage = group.groupby(["signal_date", "control_symbol"], sort=False).size().to_numpy(float)
        reuse_rows.append({"year": year, "population": population, "signal_name": signal,
            "control_rows": len(group), "unique_date_control_identities": len(usage),
            "maximum_same_date_reuse": int(usage.max()),
            "reuse_effective_size": float(usage.sum() ** 2 / np.dot(usage, usage))})
    return dict(stats), reuse_rows


def _selected_bakeoff_frame(repository: Path) -> pd.DataFrame:
    frame = pd.read_parquet(repository / bakeoff.OUTPUT_RELATIVE / "balance-bakeoff.parquet")
    selected = frame.loc[frame.volatility_weight.eq(4)].drop(columns="volatility_weight").reset_index(drop=True)
    if len(selected) != 96: raise FinalBalanceError("selected bake-off row count differs")
    return selected


def execute(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True); prereg, h1 = validate_preregistration(repository); root = repository / OUTPUT_RELATIVE
    if root.exists():
        seal = io._read(root / "SEALED.json")
        if not _valid(seal, timing=True) or seal.get("file_manifest") != _manifest(root, OUTPUT_FILES): raise FinalBalanceError("final balance output differs")
        return seal
    started = perf_counter(); frame = final_matches.v1._read_causal_panel(repository); grouped = frame.groupby(frame.signal_date.dt.year, sort=True)
    cache = repository / CACHE_RELATIVE; cache.mkdir(parents=True, exist_ok=True)
    seals = [_write_year(year, grouped.get_group(year), cache, repository) for year in prereg["years"]]
    combined = defaultdict(base_balance._empty_stats); reuse = []
    for seal in seals:
        for key, value in seal["statistics"].items(): base_balance._merge_stats(combined[key], value)
        reuse.extend(seal["reuse"])
    summary = pd.DataFrame(base_balance._summary_rows(combined)).reset_index(drop=True)
    try: pd.testing.assert_frame_equal(summary, _selected_bakeoff_frame(repository), check_exact=True)
    except AssertionError as error: raise FinalBalanceError("final identities do not exactly reproduce weight-4 bake-off rows") from error
    primary = summary.loc[summary.match_tier.eq("all")]
    overall = bool((primary.loc[primary.scope.eq("overall"), "standardized_mean_difference"].abs() <= .10).all())
    eras = bool((primary.loc[~primary.scope.eq("overall"), "standardized_mean_difference"].abs() <= .20).all())
    decision = _seal({"schema_version": SCHEMA, "status": "balance_decision", "overall_abs_smd_limit": .10,
        "each_era_abs_smd_limit": .20, "overall_balance_pass": overall, "era_balance_pass": eras,
        "balance_gate_pass": overall and eras, "selected_bakeoff_rows_exactly_reproduced": True,
        "outcome_blind_rematching_required": not (overall and eras), "post_signal_outcome_join_authorized": overall and eras,
        "post_signal_inference_authorized": False, "outcome_columns_read": [], "production_promotion_authorized": False})
    temporary = Path(tempfile.mkdtemp(prefix=f".{root.name}.", dir=root.parent))
    smoke._atomic_parquet(temporary / OUTPUT_FILES[0], summary); smoke._atomic_parquet(temporary / OUTPUT_FILES[1], pd.DataFrame(reuse))
    smoke._atomic_json(temporary / OUTPUT_FILES[2], decision)
    state = {"schema_version": SCHEMA, "status": "sealed", "passed": True, "preregistration_h1": h1,
        "preregistration_digest": prereg["preregistration_digest"],
        "match_store_result_digest": prereg["verified_final_matches"]["match_store_result_digest"],
        "selected_bakeoff_store_result_digest": prereg["verified_selected_bakeoff"]["bakeoff_store_result_digest"],
        "year_result_digests": [value["result_digest"] for value in seals], "balance_summary_rows": len(summary),
        "reuse_summary_rows": len(reuse), "balance_decision_result_digest": decision["result_digest"],
        "selected_bakeoff_rows_exactly_reproduced": True, "file_manifest": _manifest(temporary, OUTPUT_FILES),
        "outcome_columns_read": [], "post_signal_results_accessed": False, "elapsed_seconds": perf_counter() - started,
        "independent_verification_authorized": True, "production_promotion_authorized": False}
    seal = _seal(state, timing=True); smoke._atomic_json(temporary / "SEALED.json", seal)
    try: os.replace(temporary, root)
    except BaseException: shutil.rmtree(temporary, ignore_errors=True); raise
    return seal


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="command", required=True)
    for command in ("preregister", "run"):
        child = sub.add_parser(command); child.add_argument("--repository", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "preregister": value = build_preregistration(args.repository); smoke._atomic_json(args.repository / PREREGISTRATION_RELATIVE, value)
    else: value = execute(args.repository)
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)); return 0


if __name__ == "__main__": raise SystemExit(main())
