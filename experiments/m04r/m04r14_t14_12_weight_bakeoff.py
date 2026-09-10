"""Outcome-blind full-universe volatility-weight bake-off for T14-12 matching."""
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

from market_analogues.post_signal_matching import event_id
from market_analogues.post_signal_rank_matching import percentile_rank_matrix, select_rank_nearest_batch
from market_analogues.types import stable_hash

from experiments.m04r import m04r14_t14_09_outcome_smoke as smoke
from experiments.m04r import m04r14_t14_10_wf03_feasibility as io
from experiments.m04r import m04r14_t14_12_balance as balance
from experiments.m04r import m04r14_t14_12_rank_matches as rank_matches
from experiments.m04r import m04r14_t14_12_rank_balance as rank_balance


SCHEMA = "m04r14-t14-12-volatility-weight-bakeoff-v1"
WEIGHTS = (1, 2, 4, 8, 16)
PREREGISTRATION_RELATIVE = Path("experiments/m04r/m04r14_t14_12_weight_bakeoff_v1_preregistered.json")
CACHE_RELATIVE = Path("config/data/analogues/m04r14/t14-12-volatility-weight-bakeoff-v1-cache")
OUTPUT_RELATIVE = Path("config/data/analogues/m04r14/t14-12-volatility-weight-bakeoff-v1")
VERIFICATION_RELATIVE = Path("config/data/analogues/m04r14/t14-12-volatility-weight-bakeoff-v1-verification")
OUTPUT_FILES = ("balance-bakeoff.parquet", "selection-decision.json")
RUNTIME_FILES = (
    "experiments/m04r/m04r14_t14_12_weight_bakeoff.py",
    "experiments/m04r/verify_m04r14_t14_12_weight_bakeoff.py",
    "experiments/m04r/m04r14_t14_12_balance.py",
    "experiments/m04r/m04r14_t14_12_rank_matches.py",
    "experiments/m04r/verify_m04r14_t14_12_rank_matches.py",
    "src/market_analogues/post_signal_rank_matching.py",
    "config/m04r14-t14-12-post-signal-contract.json", "pyproject.toml",
)


class WeightBakeoffError(RuntimeError): pass


def _now() -> str: return datetime.now(timezone.utc).isoformat()


def _git(repository: Path, *args: str, raw: bool = False) -> str | bytes:
    result = subprocess.run(["git", *args], cwd=repository, capture_output=True, text=not raw, check=False)
    if result.returncode:
        error = result.stderr if not raw else result.stderr.decode(errors="replace")
        raise WeightBakeoffError(error.strip())
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


def _verified_v2_failure(repository: Path) -> dict[str, Any]:
    store_path = repository / rank_balance.OUTPUT_RELATIVE / "SEALED.json"
    verify_path = repository / rank_balance.VERIFICATION_RELATIVE / "VERIFIED.json"
    decision_path = repository / rank_balance.OUTPUT_RELATIVE / "balance-decision.json"
    store, receipt, decision = io._read(store_path), io._read(verify_path), io._read(decision_path)
    receipt_state = {name: item for name, item in receipt.items() if name != "verification_digest"}
    if not rank_balance._valid(store, timing=True) or receipt.get("verification_digest") != stable_hash(receipt_state) \
            or receipt.get("store_result_digest") != store.get("result_digest") or not rank_balance._valid(decision) \
            or receipt.get("balance_gate_pass") is not False or decision.get("outcome_blind_rematching_required") is not True \
            or receipt.get("outcome_columns_read") != []:
        raise WeightBakeoffError("verified V2 balance failure differs")
    return {"v2_balance_store_digest": store["result_digest"], "v2_balance_store_sha256": _sha(store_path),
            "v2_balance_verification_digest": receipt["verification_digest"], "v2_balance_verification_sha256": _sha(verify_path),
            "v2_balance_decision_digest": decision["result_digest"], "v2_balance_decision_sha256": _sha(decision_path)}


def _runtime_manifest(repository: Path, head: str) -> dict[str, str]:
    tracked = set(str(_git(repository, "ls-tree", "-r", "--name-only", head)).splitlines())
    if any(name not in tracked for name in RUNTIME_FILES): raise WeightBakeoffError("runtime file is uncommitted")
    return {name: sha256(_git(repository, "show", f"{head}:{name}", raw=True)).hexdigest() for name in RUNTIME_FILES}


def build_preregistration(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    if _git(repository, "status", "--porcelain", "--untracked-files=all"): raise WeightBakeoffError("clean worktree required")
    if (repository / CACHE_RELATIVE).exists() or (repository / OUTPUT_RELATIVE).exists(): raise WeightBakeoffError("bake-off namespaces exist")
    h0 = str(_git(repository, "rev-parse", "HEAD")); v2 = io._read(repository / rank_matches.PREREGISTRATION_RELATIVE)
    return _seal({"schema_version": SCHEMA, "status": "frozen_before_weight_balance_access",
        "implementation_h0": h0, "runtime_files": _runtime_manifest(repository, h0),
        "verified_v2_failure": _verified_v2_failure(repository), "contract_digest": v2["contract_digest"],
        "years": v2["years"], "volatility_coordinate_weights": list(WEIGHTS),
        "other_coordinate_weights": [1, 1, 1], "controls_per_event": 5,
        "candidate_and_tie_rules_identical_to_V2": True,
        "overall_abs_smd_limit": .10, "each_era_abs_smd_limit": .20, "full_match_coverage_limit": .90,
        "selection_rule": "smallest_weight_passing_coverage_and_all_overall_and_era_balance_cells",
        "no_passing_weight": "stop_without_outcome_access", "outcome_columns_read": [],
        "post_signal_results_accessed": False, "production_promotion_authorized": False}, "preregistration_digest")


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
    if len(set(found)) != 1: raise WeightBakeoffError("expected exact preregistration-only child")
    return found[0]


def validate_preregistration(repository: Path) -> tuple[dict[str, Any], str]:
    if _git(repository, "status", "--porcelain", "--untracked-files=all"): raise WeightBakeoffError("clean worktree required")
    path = repository / PREREGISTRATION_RELATIVE; raw = path.read_bytes(); prereg = io._read(path)
    if prereg.get("schema_version") != SCHEMA or not _valid(prereg, "preregistration_digest"): raise WeightBakeoffError("bake-off prereg differs")
    h1 = _sole_child(repository, raw, str(prereg["implementation_h0"]))
    if subprocess.run(["git", "merge-base", "--is-ancestor", h1, "HEAD"], cwd=repository).returncode: raise WeightBakeoffError("H1 not ancestor")
    if prereg["verified_v2_failure"] != _verified_v2_failure(repository): raise WeightBakeoffError("V2 failure drifted")
    for name, expected in prereg["runtime_files"].items():
        if _sha(repository / name) != expected: raise WeightBakeoffError(f"runtime drifted: {name}")
    return prereg, h1


def _year_statistics(year: int, frame: pd.DataFrame, contract: str) -> tuple[dict[str, Any], dict[str, list[int]]]:
    states: dict[str, dict[str, Any]] = defaultdict(balance._empty_stats); coverage: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    era = balance._era(year)
    for signal_date, group in frame.groupby("signal_date", sort=True):
        date_text = pd.Timestamp(signal_date).date().isoformat()
        for population in ("broad", "investable"):
            current = group if population == "broad" else group.loc[group.investable]
            if current.empty: continue
            current = current.reset_index(drop=True); names = current.symbol.astype(str).to_numpy(object)
            raw = current[list(rank_matches.v1.MATCH_COLUMNS)].to_numpy(float); ranks = percentile_rank_matrix(raw)
            deciles = np.clip(np.ceil(10 * ranks), 1, 10).astype(np.uint8)
            transformed = {name: balance._transform(name, current[name].to_numpy(float)) for name in balance.TRANSFORMS}
            for signal_name, (risk_column, raw_column, event_column) in rank_matches.v1.SIGNALS.items():
                candidates = np.flatnonzero(current[risk_column].to_numpy(bool) & ~current[raw_column].to_numpy(bool))
                events = np.flatnonzero(current[event_column].to_numpy(bool))
                ids = [event_id(contract, signal_name, str(names[index]), date_text) for index in events]
                for weight in WEIGHTS:
                    selected = select_rank_nearest_batch(symbols=names, percentile_ranks=ranks, candidate_indices=candidates,
                        event_indices=events, contract_digest=contract, signal_ids=ids,
                        coordinate_weights=(1., float(weight), 1., 1.))
                    coverage_key = f"{weight}|{population}|{signal_name}"; coverage[coverage_key][0] += len(events)
                    coverage[coverage_key][1] += sum(len(indices) == 5 for indices, _ in selected)
                    full_positions = [position for position, (indices, _) in enumerate(selected) if len(indices) == 5]
                    if not full_positions: continue
                    full_events = events[full_positions]
                    control_matrix = np.vstack([selected[position][0] for position in full_positions])
                    for column_index, name in enumerate(balance.TRANSFORMS):
                        event_values = np.repeat(transformed[name][full_events], 5)
                        control_values = transformed[name][control_matrix.ravel()]
                        gaps = np.abs(np.repeat(deciles[full_events, column_index].astype(int), 5)
                                      - deciles[control_matrix.ravel(), column_index].astype(int))
                        for scope in ("overall", era):
                            balance._add(states[f"{weight}|{population}|{signal_name}|{scope}|{name}"],
                                         event_values, control_values, gaps)
    return dict(states), dict(coverage)


def _write_year(year: int, frame: pd.DataFrame, cache: Path, contract: str) -> dict[str, Any]:
    final = cache / f"year-{year}"; path = final / "YEAR_SEALED.json"
    if path.exists():
        value = io._read(path)
        if not _valid(value, timing=True): raise WeightBakeoffError(f"bake-off year differs: {year}")
        return value
    started = perf_counter(); stats, coverage = _year_statistics(year, frame, contract)
    state = {"schema_version": SCHEMA, "status": "year_sealed", "passed": True, "year": year,
        "input_panel_rows": len(frame), "statistics": stats, "coverage": coverage, "outcome_columns_read": [],
        "post_signal_results_accessed": False, "elapsed_seconds": perf_counter() - started}
    seal = _seal(state, timing=True); final.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".year-{year}.", dir=final.parent))
    try: smoke._atomic_json(temporary / "YEAR_SEALED.json", seal); os.replace(temporary, final)
    except BaseException: shutil.rmtree(temporary, ignore_errors=True); raise
    return seal


def _summaries(states: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    rows = []
    for weight in WEIGHTS:
        selected = {"|".join(key.split("|")[1:]): value for key, value in states.items() if key.startswith(f"{weight}|")}
        frame = pd.DataFrame(balance._summary_rows(selected)); frame.insert(0, "volatility_weight", weight); rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def execute(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True); prereg, h1 = validate_preregistration(repository); root = repository / OUTPUT_RELATIVE
    if root.exists():
        seal = io._read(root / "SEALED.json")
        if not _valid(seal, timing=True) or seal.get("file_manifest") != _manifest(root, OUTPUT_FILES): raise WeightBakeoffError("bake-off output differs")
        return seal
    started = perf_counter(); frame = rank_matches.v1._read_causal_panel(repository); grouped = frame.groupby(frame.signal_date.dt.year, sort=True)
    cache = repository / CACHE_RELATIVE; cache.mkdir(parents=True, exist_ok=True)
    seals = [_write_year(year, grouped.get_group(year), cache, prereg["contract_digest"]) for year in prereg["years"]]
    combined: dict[str, dict[str, Any]] = defaultdict(balance._empty_stats); coverage: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for seal in seals:
        for key, value in seal["statistics"].items(): balance._merge_stats(combined[key], value)
        for key, value in seal["coverage"].items(): coverage[key] = [coverage[key][i] + int(value[i]) for i in range(2)]
    summary = _summaries(combined); weight_rows = []
    for weight in WEIGHTS:
        current = summary.loc[summary.volatility_weight.eq(weight)]; overall = current.scope.eq("overall")
        overall_pass = bool((current.loc[overall, "standardized_mean_difference"].abs() <= .10).all())
        era_pass = bool((current.loc[~overall, "standardized_mean_difference"].abs() <= .20).all())
        keys = [key for key in coverage if key.startswith(f"{weight}|")]
        total = sum(coverage[key][0] for key in keys); full = sum(coverage[key][1] for key in keys); rate = full / total
        weight_rows.append({"volatility_weight": weight, "overall_balance_pass": overall_pass,
                            "era_balance_pass": era_pass, "full_match_coverage": rate,
                            "candidate_pass": overall_pass and era_pass and rate >= .90})
    passing = [row["volatility_weight"] for row in weight_rows if row["candidate_pass"]]
    selected_weight = min(passing) if passing else None
    decision = _seal({"schema_version": SCHEMA, "status": "selection_decision", "candidate_results": weight_rows,
        "selected_volatility_weight": selected_weight, "selection_pass": selected_weight is not None,
        "final_identity_materialization_authorized": selected_weight is not None,
        "post_signal_inference_authorized": False, "outcome_columns_read": [], "production_promotion_authorized": False})
    temporary = Path(tempfile.mkdtemp(prefix=f".{root.name}.", dir=root.parent)); smoke._atomic_parquet(temporary / OUTPUT_FILES[0], summary)
    smoke._atomic_json(temporary / OUTPUT_FILES[1], decision)
    state = {"schema_version": SCHEMA, "status": "sealed", "passed": True, "preregistration_h1": h1,
        "preregistration_digest": prereg["preregistration_digest"], "year_result_digests": [value["result_digest"] for value in seals],
        "summary_rows": len(summary), "selection_decision_digest": decision["result_digest"],
        "file_manifest": _manifest(temporary, OUTPUT_FILES), "outcome_columns_read": [], "post_signal_results_accessed": False,
        "elapsed_seconds": perf_counter() - started, "independent_verification_authorized": True, "production_promotion_authorized": False}
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
