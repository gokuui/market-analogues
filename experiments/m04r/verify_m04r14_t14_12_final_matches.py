"""Independently verify every selected weight-4 T14-12 control identity."""
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

from experiments.m04r import m04r14_t14_10_wf03_feasibility as base
from experiments.m04r import m04r14_t14_12_final_matches as target


SCHEMA = "m04r14-t14-12-post-signal-final-matches-verification-v3"
MATCH_COLUMNS = ("prior_return_63", "prior_volatility_20", "prior_close", "prior_median_dollar_volume_20")
MARKET_COLUMNS = ("benchmark_signal_day_return", "benchmark_return_20", "benchmark_return_63", "benchmark_volatility_20")
SIGNALS = {"up_close_4pct": ("up_close_at_risk", "up_close_4pct", "up_close_signal_event"),
           "bullish_range_expansion_4pct": ("bullish_range_expansion_at_risk", "bullish_range_expansion_4pct",
                                             "bullish_range_expansion_signal_event")}
WEIGHTS = np.array([1., 4., 1., 1.])
TOLERANCE = 5e-15


class FinalMatchVerificationError(RuntimeError): pass


def _now() -> str: return datetime.now(timezone.utc).isoformat()


def _sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20): digest.update(block)
    return digest.hexdigest()


def _event_id(contract: str, signal: str, symbol: str, date_text: str) -> str:
    return sha256(f"{contract}|{signal}|{symbol}|{date_text}".encode()).hexdigest()


def _digest(contract: str, signal_id: str, symbol: str) -> str:
    return sha256(f"{contract}|{signal_id}|{symbol}".encode()).hexdigest()


def _ranks(values: np.ndarray) -> np.ndarray:
    return np.column_stack([pd.Series(values[:, index]).rank(method="average").to_numpy(float) / len(values)
                            for index in range(4)])


def _sort(names: np.ndarray, candidates: np.ndarray, points: np.ndarray, event: np.ndarray,
          contract: str, signal_id: str) -> tuple[np.ndarray, np.ndarray]:
    squared = np.sum(np.square(points - event), axis=1)
    order = sorted(range(len(candidates)), key=lambda index: (
        float(squared[index]), _digest(contract, signal_id, str(names[candidates[index]])), str(names[candidates[index]]),
    ))[:5]
    return candidates[order], squared[order]


def _batch(names: np.ndarray, ranks: np.ndarray, candidates: np.ndarray, events: np.ndarray,
           contract: str, ids: Sequence[str]) -> list[tuple[np.ndarray, np.ndarray]]:
    if not len(candidates): return [(np.array([], dtype=int), np.array([], dtype=float)) for _ in events]
    scale = np.sqrt(WEIGHTS); points = ranks[candidates] * scale; queries = ranks[events] * scale
    frontier = min(len(candidates), 16); tree = cKDTree(points); distances, indices = tree.query(queries, k=frontier)
    if frontier == 1: distances, indices = np.asarray(distances)[:, None], np.asarray(indices)[:, None]
    result = []
    for row, signal_id in enumerate(ids):
        local = np.atleast_1d(indices[row]).astype(int)
        if len(candidates) > frontier:
            kth, last = float(np.atleast_1d(distances[row])[4]), float(np.atleast_1d(distances[row])[-1])
            if abs(last - kth) <= 1e-14 * max(1., abs(last), abs(kth)):
                local = np.asarray(tree.query_ball_point(queries[row], kth * (1 + 1e-12) + 1e-15), dtype=int)
        result.append(_sort(names, candidates[local], points[local], queries[row], contract, signal_id))
    return result


def _equal(actual: Any, expected: Any) -> bool:
    if pd.isna(actual) and pd.isna(expected): return True
    if isinstance(expected, (float, np.floating)):
        return math.isclose(float(actual), float(expected), rel_tol=0., abs_tol=TOLERANCE)
    return bool(actual == expected)


def _assert_row(actual: pd.Series, expected: Mapping[str, Any], label: str) -> None:
    if set(actual.index) != set(expected): raise FinalMatchVerificationError(f"{label} schema differs")
    for name, value in expected.items():
        if not _equal(actual[name], value): raise FinalMatchVerificationError(f"{label} differs: {name}")


def execute(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    if subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=repository,
                      capture_output=True, text=True, check=True).stdout:
        raise FinalMatchVerificationError("clean worktree required")
    if tuple(target.v1.MATCH_COLUMNS) != MATCH_COLUMNS or tuple(target.v1.MARKET_COLUMNS) != MARKET_COLUMNS \
            or target.v1.SIGNALS != SIGNALS or tuple(target.COORDINATE_WEIGHTS) != tuple(WEIGHTS) \
            or target.SELECTED_WEIGHT != 4:
        raise FinalMatchVerificationError("producer semantic mapping differs")
    started = perf_counter(); prereg, _ = target.validate_preregistration(repository); contract = prereg["contract_digest"]
    root = repository / target.OUTPUT_RELATIVE; seal = base._read(root / "SEALED.json")
    if not target._valid(seal, timing=True) or seal.get("file_manifest") != target._manifest(root, ("coverage.json",)):
        raise FinalMatchVerificationError("final aggregate seal differs")
    frame = target.v1._read_causal_panel(repository); grouped_years = frame.groupby(frame.signal_date.dt.year, sort=True)
    totals = defaultdict(int); coverage = defaultdict(int); digests = []; exhaustive_cases = 0
    for year in prereg["years"]:
        year_root = repository / target.CACHE_RELATIVE / f"year-{year}"; year_seal = base._read(year_root / "YEAR_SEALED.json")
        if not target._valid(year_seal, timing=True) or year_seal.get("file_manifest") != target._manifest(year_root, target.YEAR_FILES) \
                or year_seal.get("selected_volatility_weight") != 4 or year_seal.get("outcome_columns_read") != []:
            raise FinalMatchVerificationError(f"final year seal differs: {year}")
        events_frame = pd.read_parquet(year_root / "event-matches.parquet")
        controls_frame = pd.read_parquet(year_root / "control-identities.parquet")
        selected_year = grouped_years.get_group(year); event_cursor = control_cursor = dates = 0
        sample_dates = set(sorted(selected_year.signal_date.unique(), key=lambda date: stable_hash([year, str(date)]))[:3])
        for signal_date, group in selected_year.groupby("signal_date", sort=True):
            dates += 1; date_text = pd.Timestamp(signal_date).date().isoformat()
            for population in ("broad", "investable"):
                current = group if population == "broad" else group.loc[group.investable]
                if current.empty: continue
                current = current.reset_index(drop=True); names = current.symbol.astype(str).to_numpy(object)
                ranks = _ranks(current[list(MATCH_COLUMNS)].to_numpy(float)); deciles = np.clip(np.ceil(10 * ranks), 1, 10).astype(np.uint8)
                scale = np.sqrt(WEIGHTS)
                for signal_name, (risk_column, raw_column, event_column) in SIGNALS.items():
                    candidates = np.flatnonzero(current[risk_column].to_numpy(bool) & ~current[raw_column].to_numpy(bool))
                    events = np.flatnonzero(current[event_column].to_numpy(bool))
                    ids = [_event_id(contract, signal_name, str(names[index]), date_text) for index in events]
                    selected_sets = _batch(names, ranks, candidates, events, contract, ids)
                    for event_index, signal_id, (selected, squared) in zip(events, ids, selected_sets):
                        if signal_date in sample_dates:
                            brute, brute_squared = _sort(names, candidates, ranks[candidates] * scale,
                                                         ranks[event_index] * scale, contract, signal_id)
                            if not np.array_equal(selected, brute) or not np.array_equal(squared, brute_squared):
                                raise FinalMatchVerificationError("weighted spatial/exhaustive selection differs")
                            exhaustive_cases += 1
                        count = len(selected); coverage[f"{population}|{signal_name}|{count}"] += 1
                        event_symbol = str(names[event_index]); expected_event = {
                            "population": population, "signal_name": signal_name, "signal_id": signal_id,
                            "event_symbol": event_symbol, "signal_date": pd.Timestamp(signal_date),
                            "event_signal_position": int(current.signal_position.iloc[event_index]),
                            "match_tier": target.MATCH_TIER, "control_count": count, "full_match": count == 5,
                            "mean_squared_rank_distance": float(squared.mean()) if count else np.nan,
                            "maximum_squared_rank_distance": float(squared.max()) if count else np.nan,
                            **{f"event_{name}_decile": int(deciles[event_index, i]) for i, name in enumerate(MATCH_COLUMNS)},
                            **{name: float(current[name].iloc[event_index]) for name in MARKET_COLUMNS}}
                        if event_cursor >= len(events_frame): raise FinalMatchVerificationError("event output truncated")
                        _assert_row(events_frame.iloc[event_cursor], expected_event, "final event"); event_cursor += 1
                        for match_rank, (control_index, distance) in enumerate(zip(selected, squared), 1):
                            control_symbol = str(names[control_index]); control_deciles = deciles[control_index]
                            expected_control = {"population": population, "signal_name": signal_name,
                                "signal_id": signal_id, "event_symbol": event_symbol, "control_symbol": control_symbol,
                                "signal_date": pd.Timestamp(signal_date),
                                "control_signal_position": int(current.signal_position.iloc[control_index]),
                                "match_rank": match_rank, "match_tier": target.MATCH_TIER,
                                "selection_digest": _digest(contract, signal_id, control_symbol),
                                "squared_rank_distance": float(distance),
                                "maximum_decile_distance": int(np.max(np.abs(deciles[event_index].astype(int)-control_deciles.astype(int)))),
                                **{f"event_{name}_decile": int(deciles[event_index, i]) for i, name in enumerate(MATCH_COLUMNS)},
                                **{f"control_{name}_decile": int(control_deciles[i]) for i, name in enumerate(MATCH_COLUMNS)}}
                            if control_cursor >= len(controls_frame): raise FinalMatchVerificationError("control output truncated")
                            _assert_row(controls_frame.iloc[control_cursor], expected_control, "final control"); control_cursor += 1
        if event_cursor != len(events_frame) or control_cursor != len(controls_frame):
            raise FinalMatchVerificationError(f"unexpected final rows: {year}")
        if dates != int(year_seal["signal_dates"]) or len(selected_year) != int(year_seal["input_panel_rows"]):
            raise FinalMatchVerificationError(f"final year accounting differs: {year}")
        local_coverage = defaultdict(int)
        for row in events_frame.itertuples(index=False): local_coverage[f"{row.population}|{row.signal_name}|{int(row.control_count)}"] += 1
        if dict(sorted(local_coverage.items())) != year_seal["coverage"]: raise FinalMatchVerificationError("final year coverage differs")
        totals["input_panel_rows"] += len(selected_year); totals["event_match_rows"] += len(events_frame)
        totals["control_identity_rows"] += len(controls_frame); digests.append(year_seal["result_digest"])
    if digests != seal["year_result_digests"] or any(int(seal[name]) != value for name, value in totals.items()):
        raise FinalMatchVerificationError("final aggregate accounting differs")
    coverage_receipt = base._read(root / "coverage.json"); full = sum(v for k, v in coverage.items() if k.endswith("|5"))
    if not target._valid(coverage_receipt) or coverage_receipt["by_population_signal_count"] != dict(sorted(coverage.items())) \
            or coverage_receipt["fully_matched_event_rows"] != full:
        raise FinalMatchVerificationError("final coverage differs")
    state = {"schema_version": SCHEMA, "status": "verified", "passed": True,
        "store_result_digest": seal["result_digest"], "store_result_sha256": _sha(root / "SEALED.json"),
        "verified_input_panel_rows": totals["input_panel_rows"], "verified_event_match_rows": totals["event_match_rows"],
        "verified_control_identity_rows": totals["control_identity_rows"], "independent_exhaustive_cases": exhaustive_cases,
        "selected_volatility_weight": 4,
        "gates": {"all_year_seals_valid": True, "all_percentile_ranks_reconstructed": True,
                  "all_weighted_nearest_identities_and_distances_reconstructed": True,
                  "stratified_exhaustive_oracle_passed": True, "all_coverage_reconstructed": True,
                  "outcome_boundary_remained_closed": True},
        "outcome_columns_read": [], "post_signal_results_accessed": False, "production_promotion_authorized": False,
        "elapsed_seconds": perf_counter() - started, "created_at": _now()}
    result = {**state, "verification_digest": stable_hash(state)}; output = repository / target.VERIFICATION_RELATIVE
    if output.exists(): return base._read(output / "VERIFIED.json")
    output.parent.mkdir(parents=True, exist_ok=True); temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try: target.smoke._atomic_json(temporary / "VERIFIED.json", result); os.replace(temporary, output)
    except BaseException: shutil.rmtree(temporary, ignore_errors=True); raise
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--repository", type=Path, required=True)
    args = parser.parse_args(argv); print(json.dumps(execute(args.repository), indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
