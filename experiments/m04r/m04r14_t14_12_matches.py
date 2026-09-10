"""Build restartable, outcome-blind T14-12 matched-control identities."""
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
import pyarrow as pa
import pyarrow.parquet as pq

from market_analogues.post_signal_matching import (
    bucket_members, control_order_digest, cross_sectional_deciles, event_id, select_control_indices,
)
from market_analogues.types import stable_hash

from experiments.m04r import m04r14_t14_09_outcome_smoke as smoke
from experiments.m04r import m04r14_t14_10_wf03_feasibility as base
from experiments.m04r import m04r14_t14_12_panel as panel_stage


SCHEMA = "m04r14-t14-12-post-signal-matches-v1"
PREREGISTRATION_RELATIVE = Path("experiments/m04r/m04r14_t14_12_matches_v1_preregistered.json")
CACHE_RELATIVE = Path("config/data/analogues/m04r14/t14-12-post-signal-matches-v1-cache")
OUTPUT_RELATIVE = Path("config/data/analogues/m04r14/t14-12-post-signal-matches-v1")
VERIFICATION_RELATIVE = Path("config/data/analogues/m04r14/t14-12-post-signal-matches-v1-verification")
YEAR_FILES = ("event-matches.parquet", "control-identities.parquet")
MATCH_COLUMNS = (
    "prior_return_63", "prior_volatility_20", "prior_close", "prior_median_dollar_volume_20",
)
MARKET_COLUMNS = (
    "benchmark_signal_day_return", "benchmark_return_20", "benchmark_return_63", "benchmark_volatility_20",
)
SIGNALS = {
    "up_close_4pct": ("up_close_at_risk", "up_close_4pct", "up_close_signal_event"),
    "bullish_range_expansion_4pct": (
        "bullish_range_expansion_at_risk", "bullish_range_expansion_4pct",
        "bullish_range_expansion_signal_event",
    ),
}
CAUSAL_COLUMNS = (
    "symbol", "signal_date", "signal_position", "investable", *MATCH_COLUMNS, *MARKET_COLUMNS,
    *(column for values in SIGNALS.values() for column in values),
)
RUNTIME_FILES = (
    "experiments/m04r/m04r14_t14_12_matches.py",
    "experiments/m04r/verify_m04r14_t14_12_matches.py",
    "experiments/m04r/m04r14_t14_12_panel.py",
    "src/market_analogues/post_signal_matching.py",
    "config/m04r14-t14-12-post-signal-contract.json", "pyproject.toml",
)
EVENT_SCHEMA = pa.schema([
    ("population", pa.string()), ("signal_name", pa.string()), ("signal_id", pa.string()),
    ("event_symbol", pa.string()), ("signal_date", pa.timestamp("ns")),
    ("event_signal_position", pa.int64()), ("match_tier", pa.string()),
    ("control_count", pa.int64()), ("full_match", pa.bool_()),
    *((f"event_{name}_decile", pa.int64()) for name in MATCH_COLUMNS),
    *((name, pa.float64()) for name in MARKET_COLUMNS),
])
CONTROL_SCHEMA = pa.schema([
    ("population", pa.string()), ("signal_name", pa.string()), ("signal_id", pa.string()),
    ("event_symbol", pa.string()), ("control_symbol", pa.string()),
    ("signal_date", pa.timestamp("ns")), ("control_signal_position", pa.int64()),
    ("match_rank", pa.int64()), ("match_tier", pa.string()), ("selection_digest", pa.string()),
    ("maximum_decile_distance", pa.int64()),
    *((f"event_{name}_decile", pa.int64()) for name in MATCH_COLUMNS),
    *((f"control_{name}_decile", pa.int64()) for name in MATCH_COLUMNS),
])


class MatchStudyError(RuntimeError): pass


def _now() -> str: return datetime.now(timezone.utc).isoformat()


def _git(repository: Path, *args: str, raw: bool = False) -> str | bytes:
    result = subprocess.run(["git", *args], cwd=repository, capture_output=True, text=not raw, check=False)
    if result.returncode:
        error = result.stderr if not raw else result.stderr.decode(errors="replace")
        raise MatchStudyError(error.strip())
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


def _verified_panel(repository: Path) -> dict[str, Any]:
    store_path = repository / panel_stage.OUTPUT_RELATIVE / "SEALED.json"
    verify_path = repository / panel_stage.VERIFICATION_RELATIVE / "VERIFIED.json"
    store, receipt = base._read(store_path), base._read(verify_path)
    if not panel_stage._valid(store, timing=True) or receipt.get("passed") is not True \
            or receipt.get("store_result_digest") != store.get("result_digest") \
            or receipt.get("control_matching_or_inference_accessed") is not False:
        raise MatchStudyError("verified panel boundary differs")
    return {
        "panel_store_result_digest": store["result_digest"], "panel_store_sha256": _sha(store_path),
        "panel_verification_digest": receipt["verification_digest"], "panel_verification_sha256": _sha(verify_path),
        "panel_rows": int(store["panel_rows"]),
        "up_close_signal_events": int(store["up_close_signal_events"]),
        "bullish_range_expansion_signal_events": int(store["bullish_range_expansion_signal_events"]),
    }


def _runtime_manifest(repository: Path, head: str) -> dict[str, str]:
    tracked = set(str(_git(repository, "ls-tree", "-r", "--name-only", head)).splitlines())
    if any(name not in tracked for name in RUNTIME_FILES): raise MatchStudyError("runtime file is uncommitted")
    return {name: sha256(_git(repository, "show", f"{head}:{name}", raw=True)).hexdigest() for name in RUNTIME_FILES}


def _read_causal_panel(repository: Path) -> pd.DataFrame:
    frames = [
        pd.read_parquet(
            repository / panel_stage.CACHE_RELATIVE / f"shard-{shard:02d}" / "daily-panel.parquet",
            columns=list(CAUSAL_COLUMNS),
        ) for shard in range(panel_stage.SHARDS)
    ]
    frame = pd.concat(frames, ignore_index=True); del frames
    frame["signal_date"] = pd.to_datetime(frame.signal_date)
    frame = frame.sort_values(["signal_date", "symbol"], kind="stable").reset_index(drop=True)
    if frame.duplicated(["signal_date", "symbol"]).any(): raise MatchStudyError("duplicate symbol/date panel rows")
    return frame


def build_preregistration(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    if _git(repository, "status", "--porcelain", "--untracked-files=all"): raise MatchStudyError("clean worktree required")
    if (repository / CACHE_RELATIVE).exists() or (repository / OUTPUT_RELATIVE).exists():
        raise MatchStudyError("matching namespaces must be absent")
    verified = _verified_panel(repository); contract = panel_stage._contract(repository)
    dates = pd.concat([
        pd.read_parquet(
            repository / panel_stage.CACHE_RELATIVE / f"shard-{shard:02d}" / "daily-panel.parquet",
            columns=["signal_date"],
        ) for shard in range(panel_stage.SHARDS)
    ], ignore_index=True).signal_date
    years = sorted(int(value) for value in pd.to_datetime(dates).dt.year.unique())
    h0 = str(_git(repository, "rev-parse", "HEAD"))
    return _seal({
        "schema_version": SCHEMA, "status": "frozen_before_real_control_identity_selection",
        "implementation_h0": h0, "runtime_files": _runtime_manifest(repository, h0),
        "verified_panel": verified, "contract_digest": contract["contract_digest"], "years": years,
        "causal_columns_read": list(CAUSAL_COLUMNS),
        "outcome_columns_read": [], "matching_or_inference_results_accessed": False,
        "decile_rule": "all_same_date_population_rows_average_rank_then_ceil_10_rank_divided_by_n",
        "signal_specific_control_rule": "same_signal_at_risk_and_raw_signal_false_today",
        "selection_serialization": "utf8(contract_digest|signal_id|candidate_symbol)",
        "relaxation_rule": "first_tier_with_at_least_five_candidates_else_same_date_shortfall",
        "controls_per_signal": 5, "year_checkpoints_restartable_and_atomic": True,
        "horizon_identity_rule": "one_outcome_blind_identity_set_per_signal_reused_across_5_20_60;horizon_completeness_applied_only_in_P3",
        "partial_and_unmatched_events_retained": True, "production_promotion_authorized": False,
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
    if len(set(found)) != 1: raise MatchStudyError("expected exact preregistration-only child")
    return found[0]


def validate_preregistration(repository: Path) -> tuple[dict[str, Any], str]:
    if _git(repository, "status", "--porcelain", "--untracked-files=all"): raise MatchStudyError("clean worktree required")
    path = repository / PREREGISTRATION_RELATIVE; raw = path.read_bytes(); prereg = base._read(path)
    if prereg.get("schema_version") != SCHEMA or not _valid(prereg, "preregistration_digest"):
        raise MatchStudyError("matching preregistration differs")
    h1 = _sole_child(repository, raw, str(prereg["implementation_h0"]))
    if subprocess.run(["git", "merge-base", "--is-ancestor", h1, "HEAD"], cwd=repository).returncode:
        raise MatchStudyError("HEAD does not descend from preregistration")
    if prereg["verified_panel"] != _verified_panel(repository): raise MatchStudyError("verified panel input drifted")
    for name, expected in prereg["runtime_files"].items():
        if _sha(repository / name) != expected: raise MatchStudyError(f"runtime drifted: {name}")
    return prereg, h1


class _ParquetSink:
    def __init__(self, path: Path, schema: pa.Schema) -> None:
        self.path, self.schema, self.writer, self.rows = path, schema, None, 0
    def write(self, rows: list[dict[str, Any]]) -> None:
        if not rows: return
        table = pa.Table.from_pylist(rows, schema=self.schema)
        if self.writer is None: self.writer = pq.ParquetWriter(self.path, table.schema, compression="zstd")
        self.writer.write_table(table); self.rows += len(rows); rows.clear()
    def close(self) -> None:
        if self.writer is None:
            pq.write_table(pa.Table.from_pylist([], schema=self.schema), self.path, compression="zstd")
            return
        self.writer.close()


def _write_year(year: int, frame: pd.DataFrame, cache: Path, contract_digest: str) -> dict[str, Any]:
    final = cache / f"year-{year}"; seal_path = final / "YEAR_SEALED.json"
    if seal_path.exists():
        value = base._read(seal_path)
        if not _valid(value, timing=True) or value.get("file_manifest") != _manifest(final, YEAR_FILES):
            raise MatchStudyError(f"existing year checkpoint differs: {year}")
        return value
    started = perf_counter(); temporary = Path(tempfile.mkdtemp(prefix=f".year-{year}.", dir=cache))
    event_sink = _ParquetSink(temporary / YEAR_FILES[0], EVENT_SCHEMA)
    control_sink = _ParquetSink(temporary / YEAR_FILES[1], CONTROL_SCHEMA)
    event_rows: list[dict[str, Any]] = []; control_rows: list[dict[str, Any]] = []
    coverage: dict[tuple[str, str, str, int], int] = defaultdict(int)
    date_count = 0
    for signal_date, group in frame.groupby("signal_date", sort=True):
        date_count += 1; date_text = pd.Timestamp(signal_date).date().isoformat()
        if any(group[column].nunique(dropna=False) != 1 for column in MARKET_COLUMNS):
            raise MatchStudyError(f"same-date market state differs: {date_text}")
        for population in ("broad", "investable"):
            current = group if population == "broad" else group.loc[group.investable]
            if current.empty: continue
            current = current.reset_index(drop=True)
            names = current.symbol.astype(str).to_numpy(object)
            deciles = np.column_stack([
                cross_sectional_deciles(current[column].to_numpy(float)) for column in MATCH_COLUMNS
            ])
            for signal_name, (risk_column, raw_column, event_column) in SIGNALS.items():
                at_risk = current[risk_column].to_numpy(bool); raw_signal = current[raw_column].to_numpy(bool)
                event_indices = np.flatnonzero(current[event_column].to_numpy(bool))
                candidate_mask = at_risk & ~raw_signal
                eligible = np.flatnonzero(candidate_mask); members = bucket_members(deciles, candidate_mask)
                for event_index in event_indices:
                    event_symbol = str(names[event_index])
                    signal_id = event_id(contract_digest, signal_name, event_symbol, date_text)
                    selected, tier = select_control_indices(
                        symbols=names, members=members, all_eligible_indices=eligible,
                        event_symbol=event_symbol, event_deciles=deciles[event_index],
                        contract_digest=contract_digest, signal_id=signal_id, controls=5,
                    )
                    count = len(selected); full = count == 5
                    coverage[(population, signal_name, tier, count)] += 1
                    event_row = {
                        "population": population, "signal_name": signal_name, "signal_id": signal_id,
                        "event_symbol": event_symbol, "signal_date": pd.Timestamp(signal_date),
                        "event_signal_position": int(current.signal_position.iloc[event_index]),
                        "match_tier": tier, "control_count": count, "full_match": full,
                    }
                    event_row.update({f"event_{name}_decile": int(deciles[event_index, i]) for i, name in enumerate(MATCH_COLUMNS)})
                    event_row.update({name: float(current[name].iloc[event_index]) for name in MARKET_COLUMNS})
                    event_rows.append(event_row)
                    for rank, control_index in enumerate(selected, 1):
                        control_symbol = str(names[control_index]); control_deciles = deciles[control_index]
                        row = {
                            "population": population, "signal_name": signal_name, "signal_id": signal_id,
                            "event_symbol": event_symbol, "control_symbol": control_symbol,
                            "signal_date": pd.Timestamp(signal_date),
                            "control_signal_position": int(current.signal_position.iloc[control_index]),
                            "match_rank": rank, "match_tier": tier,
                            "selection_digest": control_order_digest(contract_digest, signal_id, control_symbol),
                            "maximum_decile_distance": int(np.max(np.abs(deciles[event_index].astype(int) - control_deciles.astype(int)))),
                        }
                        row.update({f"event_{name}_decile": int(deciles[event_index, i]) for i, name in enumerate(MATCH_COLUMNS)})
                        row.update({f"control_{name}_decile": int(control_deciles[i]) for i, name in enumerate(MATCH_COLUMNS)})
                        control_rows.append(row)
                    if len(event_rows) >= 25_000: event_sink.write(event_rows)
                    if len(control_rows) >= 50_000: control_sink.write(control_rows)
    event_sink.write(event_rows); control_sink.write(control_rows); event_sink.close(); control_sink.close()
    state = {
        "schema_version": SCHEMA, "status": "year_sealed", "passed": True, "year": year,
        "input_panel_rows": len(frame), "signal_dates": date_count,
        "event_match_rows": event_sink.rows, "control_identity_rows": control_sink.rows,
        "coverage": {"|".join(map(str, key)): value for key, value in sorted(coverage.items())},
        "file_manifest": _manifest(temporary, YEAR_FILES), "outcome_columns_read": [],
        "inference_accessed": False, "elapsed_seconds": perf_counter() - started,
    }
    seal = _seal(state, timing=True); smoke._atomic_json(temporary / "YEAR_SEALED.json", seal)
    try: os.replace(temporary, final)
    except BaseException: shutil.rmtree(temporary, ignore_errors=True); raise
    return seal


def execute(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True); prereg, h1 = validate_preregistration(repository)
    root = repository / OUTPUT_RELATIVE
    if root.exists():
        seal = base._read(root / "SEALED.json")
        if not _valid(seal, timing=True) or seal.get("file_manifest") != _manifest(root, ("coverage.json",)):
            raise MatchStudyError("existing aggregate output differs")
        return seal
    started = perf_counter(); frame = _read_causal_panel(repository)
    if len(frame) != int(prereg["verified_panel"]["panel_rows"]): raise MatchStudyError("causal panel row count differs")
    years = sorted(int(value) for value in frame.signal_date.dt.year.unique())
    if years != prereg["years"]: raise MatchStudyError("year inventory differs")
    cache = repository / CACHE_RELATIVE; cache.mkdir(parents=True, exist_ok=True)
    seals = []
    grouped_years = frame.groupby(frame.signal_date.dt.year, sort=True)
    for year in years:
        seals.append(_write_year(year, grouped_years.get_group(year), cache, prereg["contract_digest"]))
    totals: dict[str, int] = defaultdict(int); coverage: dict[str, int] = defaultdict(int)
    for seal in seals:
        totals["input_panel_rows"] += int(seal["input_panel_rows"])
        totals["event_match_rows"] += int(seal["event_match_rows"])
        totals["control_identity_rows"] += int(seal["control_identity_rows"])
        for key, value in seal["coverage"].items(): coverage[key] += int(value)
    eligible = totals["event_match_rows"]
    fully = sum(value for key, value in coverage.items() if key.endswith("|5"))
    coverage_state = _seal({
        "schema_version": SCHEMA, "status": "coverage", "event_match_rows": eligible,
        "fully_matched_event_rows": fully, "full_match_coverage": fully / eligible,
        "control_identity_rows": totals["control_identity_rows"],
        "by_population_signal_tier_count": dict(sorted(coverage.items())), "shortfalls_retained": True,
    })
    root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{root.name}.", dir=root.parent))
    smoke._atomic_json(temporary / "coverage.json", coverage_state)
    state = {
        "schema_version": SCHEMA, "status": "sealed", "passed": True,
        "preregistration_h1": h1, "preregistration_digest": prereg["preregistration_digest"],
        "panel_store_result_digest": prereg["verified_panel"]["panel_store_result_digest"],
        "years": years, "year_result_digests": [value["result_digest"] for value in seals],
        **totals, "coverage_result_digest": coverage_state["result_digest"],
        "causal_columns_read": list(CAUSAL_COLUMNS), "outcome_columns_read": [], "inference_accessed": False,
        "file_manifest": _manifest(temporary, ("coverage.json",)), "elapsed_seconds": perf_counter() - started,
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
