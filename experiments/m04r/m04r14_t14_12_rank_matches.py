"""Build T14-12 V2 continuous-rank nearest control identities."""
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

from market_analogues.post_signal_matching import control_order_digest, event_id
from market_analogues.post_signal_rank_matching import percentile_rank_matrix, select_rank_nearest_batch
from market_analogues.types import stable_hash

from experiments.m04r import m04r14_t14_09_outcome_smoke as smoke
from experiments.m04r import m04r14_t14_10_wf03_feasibility as base
from experiments.m04r import m04r14_t14_12_matches as v1


SCHEMA = "m04r14-t14-12-post-signal-rank-matches-v2"
PREREGISTRATION_RELATIVE = Path("experiments/m04r/m04r14_t14_12_rank_matches_v2_preregistered.json")
CACHE_RELATIVE = Path("config/data/analogues/m04r14/t14-12-post-signal-rank-matches-v2-cache")
OUTPUT_RELATIVE = Path("config/data/analogues/m04r14/t14-12-post-signal-rank-matches-v2")
VERIFICATION_RELATIVE = Path("config/data/analogues/m04r14/t14-12-post-signal-rank-matches-v2-verification")
YEAR_FILES = v1.YEAR_FILES
RUNTIME_FILES = (
    "experiments/m04r/m04r14_t14_12_rank_matches.py",
    "experiments/m04r/verify_m04r14_t14_12_rank_matches.py",
    "experiments/m04r/m04r14_t14_12_matches.py",
    "src/market_analogues/post_signal_rank_matching.py",
    "src/market_analogues/post_signal_matching.py",
    "config/m04r14-t14-12-post-signal-contract.json", "pyproject.toml",
)
EVENT_SCHEMA = pa.schema([
    ("population", pa.string()), ("signal_name", pa.string()), ("signal_id", pa.string()),
    ("event_symbol", pa.string()), ("signal_date", pa.timestamp("ns")),
    ("event_signal_position", pa.int64()), ("match_tier", pa.string()),
    ("control_count", pa.int64()), ("full_match", pa.bool_()),
    ("mean_squared_rank_distance", pa.float64()), ("maximum_squared_rank_distance", pa.float64()),
    *((f"event_{name}_decile", pa.int64()) for name in v1.MATCH_COLUMNS),
    *((name, pa.float64()) for name in v1.MARKET_COLUMNS),
])
CONTROL_SCHEMA = pa.schema([
    ("population", pa.string()), ("signal_name", pa.string()), ("signal_id", pa.string()),
    ("event_symbol", pa.string()), ("control_symbol", pa.string()),
    ("signal_date", pa.timestamp("ns")), ("control_signal_position", pa.int64()),
    ("match_rank", pa.int64()), ("match_tier", pa.string()), ("selection_digest", pa.string()),
    ("squared_rank_distance", pa.float64()), ("maximum_decile_distance", pa.int64()),
    *((f"event_{name}_decile", pa.int64()) for name in v1.MATCH_COLUMNS),
    *((f"control_{name}_decile", pa.int64()) for name in v1.MATCH_COLUMNS),
])


class RankMatchStudyError(RuntimeError): pass


def _now() -> str: return datetime.now(timezone.utc).isoformat()


def _git(repository: Path, *args: str, raw: bool = False) -> str | bytes:
    result = subprocess.run(["git", *args], cwd=repository, capture_output=True, text=not raw, check=False)
    if result.returncode:
        error = result.stderr if not raw else result.stderr.decode(errors="replace")
        raise RankMatchStudyError(error.strip())
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


def _verified_v1_and_panel(repository: Path) -> dict[str, Any]:
    panel = v1._verified_panel(repository)
    store_path = repository / Path("config/data/analogues/m04r14/t14-12-post-signal-balance-v1/SEALED.json")
    balance_path = repository / Path("config/data/analogues/m04r14/t14-12-post-signal-balance-v1-verification/VERIFIED.json")
    decision_path = repository / Path("config/data/analogues/m04r14/t14-12-post-signal-balance-v1/balance-decision.json")
    store, receipt, decision = base._read(store_path), base._read(balance_path), base._read(decision_path)
    receipt_state = {name: item for name, item in receipt.items() if name != "verification_digest"}
    if not _valid(store, timing=True) or receipt.get("verification_digest") != stable_hash(receipt_state) \
            or receipt.get("store_result_digest") != store.get("result_digest") \
            or not _valid(decision) or receipt.get("passed") is not True \
            or receipt.get("balance_gate_pass") is not False or receipt.get("outcome_columns_read") != [] \
            or decision.get("outcome_blind_rematching_required") is not True:
        raise RankMatchStudyError("verified V1 balance failure boundary differs")
    return {**panel, "v1_balance_store_digest": store["result_digest"],
            "v1_balance_store_sha256": _sha(store_path),
            "v1_balance_verification_digest": receipt["verification_digest"],
            "v1_balance_verification_sha256": _sha(balance_path),
            "v1_balance_decision_digest": decision["result_digest"], "v1_balance_decision_sha256": _sha(decision_path)}


def _runtime_manifest(repository: Path, head: str) -> dict[str, str]:
    tracked = set(str(_git(repository, "ls-tree", "-r", "--name-only", head)).splitlines())
    if any(name not in tracked for name in RUNTIME_FILES): raise RankMatchStudyError("runtime file is uncommitted")
    return {name: sha256(_git(repository, "show", f"{head}:{name}", raw=True)).hexdigest() for name in RUNTIME_FILES}


def build_preregistration(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    if _git(repository, "status", "--porcelain", "--untracked-files=all"): raise RankMatchStudyError("clean worktree required")
    if (repository / CACHE_RELATIVE).exists() or (repository / OUTPUT_RELATIVE).exists():
        raise RankMatchStudyError("V2 matching namespaces must be absent")
    h0 = str(_git(repository, "rev-parse", "HEAD")); prior = base._read(repository / v1.PREREGISTRATION_RELATIVE)
    return _seal({
        "schema_version": SCHEMA, "status": "frozen_before_V2_control_identity_selection",
        "implementation_h0": h0, "runtime_files": _runtime_manifest(repository, h0),
        "verified_inputs": _verified_v1_and_panel(repository), "contract_digest": prior["contract_digest"],
        "years": prior["years"], "causal_columns_read": list(v1.CAUSAL_COLUMNS), "outcome_columns_read": [],
        "matching_coordinates": "average_percentile_rank_within_same_date_population_for_four_frozen_covariates",
        "distance": "sum_of_four_squared_percentile_rank_differences",
        "candidate_rule": "same_signal_at_risk_and_raw_signal_false_today",
        "selection": "five_smallest_distance_then_sha256_contract_signal_candidate_then_symbol",
        "spatial_frontier": 16, "ties_at_frontier_expanded_and_exactly_sorted": True,
        "controls_per_signal": 5, "replacement_within_event": False, "reuse_across_events": True,
        "year_checkpoints_restartable_and_atomic": True,
        "balance_limits_unchanged_from_V1": True, "post_signal_results_accessed": False,
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
    if len(set(found)) != 1: raise RankMatchStudyError("expected exact preregistration-only child")
    return found[0]


def validate_preregistration(repository: Path) -> tuple[dict[str, Any], str]:
    if _git(repository, "status", "--porcelain", "--untracked-files=all"): raise RankMatchStudyError("clean worktree required")
    path = repository / PREREGISTRATION_RELATIVE; raw = path.read_bytes(); prereg = base._read(path)
    if prereg.get("schema_version") != SCHEMA or not _valid(prereg, "preregistration_digest"):
        raise RankMatchStudyError("V2 preregistration differs")
    h1 = _sole_child(repository, raw, str(prereg["implementation_h0"]))
    if subprocess.run(["git", "merge-base", "--is-ancestor", h1, "HEAD"], cwd=repository).returncode:
        raise RankMatchStudyError("HEAD does not descend from V2 preregistration")
    if prereg["verified_inputs"] != _verified_v1_and_panel(repository): raise RankMatchStudyError("V2 inputs drifted")
    for name, expected in prereg["runtime_files"].items():
        if _sha(repository / name) != expected: raise RankMatchStudyError(f"runtime drifted: {name}")
    return prereg, h1


class _Sink:
    def __init__(self, path: Path, schema: pa.Schema) -> None: self.path, self.schema, self.writer, self.rows = path, schema, None, 0
    def write(self, rows: list[dict[str, Any]]) -> None:
        if not rows: return
        table = pa.Table.from_pylist(rows, schema=self.schema)
        if self.writer is None: self.writer = pq.ParquetWriter(self.path, self.schema, compression="zstd")
        self.writer.write_table(table); self.rows += len(rows); rows.clear()
    def close(self) -> None:
        if self.writer is None: pq.write_table(pa.Table.from_pylist([], schema=self.schema), self.path, compression="zstd")
        else: self.writer.close()


def _write_year(year: int, frame: pd.DataFrame, cache: Path, contract: str) -> dict[str, Any]:
    final = cache / f"year-{year}"; path = final / "YEAR_SEALED.json"
    if path.exists():
        value = base._read(path)
        if not _valid(value, timing=True) or value.get("file_manifest") != _manifest(final, YEAR_FILES):
            raise RankMatchStudyError(f"year checkpoint differs: {year}")
        return value
    started = perf_counter(); temporary = Path(tempfile.mkdtemp(prefix=f".year-{year}.", dir=cache))
    event_sink = _Sink(temporary / YEAR_FILES[0], EVENT_SCHEMA); control_sink = _Sink(temporary / YEAR_FILES[1], CONTROL_SCHEMA)
    event_rows: list[dict[str, Any]] = []; control_rows: list[dict[str, Any]] = []; coverage: dict[str, int] = defaultdict(int)
    dates = 0
    for signal_date, group in frame.groupby("signal_date", sort=True):
        dates += 1; date_text = pd.Timestamp(signal_date).date().isoformat()
        for population in ("broad", "investable"):
            current = group if population == "broad" else group.loc[group.investable]
            if current.empty: continue
            current = current.reset_index(drop=True); names = current.symbol.astype(str).to_numpy(object)
            values = current[list(v1.MATCH_COLUMNS)].to_numpy(float); ranks = percentile_rank_matrix(values)
            deciles = np.clip(np.ceil(10 * ranks), 1, 10).astype(np.uint8)
            for signal_name, (risk_column, raw_column, event_column) in v1.SIGNALS.items():
                candidates = np.flatnonzero(current[risk_column].to_numpy(bool) & ~current[raw_column].to_numpy(bool))
                events = np.flatnonzero(current[event_column].to_numpy(bool))
                ids = [event_id(contract, signal_name, str(names[index]), date_text) for index in events]
                selections = select_rank_nearest_batch(symbols=names, percentile_ranks=ranks,
                    candidate_indices=candidates, event_indices=events, contract_digest=contract, signal_ids=ids)
                for event_index, signal_id, (selected, squared) in zip(events, ids, selections):
                    event_symbol = str(names[event_index]); count = len(selected); coverage[f"{population}|{signal_name}|{count}"] += 1
                    row = {"population": population, "signal_name": signal_name, "signal_id": signal_id,
                           "event_symbol": event_symbol, "signal_date": pd.Timestamp(signal_date),
                           "event_signal_position": int(current.signal_position.iloc[event_index]),
                           "match_tier": "continuous_rank_nearest", "control_count": count, "full_match": count == 5,
                           "mean_squared_rank_distance": float(squared.mean()) if count else np.nan,
                           "maximum_squared_rank_distance": float(squared.max()) if count else np.nan}
                    row.update({f"event_{name}_decile": int(deciles[event_index, i]) for i, name in enumerate(v1.MATCH_COLUMNS)})
                    row.update({name: float(current[name].iloc[event_index]) for name in v1.MARKET_COLUMNS}); event_rows.append(row)
                    for rank, (control_index, distance) in enumerate(zip(selected, squared), 1):
                        control_symbol = str(names[control_index]); control_deciles = deciles[control_index]
                        item = {"population": population, "signal_name": signal_name, "signal_id": signal_id,
                                "event_symbol": event_symbol, "control_symbol": control_symbol,
                                "signal_date": pd.Timestamp(signal_date),
                                "control_signal_position": int(current.signal_position.iloc[control_index]),
                                "match_rank": rank, "match_tier": "continuous_rank_nearest",
                                "selection_digest": control_order_digest(contract, signal_id, control_symbol),
                                "squared_rank_distance": float(distance),
                                "maximum_decile_distance": int(np.max(np.abs(deciles[event_index].astype(int) - control_deciles.astype(int))))}
                        item.update({f"event_{name}_decile": int(deciles[event_index, i]) for i, name in enumerate(v1.MATCH_COLUMNS)})
                        item.update({f"control_{name}_decile": int(control_deciles[i]) for i, name in enumerate(v1.MATCH_COLUMNS)})
                        control_rows.append(item)
                    if len(event_rows) >= 25_000: event_sink.write(event_rows)
                    if len(control_rows) >= 50_000: control_sink.write(control_rows)
    event_sink.write(event_rows); control_sink.write(control_rows); event_sink.close(); control_sink.close()
    state = {"schema_version": SCHEMA, "status": "year_sealed", "passed": True, "year": year,
             "input_panel_rows": len(frame), "signal_dates": dates, "event_match_rows": event_sink.rows,
             "control_identity_rows": control_sink.rows, "coverage": dict(sorted(coverage.items())),
             "file_manifest": _manifest(temporary, YEAR_FILES), "outcome_columns_read": [],
             "post_signal_results_accessed": False, "elapsed_seconds": perf_counter() - started}
    seal = _seal(state, timing=True); smoke._atomic_json(temporary / "YEAR_SEALED.json", seal)
    try: os.replace(temporary, final)
    except BaseException: shutil.rmtree(temporary, ignore_errors=True); raise
    return seal


def execute(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True); prereg, h1 = validate_preregistration(repository); root = repository / OUTPUT_RELATIVE
    if root.exists():
        seal = base._read(root / "SEALED.json")
        if not _valid(seal, timing=True) or seal.get("file_manifest") != _manifest(root, ("coverage.json",)):
            raise RankMatchStudyError("existing V2 output differs")
        return seal
    started = perf_counter(); frame = v1._read_causal_panel(repository); grouped = frame.groupby(frame.signal_date.dt.year, sort=True)
    cache = repository / CACHE_RELATIVE; cache.mkdir(parents=True, exist_ok=True)
    seals = [_write_year(year, grouped.get_group(year), cache, prereg["contract_digest"]) for year in prereg["years"]]
    coverage: dict[str, int] = defaultdict(int); events = controls = rows = 0
    for value in seals:
        rows += int(value["input_panel_rows"]); events += int(value["event_match_rows"]); controls += int(value["control_identity_rows"])
        for key, count in value["coverage"].items(): coverage[key] += int(count)
    full = sum(count for key, count in coverage.items() if key.endswith("|5"))
    coverage_state = _seal({"schema_version": SCHEMA, "status": "coverage", "event_match_rows": events,
        "fully_matched_event_rows": full, "full_match_coverage": full / events, "control_identity_rows": controls,
        "by_population_signal_count": dict(sorted(coverage.items())), "shortfalls_retained": True})
    temporary = Path(tempfile.mkdtemp(prefix=f".{root.name}.", dir=root.parent)); smoke._atomic_json(temporary / "coverage.json", coverage_state)
    state = {"schema_version": SCHEMA, "status": "sealed", "passed": True, "preregistration_h1": h1,
        "preregistration_digest": prereg["preregistration_digest"], "panel_store_result_digest": prereg["verified_inputs"]["panel_store_result_digest"],
        "years": prereg["years"], "year_result_digests": [value["result_digest"] for value in seals],
        "input_panel_rows": rows, "event_match_rows": events, "control_identity_rows": controls,
        "coverage_result_digest": coverage_state["result_digest"], "causal_columns_read": list(v1.CAUSAL_COLUMNS),
        "outcome_columns_read": [], "post_signal_results_accessed": False,
        "file_manifest": _manifest(temporary, ("coverage.json",)), "elapsed_seconds": perf_counter() - started,
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
