"""Independently verify the complete T14-12 causal daily panel."""
from __future__ import annotations

import argparse
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
from experiments.m04r import m04r14_t14_10_wf03_feasibility as base
from experiments.m04r import m04r14_t14_12_panel as target
from experiments.m04r import m04r14_t14_12_synthetic_gate as oracle


SCHEMA = "m04r14-t14-12-post-signal-panel-verification-v1"
TOLERANCE = 5e-13


class PanelVerificationError(RuntimeError): pass


def _now() -> str: return datetime.now(timezone.utc).isoformat()


def _sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20): digest.update(block)
    return digest.hexdigest()


def _valid(value: Mapping[str, Any], key: str = "result_digest", *, timing: bool = False) -> bool:
    omitted = {key, "created_at"} | ({"elapsed_seconds"} if timing else set())
    return value.get(key) == stable_hash({name: item for name, item in value.items() if name not in omitted})


def _valid_mask(frame: pd.DataFrame) -> np.ndarray:
    values = frame[["open", "high", "low", "close", "volume"]].to_numpy(float)
    return np.isfinite(values).all(axis=1) & (values[:, :4] > 0).all(axis=1) & (values[:, 4] >= 0) \
        & (values[:, 1] >= np.maximum.reduce((values[:, 0], values[:, 2], values[:, 3]))) \
        & (values[:, 2] <= np.minimum.reduce((values[:, 0], values[:, 1], values[:, 3])))


def _expected_counts(frame: pd.DataFrame, benchmark_position: Mapping[pd.Timestamp, int]) -> dict[str, int]:
    data = frame.rename(columns={"date": "timestamp"}).copy(); data["timestamp"] = pd.to_datetime(data.timestamp)
    n = len(data); valid = _valid_mask(data)
    if n <= 252:
        return {"panel_rows": 0, "invalid_source_rows": int((~valid).sum()), "up_close_signal_events": 0,
                "bullish_range_expansion_signal_events": 0, **{f"complete_{h}_rows": 0 for h in (5, 20, 60)}}
    close = data.close.to_numpy(float); open_ = data.open.to_numpy(float)
    high = data.high.to_numpy(float); low = data.low.to_numpy(float)
    previous = np.roll(close, 1); previous[0] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        change = close / previous - 1
        tr_fraction = np.maximum.reduce((high - low, abs(high - previous), abs(low - previous))) / previous
    change[~valid] = np.nan; tr_fraction[~valid] = np.nan
    medians = pd.Series(tr_fraction).rolling(20, min_periods=20).median().shift(1).to_numpy(float)
    up = change >= .04
    expansion = (tr_fraction >= .04) & (close > open_) & (tr_fraction >= 1.5 * medians)
    positions = np.arange(252, n); invalid_cumulative = np.concatenate(([0], np.cumsum((~valid).astype(int))))
    positions = positions[(invalid_cumulative[positions + 1] - invalid_cumulative[positions - 252]) == 0]
    market_positions_all = np.asarray([benchmark_position.get(pd.Timestamp(x), -1) for x in data.timestamp], dtype=int)
    positions = positions[market_positions_all[positions] >= 0]
    exposure_cumulative = {name: np.concatenate(([0], np.cumsum(values.astype(int))))
                           for name, values in (("up", up), ("expansion", expansion))}
    at_risk_up = exposure_cumulative["up"][positions] - exposure_cumulative["up"][positions - 20] == 0
    at_risk_expansion = exposure_cumulative["expansion"][positions] - exposure_cumulative["expansion"][positions - 20] == 0
    result = {
        "panel_rows": len(positions), "invalid_source_rows": int((~valid).sum()),
        "up_close_signal_events": int((up[positions] & at_risk_up).sum()),
        "bullish_range_expansion_signal_events": int((expansion[positions] & at_risk_expansion).sum()),
    }
    breaks = market_positions_all[1:] != market_positions_all[:-1] + 1
    break_cumulative = np.concatenate(([0], np.cumsum(breaks.astype(int))))
    for horizon in (5, 20, 60):
        endpoint = positions + horizon; within = endpoint < n; complete = np.zeros(len(positions), dtype=bool)
        candidates = np.flatnonzero(within)
        if len(candidates):
            p, end = positions[candidates], endpoint[candidates]
            future_valid = invalid_cumulative[end + 1] - invalid_cumulative[p + 1] == 0
            market_start = market_positions_all[p]
            exact = (break_cumulative[end] - break_cumulative[p] == 0) \
                & (market_start + horizon < len(benchmark_position))
            complete[candidates] = future_valid & exact
        result[f"complete_{horizon}_rows"] = int(complete.sum())
    return result


def _assert_scalar(expected: Mapping[str, Any], observed: pd.Series) -> None:
    if set(expected) != set(observed.index): raise PanelVerificationError("sample row columns differ")
    for name, value in expected.items():
        actual = observed[name]
        if isinstance(value, (float, np.floating)):
            if np.isnan(float(value)) and pd.isna(actual): continue
            if not np.isclose(float(value), float(actual), rtol=0., atol=TOLERANCE):
                raise PanelVerificationError(f"sample numeric difference: {name}")
        elif actual != value: raise PanelVerificationError(f"sample value difference: {name}")


def execute(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    if subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=repository,
                      capture_output=True, text=True, check=True).stdout:
        raise PanelVerificationError("clean worktree required")
    prereg, source_accounting, _ = target.validate_preregistration(repository)
    root = repository / target.OUTPUT_RELATIVE; seal = base._read(root / "SEALED.json")
    if not target._valid(seal, timing=True) or seal.get("independent_verification_authorized") is not True:
        raise PanelVerificationError("panel aggregate seal differs")
    contract = target._contract(repository); benchmark = pd.read_parquet(contract["source"]["benchmark_path"])
    benchmark = benchmark.loc[pd.to_datetime(benchmark.date) <= pd.Timestamp(contract["source"]["locked_coverage_end"])].reset_index(drop=True)
    benchmark_position = {pd.Timestamp(value): i for i, value in enumerate(pd.to_datetime(benchmark.date))}
    records = {str(row.symbol): row for row in source_accounting.itertuples(index=False)}
    started = perf_counter(); seen = set(); totals = {"panel_rows": 0, "up_close_signal_events": 0,
                                                       "bullish_range_expansion_signal_events": 0}
    scalar_rows = 0
    for shard in range(target.SHARDS):
        shard_root = repository / target.CACHE_RELATIVE / f"shard-{shard:02d}"
        shard_seal = base._read(shard_root / "SHARD_SEALED.json")
        if not target._valid(shard_seal, timing=True) \
                or shard_seal.get("file_manifest") != target._manifest(shard_root, target.SHARD_FILES):
            raise PanelVerificationError(f"shard seal differs: {shard}")
        panel = pd.read_parquet(shard_root / "daily-panel.parquet")
        accounting = pd.read_parquet(shard_root / "symbol-accounting.parquet")
        if accounting.symbol.astype(str).duplicated().any() or set(accounting.symbol.astype(str)) & seen:
            raise PanelVerificationError("duplicate symbol accounting")
        shard_symbols = list(accounting.symbol.astype(str)); seen.update(shard_symbols)
        expected_symbols = sorted(symbol for symbol in records if target.source_stage._shard(symbol) == shard)
        if sorted(shard_symbols) != expected_symbols or len(panel) != int(accounting.panel_rows.sum()):
            raise PanelVerificationError(f"shard symbol/panel reconciliation differs: {shard}")
        groups = panel.groupby("symbol", sort=False).indices
        sample_symbols = sorted(shard_symbols, key=lambda symbol: stable_hash([prereg["contract_digest"], shard, symbol]))[:5]
        for row in accounting.itertuples(index=False):
            symbol = str(row.symbol); source = records[symbol]; path = Path(source.source_path)
            if _sha(path) != str(source.source_hash_at_lock) or row.source_sha256 != str(source.source_hash_at_lock):
                raise PanelVerificationError(f"source identity differs: {symbol}")
            frame = pd.read_parquet(path, columns=["date", "open", "high", "low", "close", "volume"])
            frame = frame.loc[pd.to_datetime(frame.date) <= pd.Timestamp(source.coverage_last_timestamp)].reset_index(drop=True)
            expected = _expected_counts(frame, benchmark_position)
            for name, value in expected.items():
                if int(getattr(row, name)) != int(value): raise PanelVerificationError(f"all-symbol count differs: {symbol}/{name}")
            symbol_panel = panel.iloc[groups[symbol]].sort_values("signal_position", kind="stable").reset_index(drop=True) \
                if symbol in groups else panel.iloc[:0]
            if len(symbol_panel) != expected["panel_rows"]: raise PanelVerificationError(f"symbol panel length differs: {symbol}")
            if symbol in sample_symbols and len(symbol_panel):
                choices = {0, len(symbol_panel) // 2, len(symbol_panel) - 1}
                signal_positions = np.flatnonzero(
                    symbol_panel.up_close_signal_event.to_numpy(bool)
                    | symbol_panel.bullish_range_expansion_signal_event.to_numpy(bool)
                )
                if len(signal_positions): choices.add(int(signal_positions[len(signal_positions) // 2]))
                prepared = frame.rename(columns={"date": "timestamp"})
                for choice in sorted(choices):
                    observed = symbol_panel.iloc[choice]
                    _assert_scalar(oracle.scalar_row(prepared, benchmark, int(observed.signal_position), symbol), observed)
                    scalar_rows += 1
        for name in totals: totals[name] += int(shard_seal[name])
    if seen != set(records) or len(seen) != prereg["symbol_count"]:
        raise PanelVerificationError("complete symbol inventory differs")
    if any(int(seal[name]) != value for name, value in totals.items()):
        raise PanelVerificationError("aggregate counts differ")
    if seal.get("shard_result_digests") != [
        base._read(repository / target.CACHE_RELATIVE / f"shard-{i:02d}" / "SHARD_SEALED.json")["result_digest"]
        for i in range(target.SHARDS)
    ]: raise PanelVerificationError("aggregate shard binding differs")
    state = {
        "schema_version": SCHEMA, "status": "verified", "passed": True,
        "store_result_digest": seal["result_digest"], "store_result_sha256": _sha(root / "SEALED.json"),
        "verified_symbols": len(seen), "verified_panel_rows": totals["panel_rows"],
        "verified_up_close_signal_events": totals["up_close_signal_events"],
        "verified_bullish_range_expansion_signal_events": totals["bullish_range_expansion_signal_events"],
        "independent_scalar_rows": scalar_rows,
        "gates": {"all_shard_seals_valid": True, "all_symbol_counts_reconstructed": True,
                  "all_symbol_source_identities_valid": True, "stratified_scalar_rows_reconstructed": True,
                  "aggregate_inventory_reconciled": True, "control_matching_remains_unopened": True},
        "control_matching_or_inference_accessed": False, "production_promotion_authorized": False,
        "elapsed_seconds": perf_counter() - started, "created_at": _now(),
    }
    result = {**state, "verification_digest": stable_hash(state)}; output = repository / target.VERIFICATION_RELATIVE
    if output.exists(): return base._read(output / "VERIFIED.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        smoke._atomic_json(temporary / "VERIFIED.json", result); os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True); raise
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--repository", type=Path, required=True)
    args = parser.parse_args(argv); print(json.dumps(execute(args.repository), indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
