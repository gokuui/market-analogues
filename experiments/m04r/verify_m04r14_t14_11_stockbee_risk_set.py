"""Verify all Stockbee risk-set shards and scalar-oracle sampled rows."""
from __future__ import annotations

import argparse
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
from experiments.m04r import m04r14_t14_11_stockbee_risk_set as target


SCHEMA = "m04r14-t14-11-stockbee-risk-set-verification-v2"
SAMPLE_PER_SHARD = 5
TOLERANCE = 5e-13


class RiskSetVerificationError(RuntimeError): pass


def _now() -> str: return datetime.now(timezone.utc).isoformat()


def _sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20): digest.update(block)
    return digest.hexdigest()


def _valid(value: Mapping[str, Any], key: str = "result_digest", *, timing: bool = False) -> bool:
    omitted = {key, "created_at"} | ({"elapsed_seconds"} if timing else set())
    return value.get(key) == stable_hash({k: v for k, v in value.items() if k not in omitted})


def _close(left: Any, right: Any) -> bool:
    if pd.isna(left) and pd.isna(right): return True
    if isinstance(left, (float, np.floating, int, np.integer)) and isinstance(right, (float, np.floating, int, np.integer)):
        return abs(float(left) - float(right)) <= TOLERANCE
    return left == right


def _true_range(open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray, index: int) -> float:
    previous = close[index - 1]
    return max(high[index] - low[index], abs(high[index] - previous), abs(low[index] - previous)) / previous


def _valid_mask(frame: pd.DataFrame) -> np.ndarray:
    ohlc = frame[["open", "high", "low", "close"]].to_numpy(float); volume = frame.volume.to_numpy(float)
    return (
        np.isfinite(ohlc).all(axis=1) & (ohlc > 0).all(axis=1) & np.isfinite(volume) & (volume >= 0)
        & (ohlc[:, 1] >= np.maximum.reduce((ohlc[:, 0], ohlc[:, 2], ohlc[:, 3])))
        & (ohlc[:, 2] <= np.minimum.reduce((ohlc[:, 0], ohlc[:, 1], ohlc[:, 3])))
    )


def _expected_count(valid: np.ndarray) -> int:
    cumulative = np.concatenate(([0], np.cumsum((~valid).astype(np.int64)))); total = 0; n = len(valid)
    for horizon in (21, 63):
        starts = np.arange(252, n - horizon + 1)
        if len(starts): total += int(((cumulative[starts + horizon] - cumulative[starts - 252]) == 0).sum())
    return total


def _exposure(open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray, index: int, name: str) -> bool:
    if name == "up_close_4pct": return close[index] / close[index - 1] - 1 >= .04
    tr = _true_range(open_, high, low, close, index)
    if name == "true_range_4pct": return tr >= .04
    prior = np.median([_true_range(open_, high, low, close, j) for j in range(index - 20, index)])
    return tr >= .04 and close[index] > open_[index] and tr >= 1.5 * prior


def _scalar_row(frame: pd.DataFrame, symbol: str, start: int, horizon: int) -> dict[str, Any]:
    bars = frame.reset_index(drop=True); open_ = bars.open.to_numpy(float); high = bars.high.to_numpy(float)
    low = bars.low.to_numpy(float); close = bars.close.to_numpy(float); volume = bars.volume.to_numpy(float)
    result: dict[str, Any] = {
        "symbol": symbol, "start": pd.Timestamp(bars.timestamp.iloc[start]), "start_position": start,
        "horizon_sessions": horizon,
        "forward_close_return": close[start + horizon - 1] / close[start - 1] - 1,
        "winner_25pct": close[start + horizon - 1] / close[start - 1] - 1 >= .25,
        "start_close": close[start], "prior_return_63": close[start - 1] / close[start - 64] - 1,
        "prior_volatility_20": np.std(np.log(close[start - 20:start] / close[start - 21:start - 1]), ddof=0),
        "prior_median_dollar_volume_20": np.median(close[start - 20:start] * volume[start - 20:start]),
    }
    for name in ("up_close_4pct", "true_range_4pct", "bullish_range_expansion_4pct"):
        values = np.array([_exposure(open_, high, low, close, j, name) for j in range(start - 20, start + horizon)])
        pre = values[:20]; move = values[20:]; first = move[:5]
        result[f"{name}_start_day"] = bool(move[0]); result[f"{name}_first_5_sessions"] = bool(first.any())
        result[f"{name}_full_move"] = bool(move.any()); result[f"{name}_pre_start_20_sessions"] = bool(pre.any())
        result[f"{name}_first_5_count"] = int(first.sum()); result[f"{name}_full_move_count"] = int(move.sum())
    result["investable"] = result["start_close"] >= 5 and result["prior_median_dollar_volume_20"] >= 1_000_000
    return result


def execute(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    if subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=repository,
                      capture_output=True, text=True, check=True).stdout:
        raise RiskSetVerificationError("clean worktree required")
    started = perf_counter(); output = repository / target.OUTPUT_RELATIVE
    seal = base._read(output / "SEALED.json"); prereg = base._read(repository / target.PREREGISTRATION_RELATIVE)
    if not _valid(seal, timing=True) or not _valid(prereg, "preregistration_digest") \
            or seal.get("preregistration_digest") != prereg.get("preregistration_digest"):
        raise RiskSetVerificationError("risk-set seals differ")
    expected = target._accounting(repository); cache = repository / target.CACHE_RELATIVE
    accountings = []; sampled_rows = 0
    for shard in range(target.SHARDS):
        root = cache / f"shard-{shard:02d}"; shard_seal = base._read(root / "SHARD_SEALED.json")
        if not _valid(shard_seal, timing=True) or shard_seal.get("shard") != shard \
                or shard_seal.get("file_manifest") != target._manifest(root, ("risk-set.parquet", "symbol-accounting.parquet")):
            raise RiskSetVerificationError(f"shard seal differs: {shard}")
        accounting = pd.read_parquet(root / "symbol-accounting.parquet"); accountings.append(accounting.assign(shard=shard))
        chosen = sorted(accounting.symbol.astype(str), key=lambda x: sha256(f"verify|{x}".encode()).hexdigest())[:SAMPLE_PER_SHARD]
        for symbol in chosen:
            source_row = expected.loc[expected.symbol.astype(str) == symbol].iloc[0]
            frame = pd.read_parquet(source_row.source_path, columns=["date", "open", "high", "low", "close", "volume"])
            frame = frame.loc[pd.to_datetime(frame.date) <= pd.Timestamp(source_row.coverage_last_timestamp)].rename(columns={"date": "timestamp"})
            observed = pd.read_parquet(root / "risk-set.parquet", filters=[("symbol", "==", symbol)])
            for horizon, group in observed.groupby("horizon_sessions"):
                positions = sorted(set((int(group.start_position.min()), int(group.start_position.median()), int(group.start_position.max()))))
                for position in positions:
                    row = group.loc[group.start_position == position].iloc[0]
                    oracle = _scalar_row(frame, symbol, position, int(horizon))
                    if any(not _close(row[name], value) for name, value in oracle.items()):
                        raise RiskSetVerificationError(f"scalar oracle differs: {symbol}:{horizon}:{position}")
                    sampled_rows += 1
    accounting = pd.concat(accountings, ignore_index=True).sort_values("symbol", kind="stable").reset_index(drop=True)
    if accounting.symbol.astype(str).tolist() != expected.symbol.astype(str).tolist():
        raise RiskSetVerificationError("symbol accounting inventory differs")
    expected_counts = []; invalid_counts = []
    for row in expected.itertuples(index=False):
        frame = pd.read_parquet(row.source_path, columns=["date", "open", "high", "low", "close", "volume"])
        frame = frame.loc[pd.to_datetime(frame.date) <= pd.Timestamp(row.coverage_last_timestamp)]
        valid = _valid_mask(frame); invalid_counts.append(int((~valid).sum())); expected_counts.append(_expected_count(valid))
    if accounting.risk_rows.astype(int).tolist() != expected_counts \
            or accounting.invalid_source_rows.astype(int).tolist() != invalid_counts \
            or accounting.risk_rows.sum() != seal.get("risk_rows") \
            or accounting.winner_rows.sum() != seal.get("winner_rows"):
        raise RiskSetVerificationError("complete universe row accounting differs")
    state = {
        "schema_version": SCHEMA, "status": "verified", "passed": True,
        "store_result_digest": seal["result_digest"], "store_result_sha256": _sha(output / "SEALED.json"),
        "verified_shards": target.SHARDS, "verified_symbols": len(accounting),
        "verified_risk_rows": int(accounting.risk_rows.sum()), "scalar_oracle_rows": sampled_rows,
        "gates": {"all_shard_bytes_and_seals_valid": True, "every_locked_symbol_accounted_once": True,
                  "every_symbol_risk_row_count_reconstructed": True, "cross_shard_scalar_oracle_passed": True,
                  "exposure_control_prevalence_remains_unopened": True},
        "exposure_control_prevalence_calculated": False, "production_promotion_authorized": False,
        "elapsed_seconds": perf_counter() - started, "created_at": _now(),
    }
    result = {**state, "verification_digest": stable_hash(state)}
    final = repository / target.VERIFICATION_RELATIVE
    if final.exists(): return base._read(final / "VERIFIED.json")
    final.parent.mkdir(parents=True, exist_ok=True); temporary = Path(tempfile.mkdtemp(prefix=f".{final.name}.", dir=final.parent))
    try:
        target.smoke._atomic_json(temporary / "VERIFIED.json", result); os.replace(temporary, final)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True); raise
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--repository", type=Path, required=True)
    args = parser.parse_args(argv); print(json.dumps(execute(args.repository), indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
