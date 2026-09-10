"""Independent synthetic/scalar gate for the T14-12 post-signal vector kernel."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from market_analogues.post_signal_study import symbol_post_signal_panel
from market_analogues.types import stable_hash

from experiments.m04r import m04r14_t14_09_outcome_smoke as smoke
from experiments.m04r import m04r14_t14_10_wf03_feasibility as base


SCHEMA = "m04r14-t14-12-post-signal-synthetic-v1"
CONTRACT_RELATIVE = Path("config/m04r14-t14-12-post-signal-contract.json")
OUTPUT_RELATIVE = Path("config/data/analogues/m04r14/t14-12-post-signal-synthetic-v1")
RUNTIME_FILES = (
    "experiments/m04r/m04r14_t14_12_synthetic_gate.py",
    "src/market_analogues/post_signal_study.py",
    "config/m04r14-t14-12-post-signal-contract.json",
)
TOLERANCE = 5e-13


class SyntheticGateError(RuntimeError): pass


def _now() -> str: return datetime.now(timezone.utc).isoformat()


def _sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20): digest.update(block)
    return digest.hexdigest()


def _valid_bar(row: pd.Series) -> bool:
    values = [float(row.open), float(row.high), float(row.low), float(row.close), float(row.volume)]
    return all(math.isfinite(x) for x in values) and min(values[:4]) > 0 and values[4] >= 0 \
        and values[1] >= max(values[0], values[2], values[3]) \
        and values[2] <= min(values[0], values[1], values[3])


def _true_range(stock: pd.DataFrame, position: int) -> tuple[float, float]:
    previous = float(stock.iloc[position - 1].close); row = stock.iloc[position]
    value = max(float(row.high - row.low), abs(float(row.high) - previous), abs(float(row.low) - previous))
    return value, value / previous


def scalar_row(stock_frame: pd.DataFrame, benchmark_frame: pd.DataFrame, position: int, symbol: str = "SYN") -> dict[str, Any]:
    stock = stock_frame.rename(columns={"date": "timestamp"}).copy(); stock["timestamp"] = pd.to_datetime(stock.timestamp)
    market = benchmark_frame.rename(columns={"date": "timestamp"}).copy(); market["timestamp"] = pd.to_datetime(market.timestamp)
    if position < 252 or position >= len(stock) or not all(_valid_bar(stock.iloc[i]) for i in range(position - 252, position + 1)):
        raise SyntheticGateError("scalar position is not causally eligible")
    market_by_date = {pd.Timestamp(row.timestamp): i for i, row in enumerate(market.itertuples(index=False))}
    date = pd.Timestamp(stock.iloc[position].timestamp); market_position = market_by_date.get(date)
    if market_position is None: raise SyntheticGateError("signal date is absent from benchmark")
    previous = float(stock.iloc[position - 1].close); row = stock.iloc[position]
    move = float(row.close) / previous - 1
    tr, tr_fraction = _true_range(stock, position)
    prior_tr_fraction = [_true_range(stock, i)[1] for i in range(position - 20, position)]
    expansion = tr_fraction >= .04 and float(row.close) > float(row.open) \
        and tr_fraction >= 1.5 * float(np.median(prior_tr_fraction))
    prior_moves = [] ; prior_expansions = []
    for i in range(position - 20, position):
        r = stock.iloc[i]; prior = float(stock.iloc[i - 1].close); change = float(r.close) / prior - 1
        prior_move_tr, prior_move_fraction = _true_range(stock, i)
        median_fraction = np.median([_true_range(stock, j)[1] for j in range(i - 20, i)])
        prior_moves.append(change >= .04)
        prior_expansions.append(prior_move_fraction >= .04 and float(r.close) > float(r.open)
                                and prior_move_fraction >= 1.5 * median_fraction)
    log_returns = [math.log(float(stock.iloc[i].close) / float(stock.iloc[i - 1].close))
                   for i in range(position - 20, position)]
    dollar_volume = [float(stock.iloc[i].close) * float(stock.iloc[i].volume)
                     for i in range(position - 20, position)]
    atr = float(np.mean([_true_range(stock, i)[0] for i in range(position - 19, position + 1)]))
    result: dict[str, Any] = {
        "symbol": symbol, "signal_date": date, "signal_position": position,
        "prior_close": previous,
        "prior_return_63": previous / float(stock.iloc[position - 64].close) - 1,
        "prior_volatility_20": float(np.std(log_returns, ddof=0)),
        "prior_median_dollar_volume_20": float(np.median(dollar_volume)),
        "signal_close": float(row.close), "signal_atr_20": atr,
        "up_close_4pct": move >= .04, "up_close_at_risk": not any(prior_moves),
        "up_close_signal_event": move >= .04 and not any(prior_moves),
        "bullish_range_expansion_4pct": expansion,
        "bullish_range_expansion_at_risk": not any(prior_expansions),
        "bullish_range_expansion_signal_event": expansion and not any(prior_expansions),
        "investable": previous >= 5 and np.median(dollar_volume) >= 1_000_000,
        "benchmark_signal_day_return": float(market.iloc[market_position].close) / float(market.iloc[market_position - 1].close) - 1 if market_position else np.nan,
        "benchmark_return_20": float(market.iloc[market_position].close) / float(market.iloc[market_position - 20].close) - 1 if market_position >= 20 else np.nan,
        "benchmark_return_63": float(market.iloc[market_position].close) / float(market.iloc[market_position - 63].close) - 1 if market_position >= 63 else np.nan,
        "benchmark_volatility_20": float(np.std([
            math.log(float(market.iloc[i].close) / float(market.iloc[i - 1].close))
            for i in range(market_position - 19, market_position + 1)
        ], ddof=0)) if market_position >= 20 else np.nan,
    }
    market_dates = list(pd.to_datetime(market.timestamp))
    for horizon in (5, 20, 60):
        if position + horizon >= len(stock):
            complete, status = False, "source_end_before_horizon"
        else:
            future = stock.iloc[position + 1:position + horizon + 1]
            future_valid = all(_valid_bar(future.iloc[i]) for i in range(len(future)))
            expected = market_dates[market_position + 1:market_position + horizon + 1]
            exact = len(expected) == horizon and list(pd.to_datetime(future.timestamp)) == expected
            complete = future_valid and exact
            status = "complete" if complete else ("invalid_future_ohlcv" if not future_valid else "missing_benchmark_session")
        entry = close_return = relative = mfe = mae = np.nan
        endpoint_gain: float = np.nan
        if complete:
            future = stock.iloc[position + 1:position + horizon + 1]
            entry = float(future.iloc[0].open); gross = float(future.iloc[-1].close) / entry
            market_gross = float(market.iloc[market_position + horizon].close) / float(market.iloc[market_position + 1].open)
            close_return = gross - 1; relative = math.log(gross) - math.log(market_gross)
            mfe = float(future.high.max()) / entry - 1; mae = float(future.low.min()) / entry - 1
            endpoint_gain = float(close_return >= .25)
        result.update({
            f"complete_{horizon}": complete, f"status_{horizon}": status,
            f"entry_open_{horizon}": entry, f"endpoint_close_return_{horizon}": close_return,
            f"benchmark_relative_log_return_{horizon}": relative,
            f"endpoint_gain_25pct_{horizon}": endpoint_gain,
            f"maximum_favorable_excursion_{horizon}": mfe,
            f"maximum_adverse_excursion_{horizon}": mae,
        })
        if horizon == 20:
            code = -1
            if complete:
                code = 0; upper, lower = entry + 2 * atr, entry - atr
                for future_row in future.itertuples(index=False):
                    up, down = float(future_row.high) >= upper, float(future_row.low) <= lower
                    if up or down:
                        code = 3 if up and down else (1 if up else 2); break
            result["barrier_code_20"] = code
    return result


def _fixture(rows: int = 340) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2020-01-01", periods=rows); close = np.full(rows, 100.)
    open_ = np.full(rows, 100.); high = np.full(rows, 101.); low = np.full(rows, 99.)
    position = 252; close[position] = 104.; high[position] = 105.
    open_[position + 1] = close[position + 1] = 110.; high[position + 1] = 112.; low[position + 1] = 108.
    if rows > position + 20: close[position + 20] = 121.; high[position + 20] = 122.; low[position + 20] = 98.
    if rows > position + 60: close[position + 60] = 132.; high[position + 60] = 134.; low[position + 60] = 97.
    stock = pd.DataFrame({"timestamp": dates, "open": open_, "high": np.maximum.reduce((high, open_, close)),
                          "low": np.minimum.reduce((low, open_, close)), "close": close, "volume": 20_000.})
    market_close = np.linspace(200., 230., rows); market_open = market_close.copy(); market_open[position + 1] = 210.
    market = pd.DataFrame({"date": dates, "open": market_open, "high": np.maximum(market_open, market_close) + 1,
                           "low": np.minimum(market_open, market_close) - 1, "close": market_close})
    return stock, market


def _compare(expected: Mapping[str, Any], observed: pd.Series) -> None:
    if set(expected) != set(observed.index):
        raise SyntheticGateError("scalar/vector columns differ")
    for name, value in expected.items():
        actual = observed[name]
        if isinstance(value, (float, np.floating)):
            if math.isnan(float(value)) and pd.isna(actual): continue
            if not math.isclose(float(value), float(actual), rel_tol=0., abs_tol=TOLERANCE):
                raise SyntheticGateError(f"numeric difference: {name}")
        elif actual != value: raise SyntheticGateError(f"value difference: {name}")


def execute(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True); output = repository / OUTPUT_RELATIVE
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=repository,
        capture_output=True, text=True, check=True,
    ).stdout
    if status: raise SyntheticGateError("clean worktree required")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository, capture_output=True, text=True, check=True,
    ).stdout.strip()
    contract = base._read(repository / CONTRACT_RELATIVE); digest = contract.pop("contract_digest")
    if digest != stable_hash(contract): raise SyntheticGateError("contract seal differs")
    runtime = {name: _sha(repository / name) for name in RUNTIME_FILES}
    checks = []
    stock, market = _fixture(); vector = symbol_post_signal_panel(stock, market, "SYN")
    _compare(scalar_row(stock, market, 252), vector.loc[vector.signal_position == 252].iloc[0]); checks.append("exact_scalar_vector")
    changed = stock.copy(); changed.loc[272, ["open", "high", "low", "close"]] = [150., 151., 149., 150.]
    changed_row = symbol_post_signal_panel(changed, market, "SYN").loc[lambda x: x.signal_position == 252].iloc[0]
    causal_columns = [name for name in vector if not any(token in name for token in (
        "complete_", "status_", "entry_open_", "endpoint_close_", "benchmark_relative_log_",
        "endpoint_gain_", "maximum_favorable_", "maximum_adverse_", "barrier_code_",
    ))]
    original_row = vector.loc[vector.signal_position == 252].iloc[0]
    if any(original_row[name] != changed_row[name] for name in causal_columns) \
            or original_row.endpoint_close_return_20 == changed_row.endpoint_close_return_20:
        raise SyntheticGateError("future mutation isolation differs")
    checks.append("future_mutation_isolation")
    no_signal = stock.copy(); no_signal.loc[252, ["open", "high", "low", "close"]] = [100., 101., 99., 100.]
    no_signal_row = symbol_post_signal_panel(no_signal, market, "SYN").loc[lambda x: x.signal_position == 252].iloc[0]
    if no_signal_row.up_close_signal_event or no_signal_row.bullish_range_expansion_signal_event:
        raise SyntheticGateError("signal-day mutation did not alter signal identity")
    checks.append("signal_day_mutation_binding")
    missing_market = market.drop(index=260).reset_index(drop=True)
    _compare(scalar_row(stock, missing_market, 252), symbol_post_signal_panel(stock, missing_market, "SYN").loc[lambda x: x.signal_position == 252].iloc[0])
    checks.append("missing_market_censor")
    invalid = stock.copy(); invalid.loc[260, "high"] = 1.
    _compare(scalar_row(invalid, market, 252), symbol_post_signal_panel(invalid, market, "SYN").loc[lambda x: x.signal_position == 252].iloc[0])
    checks.append("invalid_future_censor")
    short_stock, short_market = _fixture(270)
    _compare(scalar_row(short_stock, short_market, 252), symbol_post_signal_panel(short_stock, short_market, "SYN").loc[lambda x: x.signal_position == 252].iloc[0])
    checks.append("source_end_censor")
    ambiguous = stock.copy(); ambiguous.loc[253, ["open", "high", "low", "close"]] = [110., 125., 90., 110.]
    ambiguous_vector = symbol_post_signal_panel(ambiguous, market, "SYN").loc[lambda x: x.signal_position == 252].iloc[0]
    _compare(scalar_row(ambiguous, market, 252), ambiguous_vector)
    if ambiguous_vector.barrier_code_20 != 3: raise SyntheticGateError("same-bar ambiguity differs")
    checks.append("same_bar_barrier_ambiguity")
    state = {
        "schema_version": SCHEMA, "status": "verified", "passed": True,
        "implementation_h0": head, "contract_digest": digest, "runtime_files": runtime, "checks": checks,
        "scalar_vector_cases": 5, "future_mutation_cases": 1, "signal_mutation_cases": 1,
        "real_source_rows_accessed": False, "real_post_signal_outcomes_accessed": False,
        "production_promotion_authorized": False,
    }
    result = {**state, "verification_digest": stable_hash(state), "created_at": _now()}
    if output.exists():
        existing = base._read(output / "VERIFIED.json")
        deterministic = {k: v for k, v in existing.items() if k not in {"verification_digest", "created_at"}}
        if existing.get("verification_digest") != stable_hash(deterministic) or deterministic != state:
            raise SyntheticGateError("existing synthetic receipt differs")
        return existing
    output.mkdir(parents=True); smoke._atomic_json(output / "VERIFIED.json", result); return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--repository", type=Path, required=True)
    args = parser.parse_args(argv); print(json.dumps(execute(args.repository), indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
