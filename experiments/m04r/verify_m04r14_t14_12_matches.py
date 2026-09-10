"""Independently verify every T14-12 outcome-blind matched-control identity."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
from itertools import product
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

from experiments.m04r import m04r14_t14_10_wf03_feasibility as base
from experiments.m04r import m04r14_t14_12_matches as target


SCHEMA = "m04r14-t14-12-post-signal-matches-verification-v1"
VERIFY_MATCH_COLUMNS = (
    "prior_return_63", "prior_volatility_20", "prior_close", "prior_median_dollar_volume_20",
)
VERIFY_MARKET_COLUMNS = (
    "benchmark_signal_day_return", "benchmark_return_20", "benchmark_return_63", "benchmark_volatility_20",
)
VERIFY_SIGNALS = {
    "up_close_4pct": ("up_close_at_risk", "up_close_4pct", "up_close_signal_event"),
    "bullish_range_expansion_4pct": (
        "bullish_range_expansion_at_risk", "bullish_range_expansion_4pct",
        "bullish_range_expansion_signal_event",
    ),
}
VERIFY_COLUMNS = (
    "symbol", "signal_date", "signal_position", "investable",
    *VERIFY_MATCH_COLUMNS, *VERIFY_MARKET_COLUMNS,
    *(column for values in VERIFY_SIGNALS.values() for column in values),
)


class MatchVerificationError(RuntimeError): pass


def _now() -> str: return datetime.now(timezone.utc).isoformat()


def _sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20): digest.update(block)
    return digest.hexdigest()


def _valid(value: Mapping[str, Any], key: str = "result_digest", *, timing: bool = False) -> bool:
    omitted = {key, "created_at"} | ({"elapsed_seconds"} if timing else set())
    return value.get(key) == stable_hash({name: item for name, item in value.items() if name not in omitted})


def _deciles(values: np.ndarray) -> np.ndarray:
    ranks = pd.Series(values).rank(method="average").to_numpy(float)
    return np.clip(np.ceil(10.0 * ranks / len(values)), 1, 10).astype(np.uint8)


def _event_id(contract: str, signal: str, symbol: str, date_text: str) -> str:
    return sha256(f"{contract}|{signal}|{symbol}|{date_text}".encode("utf-8")).hexdigest()


def _selection_digest(contract: str, signal_id: str, symbol: str) -> str:
    return sha256(f"{contract}|{signal_id}|{symbol}".encode("utf-8")).hexdigest()


def _select(
    names: np.ndarray, deciles: np.ndarray, eligible: np.ndarray, event_symbol: str,
    event_deciles: np.ndarray, contract: str, signal_id: str,
    buckets: Mapping[tuple[int, ...], Sequence[int]] | None = None,
) -> tuple[list[int], str]:
    if buckets is None:
        built: dict[tuple[int, ...], list[int]] = defaultdict(list)
        for index in np.flatnonzero(eligible):
            built[tuple(int(value) for value in deciles[index])].append(int(index))
        buckets = built
    key = tuple(int(value) for value in event_deciles); exact = buckets.get(key, [])
    if len(exact) >= 5:
        pool, tier = exact, "exact_all_four_deciles"
    else:
        pool = []
        axes = [range(max(1, value - 1), min(10, value + 1) + 1) for value in key]
        for neighbor in product(*axes): pool.extend(buckets.get(tuple(neighbor), []))
        if len(pool) >= 5: tier = "within_one_bucket_all_four"
        else: pool, tier = list(np.flatnonzero(eligible)), "same_date_unmatched"
    pool = [index for index in pool if str(names[index]) != event_symbol]
    pool.sort(key=lambda index: (_selection_digest(contract, signal_id, str(names[index])), str(names[index])))
    return pool[:5], tier


def _equal(left: Any, right: Any) -> bool:
    if pd.isna(left) and pd.isna(right): return True
    if isinstance(left, (float, np.floating)) or isinstance(right, (float, np.floating)):
        return bool(float(left) == float(right))
    return bool(left == right)


def _assert_row(observed: pd.Series, expected: Mapping[str, Any], label: str) -> None:
    if set(observed.index) != set(expected): raise MatchVerificationError(f"{label} schema differs")
    for name, value in expected.items():
        if not _equal(observed[name], value): raise MatchVerificationError(f"{label} differs: {name}")


def _read_causal_panel(repository: Path) -> pd.DataFrame:
    frames = [
        pd.read_parquet(
            repository / target.panel_stage.CACHE_RELATIVE / f"shard-{shard:02d}" / "daily-panel.parquet",
            columns=list(VERIFY_COLUMNS),
        ) for shard in range(target.panel_stage.SHARDS)
    ]
    frame = pd.concat(frames, ignore_index=True); del frames
    frame["signal_date"] = pd.to_datetime(frame.signal_date)
    frame = frame.sort_values(["signal_date", "symbol"], kind="stable").reset_index(drop=True)
    if frame.duplicated(["signal_date", "symbol"]).any():
        raise MatchVerificationError("duplicate symbol/date panel rows")
    return frame


def execute(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    if subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=repository,
                      capture_output=True, text=True, check=True).stdout:
        raise MatchVerificationError("clean worktree required")
    started = perf_counter(); prereg, _ = target.validate_preregistration(repository)
    root = repository / target.OUTPUT_RELATIVE; seal = base._read(root / "SEALED.json")
    if not target._valid(seal, timing=True) or seal.get("file_manifest") != target._manifest(root, ("coverage.json",)):
        raise MatchVerificationError("aggregate seal differs")
    if seal.get("outcome_columns_read") != [] or seal.get("inference_accessed") is not False:
        raise MatchVerificationError("outcome-blind boundary differs")
    if tuple(target.CAUSAL_COLUMNS) != VERIFY_COLUMNS:
        raise MatchVerificationError("producer and verifier causal-column contracts differ")
    if tuple(target.MATCH_COLUMNS) != VERIFY_MATCH_COLUMNS \
            or tuple(target.MARKET_COLUMNS) != VERIFY_MARKET_COLUMNS \
            or target.SIGNALS != VERIFY_SIGNALS:
        raise MatchVerificationError("producer and verifier matching semantics differ")
    frame = _read_causal_panel(repository)
    if len(frame) != int(prereg["verified_panel"]["panel_rows"]): raise MatchVerificationError("panel count differs")
    contract = prereg["contract_digest"]; total_events = total_controls = total_input = 0
    coverage: dict[str, int] = defaultdict(int); year_digests = []
    grouped_years = frame.groupby(frame.signal_date.dt.year, sort=True)
    for year in prereg["years"]:
        year_root = repository / target.CACHE_RELATIVE / f"year-{year}"
        year_seal = base._read(year_root / "YEAR_SEALED.json")
        if not target._valid(year_seal, timing=True) \
                or year_seal.get("file_manifest") != target._manifest(year_root, target.YEAR_FILES):
            raise MatchVerificationError(f"year seal differs: {year}")
        if year_seal.get("outcome_columns_read") != [] or year_seal.get("inference_accessed") is not False:
            raise MatchVerificationError(f"year outcome boundary differs: {year}")
        year_digests.append(year_seal["result_digest"])
        events = pd.read_parquet(year_root / target.YEAR_FILES[0])
        controls = pd.read_parquet(year_root / target.YEAR_FILES[1])
        event_cursor = control_cursor = 0
        selected_year = grouped_years.get_group(year)
        total_input += len(selected_year); dates = 0
        year_coverage: dict[str, int] = defaultdict(int)
        for signal_date, group in selected_year.groupby("signal_date", sort=True):
            dates += 1; date_text = pd.Timestamp(signal_date).date().isoformat()
            if any(group[column].nunique(dropna=False) != 1 for column in VERIFY_MARKET_COLUMNS):
                raise MatchVerificationError(f"same-date market state differs: {date_text}")
            for population in ("broad", "investable"):
                current = group if population == "broad" else group.loc[group.investable]
                if current.empty: continue
                current = current.reset_index(drop=True); names = current.symbol.astype(str).to_numpy(object)
                deciles = np.column_stack([_deciles(current[column].to_numpy(float)) for column in VERIFY_MATCH_COLUMNS])
                for signal_name, (risk_column, raw_column, event_column) in VERIFY_SIGNALS.items():
                    eligible = current[risk_column].to_numpy(bool) & ~current[raw_column].to_numpy(bool)
                    buckets: dict[tuple[int, ...], list[int]] = defaultdict(list)
                    for eligible_index in np.flatnonzero(eligible):
                        buckets[tuple(int(value) for value in deciles[eligible_index])].append(int(eligible_index))
                    for event_index in np.flatnonzero(current[event_column].to_numpy(bool)):
                        event_symbol = str(names[event_index]); signal_id = _event_id(
                            contract, signal_name, event_symbol, date_text,
                        )
                        selected, tier = _select(
                            names, deciles, eligible, event_symbol, deciles[event_index], contract, signal_id, buckets,
                        )
                        count = len(selected); coverage_key = f"{population}|{signal_name}|{tier}|{count}"
                        coverage[coverage_key] += 1; year_coverage[coverage_key] += 1
                        if event_cursor >= len(events): raise MatchVerificationError(f"event rows truncated: {year}")
                        expected_event = {
                            "population": population, "signal_name": signal_name, "signal_id": signal_id,
                            "event_symbol": event_symbol, "signal_date": pd.Timestamp(signal_date),
                            "event_signal_position": int(current.signal_position.iloc[event_index]),
                            "match_tier": tier, "control_count": count, "full_match": count == 5,
                            **{f"event_{name}_decile": int(deciles[event_index, i]) for i, name in enumerate(VERIFY_MATCH_COLUMNS)},
                            **{name: float(current[name].iloc[event_index]) for name in VERIFY_MARKET_COLUMNS},
                        }
                        _assert_row(events.iloc[event_cursor], expected_event, "event match"); event_cursor += 1
                        for rank, control_index in enumerate(selected, 1):
                            if control_cursor >= len(controls): raise MatchVerificationError(f"control rows truncated: {year}")
                            control_symbol = str(names[control_index]); control_deciles = deciles[control_index]
                            expected_control = {
                                "population": population, "signal_name": signal_name, "signal_id": signal_id,
                                "event_symbol": event_symbol, "control_symbol": control_symbol,
                                "signal_date": pd.Timestamp(signal_date),
                                "control_signal_position": int(current.signal_position.iloc[control_index]),
                                "match_rank": rank, "match_tier": tier,
                                "selection_digest": _selection_digest(contract, signal_id, control_symbol),
                                "maximum_decile_distance": int(np.max(np.abs(deciles[event_index].astype(int) - control_deciles.astype(int)))),
                                **{f"event_{name}_decile": int(deciles[event_index, i]) for i, name in enumerate(VERIFY_MATCH_COLUMNS)},
                                **{f"control_{name}_decile": int(control_deciles[i]) for i, name in enumerate(VERIFY_MATCH_COLUMNS)},
                            }
                            _assert_row(controls.iloc[control_cursor], expected_control, "control identity"); control_cursor += 1
        if event_cursor != len(events) or control_cursor != len(controls):
            raise MatchVerificationError(f"unexpected trailing rows: {year}")
        if dates != int(year_seal["signal_dates"]) or len(selected_year) != int(year_seal["input_panel_rows"]) \
                or len(events) != int(year_seal["event_match_rows"]) \
                or len(controls) != int(year_seal["control_identity_rows"]):
            raise MatchVerificationError(f"year accounting differs: {year}")
        if dict(sorted(year_coverage.items())) != year_seal["coverage"]:
            raise MatchVerificationError(f"year coverage differs: {year}")
        total_events += len(events); total_controls += len(controls)
    if year_digests != seal["year_result_digests"] or total_input != int(seal["input_panel_rows"]) \
            or total_events != int(seal["event_match_rows"]) \
            or total_controls != int(seal["control_identity_rows"]):
        raise MatchVerificationError("aggregate inventory differs")
    coverage_receipt = base._read(root / "coverage.json")
    fully = sum(value for key, value in coverage.items() if key.endswith("|5"))
    expected_coverage = {
        "event_match_rows": total_events, "fully_matched_event_rows": fully,
        "full_match_coverage": fully / total_events, "control_identity_rows": total_controls,
        "by_population_signal_tier_count": dict(sorted(coverage.items())), "shortfalls_retained": True,
    }
    if not target._valid(coverage_receipt) or any(coverage_receipt.get(key) != value for key, value in expected_coverage.items()):
        raise MatchVerificationError("aggregate coverage differs")
    state = {
        "schema_version": SCHEMA, "status": "verified", "passed": True,
        "store_result_digest": seal["result_digest"], "store_result_sha256": _sha(root / "SEALED.json"),
        "verified_input_panel_rows": total_input, "verified_event_match_rows": total_events,
        "verified_control_identity_rows": total_controls, "verified_years": len(prereg["years"]),
        "gates": {
            "all_year_seals_valid": True, "all_deciles_reconstructed": True,
            "all_at_risk_non_signal_conditions_reconstructed": True,
            "all_control_identities_and_order_reconstructed": True,
            "all_shortfalls_and_coverage_reconstructed": True,
            "outcome_and_inference_boundary_remained_closed": True,
        },
        "outcome_columns_read": [], "inference_accessed": False,
        "production_promotion_authorized": False, "elapsed_seconds": perf_counter() - started,
        "created_at": _now(),
    }
    result = {**state, "verification_digest": stable_hash(state)}
    output = repository / target.VERIFICATION_RELATIVE
    if output.exists(): return base._read(output / "VERIFIED.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        target.smoke._atomic_json(temporary / "VERIFIED.json", result); os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True); raise
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--repository", type=Path, required=True)
    args = parser.parse_args(argv); print(json.dumps(execute(args.repository), indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
