from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from html import escape
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml

from .adapters import OHLCVSource, file_fingerprint
from .causal_prefix import causal_prefix_digest
from .certified_packed_search import certified_packed_search_contract
from .gate12_registry import LIQUIDITY_STRATA, QUALITY_TIERS
from .packed_bound_search import DEFAULT_ROUTE_QUOTAS, packed_bound_search_contract
from .types import EpisodeKey, InstrumentKey, stable_hash


SCHEMA_VERSION = "m04r-untouched-authority-registry-v1"
LEDGER_SCHEMA_VERSION = "m04r-contamination-ledger-v1"
UNIVERSE_SCHEMA_VERSION = "m04r-selection-universe-v1"
TRANSCRIPT_SCHEMA_VERSION = "m04r-selection-transcript-v1"
SYMBOLS_PER_CELL = 5
EXPECTED_SYMBOLS = len(QUALITY_TIERS) * len(LIQUIDITY_STRATA) * SYMBOLS_PER_CELL
EXPECTED_CASES = EXPECTED_SYMBOLS * 2
TARGET_SCHEDULE = (
    ("down", "2010s"),
    ("sideways", "2010s"),
    ("up", "2010s"),
    ("down", "2020s"),
    ("sideways", "2020s"),
)
FORBIDDEN_OUTCOME_KEYS = {
    "outcome", "outcomes", "forward_return", "forward_returns", "result",
    "winner", "loser", "setup", "setup_label", "profit", "loss",
}


@dataclass(frozen=True)
class M04RValidationRegistry:
    passed: bool
    metrics: dict[str, Any]
    failures: tuple[str, ...]
    cases: pd.DataFrame
    universe: pd.DataFrame
    transcript: pd.DataFrame
    contamination_ledger: dict[str, Any]


def _era(timestamp: pd.Timestamp) -> str:
    if timestamp.year < 2010:
        return "pre-2010"
    if timestamp.year < 2020:
        return "2010s"
    return "2020s"


def _market_regime(value: float, threshold: float) -> str:
    return "up" if value > threshold else "down" if value < -threshold else "sideways"


def _canonical_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for raw in frame.to_dict(orient="records"):
        row: dict[str, Any] = {}
        for key, value in raw.items():
            if pd.isna(value):
                row[str(key)] = None
            elif isinstance(value, pd.Timestamp):
                row[str(key)] = value.isoformat()
            elif isinstance(value, (np.integer,)):
                row[str(key)] = int(value)
            elif isinstance(value, (np.floating,)):
                row[str(key)] = float(value)
            elif isinstance(value, (np.bool_,)):
                row[str(key)] = bool(value)
            else:
                row[str(key)] = value
        records.append(row)
    return records


def _symbols(values: Iterable[Any]) -> list[str]:
    return sorted({str(value).strip().upper() for value in values if str(value).strip()})


def _ledger_source(name: str, path: Path, symbols: Iterable[Any]) -> dict[str, Any]:
    ordered = _symbols(symbols)
    return {
        "name": name,
        "path": str(path.resolve()),
        "file_sha256": file_fingerprint(path),
        "symbol_count": len(ordered),
        "symbol_digest": stable_hash(ordered),
        "symbols": ordered,
    }


def _extract_ledger_symbols(name: str, path: Path) -> list[str]:
    if name == "gate09-queries":
        frame = pd.read_parquet(path, columns=["query"])
        return _symbols(str(value).split(":", 1)[-1] for value in frame["query"])
    if name.startswith("gate09-candidates:") or name.startswith("external-example:"):
        return _symbols(pd.read_parquet(path, columns=["symbol"])["symbol"])
    if name == "gate12-selected":
        payload = yaml.safe_load(path.read_text()) or {}
        return _symbols(row["symbol"] for row in payload.get("cases_data", []))
    if name == "m04r-batch-selected":
        payload = json.loads(path.read_text())
        return _symbols(row["symbol"] for row in payload.get("cases_data", []))
    if name == "m04r10-design-and-known-aliases":
        payload = yaml.safe_load(path.read_text()) or {}
        values = list(payload.get("symbols", []))
        for group in payload.get("known_identity_alias_groups", []):
            values.extend(group.get("symbols", []))
        return _symbols(values)
    raise ValueError(f"unsupported contamination source: {name}")


def build_contamination_ledger(
    artifact_dir: Path, design_exclusions_path: Path,
) -> dict[str, Any]:
    oracle_dir = artifact_dir / "oracles" / "nasdaq"
    summary_path = oracle_dir / "oracle-summary.parquet"
    summary = pd.read_parquet(summary_path)
    sources = [_ledger_source(
        "gate09-queries", summary_path,
        (str(value).split(":", 1)[-1] for value in summary["query"]),
    )]
    for case_id in sorted(summary.case_id.astype(str).unique()):
        path = oracle_dir / f"{case_id}.parquet"
        sources.append(_ledger_source(
            f"gate09-candidates:{case_id}", path,
            pd.read_parquet(path, columns=["symbol"])["symbol"],
        ))

    gate12_path = artifact_dir / "gate12" / "nasdaq" / "query-registry.yaml"
    gate12 = yaml.safe_load(gate12_path.read_text()) or {}
    sources.append(_ledger_source(
        "gate12-selected", gate12_path,
        (row["symbol"] for row in gate12.get("cases_data", [])),
    ))
    batch_path = artifact_dir / "poc" / "m04r" / "batch-query-registry" / "query-registry.json"
    batch = json.loads(batch_path.read_text())
    sources.append(_ledger_source(
        "m04r-batch-selected", batch_path,
        (row["symbol"] for row in batch.get("cases_data", [])),
    ))
    for path in sorted((artifact_dir / "external-examples").glob("*/coverage.parquet")):
        sources.append(_ledger_source(
            f"external-example:{path.parent.name}", path,
            pd.read_parquet(path, columns=["symbol"])["symbol"],
        ))
    design = yaml.safe_load(design_exclusions_path.read_text()) or {}
    design_symbols = list(design.get("symbols", []))
    for group in design.get("known_identity_alias_groups", []):
        design_symbols.extend(group.get("symbols", []))
    sources.append(_ledger_source(
        "m04r10-design-and-known-aliases", design_exclusions_path, design_symbols,
    ))
    excluded = sorted({symbol for source in sources for symbol in source["symbols"]})
    ledger: dict[str, Any] = {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "normalization": "strip then uppercase; exact source-symbol exclusion",
        "sources": sources,
        "excluded_symbols": excluded,
        "excluded_symbol_count": len(excluded),
        "excluded_symbol_digest": stable_hash(excluded),
        "known_identity_alias_groups": design.get("known_identity_alias_groups", []),
        "alias_limit": design.get("alias_limit"),
    }
    ledger["ledger_digest"] = stable_hash(ledger)
    return ledger


def _benchmark_regimes(
    benchmark: pd.DataFrame, lookback: int, threshold: float,
) -> tuple[dict[pd.Timestamp, str], dict[pd.Timestamp, float]]:
    frame = benchmark[["timestamp", "close"]].copy()
    frame["timestamp"] = pd.to_datetime(frame.timestamp)
    frame["close"] = pd.to_numeric(frame.close, errors="coerce")
    returns = frame.close / frame.close.shift(lookback - 1) - 1.0
    regimes: dict[pd.Timestamp, str] = {}
    values: dict[pd.Timestamp, float] = {}
    for timestamp, value in zip(frame.timestamp, returns, strict=True):
        if np.isfinite(value):
            stamp = pd.Timestamp(timestamp)
            values[stamp] = float(value)
            regimes[stamp] = _market_regime(float(value), threshold)
    return regimes, values


def _candidate_cutoff(
    bars: pd.DataFrame, *, lock_cutoff: pd.Timestamp, target_regime: str,
    target_era: str, regimes: dict[pd.Timestamp, str], seed: str, symbol: str,
    slot: int, lookback: int, minimum_future_sessions: int,
) -> tuple[pd.Timestamp | None, int]:
    locked = bars[pd.to_datetime(bars.timestamp) <= lock_cutoff].reset_index(drop=True)
    if len(locked) == 0:
        return None, 0
    positions = [
        position for position in range(lookback - 1, len(locked) - minimum_future_sessions)
        if _era(pd.Timestamp(locked.timestamp.iloc[position])) == target_era
        and regimes.get(pd.Timestamp(locked.timestamp.iloc[position])) == target_regime
    ]
    if not positions:
        return None, len(locked)
    ranked = sorted(
        positions,
        key=lambda position: (
            sha256(
                f"{seed}:cutoff:{symbol}:{slot}:"
                f"{pd.Timestamp(locked.timestamp.iloc[position]).isoformat()}".encode()
            ).hexdigest(),
            pd.Timestamp(locked.timestamp.iloc[position]),
        ),
    )
    return pd.Timestamp(locked.timestamp.iloc[ranked[0]]), len(locked)


def _selection_universe(
    quality: pd.DataFrame, liquidity: pd.DataFrame, *, excluded: set[str],
    minimum_rows: int, maximum_staleness_days: int,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    required_quality = {"symbol", "first_timestamp", "last_timestamp"}
    required_liquidity = {
        "symbol", "quality_tier", "liquidity_stratum", "rows",
        "median_dollar_volume_252",
    }
    if missing := required_quality.difference(quality.columns):
        raise ValueError(f"quality audit is missing columns: {sorted(missing)}")
    if missing := required_liquidity.difference(liquidity.columns):
        raise ValueError(f"liquidity audit is missing columns: {sorted(missing)}")
    freshness = quality[["symbol", "first_timestamp", "last_timestamp"]].copy()
    freshness["symbol"] = freshness.symbol.astype(str)
    freshness["first_timestamp"] = pd.to_datetime(freshness.first_timestamp)
    freshness["last_timestamp"] = pd.to_datetime(freshness.last_timestamp)
    merged = liquidity.merge(freshness, on="symbol", how="left", validate="one_to_one")
    market_latest = pd.Timestamp(merged.last_timestamp.max())
    floor = market_latest - pd.Timedelta(days=maximum_staleness_days)
    universe = merged[
        merged.quality_tier.astype(str).isin(QUALITY_TIERS)
        & merged.liquidity_stratum.astype(str).isin(LIQUIDITY_STRATA)
        & (merged.rows.astype(int) >= minimum_rows)
        & (merged.last_timestamp >= floor)
        & ~merged.symbol.astype(str).str.upper().isin(excluded)
    ].copy()
    universe = universe[[
        "symbol", "quality_tier", "liquidity_stratum", "rows",
        "median_dollar_volume_252", "first_timestamp", "last_timestamp",
    ]].rename(columns={
        "rows": "rows_at_lock", "first_timestamp": "first_timestamp_at_lock",
        "last_timestamp": "last_timestamp_at_lock",
    })
    universe["symbol"] = universe.symbol.astype(str)
    return universe.sort_values("symbol", kind="stable", ignore_index=True), market_latest


def _descriptor(
    bars: pd.DataFrame, benchmark: pd.DataFrame, cutoff: pd.Timestamp, lookback: int,
    benchmark_return: float, regime_threshold: float, quality: pd.DataFrame,
) -> dict[str, Any]:
    eligible = bars[pd.to_datetime(bars.timestamp) <= cutoff].tail(lookback)
    close = pd.to_numeric(eligible.close, errors="coerce")
    volume = pd.to_numeric(eligible.volume, errors="coerce")
    episode_return = float(close.iloc[-1] / close.iloc[0] - 1.0)
    positive = close.where(close > 0)
    realized_volatility = float(np.log(positive).diff().std() * np.sqrt(252))
    maximum_drawdown = float((close / close.cummax() - 1.0).min())
    volume_fraction = float((volume.notna() & (volume > 0)).mean())
    benchmark_dates = set(pd.to_datetime(benchmark.timestamp))
    overlap_fraction = float(pd.to_datetime(eligible.timestamp).isin(benchmark_dates).mean())
    morphology = (
        "advance" if episode_return >= .25 else
        "decline" if episode_return <= -.20 else "range"
    )
    context = "full" if volume_fraction >= .995 and overlap_fraction >= .995 else "sparse"
    first = pd.to_datetime(quality.first_timestamp)
    last = pd.to_datetime(quality.last_timestamp)
    active = int(((first <= cutoff) & (last >= cutoff)).sum())
    size_band = "small" if active < 4_000 else "middle" if active < 8_000 else "large"
    volatility_band = (
        "low" if realized_volatility < .30 else
        "middle" if realized_volatility < .70 else "high"
    )
    drawdown_band = (
        "shallow" if maximum_drawdown > -.20 else
        "middle" if maximum_drawdown > -.50 else "deep"
    )
    return {
        "episode_return": episode_return,
        "realized_volatility_annualized": realized_volatility,
        "maximum_drawdown": maximum_drawdown,
        "volume_observed_fraction": volume_fraction,
        "benchmark_overlap_fraction": overlap_fraction,
        "benchmark_return": benchmark_return,
        "benchmark_regime": _market_regime(benchmark_return, regime_threshold),
        "era": _era(cutoff),
        "morphology_stratum": morphology,
        "data_context": context,
        "active_source_universe": active,
        "universe_size_band": size_band,
        "volatility_band": volatility_band,
        "drawdown_band": drawdown_band,
    }


def default_search_contract(artifact_dir: Path) -> dict[str, Any]:
    distance_path = artifact_dir / "m04r-distance-v1" / "m04r-distance-v1.json"
    distance = json.loads(distance_path.read_text())
    store_root = artifact_dir / "poc" / "m04r" / "packed-bound-full" / "store"
    generations = sorted((store_root / "generations").glob("*/manifest.json"))
    if len(generations) != 1:
        raise ValueError(f"expected one packed generation; found {len(generations)}")
    manifest_path = generations[0]
    manifest = json.loads(manifest_path.read_text())
    certified = certified_packed_search_contract(
        requested_positions=True, vector_lower_bounds=True, deferred_alignments=True,
    )
    controls = {
        "processes": 8,
        "numba_threads_per_process": 1,
        "exact_workers_per_process": 1,
        "block_rows": 4_096,
        "initial_frontier_rows": 16_384,
        "maximum_frontier_rows": 32_768,
        "seed_rows": 512,
        "requested_positions": True,
        "vector_lower_bounds": True,
        "deferred_alignments": True,
        "sorted_joined_iqr_merge": True,
    }
    return {
        "distance_verification_path": str(distance_path.resolve()),
        "distance_verification_result_digest": distance["result_digest"],
        "distance_verification_file_sha256": file_fingerprint(distance_path),
        "packed_generation_manifest_path": str(manifest_path.resolve()),
        "packed_generation_id": manifest["manifest_digest"],
        "packed_generation_manifest_sha256": file_fingerprint(manifest_path),
        "packed_provenance_digest": manifest["provenance_digest"],
        "packed_bound_contract_digest": manifest["pack_contract_digest"],
        "proposal_contract_digest": packed_bound_search_contract()["digest"],
        "certified_search_contract_digest": certified["digest"],
        "fast_route_quotas": dict(DEFAULT_ROUTE_QUOTAS),
        "certified_route_quotas": {"composite": controls["maximum_frontier_rows"] + 1},
        "controls": controls,
        "request": {
            "top_k": 20, "quality_tiers": ["A", "B"],
            "minimum_history_gap_bars": 60, "max_per_instrument": 3,
            "deduplicate_overlaps": True, "cross_dataset": False,
        },
        "performance_limits": {
            "fast_warm_seconds": 60.0, "fast_cold_seconds": 120.0,
            "certified_p95_seconds": 300.0, "certified_max_seconds": 600.0,
            "rss_mib": 1536.0,
        },
        "authority_root_policy": "write-isolated; candidate result roots forbidden until aggregate truth is sealed",
        "real_forward_outcomes_accessed": False,
    }


def _coverage_failures(cases: pd.DataFrame) -> list[str]:
    failures: list[str] = []
    requirements = {
        "benchmark_regime": {"up", "down", "sideways"},
        "era": {"2010s", "2020s"},
        "morphology_stratum": {"advance", "decline", "range"},
        "data_context": {"full", "sparse"},
    }
    for column, required in requirements.items():
        observed = set(cases[column].astype(str)) if column in cases else set()
        if not required.issubset(observed):
            failures.append(f"{column} coverage is {sorted(observed)}; require {sorted(required)}")
    for column in ("universe_size_band", "volatility_band", "drawdown_band"):
        observed = set(cases[column].astype(str)) if column in cases else set()
        if len(observed) < 2:
            failures.append(f"{column} has only {sorted(observed)}; require at least two strata")
    return failures


def build_m04r_validation_registry(
    source: OHLCVSource, quality: pd.DataFrame, liquidity: pd.DataFrame,
    contamination_ledger: dict[str, Any], search_contract: dict[str, Any], *,
    seed: str = "m04r10-authority-v1", lookback: int = 252,
    minimum_rows: int = 1000, minimum_future_sessions: int = 60,
    maximum_staleness_days: int = 120, regime_threshold: float = .10,
    representation_version: str = "dense-v1",
) -> M04RValidationRegistry:
    if minimum_rows < lookback + minimum_future_sessions:
        raise ValueError("minimum rows cannot support lookback and future gap")
    if not 0 < regime_threshold < 1:
        raise ValueError("regime threshold must be strictly between zero and one")
    instruments = source.instruments()
    if not instruments:
        raise ValueError("source has no instruments")
    dataset_id = instruments[0].dataset_id
    if dataset_id != "nasdaq":
        raise ValueError("M04R-10 untouched authority registry is NASDAQ-only")
    instrument_map = {item.source_symbol: item for item in instruments}
    excluded = set(contamination_ledger["excluded_symbols"])
    universe, market_latest = _selection_universe(
        quality, liquidity, excluded=excluded, minimum_rows=minimum_rows,
        maximum_staleness_days=maximum_staleness_days,
    )
    benchmark = source.load_benchmark()
    if benchmark is None:
        raise ValueError("NASDAQ authority registry requires a benchmark")
    regimes, benchmark_returns = _benchmark_regimes(benchmark, lookback, regime_threshold)
    failures: list[str] = []
    selected: list[dict[str, Any]] = []
    transcript: list[dict[str, Any]] = []
    cache: dict[str, pd.DataFrame] = {}
    for (tier, stratum), cell in universe.groupby(
        ["quality_tier", "liquidity_stratum"], sort=True,
    ):
        used: set[str] = set()
        for slot, (target_regime, target_era) in enumerate(TARGET_SCHEDULE):
            ranked = cell.copy()
            ranked["selection_hash"] = ranked.symbol.map(lambda symbol: sha256(
                f"{seed}:{dataset_id}:{tier}:{stratum}:{slot}:"
                f"{target_regime}:{target_era}:{symbol}".encode()
            ).hexdigest())
            ranked = ranked[~ranked.symbol.isin(used)].sort_values(
                ["selection_hash", "symbol"], kind="stable",
            )
            choice: dict[str, Any] | None = None
            for candidate_rank, row in enumerate(ranked.itertuples(index=False), 1):
                symbol = str(row.symbol)
                if symbol not in cache:
                    cache[symbol] = source.load(instrument_map[symbol])
                bars = cache[symbol]
                lock_cutoff = pd.Timestamp(row.last_timestamp_at_lock)
                cutoff, observed_rows = _candidate_cutoff(
                    bars, lock_cutoff=lock_cutoff, target_regime=target_regime,
                    target_era=target_era, regimes=regimes, seed=seed, symbol=symbol,
                    slot=slot, lookback=lookback,
                    minimum_future_sessions=minimum_future_sessions,
                )
                transcript.append({
                    "quality_tier": str(tier), "liquidity_stratum": str(stratum),
                    "slot": slot, "target_regime": target_regime,
                    "target_era": target_era, "candidate_rank": candidate_rank,
                    "symbol": symbol, "selection_hash": str(row.selection_hash),
                    "rows_at_lock": int(row.rows_at_lock),
                    "observed_rows_through_lock": observed_rows,
                    "last_timestamp_at_lock": lock_cutoff.isoformat(),
                    "eligible": cutoff is not None,
                    "selected_cutoff": cutoff.isoformat() if cutoff is not None else None,
                })
                if observed_rows != int(row.rows_at_lock):
                    failures.append(
                        f"{symbol} has {observed_rows} rows through lock; audit recorded {row.rows_at_lock}"
                    )
                if cutoff is not None:
                    choice = {
                        "selection": row, "slot": slot,
                        "target_regime": target_regime, "target_era": target_era,
                        "historical_cutoff": cutoff,
                    }
                    break
            if choice is None:
                failures.append(f"no eligible {tier}/{stratum} symbol for slot {slot}")
            else:
                selected.append(choice)
                used.add(str(choice["selection"].symbol))

    case_rows: list[dict[str, Any]] = []
    for choice in selected:
        selection = choice["selection"]
        symbol = str(selection.symbol)
        instrument = instrument_map[symbol]
        if symbol not in cache:
            cache[symbol] = source.load(instrument)
        bars = cache[symbol]
        lock_cutoff = pd.Timestamp(selection.last_timestamp_at_lock)
        locked = bars[pd.to_datetime(bars.timestamp) <= lock_cutoff].reset_index(drop=True)
        historical_cutoff = pd.Timestamp(choice["historical_cutoff"])
        for cutoff_role, cutoff in (("historical", historical_cutoff), ("current", lock_cutoff)):
            positions = np.flatnonzero(pd.to_datetime(locked.timestamp).to_numpy() <= cutoff.to_datetime64())
            if not len(positions):
                failures.append(f"{symbol}/{cutoff_role} has no row at cutoff")
                continue
            position = int(positions[-1])
            if position + 1 < lookback:
                failures.append(f"{symbol}/{cutoff_role} lacks lookback")
                continue
            future_sessions = len(locked) - position - 1
            if cutoff_role == "historical" and future_sessions < minimum_future_sessions:
                failures.append(f"{symbol}/historical lacks future exclusion gap")
                continue
            key = EpisodeKey(instrument, cutoff, lookback, representation_version)
            benchmark_return = benchmark_returns.get(cutoff)
            if benchmark_return is None:
                failures.append(f"{symbol}/{cutoff_role} has no benchmark regime")
                continue
            descriptor = _descriptor(
                locked, benchmark, cutoff, lookback, benchmark_return,
                regime_threshold, quality,
            )
            stock_prefix = asdict(causal_prefix_digest(locked, cutoff))
            benchmark_prefix = asdict(causal_prefix_digest(benchmark, cutoff))
            case_rows.append({
                "case_id": f"{dataset_id}-{symbol}-{cutoff_role}-{lookback}",
                "dataset_id": dataset_id, "symbol": symbol,
                "quality_tier": str(selection.quality_tier),
                "liquidity_stratum": str(selection.liquidity_stratum),
                "median_dollar_volume_252": float(selection.median_dollar_volume_252),
                "selection_slot": int(choice["slot"]),
                "selection_target_regime": str(choice["target_regime"]),
                "selection_target_era": str(choice["target_era"]),
                "cutoff_role": cutoff_role, "cutoff": cutoff.isoformat(),
                "cutoff_position_at_lock": position,
                "future_sessions_at_lock": future_sessions,
                "lookback": lookback, "representation_version": representation_version,
                "episode_id": key.id, "stock_prefix": stock_prefix,
                "benchmark_prefix": benchmark_prefix, **descriptor,
            })
    cases = pd.DataFrame(case_rows)
    if len(cases):
        cases = cases.sort_values([
            "quality_tier", "liquidity_stratum", "selection_slot", "cutoff_role",
        ], kind="stable", ignore_index=True)
    transcript_frame = pd.DataFrame(transcript)
    if len(transcript_frame):
        transcript_frame = transcript_frame.sort_values([
            "quality_tier", "liquidity_stratum", "slot", "candidate_rank",
        ], kind="stable", ignore_index=True)
    if len(cases) != EXPECTED_CASES or cases.symbol.nunique() != EXPECTED_SYMBOLS:
        failures.append(
            f"created {len(cases)} cases/{cases.symbol.nunique() if len(cases) else 0} symbols; "
            f"require {EXPECTED_CASES}/{EXPECTED_SYMBOLS}"
        )
    if len(cases) and (cases.case_id.duplicated().any() or cases.episode_id.duplicated().any()):
        failures.append("registry contains duplicate case or episode IDs")
    if len(cases) and set(cases.symbol.str.upper()).intersection(excluded):
        failures.append("registry contains a contaminated symbol")
    if len(cases):
        failures.extend(_coverage_failures(cases))
    universe_records = _canonical_records(universe)
    transcript_records = _canonical_records(transcript_frame)
    case_records = _canonical_records(cases)
    metrics: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION, "dataset": dataset_id, "seed": seed,
        "symbols_per_cell": SYMBOLS_PER_CELL, "symbols": EXPECTED_SYMBOLS,
        "cases": len(cases), "lookback": lookback,
        "minimum_rows": minimum_rows,
        "minimum_future_sessions": minimum_future_sessions,
        "maximum_staleness_days": maximum_staleness_days,
        "regime_threshold": regime_threshold,
        "target_schedule": [list(value) for value in TARGET_SCHEDULE],
        "representation_version": representation_version,
        "market_latest_timestamp_at_lock": market_latest.isoformat(),
        "contamination_ledger_digest": contamination_ledger["ledger_digest"],
        "excluded_symbol_count": contamination_ledger["excluded_symbol_count"],
        "universe_schema_version": UNIVERSE_SCHEMA_VERSION,
        "universe_rows": len(universe),
        "universe_digest": stable_hash(universe_records),
        "transcript_schema_version": TRANSCRIPT_SCHEMA_VERSION,
        "transcript_rows": len(transcript_frame),
        "transcript_digest": stable_hash(transcript_records),
        "selected_symbols": sorted(cases.symbol.unique()) if len(cases) else [],
        "coverage": {
            column: {
                str(key): int(value) for key, value in
                cases[column].value_counts().sort_index().items()
            }
            for column in (
                "quality_tier", "liquidity_stratum", "cutoff_role",
                "benchmark_regime", "era", "morphology_stratum", "data_context",
                "universe_size_band", "volatility_band", "drawdown_band",
            ) if column in cases
        },
        "search_contract": search_contract,
        "outcome_firewall": {
            "allowed_inputs": ["OHLCV", "benchmark OHLCV", "quality", "liquidity", "sealed registry metadata"],
            "forbidden": ["forward outcomes", "trade results", "setup labels", "candidate-system outputs"],
            "real_forward_outcomes_accessed": False,
        },
        "authority_build_command": (
            ".venv/bin/python experiments/m04r/m04r11_build_authorities.py "
            "--registry <m04r10-root>/query-registry.json "
            "--authority-root <write-isolated-m04r11-authorities>"
        ),
        "real_forward_outcomes_accessed": False,
    }
    metrics["registry_digest"] = stable_hash({
        **metrics, "cases_data": case_records,
        "contamination_ledger": contamination_ledger,
    })
    return M04RValidationRegistry(
        not failures, metrics, tuple(dict.fromkeys(failures)), cases, universe,
        transcript_frame, contamination_ledger,
    )


def write_m04r_validation_registry(
    result: M04RValidationRegistry, directory: Path,
) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    result.universe.to_parquet(directory / "selection-universe.parquet", index=False)
    result.transcript.to_parquet(directory / "selection-transcript.parquet", index=False)
    result.cases.to_parquet(directory / "query-registry.parquet", index=False)
    payload = {
        **result.metrics, "passed": result.passed,
        "failures": list(result.failures),
        "contamination_ledger": result.contamination_ledger,
        "cases_data": _canonical_records(result.cases),
    }
    json_path = directory / "query-registry.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    status = "PASS" if result.passed else "FAIL"
    rows = "".join(
        f"<tr><td>{escape(str(row.case_id))}</td><td>{escape(str(row.quality_tier))}/"
        f"{escape(str(row.liquidity_stratum))}</td><td>{escape(str(row.cutoff_role))}</td>"
        f"<td>{escape(str(row.cutoff))}</td><td>{escape(str(row.benchmark_regime))}</td>"
        f"<td>{escape(str(row.morphology_stratum))}</td><td>{escape(str(row.data_context))}</td></tr>"
        for row in result.cases.itertuples(index=False)
    )
    failures = "".join(f"<li>{escape(value)}</li>" for value in result.failures) or "<li>None</li>"
    html_path = directory / "query-registry.html"
    html_path.write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>M04R-10 untouched NASDAQ authority registry</title><style>body{{font-family:system-ui,sans-serif;max-width:1500px;margin:2rem auto;padding:0 1rem;background:#f5f7f8;color:#17202a}}section,header{{background:white;border:1px solid #ddd;border-radius:10px;padding:1.3rem;margin:1rem 0}}table{{border-collapse:collapse;width:100%}}th,td{{padding:.5rem;text-align:left;border-bottom:1px solid #ddd}}.pass{{color:#075}}.fail{{color:#a20}}pre{{overflow:auto;white-space:pre-wrap}}code{{overflow-wrap:anywhere}}</style></head><body><header><h1>M04R-10 untouched authority registry: <span class="{status.lower()}">{status}</span></h1><p>Sixty outcome-blind NASDAQ cases are frozen before search: five symbols per quality/liquidity cell, paired historical/current cutoffs, benchmark-only regime/era targeting, causal prefix hashes and a sealed contamination ledger.</p></header><section><h2>Coverage and contract</h2><pre>{escape(json.dumps(result.metrics, indent=2, sort_keys=True))}</pre></section><section><h2>Cases</h2><table><thead><tr><th>Case</th><th>Cell</th><th>Role</th><th>Cutoff</th><th>Market</th><th>Morphology</th><th>Context</th></tr></thead><tbody>{rows}</tbody></table></section><section><h2>Failures</h2><ul>{failures}</ul></section><section><h2>Known limit</h2><p>{escape(str(result.contamination_ledger.get('alias_limit')))}</p></section></body></html>""")
    return json_path, html_path


def _contains_forbidden_key(value: Any) -> bool:
    if isinstance(value, dict):
        return bool(FORBIDDEN_OUTCOME_KEYS.intersection(map(str, value))) or any(
            _contains_forbidden_key(item) for item in value.values()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_key(item) for item in value)
    return False


def validate_m04r_validation_registry(
    source: OHLCVSource, directory: Path,
) -> tuple[str, ...]:
    json_path = directory / "query-registry.json"
    payload = json.loads(json_path.read_text())
    failures: list[str] = []
    if payload.get("schema_version") != SCHEMA_VERSION:
        failures.append("registry schema version is unsupported")
    cases = payload.get("cases_data")
    ledger = payload.get("contamination_ledger")
    if not isinstance(cases, list) or not isinstance(ledger, dict):
        return ("registry is missing cases or contamination ledger",)
    deterministic = {
        key: value for key, value in payload.items()
        if key not in {"passed", "failures", "registry_digest", "cases_data", "contamination_ledger"}
    }
    # The builder hashes metrics before registry_digest exists.
    expected_registry_digest = stable_hash({
        **{k: v for k, v in deterministic.items() if k != "registry_digest"},
        "cases_data": cases, "contamination_ledger": ledger,
    })
    if expected_registry_digest != payload.get("registry_digest"):
        failures.append("registry digest mismatch")
    ledger_without_digest = {k: v for k, v in ledger.items() if k != "ledger_digest"}
    if stable_hash(ledger_without_digest) != ledger.get("ledger_digest"):
        failures.append("contamination ledger digest mismatch")
    excluded = _symbols(ledger.get("excluded_symbols", []))
    if stable_hash(excluded) != ledger.get("excluded_symbol_digest"):
        failures.append("contamination symbol digest mismatch")
    for source_row in ledger.get("sources", []):
        path = Path(str(source_row.get("path")))
        if not path.exists() or file_fingerprint(path) != source_row.get("file_sha256"):
            failures.append(f"contamination source changed: {source_row.get('name')}")
            continue
        try:
            extracted = _extract_ledger_symbols(str(source_row.get("name")), path)
            if extracted != _symbols(source_row.get("symbols", [])):
                failures.append(f"contamination source symbols differ: {source_row.get('name')}")
        except (KeyError, TypeError, ValueError) as exc:
            failures.append(f"contamination source cannot be reconstructed: {source_row.get('name')}:{exc}")
    source_union = _symbols(
        symbol for source_row in ledger.get("sources", [])
        for symbol in source_row.get("symbols", [])
    )
    if source_union != excluded:
        failures.append("contamination source union differs from excluded symbols")
    universe_path = directory / "selection-universe.parquet"
    transcript_path = directory / "selection-transcript.parquet"
    if not universe_path.exists() or not transcript_path.exists():
        return tuple(dict.fromkeys(failures + ["selection universe or transcript is missing"]))
    universe = pd.read_parquet(universe_path)
    transcript = pd.read_parquet(transcript_path)
    if stable_hash(_canonical_records(universe)) != payload.get("universe_digest"):
        failures.append("selection universe digest mismatch")
    if stable_hash(_canonical_records(transcript)) != payload.get("transcript_digest"):
        failures.append("selection transcript digest mismatch")
    frame = pd.DataFrame(cases)
    if len(frame) != EXPECTED_CASES or frame.symbol.nunique() != EXPECTED_SYMBOLS:
        failures.append("registry does not contain exactly 60 cases/30 symbols")
    if len(frame) and set(frame.symbol.str.upper()).intersection(excluded):
        failures.append("registry contains a contaminated symbol")
    if len(frame) and (frame.case_id.duplicated().any() or frame.episode_id.duplicated().any()):
        failures.append("registry contains duplicate case or episode IDs")
    if len(frame):
        failures.extend(_coverage_failures(frame))
        cells = frame.drop_duplicates("symbol").groupby([
            "quality_tier", "liquidity_stratum",
        ]).size()
        for tier in QUALITY_TIERS:
            for stratum in LIQUIDITY_STRATA:
                if int(cells.get((tier, stratum), 0)) != SYMBOLS_PER_CELL:
                    failures.append(f"registry cell {tier}/{stratum} is unbalanced")
    instrument_map = {item.source_symbol: item for item in source.instruments()}
    benchmark = source.load_benchmark()
    if benchmark is None:
        failures.append("benchmark is unavailable")
    else:
        try:
            regimes, _ = _benchmark_regimes(
                benchmark, int(payload["lookback"]), float(payload["regime_threshold"]),
            )
            cached: dict[str, pd.DataFrame] = {}
            selected_by_cell: dict[tuple[str, str], set[str]] = {}
            for tier in QUALITY_TIERS:
                for stratum in LIQUIDITY_STRATA:
                    used: set[str] = set()
                    cell = universe[
                        (universe.quality_tier.astype(str) == tier)
                        & (universe.liquidity_stratum.astype(str) == stratum)
                    ]
                    for slot, (target_regime, target_era) in enumerate(TARGET_SCHEDULE):
                        ranked = cell[~cell.symbol.astype(str).isin(used)].copy()
                        ranked["selection_hash"] = ranked.symbol.map(lambda symbol: sha256(
                            f"{payload['seed']}:{payload['dataset']}:{tier}:{stratum}:{slot}:"
                            f"{target_regime}:{target_era}:{symbol}".encode()
                        ).hexdigest())
                        ranked = ranked.sort_values(
                            ["selection_hash", "symbol"], kind="stable",
                        ).reset_index(drop=True)
                        observed = transcript[
                            (transcript.quality_tier.astype(str) == tier)
                            & (transcript.liquidity_stratum.astype(str) == stratum)
                            & (transcript.slot.astype(int) == slot)
                        ].sort_values("candidate_rank", kind="stable")
                        if observed.empty:
                            failures.append(f"selection transcript is missing {tier}/{stratum}/{slot}")
                            continue
                        expected_prefix = ranked.head(len(observed))
                        if observed.symbol.astype(str).tolist() != expected_prefix.symbol.astype(str).tolist():
                            failures.append(f"selection hash order mismatch: {tier}/{stratum}/{slot}")
                        if observed.candidate_rank.astype(int).tolist() != list(range(1, len(observed) + 1)):
                            failures.append(f"selection ranks are not contiguous: {tier}/{stratum}/{slot}")
                        for item in observed.itertuples(index=False):
                            symbol = str(item.symbol)
                            instrument = instrument_map.get(symbol)
                            if instrument is None:
                                failures.append(f"transcript symbol is unavailable: {symbol}")
                                continue
                            if symbol not in cached:
                                cached[symbol] = source.load(instrument)
                            cutoff, rows = _candidate_cutoff(
                                cached[symbol], lock_cutoff=pd.Timestamp(item.last_timestamp_at_lock),
                                target_regime=target_regime, target_era=target_era,
                                regimes=regimes, seed=str(payload["seed"]), symbol=symbol,
                                slot=slot, lookback=int(payload["lookback"]),
                                minimum_future_sessions=int(payload["minimum_future_sessions"]),
                            )
                            cutoff_text = cutoff.isoformat() if cutoff is not None else None
                            stored_cutoff = None if pd.isna(item.selected_cutoff) else str(item.selected_cutoff)
                            if rows != int(item.observed_rows_through_lock):
                                failures.append(f"selection row lock mismatch: {symbol}/{slot}")
                            if bool(item.eligible) != (cutoff is not None) or stored_cutoff != cutoff_text:
                                failures.append(f"selection eligibility mismatch: {symbol}/{slot}")
                        last = observed.iloc[-1]
                        if not bool(last.eligible):
                            failures.append(f"selection transcript has no eligible terminus: {tier}/{stratum}/{slot}")
                        else:
                            used.add(str(last.symbol))
                    selected_by_cell[(tier, stratum)] = used
            selected_from_cases = {
                (str(tier), str(stratum)): set(group.symbol.astype(str))
                for (tier, stratum), group in frame.groupby(
                    ["quality_tier", "liquidity_stratum"], sort=True,
                )
            }
            if selected_by_cell != selected_from_cases:
                failures.append("selection transcript terminuses differ from registry cases")
        except (KeyError, TypeError, ValueError) as exc:
            failures.append(f"selection transcript cannot be reconstructed: {exc}")
        for row in frame.itertuples(index=False):
            instrument = instrument_map.get(str(row.symbol))
            if instrument is None:
                failures.append(f"selected symbol is unavailable: {row.symbol}")
                continue
            key = EpisodeKey(
                instrument, pd.Timestamp(row.cutoff), int(row.lookback),
                str(row.representation_version),
            )
            if key.id != str(row.episode_id):
                failures.append(f"episode identity mismatch: {row.case_id}")
            if asdict(causal_prefix_digest(
                source.load(instrument), str(row.cutoff),
            )) != row.stock_prefix:
                failures.append(f"stock causal prefix mismatch: {row.case_id}")
            if asdict(causal_prefix_digest(
                benchmark, str(row.cutoff),
            )) != row.benchmark_prefix:
                failures.append(f"benchmark causal prefix mismatch: {row.case_id}")
    if _contains_forbidden_key(payload):
        failures.append("registry contains a forbidden outcome/setup field")
    if payload.get("real_forward_outcomes_accessed") is not False:
        failures.append("registry outcome-access marker differs")
    return tuple(dict.fromkeys(failures))
