from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .causal_prefix import causal_prefix_digest
from .types import EpisodeKey, InstrumentKey


QUALITY_CELLS = ("A", "B")
LIQUIDITY_CELLS = ("low", "middle", "high")
CANONICAL_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")


class WalkForwardRegistryError(RuntimeError):
    pass


@dataclass(frozen=True)
class RegistryBuild:
    candidates: pd.DataFrame
    queries: pd.DataFrame
    accounting: pd.DataFrame
    sources: pd.DataFrame
    benchmark_prefixes: pd.DataFrame


def canonical_month_cutoffs(
    benchmark: pd.DataFrame, start: str, end: str,
) -> tuple[pd.Timestamp, ...]:
    if "timestamp" not in benchmark:
        raise WalkForwardRegistryError("benchmark timestamp is missing")
    timestamps = pd.to_datetime(benchmark.timestamp, errors="coerce")
    if timestamps.isna().any() or timestamps.duplicated().any() \
            or not timestamps.is_monotonic_increasing:
        raise WalkForwardRegistryError("benchmark timestamps are not unique and ordered")
    bounded = timestamps[(timestamps >= pd.Timestamp(start)) & (timestamps <= pd.Timestamp(end))]
    if bounded.empty:
        raise WalkForwardRegistryError("benchmark contains no registry sessions")
    cutoffs = bounded.groupby(bounded.dt.to_period("M"), sort=True).max().tolist()
    expected = pd.period_range(pd.Timestamp(start), pd.Timestamp(end), freq="M")
    observed = pd.PeriodIndex(cutoffs, freq="M")
    if not observed.equals(expected):
        missing = expected.difference(observed).astype(str).tolist()
        raise WalkForwardRegistryError(f"benchmark registry months are incomplete: {missing}")
    return tuple(pd.Timestamp(value) for value in cutoffs)


def selection_digest(
    contract_digest: str, dataset_id: str, cutoff: str,
    quality_tier: str, liquidity_stratum: str, symbol: str,
) -> str:
    payload = [
        contract_digest, dataset_id, cutoff, quality_tier,
        liquidity_stratum, symbol,
    ]
    return sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _locked_source_check(
    frame: pd.DataFrame, path: Path, lock: Mapping[str, Any], timestamps: pd.Series,
) -> dict[str, Any]:
    locked_rows = int(lock["rows_at_lock"])
    locked_first = pd.Timestamp(lock["first_timestamp_at_lock"])
    locked_last = pd.Timestamp(lock["last_timestamp_at_lock"])
    through_lock = timestamps <= locked_last
    rows_through_lock = int(through_lock.sum())
    timestamp_identity = bool(
        rows_through_lock == locked_rows
        and len(timestamps)
        and pd.Timestamp(timestamps.iloc[0]) == locked_first
        and rows_through_lock > 0
        and pd.Timestamp(timestamps.iloc[rows_through_lock - 1]) == locked_last
        and not bool((timestamps.iloc[rows_through_lock:] <= locked_last).any())
    )
    unchanged_at_lock = len(frame) == locked_rows and pd.Timestamp(timestamps.iloc[-1]) == locked_last
    current_hash = _file_sha256(path) if unchanged_at_lock else None
    raw_lock_match = current_hash == str(lock["source_hash_at_lock"]) if unchanged_at_lock else None
    return {
        "symbol": str(lock["symbol"]),
        "source_path": str(path.resolve()),
        "rows_used_through_coverage": None,
        "rows_at_lock": locked_rows,
        "rows_through_lock": rows_through_lock,
        "first_timestamp_at_lock": locked_first.isoformat(),
        "last_timestamp_at_lock": locked_last.isoformat(),
        "coverage_last_timestamp": None,
        "source_hash_at_lock": str(lock["source_hash_at_lock"]),
        "current_source_sha256": current_hash,
        "timestamp_identity_through_lock": timestamp_identity,
        "unchanged_file_raw_hash_matches_lock": raw_lock_match,
        "error": None,
    }


def measure_symbol_prefixes(
    symbol: str, path: Path, lock: Mapping[str, Any],
    cutoffs: Sequence[pd.Timestamp], locked_coverage_cutoff: pd.Timestamp,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        frame = pd.read_parquet(path)
        if "timestamp" not in frame and "date" in frame:
            frame = frame.rename(columns={"date": "timestamp"})
        missing_columns = set(CANONICAL_COLUMNS).difference(frame.columns)
        if missing_columns:
            raise WalkForwardRegistryError(f"missing columns {sorted(missing_columns)}")
        frame = frame.loc[:, CANONICAL_COLUMNS].copy()
        timestamps = pd.to_datetime(frame.timestamp, errors="coerce")
        if timestamps.isna().any() or timestamps.duplicated().any() \
                or not timestamps.is_monotonic_increasing:
            raise WalkForwardRegistryError("timestamps are invalid, duplicate or reordered")
        source = _locked_source_check(frame, path, lock, timestamps)
        if not source["timestamp_identity_through_lock"]:
            raise WalkForwardRegistryError("timestamp identity differs through source lock")
        if source["unchanged_file_raw_hash_matches_lock"] is False:
            raise WalkForwardRegistryError("unchanged source raw hash differs from source lock")
        locked_last = pd.Timestamp(lock["last_timestamp_at_lock"])
        if bool(((timestamps > locked_last) & (timestamps <= locked_coverage_cutoff)).any()):
            raise WalkForwardRegistryError("post-lock rows entered the locked coverage interval")
        if pd.Timestamp(timestamps.iloc[-1]) > locked_coverage_cutoff:
            keep = timestamps <= locked_coverage_cutoff
            frame = frame.loc[keep].reset_index(drop=True)
            timestamps = timestamps.loc[keep].reset_index(drop=True)
        if frame.empty:
            raise WalkForwardRegistryError("no rows through locked coverage cutoff")
        source["rows_used_through_coverage"] = len(frame)
        source["coverage_last_timestamp"] = pd.Timestamp(timestamps.iloc[-1]).isoformat()

        numeric = frame.loc[:, CANONICAL_COLUMNS[1:]].apply(pd.to_numeric, errors="coerce")
        prices = numeric.loc[:, ["open", "high", "low", "close"]]
        missing = pd.concat(
            [timestamps.rename("timestamp"), numeric], axis=1,
        ).isna().any(axis=1).to_numpy(dtype=np.int64)
        invalid = (
            (numeric.low > prices.loc[:, ["open", "close", "high"]].min(axis=1))
            | (numeric.high < prices.loc[:, ["open", "close", "low"]].max(axis=1))
        ).to_numpy(dtype=np.int64)
        nonpositive = (prices <= 0).any(axis=1).to_numpy(dtype=np.int64)
        negative_volume = (numeric.volume < 0).to_numpy(dtype=np.int64)
        zero_volume = (numeric.volume == 0).to_numpy(dtype=np.int64)
        close = numeric.close.to_numpy(dtype=np.float64)
        log_close = np.full(len(close), np.nan, dtype=np.float64)
        positive_close = close > 0
        log_close[positive_close] = np.log(close[positive_close])
        extreme = np.zeros(len(close), dtype=np.int64)
        extreme[1:] = (np.abs(np.diff(log_close)) > np.log(3.0)).astype(np.int64)
        cumulative = {
            "missing_required": np.cumsum(missing),
            "invalid_ohlc": np.cumsum(invalid),
            "nonpositive_prices": np.cumsum(nonpositive),
            "negative_volume": np.cumsum(negative_volume),
            "zero_volume": np.cumsum(zero_volume),
            "extreme_discontinuities": np.cumsum(extreme),
        }
        date_ns = timestamps.to_numpy(dtype="datetime64[ns]").astype(np.int64)
        dollar_volume = numeric.close.to_numpy(dtype=np.float64) * numeric.volume.to_numpy(dtype=np.float64)
        rows: list[dict[str, Any]] = []
        for cutoff in cutoffs:
            cutoff_ns = pd.Timestamp(cutoff).to_datetime64().astype("datetime64[ns]").astype(np.int64)
            position = int(np.searchsorted(date_ns, cutoff_ns, side="left"))
            if position >= len(date_ns) or date_ns[position] != cutoff_ns or position < 251:
                continue
            critical = sum(int(cumulative[key][position]) for key in (
                "missing_required", "invalid_ohlc", "nonpositive_prices", "negative_volume",
            ))
            if critical:
                continue
            zero = int(cumulative["zero_volume"][position])
            extremes = int(cumulative["extreme_discontinuities"][position])
            tier = "B" if extremes or zero > max(5, int((position + 1) * .02)) else "A"
            liquidity = float(np.median(dollar_volume[position - 251:position + 1]))
            if not np.isfinite(liquidity) or liquidity < 0:
                continue
            rows.append({
                "cutoff": pd.Timestamp(cutoff).isoformat(),
                "symbol": symbol,
                "quality_tier": tier,
                "rows_at_cutoff": position + 1,
                "zero_volume_at_cutoff": zero,
                "extreme_discontinuities_at_cutoff": extremes,
                "median_dollar_volume_252": liquidity,
            })
        source["eligible_cutoffs"] = len(rows)
        return rows, source
    except Exception as exc:
        return [], {
            "symbol": symbol, "source_path": str(path.resolve()),
            "rows_used_through_coverage": None, "rows_at_lock": int(lock["rows_at_lock"]),
            "rows_through_lock": None,
            "first_timestamp_at_lock": str(lock["first_timestamp_at_lock"]),
            "last_timestamp_at_lock": str(lock["last_timestamp_at_lock"]),
            "coverage_last_timestamp": None,
            "source_hash_at_lock": str(lock["source_hash_at_lock"]),
            "current_source_sha256": None,
            "timestamp_identity_through_lock": False,
            "unchanged_file_raw_hash_matches_lock": None,
            "eligible_cutoffs": 0,
            "error": f"{type(exc).__name__}:{exc}",
        }


def _fold(cutoff: pd.Timestamp, folds: Iterable[Mapping[str, str]]) -> tuple[str, str]:
    for fold in folds:
        if pd.Timestamp(fold["start"]) <= cutoff <= pd.Timestamp(fold["end"]):
            return str(fold["fold_id"]), str(fold["role"])
    return "warmup", "baseline_seed_not_scored"


def stratify_and_select(
    candidates: pd.DataFrame, cutoffs: Sequence[pd.Timestamp], *,
    contract_digest: str, folds: Sequence[Mapping[str, str]], target_per_cell: int,
    lookback: int, representation_version: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    enriched: list[pd.DataFrame] = []
    query_rows: list[dict[str, Any]] = []
    accounting_rows: list[dict[str, Any]] = []
    for cutoff in cutoffs:
        cutoff_iso = pd.Timestamp(cutoff).isoformat()
        month = candidates[candidates.cutoff == cutoff_iso].copy()
        month = month.sort_values(["median_dollar_volume_252", "symbol"], kind="stable")
        n = len(month)
        if n:
            ranks = np.arange(1, n + 1, dtype=np.int64)
            percentile = ranks / n
            month["liquidity_rank"] = ranks
            month["liquidity_percentile"] = percentile
            month["liquidity_stratum"] = np.select(
                [percentile <= 1 / 3, percentile <= 2 / 3],
                ["low", "middle"], default="high",
            )
        else:
            month["liquidity_rank"] = pd.Series(dtype="int64")
            month["liquidity_percentile"] = pd.Series(dtype="float64")
            month["liquidity_stratum"] = pd.Series(dtype="object")
        enriched.append(month)
        fold_id, fold_role = _fold(pd.Timestamp(cutoff), folds)
        for quality in QUALITY_CELLS:
            for liquidity in LIQUIDITY_CELLS:
                cell = month[
                    (month.quality_tier == quality)
                    & (month.liquidity_stratum == liquidity)
                ].copy()
                cell["selection_hash"] = cell.symbol.astype(str).map(
                    lambda symbol: selection_digest(
                        contract_digest, "nasdaq", cutoff_iso, quality, liquidity, symbol,
                    )
                )
                cell = cell.sort_values(["selection_hash", "symbol"], kind="stable")
                selected = cell.head(target_per_cell)
                accounting_rows.append({
                    "cutoff": cutoff_iso, "fold_id": fold_id,
                    "quality_tier": quality, "liquidity_stratum": liquidity,
                    "eligible_candidates": len(cell), "selected_queries": len(selected),
                    "target_queries": target_per_cell,
                    "shortfall": target_per_cell - len(selected),
                })
                for rank, row in enumerate(selected.itertuples(index=False), 1):
                    key = EpisodeKey(
                        InstrumentKey("nasdaq", str(row.symbol)), pd.Timestamp(cutoff),
                        lookback, representation_version,
                    )
                    query_rows.append({
                        "case_id": f"nasdaq-{row.symbol}-{pd.Timestamp(cutoff).date()}-{lookback}",
                        "dataset_id": "nasdaq", "symbol": str(row.symbol),
                        "cutoff": cutoff_iso, "fold_id": fold_id, "fold_role": fold_role,
                        "scored": fold_id != "warmup", "quality_tier": quality,
                        "liquidity_stratum": liquidity,
                        "median_dollar_volume_252": float(row.median_dollar_volume_252),
                        "liquidity_rank": int(row.liquidity_rank),
                        "liquidity_percentile": float(row.liquidity_percentile),
                        "rows_at_cutoff": int(row.rows_at_cutoff),
                        "zero_volume_at_cutoff": int(row.zero_volume_at_cutoff),
                        "extreme_discontinuities_at_cutoff": int(row.extreme_discontinuities_at_cutoff),
                        "cell_selection_rank": rank,
                        "selection_hash": str(selected.iloc[rank - 1].selection_hash),
                        "lookback": lookback,
                        "representation_version": representation_version,
                        "episode_id": key.id,
                    })
    candidate_frame = pd.concat(enriched, ignore_index=True) if enriched else pd.DataFrame()
    candidate_frame = candidate_frame.sort_values(["cutoff", "symbol"], kind="stable", ignore_index=True)
    query_frame = pd.DataFrame(query_rows).sort_values(
        ["cutoff", "quality_tier", "liquidity_stratum", "cell_selection_rank"],
        kind="stable", ignore_index=True,
    )
    accounting = pd.DataFrame(accounting_rows).sort_values(
        ["cutoff", "quality_tier", "liquidity_stratum"], kind="stable", ignore_index=True,
    )
    return candidate_frame, query_frame, accounting


def bind_prefixes(
    queries: pd.DataFrame, source_root: Path, benchmark: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    benchmark_rows: list[dict[str, Any]] = []
    benchmark_map: dict[str, dict[str, Any]] = {}
    for cutoff in sorted(queries.cutoff.unique()):
        digest = causal_prefix_digest(benchmark, cutoff)
        row = {
            "cutoff": cutoff, "schema_version": digest.schema_version,
            "requested_cutoff": digest.requested_cutoff,
            "coverage_cutoff": digest.coverage_cutoff,
            "rows": digest.rows, "digest": digest.digest,
        }
        benchmark_rows.append(row)
        benchmark_map[cutoff] = row
    stock_rows: dict[tuple[str, str], dict[str, Any]] = {}
    for symbol, group in queries.groupby("symbol", sort=True):
        path = source_root / f"{symbol}.parquet"
        frame = pd.read_parquet(path)
        if "timestamp" not in frame and "date" in frame:
            frame = frame.rename(columns={"date": "timestamp"})
        frame = frame.loc[:, CANONICAL_COLUMNS].copy()
        frame["timestamp"] = pd.to_datetime(frame.timestamp, errors="coerce")
        for cutoff in sorted(group.cutoff.unique()):
            digest = causal_prefix_digest(frame, cutoff)
            stock_rows[(str(symbol), str(cutoff))] = {
                "stock_prefix_schema": digest.schema_version,
                "stock_prefix_requested_cutoff": digest.requested_cutoff,
                "stock_prefix_coverage_cutoff": digest.coverage_cutoff,
                "stock_prefix_rows": digest.rows,
                "stock_prefix_digest": digest.digest,
            }
    bound = queries.copy()
    for column in next(iter(stock_rows.values())):
        bound[column] = [stock_rows[(row.symbol, row.cutoff)][column] for row in bound.itertuples(index=False)]
    bound["benchmark_prefix_schema"] = [benchmark_map[value]["schema_version"] for value in bound.cutoff]
    bound["benchmark_prefix_rows"] = [benchmark_map[value]["rows"] for value in bound.cutoff]
    bound["benchmark_prefix_digest"] = [benchmark_map[value]["digest"] for value in bound.cutoff]
    return bound, pd.DataFrame(benchmark_rows)


def build_registry(
    denominator: pd.DataFrame, source_root: Path, benchmark: pd.DataFrame,
    cutoffs: Sequence[pd.Timestamp], *, locked_coverage_cutoff: pd.Timestamp,
    contract_digest: str, folds: Sequence[Mapping[str, str]], target_per_cell: int = 4,
    lookback: int = 252, representation_version: str = "dense-v1", workers: int = 12,
) -> RegistryBuild:
    required = {
        "symbol", "rows_at_lock", "first_timestamp_at_lock", "last_timestamp_at_lock",
        "source_hash_at_lock",
    }
    if missing := required.difference(denominator.columns):
        raise WalkForwardRegistryError(f"locked denominator missing {sorted(missing)}")
    if denominator.symbol.duplicated().any():
        raise WalkForwardRegistryError("locked denominator symbols are not unique")
    tasks = [
        (
            str(row.symbol), source_root / f"{row.symbol}.parquet", row._asdict(),
            tuple(cutoffs), locked_coverage_cutoff,
        )
        for row in denominator.sort_values("symbol", kind="stable").itertuples(index=False)
    ]
    if any(not task[1].is_file() for task in tasks):
        missing = [task[0] for task in tasks if not task[1].is_file()]
        raise WalkForwardRegistryError(f"locked source files are missing: {missing[:10]}")
    if workers == 1:
        results = [measure_symbol_prefixes(*task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(_measure_task, tasks, chunksize=8))
    candidate_rows = [row for rows, _ in results for row in rows]
    sources = pd.DataFrame([source for _, source in results]).sort_values(
        "symbol", kind="stable", ignore_index=True,
    )
    if sources.error.notna().any():
        errors = sources.loc[sources.error.notna(), ["symbol", "error"]]
        raise WalkForwardRegistryError(f"source integrity failures: {errors.head(10).to_dict('records')}")
    raw_candidates = pd.DataFrame(candidate_rows)
    candidates, queries, accounting = stratify_and_select(
        raw_candidates, cutoffs, contract_digest=contract_digest, folds=folds,
        target_per_cell=target_per_cell, lookback=lookback,
        representation_version=representation_version,
    )
    queries, benchmark_prefixes = bind_prefixes(queries, source_root, benchmark)
    return RegistryBuild(candidates, queries, accounting, sources, benchmark_prefixes)


def _measure_task(task: tuple[Any, ...]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return measure_symbol_prefixes(*task)
