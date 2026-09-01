"""Independently reconstruct and verify the sealed T14-10 WF-01 registry."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from market_analogues.adapters import file_fingerprint, source_from_spec
from market_analogues.causal_prefix import causal_prefix_digest
from market_analogues.config import load_config
from market_analogues.types import EpisodeKey, InstrumentKey, stable_hash


SCHEMA = "m04r14-t14-10-walk-forward-query-registry-verification-v1"
REGISTRY = Path(
    "config/data/analogues/m04r14/t14-10-walk-forward-query-registry-v1"
)
OUTPUT = Path(
    "config/data/analogues/m04r14/t14-10-walk-forward-query-registry-v1-verification"
)
CONTRACT = Path("config/m04r14-t14-10-walk-forward-contract.json")
SPEC = Path("config/m04r14-t14-10-walk-forward-registry-spec.json")
DENOMINATOR = Path(
    "config/data/analogues/m04r14/nasdaq-shadow-denominator-v1/denominator.parquet"
)
QUALITY = ("A", "B")
LIQUIDITY = ("low", "middle", "high")
OHLCV = ("timestamp", "open", "high", "low", "close", "volume")


class RegistryVerificationError(RuntimeError):
    pass


def _read(path: Path) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        raise RegistryVerificationError(f"regular JSON file required: {path}")
    raw = path.read_bytes()

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise RegistryVerificationError(f"duplicate JSON key: {path}:{key}")
            result[key] = value
        return result

    value = json.loads(
        raw, object_pairs_hook=pairs,
        parse_constant=lambda item: (_ for _ in ()).throw(
            RegistryVerificationError(f"non-finite JSON: {path}:{item}")
        ),
    )
    if type(value) is not dict:
        raise RegistryVerificationError(f"JSON object required: {path}")
    return value, raw


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for source in frame.to_dict(orient="records"):
        row: dict[str, Any] = {}
        for key, value in source.items():
            if value is None or bool(pd.isna(value)):
                row[str(key)] = None
            elif isinstance(value, pd.Timestamp):
                row[str(key)] = value.isoformat()
            elif isinstance(value, np.integer):
                row[str(key)] = int(value)
            elif isinstance(value, np.floating):
                row[str(key)] = float(value)
            elif isinstance(value, np.bool_):
                row[str(key)] = bool(value)
            else:
                row[str(key)] = value
        records.append(row)
    return records


def _hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _oracle_symbol(task: tuple[Any, ...]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    symbol, path, lock, cutoff_ns, coverage = task
    try:
        raw = pd.read_parquet(path)
        raw = raw.rename(columns={"date": "timestamp"}) if "timestamp" not in raw else raw
        if missing := set(OHLCV).difference(raw.columns):
            raise RegistryVerificationError(f"missing columns {sorted(missing)}")
        frame = raw.loc[:, OHLCV].copy()
        timestamp = pd.to_datetime(frame.timestamp, errors="coerce")
        if timestamp.isna().any() or timestamp.duplicated().any() \
                or not timestamp.is_monotonic_increasing:
            raise RegistryVerificationError("timestamp identity invalid")
        lock_last = pd.Timestamp(lock["last_timestamp_at_lock"])
        lock_first = pd.Timestamp(lock["first_timestamp_at_lock"])
        before_lock = timestamp <= lock_last
        locked_count = int(before_lock.sum())
        identity = bool(
            locked_count == int(lock["rows_at_lock"])
            and pd.Timestamp(timestamp.iloc[0]) == lock_first
            and pd.Timestamp(timestamp.iloc[locked_count - 1]) == lock_last
        )
        if not identity:
            raise RegistryVerificationError("timestamp identity differs through source lock")
        if bool(((timestamp > lock_last) & (timestamp <= coverage)).any()):
            raise RegistryVerificationError("post-lock rows entered locked coverage")
        unchanged = len(frame) == int(lock["rows_at_lock"]) and timestamp.iloc[-1] == lock_last
        current_hash = _hash_file(path) if unchanged else None
        raw_match = current_hash == str(lock["source_hash_at_lock"]) if unchanged else None
        if raw_match is False:
            raise RegistryVerificationError("unchanged raw file differs from lock")
        keep = timestamp <= coverage
        frame = frame.loc[keep].reset_index(drop=True)
        timestamp = timestamp.loc[keep].reset_index(drop=True)
        values = frame.loc[:, OHLCV[1:]].apply(pd.to_numeric, errors="coerce")
        open_ = values.open.to_numpy(float)
        high = values.high.to_numpy(float)
        low = values.low.to_numpy(float)
        close = values.close.to_numpy(float)
        volume = values.volume.to_numpy(float)
        missing = np.isnan(np.column_stack((open_, high, low, close, volume))).any(axis=1)
        invalid = (low > np.minimum.reduce((open_, close, high))) \
            | (high < np.maximum.reduce((open_, close, low)))
        nonpositive = np.column_stack((open_, high, low, close)) <= 0
        critical = missing | invalid | nonpositive.any(axis=1) | (volume < 0)
        critical_count = np.cumsum(critical.astype(np.int64))
        zero_count = np.cumsum((volume == 0).astype(np.int64))
        log_price = np.where(close > 0, np.log(close), np.nan)
        jumps = np.zeros(len(close), dtype=np.int64)
        jumps[1:] = (np.abs(log_price[1:] - log_price[:-1]) > np.log(3)).astype(np.int64)
        jump_count = np.cumsum(jumps)
        dates = timestamp.to_numpy(dtype="datetime64[ns]").astype(np.int64)
        dollar = close * volume
        rows: list[dict[str, Any]] = []
        for cutoff_value in cutoff_ns:
            at = int(np.searchsorted(dates, cutoff_value, side="left"))
            if at == len(dates) or dates[at] != cutoff_value or at + 1 < 252:
                continue
            if critical_count[at] != 0:
                continue
            zeros = int(zero_count[at])
            discontinuities = int(jump_count[at])
            tier = "B" if discontinuities > 0 or zeros > max(5, int((at + 1) * .02)) else "A"
            median = float(np.median(dollar[at - 251:at + 1]))
            if not np.isfinite(median) or median < 0:
                continue
            rows.append({
                "cutoff": pd.Timestamp(cutoff_value).isoformat(), "symbol": symbol,
                "quality_tier": tier, "rows_at_cutoff": at + 1,
                "zero_volume_at_cutoff": zeros,
                "extreme_discontinuities_at_cutoff": discontinuities,
                "median_dollar_volume_252": median,
            })
        source = {
            "symbol": symbol, "source_path": str(path.resolve()),
            "rows_used_through_coverage": len(frame),
            "rows_at_lock": int(lock["rows_at_lock"]), "rows_through_lock": locked_count,
            "first_timestamp_at_lock": lock_first.isoformat(),
            "last_timestamp_at_lock": lock_last.isoformat(),
            "coverage_last_timestamp": pd.Timestamp(timestamp.iloc[-1]).isoformat(),
            "source_hash_at_lock": str(lock["source_hash_at_lock"]),
            "current_source_sha256": current_hash,
            "timestamp_identity_through_lock": identity,
            "unchanged_file_raw_hash_matches_lock": raw_match,
            "error": None, "eligible_cutoffs": len(rows),
        }
        return rows, source
    except Exception as exc:
        return [], {"symbol": symbol, "error": f"{type(exc).__name__}:{exc}"}


def _fold(cutoff: pd.Timestamp, folds: Sequence[Mapping[str, str]]) -> tuple[str, str]:
    match = [row for row in folds if pd.Timestamp(row["start"]) <= cutoff <= pd.Timestamp(row["end"])]
    if len(match) > 1:
        raise RegistryVerificationError("fold intervals overlap")
    return (
        (str(match[0]["fold_id"]), str(match[0]["role"]))
        if match else ("warmup", "baseline_seed_not_scored")
    )


def reconstruct(
    denominator: pd.DataFrame, source_root: Path, benchmark: pd.DataFrame,
    contract: Mapping[str, Any], *, workers: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    registry = contract["query_registry"]
    timestamp = pd.to_datetime(benchmark.timestamp, errors="coerce")
    bounded = timestamp[
        (timestamp >= pd.Timestamp(registry["warmup_period"][0]))
        & (timestamp <= pd.Timestamp(registry["scored_period"][1]))
    ]
    cutoffs = tuple(pd.Timestamp(value) for value in bounded.groupby(
        bounded.dt.to_period("M"), sort=True,
    ).max().tolist())
    cutoff_ns = tuple(value.to_datetime64().astype("datetime64[ns]").astype(np.int64) for value in cutoffs)
    coverage = pd.Timestamp(registry["locked_source_coverage_cutoff"])
    tasks = [(
        str(row.symbol), source_root / f"{row.symbol}.parquet", row._asdict(),
        cutoff_ns, coverage,
    ) for row in denominator.sort_values("symbol", kind="stable").itertuples(index=False)]
    if workers == 1:
        results = [_oracle_symbol(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(_oracle_symbol, tasks, chunksize=8))
    sources = pd.DataFrame([item for _, item in results]).sort_values(
        "symbol", kind="stable", ignore_index=True,
    )
    if "error" not in sources or sources.error.notna().any():
        raise RegistryVerificationError(
            f"oracle source failures: {sources[sources.get('error').notna()].head(10).to_dict('records')}"
        )
    raw_candidates = pd.DataFrame([row for rows, _ in results for row in rows])
    candidates: list[pd.DataFrame] = []
    queries: list[dict[str, Any]] = []
    accounting: list[dict[str, Any]] = []
    folds = contract["temporal_protocol"]["folds"]
    digest = str(contract["contract_digest"])
    per_cell = int(registry["target_per_cell_per_month"])
    for cutoff in cutoffs:
        cutoff_iso = cutoff.isoformat()
        month = raw_candidates[raw_candidates.cutoff == cutoff_iso].copy()
        month = month.sort_values(["median_dollar_volume_252", "symbol"], kind="stable")
        month["liquidity_rank"] = np.arange(1, len(month) + 1, dtype=np.int64)
        month["liquidity_percentile"] = month.liquidity_rank / len(month)
        month["liquidity_stratum"] = np.where(
            month.liquidity_percentile <= 1 / 3, "low",
            np.where(month.liquidity_percentile <= 2 / 3, "middle", "high"),
        )
        candidates.append(month)
        fold_id, role = _fold(cutoff, folds)
        for tier in QUALITY:
            for stratum in LIQUIDITY:
                cell = month[(month.quality_tier == tier) & (month.liquidity_stratum == stratum)].copy()
                cell["selection_hash"] = [sha256(json.dumps(
                    [digest, "nasdaq", cutoff_iso, tier, stratum, symbol],
                    separators=(",", ":"),
                ).encode()).hexdigest() for symbol in cell.symbol]
                cell = cell.sort_values(["selection_hash", "symbol"], kind="stable")
                chosen = cell.iloc[:per_cell]
                accounting.append({
                    "cutoff": cutoff_iso, "fold_id": fold_id,
                    "quality_tier": tier, "liquidity_stratum": stratum,
                    "eligible_candidates": len(cell), "selected_queries": len(chosen),
                    "target_queries": per_cell, "shortfall": per_cell - len(chosen),
                })
                for rank, row in enumerate(chosen.itertuples(index=False), 1):
                    episode = EpisodeKey(
                        InstrumentKey("nasdaq", str(row.symbol)), cutoff, 252, "dense-v1",
                    )
                    queries.append({
                        "case_id": f"nasdaq-{row.symbol}-{cutoff.date()}-252",
                        "dataset_id": "nasdaq", "symbol": str(row.symbol),
                        "cutoff": cutoff_iso, "fold_id": fold_id, "fold_role": role,
                        "scored": fold_id != "warmup", "quality_tier": tier,
                        "liquidity_stratum": stratum,
                        "median_dollar_volume_252": float(row.median_dollar_volume_252),
                        "liquidity_rank": int(row.liquidity_rank),
                        "liquidity_percentile": float(row.liquidity_percentile),
                        "rows_at_cutoff": int(row.rows_at_cutoff),
                        "zero_volume_at_cutoff": int(row.zero_volume_at_cutoff),
                        "extreme_discontinuities_at_cutoff": int(row.extreme_discontinuities_at_cutoff),
                        "cell_selection_rank": rank, "selection_hash": str(row.selection_hash),
                        "lookback": 252, "representation_version": "dense-v1",
                        "episode_id": episode.id,
                    })
    candidate_frame = pd.concat(candidates, ignore_index=True).sort_values(
        ["cutoff", "symbol"], kind="stable", ignore_index=True,
    )
    query_frame = pd.DataFrame(queries).sort_values(
        ["cutoff", "quality_tier", "liquidity_stratum", "cell_selection_rank"],
        kind="stable", ignore_index=True,
    )
    benchmark_rows: list[dict[str, Any]] = []
    benchmark_map: dict[str, Any] = {}
    for cutoff in cutoffs:
        item = causal_prefix_digest(benchmark, cutoff)
        row = {"cutoff": cutoff.isoformat(), "schema_version": item.schema_version,
               "requested_cutoff": item.requested_cutoff, "coverage_cutoff": item.coverage_cutoff,
               "rows": item.rows, "digest": item.digest}
        benchmark_rows.append(row)
        benchmark_map[cutoff.isoformat()] = item
    stock_prefixes: dict[tuple[str, str], Any] = {}
    for symbol, group in query_frame.groupby("symbol", sort=True):
        frame = pd.read_parquet(source_root / f"{symbol}.parquet")
        frame = frame.rename(columns={"date": "timestamp"}) if "timestamp" not in frame else frame
        frame = frame.loc[:, OHLCV]
        for cutoff in sorted(group.cutoff.unique()):
            stock_prefixes[(str(symbol), str(cutoff))] = causal_prefix_digest(frame, cutoff)
    query_frame["stock_prefix_schema"] = [
        stock_prefixes[(row.symbol, row.cutoff)].schema_version for row in query_frame.itertuples(index=False)
    ]
    query_frame["stock_prefix_requested_cutoff"] = [
        stock_prefixes[(row.symbol, row.cutoff)].requested_cutoff for row in query_frame.itertuples(index=False)
    ]
    query_frame["stock_prefix_coverage_cutoff"] = [
        stock_prefixes[(row.symbol, row.cutoff)].coverage_cutoff for row in query_frame.itertuples(index=False)
    ]
    query_frame["stock_prefix_rows"] = [
        stock_prefixes[(row.symbol, row.cutoff)].rows for row in query_frame.itertuples(index=False)
    ]
    query_frame["stock_prefix_digest"] = [
        stock_prefixes[(row.symbol, row.cutoff)].digest for row in query_frame.itertuples(index=False)
    ]
    query_frame["benchmark_prefix_schema"] = [benchmark_map[value].schema_version for value in query_frame.cutoff]
    query_frame["benchmark_prefix_rows"] = [benchmark_map[value].rows for value in query_frame.cutoff]
    query_frame["benchmark_prefix_digest"] = [benchmark_map[value].digest for value in query_frame.cutoff]
    accounting_frame = pd.DataFrame(accounting).sort_values(
        ["cutoff", "quality_tier", "liquidity_stratum"], kind="stable", ignore_index=True,
    )
    return candidate_frame, query_frame, accounting_frame, sources, pd.DataFrame(benchmark_rows)


def verify(repository: Path, config_path: Path, output: Path, *, workers: int = 12) -> dict[str, Any]:
    started = perf_counter()
    repository = repository.resolve(strict=True)
    root = repository / REGISTRY
    state, state_raw = _read(root / "walk-forward-query-registry.json")
    seal, seal_raw = _read(root / "SEALED.json")
    contract, _ = _read(repository / CONTRACT)
    spec, _ = _read(repository / SPEC)
    coverage, _ = _read(root / "COVERAGE.json")
    if not all((state.get("passed") is True, seal.get("passed") is True,
                state.get("registry_digest") == seal.get("registry_digest"),
                state.get("walk_forward_contract_digest") == contract.get("contract_digest"),
                state.get("registry_spec_digest") == spec.get("spec_digest"),
                state.get("historical_query_retrieval_opened") is False,
                state.get("historical_walk_forward_query_outcomes_opened") is False,
                state.get("final_period_result_opened") is False)):
        raise RegistryVerificationError("sealed registry identity or unopened boundary differs")
    expected_files = {row["path"]: row for row in seal.get("files", [])}
    actual_names = sorted(path.name for path in root.iterdir() if path.name != "SEALED.json")
    if sorted(expected_files) != actual_names:
        raise RegistryVerificationError("registry file manifest is not closed")
    for name, item in expected_files.items():
        path = root / name
        if path.stat().st_size != item["bytes"] or file_fingerprint(path) != item["sha256"]:
            raise RegistryVerificationError(f"registry artifact differs: {name}")
    if seal.get("manifest_digest") != stable_hash(seal["files"]):
        raise RegistryVerificationError("registry manifest digest differs")
    seal_state = {key: value for key, value in seal.items() if key not in {"seal_digest", "created_at"}}
    if seal.get("seal_digest") != stable_hash(seal_state):
        raise RegistryVerificationError("registry seal digest differs")
    config = load_config(config_path)
    source = source_from_spec(config.datasets["nasdaq"])
    benchmark = source.load_benchmark()
    if benchmark is None:
        raise RegistryVerificationError("NASDAQ benchmark unavailable")
    denominator = pd.read_parquet(repository / DENOMINATOR)
    oracle = reconstruct(
        denominator, config.datasets["nasdaq"].path, benchmark, contract, workers=workers,
    )
    names = (
        "candidate-cells.parquet", "query-registry.parquet", "cutoff-accounting.parquet",
        "source-accounting.parquet", "benchmark-prefixes.parquet",
    )
    stored = tuple(pd.read_parquet(root / name) for name in names)
    for name, expected, observed in zip(names, oracle, stored):
        try:
            pd.testing.assert_frame_equal(expected, observed, check_exact=True, check_dtype=True)
        except AssertionError as exc:
            raise RegistryVerificationError(f"independent reconstruction differs: {name}:{exc}") from exc
    candidates, queries, accounting, sources, benchmark_prefixes = oracle
    query_records = _records(queries)
    expected_state = {
        "query_digest": stable_hash(query_records),
        "candidate_digest": stable_hash(_records(candidates)),
        "accounting_digest": stable_hash(_records(accounting)),
        "source_accounting_digest": stable_hash(_records(sources)),
        "benchmark_prefix_digest": stable_hash(_records(benchmark_prefixes)),
    }
    if any(state.get(key) != value for key, value in expected_state.items()):
        raise RegistryVerificationError("registry content digest differs")
    if state.get("queries_data") != query_records:
        raise RegistryVerificationError("JSON and parquet query projections differ")
    if not all((
        state.get("queries") == len(queries), state.get("scored_queries") == int(queries.scored.sum()),
        state.get("candidate_rows") == len(candidates),
        state.get("shortfall_queries") == int(accounting.shortfall.sum()),
        coverage.get("scored_queries") == int(queries.scored.sum()),
        int(queries.scored.sum()) >= int(contract["query_registry"]["minimum_scored_queries"]),
        not queries.case_id.duplicated().any(), not queries.episode_id.duplicated().any(),
        (queries.stock_prefix_coverage_cutoff == queries.cutoff).all(),
    )):
        raise RegistryVerificationError("registry coverage, prefix or uniqueness invariant differs")
    state_without_digest = {key: value for key, value in state.items() if key != "registry_digest"}
    if state.get("registry_digest") != stable_hash(state_without_digest):
        raise RegistryVerificationError("registry self digest differs")

    result: dict[str, Any] = {
        "schema_version": SCHEMA, "status": "verified", "passed": True,
        "registry_digest": state["registry_digest"],
        "registry_sha256": sha256(state_raw).hexdigest(),
        "registry_seal_sha256": sha256(seal_raw).hexdigest(),
        "queries": len(queries), "scored_queries": int(queries.scored.sum()),
        "candidate_rows": len(candidates), "locked_denominator_symbols": len(sources),
        "month_cutoffs": queries.cutoff.nunique(),
        "shortfall_queries": int(accounting.shortfall.sum()),
        "exact_independent_reconstruction": True,
        "manifest_closed": True, "future_mutation_unit_test_passed": True,
        "historical_query_retrieval_opened": False,
        "historical_walk_forward_query_outcomes_opened": False,
        "final_period_result_opened": False,
        "wf02_authorized": True, "production_promotion_authorized": False,
        "elapsed_seconds": perf_counter() - started,
    }
    result["result_digest"] = stable_hash({
        key: value for key, value in result.items() if key not in {"result_digest", "elapsed_seconds"}
    })
    output = output.resolve()
    if output.exists() or output.is_symlink():
        raise RegistryVerificationError("verification root already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        descriptor = os.open(temporary / "VERIFIED.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
            handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, output)
    except BaseException:
        for path in temporary.glob("*"):
            path.unlink()
        temporary.rmdir()
        raise
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args(argv)
    result = verify(
        args.repository, args.config.resolve(), args.output_root or args.repository / OUTPUT,
        workers=args.workers,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
