"""Independent verifier for the M04R-14 36-symbol untouched registry."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from market_analogues.adapters import file_fingerprint, source_from_spec
from market_analogues.causal_prefix import causal_prefix_digest
from market_analogues.config import load_config
from market_analogues.types import EpisodeKey, stable_hash


SCHEMA = "m04r14-untouched-registry-verification-v1"
REGISTRY_SCHEMA = "m04r14-untouched-authority-registry-v1"
CONTRACT = Path("config/m04r14-untouched-registry-contract.json")
REGISTRY = Path("config/data/analogues/m04r14/nasdaq-untouched-registry-v1")
VERIFICATION = Path("config/data/analogues/m04r14/nasdaq-untouched-registry-v1-verification")
BUILDER_SOURCE = Path("experiments/m04r/m04r14_untouched_registry.py")
FORBIDDEN_KEYS = {"outcome", "outcomes", "forward_return", "forward_returns",
    "winner", "loser", "setup", "setup_label", "profit", "loss", "result"}


class VerificationError(RuntimeError):
    pass


def _read(path: Path) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        raise VerificationError(f"regular file required: {path}")
    raw = path.read_bytes()
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise VerificationError(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    try:
        value = json.loads(raw, object_pairs_hook=pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(
                VerificationError(f"non-finite JSON: {item}")))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"invalid JSON: {path}") from exc
    if type(value) is not dict:
        raise VerificationError(f"JSON object required: {path}")
    return value, raw


def _symbols(values: Iterable[Any]) -> list[str]:
    return sorted({str(value).strip().upper() for value in values if str(value).strip()})


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in frame.to_dict(orient="records"):
        row: dict[str, Any] = {}
        for key, value in raw.items():
            if pd.isna(value): row[str(key)] = None
            elif isinstance(value, pd.Timestamp): row[str(key)] = value.isoformat()
            elif isinstance(value, np.integer): row[str(key)] = int(value)
            elif isinstance(value, np.floating): row[str(key)] = float(value)
            elif isinstance(value, np.bool_): row[str(key)] = bool(value)
            else: row[str(key)] = value
        rows.append(row)
    return rows


def _source(name: str, path: Path, values: Iterable[Any]) -> dict[str, Any]:
    symbols = _symbols(values)
    return {"name": name, "path": str(path.resolve()),
        "file_sha256": file_fingerprint(path), "symbol_count": len(symbols),
        "symbol_digest": stable_hash(symbols), "symbols": symbols}


def _ledger(artifact: Path, design: Path, previous: Path) -> dict[str, Any]:
    summary_path = artifact / "oracles/nasdaq/oracle-summary.parquet"
    summary = pd.read_parquet(summary_path)
    sources = [_source("gate09-queries", summary_path,
        (str(value).split(":", 1)[-1] for value in summary["query"]))]
    for case_id in sorted(summary.case_id.astype(str).unique()):
        path = artifact / "oracles/nasdaq" / f"{case_id}.parquet"
        sources.append(_source(f"gate09-candidates:{case_id}", path,
            pd.read_parquet(path, columns=["symbol"])["symbol"]))
    gate12_path = artifact / "gate12/nasdaq/query-registry.yaml"
    gate12 = yaml.safe_load(gate12_path.read_text()) or {}
    sources.append(_source("gate12-selected", gate12_path,
        (row["symbol"] for row in gate12.get("cases_data", []))))
    batch_path = artifact / "poc/m04r/batch-query-registry/query-registry.json"
    batch, _ = _read(batch_path)
    sources.append(_source("m04r-batch-selected", batch_path,
        (row["symbol"] for row in batch.get("cases_data", []))))
    for path in sorted((artifact / "external-examples").glob("*/coverage.parquet")):
        sources.append(_source(f"external-example:{path.parent.name}", path,
            pd.read_parquet(path, columns=["symbol"])["symbol"]))
    design_value = yaml.safe_load(design.read_text()) or {}
    design_symbols = list(design_value.get("symbols", []))
    for group in design_value.get("known_identity_alias_groups", []):
        design_symbols.extend(group.get("symbols", []))
    sources.append(_source("m04r10-design-and-known-aliases", design, design_symbols))
    base_excluded = _symbols(symbol for source in sources for symbol in source["symbols"])
    base: dict[str, Any] = {"schema_version": "m04r-contamination-ledger-v1",
        "normalization": "strip then uppercase; exact source-symbol exclusion",
        "sources": sources, "excluded_symbols": base_excluded,
        "excluded_symbol_count": len(base_excluded),
        "excluded_symbol_digest": stable_hash(base_excluded),
        "known_identity_alias_groups": design_value.get("known_identity_alias_groups", []),
        "alias_limit": design_value.get("alias_limit")}
    base["ledger_digest"] = stable_hash(base)
    previous_value, _ = _read(previous)
    previous_rows = previous_value.get("cases_data")
    if type(previous_rows) is not list or len(previous_rows) != 60:
        raise VerificationError("previous registry differs")
    sources = [*sources, _source("m04r10-selected-and-m04r14-exposed", previous,
        [row["symbol"] for row in previous_rows])]
    excluded = _symbols(symbol for source in sources for symbol in source["symbols"])
    value: dict[str, Any] = {"schema_version": "m04r14-contamination-ledger-v1",
        "normalization": "strip then uppercase; exact source symbol plus explicit known aliases",
        "sources": sources, "excluded_symbols": excluded,
        "excluded_symbol_count": len(excluded),
        "excluded_symbol_digest": stable_hash(excluded),
        "known_identity_alias_groups": base["known_identity_alias_groups"],
        "alias_limit": base["alias_limit"], "previous_ledger_digest": base["ledger_digest"],
        "previous_registry_digest": previous_value.get("registry_digest"),
        "previous_registry_cases": len(previous_rows),
        "previous_registry_symbols": len({str(row["symbol"]).upper() for row in previous_rows})}
    value["ledger_digest"] = stable_hash(value)
    return value


def _universe(quality: pd.DataFrame, liquidity: pd.DataFrame, excluded: set[str],
              minimum_rows: int, staleness_days: int) -> tuple[pd.DataFrame, pd.Timestamp]:
    freshness = quality[["symbol", "first_timestamp", "last_timestamp"]].copy()
    freshness["symbol"] = freshness.symbol.astype(str)
    freshness["first_timestamp"] = pd.to_datetime(freshness.first_timestamp)
    freshness["last_timestamp"] = pd.to_datetime(freshness.last_timestamp)
    merged = liquidity.merge(freshness, on="symbol", how="left", validate="one_to_one")
    latest = pd.Timestamp(merged.last_timestamp.max()); floor = latest - pd.Timedelta(days=staleness_days)
    value = merged[merged.quality_tier.astype(str).isin(("A", "B"))
        & merged.liquidity_stratum.astype(str).isin(("low", "middle", "high"))
        & (merged.rows.astype(int) >= minimum_rows) & (merged.last_timestamp >= floor)
        & ~merged.symbol.astype(str).str.upper().isin(excluded)].copy()
    value = value[["symbol", "quality_tier", "liquidity_stratum", "rows",
        "median_dollar_volume_252", "first_timestamp", "last_timestamp"]].rename(columns={
            "rows": "rows_at_lock", "first_timestamp": "first_timestamp_at_lock",
            "last_timestamp": "last_timestamp_at_lock"})
    value["symbol"] = value.symbol.astype(str)
    return value.sort_values("symbol", kind="stable", ignore_index=True), latest


def _era(timestamp: pd.Timestamp) -> str:
    return "pre-2010" if timestamp.year < 2010 else "2010s" if timestamp.year < 2020 else "2020s"


def _regime(value: float, threshold: float) -> str:
    return "up" if value > threshold else "down" if value < -threshold else "sideways"


def _benchmark(benchmark: pd.DataFrame, lookback: int, threshold: float):
    frame = benchmark[["timestamp", "close"]].copy()
    frame["timestamp"] = pd.to_datetime(frame.timestamp); frame["close"] = pd.to_numeric(frame.close)
    returns = frame.close / frame.close.shift(lookback - 1) - 1
    values = {pd.Timestamp(timestamp): float(value) for timestamp, value in
        zip(frame.timestamp, returns, strict=True) if np.isfinite(value)}
    return {stamp: _regime(value, threshold) for stamp, value in values.items()}, values


def _cutoff(bars: pd.DataFrame, lock: pd.Timestamp, target_regime: str, target_era: str,
            regimes: Mapping[pd.Timestamp, str], seed: str, symbol: str, slot: int,
            lookback: int, future: int) -> tuple[pd.Timestamp | None, int]:
    locked = bars[pd.to_datetime(bars.timestamp) <= lock].reset_index(drop=True)
    positions = [position for position in range(lookback - 1, len(locked) - future)
        if _era(pd.Timestamp(locked.timestamp.iloc[position])) == target_era
        and regimes.get(pd.Timestamp(locked.timestamp.iloc[position])) == target_regime]
    ranked = sorted(positions, key=lambda position: (sha256(
        f"{seed}:cutoff:{symbol}:{slot}:{pd.Timestamp(locked.timestamp.iloc[position]).isoformat()}".encode()
    ).hexdigest(), pd.Timestamp(locked.timestamp.iloc[position])))
    return (pd.Timestamp(locked.timestamp.iloc[ranked[0]]) if ranked else None), len(locked)


def _descriptor(bars: pd.DataFrame, benchmark: pd.DataFrame, cutoff: pd.Timestamp,
                lookback: int, benchmark_return: float, threshold: float,
                quality: pd.DataFrame) -> dict[str, Any]:
    eligible = bars[pd.to_datetime(bars.timestamp) <= cutoff].tail(lookback)
    close = pd.to_numeric(eligible.close); volume = pd.to_numeric(eligible.volume)
    episode_return = float(close.iloc[-1] / close.iloc[0] - 1)
    volatility = float(np.log(close.where(close > 0)).diff().std() * np.sqrt(252))
    drawdown = float((close / close.cummax() - 1).min())
    volume_fraction = float((volume.notna() & (volume > 0)).mean())
    overlap = float(pd.to_datetime(eligible.timestamp).isin(set(pd.to_datetime(benchmark.timestamp))).mean())
    first = pd.to_datetime(quality.first_timestamp); last = pd.to_datetime(quality.last_timestamp)
    active = int(((first <= cutoff) & (last >= cutoff)).sum())
    return {"episode_return": episode_return,
        "realized_volatility_annualized": volatility, "maximum_drawdown": drawdown,
        "volume_observed_fraction": volume_fraction, "benchmark_overlap_fraction": overlap,
        "benchmark_return": benchmark_return, "benchmark_regime": _regime(benchmark_return, threshold),
        "era": _era(cutoff), "morphology_stratum": "advance" if episode_return >= .25 else
            "decline" if episode_return <= -.20 else "range",
        "data_context": "full" if volume_fraction >= .995 and overlap >= .995 else "sparse",
        "active_source_universe": active,
        "universe_size_band": "small" if active < 4000 else "middle" if active < 8000 else "large",
        "volatility_band": "low" if volatility < .30 else "middle" if volatility < .70 else "high",
        "drawdown_band": "shallow" if drawdown > -.20 else "middle" if drawdown > -.50 else "deep"}


def _reconstruct(source: Any, quality: pd.DataFrame, universe: pd.DataFrame,
                 contract: Mapping[str, Any], ledger: Mapping[str, Any],
                 representation_version: str):
    instruments = {item.source_symbol: item for item in source.instruments()}
    benchmark = source.load_benchmark(); assert benchmark is not None
    lookback = int(contract["lookback"]); future = int(contract["minimum_future_sessions"])
    threshold = float(contract["regime_threshold"]); seed = str(contract["seed"])
    regimes, benchmark_returns = _benchmark(benchmark, lookback, threshold)
    schedule = [tuple(value) for value in contract["target_schedule"]]
    cache: dict[str, pd.DataFrame] = {}; transcript = []; selected = []
    for (tier, stratum), cell in universe.groupby(["quality_tier", "liquidity_stratum"], sort=True):
        used: set[str] = set()
        for slot, (target_regime, target_era) in enumerate(schedule):
            ranked = cell.copy(); ranked["selection_hash"] = ranked.symbol.map(lambda symbol: sha256(
                f"{seed}:nasdaq:{tier}:{stratum}:{slot}:{target_regime}:{target_era}:{symbol}".encode()
            ).hexdigest())
            ranked = ranked[~ranked.symbol.isin(used)].sort_values(["selection_hash", "symbol"], kind="stable")
            choice = None
            for candidate_rank, row in enumerate(ranked.itertuples(index=False), 1):
                symbol = str(row.symbol)
                if symbol not in cache:
                    cache[symbol] = source.load(instruments[symbol])
                cutoff, rows = _cutoff(cache[symbol], pd.Timestamp(row.last_timestamp_at_lock),
                    target_regime, target_era, regimes, seed, symbol, slot, lookback, future)
                transcript.append({"quality_tier": str(tier), "liquidity_stratum": str(stratum),
                    "slot": slot, "target_regime": target_regime, "target_era": target_era,
                    "candidate_rank": candidate_rank, "symbol": symbol,
                    "selection_hash": str(row.selection_hash), "rows_at_lock": int(row.rows_at_lock),
                    "observed_rows_through_lock": rows,
                    "last_timestamp_at_lock": pd.Timestamp(row.last_timestamp_at_lock).isoformat(),
                    "eligible": cutoff is not None,
                    "selected_cutoff": cutoff.isoformat() if cutoff is not None else None})
                if rows != int(row.rows_at_lock): raise VerificationError("source rows differ")
                if cutoff is not None:
                    choice = (row, slot, target_regime, target_era, cutoff); break
            if choice is None: raise VerificationError("selection slot has no eligible symbol")
            selected.append(choice); used.add(str(choice[0].symbol))
    cases = []
    for selection, slot, target_regime, target_era, historical in selected:
        symbol = str(selection.symbol); instrument = instruments[symbol]; bars = cache[symbol]
        lock = pd.Timestamp(selection.last_timestamp_at_lock)
        locked = bars[pd.to_datetime(bars.timestamp) <= lock].reset_index(drop=True)
        for role, cutoff in (("historical", historical), ("current", lock)):
            positions = np.flatnonzero(pd.to_datetime(locked.timestamp).to_numpy() <= cutoff.to_datetime64())
            position = int(positions[-1]); future_sessions = len(locked) - position - 1
            key = EpisodeKey(instrument, cutoff, lookback, representation_version)
            benchmark_return = benchmark_returns[cutoff]
            cases.append({"case_id": f"nasdaq-{symbol}-{role}-{lookback}", "dataset_id": "nasdaq",
                "symbol": symbol, "quality_tier": str(selection.quality_tier),
                "liquidity_stratum": str(selection.liquidity_stratum),
                "median_dollar_volume_252": float(selection.median_dollar_volume_252),
                "selection_slot": slot, "selection_target_regime": target_regime,
                "selection_target_era": target_era, "cutoff_role": role,
                "cutoff": cutoff.isoformat(), "cutoff_position_at_lock": position,
                "future_sessions_at_lock": future_sessions, "lookback": lookback,
                "representation_version": representation_version, "episode_id": key.id,
                "stock_prefix": asdict(causal_prefix_digest(locked, cutoff)),
                "benchmark_prefix": asdict(causal_prefix_digest(benchmark, cutoff)),
                **_descriptor(locked, benchmark, cutoff, lookback, benchmark_return, threshold, quality)})
    case_frame = pd.DataFrame(cases).sort_values(
        ["quality_tier", "liquidity_stratum", "selection_slot", "cutoff_role"],
        kind="stable", ignore_index=True)
    transcript_frame = pd.DataFrame(transcript).sort_values(
        ["quality_tier", "liquidity_stratum", "slot", "candidate_rank"],
        kind="stable", ignore_index=True)
    return case_frame, transcript_frame


def _contains_forbidden(value: Any) -> bool:
    if isinstance(value, dict):
        return bool(FORBIDDEN_KEYS.intersection(map(str, value))) or any(
            _contains_forbidden(item) for item in value.values())
    if isinstance(value, list): return any(_contains_forbidden(item) for item in value)
    return False


def verify(root: Path, *, repository: Path, config_path: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True); root = root.resolve(strict=True)
    if root != (repository / REGISTRY).resolve(strict=True): raise VerificationError("registry root differs")
    names = {path.name for path in root.iterdir()}
    required = {"SEALED.json", "query-registry.html", "query-registry.json",
        "query-registry.parquet", "selection-transcript.parquet", "selection-universe.parquet"}
    if names != required or any(path.is_symlink() or not path.is_file() for path in root.iterdir()):
        raise VerificationError("registry tree differs")
    payload, payload_raw = _read(root / "query-registry.json"); seal, seal_raw = _read(root / "SEALED.json")
    manifest = [{"path": path.name, "bytes": path.stat().st_size, "sha256": file_fingerprint(path)}
        for path in sorted(root.iterdir()) if path.name != "SEALED.json"]
    seal_state = {key: value for key, value in seal.items() if key not in {"created_at", "seal_digest"}}
    if seal.get("files") != manifest or seal.get("manifest_digest") != stable_hash(manifest) \
            or seal.get("seal_digest") != stable_hash(seal_state):
        raise VerificationError("registry seal differs")
    contract, contract_raw = _read(repository / CONTRACT); config = load_config(config_path)
    source = source_from_spec(config.datasets["nasdaq"])
    quality_path = config.artifact_dir / "quality/nasdaq.parquet"
    liquidity_path = config.artifact_dir / "oracles/nasdaq/liquidity-strata.parquet"
    design = repository / contract["contamination_sources"]["base_design"]
    previous = repository / contract["contamination_sources"]["previous_registry"]
    source_lock = payload.get("source_lock", {}); marker_head = source_lock.get("git_head")
    if type(marker_head) is not str or len(marker_head) != 40: raise VerificationError("source lock Git head differs")
    subprocess.run(["git", "cat-file", "-e", f"{marker_head}^{{commit}}"], cwd=repository, check=True)
    committed = subprocess.run(["git", "show", f"{marker_head}:{BUILDER_SOURCE.as_posix()}"],
        cwd=repository, capture_output=True, check=True).stdout
    if committed != (repository / BUILDER_SOURCE).read_bytes(): raise VerificationError("builder source differs")
    expected_lock = {"git_head": marker_head, "contract_sha256": sha256(contract_raw).hexdigest(),
        "config_path": str(config_path.resolve()), "config_sha256": file_fingerprint(config_path),
        "quality_path": str(quality_path.resolve()), "quality_sha256": file_fingerprint(quality_path),
        "liquidity_path": str(liquidity_path.resolve()), "liquidity_sha256": file_fingerprint(liquidity_path),
        "design_exclusions_sha256": file_fingerprint(design),
        "previous_registry_sha256": file_fingerprint(previous),
        "real_forward_outcomes_accessed": False}
    expected_lock["source_lock_digest"] = stable_hash(expected_lock)
    if source_lock != expected_lock: raise VerificationError("source lock differs")
    ledger = _ledger(config.artifact_dir, design, previous)
    if payload.get("contamination_ledger") != ledger: raise VerificationError("ledger differs")
    quality = pd.read_parquet(quality_path); liquidity = pd.read_parquet(liquidity_path)
    universe, latest = _universe(quality, liquidity, set(ledger["excluded_symbols"]),
        int(contract["minimum_rows"]), int(contract["maximum_staleness_days"]))
    stored_universe = pd.read_parquet(root / "selection-universe.parquet")
    if _records(universe) != _records(stored_universe): raise VerificationError("universe differs")
    cases, transcript = _reconstruct(
        source, quality, universe, contract, ledger, config.representation_version
    )
    stored_cases = pd.read_parquet(root / "query-registry.parquet")
    stored_transcript = pd.read_parquet(root / "selection-transcript.parquet")
    if _records(cases) != _records(stored_cases) or _records(cases) != payload.get("cases_data"):
        raise VerificationError("case reconstruction differs")
    if _records(transcript) != _records(stored_transcript): raise VerificationError("transcript differs")
    if len(cases) != 72 or cases.symbol.nunique() != 36 or _contains_forbidden(payload):
        raise VerificationError("registry count or outcome firewall differs")
    cell_counts = cases.drop_duplicates("symbol").groupby(
        ["quality_tier", "liquidity_stratum"]).size().to_dict()
    schedule = {tuple(value) for value in contract["target_schedule"]}
    observed_schedule = set(zip(cases.selection_target_regime, cases.selection_target_era))
    if set(cell_counts.values()) != {6} or observed_schedule != schedule:
        raise VerificationError("cell/schedule coverage differs")
    if set(cases.symbol.str.upper()).intersection(ledger["excluded_symbols"]):
        raise VerificationError("selected symbols are contaminated")
    deterministic = {key: value for key, value in payload.items()
        if key not in {"passed", "failures", "registry_digest"}}
    if payload.get("registry_digest") != stable_hash(deterministic) \
            or seal.get("registry_digest") != payload.get("registry_digest") \
            or payload.get("passed") is not True or payload.get("failures") != [] \
            or payload.get("schema_version") != REGISTRY_SCHEMA:
        raise VerificationError("registry aggregate differs")
    state = {"schema_version": SCHEMA, "status": "verified", "passed": True,
        "registry_root": str(root), "registry_digest": payload["registry_digest"],
        "registry_sha256": sha256(payload_raw).hexdigest(), "seal_sha256": sha256(seal_raw).hexdigest(),
        "seal_digest": seal["seal_digest"], "source_lock_digest": expected_lock["source_lock_digest"],
        "contamination_ledger_digest": ledger["ledger_digest"],
        "universe_digest": stable_hash(_records(universe)),
        "transcript_digest": stable_hash(_records(transcript)),
        "cases_digest": stable_hash(_records(cases)), "market_latest": latest.isoformat(),
        "verified_symbols": 36, "verified_cases": 72, "verified_cells": 6,
        "verified_schedule_slots": 6, "real_forward_outcomes_accessed": False,
        "production_promotion_authorized": False}
    state["result_digest"] = stable_hash(state); return state


def _publish(path: Path, state: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink(): raise VerificationError("verification root exists")
    path.mkdir(parents=False)
    descriptor = os.open(path / "VERIFIED.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(json.dumps({**state, "created_at": datetime.now(timezone.utc).isoformat()},
            indent=2, sort_keys=True).encode() + b"\n"); handle.flush(); os.fsync(handle.fileno())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv); repository = args.repository.resolve(strict=True)
    state = verify(repository / REGISTRY, repository=repository, config_path=args.config.resolve())
    if not args.dry_run: _publish(repository / VERIFICATION, state)
    print(json.dumps(state, indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
