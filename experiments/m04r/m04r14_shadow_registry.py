"""Build the outcome-blind T14-08 full NASDAQ shadow denominator.

Every source/metadata symbol receives exactly one scheduled-query or mechanical-
skip row.  Search results, authorities, setup labels and forward outcomes are not
accepted by this program.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
from html import escape
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from market_analogues.adapters import file_fingerprint, source_from_spec
from market_analogues.causal_prefix import causal_prefix_digest
from market_analogues.config import load_config
from market_analogues.types import EpisodeKey, stable_hash


SCHEMA = "m04r14-shadow-denominator-v1"
SEAL_SCHEMA = "m04r14-shadow-denominator-seal-v1"
CONTRACT = Path("config/m04r14-shadow-contract.json")
DEFAULT_OUTPUT = Path("config/data/analogues/m04r14/nasdaq-shadow-denominator-v1")
PACK = Path("config/data/analogues/poc/m04r/packed-bound-full")
FORBIDDEN_KEYS = {
    "outcome", "outcomes", "forward_return", "forward_returns", "winner",
    "loser", "profit", "loss", "setup", "setup_label", "matches",
}


class ShadowRegistryError(RuntimeError):
    pass


def _contains_forbidden_key(value: Any) -> bool:
    if isinstance(value, dict):
        return bool(FORBIDDEN_KEYS.intersection(map(str, value))) or any(
            _contains_forbidden_key(item) for item in value.values()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_key(item) for item in value)
    return False


def _read(path: Path) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        raise ShadowRegistryError(f"regular file required: {path}")
    raw = path.read_bytes()

    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise ShadowRegistryError(f"duplicate JSON key: {path}:{key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw, object_pairs_hook=pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ShadowRegistryError(f"non-finite JSON: {path}:{item}")),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ShadowRegistryError(f"invalid JSON: {path}") from exc
    if type(value) is not dict:
        raise ShadowRegistryError(f"JSON object required: {path}")
    return value, raw


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in frame.to_dict(orient="records"):
        row: dict[str, Any] = {}
        for key, value in raw.items():
            if value is None:
                row[str(key)] = None
            elif isinstance(value, (dict, list, tuple)):
                row[str(key)] = value
            elif bool(pd.isna(value)):
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
        rows.append(row)
    return rows


def _atomic(path: Path, raw: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def _pack_manifest(repository: Path) -> tuple[Path, dict[str, Any]]:
    manifests = sorted((repository / PACK / "store/generations").glob("*/manifest.json"))
    if len(manifests) != 1:
        raise ShadowRegistryError(f"expected one packed generation; found {len(manifests)}")
    manifest, _ = _read(manifests[0])
    if manifest.get("manifest_digest") != manifests[0].parent.name:
        raise ShadowRegistryError("packed generation directory/manifest identity differs")
    return manifests[0], manifest


def _source_inventory(source: Any, quality: pd.DataFrame) -> tuple[list[str], str]:
    instruments = source.instruments()
    symbols = [key.source_symbol for key in instruments]
    if len(symbols) != len(set(symbols)):
        raise ShadowRegistryError("source contains duplicate symbols")
    quality_hash = {
        str(row.symbol): str(row.source_hash) for row in quality.itertuples(index=False)
    }
    inventory: list[tuple[str, str]] = []
    for key in instruments:
        fingerprint = source.fingerprint(key)
        if key.source_symbol in quality_hash and fingerprint != quality_hash[key.source_symbol]:
            raise ShadowRegistryError(f"quality/source fingerprint differs: {key.source_symbol}")
        inventory.append((key.source_symbol, fingerprint))
    return sorted(symbols), stable_hash(inventory)


def _classify(
    repository: Path, contract: Mapping[str, Any], source: Any,
    quality: pd.DataFrame, liquidity: pd.DataFrame, pack_cutoff: pd.Timestamp,
    source_symbols: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]], pd.Timestamp]:
    q = quality.copy()
    required_q = {"symbol", "source_hash", "rows", "first_timestamp", "last_timestamp", "tier"}
    required_l = {"symbol", "quality_tier", "liquidity_stratum", "median_dollar_volume_252"}
    if missing := required_q.difference(q.columns):
        raise ShadowRegistryError(f"quality metadata missing {sorted(missing)}")
    if missing := required_l.difference(liquidity.columns):
        raise ShadowRegistryError(f"liquidity metadata missing {sorted(missing)}")
    q["symbol"] = q.symbol.astype(str)
    q["first_timestamp"] = pd.to_datetime(q.first_timestamp)
    q["last_timestamp"] = pd.to_datetime(q.last_timestamp)
    l = liquidity[list(required_l)].copy()
    l["symbol"] = l.symbol.astype(str)
    if q.symbol.duplicated().any() or l.symbol.duplicated().any():
        raise ShadowRegistryError("quality/liquidity symbol identities are not unique")
    if source_symbols is None:
        source_symbols = [key.source_symbol for key in source.instruments()]
    all_symbols = sorted(set(source_symbols) | set(q.symbol) | set(l.symbol))
    source_set = set(source_symbols)
    qmap = {str(row.symbol): row for row in q.itertuples(index=False)}
    lmap = {str(row.symbol): row for row in l.itertuples(index=False)}
    instrument_map = {item.source_symbol: item for item in source.instruments()}
    latest = pd.Timestamp(q.last_timestamp.max())
    freshness_floor = latest - pd.Timedelta(days=int(contract["maximum_staleness_days"]))
    benchmark = source.load_benchmark()
    if benchmark is None:
        raise ShadowRegistryError("NASDAQ shadow requires benchmark data")
    benchmark_dates = set(pd.to_datetime(benchmark.timestamp))
    lookback = int(contract["lookback"])
    gap = int(contract["minimum_history_gap_bars"])
    quality_tiers = set(map(str, contract["quality_tiers"]))
    liquidity_strata = set(map(str, contract["liquidity_strata"]))
    rows: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    for symbol in all_symbols:
        qr = qmap.get(symbol)
        lr = lmap.get(symbol)
        reason: str | None = None
        if symbol not in source_set:
            reason = "missing_source_file"
        elif qr is None:
            reason = "missing_quality_metadata"
        elif str(qr.tier) not in quality_tiers:
            reason = "quarantined_quality"
        elif lr is None or str(lr.quality_tier) not in quality_tiers \
                or str(lr.liquidity_stratum) not in liquidity_strata:
            reason = "missing_liquidity_metadata"
        elif int(qr.rows) < int(contract["minimum_history_rows"]):
            reason = "insufficient_query_history"
        elif pd.Timestamp(qr.last_timestamp) < freshness_floor:
            reason = "stale_at_source_lock"

        cutoff: pd.Timestamp | None = None
        latest_eligible: pd.Timestamp | None = None
        episode_id: str | None = None
        stock_prefix: dict[str, Any] | None = None
        benchmark_prefix: dict[str, Any] | None = None
        if reason is None:
            bars = source.load(instrument_map[symbol])
            timestamps = pd.to_datetime(bars.timestamp)
            if len(bars) != int(qr.rows) or pd.Timestamp(timestamps.iloc[-1]) != pd.Timestamp(qr.last_timestamp):
                raise ShadowRegistryError(f"quality/source row identity differs: {symbol}")
            cutoff = pd.Timestamp(timestamps.iloc[-1])
            if cutoff not in benchmark_dates:
                reason = "missing_benchmark_cutoff"
            else:
                latest_eligible = pd.Timestamp(timestamps.iloc[-gap - 1])
                if latest_eligible > pack_cutoff:
                    reason = "packed_temporal_coverage_unavailable"
                else:
                    key = EpisodeKey(
                        instrument_map[symbol], cutoff, lookback,
                        str(contract["representation_version"]),
                    )
                    episode_id = key.id
                    stock_prefix = asdict(causal_prefix_digest(bars, cutoff))
                    benchmark_prefix = asdict(causal_prefix_digest(benchmark, cutoff))

        status = "scheduled" if reason is None else "skipped"
        row = {
            "symbol": symbol, "status": status, "skip_reason": reason,
            "quality_tier": str(qr.tier) if qr is not None else None,
            "liquidity_stratum": str(lr.liquidity_stratum) if lr is not None else None,
            "median_dollar_volume_252": (
                float(lr.median_dollar_volume_252) if lr is not None else None
            ),
            "rows_at_lock": int(qr.rows) if qr is not None else None,
            "first_timestamp_at_lock": (
                pd.Timestamp(qr.first_timestamp).isoformat() if qr is not None else None
            ),
            "last_timestamp_at_lock": (
                pd.Timestamp(qr.last_timestamp).isoformat() if qr is not None else None
            ),
            "source_hash_at_lock": str(qr.source_hash) if qr is not None else None,
            "cutoff": cutoff.isoformat() if cutoff is not None else None,
            "latest_eligible_cutoff": (
                latest_eligible.isoformat() if latest_eligible is not None else None
            ),
            "packed_cutoff": pack_cutoff.isoformat(),
            "case_id": f"nasdaq-{symbol}-shadow-current-{lookback}" if status == "scheduled" else None,
            "episode_id": episode_id,
        }
        rows.append(row)
        if status == "scheduled":
            active = int(((q.first_timestamp <= cutoff) & (q.last_timestamp >= cutoff)
                          & q.tier.astype(str).isin(quality_tiers)).sum())
            cases.append({
                "case_id": row["case_id"], "dataset_id": "nasdaq", "symbol": symbol,
                "quality_tier": str(lr.quality_tier),
                "liquidity_stratum": str(lr.liquidity_stratum),
                "median_dollar_volume_252": float(lr.median_dollar_volume_252),
                "cutoff_role": "shadow_current", "cutoff": row["cutoff"],
                "lookback": lookback,
                "representation_version": str(contract["representation_version"]),
                "episode_id": episode_id, "stock_prefix": stock_prefix,
                "benchmark_prefix": benchmark_prefix,
                "latest_eligible_cutoff": row["latest_eligible_cutoff"],
                "active_source_universe": active,
            })
    denominator = pd.DataFrame(rows).sort_values("symbol", kind="stable", ignore_index=True)
    case_frame = pd.DataFrame(cases).sort_values("symbol", kind="stable", ignore_index=True)
    sample: list[dict[str, Any]] = []
    seed = str(contract["snapshot_seed"])
    for (tier, stratum), cell in case_frame.groupby(
        ["quality_tier", "liquidity_stratum"], sort=True,
    ):
        ranked = sorted(cell.to_dict(orient="records"), key=lambda row: (
            sha256(f"{seed}:audit:{row['episode_id']}".encode()).hexdigest(), row["symbol"],
        ))
        if len(ranked) < 2:
            raise ShadowRegistryError(f"audit cell lacks two rows: {tier}/{stratum}")
        for rank, row in enumerate(ranked[:2], 1):
            sample.append({
                "quality_tier": str(tier), "liquidity_stratum": str(stratum),
                "cell_rank": rank, "symbol": row["symbol"],
                "case_id": row["case_id"], "episode_id": row["episode_id"],
                "selection_hash": sha256(
                    f"{seed}:audit:{row['episode_id']}".encode()
                ).hexdigest(),
            })
    sample.sort(key=lambda row: (row["quality_tier"], row["liquidity_stratum"], row["cell_rank"]))
    return denominator, case_frame, sample, latest


def execute(repository: Path, config_path: Path, output: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    if subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=repository,
        text=True, capture_output=True, check=True,
    ).stdout:
        raise ShadowRegistryError("shadow denominator requires a globally clean Git worktree")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True,
        capture_output=True, check=True,
    ).stdout.strip()
    contract, contract_raw = _read(repository / CONTRACT)
    config = load_config(config_path)
    source = source_from_spec(config.datasets["nasdaq"])
    quality_path = config.artifact_dir / "quality/nasdaq.parquet"
    liquidity_path = config.artifact_dir / "oracles/nasdaq/liquidity-strata.parquet"
    quality = pd.read_parquet(quality_path)
    liquidity = pd.read_parquet(liquidity_path)
    pack_path, pack = _pack_manifest(repository)
    requested = (pack.get("provenance") or {}).get("benchmark_prefix", {}).get("requested_cutoff")
    if requested is None:
        raise ShadowRegistryError("packed generation has no requested cutoff")
    pack_cutoff = pd.Timestamp(requested)
    source_symbols, source_inventory_digest = _source_inventory(source, quality)
    denominator, cases, sample, latest = _classify(
        repository, contract, source, quality, liquidity, pack_cutoff, source_symbols,
    )
    reason_counts = {
        str(key): int(value) for key, value in
        denominator.skip_reason.fillna("scheduled").value_counts().sort_index().items()
    }
    if len(denominator) != len(set(denominator.symbol)) \
            or len(cases) != int(reason_counts.get("scheduled", 0)) \
            or len(sample) != int(contract["audit"]["sample_size"]):
        raise ShadowRegistryError("denominator, scheduled-case or audit inventory differs")
    source_lock: dict[str, Any] = {
        "git_head": head,
        "contract_sha256": sha256(contract_raw).hexdigest(),
        "config_path": str(config_path.resolve()),
        "config_sha256": file_fingerprint(config_path),
        "quality_path": str(quality_path.resolve()),
        "quality_sha256": file_fingerprint(quality_path),
        "liquidity_path": str(liquidity_path.resolve()),
        "liquidity_sha256": file_fingerprint(liquidity_path),
        "benchmark_sha256": source.benchmark_fingerprint(),
        "source_symbol_count": len(source_symbols),
        "source_symbol_digest": stable_hash(source_symbols),
        "source_inventory_digest": source_inventory_digest,
        "packed_manifest_path": str(pack_path.resolve()),
        "packed_manifest_sha256": file_fingerprint(pack_path),
        "packed_generation_id": pack["manifest_digest"],
        "packed_provenance_digest": pack["provenance_digest"],
        "packed_cutoff": pack_cutoff.isoformat(),
        "real_forward_outcomes_accessed": False,
    }
    source_lock["source_lock_digest"] = stable_hash(source_lock)
    denominator_records = _records(denominator)
    case_records = _records(cases)
    state: dict[str, Any] = {
        "schema_version": SCHEMA, "status": "sealed", "passed": True,
        "contract": contract, "source_lock": source_lock,
        "market_latest_timestamp_at_lock": latest.isoformat(),
        "freshness_floor": (
            latest - pd.Timedelta(days=int(contract["maximum_staleness_days"]))
        ).isoformat(),
        "denominator_symbols": len(denominator_records),
        "scheduled_queries": len(case_records),
        "skipped_symbols": len(denominator_records) - len(case_records),
        "reason_counts": reason_counts,
        "denominator_digest": stable_hash(denominator_records),
        "cases_digest": stable_hash(case_records),
        "audit_sample": sample, "audit_sample_digest": stable_hash(sample),
        "cases_data": case_records,
        "real_forward_outcomes_accessed": False,
        "production_promotion_authorized": False,
    }
    state["registry_digest"] = stable_hash(state)
    if _contains_forbidden_key(state):
        raise ShadowRegistryError("outcome/search-result key entered denominator")
    output = output.resolve()
    if output.exists() or output.is_symlink():
        raise ShadowRegistryError("shadow denominator root already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        denominator.to_parquet(temporary / "denominator.parquet", index=False)
        cases.to_parquet(temporary / "query-registry.parquet", index=False)
        _atomic(temporary / "query-registry.json", json.dumps(
            state, indent=2, sort_keys=True, allow_nan=False,
        ).encode() + b"\n")
        rows = "".join(
            f"<tr><td>{escape(key)}</td><td>{value}</td></tr>"
            for key, value in sorted(reason_counts.items())
        )
        audit = "".join(
            f"<tr><td>{escape(row['quality_tier'])}/{escape(row['liquidity_stratum'])}</td>"
            f"<td>{escape(row['symbol'])}</td><td><code>{escape(row['episode_id'])}</code></td></tr>"
            for row in sample
        )
        report = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>T14-08 NASDAQ shadow denominator</title><style>body{{font-family:system-ui;max-width:1100px;margin:2rem auto;line-height:1.45}}table{{border-collapse:collapse;width:100%}}th,td{{padding:.45rem;border-bottom:1px solid #ddd;text-align:left}}.pass{{color:#075}}code{{overflow-wrap:anywhere}}</style></head><body><h1>T14-08 shadow denominator: <span class="pass">PASS</span></h1><p>All {len(denominator_records):,} audited NASDAQ source symbols are accounted for. {len(case_records):,} are scheduled for certified outcome-blind retrieval; {len(denominator_records)-len(case_records):,} have a deterministic skip reason.</p><h2>Accounting</h2><table><tbody>{rows}</tbody></table><h2>Preregistered exhaustive-audit sample</h2><table><tbody>{audit}</tbody></table><h2>Boundary</h2><p>This seals the denominator only. It does not claim that shadow retrieval, sustained drift, outcome usefulness or production promotion has passed.</p></body></html>"""
        _atomic(temporary / "denominator.html", report.encode())
        manifest = [{
            "path": path.name, "bytes": path.stat().st_size,
            "sha256": file_fingerprint(path),
        } for path in sorted(temporary.iterdir())]
        seal_state = {
            "schema_version": SEAL_SCHEMA, "status": "sealed", "passed": True,
            "registry_digest": state["registry_digest"], "files": manifest,
            "manifest_digest": stable_hash(manifest),
            "real_forward_outcomes_accessed": False,
            "production_promotion_authorized": False,
        }
        seal = {**seal_state, "seal_digest": stable_hash(seal_state),
                "created_at": datetime.now(timezone.utc).isoformat()}
        _atomic(temporary / "SEALED.json", json.dumps(
            seal, indent=2, sort_keys=True, allow_nan=False,
        ).encode() + b"\n")
        os.replace(temporary, output)
    except BaseException:
        for path in sorted(temporary.glob("*")):
            path.unlink()
        temporary.rmdir()
        raise
    return {"registry_digest": state["registry_digest"],
            "denominator_symbols": len(denominator_records),
            "scheduled_queries": len(case_records), "reason_counts": reason_counts,
            "output_root": str(output)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args(argv)
    output = args.output_root or args.repository / DEFAULT_OUTPUT
    result = execute(args.repository, args.config.resolve(), output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
