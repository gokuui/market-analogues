"""Independently reconstruct and verify the T14-08 NASDAQ denominator."""
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

from market_analogues.adapters import file_fingerprint, source_from_spec
from market_analogues.causal_prefix import causal_prefix_digest
from market_analogues.config import load_config
from market_analogues.types import EpisodeKey, stable_hash


SCHEMA = "m04r14-shadow-denominator-verification-v1"
REGISTRY_SCHEMA = "m04r14-shadow-denominator-v1"
SEAL_SCHEMA = "m04r14-shadow-denominator-seal-v1"
CONTRACT = Path("config/m04r14-shadow-contract.json")
BUILDER = Path("experiments/m04r/m04r14_shadow_registry.py")
REGISTRY = Path("config/data/analogues/m04r14/nasdaq-shadow-denominator-v1")
VERIFICATION = Path("config/data/analogues/m04r14/nasdaq-shadow-denominator-v1-verification")
FORBIDDEN_KEYS = {
    "outcome", "outcomes", "forward_return", "forward_returns", "winner",
    "loser", "profit", "loss", "setup", "setup_label", "matches",
}


class ShadowVerificationError(RuntimeError):
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
        raise ShadowVerificationError(f"regular file required: {path}")
    raw = path.read_bytes()

    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise ShadowVerificationError(f"duplicate JSON key: {path}:{key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw, object_pairs_hook=pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ShadowVerificationError(f"non-finite JSON: {path}:{item}")),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ShadowVerificationError(f"invalid JSON: {path}") from exc
    if type(value) is not dict:
        raise ShadowVerificationError(f"JSON object required: {path}")
    return value, raw


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
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
        result.append(row)
    return result


def _assert_finite(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ShadowVerificationError("non-finite evidence value")
    if isinstance(value, dict):
        for item in value.values():
            _assert_finite(item)
    elif isinstance(value, list):
        for item in value:
            _assert_finite(item)


def _reconstruct(
    contract: Mapping[str, Any], source: Any, quality: pd.DataFrame,
    liquidity: pd.DataFrame, pack_cutoff: pd.Timestamp,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], str, pd.Timestamp]:
    q = quality.copy()
    q["symbol"] = q.symbol.astype(str)
    q["first_timestamp"] = pd.to_datetime(q.first_timestamp)
    q["last_timestamp"] = pd.to_datetime(q.last_timestamp)
    l = liquidity[[
        "symbol", "quality_tier", "liquidity_stratum", "median_dollar_volume_252",
    ]].copy()
    l["symbol"] = l.symbol.astype(str)
    if q.symbol.duplicated().any() or l.symbol.duplicated().any():
        raise ShadowVerificationError("metadata symbol identities are not unique")
    instruments = source.instruments()
    source_symbols = [item.source_symbol for item in instruments]
    if len(source_symbols) != len(set(source_symbols)):
        raise ShadowVerificationError("source symbol identities are not unique")
    source_map = {item.source_symbol: item for item in instruments}
    qmap = {str(row.symbol): row for row in q.itertuples(index=False)}
    lmap = {str(row.symbol): row for row in l.itertuples(index=False)}
    inventory: list[tuple[str, str]] = []
    for symbol in sorted(source_symbols):
        fingerprint = source.fingerprint(source_map[symbol])
        qr = qmap.get(symbol)
        if qr is not None and fingerprint != str(qr.source_hash):
            raise ShadowVerificationError(f"quality/source fingerprint differs: {symbol}")
        inventory.append((symbol, fingerprint))
    inventory_digest = stable_hash(inventory)
    all_symbols = sorted(set(source_symbols) | set(qmap) | set(lmap))
    latest = pd.Timestamp(q.last_timestamp.max())
    floor = latest - pd.Timedelta(days=int(contract["maximum_staleness_days"]))
    benchmark = source.load_benchmark()
    if benchmark is None:
        raise ShadowVerificationError("benchmark is unavailable")
    benchmark_dates = set(pd.to_datetime(benchmark.timestamp))
    lookback = int(contract["lookback"])
    gap = int(contract["minimum_history_gap_bars"])
    tiers = set(map(str, contract["quality_tiers"]))
    strata = set(map(str, contract["liquidity_strata"]))
    denominator: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    for symbol in all_symbols:
        qr = qmap.get(symbol)
        lr = lmap.get(symbol)
        reason = None
        if symbol not in source_map:
            reason = "missing_source_file"
        elif qr is None:
            reason = "missing_quality_metadata"
        elif str(qr.tier) not in tiers:
            reason = "quarantined_quality"
        elif lr is None or str(lr.quality_tier) not in tiers \
                or str(lr.liquidity_stratum) not in strata:
            reason = "missing_liquidity_metadata"
        elif int(qr.rows) < int(contract["minimum_history_rows"]):
            reason = "insufficient_query_history"
        elif pd.Timestamp(qr.last_timestamp) < floor:
            reason = "stale_at_source_lock"
        cutoff = latest_eligible = None
        episode_id = None
        stock_prefix = benchmark_prefix = None
        if reason is None:
            bars = source.load(source_map[symbol])
            timestamps = pd.to_datetime(bars.timestamp)
            if len(bars) != int(qr.rows) or pd.Timestamp(timestamps.iloc[-1]) != pd.Timestamp(qr.last_timestamp):
                raise ShadowVerificationError(f"quality/source row identity differs: {symbol}")
            cutoff = pd.Timestamp(timestamps.iloc[-1])
            if cutoff not in benchmark_dates:
                reason = "missing_benchmark_cutoff"
            else:
                latest_eligible = pd.Timestamp(timestamps.iloc[-gap - 1])
                if latest_eligible > pack_cutoff:
                    reason = "packed_temporal_coverage_unavailable"
                else:
                    key = EpisodeKey(
                        source_map[symbol], cutoff, lookback,
                        str(contract["representation_version"]),
                    )
                    episode_id = key.id
                    stock_prefix = asdict(causal_prefix_digest(bars, cutoff))
                    benchmark_prefix = asdict(causal_prefix_digest(benchmark, cutoff))
        status = "scheduled" if reason is None else "skipped"
        record = {
            "symbol": symbol, "status": status, "skip_reason": reason,
            "quality_tier": str(qr.tier) if qr is not None else None,
            "liquidity_stratum": str(lr.liquidity_stratum) if lr is not None else None,
            "median_dollar_volume_252": float(lr.median_dollar_volume_252) if lr is not None else None,
            "rows_at_lock": int(qr.rows) if qr is not None else None,
            "first_timestamp_at_lock": pd.Timestamp(qr.first_timestamp).isoformat() if qr is not None else None,
            "last_timestamp_at_lock": pd.Timestamp(qr.last_timestamp).isoformat() if qr is not None else None,
            "source_hash_at_lock": str(qr.source_hash) if qr is not None else None,
            "cutoff": cutoff.isoformat() if cutoff is not None else None,
            "latest_eligible_cutoff": latest_eligible.isoformat() if latest_eligible is not None else None,
            "packed_cutoff": pack_cutoff.isoformat(),
            "case_id": f"nasdaq-{symbol}-shadow-current-{lookback}" if status == "scheduled" else None,
            "episode_id": episode_id,
        }
        denominator.append(record)
        if status == "scheduled":
            active = int(((q.first_timestamp <= cutoff) & (q.last_timestamp >= cutoff)
                          & q.tier.astype(str).isin(tiers)).sum())
            cases.append({
                "case_id": record["case_id"], "dataset_id": "nasdaq", "symbol": symbol,
                "quality_tier": str(lr.quality_tier),
                "liquidity_stratum": str(lr.liquidity_stratum),
                "median_dollar_volume_252": float(lr.median_dollar_volume_252),
                "cutoff_role": "shadow_current", "cutoff": record["cutoff"],
                "lookback": lookback, "representation_version": str(contract["representation_version"]),
                "episode_id": episode_id, "stock_prefix": stock_prefix,
                "benchmark_prefix": benchmark_prefix,
                "latest_eligible_cutoff": record["latest_eligible_cutoff"],
                "active_source_universe": active,
            })
    cases.sort(key=lambda row: row["symbol"])
    sample: list[dict[str, Any]] = []
    seed = str(contract["snapshot_seed"])
    frame = pd.DataFrame(cases)
    for (tier, stratum), cell in frame.groupby(["quality_tier", "liquidity_stratum"], sort=True):
        ranked = sorted(cell.to_dict(orient="records"), key=lambda row: (
            sha256(f"{seed}:audit:{row['episode_id']}".encode()).hexdigest(), row["symbol"],
        ))
        for rank, row in enumerate(ranked[:2], 1):
            sample.append({
                "quality_tier": str(tier), "liquidity_stratum": str(stratum),
                "cell_rank": rank, "symbol": row["symbol"], "case_id": row["case_id"],
                "episode_id": row["episode_id"], "selection_hash": sha256(
                    f"{seed}:audit:{row['episode_id']}".encode()
                ).hexdigest(),
            })
    sample.sort(key=lambda row: (row["quality_tier"], row["liquidity_stratum"], row["cell_rank"]))
    return denominator, cases, sample, inventory_digest, latest


def verify(repository: Path, config_path: Path, root: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    root = root.resolve(strict=True)
    if root != (repository / REGISTRY).resolve(strict=True):
        raise ShadowVerificationError("registry root differs")
    required = {
        "SEALED.json", "denominator.html", "denominator.parquet",
        "query-registry.json", "query-registry.parquet",
    }
    if {path.name for path in root.iterdir()} != required \
            or any(path.is_symlink() or not path.is_file() for path in root.iterdir()):
        raise ShadowVerificationError("registry tree differs")
    payload, payload_raw = _read(root / "query-registry.json")
    seal, seal_raw = _read(root / "SEALED.json")
    _assert_finite(payload)
    if _contains_forbidden_key(payload):
        raise ShadowVerificationError("outcome/search-result key entered denominator")
    manifest = [{
        "path": path.name, "bytes": path.stat().st_size, "sha256": file_fingerprint(path),
    } for path in sorted(root.iterdir()) if path.name != "SEALED.json"]
    seal_state = {key: value for key, value in seal.items() if key not in {"seal_digest", "created_at"}}
    if seal.get("schema_version") != SEAL_SCHEMA or seal.get("files") != manifest \
            or seal.get("manifest_digest") != stable_hash(manifest) \
            or seal.get("seal_digest") != stable_hash(seal_state):
        raise ShadowVerificationError("denominator seal differs")
    contract, contract_raw = _read(repository / CONTRACT)
    source_lock = payload.get("source_lock") or {}
    marker_head = source_lock.get("git_head")
    if type(marker_head) is not str or len(marker_head) != 40:
        raise ShadowVerificationError("source-lock Git head differs")
    subprocess.run(["git", "cat-file", "-e", f"{marker_head}^{{commit}}"], cwd=repository, check=True)
    committed_builder = subprocess.run(
        ["git", "show", f"{marker_head}:{BUILDER.as_posix()}"], cwd=repository,
        capture_output=True, check=True,
    ).stdout
    if committed_builder != (repository / BUILDER).read_bytes():
        raise ShadowVerificationError("committed/current builder differs")
    config = load_config(config_path)
    source = source_from_spec(config.datasets["nasdaq"])
    quality_path = config.artifact_dir / "quality/nasdaq.parquet"
    liquidity_path = config.artifact_dir / "oracles/nasdaq/liquidity-strata.parquet"
    quality = pd.read_parquet(quality_path)
    liquidity = pd.read_parquet(liquidity_path)
    pack_path = Path(str(source_lock.get("packed_manifest_path")))
    pack, _ = _read(pack_path)
    pack_cutoff = pd.Timestamp(source_lock.get("packed_cutoff"))
    denominator, cases, sample, inventory_digest, latest = _reconstruct(
        contract, source, quality, liquidity, pack_cutoff,
    )
    source_symbols = sorted(item.source_symbol for item in source.instruments())
    expected_lock: dict[str, Any] = {
        "git_head": marker_head,
        "contract_sha256": sha256(contract_raw).hexdigest(),
        "config_path": str(config_path.resolve()), "config_sha256": file_fingerprint(config_path),
        "quality_path": str(quality_path.resolve()), "quality_sha256": file_fingerprint(quality_path),
        "liquidity_path": str(liquidity_path.resolve()), "liquidity_sha256": file_fingerprint(liquidity_path),
        "benchmark_sha256": source.benchmark_fingerprint(),
        "source_symbol_count": len(source_symbols), "source_symbol_digest": stable_hash(source_symbols),
        "source_inventory_digest": inventory_digest,
        "packed_manifest_path": str(pack_path.resolve()),
        "packed_manifest_sha256": file_fingerprint(pack_path),
        "packed_generation_id": pack["manifest_digest"],
        "packed_provenance_digest": pack["provenance_digest"],
        "packed_cutoff": pack_cutoff.isoformat(), "real_forward_outcomes_accessed": False,
    }
    expected_lock["source_lock_digest"] = stable_hash(expected_lock)
    if source_lock != expected_lock:
        raise ShadowVerificationError("source lock differs")
    stored_denominator = _records(pd.read_parquet(root / "denominator.parquet"))
    stored_cases = _records(pd.read_parquet(root / "query-registry.parquet"))
    if denominator != stored_denominator or cases != stored_cases \
            or cases != payload.get("cases_data"):
        raise ShadowVerificationError("independent denominator/case reconstruction differs")
    reason_counts = {
        str(key): int(value) for key, value in
        pd.Series([row["skip_reason"] or "scheduled" for row in denominator]).value_counts().sort_index().items()
    }
    state_without_digest = {key: value for key, value in payload.items() if key != "registry_digest"}
    if not all((
        payload.get("schema_version") == REGISTRY_SCHEMA,
        payload.get("status") == "sealed", payload.get("passed") is True,
        payload.get("contract") == contract, payload.get("source_lock") == expected_lock,
        payload.get("denominator_symbols") == len(denominator),
        payload.get("scheduled_queries") == len(cases),
        payload.get("skipped_symbols") == len(denominator) - len(cases),
        payload.get("reason_counts") == reason_counts,
        payload.get("denominator_digest") == stable_hash(denominator),
        payload.get("cases_digest") == stable_hash(cases),
        payload.get("audit_sample") == sample,
        payload.get("audit_sample_digest") == stable_hash(sample),
        payload.get("registry_digest") == stable_hash(state_without_digest),
        seal.get("registry_digest") == payload.get("registry_digest"),
        payload.get("real_forward_outcomes_accessed") is False,
        payload.get("production_promotion_authorized") is False,
    )):
        raise ShadowVerificationError("denominator aggregate differs")
    if len(sample) != int(contract["audit"]["sample_size"]):
        raise ShadowVerificationError("audit sample size differs")
    sample_cells = pd.DataFrame(sample).groupby(["quality_tier", "liquidity_stratum"]).size()
    if len(sample_cells) != 6 or set(sample_cells.tolist()) != {2}:
        raise ShadowVerificationError("audit sample cell coverage differs")
    result: dict[str, Any] = {
        "schema_version": SCHEMA, "status": "verified", "passed": True,
        "registry_root": str(root), "registry_digest": payload["registry_digest"],
        "registry_sha256": sha256(payload_raw).hexdigest(),
        "seal_sha256": sha256(seal_raw).hexdigest(), "seal_digest": seal["seal_digest"],
        "source_lock_digest": expected_lock["source_lock_digest"],
        "denominator_symbols": len(denominator), "scheduled_queries": len(cases),
        "skipped_symbols": len(denominator) - len(cases), "reason_counts": reason_counts,
        "audit_sample_size": len(sample), "market_latest": latest.isoformat(),
        "real_forward_outcomes_accessed": False,
        "production_promotion_authorized": False,
    }
    result["result_digest"] = stable_hash(result)
    return result


def _publish(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise ShadowVerificationError("verification root exists")
    path.mkdir(parents=False)
    descriptor = os.open(path / "VERIFIED.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(json.dumps({
            **value, "created_at": datetime.now(timezone.utc).isoformat(),
        }, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--registry-root", type=Path)
    parser.add_argument("--verification-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    repository = args.repository.resolve(strict=True)
    root = args.registry_root or repository / REGISTRY
    result = verify(repository, args.config.resolve(), root)
    if not args.dry_run:
        _publish(args.verification_root or repository / VERIFICATION, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
