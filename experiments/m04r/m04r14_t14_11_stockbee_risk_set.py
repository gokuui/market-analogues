"""Build the preregistered, restartable 12-shard Stockbee universe risk set."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
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

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from market_analogues.stockbee_study import symbol_risk_rows, valid_ohlcv_rows
from market_analogues.types import stable_hash

from experiments.m04r import m04r14_t14_09_outcome_smoke as smoke
from experiments.m04r import m04r14_t14_10_wf03_feasibility as base


SCHEMA = "m04r14-t14-11-stockbee-risk-set-v2"
PREREGISTRATION_RELATIVE = Path("experiments/m04r/m04r14_t14_11_stockbee_risk_set_v2_preregistered.json")
V1_PREREGISTRATION_RELATIVE = Path("experiments/m04r/m04r14_t14_11_stockbee_risk_set_v1_preregistered.json")
CONTRACT_RELATIVE = Path("config/m04r14-t14-11-stockbee-contract.json")
ACCOUNTING_RELATIVE = Path("config/data/analogues/m04r14/t14-10-walk-forward-query-registry-v1/source-accounting.parquet")
CACHE_RELATIVE = Path("config/data/analogues/m04r14/t14-11-stockbee-risk-set-v2-cache")
OUTPUT_RELATIVE = Path("config/data/analogues/m04r14/t14-11-stockbee-risk-set-v2")
VERIFICATION_RELATIVE = Path("config/data/analogues/m04r14/t14-11-stockbee-risk-set-v2-verification")
SHARDS = 12
RUNTIME_FILES = (
    "experiments/m04r/m04r14_t14_11_stockbee_risk_set.py",
    "experiments/m04r/verify_m04r14_t14_11_stockbee_risk_set.py",
    "src/market_analogues/stockbee_study.py",
    "config/m04r14-t14-11-stockbee-contract.json",
    "pyproject.toml",
)


class RiskSetError(RuntimeError):
    pass


def _now() -> str: return datetime.now(timezone.utc).isoformat()


def _git(repository: Path, *args: str, raw: bool = False) -> str | bytes:
    result = subprocess.run(["git", *args], cwd=repository, capture_output=True, text=not raw, check=False)
    if result.returncode:
        error = result.stderr if not raw else result.stderr.decode(errors="replace")
        raise RiskSetError(error.strip() or "git command failed")
    return result.stdout if raw else result.stdout.strip()


def _sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20): digest.update(block)
    return digest.hexdigest()


def _seal(value: Mapping[str, Any], key: str = "result_digest", *, timing: bool = False) -> dict[str, Any]:
    omitted = {"elapsed_seconds"} if timing else set(); result = dict(value)
    result[key] = stable_hash({k: v for k, v in result.items() if k not in omitted}); result["created_at"] = _now(); return result


def _valid(value: Mapping[str, Any], key: str = "result_digest", *, timing: bool = False) -> bool:
    omitted = {key, "created_at"} | ({"elapsed_seconds"} if timing else set())
    return value.get(key) == stable_hash({k: v for k, v in value.items() if k not in omitted})


def _manifest(root: Path, names: Sequence[str]) -> list[dict[str, Any]]:
    return [{"path": name, "bytes": (root / name).stat().st_size, "sha256": _sha(root / name)} for name in names]


def _contract(repository: Path) -> dict[str, Any]:
    value = base._read(repository / CONTRACT_RELATIVE); digest = value.pop("contract_digest")
    if digest != stable_hash(value): raise RiskSetError("Stockbee contract seal differs")
    value["contract_digest"] = digest; return value


def _accounting(repository: Path) -> pd.DataFrame:
    frame = pd.read_parquet(repository / ACCOUNTING_RELATIVE).copy()
    required = {"symbol", "source_path", "rows_through_lock", "coverage_last_timestamp", "source_hash_at_lock", "error"}
    if not required.issubset(frame) or frame.symbol.astype(str).duplicated().any() or frame.error.notna().any():
        raise RiskSetError("source accounting differs")
    return frame.sort_values("symbol", kind="stable").reset_index(drop=True)


def _runtime_manifest(repository: Path, head: str) -> dict[str, str]:
    tracked = set(str(_git(repository, "ls-tree", "-r", "--name-only", head)).splitlines())
    if any(name not in tracked for name in RUNTIME_FILES): raise RiskSetError("runtime file is uncommitted")
    return {name: sha256(_git(repository, "show", f"{head}:{name}", raw=True)).hexdigest() for name in RUNTIME_FILES}


def _shard(symbol: str) -> int:
    return int.from_bytes(sha256(str(symbol).encode()).digest()[:8], "big") % SHARDS


def _v1_failure(repository: Path) -> dict[str, Any]:
    prereg = base._read(repository / V1_PREREGISTRATION_RELATIVE)
    root = repository / "config/data/analogues/m04r14/t14-11-stockbee-risk-set-v1-cache"
    sealed = sorted(root.glob("shard-*/SHARD_SEALED.json"))
    if prereg.get("preregistration_digest") != "4599da9e03eda120c156c260fd04c84caee80fd496a19dc2faefa0984fd5b3a2" \
            or [path.parent.name for path in sealed] != ["shard-08"]:
        raise RiskSetError("V1 failure evidence differs")
    return {
        "preregistration_digest": prereg["preregistration_digest"],
        "preregistration_sha256": _sha(repository / V1_PREREGISTRATION_RELATIVE),
        "sealed_shards": ["shard-08"], "sealed_shard_result_digest": base._read(sealed[0])["result_digest"],
        "failure": "whole_symbol_OHLCV_rejection_encountered_31_histories_with_invalid_rows",
        "repair": "exclude_only_start_windows_whose_252_prior_through_future_endpoint_intersects_invalid_row",
        "real_universe_outcomes_accessed": True, "v1_receipts_reused": False,
    }


def build_preregistration(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    if _git(repository, "status", "--porcelain", "--untracked-files=all"): raise RiskSetError("clean worktree required")
    if (repository / CACHE_RELATIVE).exists() or (repository / OUTPUT_RELATIVE).exists():
        raise RiskSetError("risk-set namespaces must be absent")
    contract = _contract(repository); accounting = _accounting(repository); h0 = str(_git(repository, "rev-parse", "HEAD"))
    if _sha(repository / ACCOUNTING_RELATIVE) != contract["source"]["source_accounting_sha256"]:
        raise RiskSetError("source accounting hash differs")
    assignments = {str(row.symbol): _shard(str(row.symbol)) for row in accounting.itertuples(index=False)}
    state = {
        "schema_version": SCHEMA, "status": "frozen_before_real_universe_risk_scan",
        "implementation_h0": h0, "runtime_files": _runtime_manifest(repository, h0),
        "superseded_v1": _v1_failure(repository),
        "contract_digest": contract["contract_digest"], "source_accounting_sha256": _sha(repository / ACCOUNTING_RELATIVE),
        "symbol_count": len(accounting), "shards": SHARDS,
        "shard_symbol_counts": {str(i): sum(v == i for v in assignments.values()) for i in range(SHARDS)},
        "assignment_digest": stable_hash(assignments), "expected_horizons": [21, 63],
        "real_universe_outcomes_accessed": False, "production_promotion_authorized": False,
    }
    return _seal(state, "preregistration_digest")


def _sole_child(repository: Path, raw: bytes, h0: str) -> str:
    accepted = []
    for line in str(_git(repository, "rev-list", "--all", "--children")).splitlines():
        values = line.split()
        if not values or values[0] != h0: continue
        for child in values[1:]:
            parents = str(_git(repository, "rev-list", "--parents", "-n", "1", child)).split()
            changed = str(_git(repository, "diff-tree", "--no-commit-id", "--name-only", "-r", child)).splitlines()
            if parents == [child, h0] and changed == [PREREGISTRATION_RELATIVE.as_posix()] \
                    and _git(repository, "show", f"{child}:{PREREGISTRATION_RELATIVE}", raw=True) == raw:
                accepted.append(child)
    if len(set(accepted)) != 1: raise RiskSetError("expected one exact preregistration-only child")
    return accepted[0]


def validate_preregistration(repository: Path) -> tuple[dict[str, Any], pd.DataFrame, str]:
    if _git(repository, "status", "--porcelain", "--untracked-files=all"): raise RiskSetError("clean worktree required")
    path = repository / PREREGISTRATION_RELATIVE; raw = path.read_bytes(); prereg = base._read(path)
    if prereg.get("schema_version") != SCHEMA or not _valid(prereg, "preregistration_digest"):
        raise RiskSetError("preregistration differs")
    h0 = str(prereg["implementation_h0"]); h1 = _sole_child(repository, raw, h0)
    if subprocess.run(["git", "merge-base", "--is-ancestor", h1, "HEAD"], cwd=repository).returncode:
        raise RiskSetError("HEAD does not descend from preregistration")
    for name, expected in prereg["runtime_files"].items():
        if _sha(repository / name) != expected: raise RiskSetError(f"runtime drifted: {name}")
    accounting = _accounting(repository)
    observed = {
        "contract_digest": _contract(repository)["contract_digest"],
        "source_accounting_sha256": _sha(repository / ACCOUNTING_RELATIVE), "symbol_count": len(accounting),
        "assignment_digest": stable_hash({str(r.symbol): _shard(str(r.symbol)) for r in accounting.itertuples(index=False)}),
        "superseded_v1": _v1_failure(repository),
    }
    if any(prereg.get(k) != v for k, v in observed.items()): raise RiskSetError("preregistered inputs drifted")
    return prereg, accounting, h1


def _write_shard(shard: int, records: list[dict[str, Any]], cache_text: str, contract_digest: str) -> dict[str, Any]:
    cache = Path(cache_text); final_root = cache / f"shard-{shard:02d}"
    seal_path = final_root / "SHARD_SEALED.json"
    if seal_path.exists(): return json.loads(seal_path.read_text())
    if final_root.exists(): raise RiskSetError(f"partial shard exists: {shard}")
    started = perf_counter(); temporary = Path(tempfile.mkdtemp(prefix=f".shard-{shard:02d}.", dir=cache))
    writer: pq.ParquetWriter | None = None; accounting_rows = []; total_rows = winners = 0
    try:
        for record in records:
            path = Path(record["source_path"])
            if _sha(path) != record["source_hash_at_lock"]: raise RiskSetError(f"source drifted: {record['symbol']}")
            frame = pd.read_parquet(
                path, columns=["date", "open", "high", "low", "close", "volume"],
                engine="pyarrow", use_threads=False,
            )
            frame = frame.loc[pd.to_datetime(frame.date) <= pd.Timestamp(record["coverage_last_timestamp"])].rename(columns={"date": "timestamp"})
            if len(frame) != int(record["rows_through_lock"]): raise RiskSetError(f"locked row count differs: {record['symbol']}")
            invalid_rows = int((~valid_ohlcv_rows(frame)).sum())
            risk = symbol_risk_rows(frame, str(record["symbol"]))
            if not risk.empty:
                table = pa.Table.from_pandas(risk, preserve_index=False)
                if writer is None: writer = pq.ParquetWriter(temporary / "risk-set.parquet", table.schema, compression="zstd")
                writer.write_table(table); total_rows += len(risk); winners += int(risk.winner_25pct.sum())
            accounting_rows.append({
                "symbol": str(record["symbol"]), "source_rows": len(frame), "risk_rows": len(risk),
                "winner_rows": int(risk.winner_25pct.sum()) if len(risk) else 0,
                "invalid_source_rows": invalid_rows,
                "source_sha256": str(record["source_hash_at_lock"]),
            })
        if writer is None:
            raise RiskSetError(f"shard has no eligible risk rows: {shard}")
        writer.close(); writer = None
        smoke._atomic_parquet(temporary / "symbol-accounting.parquet", pd.DataFrame(accounting_rows))
        names = ("risk-set.parquet", "symbol-accounting.parquet")
        seal = _seal({
            "schema_version": SCHEMA, "status": "shard_sealed", "passed": True, "shard": shard,
            "contract_digest": contract_digest, "symbol_count": len(records), "risk_rows": total_rows,
            "winner_rows": winners, "file_manifest": _manifest(temporary, names),
            "elapsed_seconds": perf_counter() - started, "real_universe_outcomes_accessed": True,
        }, timing=True)
        smoke._atomic_json(temporary / "SHARD_SEALED.json", seal); os.replace(temporary, final_root); return seal
    except BaseException:
        if writer is not None: writer.close()
        shutil.rmtree(temporary, ignore_errors=True); raise


def execute(repository: Path, workers: int = SHARDS) -> dict[str, Any]:
    repository = repository.resolve(strict=True); prereg, accounting, h1 = validate_preregistration(repository)
    output = repository / OUTPUT_RELATIVE
    if output.exists():
        seal = base._read(output / "SEALED.json")
        if not _valid(seal, timing=True): raise RiskSetError("existing output seal differs")
        return seal
    cache = repository / CACHE_RELATIVE; cache.mkdir(parents=True, exist_ok=True)
    grouped: dict[int, list[dict[str, Any]]] = {i: [] for i in range(SHARDS)}
    for row in accounting.itertuples(index=False):
        grouped[_shard(str(row.symbol))].append({
            "symbol": str(row.symbol), "source_path": str(row.source_path),
            "rows_through_lock": int(row.rows_through_lock), "coverage_last_timestamp": str(row.coverage_last_timestamp),
            "source_hash_at_lock": str(row.source_hash_at_lock),
        })
    started = perf_counter(); seals = []
    with ProcessPoolExecutor(max_workers=min(max(1, workers), SHARDS)) as pool:
        futures = {pool.submit(_write_shard, shard, rows, str(cache), prereg["contract_digest"]): shard for shard, rows in grouped.items()}
        for future in as_completed(futures): seals.append(future.result())
    seals.sort(key=lambda x: x["shard"])
    if len(seals) != SHARDS or any(not _valid(x, timing=True) for x in seals): raise RiskSetError("shard completion differs")
    output.mkdir(parents=True)
    state = _seal({
        "schema_version": SCHEMA, "status": "sealed", "passed": True,
        "preregistration_h1": h1, "preregistration_digest": prereg["preregistration_digest"],
        "contract_digest": prereg["contract_digest"], "shards": SHARDS,
        "symbol_count": sum(x["symbol_count"] for x in seals), "risk_rows": sum(x["risk_rows"] for x in seals),
        "winner_rows": sum(x["winner_rows"] for x in seals),
        "shard_result_digests": [x["result_digest"] for x in seals],
        "shard_seal_sha256": [_sha(cache / f"shard-{i:02d}" / "SHARD_SEALED.json") for i in range(SHARDS)],
        "elapsed_seconds": perf_counter() - started, "real_universe_outcomes_accessed": True,
        "exposure_control_prevalence_calculated": False, "independent_verification_authorized": True,
        "production_promotion_authorized": False,
    }, timing=True)
    smoke._atomic_json(output / "SEALED.json", state); return state


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="command", required=True)
    for command in ("preregister", "run"):
        child = sub.add_parser(command); child.add_argument("--repository", type=Path, required=True)
        if command == "run": child.add_argument("--workers", type=int, default=SHARDS)
    args = parser.parse_args(argv)
    if args.command == "preregister":
        value = build_preregistration(args.repository); smoke._atomic_json(args.repository / PREREGISTRATION_RELATIVE, value)
    else: value = execute(args.repository, args.workers)
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)); return 0


if __name__ == "__main__": raise SystemExit(main())
