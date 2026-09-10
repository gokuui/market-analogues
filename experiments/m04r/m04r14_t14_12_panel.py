"""Build the preregistered, restartable T14-12 causal daily panel."""
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

from market_analogues.post_signal_study import symbol_post_signal_panel
from market_analogues.stockbee_study import valid_ohlcv_rows
from market_analogues.types import stable_hash

from experiments.m04r import m04r14_t14_09_outcome_smoke as smoke
from experiments.m04r import m04r14_t14_10_wf03_feasibility as base
from experiments.m04r import m04r14_t14_11_stockbee_risk_set as source_stage
from experiments.m04r import m04r14_t14_12_synthetic_gate as synthetic


SCHEMA = "m04r14-t14-12-post-signal-panel-v1"
CONTRACT_RELATIVE = Path("config/m04r14-t14-12-post-signal-contract.json")
PREREGISTRATION_RELATIVE = Path("experiments/m04r/m04r14_t14_12_panel_v1_preregistered.json")
CACHE_RELATIVE = Path("config/data/analogues/m04r14/t14-12-post-signal-panel-v1-cache")
OUTPUT_RELATIVE = Path("config/data/analogues/m04r14/t14-12-post-signal-panel-v1")
VERIFICATION_RELATIVE = Path("config/data/analogues/m04r14/t14-12-post-signal-panel-v1-verification")
SHARDS = 12
SHARD_FILES = ("daily-panel.parquet", "symbol-accounting.parquet")
RUNTIME_FILES = (
    "experiments/m04r/m04r14_t14_12_panel.py",
    "experiments/m04r/verify_m04r14_t14_12_panel.py",
    "experiments/m04r/m04r14_t14_12_synthetic_gate.py",
    "src/market_analogues/post_signal_study.py",
    "config/m04r14-t14-12-post-signal-contract.json", "pyproject.toml",
)


class PanelError(RuntimeError): pass


def _now() -> str: return datetime.now(timezone.utc).isoformat()


def _git(repository: Path, *args: str, raw: bool = False) -> str | bytes:
    result = subprocess.run(["git", *args], cwd=repository, capture_output=True, text=not raw, check=False)
    if result.returncode:
        error = result.stderr if not raw else result.stderr.decode(errors="replace"); raise PanelError(error.strip())
    return result.stdout if raw else result.stdout.strip()


def _sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20): digest.update(block)
    return digest.hexdigest()


def _seal(value: Mapping[str, Any], key: str = "result_digest", *, timing: bool = False) -> dict[str, Any]:
    omitted = {"elapsed_seconds"} if timing else set(); result = dict(value)
    result[key] = stable_hash({name: item for name, item in result.items() if name not in omitted})
    result["created_at"] = _now(); return result


def _valid(value: Mapping[str, Any], key: str = "result_digest", *, timing: bool = False) -> bool:
    omitted = {key, "created_at"} | ({"elapsed_seconds"} if timing else set())
    return value.get(key) == stable_hash({name: item for name, item in value.items() if name not in omitted})


def _manifest(root: Path, names: Sequence[str]) -> list[dict[str, Any]]:
    return [{"path": name, "bytes": (root / name).stat().st_size, "sha256": _sha(root / name)} for name in names]


def _contract(repository: Path) -> dict[str, Any]:
    value = base._read(repository / CONTRACT_RELATIVE); digest = value.pop("contract_digest")
    if digest != stable_hash(value): raise PanelError("post-signal contract seal differs")
    value["contract_digest"] = digest; return value


def _synthetic_receipt(repository: Path) -> dict[str, Any]:
    path = repository / synthetic.OUTPUT_RELATIVE / "VERIFIED.json"; receipt = base._read(path)
    state = {name: item for name, item in receipt.items() if name not in {"verification_digest", "created_at"}}
    if receipt.get("verification_digest") != stable_hash(state) or receipt.get("passed") is not True \
            or receipt.get("real_source_rows_accessed") is not False:
        raise PanelError("synthetic receipt differs")
    return {"verification_digest": receipt["verification_digest"], "sha256": _sha(path),
            "implementation_h0": receipt["implementation_h0"]}


def _runtime_manifest(repository: Path, head: str) -> dict[str, str]:
    tracked = set(str(_git(repository, "ls-tree", "-r", "--name-only", head)).splitlines())
    if any(name not in tracked for name in RUNTIME_FILES): raise PanelError("runtime file is uncommitted")
    return {name: sha256(_git(repository, "show", f"{head}:{name}", raw=True)).hexdigest() for name in RUNTIME_FILES}


def build_preregistration(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    if _git(repository, "status", "--porcelain", "--untracked-files=all"): raise PanelError("clean worktree required")
    if (repository / CACHE_RELATIVE).exists() or (repository / OUTPUT_RELATIVE).exists():
        raise PanelError("panel namespaces must be absent")
    contract = _contract(repository); accounting = source_stage._accounting(repository)
    if _sha(repository / source_stage.ACCOUNTING_RELATIVE) != contract["source"]["source_accounting_sha256"] \
            or _sha(Path(contract["source"]["benchmark_path"])) != contract["source"]["benchmark_sha256"]:
        raise PanelError("source or benchmark binding differs")
    h0 = str(_git(repository, "rev-parse", "HEAD"))
    assignments = {str(row.symbol): source_stage._shard(str(row.symbol)) for row in accounting.itertuples(index=False)}
    return _seal({
        "schema_version": SCHEMA, "status": "frozen_before_real_post_signal_panel",
        "implementation_h0": h0, "runtime_files": _runtime_manifest(repository, h0),
        "contract_digest": contract["contract_digest"], "synthetic_receipt": _synthetic_receipt(repository),
        "source_accounting_sha256": _sha(repository / source_stage.ACCOUNTING_RELATIVE),
        "benchmark_sha256": _sha(Path(contract["source"]["benchmark_path"])),
        "symbol_count": len(accounting), "shards": SHARDS,
        "shard_symbol_counts": {str(i): sum(value == i for value in assignments.values()) for i in range(SHARDS)},
        "assignment_digest": stable_hash(assignments), "horizons": [5, 20, 60],
        "real_post_signal_outcomes_accessed": False, "production_promotion_authorized": False,
    }, "preregistration_digest")


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
    if len(set(accepted)) != 1: raise PanelError("expected exact preregistration-only child")
    return accepted[0]


def validate_preregistration(repository: Path) -> tuple[dict[str, Any], pd.DataFrame, str]:
    if _git(repository, "status", "--porcelain", "--untracked-files=all"): raise PanelError("clean worktree required")
    path = repository / PREREGISTRATION_RELATIVE; raw = path.read_bytes(); prereg = base._read(path)
    if prereg.get("schema_version") != SCHEMA or not _valid(prereg, "preregistration_digest"):
        raise PanelError("panel preregistration differs")
    h1 = _sole_child(repository, raw, str(prereg["implementation_h0"]))
    if subprocess.run(["git", "merge-base", "--is-ancestor", h1, "HEAD"], cwd=repository).returncode:
        raise PanelError("HEAD does not descend from preregistration")
    for name, expected in prereg["runtime_files"].items():
        if _sha(repository / name) != expected: raise PanelError(f"runtime drifted: {name}")
    contract = _contract(repository); accounting = source_stage._accounting(repository)
    observed = {
        "contract_digest": contract["contract_digest"], "synthetic_receipt": _synthetic_receipt(repository),
        "source_accounting_sha256": _sha(repository / source_stage.ACCOUNTING_RELATIVE),
        "benchmark_sha256": _sha(Path(contract["source"]["benchmark_path"])), "symbol_count": len(accounting),
        "assignment_digest": stable_hash({str(r.symbol): source_stage._shard(str(r.symbol)) for r in accounting.itertuples(index=False)}),
    }
    if any(prereg.get(name) != value for name, value in observed.items()): raise PanelError("panel input drifted")
    return prereg, accounting, h1


def _write_shard(shard: int, records: list[dict[str, Any]], cache_text: str, benchmark_text: str,
                 cutoff: str, contract_digest: str) -> dict[str, Any]:
    cache = Path(cache_text); final_root = cache / f"shard-{shard:02d}"; seal_path = final_root / "SHARD_SEALED.json"
    if seal_path.exists():
        value = json.loads(seal_path.read_text())
        if not _valid(value, timing=True) or value.get("file_manifest") != _manifest(final_root, SHARD_FILES):
            raise PanelError(f"existing shard differs: {shard}")
        return value
    if final_root.exists(): raise PanelError(f"partial panel shard exists: {shard}")
    benchmark = pd.read_parquet(benchmark_text)
    benchmark = benchmark.loc[pd.to_datetime(benchmark.date) <= pd.Timestamp(cutoff)].reset_index(drop=True)
    started = perf_counter(); temporary = Path(tempfile.mkdtemp(prefix=f".shard-{shard:02d}.", dir=cache))
    writer: pq.ParquetWriter | None = None; accounting_rows = []; total_rows = up_events = expansion_events = 0
    try:
        for record in records:
            path = Path(record["source_path"])
            if _sha(path) != record["source_hash_at_lock"]: raise PanelError(f"source drifted: {record['symbol']}")
            frame = pd.read_parquet(path, columns=["date", "open", "high", "low", "close", "volume"],
                                    engine="pyarrow", use_threads=False)
            frame = frame.loc[pd.to_datetime(frame.date) <= pd.Timestamp(record["coverage_last_timestamp"])].rename(columns={"date": "timestamp"})
            if len(frame) != int(record["rows_through_lock"]): raise PanelError(f"locked rows differ: {record['symbol']}")
            panel = symbol_post_signal_panel(frame, benchmark, str(record["symbol"]))
            if not panel.empty:
                table = pa.Table.from_pandas(panel, preserve_index=False)
                if writer is None: writer = pq.ParquetWriter(temporary / "daily-panel.parquet", table.schema, compression="zstd")
                writer.write_table(table); total_rows += len(panel)
                up_events += int(panel.up_close_signal_event.sum())
                expansion_events += int(panel.bullish_range_expansion_signal_event.sum())
            accounting_rows.append({
                "symbol": str(record["symbol"]), "source_rows": len(frame), "panel_rows": len(panel),
                "invalid_source_rows": int((~valid_ohlcv_rows(frame)).sum()),
                "up_close_signal_events": int(panel.up_close_signal_event.sum()) if len(panel) else 0,
                "bullish_range_expansion_signal_events": int(panel.bullish_range_expansion_signal_event.sum()) if len(panel) else 0,
                **{f"complete_{horizon}_rows": int(panel[f"complete_{horizon}"].sum()) if len(panel) else 0 for horizon in (5, 20, 60)},
                "source_sha256": str(record["source_hash_at_lock"]),
            })
        if writer is None: raise PanelError(f"panel shard has no rows: {shard}")
        writer.close(); writer = None
        smoke._atomic_parquet(temporary / "symbol-accounting.parquet", pd.DataFrame(accounting_rows))
        seal = _seal({
            "schema_version": SCHEMA, "status": "shard_sealed", "passed": True, "shard": shard,
            "contract_digest": contract_digest, "symbol_count": len(records), "panel_rows": total_rows,
            "up_close_signal_events": up_events, "bullish_range_expansion_signal_events": expansion_events,
            "file_manifest": _manifest(temporary, SHARD_FILES), "elapsed_seconds": perf_counter() - started,
            "real_post_signal_outcomes_accessed": True,
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
        if not _valid(seal, timing=True): raise PanelError("existing panel output differs")
        return seal
    cache = repository / CACHE_RELATIVE; cache.mkdir(parents=True, exist_ok=True)
    grouped: dict[int, list[dict[str, Any]]] = {i: [] for i in range(SHARDS)}
    for row in accounting.itertuples(index=False):
        grouped[source_stage._shard(str(row.symbol))].append({
            "symbol": str(row.symbol), "source_path": str(row.source_path),
            "rows_through_lock": int(row.rows_through_lock), "coverage_last_timestamp": str(row.coverage_last_timestamp),
            "source_hash_at_lock": str(row.source_hash_at_lock),
        })
    contract = _contract(repository); started = perf_counter(); seals = []
    with ProcessPoolExecutor(max_workers=min(max(1, workers), SHARDS)) as pool:
        futures = {pool.submit(
            _write_shard, shard, records, str(cache), contract["source"]["benchmark_path"],
            contract["source"]["locked_coverage_end"], prereg["contract_digest"],
        ): shard for shard, records in grouped.items()}
        for future in as_completed(futures): seals.append(future.result())
    seals.sort(key=lambda item: item["shard"])
    if len(seals) != SHARDS or any(not _valid(item, timing=True) for item in seals):
        raise PanelError("panel shard completion differs")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    state = _seal({
        "schema_version": SCHEMA, "status": "sealed", "passed": True,
        "preregistration_h1": h1, "preregistration_digest": prereg["preregistration_digest"],
        "contract_digest": prereg["contract_digest"], "shards": SHARDS,
        "symbol_count": sum(item["symbol_count"] for item in seals),
        "panel_rows": sum(item["panel_rows"] for item in seals),
        "up_close_signal_events": sum(item["up_close_signal_events"] for item in seals),
        "bullish_range_expansion_signal_events": sum(item["bullish_range_expansion_signal_events"] for item in seals),
        "shard_result_digests": [item["result_digest"] for item in seals],
        "shard_seal_sha256": [_sha(cache / f"shard-{i:02d}" / "SHARD_SEALED.json") for i in range(SHARDS)],
        "elapsed_seconds": perf_counter() - started, "real_post_signal_outcomes_accessed": True,
        "control_matching_or_inference_accessed": False, "independent_verification_authorized": True,
        "production_promotion_authorized": False,
    }, timing=True)
    try:
        smoke._atomic_json(temporary / "SEALED.json", state); os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True); raise
    return state


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="command", required=True)
    for command in ("preregister", "run"):
        child = sub.add_parser(command); child.add_argument("--repository", type=Path, required=True)
        if command == "run": child.add_argument("--workers", type=int, default=SHARDS)
    args = parser.parse_args(argv)
    if args.command == "preregister":
        value = build_preregistration(args.repository); smoke._atomic_json(args.repository / PREREGISTRATION_RELATIVE, value)
    else: value = execute(args.repository, args.workers)
    print(json.dumps(value, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
