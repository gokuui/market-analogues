"""Build the frozen 36-symbol/72-case M04R-14 untouched NASDAQ registry."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from html import escape
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Mapping, Sequence

import pandas as pd

from market_analogues.adapters import file_fingerprint, source_from_spec
from market_analogues.config import load_config
from market_analogues.m04r_validation_registry import (
    _canonical_records, _symbols, build_contamination_ledger,
    build_m04r_validation_registry, default_search_contract,
)
from market_analogues.types import stable_hash


SCHEMA = "m04r14-untouched-authority-registry-v1"
LEDGER_SCHEMA = "m04r14-contamination-ledger-v1"
CONTRACT = Path("config/m04r14-untouched-registry-contract.json")
DEFAULT_OUTPUT = Path("config/data/analogues/m04r14/nasdaq-untouched-registry-v1")


class RegistryError(RuntimeError):
    pass


def _read(path: Path) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        raise RegistryError(f"regular file required: {path}")
    raw = path.read_bytes()
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise RegistryError(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    value = json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda item: (_ for _ in ()).throw(
            RegistryError(f"non-finite JSON: {item}")))
    if type(value) is not dict:
        raise RegistryError(f"JSON object required: {path}")
    return value, raw


def _source(name: str, path: Path, symbols: Sequence[str]) -> dict[str, Any]:
    ordered = _symbols(symbols)
    return {"name": name, "path": str(path.resolve()),
        "file_sha256": file_fingerprint(path), "symbol_count": len(ordered),
        "symbol_digest": stable_hash(ordered), "symbols": ordered}


def expanded_ledger(artifact_dir: Path, design_path: Path, previous_path: Path) -> dict[str, Any]:
    base = build_contamination_ledger(artifact_dir, design_path)
    previous, _ = _read(previous_path)
    rows = previous.get("cases_data")
    if type(rows) is not list or len(rows) != 60:
        raise RegistryError("previous registry differs")
    sources = [*base["sources"], _source(
        "m04r10-selected-and-m04r14-exposed", previous_path,
        [str(row["symbol"]) for row in rows],
    )]
    excluded = _symbols(symbol for source in sources for symbol in source["symbols"])
    ledger: dict[str, Any] = {
        "schema_version": LEDGER_SCHEMA,
        "normalization": "strip then uppercase; exact source symbol plus explicit known aliases",
        "sources": sources, "excluded_symbols": excluded,
        "excluded_symbol_count": len(excluded),
        "excluded_symbol_digest": stable_hash(excluded),
        "known_identity_alias_groups": base.get("known_identity_alias_groups", []),
        "alias_limit": base.get("alias_limit"),
        "previous_ledger_digest": base["ledger_digest"],
        "previous_registry_digest": previous.get("registry_digest"),
        "previous_registry_cases": len(rows),
        "previous_registry_symbols": len({str(row["symbol"]).upper() for row in rows}),
    }
    ledger["ledger_digest"] = stable_hash(ledger)
    return ledger


def _atomic_bytes(path: Path, raw: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw); handle.flush(); os.fsync(handle.fileno())


def _publish(result: Any, output: Path, source_lock: Mapping[str, Any]) -> dict[str, Any]:
    if output.exists() or output.is_symlink():
        raise RegistryError("registry output root exists")
    parent = output.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=parent))
    result.universe.to_parquet(temporary / "selection-universe.parquet", index=False)
    result.transcript.to_parquet(temporary / "selection-transcript.parquet", index=False)
    result.cases.to_parquet(temporary / "query-registry.parquet", index=False)
    payload = {**result.metrics, "passed": result.passed,
        "failures": list(result.failures), "source_lock": dict(source_lock),
        "contamination_ledger": result.contamination_ledger,
        "cases_data": _canonical_records(result.cases)}
    payload["registry_digest"] = stable_hash({key: value for key, value in payload.items()
        if key not in {"passed", "failures", "registry_digest"}})
    _atomic_bytes(temporary / "query-registry.json",
        json.dumps(payload, indent=2, sort_keys=True).encode() + b"\n")
    status = "PASS" if result.passed else "FAIL"
    rows = "".join(f"<tr><td>{escape(str(row.case_id))}</td><td>{escape(str(row.quality_tier))}/"
        f"{escape(str(row.liquidity_stratum))}</td><td>{escape(str(row.selection_target_regime))}/"
        f"{escape(str(row.selection_target_era))}</td><td>{escape(str(row.cutoff_role))}</td>"
        f"<td>{escape(str(row.cutoff))}</td></tr>" for row in result.cases.itertuples(index=False))
    report = ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<title>M04R-14 untouched registry</title></head><body>"
        f"<h1>M04R-14 untouched registry: {status}</h1><p>36 symbols, 72 paired cases; "
        "six quality/liquidity cells and all regime/era slots. Outcomes and candidate results "
        f"were not accessed.</p><table><tbody>{rows}</tbody></table></body></html>")
    _atomic_bytes(temporary / "query-registry.html", report.encode())
    manifest = [{"path": path.name, "bytes": path.stat().st_size,
                 "sha256": file_fingerprint(path)}
                for path in sorted(temporary.iterdir())]
    seal = {"schema_version": "m04r14-untouched-registry-seal-v1",
        "status": "sealed", "passed": result.passed,
        "registry_digest": payload["registry_digest"], "files": manifest,
        "manifest_digest": stable_hash(manifest),
        "production_promotion_authorized": False}
    seal["seal_digest"] = stable_hash(seal)
    _atomic_bytes(temporary / "SEALED.json",
        json.dumps({**seal, "created_at": datetime.now(timezone.utc).isoformat()},
                   indent=2, sort_keys=True).encode() + b"\n")
    os.replace(temporary, output)
    return {**seal, "output_root": str(output)}


def execute(repository: Path, config_path: Path, output: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    if subprocess.run(["git", "status", "--porcelain"], cwd=repository,
            text=True, capture_output=True, check=True).stdout:
        raise RegistryError("untouched registry requires clean Git")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repository,
        text=True, capture_output=True, check=True).stdout.strip()
    contract, contract_raw = _read(repository / CONTRACT)
    config = load_config(config_path)
    source = source_from_spec(config.datasets["nasdaq"])
    quality_path = config.artifact_dir / "quality/nasdaq.parquet"
    liquidity_path = config.artifact_dir / "oracles/nasdaq/liquidity-strata.parquet"
    design_path = repository / contract["contamination_sources"]["base_design"]
    previous_path = repository / contract["contamination_sources"]["previous_registry"]
    ledger = expanded_ledger(config.artifact_dir, design_path, previous_path)
    schedule = tuple(tuple(value) for value in contract["target_schedule"])
    source_lock = {"git_head": head, "contract_sha256": sha256(contract_raw).hexdigest(),
        "config_path": str(config_path.resolve()), "config_sha256": file_fingerprint(config_path),
        "quality_path": str(quality_path.resolve()), "quality_sha256": file_fingerprint(quality_path),
        "liquidity_path": str(liquidity_path.resolve()), "liquidity_sha256": file_fingerprint(liquidity_path),
        "design_exclusions_sha256": file_fingerprint(design_path),
        "previous_registry_sha256": file_fingerprint(previous_path),
        "real_forward_outcomes_accessed": False}
    source_lock["source_lock_digest"] = stable_hash(source_lock)
    result = build_m04r_validation_registry(source, pd.read_parquet(quality_path),
        pd.read_parquet(liquidity_path), ledger, default_search_contract(config.artifact_dir),
        seed=contract["seed"], lookback=int(contract["lookback"]),
        minimum_rows=int(contract["minimum_rows"]),
        minimum_future_sessions=int(contract["minimum_future_sessions"]),
        maximum_staleness_days=int(contract["maximum_staleness_days"]),
        regime_threshold=float(contract["regime_threshold"]),
        representation_version=config.representation_version,
        target_schedule=schedule, symbols_per_cell=int(contract["symbols_per_cell"]),
        schema_version=SCHEMA)
    if not result.passed:
        raise RegistryError(f"registry selection failed: {result.failures}")
    return _publish(result, output.resolve(), source_lock)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args(argv)
    output = args.output_root or args.repository / DEFAULT_OUTPUT
    state = execute(args.repository, args.config.resolve(), output)
    print(json.dumps(state, indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
