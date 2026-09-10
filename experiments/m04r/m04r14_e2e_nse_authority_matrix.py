"""Preregister and run a fresh current-code NSE exact-authority matrix.

The source is locked semantically through the registry's maximum query cutoff,
so later daily appends do not invalidate or expand this historical experiment.
No outcome module is imported here.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

import pandas as pd
import yaml

from market_analogues.adapters import PrefixLockedOHLCVSource, source_from_spec
from market_analogues.authority import (
    authority_universe_digest, load_authority_artifact, run_authority_case,
    validate_authority_artifact,
)
from market_analogues.config import load_config
from market_analogues.episodes import build_episode
from market_analogues.types import InstrumentKey, SearchQuery, stable_hash


SCHEMA_VERSION = "m04r14-e2e-nse-authority-matrix-v1"
PREREG_SCHEMA_VERSION = f"{SCHEMA_VERSION}-preregistration"
DEFAULT_PREREG = Path("experiments/m04r/m04r14_e2e_nse_authority_matrix_preregistered.json")
DEFAULT_OUTPUT = Path("config/data/analogues/portability/nse-current-authorities-v1")
THIS_RUNTIME = "experiments/m04r/m04r14_e2e_nse_authority_matrix.py"


class NseAuthorityMatrixError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _git(repository: Path, *args: str) -> str:
    return subprocess.check_output(
        ("git", *args), cwd=repository, text=True,
    ).strip()


def _repository_clean(repository: Path) -> bool:
    return not _git(repository, "status", "--porcelain")


def _runtime_files(repository: Path) -> tuple[str, ...]:
    package = _git(
        repository, "ls-files", "--", "src/market_analogues/*.py",
    ).splitlines()
    return tuple([THIS_RUNTIME, *package])


def _source_lock(repository: Path, config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    dataset = "nse"
    source = source_from_spec(config.datasets[dataset])
    registry_root = config.artifact_dir / "gate12" / dataset
    registry_yaml = registry_root / "query-registry.yaml"
    registry_parquet = registry_root / "query-registry.parquet"
    quality_path = config.artifact_dir / "quality" / f"{dataset}.parquet"
    original = yaml.safe_load(registry_yaml.read_text()) or {}
    registry = pd.read_parquet(registry_parquet).sort_values("case_id").reset_index(drop=True)
    if len(registry) != 12:
        raise NseAuthorityMatrixError(f"require 12 NSE cases; found {len(registry)}")
    maximum_cutoff = pd.Timestamp(registry["cutoff"].max())
    locked = PrefixLockedOHLCVSource(source, maximum_cutoff)
    quality = pd.read_parquet(quality_path)
    cases: list[dict[str, Any]] = []
    for row in registry.itertuples(index=False):
        query = build_episode(
            locked, InstrumentKey(dataset, str(row.symbol)), str(row.cutoff),
            int(row.lookback), str(row.representation_version),
        )
        if query.key.id != str(row.episode_id):
            raise NseAuthorityMatrixError(f"query identity changed: {row.case_id}")
        cases.append({
            "case_id": str(row.case_id), "dataset_id": dataset,
            "symbol": str(row.symbol), "cutoff": pd.Timestamp(row.cutoff).isoformat(),
            "cutoff_role": str(row.cutoff_role),
            "quality_tier": str(row.quality_tier),
            "liquidity_stratum": str(row.liquidity_stratum),
            "lookback": int(row.lookback),
            "representation_version": str(row.representation_version),
            "episode_id": query.key.id,
            "source_prefix_digest": locked.fingerprint(query.key.instrument),
        })
    request = SearchQuery(
        build_episode(
            locked, InstrumentKey(dataset, cases[0]["symbol"]), cases[0]["cutoff"],
            cases[0]["lookback"], cases[0]["representation_version"],
        ).key,
        (dataset,), ("A", "B"), 20,
        max_per_instrument=3, minimum_history_gap_bars=60,
    )
    universe_digest = authority_universe_digest(locked, request, quality)
    benchmark_digest = locked.benchmark_fingerprint()
    registry_state = {
        "schema_version": f"{SCHEMA_VERSION}-registry",
        "base_registry_digest": str(original["registry_digest"]),
        "dataset": dataset,
        "maximum_cutoff": maximum_cutoff.isoformat(),
        "benchmark_prefix_digest": benchmark_digest,
        "universe_prefix_digest": universe_digest,
        "cases": cases,
        "request": {
            "top_k": 20, "quality_tiers": ["A", "B"],
            "max_per_instrument": 3, "minimum_history_gap_bars": 60,
            "stride": 5,
        },
    }
    return {
        **registry_state,
        "registry_digest": stable_hash(registry_state),
        "input_hashes": {
            "base_registry_yaml": _sha(registry_yaml),
            "base_registry_parquet": _sha(registry_parquet),
            "quality": _sha(quality_path),
        },
    }


def preregister(
    repository: Path,
    config_path: Path,
    preregistration: Path,
    output_root: Path,
    *,
    workers: int,
) -> Path:
    repository = repository.resolve()
    config_path = config_path.resolve()
    if not _repository_clean(repository):
        raise NseAuthorityMatrixError("repository must be clean before preregistration")
    if preregistration.exists():
        raise FileExistsError(f"preregistration already exists: {preregistration}")
    if output_root.exists():
        raise FileExistsError(f"output root already exists: {output_root}")
    source_lock = _source_lock(repository, config_path)
    payload = {
        "schema_version": PREREG_SCHEMA_VERSION,
        "h0_commit": _git(repository, "rev-parse", "HEAD"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "config_sha256": _sha(config_path),
        "output_root": str(output_root.resolve()),
        "workers": workers,
        "disk_reserve_bytes": 5 * 1024 ** 3,
        "source_lock": source_lock,
        "runtime_hashes": {
            path: _sha(repository / path) for path in _runtime_files(repository)
        },
        "outcomes_accessed": False,
        "claim_boundary": "exact_retrieval_authorities_only",
    }
    state = dict(payload)
    payload["preregistration_digest"] = stable_hash(state)
    _atomic_json(preregistration, payload)
    return preregistration


def _validate_preregistration(repository: Path, path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != PREREG_SCHEMA_VERSION:
        raise NseAuthorityMatrixError("unsupported preregistration schema")
    claimed = payload.get("preregistration_digest")
    state = dict(payload)
    state.pop("preregistration_digest", None)
    if claimed != stable_hash(state):
        raise NseAuthorityMatrixError("preregistration digest mismatch")
    if not _repository_clean(repository):
        raise NseAuthorityMatrixError("repository must be clean for the frozen run")
    if _git(repository, "rev-parse", "HEAD^") != payload["h0_commit"]:
        raise NseAuthorityMatrixError("HEAD is not the sole preregistration child of H0")
    changed = _git(repository, "diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD").splitlines()
    expected = str(path.resolve().relative_to(repository))
    if changed != [expected]:
        raise NseAuthorityMatrixError("H1 must contain exactly the preregistration file")
    if _sha(Path(payload["config_path"])) != payload["config_sha256"]:
        raise NseAuthorityMatrixError("configuration changed after preregistration")
    for runtime, digest in payload["runtime_hashes"].items():
        if _sha(repository / runtime) != digest:
            raise NseAuthorityMatrixError(f"runtime changed after preregistration: {runtime}")
    current_lock = _source_lock(repository, Path(payload["config_path"]))
    if current_lock != payload["source_lock"]:
        raise NseAuthorityMatrixError("causal source lock changed after preregistration")
    if payload.get("outcomes_accessed") is not False:
        raise NseAuthorityMatrixError("outcome boundary is invalid")
    return payload


def _worker(
    config_path: str,
    output_root: str,
    source_lock: dict[str, Any],
    case: dict[str, Any],
) -> dict[str, Any]:
    config = load_config(config_path)
    locked = PrefixLockedOHLCVSource(
        source_from_spec(config.datasets["nse"]), source_lock["maximum_cutoff"],
    )
    quality = pd.read_parquet(config.artifact_dir / "quality" / "nse.parquet")
    query = build_episode(
        locked, InstrumentKey("nse", case["symbol"]), case["cutoff"],
        case["lookback"], case["representation_version"],
    )
    request = SearchQuery(
        query.key, ("nse",), ("A", "B"), 20,
        max_per_instrument=3, minimum_history_gap_bars=60,
    )
    root = Path(output_root)
    if shutil.disk_usage(root).free < 5 * 1024 ** 3:
        raise NseAuthorityMatrixError("free disk is below the frozen 5 GiB reserve")
    artifact, seeded, build, resume = run_authority_case(
        query, locked, request, quality, root / "frontiers",
        root / "cases" / f"{query.key.id}.json",
        registry_digest=source_lock["registry_digest"],
        seed_frontier_root=None, stride=5, batch_size=512,
        frontier_batch_rows=4096, representation_cache_shards=8,
        tolerance=1e-12, rebuild_invalid=False,
    )
    return {
        "case_id": case["case_id"], "episode_id": query.key.id,
        "authority_digest": artifact.authority_digest,
        "result_digest": artifact.result_digest,
        "eligible_candidates": artifact.eligible_candidates,
        "exact_evaluated": artifact.exact_evaluated,
        "safely_pruned": artifact.safely_pruned,
        "seeded_shards": seeded,
        "instruments_built": build.instruments_built,
        "instruments_reused": build.instruments_reused,
        "resume_instruments_reused": resume.instruments_reused,
    }


def run(repository: Path, preregistration: Path) -> Path:
    repository = repository.resolve()
    payload = _validate_preregistration(repository, preregistration)
    output_root = Path(payload["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output_root).free < int(payload["disk_reserve_bytes"]):
        raise NseAuthorityMatrixError("free disk is below the preregistered reserve")
    progress_path = output_root / "PROGRESS.json"
    existing: dict[str, Any] = {}
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        existing = {
            row["case_id"]: row for row in progress.get("completed", [])
        }
    cases = payload["source_lock"]["cases"]
    pending = [case for case in cases if case["case_id"] not in existing]
    progress = {
        "schema_version": f"{SCHEMA_VERSION}-progress",
        "preregistration_digest": payload["preregistration_digest"],
        "status": "running", "required": len(cases),
        "completed": [existing[key] for key in sorted(existing)],
        "failed": [], "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(progress_path, progress)
    workers = min(int(payload["workers"]), max(len(pending), 1))
    if pending:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _worker, payload["config_path"], str(output_root),
                    payload["source_lock"], case,
                ): case
                for case in pending
            }
            for future in as_completed(futures):
                case = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    progress["failed"].append({
                        "case_id": case["case_id"],
                        "error": f"{type(exc).__name__}: {exc}",
                    })
                else:
                    existing[result["case_id"]] = result
                progress["completed"] = [existing[key] for key in sorted(existing)]
                progress["updated_at"] = datetime.now(timezone.utc).isoformat()
                _atomic_json(progress_path, progress)
    if progress["failed"] or len(existing) != len(cases):
        progress["status"] = "failed"
        _atomic_json(progress_path, progress)
        raise NseAuthorityMatrixError(
            f"authority matrix incomplete: {len(existing)}/{len(cases)}"
        )
    progress["status"] = "complete"
    _atomic_json(progress_path, progress)
    result_state = {
        "schema_version": SCHEMA_VERSION,
        "preregistration_digest": payload["preregistration_digest"],
        "registry_digest": payload["source_lock"]["registry_digest"],
        "universe_prefix_digest": payload["source_lock"]["universe_prefix_digest"],
        "benchmark_prefix_digest": payload["source_lock"]["benchmark_prefix_digest"],
        "maximum_cutoff": payload["source_lock"]["maximum_cutoff"],
        "workers": int(payload["workers"]),
        "cases": [existing[key] for key in sorted(existing)],
        "outcomes_accessed": False,
        "production_authorized": False,
    }
    result = {**result_state, "result_digest": stable_hash(result_state)}
    _atomic_json(output_root / "RESULT.json", result)
    return output_root / "RESULT.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("preregister", "run"):
        command = sub.add_parser(name)
        command.add_argument("--repository", type=Path, default=Path.cwd())
        command.add_argument("--preregistration", type=Path, default=DEFAULT_PREREG)
    freeze = sub.choices["preregister"]
    freeze.add_argument("--config", type=Path, default=Path("config/datasets.example.yaml"))
    freeze.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    freeze.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    repository = args.repository.resolve()
    preregistration = (
        args.preregistration if args.preregistration.is_absolute()
        else repository / args.preregistration
    )
    if args.command == "preregister":
        config_path = args.config if args.config.is_absolute() else repository / args.config
        output_root = args.output_root if args.output_root.is_absolute() else repository / args.output_root
        print(preregister(
            repository, config_path, preregistration, output_root,
            workers=args.workers,
        ))
    else:
        print(run(repository, preregistration))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
