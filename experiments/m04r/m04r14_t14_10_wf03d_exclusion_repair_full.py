"""Preregister and run the restart-safe full WF-03D symbol-exclusion repair."""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict
from hashlib import sha256
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
from time import perf_counter
from typing import Any, Mapping, Sequence

import numba
import numpy as np

from market_analogues.adapters import CachedOHLCVSource, source_from_spec
from market_analogues.baseline_feature_store import load_feature_generation
from market_analogues.baseline_neighbors import (
    build_baseline_rank_index, deterministic_random_neighbors,
    indexed_recent_return_volatility_neighbors, recent_return_volatility,
)
from market_analogues.config import load_config
from market_analogues.dtw_component_search import certified_staged_dtw_component_search
from market_analogues.episodes import build_episode
from market_analogues.packed_bound_search import PackedBoundQuery, _eligible_mask
from market_analogues.packed_bound_store import load_packed_generation
from market_analogues.search import latest_eligible_cutoff
from market_analogues.subset_selection import certified_prefix_without_query_symbol
from market_analogues.types import InstrumentKey, SearchQuery, stable_hash
from market_analogues.certified_packed_search import certified_packed_search

from experiments.m04r import m04r14_t14_10_wf03_baseline_batch as baseline_batch
from experiments.m04r import m04r14_t14_10_wf03_baseline_poc as baseline_poc
from experiments.m04r import m04r14_t14_10_wf03_baseline_store_full as feature_store
from experiments.m04r import m04r14_t14_10_wf03_combined_batch as price_batch
from experiments.m04r import m04r14_t14_10_wf03_composite_topology_poc as composite_kernel
from experiments.m04r import m04r14_t14_10_wf03_feasibility as base
from experiments.m04r import m04r14_t14_10_wf03b_dtw_component_ladder as dtw_ladder
from experiments.m04r import m04r14_t14_10_wf03d_exclusion_audit as audit
from experiments.m04r import m04r14_t14_10_wf03d_exclusion_repair_poc as poc
from experiments.m04r import verify_m04r14_t14_10_wf03d_exclusion_repair_poc as poc_verifier


SCHEMA = "m04r14-t14-10-wf03d-exclusion-repair-full-preregistration-v2"
OUTPUT_RELATIVE = Path(
    "config/data/analogues/m04r14/t14-10-wf03d-exclusion-repair-full-v2"
)
PREREGISTRATION_RELATIVE = Path(
    "experiments/m04r/"
    "m04r14_t14_10_wf03d_exclusion_repair_full_v2_preregistered.json"
)
LEGACY_V1_OUTPUT_RELATIVE = Path(
    "config/data/analogues/m04r14/t14-10-wf03d-exclusion-repair-full-v1"
)
LEGACY_V1_PREREGISTRATION_RELATIVE = Path(
    "experiments/m04r/"
    "m04r14_t14_10_wf03d_exclusion_repair_full_v1_preregistered.json"
)
RECEIPT_SCHEMA = "m04r14-t14-10-wf03d-exclusion-repair-receipt-v2"
LEGACY_RECEIPT_SCHEMA = "m04r14-t14-10-wf03d-exclusion-repair-receipt-v1"
AUDIT_VERIFICATION = poc.AUDIT_VERIFICATION
POC_VERIFICATION = poc_verifier.OUTPUT_RELATIVE / "VERIFIED.json"
TOP_K = 20
SUPERSET_K = 21
COMPOSITE_PROCESSES = 12
COMPOSITE_THREADS = 1
PRICE_PROCESSES = 3
PRICE_THREADS = 4
INITIAL_FRONTIER = poc.INITIAL_FRONTIER
MAXIMUM_FRONTIER = poc.MAXIMUM_FRONTIER
SEED_ROWS = poc.SEED_ROWS
PRICE_SEED_ROWS = poc.PRICE_SEED_ROWS
BLOCK_ROWS = poc.BLOCK_ROWS
TOLERANCE = poc.TOLERANCE
METHODS = (
    "composite", "price_only", "deterministic_random",
    "recent_return_volatility",
)
EXPECTED_QUERIES = 3_936
EXPECTED_AFFECTED = {
    "composite": 78, "price_only": 94,
    "deterministic_random": 16, "recent_return_volatility": 24,
}
EXPECTED_AFFECTED_UNION = 173
RUNTIME_FILES = (
    "config/datasets.example.yaml",
    "experiments/m04r/m04r14_t14_10_wf03d_exclusion_repair_full.py",
    "experiments/m04r/m04r14_t14_10_wf03d_exclusion_repair_poc.py",
    "experiments/m04r/verify_m04r14_t14_10_wf03d_exclusion_repair_poc.py",
    "experiments/m04r/m04r14_t14_10_wf03d_exclusion_audit.py",
    "experiments/m04r/m04r14_t14_10_wf03_composite_topology_poc.py",
    "experiments/m04r/m04r14_t14_10_wf03_combined_batch.py",
    "experiments/m04r/m04r14_t14_10_wf03_baseline_batch.py",
    "experiments/m04r/m04r14_t14_10_wf03_baseline_poc.py",
    "experiments/m04r/m04r14_t14_10_wf03_baseline_store_full.py",
    "experiments/m04r/m04r14_t14_10_wf03_feasibility.py",
    "experiments/m04r/m04r14_t14_10_wf03b_dtw_component_ladder.py",
    "pyproject.toml",
) + tuple(
    str(path.relative_to(Path(__file__).resolve().parents[2]))
    for path in sorted(
        (Path(__file__).resolve().parents[2] / "src/market_analogues").glob("*.py")
    )
)

_PRICE_REPOSITORY: Path | None = None
_PRICE_PACKED_ROOT: Path | None = None
_PRICE_SOURCE: CachedOHLCVSource | None = None
_PRICE_PREPARED: dict[str, Any] | None = None


class ExclusionRepairFullError(RuntimeError):
    pass


def _git(repository: Path, *args: str, raw: bool = False) -> str | bytes:
    result = subprocess.run(
        ["git", *args], cwd=repository, capture_output=True,
        text=not raw, check=False,
    )
    if result.returncode:
        message = result.stderr if not raw else result.stderr.decode(errors="replace")
        raise ExclusionRepairFullError(message.strip() or "git command failed")
    return result.stdout if raw else result.stdout.strip()


def _sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _case_paths(repository: Path, query_id: str) -> dict[str, Path]:
    return {
        "composite": repository / audit.COMPOSITE_ROOT / "cases" / f"{query_id}.json",
        "price_only": repository / audit.PRICE_ROOT / "cases" / f"{query_id}.json",
        "baselines": repository / audit.BASELINE_ROOT / "cases" / f"{query_id}.json",
    }


def _source_manifest(repository: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    return stable_hash([
        {
            "query_id": row["episode_id"],
            "files": {
                name: _sha(path)
                for name, path in _case_paths(repository, row["episode_id"]).items()
            },
        }
        for row in rows
    ])


def _query_input_manifest(
    repository: Path, rows: Sequence[Mapping[str, Any]],
    affected: Mapping[str, Sequence[str]],
) -> str:
    selected = set().union(*(set(values) for values in affected.values()))
    source = CachedOHLCVSource(source_from_spec(
        load_config(repository / base.CONFIG_RELATIVE).datasets["nasdaq"]
    ), max_entries=None)
    values = []
    for row in rows:
        if row["episode_id"] not in selected:
            continue
        key = InstrumentKey("nasdaq", str(row["symbol"]))
        values.append({
            "query_id": row["episode_id"],
            "stock_prefix": asdict(source.causal_prefix_fingerprint(key, row["cutoff"])),
            "benchmark_prefix": asdict(
                source.benchmark_causal_prefix_fingerprint(row["cutoff"])
            ),
        })
    if len(values) != len(selected):
        raise ExclusionRepairFullError("affected query input inventory differs")
    return stable_hash(values)


def _affected(audit_value: Mapping[str, Any]) -> dict[str, list[str]]:
    result = {
        method: list(audit_value["methods"][method]["affected_query_ids"])
        for method in METHODS
    }
    if {method: len(values) for method, values in result.items()} != EXPECTED_AFFECTED \
            or any(values != sorted(values) or len(values) != len(set(values))
                   for values in result.values()) \
            or len(set().union(*(set(values) for values in result.values()))) \
                != EXPECTED_AFFECTED_UNION:
        raise ExclusionRepairFullError("affected query inventory differs")
    return result


def _legacy_v1_evidence(
    repository: Path, rows_by_id: Mapping[str, Mapping[str, Any]],
    affected: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    root = repository / LEGACY_V1_OUTPUT_RELATIVE
    contract_path = root / "CONTRACT.json"
    if not root.is_dir() or not contract_path.is_file() \
            or (root / "RESULT.json").exists() \
            or {path.name for path in root.iterdir()} \
                != {"CONTRACT.json", "PROGRESS.json", "repairs"}:
        raise ExclusionRepairFullError("legacy v1 partial evidence differs")
    contract = base._read(contract_path)
    base._validate_seal(contract, "preregistration_digest")
    tracked = base._read(repository / LEGACY_V1_PREREGISTRATION_RELATIVE)
    if contract != tracked:
        raise ExclusionRepairFullError("legacy v1 contract differs")
    affected_sets = {method: set(values) for method, values in affected.items()}
    repairs_root = root / "repairs"
    if not repairs_root.is_dir() \
            or {path.name for path in repairs_root.iterdir()} != set(METHODS):
        raise ExclusionRepairFullError("legacy v1 repairs layout differs")
    entries = []
    counts = {method: 0 for method in METHODS}
    for method in METHODS:
        method_root = root / "repairs" / method
        if not method_root.is_dir() or any(
            path.is_symlink() or not path.is_file() or path.suffix != ".json"
            for path in method_root.iterdir()
        ):
            raise ExclusionRepairFullError("legacy v1 method root differs")
        for path in sorted(method_root.glob("*.json")):
            query_id = path.stem
            if query_id not in rows_by_id or query_id not in affected_sets[method]:
                raise ExclusionRepairFullError("legacy v1 receipt inventory differs")
            value = _valid_receipt(
                repository, path, rows_by_id[query_id], method, contract,
                expected_schema=LEGACY_RECEIPT_SCHEMA,
            )
            if value is None:
                raise ExclusionRepairFullError("legacy v1 receipt disappeared")
            entries.append({
                "path": str(path.relative_to(root)), "query_id": query_id,
                "method": method, "sha256": _sha(path),
                "receipt_digest": value["receipt_digest"],
            })
            counts[method] += 1
    expected_counts = {
        "composite": 1, "price_only": 94,
        "deterministic_random": 16, "recent_return_volatility": 24,
    }
    if counts != expected_counts or len(entries) != 135:
        raise ExclusionRepairFullError("legacy v1 completion boundary differs")
    progress_path = root / "PROGRESS.json"
    progress = base._read(progress_path)
    if not all((
        progress.get("status") == "interrupted",
        progress.get("phase") == "composite",
        progress.get("completed_method_repairs") == 135,
        progress.get("total_method_repairs") == 212,
        progress.get("error_type") == "CompositeTopologyError",
        progress.get("error") == "invalid infinite round threshold",
    )):
        raise ExclusionRepairFullError("legacy v1 interruption evidence differs")
    return {
        "root": str(root.resolve()),
        "contract_preregistration_digest": contract["preregistration_digest"],
        "contract_sha256": _sha(contract_path),
        "receipt_count": len(entries), "method_counts": counts,
        "receipt_manifest_digest": stable_hash(entries),
        "progress_sha256": _sha(progress_path),
        "interruption": {
            key: progress[key] for key in (
                "status", "phase", "completed_method_repairs",
                "total_method_repairs", "error_type", "error",
            )
        },
    }


def build_preregistration(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    if _git(repository, "status", "--porcelain"):
        raise ExclusionRepairFullError("full repair preregistration requires clean commit")
    if (repository / OUTPUT_RELATIVE).exists() \
            or (repository / PREREGISTRATION_RELATIVE).exists():
        raise ExclusionRepairFullError("full repair paths must be absent before freeze")
    registry, _ = base._registry(repository)
    rows = registry.get("queries_data")
    if type(rows) is not list or len(rows) != EXPECTED_QUERIES:
        raise ExclusionRepairFullError("full repair registry differs")
    audit_value = base._read(repository / audit.OUTPUT_RELATIVE / "AUDIT.json")
    base._validate_seal(audit_value, "audit_digest")
    audit_verified = base._read(repository / AUDIT_VERIFICATION)
    base._validate_seal(audit_verified, "verification_digest")
    poc_verified = base._read(repository / POC_VERIFICATION)
    base._validate_seal(poc_verified, "verification_digest")
    if audit_verified.get("passed") is not True \
            or audit_verified.get("audit_digest") != audit_value["audit_digest"] \
            or poc_verified.get("passed") is not True \
            or poc_verified.get("full_exclusion_repair_authorized") is not True:
        raise ExclusionRepairFullError("verified repair authority differs")
    affected = _affected(audit_value)
    rows_by_id = {row["episode_id"]: row for row in rows}
    legacy_v1 = _legacy_v1_evidence(repository, rows_by_id, affected)
    resident = base._resident()
    feature_result = base._read(repository / poc.FEATURE_RESULT)
    base._validate_seal(feature_result)
    dtw_result = base._read(repository / poc.DTW_RESULT)
    base._validate_seal(dtw_result)
    state = {
        "schema_version": SCHEMA,
        "status": "frozen_before_full_outcome_blind_symbol_exclusion_repair",
        "implementation_commit": str(_git(repository, "rev-parse", "HEAD")),
        "runtime_files": {path: _sha(repository / path) for path in RUNTIME_FILES},
        "inputs": {
            "registry_digest": registry["registry_digest"],
            "audit_digest": audit_value["audit_digest"],
            "audit_verification_digest": audit_verified["verification_digest"],
            "audit_verification_sha256": _sha(repository / AUDIT_VERIFICATION),
            "poc_verification_digest": poc_verified["verification_digest"],
            "poc_verification_sha256": _sha(repository / POC_VERIFICATION),
            "source_case_manifest_digest": _source_manifest(repository, rows),
            "affected_query_input_manifest_digest": _query_input_manifest(
                repository, rows, affected,
            ),
            "resident_content_digest": resident["content_digest"],
            "packed_generation_id": base.GENERATION_ID,
            "dtw_generation_id": dtw_result["generation_id"],
            "dtw_physical_identity_digest": price_batch._dtw_physical_identity(repository)["digest"],
            "feature_generation_id": feature_result["generation_id"],
            "feature_result_digest": feature_result["result_digest"],
        },
        "affected_query_ids": affected,
        "legacy_v1_import": legacy_v1,
        "execution": {
            "top_k": TOP_K, "certified_superset_k": SUPERSET_K,
            "composite_processes": COMPOSITE_PROCESSES,
            "composite_threads_per_process": COMPOSITE_THREADS,
            "price_processes": PRICE_PROCESSES,
            "price_threads_per_process": PRICE_THREADS,
            "initial_frontier": INITIAL_FRONTIER,
            "maximum_frontier": MAXIMUM_FRONTIER,
            "seed_rows": SEED_ROWS, "price_seed_rows": PRICE_SEED_ROWS,
            "block_rows": BLOCK_ROWS, "tolerance_hex": TOLERANCE.hex(),
            "resume": "validate and reuse only sealed per-method repair receipts",
            "output_root": str((repository / OUTPUT_RELATIVE).resolve()),
        },
        "claims": {
            "outcomes_or_labels_used": False,
            "historical_walk_forward_query_outcomes_opened": False,
            "final_period_result_opened": False,
            "production_promotion_authorized": False,
        },
    }
    return base._sealed(state, "preregistration_digest")


def _sole_child(repository: Path, preregistration: Mapping[str, Any]) -> str:
    h0 = str(preregistration["implementation_commit"])
    raw = (repository / PREREGISTRATION_RELATIVE).read_bytes()
    accepted = []
    for line in str(_git(repository, "rev-list", "--all", "--children")).splitlines():
        values = line.split()
        if values and values[0] == h0:
            for child in values[1:]:
                changed = str(_git(
                    repository, "diff-tree", "--no-commit-id", "--name-only", "-r", child,
                )).splitlines()
                parents = str(_git(repository, "rev-list", "--parents", "-n", "1", child)).split()
                if parents == [child, h0] and changed == [PREREGISTRATION_RELATIVE.as_posix()] \
                        and _git(repository, "show", f"{child}:{PREREGISTRATION_RELATIVE}", raw=True) == raw:
                    accepted.append(child)
    if len(set(accepted)) != 1:
        raise ExclusionRepairFullError("full repair requires one preregistration-only child")
    return accepted[0]


def validate_preregistration(
    repository: Path, value: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    repository = repository.resolve(strict=True)
    if _git(repository, "status", "--porcelain"):
        raise ExclusionRepairFullError("full repair execution requires clean commit")
    base._validate_seal(value, "preregistration_digest")
    expected_execution = {
        "top_k": TOP_K, "certified_superset_k": SUPERSET_K,
        "composite_processes": COMPOSITE_PROCESSES,
        "composite_threads_per_process": COMPOSITE_THREADS,
        "price_processes": PRICE_PROCESSES,
        "price_threads_per_process": PRICE_THREADS,
        "initial_frontier": INITIAL_FRONTIER,
        "maximum_frontier": MAXIMUM_FRONTIER,
        "seed_rows": SEED_ROWS, "price_seed_rows": PRICE_SEED_ROWS,
        "block_rows": BLOCK_ROWS, "tolerance_hex": TOLERANCE.hex(),
        "resume": "validate and reuse only sealed per-method repair receipts",
        "output_root": str((repository / OUTPUT_RELATIVE).resolve()),
    }
    expected_claims = {
        "outcomes_or_labels_used": False,
        "historical_walk_forward_query_outcomes_opened": False,
        "final_period_result_opened": False,
        "production_promotion_authorized": False,
    }
    if value.get("schema_version") != SCHEMA \
            or value.get("status") \
                != "frozen_before_full_outcome_blind_symbol_exclusion_repair" \
            or value.get("execution") != expected_execution \
            or value.get("claims") != expected_claims:
        raise ExclusionRepairFullError("full repair frozen contract differs")
    child = _sole_child(repository, value)
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", child, "HEAD"], cwd=repository,
    ).returncode:
        raise ExclusionRepairFullError("full repair lineage differs")
    for path, digest in value["runtime_files"].items():
        blob = _git(repository, "show", f"{value['implementation_commit']}:{path}", raw=True)
        if sha256(blob).hexdigest() != digest or _sha(repository / path) != digest:
            raise ExclusionRepairFullError(f"full repair runtime differs: {path}")
    registry, _ = base._registry(repository)
    rows = registry.get("queries_data")
    if type(rows) is not list or len(rows) != EXPECTED_QUERIES \
            or value.get("schema_version") != SCHEMA \
            or value.get("inputs", {}).get("registry_digest") != registry["registry_digest"] \
            or value.get("inputs", {}).get("source_case_manifest_digest") \
                != _source_manifest(repository, rows):
        raise ExclusionRepairFullError("full repair frozen inputs differ")
    audit_value = base._read(repository / audit.OUTPUT_RELATIVE / "AUDIT.json")
    base._validate_seal(audit_value, "audit_digest")
    affected = _affected(audit_value)
    if value.get("affected_query_ids") != affected:
        raise ExclusionRepairFullError("full repair affected inventory differs")
    legacy_v1 = _legacy_v1_evidence(
        repository, {row["episode_id"]: row for row in rows}, affected,
    )
    if value.get("legacy_v1_import") != legacy_v1:
        raise ExclusionRepairFullError("full repair legacy import evidence differs")
    if value["inputs"].get("affected_query_input_manifest_digest") \
            != _query_input_manifest(repository, rows, affected):
        raise ExclusionRepairFullError("full repair query input prefixes differ")
    for path, digest_key, expected_key in (
        (AUDIT_VERIFICATION, "verification_digest", "audit_verification_digest"),
        (POC_VERIFICATION, "verification_digest", "poc_verification_digest"),
    ):
        receipt = base._read(repository / path)
        base._validate_seal(receipt, digest_key)
        if receipt.get("passed") is not True \
                or receipt[digest_key] != value["inputs"][expected_key] \
                or _sha(repository / path) != value["inputs"][expected_key.replace("digest", "sha256")]:
            raise ExclusionRepairFullError("full repair authority receipt differs")
    resident = base._resident()
    if resident["content_digest"] != value["inputs"]["resident_content_digest"] \
            or price_batch._dtw_physical_identity(repository)["digest"] \
                != value["inputs"]["dtw_physical_identity_digest"]:
        raise ExclusionRepairFullError("full repair substrate differs")
    feature_result = base._read(repository / poc.FEATURE_RESULT)
    base._validate_seal(feature_result)
    dtw_result = base._read(repository / poc.DTW_RESULT)
    base._validate_seal(dtw_result)
    if feature_result.get("generation_id") != value["inputs"]["feature_generation_id"] \
            or feature_result.get("result_digest") != value["inputs"]["feature_result_digest"] \
            or dtw_result.get("generation_id") != value["inputs"]["dtw_generation_id"]:
        raise ExclusionRepairFullError("full repair generation binding differs")
    return rows, affected


def _episode(source: Any, row: Mapping[str, Any]) -> Any:
    value = build_episode(
        source, InstrumentKey("nasdaq", str(row["symbol"])), str(row["cutoff"]),
        int(row["lookback"]), str(row["representation_version"]),
    )
    if value.key.id != row["episode_id"]:
        raise ExclusionRepairFullError("full repair query reconstruction differs")
    return value


def _request(episode: Any) -> SearchQuery:
    return SearchQuery(
        episode.key, ("nasdaq",), ("A", "B"), SUPERSET_K,
        False, True, base.MAX_PER_INSTRUMENT, base.MINIMUM_HISTORY_GAP,
    )


def _selected(rows: list[dict[str, Any]], symbol: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected, proof = certified_prefix_without_query_symbol(rows, symbol, top_k=TOP_K)
    if proof.excluded_rows != 1 or len(rows) != SUPERSET_K:
        raise ExclusionRepairFullError("full repair affected subset proof differs")
    return list(selected), asdict(proof)


def _source_prefix(repository: Path, row: Mapping[str, Any], method: str) -> list[dict[str, Any]]:
    paths = _case_paths(repository, row["episode_id"])
    if method == "composite":
        result = base._read(paths["composite"])["retrieval"]["matches"]
    elif method == "price_only":
        result = base._read(paths["price_only"])["matches"]
    else:
        value = base._read(paths["baselines"])
        result = value[
            "random_neighbors" if method == "deterministic_random"
            else "rank_neighbors"
        ]
    if type(result) is not list or len(result) != TOP_K:
        raise ExclusionRepairFullError("full repair source prefix differs")
    return result


def _repair_state(
    row: Mapping[str, Any], method: str, matches: list[dict[str, Any]],
    certificate: Mapping[str, Any] | None, elapsed: float,
) -> dict[str, Any]:
    repository = Path(row["_repository"])
    selected, proof = _selected(matches, str(row["symbol"]))
    if matches[:TOP_K] != _source_prefix(repository, row, method):
        raise ExclusionRepairFullError("full repair top-21 prefix differs")
    state = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "complete", "query_id": row["episode_id"],
        "case_id": row["case_id"], "symbol": row["symbol"],
        "cutoff": row["cutoff"], "method": method,
        "preregistration_digest": row["_preregistration_digest"],
        "source_case_sha256": {
            name: _sha(path) for name, path in _case_paths(
                repository, row["episode_id"]
            ).items()
        },
        "superset_matches": matches, "corrected_matches": selected,
        "subset_proof": proof, "certificate": dict(certificate) if certificate else None,
        "receipt_provenance": {"kind": "computed_v2"},
        "elapsed_seconds": elapsed,
        "outcomes_or_labels_used": False,
        "historical_walk_forward_query_outcomes_opened": False,
        "final_period_result_opened": False,
        "production_promotion_authorized": False,
    }
    return base._sealed(state, "receipt_digest")


def _composite_worker(
    repository_raw: str, packed_root_raw: str, row_raw: Mapping[str, Any],
    preregistration_digest: str,
) -> dict[str, Any]:
    started = perf_counter()
    repository = Path(repository_raw)
    numba.set_num_threads(COMPOSITE_THREADS)
    source = CachedOHLCVSource(source_from_spec(
        load_config(repository / base.CONFIG_RELATIVE).datasets["nasdaq"]
    ), max_entries=None)
    row = {**row_raw, "_repository": repository_raw,
           "_preregistration_digest": preregistration_digest}
    episode = _episode(source, row)
    result = certified_packed_search(
        episode, source, _request(episode), Path(packed_root_raw), base.GENERATION_ID,
        store_dataset_id="nasdaq", initial_frontier_rows=INITIAL_FRONTIER,
        maximum_frontier_rows=MAXIMUM_FRONTIER, seed_rows=SEED_ROWS,
        block_rows=BLOCK_ROWS, workers=COMPOSITE_THREADS, tolerance=TOLERANCE,
        verify_content=False, requested_positions=True, vector_lower_bounds=True,
        deferred_alignments=True, compact_scored=True,
        native_bound_deferral=True, streaming_threshold_closure=True,
        branch_aware_packed_bounds=True, proposal_threads=COMPOSITE_THREADS,
    )
    return _repair_state(
        row, "composite", [composite_kernel._match(value) for value in result.matches],
        composite_kernel._certificate_json_value(
            result.certificate, required_top_k=SUPERSET_K,
        ),
        perf_counter() - started,
    )


def _init_price_worker(repository_raw: str, packed_root_raw: str) -> None:
    global _PRICE_REPOSITORY, _PRICE_PACKED_ROOT, _PRICE_SOURCE, _PRICE_PREPARED
    _PRICE_REPOSITORY = Path(repository_raw)
    _PRICE_PACKED_ROOT = Path(packed_root_raw)
    numba.set_num_threads(PRICE_THREADS)
    _PRICE_SOURCE = CachedOHLCVSource(source_from_spec(
        load_config(_PRICE_REPOSITORY / base.CONFIG_RELATIVE).datasets["nasdaq"]
    ), max_entries=None)
    _PRICE_PREPARED = {}


def _price_worker(
    row_raw: Mapping[str, Any], preregistration_digest: str,
) -> dict[str, Any]:
    started = perf_counter()
    if _PRICE_REPOSITORY is None or _PRICE_PACKED_ROOT is None \
            or _PRICE_SOURCE is None or _PRICE_PREPARED is None:
        raise ExclusionRepairFullError("price repair worker is uninitialized")
    row = {**row_raw, "_repository": str(_PRICE_REPOSITORY),
           "_preregistration_digest": preregistration_digest}
    episode = _episode(_PRICE_SOURCE, row)
    result = certified_staged_dtw_component_search(
        episode, _PRICE_SOURCE, _request(episode), _PRICE_PACKED_ROOT,
        base.GENERATION_ID, _PRICE_REPOSITORY / dtw_ladder.DTW_ROOT_RELATIVE,
        dtw_ladder.DTW_GENERATION_ID, store_dataset_id="nasdaq",
        seed_rows=PRICE_SEED_ROWS, block_rows=BLOCK_ROWS,
        rigid_threads=PRICE_THREADS, dtw_threads=PRICE_THREADS,
        exact_workers=PRICE_THREADS, tolerance=TOLERANCE, verify_content=False,
        prepared_symbol_cache=_PRICE_PREPARED, adaptive_seed=True,
    )
    return _repair_state(
        row, "price_only", price_batch._matches(result), asdict(result.certificate),
        perf_counter() - started,
    )


def _receipt_path(root: Path, method: str, query_id: str) -> Path:
    return root / "repairs" / method / f"{query_id}.json"


def _valid_receipt(
    repository: Path, path: Path, row: Mapping[str, Any], method: str,
    preregistration: Mapping[str, Any], *, expected_schema: str = RECEIPT_SCHEMA,
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    value = base._read(path)
    base._validate_seal(value, "receipt_digest")
    selected, proof = _selected(value["superset_matches"], str(row["symbol"]))
    provenance = value.get("receipt_provenance")
    provenance_valid = expected_schema == LEGACY_RECEIPT_SCHEMA \
        and provenance is None
    if expected_schema == RECEIPT_SCHEMA and type(provenance) is dict:
        provenance_valid = (
            provenance.get("kind") == "computed_v2"
            and set(provenance) == {"kind"}
        ) or (
            provenance.get("kind") == "validated_v1_import"
            and set(provenance) == {
                "kind", "legacy_contract_preregistration_digest",
                "legacy_receipt_digest", "legacy_receipt_sha256",
            }
            and all(type(provenance[key]) is str for key in provenance)
        )
    if not all((
        value.get("schema_version") == expected_schema,
        value.get("status") == "complete",
        value.get("query_id") == row["episode_id"],
        value.get("case_id") == row["case_id"],
        value.get("symbol") == row["symbol"], value.get("cutoff") == row["cutoff"],
        value.get("method") == method,
        value.get("preregistration_digest") == preregistration["preregistration_digest"],
        value.get("corrected_matches") == selected,
        value.get("subset_proof") == proof,
        value["superset_matches"][:TOP_K] == _source_prefix(repository, row, method),
        value.get("source_case_sha256") == {
            name: _sha(source_path)
            for name, source_path in _case_paths(repository, row["episode_id"]).items()
        },
        value.get("outcomes_or_labels_used") is False,
        value.get("historical_walk_forward_query_outcomes_opened") is False,
        value.get("final_period_result_opened") is False,
        value.get("production_promotion_authorized") is False,
        (value.get("certificate") is None)
            == (method in {"deterministic_random", "recent_return_volatility"}),
        provenance_valid,
    )):
        raise ExclusionRepairFullError("existing full repair receipt differs")
    return value


def _import_legacy_v1(
    repository: Path, root: Path,
    rows_by_id: Mapping[str, dict[str, Any]],
    affected: Mapping[str, list[str]], preregistration: Mapping[str, Any],
) -> int:
    legacy_root = repository / LEGACY_V1_OUTPUT_RELATIVE
    legacy_contract = base._read(legacy_root / "CONTRACT.json")
    imported = 0
    for method in METHODS:
        for source_path in sorted((legacy_root / "repairs" / method).glob("*.json")):
            query_id = source_path.stem
            source = _valid_receipt(
                repository, source_path, rows_by_id[query_id], method,
                legacy_contract, expected_schema=LEGACY_RECEIPT_SCHEMA,
            )
            if source is None:
                raise ExclusionRepairFullError("legacy v1 receipt disappeared")
            target = _receipt_path(root, method, query_id)
            existing = _valid_receipt(
                repository, target, rows_by_id[query_id], method,
                preregistration,
            )
            if existing is not None:
                provenance = existing.get("receipt_provenance", {})
                if provenance.get("kind") != "validated_v1_import" \
                        or provenance.get("legacy_receipt_digest") \
                            != source["receipt_digest"]:
                    raise ExclusionRepairFullError("legacy import provenance differs")
                imported += 1
                continue
            state = {
                key: value for key, value in source.items()
                if key != "receipt_digest"
            }
            state.update({
                "schema_version": RECEIPT_SCHEMA,
                "preregistration_digest": preregistration["preregistration_digest"],
                "receipt_provenance": {
                    "kind": "validated_v1_import",
                    "legacy_contract_preregistration_digest": legacy_contract[
                        "preregistration_digest"
                    ],
                    "legacy_receipt_digest": source["receipt_digest"],
                    "legacy_receipt_sha256": _sha(source_path),
                },
            })
            _publish_receipt(root, base._sealed(state, "receipt_digest"))
            imported += 1
    if imported != preregistration["legacy_v1_import"]["receipt_count"]:
        raise ExclusionRepairFullError("legacy v1 imported receipt count differs")
    return imported


def _publish_receipt(root: Path, value: Mapping[str, Any]) -> None:
    path = _receipt_path(root, str(value["method"]), str(value["query_id"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ExclusionRepairFullError("repair receipt appeared concurrently")
    base._atomic(path, value)


def _progress(root: Path, state: Mapping[str, Any]) -> None:
    path = root / "PROGRESS.json"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w") as handle:
        json.dump(dict(state), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def _run_parallel(
    repository: Path, packed_root: Path, rows: list[dict[str, Any]],
    method: str, root: Path, preregistration: Mapping[str, Any], completed: int,
    total: int, started: float,
) -> int:
    pending = []
    for row in rows:
        path = _receipt_path(root, method, row["episode_id"])
        if _valid_receipt(repository, path, row, method, preregistration) is None:
            pending.append(row)
        else:
            completed += 1
    if not pending:
        return completed
    context = multiprocessing.get_context("spawn")
    if method == "composite":
        executor = ProcessPoolExecutor(
            max_workers=COMPOSITE_PROCESSES, mp_context=context,
        )
        submit = lambda pool, row: pool.submit(  # noqa: E731
            _composite_worker, str(repository), str(packed_root), row,
            preregistration["preregistration_digest"],
        )
    else:
        executor = ProcessPoolExecutor(
            max_workers=PRICE_PROCESSES, mp_context=context,
            initializer=_init_price_worker,
            initargs=(str(repository), str(packed_root)),
        )
        submit = lambda pool, row: pool.submit(  # noqa: E731
            _price_worker, row, preregistration["preregistration_digest"],
        )
    iterator = iter(pending)
    try:
        with executor as pool:
            active = {}
            for _ in range(min(
                COMPOSITE_PROCESSES if method == "composite" else PRICE_PROCESSES,
                len(pending),
            )):
                row = next(iterator)
                active[submit(pool, row)] = row
            while active:
                done, _ = wait(active, return_when=FIRST_COMPLETED)
                for future in done:
                    row = active.pop(future)
                    value = future.result()
                    _publish_receipt(root, value)
                    completed += 1
                    _progress(root, {
                        "status": "running", "phase": method,
                        "completed_method_repairs": completed,
                        "total_method_repairs": total,
                        "remaining_method_repairs": total - completed,
                        "last_query_id": row["episode_id"],
                        "last_receipt_digest": value["receipt_digest"],
                        "elapsed_seconds": perf_counter() - started,
                    })
                    print(
                        f"[exclusion-repair] method={method} completed={completed}/{total} "
                        f"query={row['episode_id']} seconds={value['elapsed_seconds']:.2f}",
                        flush=True,
                    )
                    try:
                        next_row = next(iterator)
                    except StopIteration:
                        continue
                    active[submit(pool, next_row)] = next_row
    except BaseException as exc:
        _progress(root, {
            "status": "interrupted", "phase": method,
            "completed_method_repairs": completed,
            "total_method_repairs": total,
            "error_type": type(exc).__name__, "error": str(exc),
            "elapsed_seconds": perf_counter() - started,
        })
        raise
    return completed


def _run_baselines(
    repository: Path, packed_root: Path, rows_by_id: Mapping[str, dict[str, Any]],
    affected: Mapping[str, list[str]], root: Path,
    preregistration: Mapping[str, Any], completed: int, total: int, started: float,
) -> int:
    methods = ("deterministic_random", "recent_return_volatility")
    pending = [(method, query_id) for method in methods for query_id in affected[method]
               if _valid_receipt(
                   repository, _receipt_path(root, method, query_id),
                   rows_by_id[query_id], method,
                   preregistration,
               ) is None]
    completed += sum(len(affected[method]) for method in methods) - len(pending)
    if not pending:
        return completed
    packed = load_packed_generation(
        packed_root, base.GENERATION_ID, verify_content=False,
        validate_records=False, expected_provenance_digest=base.PROVENANCE_DIGEST,
    )
    feature_result = base._read(repository / poc.FEATURE_RESULT)
    loaded = load_feature_generation(
        repository / feature_store.OUTPUT_RELATIVE / "store",
        feature_result["generation_id"], packed_manifest=packed.manifest,
        verify_content=True,
    )
    records = np.concatenate((
        baseline_poc._neighbor_records(packed.rows),
        baseline_poc._neighbor_records(packed.overflow),
    ))
    features = np.concatenate((loaded.rows, loaded.overflow))["values"]
    rank_index = build_baseline_rank_index(features, records["episode_id"])
    source = CachedOHLCVSource(source_from_spec(
        load_config(repository / base.CONFIG_RELATIVE).datasets["nasdaq"]
    ), max_entries=None)
    for method, query_id in pending:
        item_started = perf_counter()
        row = rows_by_id[query_id]
        episode = _episode(source, row)
        packed_query = PackedBoundQuery(
            episode.key.id, episode.key.instrument.source_symbol,
            int(episode.bars.timestamp.iloc[0].value),
            int(latest_eligible_cutoff(episode, base.MINIMUM_HISTORY_GAP).value),
            composite_kernel.represent(episode), ("A", "B"),
        )
        symbol_id = packed.symbols.index(row["symbol"]) \
            if row["symbol"] in packed.symbols else None
        eligible = _eligible_mask(records, packed_query, symbol_id)
        if method == "deterministic_random":
            matches = baseline_batch._neighbor_state(deterministic_random_neighbors(
                records["episode_id"], records["symbol_id"], eligible,
                packed.symbols, query_id, top_k=SUPERSET_K,
            ))
        else:
            query_features = recent_return_volatility(
                episode.bars["close"].to_numpy(dtype=np.float64)
            )
            matches = baseline_batch._neighbor_state(
                indexed_recent_return_volatility_neighbors(
                    rank_index, features, records["episode_id"], records["symbol_id"],
                    eligible, packed.symbols, query_features, query_id, top_k=SUPERSET_K,
                )
            )
        value = _repair_state(
            {**row, "_repository": str(repository),
             "_preregistration_digest": preregistration["preregistration_digest"]},
            method, matches, None, perf_counter() - item_started,
        )
        _publish_receipt(root, value)
        completed += 1
        _progress(root, {
            "status": "running", "phase": method,
            "completed_method_repairs": completed,
            "total_method_repairs": total,
            "remaining_method_repairs": total - completed,
            "last_query_id": query_id, "last_receipt_digest": value["receipt_digest"],
            "elapsed_seconds": perf_counter() - started,
        })
    return completed


def _manifest(
    repository: Path, rows: Sequence[Mapping[str, Any]],
    affected: Mapping[str, list[str]], root: Path,
    preregistration: Mapping[str, Any],
) -> dict[str, Any]:
    affected_sets = {method: set(values) for method, values in affected.items()}
    entries = []
    effective_digests = []
    for row in rows:
        query_id = row["episode_id"]
        paths = _case_paths(repository, query_id)
        source_sha = {name: _sha(path) for name, path in paths.items()}
        methods = []
        for method in METHODS:
            if query_id in affected_sets[method]:
                receipt_path = _receipt_path(root, method, query_id)
                receipt = _valid_receipt(
                    repository, receipt_path, row, method, preregistration,
                )
                if receipt is None:
                    raise ExclusionRepairFullError("full repair receipt is missing")
                matches = receipt["corrected_matches"]
                resolution = {
                    "kind": "top21_drop_query_symbol",
                    "repair_receipt": str(receipt_path.relative_to(root)),
                    "repair_receipt_sha256": _sha(receipt_path),
                    "repair_receipt_digest": receipt["receipt_digest"],
                    "subset_proof": receipt["subset_proof"],
                }
            else:
                matches = _source_prefix(repository, row, method)
                if any(value["symbol"] == row["symbol"] for value in matches):
                    raise ExclusionRepairFullError("unaffected source contains query symbol")
                resolution = {
                    "kind": "upstream_top20_unchanged_over_subset",
                    "subset_proof": {
                        "input_rows": TOP_K, "selected_rows": TOP_K,
                        "excluded_rows": 0, "query_symbol": row["symbol"],
                        "top_k": TOP_K,
                        "proof_kind": "exact_top_k_unchanged_over_subset",
                    },
                }
            if len(matches) != TOP_K \
                    or len({value["symbol"] for value in matches}) != TOP_K \
                    or len({value["episode_id"] for value in matches}) != TOP_K \
                    or any(value["symbol"] == row["symbol"] for value in matches):
                raise ExclusionRepairFullError("effective repaired matches differ")
            digest = stable_hash(matches)
            effective_digests.append({
                "query_id": query_id, "method": method, "matches_digest": digest,
            })
            methods.append({
                "method": method, "source_case_sha256": source_sha,
                "effective_matches_digest": digest, **resolution,
            })
        entries.append({
            "query_id": query_id, "case_id": row["case_id"],
            "symbol": row["symbol"], "cutoff": row["cutoff"], "methods": methods,
        })
    state = {
        "schema_version": "m04r14-t14-10-wf03d-exclusion-repair-manifest-v2",
        "status": "complete", "query_count": len(entries),
        "method_links": len(entries) * len(METHODS),
        "effective_neighbour_links": len(entries) * len(METHODS) * TOP_K,
        "affected_method_repairs": sum(len(value) for value in affected.values()),
        "affected_query_union": len(set().union(*(set(value) for value in affected.values()))),
        "preregistration_digest": preregistration["preregistration_digest"],
        "effective_inventory_digest": stable_hash(effective_digests),
        "queries": entries,
        "outcomes_or_labels_used": False,
        "historical_walk_forward_query_outcomes_opened": False,
        "final_period_result_opened": False,
    }
    return base._sealed(state, "manifest_digest")


def execute(repository: Path, preregistration: Mapping[str, Any]) -> dict[str, Any]:
    started = perf_counter()
    repository = repository.resolve(strict=True)
    rows, affected = validate_preregistration(repository, preregistration)
    root = repository / OUTPUT_RELATIVE
    terminal_result: dict[str, Any] | None = None
    if root.is_symlink() or root.exists() and not root.is_dir():
        raise ExclusionRepairFullError("full repair output path differs")
    if root.exists():
        if base._read(root / "CONTRACT.json") != preregistration:
            raise ExclusionRepairFullError("full repair resume contract differs")
        if (root / "RESULT.json").exists():
            terminal_result = base._read(root / "RESULT.json")
            base._validate_seal(terminal_result)
    else:
        root.mkdir(parents=True)
        base._atomic(root / "CONTRACT.json", preregistration)
    for method in METHODS:
        (root / "repairs" / method).mkdir(parents=True, exist_ok=True)
    rows_by_id = {row["episode_id"]: row for row in rows}
    total = sum(len(values) for values in affected.values())
    _import_legacy_v1(
        repository, root, rows_by_id, affected, preregistration,
    )
    if terminal_result is not None:
        manifest_path = root / "MANIFEST.json"
        manifest = base._read(manifest_path)
        base._validate_seal(manifest, "manifest_digest")
        reconstructed = _manifest(
            repository, rows, affected, root, preregistration,
        )
        if manifest != reconstructed or not all((
            terminal_result.get("schema_version")
                == "m04r14-t14-10-wf03d-exclusion-repair-full-result-v2",
            terminal_result.get("status") == "complete",
            terminal_result.get("passed") is True,
            terminal_result.get("preregistration_digest")
                == preregistration["preregistration_digest"],
            terminal_result.get("manifest_digest") == manifest["manifest_digest"],
            terminal_result.get("manifest_sha256") == _sha(manifest_path),
            terminal_result.get("query_count") == EXPECTED_QUERIES,
            terminal_result.get("affected_method_repairs") == total,
            terminal_result.get("affected_query_union") == EXPECTED_AFFECTED_UNION,
            terminal_result.get("repair_receipts") == total,
            terminal_result.get("legacy_v1_receipts_imported")
                == preregistration["legacy_v1_import"]["receipt_count"],
            terminal_result.get("outcomes_or_labels_used") is False,
            terminal_result.get("historical_walk_forward_query_outcomes_opened") is False,
            terminal_result.get("final_period_result_opened") is False,
            terminal_result.get("production_promotion_authorized") is False,
        )):
            raise ExclusionRepairFullError("full repair terminal publication differs")
        return terminal_result
    completed = 0
    packed_root = Path(base._resident()["store_root"])
    completed = _run_baselines(
        repository, packed_root, rows_by_id, affected, root,
        preregistration, completed, total, started,
    )
    for method in ("price_only", "composite"):
        method_rows = [rows_by_id[query_id] for query_id in affected[method]]
        completed = _run_parallel(
            repository, packed_root, method_rows, method, root,
            preregistration, completed, total, started,
        )
    if base._resident()["content_digest"] \
            != preregistration["inputs"]["resident_content_digest"] \
            or price_batch._dtw_physical_identity(repository)["digest"] \
                != preregistration["inputs"]["dtw_physical_identity_digest"]:
        raise ExclusionRepairFullError("full repair substrate changed during execution")
    if completed != total:
        raise ExclusionRepairFullError("full repair completion count differs")
    manifest = _manifest(repository, rows, affected, root, preregistration)
    manifest_path = root / "MANIFEST.json"
    if manifest_path.exists():
        if base._read(manifest_path) != manifest:
            raise ExclusionRepairFullError("full repair manifest resume differs")
    else:
        base._atomic(manifest_path, manifest)
    state = {
        "schema_version": "m04r14-t14-10-wf03d-exclusion-repair-full-result-v2",
        "status": "complete", "passed": True,
        "preregistration_digest": preregistration["preregistration_digest"],
        "manifest_digest": manifest["manifest_digest"],
        "manifest_sha256": _sha(manifest_path),
        "query_count": EXPECTED_QUERIES, "methods_per_query": len(METHODS),
        "effective_neighbour_links": EXPECTED_QUERIES * len(METHODS) * TOP_K,
        "affected_query_union": manifest["affected_query_union"],
        "affected_method_repairs": total,
        "repair_receipts": total,
        "legacy_v1_receipts_imported": preregistration[
            "legacy_v1_import"
        ]["receipt_count"],
        "fresh_v2_receipts_computed": total - preregistration[
            "legacy_v1_import"
        ]["receipt_count"],
        "all_effective_matches_exclude_query_symbol": True,
        "all_unaffected_top20_reused_unchanged": True,
        "all_affected_top21_prefixes_exact": True,
        "elapsed_seconds": perf_counter() - started,
        "outcomes_or_labels_used": False,
        "historical_walk_forward_query_outcomes_opened": False,
        "final_period_result_opened": False,
        "production_promotion_authorized": False,
        "independent_verification_authorized": True,
    }
    result = base._sealed(state)
    base._atomic(root / "RESULT.json", result)
    _progress(root, {
        "status": "complete", "completed_method_repairs": total,
        "total_method_repairs": total, "manifest_digest": manifest["manifest_digest"],
        "result_digest": result["result_digest"],
        "elapsed_seconds": result["elapsed_seconds"],
    })
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preregister", "run"))
    parser.add_argument("--repository", required=True, type=Path)
    args = parser.parse_args(argv)
    repository = args.repository.resolve(strict=True)
    path = repository / PREREGISTRATION_RELATIVE
    if args.mode == "preregister":
        base._atomic(path, build_preregistration(repository)); return 0
    result = execute(repository, base._read(path))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
