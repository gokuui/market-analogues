"""Independent verification of the fresh current-code NSE authority matrix.

The producer is deliberately not imported.  This verifier reconstructs the
frozen lifecycle, prefix locks, frontier merge, exact candidate scores,
constrained top-20 selection, and scalar-reference distances from persisted
artifacts and source data.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import heapq
from html import escape
import json
import math
import os
from pathlib import Path
import subprocess
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from market_analogues.adapters import PrefixLockedOHLCVSource, source_from_spec
from market_analogues.config import load_config
from market_analogues.distance import (
    complete_representation_distance, representation_distance_lower_bound,
)
from market_analogues.distance_v1_reference import reference_representation_distance
from market_analogues.episodes import build_episode
from market_analogues.exact_batch import exact_representations_at_positions
from market_analogues.exhaustive import load_frontier_shard
from market_analogues.representation import represent
from market_analogues.search import latest_eligible_cutoff
from market_analogues.types import EpisodeKey, InstrumentKey, SearchQuery, stable_hash


SCHEMA = "m04r14-e2e-nse-authority-matrix-verification-v1"
PRODUCER_SCHEMA = "m04r14-e2e-nse-authority-matrix-v1"
PREREG_SCHEMA = f"{PRODUCER_SCHEMA}-preregistration"
PREREGISTRATION = Path(
    "experiments/m04r/m04r14_e2e_nse_authority_matrix_preregistered.json"
)
PRODUCER = Path("experiments/m04r/m04r14_e2e_nse_authority_matrix.py")
OUTPUT = Path("config/data/analogues/portability/nse-current-authorities-v1")
VERIFICATION = Path(
    "config/data/analogues/portability/"
    "nse-current-authorities-v1-verification"
)
FRONTIER_ROWS = 4_096
TOP_K = 20
TOLERANCE = 1e-12


class NseAuthorityVerificationError(RuntimeError):
    pass


def _git(repository: Path, *args: str, raw: bool = False) -> str | bytes:
    completed = subprocess.run(
        ["git", *args], cwd=repository, capture_output=True,
        text=not raw, check=False,
    )
    if completed.returncode:
        raise NseAuthorityVerificationError(
            f"git validation failed: {' '.join(args)}"
        )
    return completed.stdout if raw else completed.stdout.strip()


def _read_json(path: Path) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        raise NseAuthorityVerificationError(f"regular JSON file required: {path}")
    raw = path.read_bytes()

    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise NseAuthorityVerificationError(
                    f"duplicate JSON key: {path}:{key}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            raw, object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                NseAuthorityVerificationError(
                    f"non-finite JSON token: {path}:{token}"
                )
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NseAuthorityVerificationError(f"invalid JSON: {path}") from exc
    if type(value) is not dict:
        raise NseAuthorityVerificationError(f"JSON object required: {path}")
    return value, raw


def _sha_bytes(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def _sha_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _semantic_digest(payload: Mapping[str, Any], *omitted: str) -> str:
    ignored = set(omitted)
    return stable_hash({key: value for key, value in payload.items() if key not in ignored})


def _validate_lifecycle(
    repository: Path, prereg: Mapping[str, Any], prereg_raw: bytes,
) -> str:
    h0 = str(prereg.get("h0_commit"))
    candidates: list[str] = []
    for line in str(_git(repository, "rev-list", "--all", "--children")).splitlines():
        values = line.split()
        if values and values[0] == h0:
            candidates.extend(values[1:])
    valid: list[str] = []
    for child in sorted(set(candidates)):
        parents = str(_git(
            repository, "rev-list", "--parents", "-n", "1", child,
        )).split()
        changed = str(_git(
            repository, "diff-tree", "--no-commit-id", "--name-only", "-r", child,
        )).splitlines()
        if parents == [child, h0] and changed == [PREREGISTRATION.as_posix()] \
                and _git(
                    repository, "show", f"{child}:{PREREGISTRATION.as_posix()}",
                    raw=True,
                ) == prereg_raw:
            valid.append(child)
    if len(valid) != 1:
        raise NseAuthorityVerificationError(
            "preregistration is not the unique sole-file child of H0"
        )
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", valid[0], "HEAD"],
        cwd=repository, capture_output=True,
    ).returncode:
        raise NseAuthorityVerificationError("HEAD does not descend from producer H1")
    for name, expected in dict(prereg.get("runtime_hashes") or {}).items():
        raw = _git(repository, "show", f"{h0}:{name}", raw=True)
        if _sha_bytes(raw) != expected:
            raise NseAuthorityVerificationError(
                f"H0 runtime hash differs: {name}"
            )
    producer_at_h0 = _git(repository, "show", f"{h0}:{PRODUCER.as_posix()}", raw=True)
    if _sha_bytes(producer_at_h0) != prereg["runtime_hashes"][PRODUCER.as_posix()]:
        raise NseAuthorityVerificationError("producer implementation binding differs")
    return valid[0]


def _validate_preregistration(repository: Path) -> tuple[dict[str, Any], str]:
    prereg, raw = _read_json(repository / PREREGISTRATION)
    if prereg.get("schema_version") != PREREG_SCHEMA:
        raise NseAuthorityVerificationError("preregistration schema differs")
    if prereg.get("preregistration_digest") != _semantic_digest(
        prereg, "preregistration_digest"
    ):
        raise NseAuthorityVerificationError("preregistration digest differs")
    if prereg.get("outcomes_accessed") is not False \
            or prereg.get("claim_boundary") != "exact_retrieval_authorities_only" \
            or int(prereg.get("workers", -1)) != 12:
        raise NseAuthorityVerificationError("preregistered boundary or controls differ")
    if Path(str(prereg.get("output_root"))).resolve() != (repository / OUTPUT).resolve():
        raise NseAuthorityVerificationError("preregistered output root differs")
    if _sha_file(Path(str(prereg.get("config_path")))) != prereg.get("config_sha256"):
        raise NseAuthorityVerificationError("configuration changed")
    h1 = _validate_lifecycle(repository, prereg, raw)
    return prereg, h1


def _quality_maps(frame: pd.DataFrame) -> tuple[dict[str, str], dict[str, tuple[str, ...]]]:
    tiers: dict[str, str] = {}
    issues: dict[str, tuple[str, ...]] = {}
    for row in frame.itertuples(index=False):
        symbol = str(row.symbol)
        tiers[symbol] = str(row.tier)
        raw = getattr(row, "issues", ())
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = [raw] if raw else []
        elif raw is None or (isinstance(raw, float) and math.isnan(raw)):
            parsed = []
        else:
            parsed = list(raw)
        issues[symbol] = tuple(str(value) for value in parsed)
    return tiers, issues


def _source_state(
    prereg: Mapping[str, Any],
) -> tuple[dict[str, str], str, str]:
    config = load_config(Path(str(prereg["config_path"])))
    lock = dict(prereg["source_lock"])
    source = PrefixLockedOHLCVSource(
        source_from_spec(config.datasets["nse"]), lock["maximum_cutoff"],
    )
    quality = pd.read_parquet(config.artifact_dir / "quality" / "nse.parquet")
    tiers, _ = _quality_maps(quality)
    fingerprints: dict[str, str] = {}
    universe: list[tuple[str, str]] = []
    for key in sorted(source.instruments()):
        digest = source.fingerprint(key)
        fingerprints[key.source_symbol] = digest
        tier = tiers.get(key.source_symbol, "A")
        if tier in {"A", "B"}:
            universe.append((str(key), digest))
    universe_digest = stable_hash(universe)
    benchmark_digest = source.benchmark_fingerprint()
    if benchmark_digest is None:
        raise NseAuthorityVerificationError("NSE benchmark is unavailable")
    if universe_digest != lock.get("universe_prefix_digest"):
        raise NseAuthorityVerificationError("current universe prefix differs")
    if benchmark_digest != lock.get("benchmark_prefix_digest"):
        raise NseAuthorityVerificationError("current benchmark prefix differs")
    for case in lock.get("cases") or []:
        if fingerprints.get(str(case["symbol"])) != case.get("source_prefix_digest"):
            raise NseAuthorityVerificationError(
                f"query source prefix differs: {case.get('case_id')}"
            )
    return fingerprints, universe_digest, benchmark_digest


def _frontier_prefix(
    frontier_root: Path,
    manifest: Mapping[str, Any],
    fingerprints: Mapping[str, str],
) -> tuple[list[tuple[float, str, int, int]], float, int, int]:
    records = manifest.get("shards")
    if type(records) is not list or not records:
        raise NseAuthorityVerificationError("frontier shard inventory is empty")
    expected_digest = sha256(json.dumps(
        records, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()
    if expected_digest != manifest.get("manifest_digest"):
        raise NseAuthorityVerificationError("frontier manifest digest differs")
    shards = []
    eligible = 0
    heap: list[tuple[float, str, int, int]] = []
    for index, record in enumerate(records):
        relative = Path(str(record.get("path")))
        if relative.is_absolute() or ".." in relative.parts:
            raise NseAuthorityVerificationError("unsafe frontier shard path")
        shard = load_frontier_shard(frontier_root / relative)
        metadata = asdict(shard.metadata)
        if metadata != {key: value for key, value in record.items() if key != "path"}:
            raise NseAuthorityVerificationError(
                f"frontier metadata differs: {relative}"
            )
        if fingerprints.get(shard.metadata.symbol) != shard.metadata.source_fingerprint:
            raise NseAuthorityVerificationError(
                f"frontier source prefix differs: {shard.metadata.symbol}"
            )
        shards.append(shard)
        eligible += len(shard.episode_ids)
        if len(shard.episode_ids):
            heapq.heappush(heap, (
                float(shard.lower_bounds[0]), str(shard.episode_ids[0]), index, 0,
            ))
    if eligible != int(manifest.get("eligible_rows", -1)):
        raise NseAuthorityVerificationError("frontier eligible accounting differs")
    prefix: list[tuple[float, str, int, int]] = []
    while heap and len(prefix) <= FRONTIER_ROWS:
        item = heapq.heappop(heap)
        prefix.append(item)
        shard = shards[item[2]]
        next_row = item[3] + 1
        if next_row < len(shard.episode_ids):
            heapq.heappush(heap, (
                float(shard.lower_bounds[next_row]),
                str(shard.episode_ids[next_row]), item[2], next_row,
            ))
    if len(prefix) != FRONTIER_ROWS + 1:
        raise NseAuthorityVerificationError("frontier cannot supply stopping witness")
    return prefix[:FRONTIER_ROWS], prefix[-1][0], eligible, len(shards)


def _select(records: list[dict[str, Any]], *, lookback: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    per_symbol: dict[str, int] = {}
    for row in sorted(records, key=lambda value: (value["total"], value["episode_id"])):
        symbol = str(row["symbol"])
        if per_symbol.get(symbol, 0) >= 3:
            continue
        if any(
            symbol == str(chosen["symbol"])
            and abs(int(row["position"]) - int(chosen["position"])) < lookback
            for chosen in selected
        ):
            continue
        selected.append(row)
        per_symbol[symbol] = per_symbol.get(symbol, 0) + 1
        if len(selected) == TOP_K:
            break
    return selected


def _match_delta(
    expected: Mapping[str, Any], actual: Mapping[str, Any],
) -> tuple[float, float, bool]:
    total = abs(float(expected["total_distance"]) - float(actual["total_distance"]))
    left = dict(expected["component_distances"])
    right = dict(actual["component_distances"])
    if set(left) != set(right):
        return total, math.inf, False
    component = max(abs(float(left[key]) - float(right[key])) for key in left)
    alignment = expected["alignment"] == actual["alignment"]
    return total, component, alignment


def _case_worker(
    config_path: str,
    output_root: str,
    verification_root: str,
    source_lock: dict[str, Any],
    fingerprints: dict[str, str],
    case: dict[str, Any],
) -> dict[str, Any]:
    started = perf_counter()
    config = load_config(Path(config_path))
    source = PrefixLockedOHLCVSource(
        source_from_spec(config.datasets["nse"]), source_lock["maximum_cutoff"],
    )
    quality = pd.read_parquet(config.artifact_dir / "quality" / "nse.parquet")
    tiers, issues = _quality_maps(quality)
    query = build_episode(
        source, InstrumentKey("nse", str(case["symbol"])), case["cutoff"],
        int(case["lookback"]), str(case["representation_version"]),
    )
    if query.key.id != case["episode_id"]:
        raise NseAuthorityVerificationError(f"query identity differs: {case['case_id']}")
    request = SearchQuery(
        query.key, ("nse",), ("A", "B"), TOP_K,
        max_per_instrument=3, minimum_history_gap_bars=60,
    )
    root = Path(output_root)
    authority_path = root / "cases" / f"{query.key.id}.json"
    authority, authority_raw = _read_json(authority_path)
    content = dict(authority)
    claimed_authority = content.pop("authority_digest", None)
    if claimed_authority != stable_hash(content):
        raise NseAuthorityVerificationError("authority content digest differs")
    if authority.get("result_digest") != stable_hash(authority.get("matches")) \
            or authority.get("result_digest") != authority.get("repeated_digest"):
        raise NseAuthorityVerificationError("authority repeated result differs")
    expected_provenance = {
        "query_episode_id": query.key.id,
        "request_digest": stable_hash(asdict(request)),
        "registry_digest": source_lock["registry_digest"],
        "source_fingerprint": fingerprints[str(case["symbol"])],
        "universe_source_digest": source_lock["universe_prefix_digest"],
        "benchmark_fingerprint": source_lock["benchmark_prefix_digest"],
    }
    if any(authority.get(key) != value for key, value in expected_provenance.items()):
        raise NseAuthorityVerificationError("authority provenance differs")

    manifest_path = root / "frontiers" / query.key.id / "manifest.json"
    manifest, manifest_raw = _read_json(manifest_path)
    prefix, next_bound, eligible, shard_count = _frontier_prefix(
        root / "frontiers", manifest, fingerprints,
    )
    if manifest.get("query_episode_id") != query.key.id \
            or manifest.get("query_source_fingerprint") != fingerprints[str(case["symbol"])] \
            or manifest.get("benchmark_fingerprint") != source_lock["benchmark_prefix_digest"] \
            or manifest.get("failures") != []:
        raise NseAuthorityVerificationError("frontier query provenance differs")
    if authority.get("frontier_manifest_digest") != manifest.get("manifest_digest"):
        raise NseAuthorityVerificationError("authority frontier binding differs")

    # Reload only the shards represented in the exact prefix, then materialize
    # only the requested candidate positions.  This differs from the producer's
    # sliding-all-windows reconstruction and avoids treating its result as truth.
    grouped: dict[int, list[tuple[float, str, int]]] = {}
    for lower, episode_id, shard_index, row in prefix:
        grouped.setdefault(shard_index, []).append((lower, episode_id, row))
    records = manifest["shards"]
    benchmark = source.load_benchmark()
    query_representation = represent(query)
    scored: list[dict[str, Any]] = []
    representations: dict[str, Any] = {}
    maximum_bound_delta = 0.0
    latest = latest_eligible_cutoff(query, 60)
    query_timestamps = set(query.bars.timestamp.astype(str))
    for shard_index in sorted(grouped):
        shard = load_frontier_shard(root / "frontiers" / records[shard_index]["path"])
        symbol = shard.metadata.symbol
        key = InstrumentKey("nse", symbol)
        bars = source.load(key)
        frame = bars[bars.timestamp <= latest].reset_index(drop=True)
        by_cutoff = {
            int(pd.Timestamp(value).value): index
            for index, value in enumerate(frame.timestamp)
        }
        ordered = grouped[shard_index]
        positions = np.asarray([
            by_cutoff[int(shard.cutoffs_ns[row])] for _, _, row in ordered
        ], dtype=int)
        if len(set(positions.tolist())) != len(positions):
            raise NseAuthorityVerificationError("duplicate exact candidate position")
        batch = exact_representations_at_positions(
            frame, benchmark, positions=positions,
            lookback=int(case["lookback"]),
        )
        for (lower, expected_id, row), position, candidate_representation in zip(
            ordered, positions, batch, strict=True,
        ):
            cutoff_ns = int(shard.cutoffs_ns[row])
            episode_key = EpisodeKey(
                key, pd.Timestamp(cutoff_ns), int(case["lookback"]),
                str(case["representation_version"]),
            )
            if episode_key.id != expected_id:
                raise NseAuthorityVerificationError("frontier episode identity differs")
            if pd.Timestamp(cutoff_ns) > latest or pd.Timestamp(cutoff_ns) >= query.key.cutoff:
                raise NseAuthorityVerificationError("temporally ineligible frontier episode")
            start = int(position) - int(case["lookback"]) + 1
            if key == query.key.instrument and query_timestamps.intersection(
                frame.iloc[start:int(position) + 1].timestamp.astype(str)
            ):
                raise NseAuthorityVerificationError("query-overlapping frontier episode")
            tier = tiers.get(symbol, "A")
            if tier not in {"A", "B"} or tier != shard.metadata.quality_tier:
                raise NseAuthorityVerificationError("frontier quality tier differs")
            calculated_lower, components, rigid = representation_distance_lower_bound(
                query_representation, candidate_representation,
            )
            maximum_bound_delta = max(
                maximum_bound_delta, abs(float(calculated_lower) - float(lower)),
            )
            if maximum_bound_delta > TOLERANCE:
                raise NseAuthorityVerificationError("frontier lower bound differs")
            total, _, _ = complete_representation_distance(
                query_representation, candidate_representation,
                calculated_lower, components, rigid, reconstruct_path=False,
            )
            row_state = {
                "episode_id": expected_id, "symbol": symbol,
                "position": int(position), "cutoff_ns": cutoff_ns,
                "total": float(total), "quality_tier": tier,
                "quality_issues": issues.get(symbol, ()),
            }
            scored.append(row_state)
            representations[expected_id] = candidate_representation

    selected = _select(scored, lookback=int(case["lookback"]))
    matches = list(authority.get("matches") or [])
    if len(selected) != TOP_K or len(matches) != TOP_K:
        raise NseAuthorityVerificationError("top-20 selection is incomplete")
    selected_ids = [row["episode_id"] for row in selected]
    stored_ids = [str(row.get("episode_id")) for row in matches]
    if selected_ids != stored_ids:
        raise NseAuthorityVerificationError("independent top-20 order differs")

    maximum_total_delta = 0.0
    maximum_component_delta = 0.0
    alignment_equal = True
    for selected_row, stored in zip(selected, matches, strict=True):
        reference = reference_representation_distance(
            query_representation, representations[selected_row["episode_id"]],
        )
        actual = {
            "total_distance": reference.total,
            "component_distances": reference.components,
            "alignment": [[left, right] for left, right in reference.alignment],
        }
        total_delta, component_delta, same_alignment = _match_delta(stored, actual)
        maximum_total_delta = max(maximum_total_delta, total_delta)
        maximum_component_delta = max(maximum_component_delta, component_delta)
        alignment_equal = alignment_equal and same_alignment
    certificate = dict(authority.get("certificate") or {})
    repeated = dict(authority.get("repeated_certificate") or {})
    certificate_fields = {
        "query_episode_id": query.key.id,
        "manifest_digest": manifest["manifest_digest"],
        "eligible_candidates": eligible,
        "exact_evaluated": FRONTIER_ROWS,
        "safely_pruned": eligible - FRONTIER_ROWS,
        "stopped_early": True,
    }
    for key, value in certificate_fields.items():
        if certificate.get(key) != value or repeated.get(key) != value:
            raise NseAuthorityVerificationError(f"certificate field differs: {key}")
    stop_threshold = max(float(row["total"]) for row in selected)
    gates = {
        "authority_integrity": True,
        "frontier_integrity": True,
        "candidate_accounting": FRONTIER_ROWS + eligible - FRONTIER_ROWS == eligible,
        "strict_stopping": (
            next_bound > stop_threshold + TOLERANCE
            and float(certificate.get("next_lower_bound")) == next_bound
            and abs(float(certificate.get("stop_threshold")) - stop_threshold) <= TOLERANCE
        ),
        "repeated_certificate_equal": all(
            certificate.get(key) == repeated.get(key)
            for key in certificate if key != "elapsed_seconds"
        ),
        "top20_identity_order_equal": selected_ids == stored_ids,
        "scalar_reference_total_equal": (
            maximum_total_delta <= TOLERANCE
            and maximum_component_delta <= TOLERANCE
            and alignment_equal
        ),
        "maximum_bound_delta_within_tolerance": maximum_bound_delta <= TOLERANCE,
        "outcomes_accessed": False,
    }
    if not all(value is True for key, value in gates.items() if key != "outcomes_accessed") \
            or gates["outcomes_accessed"] is not False:
        raise NseAuthorityVerificationError("one or more case gates failed")
    state = {
        "schema_version": f"{SCHEMA}-case",
        "case_id": case["case_id"], "query_episode_id": query.key.id,
        "authority_sha256": _sha_bytes(authority_raw),
        "authority_digest": claimed_authority,
        "manifest_sha256": _sha_bytes(manifest_raw),
        "manifest_digest": manifest["manifest_digest"],
        "eligible_candidates": eligible, "exact_evaluated": FRONTIER_ROWS,
        "safely_pruned": eligible - FRONTIER_ROWS, "frontier_shards": shard_count,
        "selected_matches": TOP_K, "next_lower_bound": next_bound,
        "stop_threshold": stop_threshold,
        "maximum_bound_delta": maximum_bound_delta,
        "maximum_scalar_reference_total_delta": maximum_total_delta,
        "maximum_scalar_reference_component_delta": maximum_component_delta,
        "gates": gates,
    }
    result = {**state, "result_digest": stable_hash(state),
              "elapsed_seconds": perf_counter() - started}
    target = Path(verification_root) / "cases" / f"{query.key.id}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(target)
    return result


def verify(
    repository: Path, *, workers: int = 12,
    output_root: Path | None = None, verification_root: Path | None = None,
) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    if workers < 1:
        raise NseAuthorityVerificationError("workers must be positive")
    output = (output_root or repository / OUTPUT).resolve(strict=True)
    verification = (verification_root or repository / VERIFICATION).resolve()
    if output != (repository / OUTPUT).resolve(strict=True):
        raise NseAuthorityVerificationError("authority output root differs")
    if verification.exists() or verification.is_symlink():
        raise NseAuthorityVerificationError("verification root already exists")
    prereg, h1 = _validate_preregistration(repository)
    producer_result, producer_raw = _read_json(output / "RESULT.json")
    progress, progress_raw = _read_json(output / "PROGRESS.json")
    producer_state = {
        key: value for key, value in producer_result.items() if key != "result_digest"
    }
    if producer_result.get("schema_version") != PRODUCER_SCHEMA \
            or producer_result.get("result_digest") != stable_hash(producer_state) \
            or producer_result.get("preregistration_digest") != prereg["preregistration_digest"] \
            or producer_result.get("outcomes_accessed") is not False \
            or producer_result.get("production_authorized") is not False:
        raise NseAuthorityVerificationError("producer aggregate differs")
    if progress.get("status") != "complete" or progress.get("failed") != [] \
            or int(progress.get("required", -1)) != 12 \
            or len(progress.get("completed") or []) != 12:
        raise NseAuthorityVerificationError("producer progress is incomplete")
    case_files = sorted((output / "cases").glob("*.json"))
    frontier_dirs = sorted(
        path for path in (output / "frontiers").iterdir() if path.is_dir()
    )
    cases = list(prereg["source_lock"]["cases"])
    expected_ids = {str(case["episode_id"]) for case in cases}
    if {path.stem for path in case_files} != expected_ids \
            or {path.name for path in frontier_dirs} != expected_ids \
            or len(case_files) != 12 or len(frontier_dirs) != 12:
        raise NseAuthorityVerificationError("authority tree case inventory differs")
    temporary_files = [path for path in output.rglob("*.tmp")]
    if temporary_files:
        raise NseAuthorityVerificationError("temporary authority files remain")

    fingerprints, universe_digest, benchmark_digest = _source_state(prereg)
    verification.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    completed: list[dict[str, Any]] = []
    failures: list[str] = []
    maximum_workers = min(workers, len(cases))
    with ProcessPoolExecutor(max_workers=maximum_workers) as executor:
        pending = {
            executor.submit(
                _case_worker, str(prereg["config_path"]), str(output),
                str(verification), dict(prereg["source_lock"]), fingerprints, case,
            ): case
            for case in cases
        }
        for future in as_completed(pending):
            case = pending[future]
            try:
                completed.append(future.result())
            except Exception as exc:
                failures.append(
                    f"{case['case_id']}: {type(exc).__name__}: {exc}"
                )
    completed.sort(key=lambda row: row["case_id"])
    if failures or len(completed) != 12:
        failure_state = {
            "schema_version": SCHEMA, "status": "failed",
            "completed_cases": len(completed), "failures": failures,
            "elapsed_seconds": perf_counter() - started,
        }
        (verification / "FAILED.json").write_text(
            json.dumps(failure_state, indent=2, sort_keys=True) + "\n"
        )
        raise NseAuthorityVerificationError(
            f"independent verification failed: {len(completed)}/12; {failures}"
        )
    case_semantic = [
        {key: value for key, value in row.items() if key != "elapsed_seconds"}
        for row in completed
    ]
    exact_total = sum(int(row["exact_evaluated"]) for row in completed)
    eligible_total = sum(int(row["eligible_candidates"]) for row in completed)
    state = {
        "schema_version": SCHEMA, "status": "verified", "passed": True,
        "producer_h0": prereg["h0_commit"], "producer_h1": h1,
        "verifier_git_head": str(_git(repository, "rev-parse", "HEAD")),
        "preregistration_digest": prereg["preregistration_digest"],
        "producer_result_digest": producer_result["result_digest"],
        "producer_result_sha256": _sha_bytes(producer_raw),
        "progress_sha256": _sha_bytes(progress_raw),
        "universe_prefix_digest": universe_digest,
        "benchmark_prefix_digest": benchmark_digest,
        "verified_cases": 12, "verified_matches": 240,
        "eligible_candidates": eligible_total, "exact_evaluated": exact_total,
        "safely_pruned": eligible_total - exact_total,
        "maximum_bound_delta": max(float(row["maximum_bound_delta"]) for row in completed),
        "maximum_scalar_reference_total_delta": max(
            float(row["maximum_scalar_reference_total_delta"]) for row in completed
        ),
        "maximum_scalar_reference_component_delta": max(
            float(row["maximum_scalar_reference_component_delta"])
            for row in completed
        ),
        "case_result_digests": {
            row["case_id"]: row["result_digest"] for row in completed
        },
        "cases_digest": stable_hash(case_semantic),
        "all_case_gates_passed": True,
        "outcomes_accessed": False,
        "production_authorized": False,
    }
    receipt = {
        **state, "result_digest": stable_hash(state),
        "elapsed_seconds": perf_counter() - started,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    target = verification / "VERIFIED.json"
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    rows = "".join(
        "<tr>" + "".join(f"<td>{escape(str(row[key]))}</td>" for key in (
            "case_id", "eligible_candidates", "exact_evaluated",
            "safely_pruned", "maximum_scalar_reference_total_delta",
        )) + "</tr>" for row in completed
    )
    (verification / "report.html").write_text(
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        "<title>NSE authority verification</title><style>body{font-family:system-ui;"
        "max-width:1200px;margin:2rem auto;padding:0 1rem}table{border-collapse:"
        "collapse;width:100%}th,td{padding:.55rem;border-bottom:1px solid #ddd;"
        "text-align:left}.pass{color:#176b37}</style></head><body>"
        "<h1>Fresh NSE exact-authority verification</h1><p class=\"pass\"><b>"
        "PASS — 12/12</b></p><p>Every 4,096-row exact frontier was independently "
        "reconstructed and constrained; all top-20 identities and scalar-reference "
        "distances agree. No outcome was accessed.</p><table><thead><tr><th>Case</th>"
        "<th>Eligible</th><th>Exact</th><th>Pruned</th><th>Reference delta</th>"
        f"</tr></thead><tbody>{rows}</tbody></table><p>Result digest: "
        f"<code>{receipt['result_digest']}</code></p><p><b>Boundary:</b> This "
        "certifies descriptive exact retrieval only; it does not authorize a "
        "predictive or trading claim.</p></body></html>"
    )
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--verification-root", type=Path)
    args = parser.parse_args(argv)
    result = verify(
        args.repository, workers=args.workers,
        output_root=args.output_root, verification_root=args.verification_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
