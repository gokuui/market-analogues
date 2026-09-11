"""Run the preregistered R1-B B2 structural-localization experiment.

This producer is deliberately outcome blind.  It reads only the joint H1
contract, its frozen population, and the independently materialized B2 geometry
bundle.  Replicate shards are immutable restart checkpoints; a shard is reused
only after every binding and byte digest has been revalidated.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict, dataclass
import fcntl
from hashlib import sha256
import heapq
import io
import json
import math
import multiprocessing
import os
from pathlib import Path
import re
import shutil
import stat
import sys
from time import perf_counter
from typing import Any, Hashable, Mapping, Sequence
from uuid import uuid4

import numpy as np

from market_analogues.adequacy_localization import (
    DIMENSIONS,
    PRIMARY_EPISODES,
    PRIORITY_DOMAIN,
    PRIORITY_FAMILY,
    REPLICATES,
    LocalizationError,
    PreparedLocalization,
    assemble_replicate_chunks,
    candidate_specificity_ranks,
    conditional_priority,
    deterministic_null_summary,
    episode_parities,
    effect_summary,
    localization_decision,
)
from market_analogues.balanced_partition import array_digest, id_digest
from market_analogues.types import stable_hash

from experiments.m04r import m04r14_r1b_joint_b005_b2_contract as joint


SCHEMA = "m04r14-r1b-b2-localization-result-v1"
SHARD_SCHEMA = "m04r14-r1b-b2-localization-shard-v1"
OUTPUT = Path(joint.OUTPUTS["b2"])
GEOMETRY = Path(joint.OUTPUTS["geometry"])
ARRAYS = (
    "episode_n0", "episode_n1", "episode_k12", "episode_k16",
    "shared_n0", "shared_n1", "episode_n0_breadth",
)
NULL_TABLES = ARRAYS[:6]
SHARD_REPLICATES = 32
WORKERS = 12
QUERIES = 3270
COHORT_EPISODES = 369
COHORT_LINKS = 2865
PRIMARY_LINKS = 2791
FORBIDDEN = tuple(joint.FORBIDDEN)


class B2RunnerError(RuntimeError):
    """The frozen B2 execution contract was not satisfied."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise B2RunnerError(message)


def _exact_keys(value: object, keys: Sequence[str], name: str) -> Mapping[str, Any]:
    require(isinstance(value, dict) and set(value) == set(keys), f"{name} field closure differs")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    return _decode_json(_snapshot_bytes(path), path)


def _decode_json(content: bytes, path: Path) -> dict[str, Any]:

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            require(key not in result, f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(content, object_pairs_hook=pairs,
                           parse_constant=lambda item: require(False, f"nonfinite JSON: {item}"))
    except (OSError, ValueError) as error:
        raise B2RunnerError(f"unreadable JSON: {path}") from error
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def _snapshot_bytes(path: Path) -> bytes:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as error:
        raise B2RunnerError(f"bound file is unsafe: {path}") from error
    try:
        metadata = os.fstat(descriptor)
        require(stat.S_ISREG(metadata.st_mode), f"bound file is not regular: {path}")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            content = handle.read()
    finally:
        os.close(descriptor)
    return content


def _bound_bytes(path: Path, expected_sha256: object, expected_bytes: object | None = None) -> bytes:
    require(isinstance(expected_sha256, str) and re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is not None,
            f"bound SHA256 differs: {path}")
    content = _snapshot_bytes(path)
    require(expected_bytes is None or type(expected_bytes) is int and len(content) == expected_bytes,
            f"bound file length differs: {path}")
    require(sha256(content).hexdigest() == expected_sha256, f"bound file hash differs: {path}")
    return content


def _read_bound_json(path: Path, expected_sha256: object, expected_bytes: object | None = None) -> dict[str, Any]:
    return _decode_json(_bound_bytes(path, expected_sha256, expected_bytes), path)


def _file_sha(path: Path) -> str:
    require(path.is_file() and not path.is_symlink(), f"regular file required: {path}")
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_bytes(path: Path, content: bytes) -> None:
    """Durable create-only publication, including dangling-symlink refusal."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}-{uuid4().hex}"
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        try:
            with os.fdopen(descriptor, "wb", closefd=True) as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            try:
                os.close(descriptor)
            except OSError:
                pass
            raise
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise B2RunnerError(f"create-only publication exists: {path}") from error
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_bytes(path, _json_bytes(value))


def _npy_bytes(array: np.ndarray) -> bytes:
    import io
    stream = io.BytesIO()
    np.lib.format.write_array(stream, np.ascontiguousarray(array), allow_pickle=False)
    return stream.getvalue()


def _canonical_array(array: object, *, dtype: np.dtype[Any], shape: tuple[int, ...], name: str,
                     nonnegative: bool = False) -> np.ndarray:
    value = np.asarray(array)
    require(value.shape == shape and value.dtype == dtype, f"{name} shape/dtype differs")
    if value.dtype.kind == "f":
        require(np.isfinite(value).all(), f"{name} is nonfinite")
        if nonnegative:
            require(not (value < 0).any(), f"{name} is negative")
        value = value.copy()
        value[value == 0] = 0.0
    return np.ascontiguousarray(value)


def _array_digest(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    state = {"shape": list(value.shape), "dtype": value.dtype.str,
             "sha256": sha256(value.tobytes(order="C")).hexdigest()}
    return stable_hash(state)


def _geometry_array_digest(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = sha256()
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("utf-8"))
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class DesignGroups:
    groups: tuple[tuple[int, tuple[int, ...]], ...]


@dataclass(frozen=True)
class EpisodePlan:
    episode_id: str
    candidate_index: int
    observed: tuple[int, ...]
    active: tuple[int, ...]
    designs: tuple[DesignGroups, DesignGroups, DesignGroups, DesignGroups]


@dataclass(frozen=True)
class Bindings:
    h1_commit: str
    preregistration_digest: str
    priority_contract_digest: str
    population_digest: str
    runtime_sha256: str
    geometry_digest: str
    query_ids_digest: str
    candidate_ids_digest: str
    primary_ids_digest: str


@dataclass
class RunState:
    query_ids: tuple[str, ...]
    cohort_ids: tuple[str, ...]
    primary_ids: tuple[str, ...]
    plans: tuple[EpisodePlan, ...]
    localization: PreparedLocalization
    bindings: Bindings
    output: Path
    query_priority_suffixes: tuple[bytes, ...] = ()


def _safe_relative(root: Path, relative: object) -> Path:
    require(isinstance(relative, str), "geometry array path must be text")
    path = Path(relative)
    require(not path.is_absolute() and ".." not in path.parts and len(path.parts) == 1,
            f"unsafe geometry array path: {relative}")
    require(not any(token in relative.lower() for token in FORBIDDEN), f"embargoed geometry path: {relative}")
    target = root / path
    require(target.parent.resolve() == root.resolve() and not target.is_symlink(), "geometry path escapes or is symlink")
    return target


def _frozen_eligibility(
    query_ids: Sequence[str], candidate_ids: Sequence[str], episodes: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    require(len(set(query_ids)) == len(query_ids) and len(set(candidate_ids)) == len(candidate_ids),
            "geometry eligibility identities duplicate")
    result = np.zeros((len(query_ids), len(candidate_ids)), dtype=np.bool_)
    query_index = {value: index for index, value in enumerate(query_ids)}
    by_episode = {row["episode_id"]: row for row in episodes}
    require(len(by_episode) == len(episodes) and set(by_episode) == set(candidate_ids),
            "geometry eligibility population differs")
    for column, episode_id in enumerate(candidate_ids):
        eligible_ids = by_episode[episode_id]["eligible_query_ids"]
        require(len(eligible_ids) == len(set(eligible_ids)) and set(eligible_ids) <= set(query_ids),
                f"frozen eligibility identities differ: {episode_id}")
        result[[query_index[value] for value in eligible_ids], column] = True
    return result


def _load_geometry(repository: Path, prereg: Mapping[str, Any], h1: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, np.ndarray]]:
    root = repository / GEOMETRY
    require(root.is_dir() and not root.is_symlink(), "frozen geometry output is absent or unsafe")
    result_path = root / "RESULT.json"; ids_path = root / "IDS.json"
    result = _read_json(result_path)
    _exact_keys(result, (
        "schema_version", "status", "passed", "preregistration_commit",
        "preregistration_digest", "implementation_commit", "ids", "arrays",
        "source_manifest", "transform_audit", "claims", "geometry_digest",
    ), "geometry result")
    require(result.get("schema_version") == "m04r14-r1b-b2-geometry-v1"
            and result.get("status") == "geometry_complete_pending_independent_verification"
            and result.get("passed") is True, "geometry completion status differs")
    require(result.get("geometry_digest") == stable_hash({key: value for key, value in result.items()
            if key != "geometry_digest"}), "geometry result digest differs")
    require(result.get("preregistration_digest") == prereg["preregistration_digest"], "geometry/H1 binding differs")
    require(result.get("preregistration_commit") == h1
            and result.get("implementation_commit") == prereg["implementation_commit"], "geometry H1 commit differs")
    ids_record = result.get("ids")
    _exact_keys(ids_record, (
        "path", "sha256", "query_ids_digest", "candidate_ids_digest", "population_digest",
    ), "geometry identity record")
    require(ids_record.get("path") == "IDS.json", "geometry identity file path differs")
    identities = _read_bound_json(ids_path, ids_record.get("sha256"))
    _exact_keys(identities, (
        "schema_version", "query_ids", "candidate_ids", "query_ids_digest",
        "candidate_ids_digest", "population_digest", "candidate_keys",
    ), "geometry identities")
    query_ids = identities.get("query_ids"); candidate_ids = identities.get("candidate_ids")
    population = prereg["authorities"]["population"]
    require(identities.get("schema_version") == "m04r14-r1b-b2-geometry-ids-v1"
            and identities.get("population_digest") == population["population_digest"],
            "geometry identity authority differs")
    require(query_ids == population["query_ids"] and candidate_ids == population["cohort_ids"],
            "geometry identities differ from frozen full population")
    require(len(query_ids) == QUERIES and len(candidate_ids) == COHORT_EPISODES,
            "geometry must retain 3270 queries and all 369 cohort candidates")
    require(identities.get("query_ids_digest") == id_digest(query_ids)
            and identities.get("candidate_ids_digest") == id_digest(candidate_ids)
            and ids_record.get("query_ids_digest") == identities["query_ids_digest"]
            and ids_record.get("candidate_ids_digest") == identities["candidate_ids_digest"]
            and ids_record.get("population_digest") == population["population_digest"],
            "geometry identity digest differs")
    expected_keys = [{"episode_id": row["episode_id"], "dataset_id": "nasdaq", "symbol": row["symbol"],
                      "cutoff_ns": row["cutoff_ns"], "lookback": 252, "representation_version": "dense-v1"}
                     for row in population["episodes"]]
    require(identities.get("candidate_keys") == expected_keys, "geometry candidate key identities differ")
    records = result.get("arrays")
    require(isinstance(records, dict), "geometry array manifest missing")
    required = {
        "raw_queries": ((QUERIES, DIMENSIONS), np.dtype("<f8")),
        "raw_candidates": ((COHORT_EPISODES, DIMENSIONS), np.dtype("<f8")),
        "transformed_queries": ((QUERIES, DIMENSIONS), np.dtype("<f8")),
        "transformed_candidates": ((COHORT_EPISODES, DIMENSIONS), np.dtype("<f8")),
        "query_pair_distances": ((QUERIES, QUERIES), np.dtype("<f8")),
        "query_candidate_distances": ((QUERIES, COHORT_EPISODES), np.dtype("<f8")),
        "specificity_ranks": ((QUERIES, COHORT_EPISODES), np.dtype("<f8")),
        "causal_eligibility": ((QUERIES, COHORT_EPISODES), np.dtype("?")),
    }
    require(set(records) == set(required), "geometry array inventory differs")
    arrays: dict[str, np.ndarray] = {}
    for name, (shape, dtype) in required.items():
        record = records[name]
        require(isinstance(record, dict) and set(record) == {"path", "sha256", "semantic_digest", "shape", "dtype", "order"},
                f"geometry array record malformed: {name}")
        path = _safe_relative(root, record["path"])
        require(record["shape"] == list(shape) and np.dtype(record["dtype"]) == dtype
                and record["order"] == "C",
                f"geometry array manifest differs: {name}")
        try:
            array = np.load(io.BytesIO(_bound_bytes(path, record["sha256"])), allow_pickle=False)
        except (OSError, ValueError) as error:
            raise B2RunnerError(f"geometry array unreadable: {name}") from error
        require(array.shape == shape and array.dtype == dtype and array.flags.c_contiguous,
                f"geometry array storage differs: {name}")
        require(record["semantic_digest"] == _geometry_array_digest(array),
                f"geometry semantic array digest differs: {name}")
        arrays[name] = array
    require(set(path.name for path in root.iterdir()) == {"RESULT.json", "IDS.json", *(records[name]["path"] for name in required)},
            "geometry output file closure differs")
    eligibility = arrays["causal_eligibility"]
    for name in required:
        value = arrays[name]
        if name == "specificity_ranks":
            require(np.isnan(value[~eligibility]).all() and np.isfinite(value[eligibility]).all()
                    and ((value[eligibility] > 0) & (value[eligibility] < 1)).all(),
                    "geometry specificity sentinels differ")
        elif value.dtype.kind == "f":
            require(np.isfinite(value).all(), f"nonfinite geometry array: {name}")
            require(not np.signbit(value[value == 0]).any(), f"negative zero geometry array: {name}")
    require((arrays["query_pair_distances"] >= 0).all()
            and (arrays["query_candidate_distances"] >= 0).all()
            and np.array_equal(arrays["query_pair_distances"], arrays["query_pair_distances"].T)
            and np.all(arrays["query_pair_distances"].diagonal() == 0), "geometry distance invariants differ")
    source = result.get("source_manifest")
    frozen_source = prereg["authorities"]["external_source_manifest"]
    expected_source = {
        "manifest_digest": frozen_source["manifest_digest"],
        "source_lock_digest": frozen_source["source_lock_digest"],
        "stock_files_count": frozen_source["stock_files_count"],
        "config_sha256": frozen_source["config"]["sha256"],
        "benchmark_sha256": frozen_source["benchmark"]["sha256"],
        "stock_files_digest": stable_hash(frozen_source["stock_files"]),
    }
    require(source == expected_source, "geometry source manifest binding differs")
    audit = _exact_keys(result.get("transform_audit"), (
        "sealed_transform_sha256", "sealed_query_ids_digest", "sealed_raw_query_digest",
        "sealed_transformed_query_digest", "sealed_integer_midrank_digest",
        "sealed_query_audits_digest", "reconstructed_query_audits_digest",
        "candidate_audits_digest", "candidate_audits", "constant_columns",
        "full_specificity_denominator", "unsupported_candidates_included",
        "query_reconstruction_workers", "distance_workers", "distance_chunk_rows",
        "distance_shards", "distance_shard_binding_digest", "performance",
        "synthetic_1_12_byte_identity_h0_test",
    ), "geometry transform audit")
    sealed = prereg["authorities"]["sealed_query_transform"]
    candidates = audit["candidate_audits"]
    require(isinstance(candidates, list) and len(candidates) == COHORT_EPISODES
            and audit["candidate_audits_digest"] == stable_hash(candidates),
            "geometry candidate audit binding differs")
    for expected_key, candidate in zip(expected_keys, candidates):
        _exact_keys(candidate, (
            "episode_id", "symbol", "cutoff_ns", "lookback", "representation_version",
            "representation_digest", "stock_prefix", "benchmark_prefix",
        ), "geometry candidate audit row")
        require(all(candidate[key] == expected_key[key] for key in (
            "episode_id", "symbol", "cutoff_ns", "lookback", "representation_version",
        )) and isinstance(candidate["representation_digest"], str)
                and re.fullmatch(r"[0-9a-f]{64}", candidate["representation_digest"]) is not None,
                "geometry candidate audit identity differs")
        for prefix_name in ("stock_prefix", "benchmark_prefix"):
            prefix = _exact_keys(candidate[prefix_name],
                                 ("schema_version", "requested_cutoff", "coverage_cutoff", "rows", "digest"),
                                 f"geometry candidate {prefix_name}")
            require(type(prefix["rows"]) is int and prefix["rows"] > 0
                    and prefix["schema_version"] == "canonical-ohlcv-prefix-v1"
                    and isinstance(prefix["requested_cutoff"], str)
                    and isinstance(prefix["coverage_cutoff"], str)
                    and isinstance(prefix["digest"], str)
                    and re.fullmatch(r"[0-9a-f]{64}", prefix["digest"]) is not None,
                    f"geometry candidate {prefix_name} differs")
    require(audit["sealed_transform_sha256"] == prereg["authorities"]["authority_file_sha256"][str(joint.TRANSFORM)]
            and audit["sealed_query_ids_digest"] == sealed["query_ids_digest"]
            and audit["sealed_raw_query_digest"] == sealed["raw_vector_digest"]
            and audit["sealed_transformed_query_digest"] == sealed["transformed_digest"]
            and audit["sealed_integer_midrank_digest"] == sealed["integer_midrank_digest"]
            and audit["sealed_query_audits_digest"] == stable_hash(sealed["query_audits"])
            and audit["reconstructed_query_audits_digest"] == stable_hash(sealed["query_audits"])
            and audit["constant_columns"] == sealed["constant_columns"]
            and audit["full_specificity_denominator"] == COHORT_EPISODES
            and audit["unsupported_candidates_included"] == 12
            and audit["query_reconstruction_workers"] == WORKERS
            and audit["distance_workers"] == WORKERS
            and audit["distance_chunk_rows"] == 64 and audit["distance_shards"] == 52
            and audit["synthetic_1_12_byte_identity_h0_test"]
                == "test_distance_serial_parallel_byte_identity and test_reconstruction_serial_threaded_byte_identity",
            "geometry transform audit authority differs")
    performance = _exact_keys(audit["performance"], (
        "wall_seconds", "parent_cpu_seconds", "children_cpu_seconds",
        "parent_lifetime_peak_rss_mib", "children_lifetime_peak_rss_mib", "blas_threads",
    ), "geometry performance")
    require(performance["blas_threads"] == 1 and all(
        isinstance(performance[key], (int, float)) and not isinstance(performance[key], bool)
        and math.isfinite(float(performance[key])) and float(performance[key]) >= 0
        for key in performance if key != "blas_threads"
    ), "geometry performance differs")
    expected_shard_binding = stable_hash({
        "schema_version": "m04r14-r1b-b2-geometry-distance-shard-v1",
        "preregistration_digest": prereg["preregistration_digest"],
        "runtime_sha256": prereg["runtime_sha256"],
        "query_ids_digest": identities["query_ids_digest"],
        "candidate_ids_digest": identities["candidate_ids_digest"],
        "transformed_queries": array_digest(arrays["transformed_queries"]),
        "transformed_candidates": array_digest(arrays["transformed_candidates"]),
        "chunk_rows": 64,
    })
    require(audit["distance_shard_binding_digest"] == expected_shard_binding,
            "geometry distance shard binding differs")

    expected_eligibility = _frozen_eligibility(query_ids, candidate_ids, population["episodes"])
    require(np.array_equal(eligibility, expected_eligibility),
            "geometry full-cohort causal eligibility differs")
    expected_specificity = candidate_specificity_ranks(
        arrays["query_candidate_distances"], expected_eligibility,
    )
    require(np.array_equal(np.isnan(arrays["specificity_ranks"]), np.isnan(expected_specificity))
            and np.array_equal(arrays["specificity_ranks"][expected_eligibility],
                               expected_specificity[expected_eligibility]),
            "geometry specificity replay differs")
    require(result.get("claims") == {"real_forward_outcomes_accessed": False, "b2_statistics_computed": False,
            "predictive_claim_authorized": False, "production_promotion_authorized": False},
            "geometry claim boundary differs")
    require(_read_json(result_path) == result and _read_json(ids_path) == identities,
            "geometry metadata changed while validating")
    require(set(path.name for path in root.iterdir()) == {"RESULT.json", "IDS.json", *(records[name]["path"] for name in required)},
            "geometry output changed while validating")
    return result, identities, arrays


def _selected_link_diagnostics(
    repository: Path, prereg: Mapping[str, Any], primary_ids: Sequence[str],
) -> dict[str, Any]:
    """Read exact H1-bound selected links for mandatory descriptive views only."""
    population = prereg["authorities"]["population"]
    primary = set(primary_ids)
    episodes = {row["episode_id"]: row for row in population["episodes"]}
    require(primary == set(population["primary_ids"]) and len(primary) == PRIMARY_EPISODES,
            "descriptive primary population differs")
    expected_links = {
        (episode_id, query_id)
        for episode_id in primary
        for query_id in episodes[episode_id]["observed_query_ids"]
    }
    require(len(expected_links) == PRIMARY_LINKS, "descriptive selected-link population differs")
    manifest = prereg["authorities"]["case_manifest"]
    require(isinstance(manifest, list) and len(manifest) == QUERIES, "H1 case manifest differs")
    seen_queries: set[str] = set()
    links: list[dict[str, Any]] = []
    root = repository.resolve()
    for record in manifest:
        _exact_keys(record, ("path", "bytes", "sha256"), "H1 case record")
        relative = Path(record["path"])
        require(not relative.is_absolute() and ".." not in relative.parts
                and relative.parent == joint.CASES, "H1 case path differs")
        path = root / relative
        require(path.resolve().is_relative_to(root)
                and all(not parent.is_symlink() for parent in (path, *path.parents) if parent != root.parent),
                "H1 case path is unsafe")
        case = _read_bound_json(path, record["sha256"], record["bytes"])
        query_id = case.get("query_episode_id")
        require(isinstance(query_id, str) and query_id not in seen_queries,
                "H1 case query identity differs")
        seen_queries.add(query_id)
        matches = case.get("matches")
        require(isinstance(matches, list), "H1 case matches differ")
        seen_episodes: set[str] = set()
        for match in matches:
            require(isinstance(match, dict), "H1 case match differs")
            episode_id = match.get("episode_id")
            require(isinstance(episode_id, str) and episode_id not in seen_episodes,
                    "H1 case selected episode identity differs")
            seen_episodes.add(episode_id)
            if episode_id not in primary:
                continue
            total = match.get("total_distance")
            components = match.get("component_distances")
            require(isinstance(components, dict) and "market_context" in components,
                    "selected-link market_context is absent")
            market = components["market_context"]
            require(all(isinstance(value, (int, float)) and not isinstance(value, bool)
                        and math.isfinite(float(value)) and float(value) >= 0
                        for value in (total, market)),
                    "selected-link descriptive distance differs")
            total_value = float(total); market_value = float(market)
            if total_value == 0:
                total_value = 0.0
            if market_value == 0:
                market_value = 0.0
            links.append({"episode_id": episode_id, "query_id": query_id,
                          "production_total": total_value, "raw_market_context": market_value})
    require(seen_queries == set(population["query_ids"]), "H1 descriptive query population differs")
    links.sort(key=lambda row: (row["episode_id"], row["query_id"]))
    require(len(links) == len(expected_links)
            and {(row["episode_id"], row["query_id"]) for row in links} == expected_links,
            "descriptive selected-link membership differs")
    rows = []
    for episode_id in primary_ids:
        values = [row for row in links if row["episode_id"] == episode_id]
        require(len(values) == episodes[episode_id]["observed_count"],
                f"descriptive episode cardinality differs: {episode_id}")
        rows.append({
            "episode_id": episode_id, "selected_links": len(values),
            "production_total_mean": math.fsum(row["production_total"] for row in values) / len(values),
            "raw_market_context_mean": math.fsum(row["raw_market_context"] for row in values) / len(values),
        })
    result = {
        "status": "descriptive_only",
        "scope": {"primary_episodes": len(rows), "selected_links": len(links)},
        "unit": "selected-link means within episode, then equal-episode means",
        "links": links,
        "links_digest": stable_hash(links),
        "episode_rows": rows,
        "episode_rows_digest": stable_hash(rows),
        "selected_link_aggregate": {
            "production_total": math.fsum(row["production_total"] for row in links) / len(links),
            "raw_market_context": math.fsum(row["raw_market_context"] for row in links) / len(links),
        },
        "equal_episode_aggregate": {
            "production_total": math.fsum(row["production_total_mean"] for row in rows) / len(rows),
            "raw_market_context": math.fsum(row["raw_market_context_mean"] for row in rows) / len(rows),
        },
        "claims": {"null_computed": False, "pvalue_computed": False,
                   "decision_input": False, "rescue_authorized": False},
    }
    return json.loads(json.dumps(result, allow_nan=False))


def _groups(eligible: Sequence[int], observed: Sequence[int], cells: Sequence[Hashable]) -> DesignGroups:
    required = Counter(cells[index] for index in observed)
    available = {cell: tuple(index for index in eligible if cells[index] == cell) for cell in required}
    require(all(len(available[cell]) >= count for cell, count in required.items()), "conditional cell lacks support")
    ordered = sorted(required, key=lambda value: json.dumps(value, separators=(",", ":"), sort_keys=True))
    return DesignGroups(tuple((required[cell], available[cell]) for cell in ordered))


def _build_plans(prereg: Mapping[str, Any], identities: Mapping[str, Any], eligibility: np.ndarray) -> tuple[tuple[str, ...], tuple[EpisodePlan, ...], np.ndarray]:
    population = prereg["authorities"]["population"]
    query_ids = tuple(identities["query_ids"]); candidate_ids = tuple(identities["candidate_ids"])
    query_index = {value: index for index, value in enumerate(query_ids)}
    candidate_index = {value: index for index, value in enumerate(candidate_ids)}
    require(len(query_index) == QUERIES and len(candidate_index) == COHORT_EPISODES, "duplicate geometry identities")
    cell_rows = population["query_cells"]
    require([row["query_id"] for row in cell_rows] == list(query_ids), "query-cell order differs")
    n0 = [tuple(row["n0"]) for row in cell_rows]
    all_labels = {k: np.asarray([int(row["labels"][str(k)]) for row in cell_rows], dtype=np.int8)
                  for k in (8, 12, 16)}
    require(all(value.shape == (QUERIES,) and ((0 <= value) & (value < k)).all()
                for k, value in all_labels.items()), "partition labels differ")
    labels = all_labels[8]
    cell_designs = (
        n0,
        [value + (int(row["labels"]["8"]),) for value, row in zip(n0, cell_rows)],
        [value + (int(row["labels"]["12"]),) for value, row in zip(n0, cell_rows)],
        [value + (int(row["labels"]["16"]),) for value, row in zip(n0, cell_rows)],
    )
    by_episode = {row["episode_id"]: row for row in population["episodes"]}
    primary_ids = tuple(population["primary_ids"])
    require(len(primary_ids) == PRIMARY_EPISODES and len(set(primary_ids)) == PRIMARY_EPISODES,
            "primary population must be exact 357")
    plans = []
    for episode_id in primary_ids:
        row = by_episode[episode_id]; column = candidate_index[episode_id]
        eligible = tuple(query_index[value] for value in row["eligible_query_ids"])
        observed = tuple(query_index[value] for value in row["observed_query_ids"])
        require(row.get("primary") is True and row["designs"]["N1"]["support_capped"] == REPLICATES,
                f"primary K8 support differs: {episode_id}")
        require(set(observed) <= set(eligible), f"observed membership is not eligible: {episode_id}")
        require(set(np.flatnonzero(eligibility[:, column])) == set(eligible),
                f"geometry causal eligibility differs: {episode_id}")
        designs = tuple(_groups(eligible, observed, cells) for cells in cell_designs)
        active = tuple(sorted({index for _, members in designs[0].groups for index in members}))
        plans.append(EpisodePlan(episode_id, column, observed, active, designs))
    return primary_ids, tuple(plans), labels


def _priority_prefix(contract_digest: str, replicate: int, scheme: bytes) -> bytes:
    parts = (PRIORITY_DOMAIN, bytes.fromhex(contract_digest), PRIORITY_FAMILY,
             scheme, int(replicate).to_bytes(4, "big"))
    return b"".join(len(part).to_bytes(4, "big") + part for part in parts)


def _priority(prefix: bytes, episode_id: str | None, query_id: str) -> bytes:
    parts = (() if episode_id is None else (episode_id.encode(),)) + (query_id.encode(),)
    return sha256(prefix + b"".join(len(part).to_bytes(4, "big") + part for part in parts)).digest()


def _lp(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return len(encoded).to_bytes(4, "big") + encoded


def _priorities(base: Any, members: Sequence[int], suffixes: Sequence[bytes]) -> dict[int, bytes]:
    result = {}
    for index in members:
        digest = base.copy(); digest.update(suffixes[index]); result[index] = digest.digest()
    return result


def _select(groups: DesignGroups, priorities: Mapping[int, bytes], query_ids: Sequence[str]) -> tuple[int, ...]:
    selected: list[int] = []
    for count, members in groups.groups:
        chosen = heapq.nsmallest(count, members, key=lambda index: (priorities[index], query_ids[index]))
        require(len(chosen) == count, "conditional selection count differs")
        selected.extend(chosen)
    require(len(selected) == len(set(selected)), "conditional selection duplicates query")
    return tuple(sorted(selected, key=lambda index: query_ids[index]))


def _compute_chunk(state: RunState, start: int, stop: int) -> dict[str, np.ndarray]:
    require(0 <= start < stop <= REPLICATES, "replicate shard range differs")
    count = stop - start; episodes = len(state.plans)
    tables = {name: np.empty((count, episodes, 2), dtype="<f8") for name in NULL_TABLES}
    breadth = np.empty((count, episodes), dtype="<f8")
    shared_priority: list[dict[int, bytes]] = []
    for replicate in range(start, stop):
        prefix = _priority_prefix(state.bindings.priority_contract_digest, replicate, b"shared-query")
        shared_priority.append(_priorities(sha256(prefix), range(len(state.query_ids)),
                                           state.query_priority_suffixes))
    for episode_position, plan in enumerate(state.plans):
        selections = {name: [] for name in NULL_TABLES}
        for local, replicate in enumerate(range(start, stop)):
            prefix = (_priority_prefix(state.bindings.priority_contract_digest, replicate, b"episode")
                      + _lp(plan.episode_id))
            priorities = _priorities(sha256(prefix), plan.active, state.query_priority_suffixes)
            for name, groups in zip(NULL_TABLES[:4], plan.designs):
                selections[name].append(_select(groups, priorities, state.query_ids))
            for name, groups in zip(NULL_TABLES[4:], plan.designs[:2]):
                selections[name].append(_select(groups, shared_priority[local], state.query_ids))
        for name in NULL_TABLES:
            selected = np.asarray(selections[name], dtype=np.int64)
            tables[name][:, episode_position] = state.localization.evaluate(plan.candidate_index, selected)
            if name == "episode_n0":
                breadth[:, episode_position] = state.localization.breadth(selected)
    tables["episode_n0_breadth"] = breadth
    return tables


def _observed(state: RunState) -> tuple[np.ndarray, np.ndarray]:
    stats = np.empty((len(state.plans), 2), dtype="<f8")
    breadth = np.empty(len(state.plans), dtype="<f8")
    for position, plan in enumerate(state.plans):
        selected = np.asarray([plan.observed], dtype=np.int64)
        stats[position] = state.localization.evaluate(plan.candidate_index, selected)[0]
        breadth[position] = state.localization.breadth(selected)[0]
    return stats, breadth


def _binding_dict(bindings: Bindings) -> dict[str, str]:
    return asdict(bindings)


def _shard_name(start: int, stop: int) -> str:
    return f"replicates-{start:04d}-{stop - 1:04d}"


def _cleanup_unpublished_staging(parent: Path, expected_shards: set[str]) -> None:
    pattern = re.compile(r"^\.(replicates-[0-9]{4}-[0-9]{4})\.tmp-[0-9]+-[0-9a-f]{32}$")
    for path in tuple(parent.iterdir()):
        match = pattern.fullmatch(path.name)
        if match is None or match.group(1) not in expected_shards:
            continue
        require(path.is_dir() and not path.is_symlink(),
                f"unsafe localization staging evidence: {path.name}")
        shutil.rmtree(path)
    _fsync_directory(parent)
    leftovers = sorted(path.name for path in parent.iterdir() if path.name.startswith("."))
    require(not leftovers, f"incomplete shard evidence requires review: {leftovers}")


def _shard_state(state: RunState, start: int, stop: int, arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    manifest = {}
    for name in ARRAYS:
        value = arrays[name]
        manifest[name] = {"path": f"{name}.npy", "shape": list(value.shape), "dtype": value.dtype.str,
                          "content_digest": _array_digest(value), "sha256": sha256(_npy_bytes(value)).hexdigest()}
    payload = {"schema_version": SHARD_SCHEMA, "status": "complete", "replicate_start": start,
               "replicate_stop": stop, "episodes": len(state.plans), "bindings": _binding_dict(state.bindings),
               "primary_ids_digest": stable_hash(list(state.primary_ids)), "arrays": manifest}
    return {**payload, "shard_digest": stable_hash(payload)}


def _publish_shard(state: RunState, start: int, stop: int) -> dict[str, Any]:
    parent = state.output / "shards"; parent.mkdir(parents=True, exist_ok=True)
    final = parent / _shard_name(start, stop)
    require(not final.exists() and not final.is_symlink(), f"shard target already exists: {final}")
    temporary = parent / f".{final.name}.tmp-{os.getpid()}-{uuid4().hex}"
    temporary.mkdir()
    try:
        arrays = _compute_chunk(state, start, stop)
        seal = _shard_state(state, start, stop, arrays)
        for name in ARRAYS:
            path = temporary / f"{name}.npy"
            path.write_bytes(_npy_bytes(arrays[name])); _fsync_file(path)
        (temporary / "SHARD.json").write_bytes(_json_bytes(seal)); _fsync_file(temporary / "SHARD.json")
        _fsync_directory(temporary)
        try:
            os.rename(temporary, final)
        except FileExistsError as error:
            raise B2RunnerError(f"concurrent shard publication: {final}") from error
        _fsync_directory(parent)
        return seal
    finally:
        if temporary.exists() and temporary.is_dir() and not temporary.is_symlink():
            shutil.rmtree(temporary)


def _load_shard(state: RunState, start: int, stop: int) -> tuple[dict[str, Any], dict[str, np.ndarray], str]:
    root = state.output / "shards" / _shard_name(start, stop)
    require(root.is_dir() and not root.is_symlink(), f"sealed shard missing: {root.name}")
    seal_path = root / "SHARD.json"
    seal_bytes = _snapshot_bytes(seal_path)
    seal = _decode_json(seal_bytes, seal_path)
    _exact_keys(seal, (
        "schema_version", "status", "replicate_start", "replicate_stop", "episodes",
        "bindings", "primary_ids_digest", "arrays", "shard_digest",
    ), "localization shard")
    require(set(path.name for path in root.iterdir()) == {"SHARD.json", *(f"{name}.npy" for name in ARRAYS)},
            f"shard file closure differs: {root.name}")
    require(seal.get("shard_digest") == stable_hash({key: value for key, value in seal.items() if key != "shard_digest"}),
            f"shard digest differs: {root.name}")
    require(seal.get("schema_version") == SHARD_SCHEMA and seal.get("status") == "complete"
            and seal.get("replicate_start") == start and seal.get("replicate_stop") == stop
            and seal.get("episodes") == len(state.plans), f"shard interval differs: {root.name}")
    require(seal.get("bindings") == _binding_dict(state.bindings)
            and seal.get("primary_ids_digest") == stable_hash(list(state.primary_ids)), f"shard binding differs: {root.name}")
    require(isinstance(seal.get("arrays"), dict) and set(seal["arrays"]) == set(ARRAYS),
            f"shard array inventory differs: {root.name}")
    arrays = {}
    for name in ARRAYS:
        path = root / f"{name}.npy"; record = seal["arrays"][name]
        try:
            value = np.load(io.BytesIO(_bound_bytes(path, record["sha256"])), allow_pickle=False)
        except (OSError, ValueError) as error:
            raise B2RunnerError(f"shard array unreadable: {root.name}/{name}") from error
        shape = (stop - start, len(state.plans)) + (() if name.endswith("breadth") else (2,))
        value = _canonical_array(value, dtype=np.dtype("<f8"), shape=shape, name=name, nonnegative=True)
        if name in NULL_TABLES:
            require(((value[:, :, 1] > 0) & (value[:, :, 1] < 1)).all(),
                    f"specificity outside open unit interval: {name}")
        else:
            require(((value > 0) & (value <= 1)).all(), "breadth outside valid range")
        require(record == {"path": f"{name}.npy", "shape": list(value.shape), "dtype": value.dtype.str,
                           "content_digest": _array_digest(value), "sha256": record["sha256"]},
                f"shard array record differs: {root.name}/{name}")
        arrays[name] = value
    require(_snapshot_bytes(seal_path) == seal_bytes,
            f"shard metadata changed while validating: {root.name}")
    require(set(path.name for path in root.iterdir()) == {"SHARD.json", *(f"{name}.npy" for name in ARRAYS)},
            f"shard changed while validating: {root.name}")
    return seal, arrays, sha256(seal_bytes).hexdigest()


_PROCESS_STATE: RunState | None = None


def _process_shard(bounds: tuple[int, int]) -> dict[str, Any]:
    require(_PROCESS_STATE is not None, "worker state not initialized")
    return _publish_shard(_PROCESS_STATE, *bounds)


def _summary_dict(summary: object) -> dict[str, Any]:
    return asdict(summary)  # type: ignore[arg-type]


def _descriptive_effect(summary: object) -> dict[str, Any]:
    value = _summary_dict(summary)
    value.pop("practical_pass", None)
    value.pop("required_improved_episodes", None)
    return value


def _split_effect(observed: np.ndarray, null_means: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    require(mask.dtype == np.bool_ and mask.ndim == 1 and mask.any(), "episode split differs")
    value = effect_summary(observed[mask], null_means[mask])
    return {
        "episodes": int(np.count_nonzero(mask)),
        "cohesion_relative_improvement": value.cohesion_relative_improvement,
        "specificity_improvement": value.specificity_improvement,
        "gate": "both effects >= 0.05; no p-value or episode-count gate",
        "gate_passed": value.cohesion_relative_improvement is not None
            and value.cohesion_relative_improvement >= 0.05
            and value.specificity_improvement >= 0.05,
    }


def _published_status(core_status: str) -> str:
    try:
        return {"structurally_localized": "established_pending_independent_verification",
                "not_established": "not_established_pending_independent_verification",
                "unresolved": "unresolved"}[core_status]
    except KeyError as error:
        raise B2RunnerError(f"unknown localization decision: {core_status}") from error


def _aggregate(state: RunState, ranges: Sequence[tuple[int, int]], observed: np.ndarray,
               observed_breadth: np.ndarray, selected_link_diagnostics: Mapping[str, Any]) -> dict[str, Any]:
    expected_ranges = tuple((start, min(start + SHARD_REPLICATES, REPLICATES))
                            for start in range(0, REPLICATES, SHARD_REPLICATES))
    require(tuple(ranges) == expected_ranges, "localization aggregate shard ranges differ")
    seals = []; chunks: dict[str, list[tuple[int, np.ndarray]]] = {name: [] for name in ARRAYS}
    for start, stop in ranges:
        seal, arrays, seal_sha = _load_shard(state, start, stop); seals.append((seal, seal_sha))
        for name, value in arrays.items():
            chunks[name].append((start, value))
    tables = {name: assemble_replicate_chunks(chunks[name], replicates=REPLICATES,
              episodes=len(state.plans)) for name in NULL_TABLES}
    # Breadth is one-dimensional per episode; use the same coverage checks by
    # temporarily adding a singleton metric and discard it after assembly.
    breadth = assemble_replicate_chunks([(start, value[:, :, None].repeat(2, axis=2))
              for start, value in chunks["episode_n0_breadth"]], replicates=REPLICATES,
              episodes=len(state.plans))[:, :, 0]
    summaries = {name: deterministic_null_summary(value) for name, value in tables.items()}
    effects = {name: effect_summary(observed, summaries[name][1]) for name in NULL_TABLES}
    parity = episode_parities(state.primary_ids)
    split_effects = {}
    for split in (0, 1):
        mask = parity == split
        require(mask.any(), f"episode parity {split} empty")
        split_effects[str(split)] = _split_effect(observed, summaries["episode_n1"][1], mask)
    leave_one_out = []
    for omitted, episode_id in enumerate(state.primary_ids):
        mask = np.arange(len(state.primary_ids)) != omitted
        value = effect_summary(observed[mask], summaries["episode_n1"][1][mask])
        leave_one_out.append({"omitted_episode_id": episode_id,
                              "cohesion_relative_improvement": value.cohesion_relative_improvement,
                              "specificity_improvement": value.specificity_improvement})
    decision = localization_decision(
        episode_ids=state.primary_ids, observed=observed,
        n0_null=tables["episode_n0"], n1_null=tables["episode_n1"],
        shared_query_n0_null=tables["shared_n0"], shared_query_n1_null=tables["shared_n1"],
        prerequisite_verified=True, primary_k=8,
    )
    status = _published_status(decision.status)
    array_digests = {name: _array_digest(value) for name, value in tables.items()}
    array_digests["episode_n0_breadth"] = _array_digest(breadth)
    scientific = {
        "status": status,
        "decision": {**asdict(decision),
                     "n1_effects": None if decision.n1_effects is None else _summary_dict(decision.n1_effects),
                     "n0_effects": None if decision.n0_effects is None else _summary_dict(decision.n0_effects)},
        "observed": {"statistics_digest": _array_digest(observed),
                     "breadth_digest": _array_digest(observed_breadth),
                     "cohesion": math.fsum(map(float, observed[:, 0])) / len(observed),
                     "specificity": math.fsum(map(float, observed[:, 1])) / len(observed),
                     "breadth": math.fsum(map(float, observed_breadth)) / len(observed_breadth)},
        "primary_effects": {name: _summary_dict(effects[name]) for name in ("episode_n0", "episode_n1")},
        "shared_priority_effects": {name: _descriptive_effect(effects[name]) for name in ("shared_n0", "shared_n1")},
        "descriptive_effect_sensitivity": {
            "episode_k12": {**_descriptive_effect(effects["episode_k12"]),
                            "interpretation": "descriptive only; no p-value, gate, or rescue"},
            "episode_k16": {**_descriptive_effect(effects["episode_k16"]),
                            "interpretation": "descriptive only; no p-value, gate, or rescue"},
        },
        "selected_link_production_diagnostics": selected_link_diagnostics,
        "robustness": {
            "episode_hash_splits": split_effects,
            "leave_one_episode_out": leave_one_out,
            "all_leave_one_out_effects_strictly_positive": all(
                row["cohesion_relative_improvement"] is not None
                and row["cohesion_relative_improvement"] > 0
                and row["specificity_improvement"] > 0 for row in leave_one_out),
            "shared_n0_effects_strictly_positive": (
                effects["shared_n0"].cohesion_relative_improvement is not None
                and effects["shared_n0"].cohesion_relative_improvement > 0
                and effects["shared_n0"].specificity_improvement > 0),
            "shared_n1_effects_strictly_positive": (
                effects["shared_n1"].cohesion_relative_improvement is not None
                and effects["shared_n1"].cohesion_relative_improvement > 0
                and effects["shared_n1"].specificity_improvement > 0),
        },
        "n0_breadth": {"replicate_mean": math.fsum(map(float, breadth.ravel(order="C"))) / breadth.size,
                       "table_digest": array_digests["episode_n0_breadth"]},
        "null_table_digests": array_digests,
        "claims": {"predictive_claim_authorized": False, "production_promotion_authorized": False,
                   "ranking_change_authorized": False, "outcomes_opened": False,
                   "independent_verification_required": True},
    }
    manifest = [{"path": f"shards/{_shard_name(start, stop)}/SHARD.json",
                 "sha256": seal_sha,
                 "shard_digest": seal["shard_digest"], "start": start, "stop": stop}
                for (start, stop), (seal, seal_sha) in zip(ranges, seals)]
    payload = {"schema_version": SCHEMA,
               "result_digest_scope": "all fields except result_digest and nonsemantic performance",
               "bindings": _binding_dict(state.bindings),
               "inventory": {"queries": len(state.query_ids), "cohort_episodes": COHORT_EPISODES,
                             "cohort_links": COHORT_LINKS, "primary_episodes": len(state.primary_ids),
                             "primary_links": sum(len(plan.observed) for plan in state.plans),
                             "unsupported_episodes": 12, "unsupported_links": 74,
                             "replicates": REPLICATES, "shards": len(ranges)},
               "identity": {"primary_ids": list(state.primary_ids),
                            "unsupported_ids": sorted(set(state.cohort_ids) - set(state.primary_ids))},
               "shards": manifest, "scientific": scientific}
    payload = json.loads(json.dumps(payload, allow_nan=False))
    return {**payload, "result_digest": stable_hash(payload)}


def _context(repository: Path) -> tuple[RunState, np.ndarray, np.ndarray, dict[str, Any]]:
    prereg = _read_json(repository / joint.PREREGISTRATION)
    h1 = joint.validate_h1(repository, prereg)
    geometry_result, identities, arrays = _load_geometry(repository, prereg, h1)
    primary_ids, plans, labels = _build_plans(prereg, identities, arrays["causal_eligibility"])
    localization = PreparedLocalization(arrays["query_pair_distances"], arrays["specificity_ranks"],
                                        arrays["causal_eligibility"], labels)
    require(_read_json(repository / GEOMETRY / "RESULT.json") == geometry_result
            and _read_json(repository / GEOMETRY / "IDS.json") == identities,
            "geometry authority changed while preparing localization")
    for name in ("query_pair_distances", "specificity_ranks", "causal_eligibility"):
        require(_file_sha(repository / GEOMETRY / geometry_result["arrays"][name]["path"])
                == geometry_result["arrays"][name]["sha256"],
                f"geometry changed while preparing localization: {name}")
    runtime_name = Path(__file__).resolve().relative_to(repository.resolve()).as_posix()
    require(prereg["runtime_sha256"].get(runtime_name) == _file_sha(repository / runtime_name), "runner runtime hash differs")
    bindings = Bindings(
        h1_commit=h1, preregistration_digest=prereg["preregistration_digest"],
        priority_contract_digest=prereg["b2_priority_contract_digest"],
        population_digest=prereg["authorities"]["population"]["population_digest"],
        runtime_sha256=prereg["runtime_sha256"][runtime_name],
        geometry_digest=geometry_result["geometry_digest"],
        query_ids_digest=identities["query_ids_digest"],
        candidate_ids_digest=identities["candidate_ids_digest"],
        primary_ids_digest=stable_hash(list(primary_ids)),
    )
    for shared in (False, True):
        scheme = b"shared-query" if shared else b"episode"
        prefix = _priority_prefix(bindings.priority_contract_digest, REPLICATES - 1, scheme)
        episode_id = primary_ids[-1]; query_id = identities["query_ids"][-1]
        expected = conditional_priority(bindings.priority_contract_digest, REPLICATES - 1,
                                        episode_id, query_id, shared_query=shared)
        require(_priority(prefix, None if shared else episode_id, query_id) == expected,
                "runner/core priority encoding differs")
        optimized_prefix = prefix if shared else prefix + _lp(episode_id)
        optimized = _priorities(sha256(optimized_prefix), (len(identities["query_ids"]) - 1,),
                                tuple(_lp(value) for value in identities["query_ids"]))
        require(optimized[len(identities["query_ids"]) - 1] == expected,
                "optimized runner/core priority encoding differs")
    query_ids = tuple(identities["query_ids"])
    state = RunState(query_ids, tuple(identities["candidate_ids"]),
                     primary_ids, plans, localization, bindings,
                     repository / OUTPUT, tuple(_lp(value) for value in query_ids))
    require(sum(len(plan.observed) for plan in plans) == PRIMARY_LINKS,
            f"primary observed-link population differs from {PRIMARY_LINKS}")
    observed, breadth = _observed(state)
    selected_link_diagnostics = _selected_link_diagnostics(repository, prereg, primary_ids)
    return state, observed, breadth, selected_link_diagnostics


def run(repository: Path, *, workers: int = WORKERS, shard_replicates: int = SHARD_REPLICATES) -> dict[str, Any]:
    repository = repository.resolve()
    require(type(workers) is int and workers == WORKERS, "production localization requires exactly 12 workers")
    require(type(shard_replicates) is int and shard_replicates == SHARD_REPLICATES,
            "production localization requires frozen 32-replicate shards")
    state, observed, observed_breadth, selected_link_diagnostics = _context(repository)
    output = state.output; output.mkdir(parents=True, exist_ok=True)
    require(not output.is_symlink(), "output directory may not be a symlink")
    lock_path = output.parent / f".{output.name}.lock"
    try:
        lock_descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    except OSError as error:
        raise B2RunnerError("localization lock path is unsafe") from error
    lock = os.fdopen(lock_descriptor, "a+b")
    try:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise B2RunnerError("another B2 localization producer holds the lock") from error
        result_path = output / "RESULT.json"
        require(set(path.name for path in output.iterdir()) <= {"shards", "RESULT.json"},
                "unexpected localization output evidence")
        shard_root = output / "shards"; shard_root.mkdir(exist_ok=True)
        require(shard_root.is_dir() and not shard_root.is_symlink(), "shard root is unsafe")
        ranges = [(start, min(start + shard_replicates, REPLICATES))
                  for start in range(0, REPLICATES, shard_replicates)]
        expected_shards = {_shard_name(start, stop) for start, stop in ranges}
        _cleanup_unpublished_staging(shard_root, expected_shards)
        leftovers = sorted(path.name for path in shard_root.iterdir() if path.name.startswith("."))
        require(not leftovers, f"incomplete shard evidence requires review: {leftovers}")
        require(set(path.name for path in shard_root.iterdir()) <= expected_shards,
                "unexpected localization shard evidence")
        if result_path.exists() or result_path.is_symlink():
            require(result_path.is_file() and not result_path.is_symlink(), "completed B2 result is unsafe")
            expected = _aggregate(state, ranges, observed, observed_breadth, selected_link_diagnostics)
            actual = _read_json(result_path)
            require({key: value for key, value in actual.items() if key != "performance"} == expected,
                    "completed B2 result differs from sealed shards")
            return actual
        missing = []
        for start, stop in ranges:
            target = shard_root / _shard_name(start, stop)
            if target.exists() or target.is_symlink():
                _load_shard(state, start, stop)
            else:
                missing.append((start, stop))
        started = perf_counter()
        global _PROCESS_STATE
        _PROCESS_STATE = state
        if workers == 1:
            for bounds in missing:
                _process_shard(bounds)
        elif missing:
            context = multiprocessing.get_context("fork")
            with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
                list(pool.map(_process_shard, missing))
        elapsed = perf_counter() - started
        require(set(path.name for path in shard_root.iterdir()) == expected_shards,
                "localization shard closure differs")
        result = _aggregate(state, ranges, observed, observed_breadth, selected_link_diagnostics)
        result["performance"] = {
            "workers": workers, "shard_replicates": shard_replicates,
            "new_shards": len(missing), "elapsed_seconds": elapsed,
            "claim": "measurement only; no unsupported fastest claim",
        }
        _atomic_json(result_path, result)
        return result
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN); lock.close()


def synthetic_worker_identity(state: RunState, start: int, stop: int) -> bool:
    """H0 gate: scientific arrays are byte-identical at 1 and 12-way chunks."""
    serial = _compute_chunk(state, start, stop)
    bounds = [(value, value + 1) for value in range(start, stop)]
    with ThreadPoolExecutor(max_workers=12, thread_name_prefix="b2-synthetic") as pool:
        generated = list(pool.map(lambda pair: _compute_chunk(state, *pair), reversed(bounds)))
    pieces = list(zip(reversed(bounds), generated))
    for name in ARRAYS:
        parallel = np.concatenate([piece[name] for _, piece in sorted(pieces)], axis=0)
        if _npy_bytes(serial[name]) != _npy_bytes(parallel):
            return False
    return True


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--shard-replicates", type=int, default=SHARD_REPLICATES)
    args = parser.parse_args(argv)
    try:
        result = run(args.repository, workers=args.workers, shard_replicates=args.shard_replicates)
    except (B2RunnerError, joint.JointContractError, LocalizationError) as error:
        print(f"B2 localization unresolved: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
