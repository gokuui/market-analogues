"""Independent integrity reconstruction for the sealed R1-B v3 support result.

This module intentionally does not import either v3 producer, the balanced
partition kernel, the adequacy-support kernel, or the R1-A experiment.  It
reimplements the frozen mathematics from the preregistration and uses shared
code only to read canonical OHLCV and construct the already-frozen chart
representation.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
from hashlib import sha256
from html import escape
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
from typing import Any, Mapping, Sequence
from uuid import uuid4

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_info, threadpool_limits

from market_analogues.adapters import source_from_spec
from market_analogues.causal_prefix import causal_prefix_digest
from market_analogues.config import load_config
from market_analogues.representation import represent, representation_input_digest
from market_analogues.types import Episode, EpisodeKey, InstrumentKey, stable_hash


SCHEMA = "m04r14-r1b-balanced-support-v3-integrity-verification-v1"
OUTPUT = Path(
    "config/data/analogues/m04r14/"
    "r1b-balanced-support-v3-integrity-verification-v1"
)
PREREGISTRATION = Path("experiments/m04r/m04r14_r1b_balanced_v3_preregistered.json")
PARTITION_OUTPUT = Path("config/data/analogues/m04r14/r1b-balanced-partition-v3")
SUPPORT_OUTPUT = Path("config/data/analogues/m04r14/r1b-balanced-support-v3")
REGISTRY = Path("config/data/analogues/m04r14/nasdaq-shadow-denominator-v1/query-registry.json")
CASES = Path("config/data/analogues/m04r14/nasdaq-shadow-snapshot-v1/cases")
CONFIG = Path("config/datasets.example.yaml")
BENCHMARK = Path("/home/vinay/code/loser-nasdaq/data/nasdaq/index/IXIC.parquet")
R1A_RESULT = Path("config/data/analogues/m04r14/r1a-exposure-audit-v2/RESULT.json")
V2_RESULT = Path("config/data/analogues/m04r14/r1b-support-pilot-v2/RESULT.json")
V2_QUERY_CELLS = Path("config/data/analogues/m04r14/r1b-support-pilot-v2/QUERY_CELLS.json")
V2_COHORT = Path("config/data/analogues/m04r14/r1b-support-pilot-v2/COHORT_SUPPORT.json")
V2_VERIFIED = Path(
    "config/data/analogues/m04r14/"
    "r1b-support-pilot-v2-integrity-verification-v1/VERIFIED.json"
)
PARTITION_RUNTIME = (
    "experiments/m04r/m04r14_r1b_balanced_partition_v3.py",
    "src/market_analogues/balanced_partition.py",
    "tests/test_balanced_partition.py",
)
SUPPORT_RUNTIME = (
    "experiments/m04r/m04r14_r1b_balanced_support_v3.py",
    "src/market_analogues/adequacy_support.py",
    "tests/test_r1b_balanced_support_v3.py",
)
VERIFIER_RUNTIME = (
    "experiments/m04r/m04r14_r1b_balanced_support_v3_verifier.py",
    "tests/test_r1b_balanced_support_v3_verifier.py",
    "src/market_analogues/adapters.py",
    "src/market_analogues/causal_prefix.py",
    "src/market_analogues/config.py",
    "src/market_analogues/representation.py",
    "src/market_analogues/types.py",
)
K_VALUES = (8, 12, 16)
CAP = 4096
MIN_COVERAGE = 0.90
V2_VERIFICATION_DIGEST = "1d14eabe98c2630dd94d5a8fbead62791368477573bcfcc85a80d758c6f0ca03"
CASE_RESULT_OMITTED = {
    "created_at", "proposal_seconds", "amortized_proposal_seconds",
    "exact_seconds", "final_exact_seconds", "frontier_attempt_measurements",
    "peak_rss_mb", "result_digest", "checkpoint_integrity_digest",
}


class BalancedSupportVerificationError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BalancedSupportVerificationError(message)


def _sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path, expected: type = dict) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in items:
            if key in value:
                raise BalancedSupportVerificationError(f"duplicate JSON key {key}: {path}")
            value[key] = item
        return value

    try:
        value = json.loads(
            path.read_bytes(), object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                BalancedSupportVerificationError(f"non-finite JSON {token}: {path}")
            ),
        )
    except (OSError, json.JSONDecodeError) as error:
        raise BalancedSupportVerificationError(f"invalid JSON: {path}") from error
    _require(isinstance(value, expected), f"JSON {expected.__name__} required: {path}")
    return value


def _digest(value: Mapping[str, Any], omitted: set[str]) -> str:
    return stable_hash({key: item for key, item in value.items() if key not in omitted})


def _git(repository: Path, *args: str, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        ("git", *args), cwd=repository, capture_output=True,
        text=not binary, check=False,
    )
    _require(result.returncode == 0, f"git {' '.join(args)} failed")
    return result.stdout if binary else result.stdout.strip()


def _committed_verifier(repository: Path) -> str:
    status = _git(repository, "status", "--porcelain", "--untracked-files=all")
    _require(not status, "verifier requires a clean committed tree")
    head = str(_git(repository, "rev-parse", "HEAD"))
    for relative in VERIFIER_RUNTIME:
        path = repository / relative
        _require(path.is_file() and not path.is_symlink(), f"verifier runtime absent: {relative}")
        blob = _git(repository, "show", f"{head}:{relative}", binary=True)
        _require(blob == path.read_bytes(), f"verifier runtime is not committed: {relative}")
    return head


def _preregistration_h1(repository: Path, prereg: Mapping[str, Any]) -> str:
    commits = str(_git(repository, "log", "--format=%H", "--", str(PREREGISTRATION))).splitlines()
    candidates = [
        commit for commit in commits
        if str(_git(repository, "rev-parse", f"{commit}^")) == prereg["implementation_commit"]
    ]
    _require(len(candidates) == 1, "preregistration H1 lineage differs")
    h1 = candidates[0]
    changed = str(_git(
        repository, "diff-tree", "--no-commit-id", "--name-only", "-r", h1,
    )).splitlines()
    _require(changed == [str(PREREGISTRATION)], "H1 is not the sole preregistration change")
    blob = _git(repository, "show", f"{h1}:{PREREGISTRATION}", binary=True)
    _require(blob == (repository / PREREGISTRATION).read_bytes(), "preregistration bytes differ")
    return h1


def _runtime_environment() -> dict[str, Any]:
    blas = [{
        key: value for key, value in row.items()
        if key in {
            "user_api", "internal_api", "prefix", "version",
            "threading_layer", "architecture",
        }
    } for row in threadpool_info() if row.get("user_api") == "blas"]
    return {
        "python": platform.python_version(), "numpy": np.__version__,
        "platform": platform.platform(),
        "blas": sorted(blas, key=lambda row: json.dumps(row, sort_keys=True)),
        "partition_blas_threads": 1,
    }


def _frozen_partition_contract() -> dict[str, Any]:
    return {
        "column_order": (
            "coarse[0:96] + stage.reshape(12,4,C)[:,0:3].ravel(C) + "
            "structural[0:9]"
        ),
        "fallback": (
            "relative eigengap <=1e-10; exact integer "
            "n*sum(m^2)-sum(m)^2; lowest column tie"
        ),
        "no_merge_retry_or_coarsening": True,
        "parallel_reconstruction_workers": 12,
        "primary_k": 8,
        "projection": (
            "math.fsum feature order; exact-equal boundary uses query ID; "
            "else margin >1e-10*max(1,maxabs)"
        ),
        "query_order": "ascending query episode ID",
        "rank": (
            "exact float64 ties; doubled 1-based midrank m=a+b; "
            "z=(m-(N+1))/(N-1); constants +0.0"
        ),
        "requested_k": [8, 12, 16],
        "scatter": "locally centered float64 Zc.T@Zc under single-thread BLAS",
        "secondary_k": [12, 16],
        "shape": [3270, 141],
    }


def _validate_preregistration(repository: Path) -> tuple[dict[str, Any], str]:
    prereg = _load(repository / PREREGISTRATION)
    _require(
        prereg.get("preregistration_digest")
        == _digest(prereg, {"preregistration_digest"}),
        "preregistration digest differs",
    )
    _require(set(prereg) == {
        "claims", "environment", "execution", "implementation_commit", "inventory",
        "partition_contract", "partition_file_sha256", "preregistration_digest",
        "prior_exposure", "runtime_sha256", "schema_version", "support_contract",
        "support_file_sha256",
    }, "preregistration field closure differs")
    _require(all((
        prereg["schema_version"] == "m04r14-r1b-balanced-v3-preregistration-v1",
        prereg["inventory"] == {
            "queries": 3270, "cohort_episodes": 369, "cohort_links": 2865,
        },
        prereg["execution"] == {
            "partition_output": str(PARTITION_OUTPUT),
            "support_output": str(SUPPORT_OUTPUT),
            "publication": (
                "flock-serialized create-only atomic directory rename with fsync"
            ),
        },
        prereg["environment"] == _runtime_environment(),
        prereg["partition_contract"] == _frozen_partition_contract(),
        prereg["support_contract"] == {
            "base": "verified v2 session_21 cells", "cap": 4096,
            "failure_states": [
                "partition_invalid", "reproducibility_failed", "support_inadequate",
                "support_pass_pending_independent_verification",
            ],
            "minimum_episode_coverage": 0.9, "minimum_link_coverage": 0.9,
            "primary": "K8 only; K12/K16 sensitivity cannot rescue",
        },
        prereg["claims"] == {
            "partition_only_before_support": True,
            "real_forward_outcomes_accessed": False,
            "r1b_statistics_opened": False, "b2_execution_authorized": False,
            "adequacy_labels_authorized": False, "predictive_claim_authorized": False,
            "production_promotion_authorized": False,
        },
        prereg["prior_exposure"] == {
            "v2_base_support_observed": True,
            "v2_structure_partition_degenerate": True,
            "v3_real_vectors_or_support_observed": False,
            "v2_verification_digest": V2_VERIFICATION_DIGEST,
        },
        set(prereg["runtime_sha256"]) == set((*PARTITION_RUNTIME, *SUPPORT_RUNTIME)),
        set(prereg["partition_file_sha256"]) == {
            str((repository / REGISTRY).resolve()), str((repository / CONFIG).resolve()),
            str(BENCHMARK.resolve()),
        },
        set(prereg["support_file_sha256"]) == {
            str((repository / R1A_RESULT).resolve()), str((repository / V2_RESULT).resolve()),
            str((repository / V2_QUERY_CELLS).resolve()),
            str((repository / V2_COHORT).resolve()), str((repository / V2_VERIFIED).resolve()),
        },
    )), "preregistration semantics differ")
    h1 = _preregistration_h1(repository, prereg)
    h0 = str(prereg["implementation_commit"])
    for relative, expected in prereg["runtime_sha256"].items():
        path = repository / relative
        _require(_sha(path) == expected, f"runtime hash differs: {relative}")
        blob = _git(repository, "show", f"{h0}:{relative}", binary=True)
        _require(sha256(blob).hexdigest() == expected, f"H0 runtime hash differs: {relative}")
        h1_blob = _git(repository, "show", f"{h1}:{relative}", binary=True)
        _require(sha256(h1_blob).hexdigest() == expected, f"H1 runtime hash differs: {relative}")
    for group in ("partition_file_sha256", "support_file_sha256"):
        for path, expected in prereg[group].items():
            target = Path(path)
            _require(target.is_file() and not target.is_symlink(), f"frozen input absent: {path}")
            _require(_sha(target) == expected, f"frozen input hash differs: {path}")
    return prereg, h1


def _validate_no_leakage_source(repository: Path) -> None:
    """Check the frozen stage boundaries in addition to trusting their hashes."""
    partition_text = (repository / PARTITION_RUNTIME[0]).read_text()
    support_text = (repository / SUPPORT_RUNTIME[0]).read_text()
    partition_tree = ast.parse(partition_text)
    support_tree = ast.parse(support_text)
    def imported_modules(tree: ast.AST) -> set[str]:
        modules: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                modules.add(node.module or "")
        return modules

    partition_imports = imported_modules(partition_tree)
    support_imports = imported_modules(support_tree)
    _require(
        not any("outcome" in name or "evidence_card" in name for name in partition_imports),
        "partition runtime imports an outcome module",
    )
    _require(
        not any("outcome" in name or "evidence_card" in name for name in support_imports),
        "support runtime imports an outcome module",
    )
    _require(
        '"candidate_or_eligibility_inputs_accessed": False' in partition_text,
        "partition no-candidate declaration absent",
    )
    _require(
        '"real_forward_outcomes_accessed": False' in partition_text
        and '"real_forward_outcomes_accessed": False' in support_text,
        "producer no-outcome declarations absent",
    )


def _array_digest(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    digest = sha256()
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _integer_digest(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype="<i8"))
    digest = sha256()
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _id_digest(ids: Sequence[str]) -> str:
    raw = json.dumps(list(ids), ensure_ascii=False, separators=(",", ":"))
    return sha256(raw.encode()).hexdigest()


def _independent_midranks(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    source = np.asarray(values, dtype=np.float64)
    _require(source.ndim == 2 and len(source) >= 2 and np.isfinite(source).all(),
             "rank input differs")
    rows, columns = source.shape
    integers = np.empty((rows, columns), dtype=np.int64)
    transformed = np.zeros((rows, columns), dtype=np.float64)
    constants: list[int] = []
    for column in range(columns):
        unique, inverse, counts = np.unique(
            source[:, column], return_inverse=True, return_counts=True,
        )
        before = np.r_[0, np.cumsum(counts[:-1], dtype=np.int64)]
        doubled = 2 * before + counts + 1
        integers[:, column] = doubled[inverse]
        if len(unique) == 1:
            constants.append(column)
        else:
            transformed[:, column] = (integers[:, column] - (rows + 1)) / (rows - 1)
    transformed[transformed == 0.0] = 0.0
    return transformed, integers, tuple(constants)


def _independent_axis(
    centered: np.ndarray, integer_values: np.ndarray,
) -> tuple[np.ndarray, str, int, float, float]:
    with threadpool_limits(limits=1, user_api="blas"):
        eigenvalues, eigenvectors = np.linalg.eigh(centered.T @ centered)
    leading = float(eigenvalues[-1])
    second = float(eigenvalues[-2]) if len(eigenvalues) > 1 else 0.0
    gap = (leading - second) / leading if leading > 0 else 0.0
    _require(np.isfinite(leading) and leading > 0, "node has no structure feature")
    if gap <= 1e-10:
        rows = len(integer_values)
        exact = []
        for column in range(integer_values.shape[1]):
            data = integer_values[:, column]
            total = sum(int(value) for value in data)
            squares = sum(int(value) ** 2 for value in data)
            exact.append(rows * squares - total * total)
        pivot = max(range(len(exact)), key=lambda value: (exact[value], -value))
        _require(exact[pivot] > 0, "fallback has no structure feature")
        axis = np.zeros(centered.shape[1], dtype=np.float64)
        axis[pivot] = 1.0
        return axis, "exact_variance_fallback", pivot, leading, gap
    axis = np.asarray(eigenvectors[:, -1], dtype=np.float64)
    _require(np.isfinite(axis).all(), "non-finite PCA axis")
    pivot = int(np.argmax(np.abs(axis)))
    if axis[pivot] < 0:
        axis = -axis
    return axis, "leading_pca", pivot, leading, gap


def _independent_partition(
    transformed: np.ndarray, integers: np.ndarray, ids: Sequence[str], leaves: int,
) -> tuple[np.ndarray, tuple[str, ...], tuple[int, ...], tuple[dict[str, Any], ...]]:
    values = np.asarray(transformed, dtype=np.float64)
    _require(values.shape == integers.shape and len(values) == len(ids)
             and len(set(ids)) == len(ids), "partition inputs differ")
    leaf_members: list[tuple[str, tuple[int, ...]]] = []
    splits: list[dict[str, Any]] = []

    def recurse(members: tuple[int, ...], targets: int, path: str) -> None:
        if targets == 1:
            leaf_members.append((path or "root", members)); return
        rows = len(members); left_targets = targets // 2
        right_targets = targets - left_targets
        left_count = rows * left_targets // targets
        _require(left_count >= left_targets and rows - left_count >= right_targets,
                 "balanced allocation cannot populate leaves")
        indices = np.asarray(members, dtype=np.int64)
        node = values[indices]
        centered = node - np.mean(node, axis=0)
        axis, mode, pivot, leading, gap = _independent_axis(centered, integers[indices])
        projection = np.asarray([
            math.fsum(float(row[column]) * float(axis[column]) for column in range(len(axis)))
            for row in centered
        ], dtype=np.float64)
        projection[projection == 0.0] = 0.0
        order = tuple(sorted(
            range(rows), key=lambda index: (float(projection[index]), ids[members[index]]),
        ))
        left = tuple(members[index] for index in order[:left_count])
        right = tuple(members[index] for index in order[left_count:])
        left_boundary = float(projection[order[left_count - 1]])
        right_boundary = float(projection[order[left_count]])
        margin = right_boundary - left_boundary
        scale = max(1.0, float(np.max(np.abs(projection))))
        _require(
            left_boundary.hex() == right_boundary.hex() or margin > 1e-10 * scale,
            "projection boundary unresolved",
        )
        left_path, right_path = f"{path}0", f"{path}1"
        splits.append({
            "path": path or "root", "rows": rows, "target_leaves": targets,
            "left_rows": len(left), "right_rows": len(right),
            "left_target_leaves": left_targets, "right_target_leaves": right_targets,
            "axis_mode": mode, "pivot_feature": pivot,
            "leading_eigenvalue_hex": leading.hex(), "relative_eigengap_hex": gap.hex(),
            "axis_digest": _array_digest(axis),
            "member_ids_digest": _id_digest([ids[index] for index in members]),
            "left_child_path": left_path, "right_child_path": right_path,
            "left_boundary_hex": left_boundary.hex(),
            "right_boundary_hex": right_boundary.hex(),
            "boundary_margin_hex": margin.hex(),
        })
        recurse(left, left_targets, left_path)
        recurse(right, right_targets, right_path)

    recurse(tuple(range(len(values))), leaves, "")
    labels = np.full(len(values), -1, dtype=np.int32)
    paths: list[str] = []; sizes: list[int] = []
    for label, (path, members) in enumerate(leaf_members):
        labels[np.asarray(members, dtype=np.int64)] = label
        paths.append(path); sizes.append(len(members))
    floor, ceiling = len(values) // leaves, (len(values) + leaves - 1) // leaves
    _require(len(set(map(int, labels))) == leaves and set(sizes) <= {floor, ceiling},
             "partition is not exactly balanced")
    return labels, tuple(paths), tuple(sizes), tuple(splits)


def _feature(
    query_id: str, row: Mapping[str, Any], stock: pd.DataFrame, benchmark: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, Any]]:
    cutoff = pd.Timestamp(str(row["cutoff"]))
    stock_prefix = asdict(causal_prefix_digest(stock, cutoff))
    benchmark_prefix = asdict(causal_prefix_digest(benchmark, cutoff))
    _require(stock_prefix == row["stock_prefix"], f"stock prefix differs: {query_id}")
    _require(benchmark_prefix == row["benchmark_prefix"], f"benchmark prefix differs: {query_id}")
    eligible = stock.loc[pd.to_datetime(stock["timestamp"]) <= cutoff]
    _require(int(row["lookback"]) == 252 and len(eligible) >= 252,
             f"query history differs: {query_id}")
    window = eligible.tail(252).copy().reset_index(drop=True)
    actual = pd.Timestamp(window["timestamp"].iloc[-1])
    context = benchmark.loc[pd.to_datetime(benchmark["timestamp"]) <= actual].copy()
    episode = Episode(
        EpisodeKey(
            InstrumentKey("nasdaq", str(row["symbol"])), actual, 252,
            str(row["representation_version"]),
        ),
        window, context, str(row["quality_tier"]),
    )
    _require(episode.key.id == query_id, f"query identity differs: {query_id}")
    represented = represent(episode)
    stage = np.asarray(represented.stage, dtype=np.float64).reshape(12, 4)[:, :3].ravel()
    vector = np.r_[
        np.asarray(represented.coarse[:96], dtype=np.float64), stage,
        np.asarray(represented.structural[:9], dtype=np.float64),
    ]
    _require(vector.shape == (141,) and np.isfinite(vector).all(),
             f"query vector differs: {query_id}")
    return vector, {
        "query_episode_id": query_id,
        "query_representation_digest": representation_input_digest(represented),
        "stock_prefix_digest": stock_prefix["digest"],
        "benchmark_prefix_digest": benchmark_prefix["digest"],
    }


def _features(
    repository: Path, query_ids: Sequence[str], rows: Sequence[Mapping[str, Any]], workers: int,
) -> tuple[np.ndarray, tuple[dict[str, Any], ...]]:
    spec = load_config(repository / CONFIG).datasets.get("nasdaq")
    _require(spec is not None and spec.benchmark is not None
             and spec.benchmark.path.resolve() == BENCHMARK.resolve(),
             "NASDAQ configuration differs")
    source = source_from_spec(spec); benchmark = source.load_benchmark()
    _require(benchmark is not None, "benchmark absent")

    def one(item: tuple[str, Mapping[str, Any]]) -> tuple[np.ndarray, dict[str, Any]]:
        query_id, row = item
        stock = source.load(InstrumentKey("nasdaq", str(row["symbol"])))
        return _feature(query_id, row, stock, benchmark)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        reconstructed = tuple(executor.map(one, zip(query_ids, rows, strict=True)))
    return (
        np.asarray([item[0] for item in reconstructed], dtype=np.float64),
        tuple(item[1] for item in reconstructed),
    )


def _expected_partition_payload(
    vectors: np.ndarray, audits: Sequence[Mapping[str, Any]], query_ids: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    transformed, integers, constants = _independent_midranks(vectors)
    transform = {
        "schema_version": "m04r14-r1b-balanced-transform-v1",
        "query_ids_digest": _id_digest(query_ids), "query_audits": list(audits),
        "raw_vector_digest": _array_digest(vectors),
        "integer_midrank_digest": _integer_digest(integers),
        "transformed_digest": _array_digest(transformed), "shape": [3270, 141],
        "constant_columns": list(constants),
    }
    assignments = [{"query_episode_id": value, "labels": {}} for value in query_ids]
    partitions: dict[str, Any] = {}
    for k in K_VALUES:
        labels, paths, sizes, splits = _independent_partition(transformed, integers, query_ids, k)
        for index, label in enumerate(labels):
            assignments[index]["labels"][str(k)] = int(label)
        partitions[str(k)] = {
            "status": "partition_valid", "requested_k": k, "retained_k": k,
            "leaf_paths": list(paths), "leaf_sizes": list(sizes),
            "membership_digest": stable_hash([
                [query_ids[index], int(label)] for index, label in enumerate(labels)
            ]),
            "splits": list(splits),
        }
    return transform, {
        "schema_version": "m04r14-r1b-balanced-partitions-v1",
        "assignments": assignments, "partitions": partitions,
    }


def _case_digest(case: Mapping[str, Any]) -> str:
    return stable_hash({key: value for key, value in case.items() if key not in CASE_RESULT_OMITTED})


def _case_integrity_digest(case: Mapping[str, Any]) -> str:
    return stable_hash({
        key: value for key, value in case.items()
        if key not in {"created_at", "checkpoint_integrity_digest"}
    })


def _eligible(
    candidate_symbol: str, candidate_cutoff: int,
    query_symbol: str, query_start: int, latest: int,
) -> bool:
    return candidate_cutoff <= latest and (
        candidate_symbol != query_symbol or candidate_cutoff < query_start
    )


def _matched_support(
    eligible: Sequence[int], observed: Sequence[int], cells: Sequence[Any], cap: int = CAP,
) -> int:
    eligible_values = tuple(map(int, eligible)); observed_values = tuple(map(int, observed))
    _require(len(set(eligible_values)) == len(eligible_values)
             and len(set(observed_values)) == len(observed_values), "duplicate membership")
    _require(set(observed_values).issubset(eligible_values), "observed membership is ineligible")
    available = Counter(cells[index] for index in eligible_values)
    selected = Counter(cells[index] for index in observed_values)
    support = 1
    for cell, count in selected.items():
        population = available.get(cell, 0)
        if population < count:
            return 0
        support *= math.comb(population, count)
        if support >= cap:
            return cap
    return support


def _summary(
    k: int, support: Mapping[str, int], cohort: Mapping[str, tuple[int, ...]],
    partition: Mapping[str, Any], crossed_cells: int,
) -> dict[str, Any]:
    passing = {key for key, value in support.items() if value >= CAP}
    links = sum(map(len, cohort.values()))
    supported_links = sum(len(cohort[key]) for key in passing)
    episode_coverage = len(passing) / len(cohort)
    link_coverage = supported_links / links
    return {
        "design": f"session_21_structure_{k}",
        "supported_episodes": len(passing), "cohort_episodes": len(cohort),
        "episode_coverage": episode_coverage, "supported_links": supported_links,
        "cohort_links": links, "link_coverage": link_coverage,
        "passes": episode_coverage >= MIN_COVERAGE and link_coverage >= MIN_COVERAGE,
        "role": "primary" if k == 8 else "secondary_sensitivity",
        "global_structure_cells": k, "retained_structure_cells": k,
        "leaf_size_min": min(partition["leaf_sizes"]),
        "leaf_size_max": max(partition["leaf_sizes"]),
        "crossed_cells": crossed_cells,
    }


def _support_authority(
    repository: Path, registry: Mapping[str, Any], query_ids: Sequence[str],
) -> tuple[
    dict[str, tuple[int, ...]], dict[str, tuple[int, ...]],
    tuple[tuple[Any, ...], ...], dict[str, Any], dict[str, Any], list[dict[str, Any]],
]:
    r1a = _load(repository / R1A_RESULT)
    v2 = _load(repository / V2_RESULT)
    verified = _load(repository / V2_VERIFIED)
    cells = _load(repository / V2_QUERY_CELLS, list)
    cohort_artifact = _load(repository / V2_COHORT, list)
    _require(r1a["result_digest"] == _digest(r1a, {"result_digest", "elapsed_seconds"}),
             "R1-A digest differs")
    _require(v2["result_digest"] == _digest(v2, {"result_digest"}), "v2 digest differs")
    _require(verified["verification_digest"]
             == _digest(verified, {"verification_digest", "created_at"})
             and verified.get("passed") is True
             and verified.get("verified_result_digest") == v2["result_digest"]
             and verified.get("verification_digest") == V2_VERIFICATION_DIGEST
             and verified.get("real_forward_outcomes_accessed") is False,
             "v2 verification differs")
    _require(v2.get("selected_base_design") == "session_21", "v2 base design differs")
    registry_rows = {str(row["episode_id"]): row for row in registry["cases_data"]}
    query_index = {value: index for index, value in enumerate(query_ids)}
    selected: defaultdict[str, list[str]] = defaultdict(list)
    recurrent_meta: dict[str, tuple[str, int]] = {}
    queries: dict[str, tuple[str, int, int]] = {}
    manifest = []
    paths = sorted((repository / CASES).glob("*.json"))
    _require(len(paths) == 3270, "case inventory differs")
    for path in paths:
        case = _load(path); query_id = str(case.get("query_episode_id"))
        row = registry_rows.get(query_id)
        _require(row is not None and query_id not in queries, f"case identity differs: {query_id}")
        _require(all((
            case.get("gate_passed") is True,
            case.get("real_forward_outcomes_accessed") is False,
            case.get("result_digest") == _case_digest(case),
            case.get("checkpoint_integrity_digest") == _case_integrity_digest(case),
            case.get("registry_case_id") == row["case_id"],
            case.get("query_symbol") == row["symbol"],
            case.get("query_cutoff") == row["cutoff"],
            case.get("query_stock_prefix") == row["stock_prefix"],
            case.get("query_benchmark_prefix") == row["benchmark_prefix"],
            len(case.get("matches", [])) == 20,
            len({str(match["episode_id"]) for match in case.get("matches", [])}) == 20,
        )), f"case seal differs: {query_id}")
        queries[query_id] = (
            str(case["query_symbol"]),
            int(np.datetime64(case["query_start"], "ns").view(np.int64)),
            int(np.datetime64(case["latest_eligible_cutoff"], "ns").view(np.int64)),
        )
        for match in case["matches"]:
            episode_id = str(match["episode_id"])
            meta = (
                str(match["symbol"]),
                int(np.datetime64(match["cutoff"], "ns").view(np.int64)),
            )
            _require(episode_id not in recurrent_meta or recurrent_meta[episode_id] == meta,
                     f"recurrent metadata differs: {episode_id}")
            recurrent_meta[episode_id] = meta; selected[episode_id].append(query_id)
        manifest.append({
            "path": path.relative_to(repository / CASES.parent).as_posix(),
            "bytes": path.stat().st_size, "sha256": _sha(path),
        })
    _require(set(queries) == set(query_ids), "case/query closure differs")
    _require(stable_hash(manifest) == r1a["inputs"]["case_manifest_digest"],
             "case manifest differs")
    cohort = {
        episode_id: tuple(sorted(query_index[value] for value in members))
        for episode_id, members in selected.items() if len(members) >= 5
    }
    _require(len(cohort) == 369 and sum(map(len, cohort.values())) == 2865,
             "recurrent cohort differs")
    eligible = {
        episode_id: tuple(
            index for index, query_id in enumerate(query_ids)
            if _eligible(
                recurrent_meta[episode_id][0], recurrent_meta[episode_id][1],
                queries[query_id][0], queries[query_id][1], queries[query_id][2],
            )
        ) for episode_id in cohort
    }
    _require(all(set(cohort[key]).issubset(eligible[key]) for key in cohort),
             "cohort membership is not causal")
    _require(len(cells) == 3270
             and [str(row.get("query_episode_id")) for row in cells] == list(query_ids),
             "v2 cell order differs")
    base_cells = tuple(tuple(row["base_cells"]["session_21"]) for row in cells)
    base_support = {
        key: _matched_support(eligible[key], cohort[key], base_cells) for key in cohort
    }
    stored = {str(row["episode_id"]): row for row in cohort_artifact}
    _require(len(stored) == len(cohort_artifact) == 369 and set(stored) == set(cohort),
             "v2 cohort artifact differs")
    for episode_id in cohort:
        row = stored[episode_id]
        _require(all((
            row["symbol"] == recurrent_meta[episode_id][0],
            int(np.datetime64(row["cutoff"], "ns").view(np.int64)) == recurrent_meta[episode_id][1],
            row["observed_inbound_queries"] == len(cohort[episode_id]),
            row["causally_eligible_queries"] == len(eligible[episode_id]),
            row["base_support"]["session_21"] == base_support[episode_id],
        )), f"v2 cohort row differs: {episode_id}")
    return cohort, eligible, base_cells, base_support, v2, cohort_artifact


def _expected_html(result: Mapping[str, Any]) -> str:
    rows = "".join(
        "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}–{}</td><td>{:.2%}</td><td>{:.2%}</td><td>{}</td></tr>".format(
            escape(str(row["design"])), escape(str(row.get("role", ""))),
            row["retained_structure_cells"], row["leaf_size_min"], row["leaf_size_max"],
            row["episode_coverage"], row["link_coverage"],
            "PASS" if row["passes"] else "insufficient",
        ) for row in result.get("designs", [])
    )
    return f"""<!doctype html><html lang='en'><head><meta charset='utf-8'><title>R1-B v3 balanced support</title><style>body{{font-family:system-ui;max-width:1000px;margin:2rem auto}}table{{border-collapse:collapse;width:100%}}td,th{{padding:.5rem;border-bottom:1px solid #ccc;text-align:left}}.note{{background:#fff5d8;padding:1rem}}</style></head><body><h1>R1-B v3 balanced structure support</h1><p><b>Status:</b> {escape(str(result['status']))}. Primary N1 is K=8; K12/K16 are secondary only.</p><table><thead><tr><th>Design</th><th>Role</th><th>Retained K</th><th>Leaf size range</th><th>Episode coverage</th><th>Link coverage</th><th>Gate</th></tr></thead><tbody>{rows}</tbody></table><p class='note'><b>Boundary:</b> support statistics only. No R1-B cohesion/specificity statistic, outcome, prediction, label, ranking change, production promotion or trading claim was opened or authorized.</p></body></html>"""


def _validate_outputs(
    repository: Path, prereg: Mapping[str, Any], h1: str,
    registry: Mapping[str, Any], query_ids: Sequence[str],
    expected_transform: Mapping[str, Any], expected_partitions: Mapping[str, Any],
    cohort: Mapping[str, tuple[int, ...]], eligible: Mapping[str, tuple[int, ...]],
    base_cells: Sequence[Any], base_support: Mapping[str, int], v2: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    partition_root = repository / PARTITION_OUTPUT
    support_root = repository / SUPPORT_OUTPUT
    _require(partition_root.is_dir() and not partition_root.is_symlink()
             and {path.name for path in partition_root.iterdir()}
             == {"TRANSFORM.json", "PARTITIONS.json", "RESULT.json"},
             "partition output closure differs")
    _require(support_root.is_dir() and not support_root.is_symlink()
             and {path.name for path in support_root.iterdir()}
             == {"SUPPORT.json", "RESULT.json", "report.html"},
             "support output closure differs")
    transform = _load(partition_root / "TRANSFORM.json")
    partitions = _load(partition_root / "PARTITIONS.json")
    partition_result = _load(partition_root / "RESULT.json")
    _require(transform == expected_transform, "transform reconstruction differs")
    _require(partitions == expected_partitions, "partition reconstruction differs")
    partition_state = {
        "schema_version": "m04r14-r1b-balanced-partition-v3",
        "status": "partition_valid", "passed": True,
        "preregistration_commit": h1,
        "preregistration_digest": prereg["preregistration_digest"],
        "registry_digest": registry["registry_digest"], "queries": 3270,
        "serial_parallel_reconstruction_identical": True,
        "serial_parallel_transform_digest_identical": True,
        "serial_parallel_partition_digest_identical": True,
        "primary_k": 8, "secondary_k": [12, 16],
        "real_forward_outcomes_accessed": False,
        "candidate_or_eligibility_inputs_accessed": False,
        "r1b_statistics_opened": False, "b2_execution_authorized": False,
        "adequacy_labels_authorized": False, "predictive_claim_authorized": False,
        "production_promotion_authorized": False,
        "transform_sha256": _sha(partition_root / "TRANSFORM.json"),
        "partitions_sha256": _sha(partition_root / "PARTITIONS.json"),
    }
    _require(set(partition_result) == set(partition_state) | {"result_digest", "created_at"}
             and all(partition_result[key] == value for key, value in partition_state.items())
             and partition_result["result_digest"] == stable_hash(partition_state),
             "partition RESULT differs")
    assignments = expected_partitions["assignments"]
    labels = {
        k: tuple(int(row["labels"][str(k)]) for row in assignments) for k in K_VALUES
    }
    support_by_k: dict[int, dict[str, int]] = {}
    designs = []
    for k in K_VALUES:
        crossed = tuple((*base_cells[index], labels[k][index]) for index in range(3270))
        _require(len(set(crossed)) > len(set(base_cells)), f"crossed K{k} cells are vacuous")
        values = {
            episode_id: _matched_support(eligible[episode_id], observed, crossed)
            for episode_id, observed in cohort.items()
        }
        support_by_k[k] = values
        designs.append(_summary(
            k, values, cohort, expected_partitions["partitions"][str(k)], len(set(crossed)),
        ))
    primary_pass = designs[0]["passes"] is True
    _require(designs[0]["role"] == "primary"
             and [row["role"] for row in designs[1:]]
             == ["secondary_sensitivity", "secondary_sensitivity"],
             "K role semantics differ")
    support_rows = [{
        "episode_id": episode_id,
        "observed_inbound_queries": len(cohort[episode_id]),
        "eligible_queries": len(eligible[episode_id]),
        "n0_support": base_support[episode_id],
        "n1_support": {str(k): support_by_k[k][episode_id] for k in K_VALUES},
    } for episode_id in sorted(cohort)]
    stored_rows = _load(support_root / "SUPPORT.json", list)
    _require(stored_rows == support_rows, "support-row reconstruction differs")
    result = _load(support_root / "RESULT.json")
    state = {
        "schema_version": "m04r14-r1b-balanced-support-v3",
        "status": (
            "support_pass_pending_independent_verification"
            if primary_pass else "support_inadequate"
        ),
        "passed": primary_pass, "preregistration_commit": h1,
        "preregistration_digest": prereg["preregistration_digest"],
        "partition_result_digest": partition_result["result_digest"],
        "v2_result_digest": v2["result_digest"],
        "v2_verification_digest": V2_VERIFICATION_DIGEST,
        "inventory": {"queries": 3270, "cohort_episodes": 369, "cohort_links": 2865},
        "designs": designs,
        "partition_status": {str(k): "partition_valid" for k in K_VALUES},
        "primary_k": 8, "secondary_k": [12, 16],
        "n0_matching_design_support_verified": True,
        "primary_support_gate_passed": primary_pass,
        "passed_meaning": "producer support gate only; independent verification pending",
        "b2_execution_authorized": False, "real_forward_outcomes_accessed": False,
        "r1b_statistics_opened": False, "adequacy_labels_authorized": False,
        "predictive_claim_authorized": False, "production_promotion_authorized": False,
        "support_sha256": _sha(support_root / "SUPPORT.json"),
        "report_sha256": _sha(support_root / "report.html"),
    }
    _require(set(result) == set(state) | {"result_digest", "created_at"}
             and all(result[key] == value for key, value in state.items())
             and result["result_digest"] == stable_hash(state),
             "support RESULT reconstruction differs")
    _require((support_root / "report.html").read_text() == _expected_html(result),
             "support report reconstruction differs")
    return result, support_rows, designs


def _atomic_publish(output: Path, payload: Mapping[str, Any]) -> None:
    _require(not output.exists() and not output.is_symlink(), f"create-only output exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.tmp-{os.getpid()}-{uuid4().hex}"
    lock_path = output.parent / f".{output.name}.publish.lock"
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        _require(not output.exists() and not output.is_symlink(),
                 f"create-only output exists: {output}")
        temporary.mkdir()
        path = temporary / "VERIFIED.json"
        with path.open("x") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        directory = os.open(temporary, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        os.rename(temporary, output)
        parent = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
        fcntl.flock(descriptor, fcntl.LOCK_UN); os.close(descriptor)


def _receipt_state(
    prereg: Mapping[str, Any], h1: str, verifier_commit: str,
    result: Mapping[str, Any], designs: Sequence[Mapping[str, Any]],
    runtime_hashes: Mapping[str, str],
) -> dict[str, Any]:
    primary_pass = bool(designs[0]["passes"])
    gates = {
        "h0_h1_preregistration_lineage_and_runtime_reconstructed": True,
        "all_frozen_input_hashes_reconstructed": True,
        "no_outcome_or_r1b_statistic_access_contract_verified": True,
        "all_3270_causal_query_prefixes_and_vectors_reconstructed": True,
        "midrank_transform_reconstructed": True,
        "k8_k12_k16_recursive_partitions_and_split_audits_reconstructed": True,
        "partition_result_and_sidecar_digests_reconstructed": True,
        "all_3270_case_seals_and_causal_eligibility_sets_reconstructed": True,
        "all_369_n0_and_n1_support_rows_reconstructed": True,
        "k8_primary_k12_k16_sensitivity_semantics_enforced": True,
        "support_result_report_and_sidecar_digests_reconstructed": True,
        "producer_pending_and_no_scientific_authorization_semantics_enforced": True,
    }
    return {
        "schema_version": SCHEMA, "status": "full_partition_and_support_integrity_reconstruction",
        "passed": all(gates.values()), "gates": gates,
        "verified_preregistration_commit": h1,
        "verified_preregistration_digest": prereg["preregistration_digest"],
        "verified_partition_result_digest": result["partition_result_digest"],
        "verified_support_result_digest": result["result_digest"],
        "verified_queries": 3270, "verified_cohort_episodes": 369,
        "verified_cohort_links": 2865,
        "primary_k": 8, "secondary_sensitivity_k": [12, 16],
        "primary_k8_support_gate_passed": primary_pass,
        "support_decision_verified": True,
        "b2_contract_freeze_may_proceed": primary_pass,
        "b2_scientific_execution_authorized": False,
        "r1b_statistics_opened": False, "real_forward_outcomes_accessed": False,
        "adequacy_labels_authorized": False, "predictive_claim_authorized": False,
        "ranking_change_authorized": False, "production_promotion_authorized": False,
        "passed_meaning": "integrity and matching-support decision only",
        "verifier_commit": verifier_commit,
        "verifier_runtime_sha256": dict(runtime_hashes),
    }


def execute(repository: Path, *, workers: int = 12) -> dict[str, Any]:
    repository = repository.resolve()
    _require(workers == 12, "verification requires exactly 12 source workers")
    verifier_commit = _committed_verifier(repository)
    output = repository / OUTPUT
    _require(not output.exists() and not output.is_symlink(),
             f"create-only output exists: {output}")
    prereg, h1 = _validate_preregistration(repository)
    _validate_no_leakage_source(repository)
    registry = _load(repository / REGISTRY)
    _require(registry.get("registry_digest") == _digest(registry, {"registry_digest"}),
             "registry digest differs")
    rows_by_id = {str(row["episode_id"]): row for row in registry["cases_data"]}
    _require(len(rows_by_id) == len(registry["cases_data"]) == 3270,
             "registry query closure differs")
    query_ids = tuple(sorted(rows_by_id))
    rows = tuple(dict(rows_by_id[value]) for value in query_ids)
    vectors, audits = _features(repository, query_ids, rows, workers)
    expected_transform, expected_partitions = _expected_partition_payload(
        vectors, audits, query_ids,
    )
    cohort, eligible, base_cells, base_support, v2, _v2_cohort = _support_authority(
        repository, registry, query_ids,
    )
    result, _support_rows, designs = _validate_outputs(
        repository, prereg, h1, registry, query_ids,
        expected_transform, expected_partitions, cohort, eligible,
        base_cells, base_support, v2,
    )
    runtime_hashes = {relative: _sha(repository / relative) for relative in VERIFIER_RUNTIME}
    state = _receipt_state(prereg, h1, verifier_commit, result, designs, runtime_hashes)
    payload = {
        **state, "verification_digest": stable_hash(state),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_publish(output, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args(argv)
    print(json.dumps(execute(args.repository, workers=args.workers), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
