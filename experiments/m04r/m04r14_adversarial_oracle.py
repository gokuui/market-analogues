"""Truth-free adversarial qualification for certified progressive search.

The fixture is generated entirely from deterministic synthetic OHLCV.  It
compares the production certified path with the production exhaustive search,
and separately exercises packed selection/cardinality and threshold-band
boundaries.  No registry, authority, label, or forward-outcome path is read.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, replace
from hashlib import sha256
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile
from typing import Any, Sequence

import numpy as np
import pandas as pd

from market_analogues.adapters import DirectorySource
from market_analogues.causal_prefix import causal_prefix_digest
from market_analogues.certified_packed_search import (
    CertifiedPackedSearchError, CompactScoredCandidate,
    _close_native_bound_deferred, certified_packed_search,
    certified_packed_search_contract,
)
from market_analogues.config import BenchmarkSpec, DatasetSpec
from market_analogues.episodes import build_episode
from market_analogues.packed_bound_search import (
    BoundProposalReport, PackedBoundQuery, PackedBoundSearchError,
    scan_packed_bound_proposals,
    scan_packed_bound_proposals_threaded, scan_packed_bound_threshold,
    packed_bound_search_contract, packed_bound_threshold_scan_contract,
)
from market_analogues.packed_bound_store import (
    OVERFLOW_DTYPE, PACK_DTYPE, PackedBoundStoreError,
    make_overflow_record, make_packed_record,
    write_packed_generation,
)
from market_analogues.quantized_bound import quantize_bound_row
from market_analogues.representation import represent
from market_analogues.representation import representation_input_digest
from market_analogues.search import (
    SearchCandidate, eligible, exact_search, latest_eligible_cutoff,
)
from market_analogues.synthetic import generate_case
from market_analogues.types import (
    AnalogueMatch, EpisodeKey, InstrumentKey, SearchQuery, stable_hash,
)


SCHEMA = "m04r14-adversarial-oracle-v1"
DATASET = "m04r14-synthetic"
LOOKBACK = 40
SYMBOLS = ("S0", "S1", "S2", "S3")
NUMERIC_ATOL = 1e-6
CANONICAL_OUTPUT = Path(
    "config/data/analogues/m04r14/adversarial-oracle-v1/oracle.json"
)
_MODULE_REPO = Path(__file__).resolve().parents[2]
RUNTIME_FILES = (
    Path("experiments/m04r/m04r14_adversarial_oracle.py"),
    *(path.relative_to(_MODULE_REPO) for path in sorted(
        (_MODULE_REPO / "src/market_analogues").glob("*.py")
    )),
)
THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
    "NUMBA_NUM_THREADS", "NUMBA_THREADING_LAYER",
)


class OracleError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _contract_binding() -> dict[str, str]:
    return {
        "branch_aware_proposal": packed_bound_search_contract(
            branch_aware=True,
        )["digest"],
        "branch_aware_threshold": packed_bound_threshold_scan_contract(
            branch_aware=True,
        )["digest"],
        "certified_v8": certified_packed_search_contract(
            requested_positions=True, vector_lower_bounds=True,
            deferred_alignments=True, compact_scored=True,
            native_bound_deferral=True, streaming_threshold_closure=True,
            branch_aware_packed_bounds=True,
        )["digest"],
    }


def _environment_binding() -> dict[str, Any]:
    state = {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "platform_system": platform.system(),
        "machine": platform.machine(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "numba", "pyarrow")
        },
        "thread_environment": {
            name: os.environ.get(name) for name in THREAD_ENV_KEYS
        },
        "cpu_affinity": (
            sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity") else None
        ),
    }
    return {**state, "digest": stable_hash(state)}


def test_runtime_manifest() -> dict[str, Any]:
    state = {
        "mode": "test-injected",
        "git_head": "0" * 40,
        "files": {"synthetic-test-fixture": "1" * 64},
        "contracts": _contract_binding(),
        "environment": _environment_binding(),
    }
    return {"state": state, "digest": stable_hash(state)}


def _production_runtime_manifest(repo_root: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve(strict=True)

    def git(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", *arguments], cwd=repo_root, check=True,
            text=True, capture_output=True,
        )
        return completed.stdout.strip()

    if Path(git("rev-parse", "--show-toplevel")).resolve() != repo_root:
        raise OracleError("repository root differs")
    if git("status", "--porcelain", "--untracked-files=all"):
        raise OracleError("production oracle requires a globally clean worktree")
    head = git("rev-parse", "HEAD")
    files = {}
    for relative in RUNTIME_FILES:
        git("ls-files", "--error-unmatch", str(relative))
        working_sha = _sha256(repo_root / relative)
        committed = subprocess.run(
            ["git", "show", f"{head}:{relative}"], cwd=repo_root,
            check=True, capture_output=True,
        ).stdout
        if sha256(committed).hexdigest() != working_sha:
            raise OracleError(f"runtime file differs from HEAD: {relative}")
        files[str(relative)] = working_sha
    state = {
        "mode": "production-clean-head", "git_head": head,
        "files": files, "contracts": _contract_binding(),
        "environment": _environment_binding(),
    }
    return {"state": state, "digest": stable_hash(state)}


def _validate_production_manifest_against_repo(
    value: dict[str, Any], repo_root: Path,
    *, runtime_files: Sequence[Path] = RUNTIME_FILES,
) -> None:
    """Validate immutable H0 runtime bytes while permitting unrelated H1 commits."""
    repo_root = repo_root.resolve(strict=True)

    def git(*arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *arguments], cwd=repo_root, check=check,
            text=True, capture_output=True,
        )

    state = _validate_seal(value, "runtime manifest")
    if state.get("environment") != _environment_binding():
        raise OracleError("stored runtime environment differs")
    if git("status", "--porcelain", "--untracked-files=all").stdout.strip():
        raise OracleError("production validation requires a globally clean worktree")
    current_head = git("rev-parse", "HEAD").stdout.strip()
    stored_head = state.get("git_head")
    if not _is_hex(stored_head, 40):
        raise OracleError("stored runtime H0 differs")
    if git("cat-file", "-e", f"{stored_head}^{{commit}}", check=False).returncode:
        raise OracleError("stored runtime H0 is absent from repository")
    if git(
        "merge-base", "--is-ancestor", stored_head, current_head, check=False,
    ).returncode:
        raise OracleError("stored runtime H0 is not an ancestor of current HEAD")
    expected_paths = {str(path) for path in runtime_files}
    if set(state.get("files", {})) != expected_paths:
        raise OracleError("stored runtime file set differs")
    for relative in runtime_files:
        name = str(relative)
        expected_sha = state["files"][name]
        try:
            h0_bytes = subprocess.run(
                ["git", "show", f"{stored_head}:{name}"], cwd=repo_root,
                check=True, capture_output=True,
            ).stdout
            head_bytes = subprocess.run(
                ["git", "show", f"{current_head}:{name}"], cwd=repo_root,
                check=True, capture_output=True,
            ).stdout
        except subprocess.CalledProcessError as exc:
            raise OracleError(f"runtime path is absent from H0 or HEAD: {name}") from exc
        if sha256(h0_bytes).hexdigest() != expected_sha:
            raise OracleError(f"stored H0 runtime blob differs: {name}")
        if sha256(head_bytes).hexdigest() != expected_sha:
            raise OracleError(f"current HEAD runtime blob drifted: {name}")
        path = repo_root / relative
        if not path.is_file() or path.is_symlink() or _sha256(path) != expected_sha:
            raise OracleError(f"worktree runtime blob drifted: {name}")


def _bars(
    *, phase: float = 0.0, count: int = 100, random_seed: int | None = None,
) -> pd.DataFrame:
    index = np.arange(count, dtype=np.float64)
    angle = 2.0 * np.pi * (index + phase) / 20.0
    close = 100.0 * np.exp(0.025 * np.sin(angle))
    open_ = close * (1.0 + 0.002 * np.cos(angle))
    high = np.maximum(open_, close) * 1.005
    low = np.minimum(open_, close) * 0.995
    volume = 100_000.0 * (1.2 + 0.15 * np.cos(angle))
    if random_seed is not None:
        random = np.random.default_rng(random_seed)
        close *= np.exp(np.cumsum(random.normal(0.0, 0.0025, count)))
        open_ = close * np.exp(random.normal(0.0, 0.0015, count))
        high = np.maximum(open_, close) * (1.002 + random.uniform(0, 0.006, count))
        low = np.minimum(open_, close) * (0.998 - random.uniform(0, 0.006, count))
        volume *= np.exp(random.normal(0.0, 0.08, count))
    return pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=count, freq="B"),
        "open": open_, "high": high, "low": low, "close": close,
        "volume": volume,
    })


def _source(
    root: Path, *, random_seed: int | None = None,
    missing_volume_symbol: str | None = None,
) -> DirectorySource:
    bars_root = root / "bars"
    bars_root.mkdir()
    query = _bars(random_seed=random_seed)
    query.to_parquet(bars_root / "QUERY.parquet", index=False)
    for index, symbol in enumerate(SYMBOLS):
        candidate = _bars(
            random_seed=(None if random_seed is None else random_seed + index + 1),
        )
        if symbol == missing_volume_symbol:
            candidate = candidate.assign(volume=np.nan)
        candidate.to_parquet(bars_root / f"{symbol}.parquet", index=False)
    benchmark_path = bars_root / "MARKET.parquet"
    _bars(
        phase=3.0,
        random_seed=(None if random_seed is None else random_seed + 100),
    ).to_parquet(benchmark_path, index=False)
    return DirectorySource(DatasetSpec(
        DATASET, "directory", bars_root, "parquet", timestamp_column="date",
        benchmark=BenchmarkSpec(benchmark_path, timestamp_column="date"),
    ))


def _semantic_report(report: BoundProposalReport) -> dict[str, Any]:
    return {
        name: getattr(report, name)
        for name in (
            "schema_version", "generation_id", "query_episode_id",
            "candidates", "rows_scanned", "eligible_rows",
            "eligible_main_rows", "eligible_overflow_rows", "route_counts",
            "route_quotas", "candidate_digest", "result_digest",
            "contract_digest", "input_digest",
        )
    }


def _strict_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode()
    except (TypeError, ValueError) as exc:
        raise OracleError("oracle state is not finite strict JSON") from exc


def _hex_state(value: Any) -> Any:
    if isinstance(value, float):
        if not np.isfinite(value):
            raise OracleError("sealed state contains a nonfinite float")
        return value.hex()
    if isinstance(value, dict):
        return {str(key): _hex_state(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_hex_state(item) for item in value]
    if value is None or type(value) in {str, int, bool}:
        return value
    raise OracleError(f"sealed state contains unsupported type: {type(value).__name__}")


def _seal_matches(matches: Sequence[Any]) -> dict[str, Any]:
    rows = []
    for match in matches:
        key = match.episode_key
        rows.append({
            "episode_id": key.id,
            "dataset_id": key.instrument.dataset_id,
            "symbol": key.instrument.source_symbol,
            "cutoff": key.cutoff.isoformat(),
            "lookback": key.lookback,
            "representation_version": key.representation_version,
            "total_distance_hex": float(match.total_distance).hex(),
            "component_distances_hex": {
                name: float(value).hex()
                for name, value in sorted(match.component_distances.items())
            },
            "alignment": [list(pair) for pair in match.alignment],
            "quality_tier": match.quality_tier,
            "quality_issues": list(match.quality_issues),
        })
    state = {"rows": rows}
    return {"state": state, "digest": stable_hash(state)}


def _seal_certificate(certificate: Any) -> dict[str, Any]:
    state = asdict(certificate)
    state.pop("elapsed_seconds", None)
    state = _hex_state(state)
    return {"state": state, "digest": stable_hash(state)}


def _certificate_result_digest_from_sealed(
    state: dict[str, Any], matches: list[dict[str, Any]],
) -> str:
    """Reconstruct the production v8 result digest from timing-free evidence."""
    def decoded(row: dict[str, Any], names: Sequence[str]) -> dict[str, Any]:
        output = dict(row)
        for name in names:
            output[name] = (
                None if row[name] is None else float.fromhex(row[name])
            )
        return output

    deterministic = {
        "schema_version": state["schema_version"],
        "contract_digest": state["contract_digest"],
        "generation_id": state["generation_id"],
        "query_episode_id": state["query_episode_id"],
        "input_digest": state["input_digest"],
        "eligible_candidates": state["eligible_candidates"],
        "exact_evaluated": state["exact_evaluated"],
        "safely_pruned": state["safely_pruned"],
        "stopped_early": state["stopped_early"],
        "stop_threshold_hex": state["stop_threshold"],
        "next_lower_bound_hex": state["next_lower_bound"],
        "maximum_quantized_bound_excess_hex": state[
            "maximum_quantized_bound_excess"
        ],
        "rounds": [decoded(row, (
            "next_lower_bound", "constrained_threshold",
        )) for row in state["rounds"]],
        "matches": [{
            "episode_id": row["episode_id"],
            "total_hex": row["total_distance_hex"],
            "components": row["component_distances_hex"],
            "alignment": row["alignment"],
        } for row in matches],
        "real_forward_outcomes_accessed": False,
        "native_bound_accounting": state["native_bound_accounting"],
        "minimum_native_pruned_bound_hex": state[
            "minimum_native_pruned_bound"
        ],
        "threshold_closure_passes": [decoded(row, (
            "lower_exclusive", "upper_inclusive", "resulting_threshold",
            "minimum_packed_unclassified_bound", "minimum_native_pruned_bound",
        )) for row in state["threshold_closure_passes"]],
    }
    return stable_hash(deterministic)


def _query_binding(
    query: Any, request: SearchQuery, *, universe: dict[str, Any],
) -> dict[str, Any]:
    deterministic = {
        "query_episode_id": query.key.id,
        "query_dataset_id": query.key.instrument.dataset_id,
        "query_symbol": query.key.instrument.source_symbol,
        "query_cutoff": query.key.cutoff.isoformat(),
        "lookback": query.key.lookback,
        "representation_version": query.key.representation_version,
        "universe": universe,
        "request": {
            "search_datasets": list(request.search_datasets),
            "quality_tiers": list(request.quality_tiers),
            "top_k": request.top_k,
            "cross_dataset": request.cross_dataset,
            "deduplicate_overlaps": request.deduplicate_overlaps,
            "max_per_instrument": request.max_per_instrument,
            "minimum_history_gap_bars": request.minimum_history_gap_bars,
        },
    }
    return {"state": deterministic, "digest": stable_hash(deterministic)}


_EXPECTED_UNIVERSE_CACHE: dict[str, tuple[str, dict[str, Any]]] = {}


def _reconstruct_expected_input_digest(query_state: dict[str, Any]) -> str:
    cache_key = stable_hash(query_state)
    if cache_key in _EXPECTED_UNIVERSE_CACHE:
        return _EXPECTED_UNIVERSE_CACHE[cache_key][0]
    universe = query_state["universe"]
    with tempfile.TemporaryDirectory(prefix="m04r14-input-rebuild-") as temporary:
        source = _source(
            Path(temporary),
            random_seed=(
                universe["seed"] if universe["perturb_ohlcv"] else None
            ),
            missing_volume_symbol=(
                "S3" if universe["sparse_missing_volume"] else None
            ),
        )
        query_frame = source.load(InstrumentKey(DATASET, "QUERY"))
        query = build_episode(
            source, InstrumentKey(DATASET, "QUERY"),
            pd.Timestamp(query_frame.timestamp.iloc[-1]), LOOKBACK, "dense-v1",
        )
        benchmark = source.load_benchmark()
        if benchmark is None:
            raise OracleError("reconstructed benchmark is absent")
        provenance = {
            "source_prefixes": {
                symbol: asdict(causal_prefix_digest(
                    source.load(InstrumentKey(DATASET, symbol)), query.key.cutoff,
                )) for symbol in SYMBOLS
            },
            "benchmark_prefix": asdict(causal_prefix_digest(
                benchmark, query.key.cutoff,
            )),
        }
        source_state = {
            "stock_prefixes": provenance["source_prefixes"],
            "benchmark_prefix": provenance["benchmark_prefix"],
            "stock_rows": {
                symbol: len(source.load(InstrumentKey(DATASET, symbol)))
                for symbol in SYMBOLS
            },
            "volume_nan_rows": {
                symbol: int(source.load(
                    InstrumentKey(DATASET, symbol)
                )["volume"].isna().sum())
                for symbol in SYMBOLS
            },
        }
        request = query_state["request"]
        input_state = {
            "query_stock_prefix": asdict(causal_prefix_digest(
                source.load(query.key.instrument), query.key.cutoff,
            )),
            "query_benchmark_prefix": asdict(causal_prefix_digest(
                benchmark, query.key.cutoff,
            )),
            "request": {
                "search_datasets": request["search_datasets"],
                "quality_tiers": request["quality_tiers"],
                "top_k": request["top_k"],
                "cross_dataset": request["cross_dataset"],
                "deduplicate_overlaps": request["deduplicate_overlaps"],
                "max_per_instrument": request["max_per_instrument"],
                "minimum_history_gap_bars": request["minimum_history_gap_bars"],
            },
            "packed_provenance_digest": stable_hash(provenance),
            "query_representation_digest": representation_input_digest(
                represent(query),
            ),
        }
    digest = stable_hash(input_state)
    source_manifest = {
        "state": source_state, "digest": stable_hash(source_state),
    }
    _EXPECTED_UNIVERSE_CACHE[cache_key] = (digest, source_manifest)
    return digest


def _reconstruct_expected_source_manifest(
    query_state: dict[str, Any],
) -> dict[str, Any]:
    cache_key = stable_hash(query_state)
    _reconstruct_expected_input_digest(query_state)
    return _EXPECTED_UNIVERSE_CACHE[cache_key][1]


_EXPECTED_PROPOSAL_CACHE: dict[int, dict[str, Any]] = {}
_EXPECTED_RESULT_CACHE: dict[tuple[int, str], dict[str, Any]] = {}
_EXPECTED_CARDINALITY_CACHE: dict[str, Any] = {}
_EXPECTED_PRIMARY_CACHE: dict[str, Any] = {}


def _reconstruct_expected_property_proposal(seed: int) -> dict[str, Any]:
    if seed in _EXPECTED_PROPOSAL_CACHE:
        return _EXPECTED_PROPOSAL_CACHE[seed]
    with tempfile.TemporaryDirectory(prefix="m04r14-proposal-rebuild-") as temporary:
        fixture = _build_exact_fixture(
            Path(temporary), seed=seed, sampled=True, perturb_ohlcv=True,
            sparse_missing_volume=seed % 2 == 0,
        )
        report = scan_packed_bound_proposals(
            fixture["store"], fixture["generation"], fixture["packed_query"],
            route_quotas={"composite": 1_000}, block_rows=4,
            branch_aware=True, verify_content=True,
        )
        expected = {
            "proposal_result_digest": report.result_digest,
            "generation_id": fixture["generation"],
            "query_episode_id": fixture["query"].key.id,
            "eligible_rows": report.eligible_rows,
        }
    _EXPECTED_PROPOSAL_CACHE[seed] = expected
    return expected


def _reconstruct_expected_property_result(
    seed: int, configuration: str,
) -> dict[str, Any]:
    cache_key = (seed, configuration)
    if cache_key in _EXPECTED_RESULT_CACHE:
        return _EXPECTED_RESULT_CACHE[cache_key]
    with tempfile.TemporaryDirectory(prefix="m04r14-result-rebuild-") as temporary:
        fixture = _build_exact_fixture(
            Path(temporary), seed=seed, sampled=True, perturb_ohlcv=True,
            sparse_missing_volume=seed % 2 == 0,
        )
        request = replace(
            fixture["request"],
            top_k=5 if configuration == "unconstrained" else 3,
            deduplicate_overlaps=configuration == "constrained",
            max_per_instrument=24 if configuration == "unconstrained" else 1,
        )
        proposal = scan_packed_bound_proposals(
            fixture["store"], fixture["generation"], fixture["packed_query"],
            route_quotas={"composite": 1_000}, block_rows=4,
            branch_aware=True, verify_content=True,
        )
        result = certified_packed_search(
            fixture["query"], fixture["source"], request,
            fixture["store"], fixture["generation"], store_dataset_id=DATASET,
            initial_frontier_rows=8, maximum_frontier_rows=18, seed_rows=8,
            block_rows=6, workers=1, sparse_cutoff=3,
            tolerance=NUMERIC_ATOL, verify_content=False,
            requested_positions=True, vector_lower_bounds=True,
            deferred_alignments=True, compact_scored=True,
            native_bound_deferral=True, streaming_threshold_closure=True,
            branch_aware_packed_bounds=True, precomputed_proposal=proposal,
        )
        brute = exact_search(fixture["query"], fixture["brute"], request)
        _assert_match_parity(result.matches, brute)
        expected_eligible = sum(
            eligible(fixture["query"], candidate.episode, request)
            for candidate in fixture["brute"]
        )
        expected = {
            "query": _query_binding(
                fixture["query"], request,
                universe={
                    "seed": seed, "sampled": True, "perturb_ohlcv": True,
                    "sparse_missing_volume": seed % 2 == 0,
                },
            ),
            "source": fixture["source_manifest"],
            "matches": _seal_matches(result.matches),
            "exhaustive_matches_digest": _seal_matches(brute)["digest"],
            "certificate": _seal_certificate(result.certificate),
            "accounting": _accounting_evidence(
                result.certificate, expected_eligible,
            ),
        }
    _EXPECTED_RESULT_CACHE[cache_key] = expected
    return expected


def _reconstruct_expected_cardinality(*, certified: bool) -> dict[str, Any]:
    """Re-run the frozen cardinality fixture instead of trusting mutable seals."""
    cache_key = "certified" if certified else "proposal"
    if cache_key in _EXPECTED_CARDINALITY_CACHE:
        return _EXPECTED_CARDINALITY_CACHE[cache_key]
    with tempfile.TemporaryDirectory(
        prefix=f"m04r14-{cache_key}-cardinality-rebuild-",
    ) as temporary:
        root = Path(temporary)
        expected = (
            _certified_cardinality_checks(root)
            if certified else _cardinality_checks(root)
        )
    _EXPECTED_CARDINALITY_CACHE[cache_key] = expected
    return expected


def _reconstruct_expected_primary() -> dict[str, Any]:
    """Re-run the frozen primary fixture for independent semantic validation."""
    if "primary" in _EXPECTED_PRIMARY_CACHE:
        return _EXPECTED_PRIMARY_CACHE["primary"]
    with tempfile.TemporaryDirectory(prefix="m04r14-primary-rebuild-") as temporary:
        rebuilt = _exact_checks(Path(temporary))
    expected = {
        name: rebuilt[name] for name in (
            "query", "matches", "certificate", "accounting",
            "exhaustive_match_digest",
        )
    }
    _EXPECTED_PRIMARY_CACHE["primary"] = expected
    return expected


def _accounting_evidence(certificate: Any, expected_eligible: int) -> dict[str, Any]:
    native = asdict(certificate.native_bound_accounting)
    state = {
        "claim": (
            "conservation against independently reconstructed eligibility; "
            "exact/native/packed classifications are production-certificate evidence"
        ),
        "independent_eligible": expected_eligible,
        "certificate_eligible": certificate.eligible_candidates,
        "exact_evaluated": certificate.exact_evaluated,
        "safely_pruned": certificate.safely_pruned,
        "native": native,
    }
    if not all((
        expected_eligible == certificate.eligible_candidates,
        certificate.exact_evaluated + certificate.safely_pruned
        == expected_eligible,
        native["native_bound_evaluated"] + native["packed_bound_pruned"]
        == expected_eligible,
        native["exact_dtw_evaluated"] + native["native_bound_pruned"]
        == native["native_bound_evaluated"],
    )):
        raise OracleError("certificate accounting does not conserve eligible rows")
    return {"state": state, "digest": stable_hash(state)}


def _build_exact_fixture(
    root: Path, *, seed: int = 14_001, sampled: bool = False,
    perturb_ohlcv: bool = False, sparse_missing_volume: bool = False,
    eligible_limit: int | None = None, include_overflow: bool = True,
) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    source = _source(
        root, random_seed=seed if perturb_ohlcv else None,
        missing_volume_symbol="S3" if sparse_missing_volume else None,
    )
    query_bars = source.load(InstrumentKey(DATASET, "QUERY"))
    query = build_episode(
        source, InstrumentKey(DATASET, "QUERY"),
        pd.Timestamp(query_bars.timestamp.iloc[-1]), LOOKBACK, "dense-v1",
    )
    request = SearchQuery(
        query.key, (DATASET,), ("A",), 3,
        deduplicate_overlaps=True, max_per_instrument=1,
        minimum_history_gap_bars=20,
    )
    benchmark = source.load_benchmark()
    if benchmark is None:
        raise OracleError("synthetic benchmark is absent")
    query_quantized = quantize_bound_row(represent(query))
    candidate_rows: list[tuple[object, object]] = []
    exact_periodic: list[str] = []
    for symbol_id, symbol in enumerate(SYMBOLS):
        key = InstrumentKey(DATASET, symbol)
        frame = source.load(key)
        positions = list(range(LOOKBACK - 1, 80, 5))
        if sampled:
            random = np.random.default_rng(seed + symbol_id)
            positions = sorted({59, *random.choice(
                [value for value in positions if value != 59],
                size=5, replace=False,
            ).tolist()})
            if sparse_missing_volume and symbol == "S3":
                positions = [59, 79]
        for position in positions:
            cutoff = pd.Timestamp(frame.timestamp.iloc[position])
            episode = build_episode(
                source, key, cutoff, LOOKBACK, "dense-v1", "A",
            )
            record = make_packed_record(
                episode.key.id, int(cutoff.value), symbol_id, "A",
                # A deliberately loose but valid zero lower bound forces the
                # native-deferred and streaming closure paths.
                query_quantized,
            )
            candidate_rows.append((episode, record))
            if position == 59:
                exact_periodic.append(episode.key.id)
    if eligible_limit is not None:
        if type(eligible_limit) is not int or not 0 <= eligible_limit <= len(candidate_rows):
            raise OracleError("eligible limit is outside synthetic universe")
        candidate_rows = candidate_rows[:eligible_limit]
    brute = [SearchCandidate.from_episode(episode) for episode, _ in candidate_rows]
    if include_overflow:
        retained_periodic = [
            episode.key.id for episode, _ in candidate_rows
            if episode.key.id in exact_periodic
        ]
        if not retained_periodic:
            raise OracleError("overflow fixture has no retained periodic episode")
        overflow_id = min(retained_periodic)
        main = [
            record for episode, record in candidate_rows
            if episode.key.id != overflow_id
        ]
        overflow_episode = next(
            episode for episode, _ in candidate_rows if episode.key.id == overflow_id
        )
        overflow = make_overflow_record(
            overflow_episode.key.id, int(overflow_episode.key.cutoff.value),
            SYMBOLS.index(overflow_episode.key.instrument.source_symbol), "A",
        )
    else:
        overflow_id = None
        main = [record for _, record in candidate_rows]
        overflow = np.empty(0, dtype=OVERFLOW_DTYPE)
    provenance = {
        "source_prefixes": {
            symbol: asdict(causal_prefix_digest(
                source.load(InstrumentKey(DATASET, symbol)), query.key.cutoff,
            )) for symbol in SYMBOLS
        },
        "benchmark_prefix": asdict(causal_prefix_digest(
            benchmark, query.key.cutoff,
        )),
    }
    source_state = {
        "stock_prefixes": provenance["source_prefixes"],
        "benchmark_prefix": provenance["benchmark_prefix"],
        "stock_rows": {
            symbol: len(source.load(InstrumentKey(DATASET, symbol)))
            for symbol in SYMBOLS
        },
        "volume_nan_rows": {
            symbol: int(source.load(
                InstrumentKey(DATASET, symbol)
            )["volume"].isna().sum())
            for symbol in SYMBOLS
        },
    }
    source_manifest = {
        "state": source_state, "digest": stable_hash(source_state),
    }
    store = root / "exact-store"
    generation = write_packed_generation(
        store, np.concatenate(main) if main else np.empty(0, dtype=PACK_DTYPE),
        overflow,
        SYMBOLS, provenance, activate=False,
    )
    packed_query = PackedBoundQuery(
        query.key.id, query.key.instrument.source_symbol,
        int(query.bars.timestamp.iloc[0].value),
        int(latest_eligible_cutoff(
            query, request.minimum_history_gap_bars,
        ).value),
        represent(query), request.quality_tiers,
    )
    return {
        "source": source, "query": query, "request": request,
        "brute": brute, "store": store, "generation": generation,
        "packed_query": packed_query, "overflow_id": overflow_id,
        "source_manifest": source_manifest,
    }


def _cardinality_checks(root: Path) -> dict[str, Any]:
    representation = represent(generate_case("rounded_base", 140_001).episode)
    other = represent(generate_case("steady_trend", 140_002).episode)
    rows = [make_packed_record(
        f"{index + 1:024x}", index + 1, 0, "A",
        quantize_bound_row(other if index == 24 else representation),
    ) for index in range(25)]
    store = root / "cardinality-store"
    generation = write_packed_generation(
        store, np.concatenate(rows), np.empty(0, dtype=OVERFLOW_DTYPE),
        ("CAND",), {"purpose": "m04r14-cardinality"}, activate=False,
    )

    def query(latest: int) -> PackedBoundQuery:
        return PackedBoundQuery(
            f"{900_000 + latest:024x}", "QUERY", 0, latest,
            representation, ("A",),
        )

    counts = {}
    for expected in (0, 19, 20, 25):
        report = scan_packed_bound_proposals(
            store, generation, query(expected), route_quotas={"composite": 1_000},
            branch_aware=True, block_rows=7, verify_content=False,
        )
        if report.eligible_rows != expected:
            raise OracleError(f"eligible cardinality {expected} differs")
        counts[str(expected)] = report.result_digest

    full_query = query(25)
    scalar = scan_packed_bound_proposals(
        store, generation, full_query, route_quotas={"composite": 1_000},
        branch_aware=True, block_rows=6, verify_content=False,
    )
    for block, order, threads in (
        (1, "forward", 1), (4, "forward", 2), (9, "reverse", 4),
    ):
        threaded = scan_packed_bound_proposals_threaded(
            store, generation, full_query,
            route_quotas={"composite": 1_000}, branch_aware=True,
            block_rows=block, block_order=order, threads=threads,
            verify_content=False,
        )
        if _semantic_report(threaded) != _semantic_report(scalar):
            raise OracleError("scalar/threaded traversal parity differs")

    positive = next(row for row in scalar.candidates if row.lower_bound > 0)
    admitted_at_upper: list[str] = []
    at_upper = scan_packed_bound_threshold(
        store, generation, full_query, upper_inclusive=positive.lower_bound,
        lower_exclusive=float(np.nextafter(positive.lower_bound, -np.inf)),
        consume=lambda values: admitted_at_upper.extend(row.episode_id for row in values),
        branch_aware=True, block_rows=8, verify_content=False,
    )
    excluded_at_lower: list[str] = []
    above = float(np.nextafter(positive.lower_bound, np.inf))
    scan_packed_bound_threshold(
        store, generation, full_query, lower_exclusive=positive.lower_bound,
        upper_inclusive=above,
        consume=lambda values: excluded_at_lower.extend(row.episode_id for row in values),
        branch_aware=True, block_rows=5, block_order="reverse",
        verify_content=False,
    )
    if positive.episode_id not in admitted_at_upper or positive.episode_id in excluded_at_lower:
        raise OracleError("threshold interval is not (lower, upper]")
    try:
        duplicate_id = f"{777:024x}"
        write_packed_generation(
            root / "duplicate-store", np.concatenate((
                make_packed_record(
                    duplicate_id, 1, 0, "A", quantize_bound_row(representation),
                ),
                make_packed_record(
                    duplicate_id, 2, 0, "A", quantize_bound_row(representation),
                ),
            )),
            np.empty(0, dtype=OVERFLOW_DTYPE), ("CAND",),
            {"purpose": "must-reject-duplicate"}, activate=False,
        )
    except PackedBoundStoreError:
        duplicate_rejected = 1
    else:
        raise OracleError("duplicate episode/symbol/cutoff row was accepted")
    return {
        "eligible_result_digests": counts,
        "parity_result_digest": scalar.result_digest,
        "upper_boundary_episode_id": positive.episode_id,
        "upper_boundary_scan_digest": at_upper.result_digest,
        "duplicate_id_rejected": duplicate_rejected,
    }


def _certified_cardinality_checks(root: Path) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for count in (0, 19, 20):
        fixture = _build_exact_fixture(
            root / f"certified-cardinality-{count}", seed=14_200,
            sampled=True, perturb_ohlcv=True, sparse_missing_volume=True,
            eligible_limit=count, include_overflow=False,
        )
        request = replace(
            fixture["request"], top_k=20, deduplicate_overlaps=False,
            max_per_instrument=20,
        )
        proposal = scan_packed_bound_proposals_threaded(
            fixture["store"], fixture["generation"], fixture["packed_query"],
            route_quotas={"composite": 1_000}, block_rows=5, threads=2,
            branch_aware=True, verify_content=False,
        )
        if proposal.eligible_rows != count:
            raise OracleError("certified cardinality eligibility differs")
        try:
            result = certified_packed_search(
                fixture["query"], fixture["source"], request,
                fixture["store"], fixture["generation"],
                store_dataset_id=DATASET, initial_frontier_rows=20,
                maximum_frontier_rows=20, seed_rows=20, block_rows=5,
                workers=1, sparse_cutoff=3, tolerance=NUMERIC_ATOL,
                verify_content=False, requested_positions=True,
                vector_lower_bounds=True, deferred_alignments=True,
                compact_scored=True, native_bound_deferral=True,
                streaming_threshold_closure=True,
                branch_aware_packed_bounds=True,
                precomputed_proposal=proposal,
            )
        except CertifiedPackedSearchError as exc:
            if count >= 20 or "cannot fill constrained top-k" not in str(exc):
                raise OracleError("certified cardinality failed unexpectedly") from exc
            rows[str(count)] = {
                "eligible_rows": count, "status": "underfill-rejected",
                "proposal_result_digest": proposal.result_digest,
            }
            continue
        if count != 20:
            raise OracleError("certified underfill was accepted")
        brute = exact_search(fixture["query"], fixture["brute"], request)
        _assert_match_parity(result.matches, brute)
        rows[str(count)] = {
            "eligible_rows": count, "status": "certified",
            "proposal_result_digest": proposal.result_digest,
            "matches": _seal_matches(result.matches),
            "certificate": _seal_certificate(result.certificate),
        }
    state = {"rows": rows}
    return {"state": state, "digest": stable_hash(state)}


def _assert_match_parity(actual: Sequence[Any], expected: Sequence[Any]) -> None:
    if [row.episode_key.id for row in actual] != [
        row.episode_key.id for row in expected
    ]:
        raise OracleError("certified ordered IDs differ from exhaustive search")
    for left, right in zip(actual, expected, strict=True):
        if not np.isclose(
            left.total_distance, right.total_distance, rtol=0.0,
            atol=NUMERIC_ATOL,
        ):
            raise OracleError("certified distance differs from exhaustive search")
        if left.component_distances.keys() != right.component_distances.keys():
            raise OracleError("certified component schema differs")
        if not np.allclose(
            list(left.component_distances.values()),
            list(right.component_distances.values()), rtol=0.0,
            atol=NUMERIC_ATOL,
        ):
            raise OracleError("certified components differ from exhaustive search")
        if left.alignment != right.alignment:
            raise OracleError("certified alignment differs from exhaustive search")


def _max_per_instrument_underfill_completion_check() -> dict[str, Any]:
    first_instrument = InstrumentKey(DATASET, "CAP-A")
    second_instrument = InstrumentKey(DATASET, "CAP-B")
    query_key = EpisodeKey(
        InstrumentKey(DATASET, "CAP-QUERY"), pd.Timestamp("2025-01-31"),
        LOOKBACK, "dense-v1",
    )
    request = SearchQuery(
        query_key, (DATASET,), ("A",), 2,
        deduplicate_overlaps=False, max_per_instrument=1,
    )

    def compact(
        instrument: InstrumentKey, label: str, distance: float, start: int,
    ) -> CompactScoredCandidate:
        key = EpisodeKey(
            instrument, pd.Timestamp("2020-01-01") + pd.Timedelta(days=start + 9),
            10, label,
        )
        return CompactScoredCandidate(
            AnalogueMatch(key, distance, {"price": distance}),
            instrument, start, start + 9,
        )

    cap_first = compact(first_instrument, "cap-first", 0.1, 0)
    cap_rejected = compact(first_instrument, "cap-rejected", 0.2, 20)
    reopened = compact(second_instrument, "reopened", 100.0, 40)
    deferred = {
        reopened.match.episode_key.id: type("Deferred", (), {
            "lower_bound": 99.0,
            "episode_key": reopened.match.episode_key,
        })(),
    }
    completed: list[str] = []

    def complete(values: Sequence[Any]) -> list[CompactScoredCandidate]:
        completed.extend(row.episode_key.id for row in values)
        return [reopened]

    selected, threshold, count = _close_native_bound_deferred(
        {
            cap_first.match.episode_key.id: cap_first,
            cap_rejected.match.episode_key.id: cap_rejected,
        },
        deferred, request, compact=True, complete_many=complete,
    )
    if (
        count != 1 or completed != [reopened.match.episode_key.id]
        or len(selected) != 2 or threshold != 100.0 or deferred
    ):
        raise OracleError("max-per-instrument underfill completion branch differs")
    return {
        "completed_rows": count,
        "initial_threshold_state": "incomplete-infinity",
        "raw_second_bound_hex": float(99.0).hex(),
        "resulting_threshold_hex": threshold.hex(),
    }


def _property_matrix(root: Path) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    seeds = (14_101, 14_102, 14_103, 14_104)
    for seed in seeds:
        fixture = _build_exact_fixture(
            root / f"property-{seed}", seed=seed, sampled=True,
            perturb_ohlcv=True, sparse_missing_volume=(seed % 2 == 0),
        )
        base_request = fixture["request"]
        configurations = (
            ("unconstrained", replace(
                base_request, top_k=5, deduplicate_overlaps=False,
                max_per_instrument=24,
            )),
            ("constrained", replace(
                base_request, top_k=3, deduplicate_overlaps=True,
                max_per_instrument=1,
            )),
        )
        reports = [
            scan_packed_bound_proposals(
                fixture["store"], fixture["generation"], fixture["packed_query"],
                route_quotas={"composite": 1_000}, block_rows=block,
                block_order=order, branch_aware=True, verify_content=False,
            )
            for block, order in ((4, "forward"), (9, "reverse"))
        ]
        reports.extend(
            scan_packed_bound_proposals_threaded(
                fixture["store"], fixture["generation"], fixture["packed_query"],
                route_quotas={"composite": 1_000}, block_rows=block,
                block_order=order, threads=threads, branch_aware=True,
                verify_content=False,
            )
            for block, order, threads in (
                (5, "forward", 1), (7, "forward", 3),
                (6, "reverse", 1), (8, "reverse", 3),
            )
        )
        if any(
            _semantic_report(report) != _semantic_report(reports[0])
            for report in reports[1:]
        ):
            raise OracleError("property scalar/thread/order parity differs")
        forward, reverse = reports[2], reports[-1]
        for label, request in configurations:
            results = []
            for order, workers, proposal in (
                ("forward", 1, forward), ("forward", 2, forward),
                ("reverse", 1, reverse), ("reverse", 2, reverse),
            ):
                results.append(certified_packed_search(
                    fixture["query"], fixture["source"], request,
                    fixture["store"], fixture["generation"],
                    store_dataset_id=DATASET, initial_frontier_rows=8,
                    maximum_frontier_rows=18, seed_rows=8, block_rows=6,
                    workers=workers, sparse_cutoff=3, verify_content=False,
                    tolerance=NUMERIC_ATOL,
                    requested_positions=True, vector_lower_bounds=True,
                    deferred_alignments=True, compact_scored=True,
                    native_bound_deferral=True,
                    streaming_threshold_closure=True,
                    branch_aware_packed_bounds=True,
                    threshold_scan_block_order=order,
                    precomputed_proposal=proposal,
                ))
            brute = exact_search(fixture["query"], fixture["brute"], request)
            expected_eligible = sum(
                eligible(fixture["query"], candidate.episode, request)
                for candidate in fixture["brute"]
            )
            for result in results:
                _assert_match_parity(result.matches, brute)
                certificate = result.certificate
                _accounting_evidence(certificate, expected_eligible)
                _strict_json(asdict(certificate))
            if len({row.certificate.result_digest for row in results}) != 1:
                raise OracleError("property threshold traversal result differs")
            certificate_seal = _seal_certificate(results[0].certificate)
            match_seal = _seal_matches(results[0].matches)
            accounting_seal = _accounting_evidence(
                results[0].certificate, expected_eligible,
            )
            query_seal = _query_binding(
                fixture["query"], request,
                universe={
                    "seed": seed, "sampled": True, "perturb_ohlcv": True,
                    "sparse_missing_volume": seed % 2 == 0,
                },
            )
            execution_state = {
                "initial_frontier_rows": 8,
                "maximum_frontier_rows": 18,
                "seed_rows": 8,
                "proposal_quota": 1_000,
                "proposal_traversals": [
                    ["scalar", "forward", 4, 1],
                    ["scalar", "reverse", 9, 1],
                    ["threaded", "forward", 5, 1],
                    ["threaded", "forward", 7, 3],
                    ["threaded", "reverse", 6, 1],
                    ["threaded", "reverse", 8, 3],
                ],
                "certified_workers": [1, 2],
                "certified_traversal_worker_cross_product": [
                    ["forward", 1], ["forward", 2],
                    ["reverse", 1], ["reverse", 2],
                ],
                "numeric_tolerance_hex": NUMERIC_ATOL.hex(),
            }
            execution_seal = {
                "state": execution_state, "digest": stable_hash(execution_state),
            }
            frontiers = [
                row.frontier_rows for row in results[0].certificate.rounds
            ]
            repeated_frontiers = len(frontiers) - len(set(frontiers))
            cases.append({
                "seed": seed, "configuration": label,
                "eligible_rows": results[0].certificate.eligible_candidates,
                "sparse_missing_volume": seed % 2 == 0,
                "proposal_scalar_forward_result_digest": reports[0].result_digest,
                "proposal_scalar_reverse_result_digest": reports[1].result_digest,
                "proposal_threaded_forward_result_digest": forward.result_digest,
                "proposal_threaded_reverse_result_digest": reverse.result_digest,
                "query": query_seal,
                "source": fixture["source_manifest"],
                "execution": execution_seal,
                "matches": match_seal,
                "exhaustive_matches_digest": _seal_matches(brute)["digest"],
                "certificate": certificate_seal,
                "accounting": accounting_seal,
                "certificate_result_digest": results[0].certificate.result_digest,
                "closure_passes": len(
                    results[0].certificate.threshold_closure_passes
                ),
                "repeated_frontier_rounds": repeated_frontiers,
            })
    return {
        "fixed_seeds": list(seeds),
        "cases": cases,
        "case_digest": stable_hash(cases),
    }


def _exact_checks(root: Path) -> dict[str, Any]:
    fixture = _build_exact_fixture(root)
    proposal = scan_packed_bound_proposals_threaded(
        fixture["store"], fixture["generation"], fixture["packed_query"],
        route_quotas={"composite": 1_000}, block_rows=7, threads=2,
        branch_aware=True, verify_content=False,
    )
    reverse = scan_packed_bound_proposals_threaded(
        fixture["store"], fixture["generation"], fixture["packed_query"],
        route_quotas={"composite": 1_000}, block_rows=11, block_order="reverse",
        threads=4, branch_aware=True, verify_content=False,
    )
    if _semantic_report(proposal) != _semantic_report(reverse):
        raise OracleError("adversarial proposal traversal parity differs")
    result = certified_packed_search(
        fixture["query"], fixture["source"], fixture["request"],
        fixture["store"], fixture["generation"], store_dataset_id=DATASET,
        initial_frontier_rows=30, maximum_frontier_rows=30, seed_rows=30,
        block_rows=9, workers=2, sparse_cutoff=3, verify_content=False,
        tolerance=NUMERIC_ATOL,
        requested_positions=True, vector_lower_bounds=True,
        deferred_alignments=True, compact_scored=True,
        native_bound_deferral=True, streaming_threshold_closure=True,
        branch_aware_packed_bounds=True, precomputed_proposal=proposal,
    )
    reverse_result = certified_packed_search(
        fixture["query"], fixture["source"], fixture["request"],
        fixture["store"], fixture["generation"], store_dataset_id=DATASET,
        initial_frontier_rows=30, maximum_frontier_rows=30, seed_rows=30,
        block_rows=13, workers=1, sparse_cutoff=3, verify_content=False,
        tolerance=NUMERIC_ATOL,
        requested_positions=True, vector_lower_bounds=True,
        deferred_alignments=True, compact_scored=True,
        native_bound_deferral=True, streaming_threshold_closure=True,
        branch_aware_packed_bounds=True, threshold_scan_block_order="reverse",
        precomputed_proposal=proposal,
    )
    brute = exact_search(fixture["query"], fixture["brute"], fixture["request"])
    _assert_match_parity(result.matches, brute)
    expected = [(row.episode_key.id, row.total_distance) for row in brute]
    if reverse_result.certificate.result_digest != result.certificate.result_digest:
        raise OracleError("threshold traversal changes certified result")
    certificate = result.certificate
    if not certificate.threshold_closure_passes:
        raise OracleError("synthetic fixture did not force threshold closure")
    if certificate.native_bound_accounting.native_bound_pruned <= 0:
        raise OracleError("synthetic fixture did not force native deferral")
    if fixture["overflow_id"] not in {row.episode_key.id for row in result.matches}:
        raise OracleError("overflow fallback was not retained")
    retained_main = {
        row.episode_key.id for row in result.matches
    } - {fixture["overflow_id"]}
    if not retained_main:
        raise OracleError("no main-pack match was retained")
    if len({row.total_distance for row in result.matches}) != 1:
        raise OracleError("K-boundary exact ties were not forced")
    per_symbol: dict[str, int] = {}
    for row in result.matches:
        symbol = row.episode_key.instrument.source_symbol
        per_symbol[symbol] = per_symbol.get(symbol, 0) + 1
    if len(per_symbol) != 3 or sorted(per_symbol.values()) != [1, 1, 1]:
        raise OracleError("max-per-symbol constraint was not forced")
    # More exact zero-distance rows exist than can be retained, proving the
    # K-boundary tie, symbol cap, and overlap-aware constrained selection.
    relaxed = replace(
        fixture["request"], top_k=len(fixture["brute"]),
        deduplicate_overlaps=False, max_per_instrument=len(fixture["brute"]),
    )
    unconstrained = exact_search(fixture["query"], fixture["brute"], relaxed)
    dedup_only = exact_search(
        fixture["query"], fixture["brute"],
        replace(relaxed, deduplicate_overlaps=True),
    )
    overlap_rejected = len(unconstrained) - len(dedup_only)
    if overlap_rejected <= 0:
        raise OracleError("overlap deduplication was not forced")
    tied = sum(
        abs(row.total_distance - result.matches[-1].total_distance) <= NUMERIC_ATOL
        for row in unconstrained
    )
    if tied <= fixture["request"].top_k:
        raise OracleError("constrained K-boundary tie was not forced")
    capped = exact_search(
        fixture["query"], fixture["brute"],
        replace(relaxed, max_per_instrument=1),
    )
    cap_rejected = len(unconstrained) - len(capped)
    if cap_rejected <= 0:
        raise OracleError("max-per-instrument rejection was not forced")
    state = asdict(certificate)
    _strict_json(state)
    tampered_candidate = replace(
        proposal.candidates[0], lower_bound=proposal.candidates[0].lower_bound + 1.0,
    )
    tampered = replace(
        proposal, candidates=(tampered_candidate, *proposal.candidates[1:]),
    )
    try:
        certified_packed_search(
            fixture["query"], fixture["source"], fixture["request"],
            fixture["store"], fixture["generation"], store_dataset_id=DATASET,
            initial_frontier_rows=30, maximum_frontier_rows=30, seed_rows=30,
            tolerance=NUMERIC_ATOL,
            precomputed_proposal=tampered,
            branch_aware_packed_bounds=True,
        )
    except CertifiedPackedSearchError:
        mutation_rejected = True
    else:
        mutation_rejected = False
    if not mutation_rejected:
        raise OracleError("mutated proposal report was accepted")

    underfill_hits = {"raw": 0, "constraint": 0}
    underfill_requests = (
        ("raw", replace(
            fixture["request"], top_k=37,
            deduplicate_overlaps=False, max_per_instrument=37,
        ), 37),
        ("constraint", replace(
            fixture["request"], top_k=5,
            deduplicate_overlaps=False, max_per_instrument=1,
        ), 30),
    )
    for label, request, frontier in underfill_requests:
        try:
            certified_packed_search(
                fixture["query"], fixture["source"], request,
                fixture["store"], fixture["generation"],
                store_dataset_id=DATASET, initial_frontier_rows=frontier,
                maximum_frontier_rows=frontier, seed_rows=frontier,
                block_rows=10, workers=1, sparse_cutoff=3,
                tolerance=NUMERIC_ATOL,
                verify_content=False, requested_positions=True,
                vector_lower_bounds=True, deferred_alignments=True,
                compact_scored=True, native_bound_deferral=True,
                streaming_threshold_closure=True,
                branch_aware_packed_bounds=True,
                precomputed_proposal=proposal,
            )
        except CertifiedPackedSearchError as exc:
            if "cannot fill constrained top-k" not in str(exc):
                raise OracleError(f"{label} underfill failed for another reason") from exc
            underfill_hits[label] += 1
        else:
            raise OracleError(f"{label} underfill did not fail closed")
    expected_eligible = sum(
        eligible(fixture["query"], candidate.episode, fixture["request"])
        for candidate in fixture["brute"]
    )
    match_seal = _seal_matches(result.matches)
    certificate_seal = _seal_certificate(certificate)
    accounting_seal = _accounting_evidence(certificate, expected_eligible)
    query_seal = _query_binding(
        fixture["query"], fixture["request"],
        universe={
            "seed": 14_001, "sampled": False, "perturb_ohlcv": False,
            "sparse_missing_volume": False,
        },
    )

    source_path = root / "bars" / "S0.parquet"
    changed_source = pd.read_parquet(source_path)
    changed_source.loc[0, "close"] = float(changed_source.loc[0, "close"]) + 1.0
    source_replacement = source_path.with_name(".S0.replacement.parquet")
    changed_source.to_parquet(source_replacement, index=False)
    os.replace(source_replacement, source_path)
    mutated_source = DirectorySource(fixture["source"].spec)
    try:
        certified_packed_search(
            fixture["query"], mutated_source, fixture["request"],
            fixture["store"], fixture["generation"], store_dataset_id=DATASET,
            initial_frontier_rows=30, maximum_frontier_rows=30, seed_rows=30,
            tolerance=NUMERIC_ATOL,
            verify_content=False, requested_positions=True,
            vector_lower_bounds=True, deferred_alignments=True,
            compact_scored=True, native_bound_deferral=True,
            streaming_threshold_closure=True,
            branch_aware_packed_bounds=True, precomputed_proposal=proposal,
        )
    except CertifiedPackedSearchError as exc:
        if "causal prefix is stale" not in str(exc):
            raise OracleError("source mutation failed for another reason") from exc
        source_mutation_rejected = 1
    else:
        raise OracleError("mutated source prefix was accepted")

    manifest = json.loads((
        fixture["store"] / "generations" / fixture["generation"] / "manifest.json"
    ).read_text())
    rows_path = (
        fixture["store"] / "generations" / fixture["generation"]
        / manifest["rows_file"]
    )
    row_bytes = bytearray(rows_path.read_bytes())
    row_bytes[0] ^= 1
    rows_replacement = rows_path.with_name(f".{rows_path.name}.replacement")
    with rows_replacement.open("xb") as handle:
        handle.write(row_bytes)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(rows_replacement, rows_path)
    try:
        scan_packed_bound_proposals(
            fixture["store"], fixture["generation"], fixture["packed_query"],
            route_quotas={"composite": 1_000}, branch_aware=True,
            verify_content=True,
        )
    except (PackedBoundSearchError, PackedBoundStoreError):
        store_mutation_rejected = 1
    else:
        raise OracleError("mutated packed-store bytes were accepted")
    return {
        "eligible_rows": certificate.eligible_candidates,
        "retained_main": len(retained_main), "retained_overflow": 1,
        "top_k": len(result.matches), "tied_at_k": tied,
        "overlap_rejected": overlap_rejected,
        "cap_rejected": cap_rejected,
        "per_symbol": per_symbol,
        "closure_passes": len(certificate.threshold_closure_passes),
        "native_bound_pruned": certificate.native_bound_accounting.native_bound_pruned,
        "forward_reverse_result_digest": certificate.result_digest,
        "exhaustive_match_digest": stable_hash(expected),
        "mutation_rejected": mutation_rejected,
        "source_mutation_rejected": source_mutation_rejected,
        "store_mutation_rejected": store_mutation_rejected,
        "underfill_rejections": underfill_hits,
        "query": query_seal,
        "matches": match_seal,
        "certificate": certificate_seal,
        "accounting": accounting_seal,
        "strict_finite_json": True,
    }


MUTATION_KINDS = (
    "proposal", "proposal-all-coordinated", "certificate",
    "certificate-result-coordinated", "certificate-input-coordinated",
    "round", "closure", "accounting", "query",
    "query-gap-coordinated", "matches", "match-distance-coordinated",
    "cardinality-coordinated", "primary-coordinated",
)


def build_oracle_payload(
    work_root: Path, runtime_manifest: dict[str, Any],
) -> dict[str, Any]:
    work_root.mkdir(parents=True, exist_ok=False)
    cardinality = _cardinality_checks(work_root)
    certified_cardinality = _certified_cardinality_checks(work_root)
    exact = _exact_checks(work_root)
    properties = _property_matrix(work_root)
    reopening = _max_per_instrument_underfill_completion_check()
    branch_hits = {
        "overflow_retained": exact["retained_overflow"],
        "threshold_closure": exact["closure_passes"],
        "native_deferral": exact["native_bound_pruned"],
        "max_per_instrument_cap": exact["cap_rejected"],
        "overlap_dedup": exact["overlap_rejected"],
        "k_boundary_ties": exact["tied_at_k"] - exact["top_k"],
        "raw_underfill_rejected": exact["underfill_rejections"]["raw"],
        "constraint_underfill_rejected": (
            exact["underfill_rejections"]["constraint"]
        ),
        "max_per_completion_from_underfill": reopening["completed_rows"],
        "property_matrix_cases": len(properties["cases"]),
        "sparse_symbol_groups": sum(
            row["certificate"]["state"]["sparse_symbols"]
            for row in properties["cases"]
        ),
        "repeated_frontier_rounds": sum(
            row["repeated_frontier_rounds"] for row in properties["cases"]
        ),
        "duplicate_id_rejected": cardinality["duplicate_id_rejected"],
        "certified_cardinality_cases": len(certified_cardinality["state"]["rows"]),
        "source_mutation_rejected": exact["source_mutation_rejected"],
        "store_mutation_rejected": exact["store_mutation_rejected"],
        "rehashed_mutations_rejected": len(MUTATION_KINDS),
    }
    if any(type(value) is not int or value <= 0 for value in branch_hits.values()):
        raise OracleError("a required adversarial branch was not hit")
    deterministic = {
        "schema_version": SCHEMA,
        "truth_free": True,
        "real_forward_outcomes_accessed": False,
        "runtime_manifest": runtime_manifest,
        "output_contract": {
            "repo_relative_path": str(CANONICAL_OUTPUT),
            "create_only": True,
            "protected_repo_roots": [
                ".git", "src", "experiments",
                "config/data/analogues/m04r10/nasdaq-untouched-authority-registry",
                "config/data/analogues/m04r11", "config/data/analogues/m04r13",
                "config/data/analogues/poc/m04r/packed-bound-full",
            ],
            "protected_resident_root": "/dev/shm/market-analogues/m04r11-candidate-v2",
        },
        "numeric_comparison": {
            "ordered_ids_metadata_alignments": "exact",
            "distance_components": "absolute-tolerance",
            "absolute_tolerance_hex": NUMERIC_ATOL.hex(),
            "reason": (
                "exhaustive and deferred/vector scorers are mathematically equal "
                "but use different floating evaluation paths"
            ),
        },
        "cardinality_and_boundaries": cardinality,
        "certified_cardinality": certified_cardinality,
        "certified_vs_exhaustive": exact,
        "randomized_property_matrix": properties,
        "max_per_instrument_underfill_completion": reopening,
        "branch_hits": branch_hits,
        "rehashed_mutation_kinds": list(MUTATION_KINDS),
        "passed": True,
    }
    deterministic["result_digest"] = stable_hash(deterministic)
    _strict_json(deterministic)
    validate_payload(deterministic, require_production=False)
    _exercise_rehashed_mutations(deterministic)
    return deterministic


def _is_hex(value: Any, length: int = 64) -> bool:
    if type(value) is not str or len(value) != length or value.lower() != value:
        return False
    try:
        return len(bytes.fromhex(value)) == length // 2
    except ValueError:
        return False


def _finite_hex(value: Any) -> bool:
    if type(value) is not str:
        return False
    try:
        return bool(np.isfinite(float.fromhex(value)))
    except ValueError:
        return False


def _validate_seal(value: Any, label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != {"state", "digest"}:
        raise OracleError(f"{label} seal schema differs")
    if not _is_hex(value["digest"]) or value["digest"] != stable_hash(value["state"]):
        raise OracleError(f"{label} seal digest differs")
    if type(value["state"]) is not dict:
        raise OracleError(f"{label} seal state differs")
    return value["state"]


def _validate_runtime_manifest(value: Any, *, require_production: bool) -> None:
    state = _validate_seal(value, "runtime manifest")
    if set(state) != {
        "mode", "git_head", "files", "contracts", "environment",
    }:
        raise OracleError("runtime manifest keys differ")
    if require_production and state["mode"] != "production-clean-head":
        raise OracleError("validation requires a production runtime manifest")
    if state["mode"] not in {"production-clean-head", "test-injected"}:
        raise OracleError("runtime manifest mode differs")
    if not _is_hex(state["git_head"], 40):
        raise OracleError("runtime Git head differs")
    if type(state["files"]) is not dict or not state["files"] or any(
        type(name) is not str or not _is_hex(digest)
        for name, digest in state["files"].items()
    ):
        raise OracleError("runtime file manifest differs")
    if state["contracts"] != _contract_binding():
        raise OracleError("runtime production contracts differ")
    if state["environment"] != _environment_binding():
        raise OracleError("runtime environment differs")
    if require_production and set(state["files"]) != {
        str(path) for path in RUNTIME_FILES
    }:
        raise OracleError("production runtime file set differs")
    if require_production:
        _validate_production_manifest_against_repo(
            value, Path(__file__).resolve().parents[2],
        )


def _validate_query_seal(value: Any) -> dict[str, Any]:
    state = _validate_seal(value, "query")
    if set(state) != {
        "query_episode_id", "query_dataset_id", "query_symbol", "query_cutoff", "lookback",
        "representation_version", "universe", "request",
    } or not _is_hex(state["query_episode_id"], 24):
        raise OracleError("query binding schema differs")
    request = state["request"]
    universe = state["universe"]
    if type(universe) is not dict or set(universe) != {
        "seed", "sampled", "perturb_ohlcv", "sparse_missing_volume",
    } or not all((
        type(universe["seed"]) is int,
        type(universe["sampled"]) is bool,
        type(universe["perturb_ohlcv"]) is bool,
        type(universe["sparse_missing_volume"]) is bool,
    )):
        raise OracleError("query universe binding differs")
    if type(request) is not dict or set(request) != {
        "search_datasets", "quality_tiers", "top_k", "cross_dataset",
        "deduplicate_overlaps", "max_per_instrument",
        "minimum_history_gap_bars",
    }:
        raise OracleError("query request binding differs")
    if not all((
        type(request["search_datasets"]) is list,
        type(request["quality_tiers"]) is list,
        type(request["top_k"]) is int and request["top_k"] > 0,
        type(request["cross_dataset"]) is bool,
        type(request["deduplicate_overlaps"]) is bool,
        type(request["max_per_instrument"]) is int
        and request["max_per_instrument"] > 0,
        type(request["minimum_history_gap_bars"]) is int,
    )):
        raise OracleError("query request types differ")
    try:
        reconstructed = EpisodeKey(
            InstrumentKey(state["query_dataset_id"], state["query_symbol"]),
            pd.Timestamp(state["query_cutoff"]), state["lookback"],
            state["representation_version"],
        )
    except Exception as exc:
        raise OracleError("query identity metadata differs") from exc
    if reconstructed.id != state["query_episode_id"]:
        raise OracleError("query identity does not reconstruct")
    return state


def _validate_match_seal(value: Any) -> list[dict[str, Any]]:
    state = _validate_seal(value, "matches")
    if set(state) != {"rows"} or type(state["rows"]) is not list:
        raise OracleError("match seal state differs")
    seen = set()
    order: list[tuple[float, str]] = []
    for row in state["rows"]:
        if type(row) is not dict or set(row) != {
            "episode_id", "dataset_id", "symbol", "cutoff", "lookback",
            "representation_version", "total_distance_hex",
            "component_distances_hex", "alignment", "quality_tier",
            "quality_issues",
        }:
            raise OracleError("sealed match schema differs")
        try:
            key = EpisodeKey(
                InstrumentKey(row["dataset_id"], row["symbol"]),
                pd.Timestamp(row["cutoff"]), row["lookback"],
                row["representation_version"],
            )
        except Exception as exc:
            raise OracleError("sealed match metadata differs") from exc
        if key.id != row["episode_id"] or key.id in seen:
            raise OracleError("sealed match identity differs or repeats")
        seen.add(key.id)
        if not _finite_hex(row["total_distance_hex"]):
            raise OracleError("sealed match total differs")
        order.append((float.fromhex(row["total_distance_hex"]), row["episode_id"]))
        components = row["component_distances_hex"]
        if type(components) is not dict or not components or any(
            type(name) is not str or not _finite_hex(distance)
            for name, distance in components.items()
        ):
            raise OracleError("sealed match components differ")
        if type(row["alignment"]) is not list or any(
            type(pair) is not list or len(pair) != 2
            or any(type(index) is not int or index < 0 for index in pair)
            for pair in row["alignment"]
        ):
            raise OracleError("sealed match alignment differs")
        if row["quality_tier"] not in {"A", "B"} or type(row["quality_issues"]) is not list:
            raise OracleError("sealed match quality differs")
    if order != sorted(order):
        raise OracleError("sealed match order differs")
    return state["rows"]


def _validate_certificate_seal(
    value: Any, *, query_id: str, matches: list[dict[str, Any]],
) -> dict[str, Any]:
    match_count = len(matches)
    state = _validate_seal(value, "certificate")
    required = {
        "schema_version", "contract_digest", "generation_id",
        "query_episode_id", "input_digest", "eligible_candidates",
        "exact_evaluated", "safely_pruned", "stopped_early",
        "stop_threshold", "next_lower_bound", "maximum_quantized_bound_excess",
        "materialization_groups", "sparse_symbols", "batch_symbols", "rounds",
        "result_digest", "native_bound_accounting",
        "minimum_native_pruned_bound", "threshold_closure_passes",
    }
    if set(state) != required or state["query_episode_id"] != query_id:
        raise OracleError("certificate schema/query differs")
    if state["schema_version"] != "m04r-certified-packed-search-v8" or any(
        not _is_hex(state[name])
        for name in ("contract_digest", "generation_id", "input_digest", "result_digest")
    ):
        raise OracleError("certificate contract/digests differ")
    if state["contract_digest"] != _contract_binding()["certified_v8"]:
        raise OracleError("certificate contract is not frozen v8")
    for name in (
        "eligible_candidates", "exact_evaluated", "safely_pruned",
        "materialization_groups", "sparse_symbols", "batch_symbols",
    ):
        if type(state[name]) is not int or state[name] < 0:
            raise OracleError("certificate integer accounting differs")
    if state["exact_evaluated"] + state["safely_pruned"] != state["eligible_candidates"]:
        raise OracleError("certificate top-level accounting differs")
    if type(state["stopped_early"]) is not bool or not _finite_hex(state["stop_threshold"]):
        raise OracleError("certificate terminal state differs")
    if state["next_lower_bound"] is not None and not _finite_hex(state["next_lower_bound"]):
        raise OracleError("certificate next bound differs")
    if not _finite_hex(state["maximum_quantized_bound_excess"]):
        raise OracleError("certificate quantized excess differs")
    rounds = state["rounds"]
    closures = state["threshold_closure_passes"]
    if type(rounds) is not list or not rounds or type(closures) is not list:
        raise OracleError("certificate round/closure state differs")
    for row in rounds:
        if type(row) is not dict or set(row) != {
            "frontier_rows", "exact_rows", "next_lower_bound",
            "constrained_threshold", "selected_rows", "certified",
            "proposal_digest",
        } or not all((
            type(row["frontier_rows"]) is int and row["frontier_rows"] > 0,
            type(row["exact_rows"]) is int and row["exact_rows"] >= 0,
            type(row["selected_rows"]) is int and 0 <= row["selected_rows"] <= match_count,
            type(row["certified"]) is bool,
            _finite_hex(row["constrained_threshold"]),
            row["next_lower_bound"] is None or _finite_hex(row["next_lower_bound"]),
            _is_hex(row["proposal_digest"]),
        )):
            raise OracleError("certificate round differs")
    for row in closures:
        if type(row) is not dict or set(row) != {
            "lower_exclusive", "upper_inclusive", "admitted_rows",
            "cumulative_native_bound_evaluated",
            "cumulative_exact_dtw_evaluated", "selected_rows",
            "resulting_threshold", "minimum_packed_unclassified_bound",
            "minimum_native_pruned_bound", "excluded_prefix_digest",
            "admitted_set_digest", "scan_result_digest", "certified",
        }:
            raise OracleError("certificate closure schema differs")
        if not all((
            row["lower_exclusive"] is None or _finite_hex(row["lower_exclusive"]),
            _finite_hex(row["upper_inclusive"]),
            _finite_hex(row["resulting_threshold"]),
            row["minimum_packed_unclassified_bound"] is None
            or _finite_hex(row["minimum_packed_unclassified_bound"]),
            row["minimum_native_pruned_bound"] is None
            or _finite_hex(row["minimum_native_pruned_bound"]),
            all(type(row[name]) is int and row[name] >= 0 for name in (
                "admitted_rows", "cumulative_native_bound_evaluated",
                "cumulative_exact_dtw_evaluated", "selected_rows",
            )),
            all(_is_hex(row[name]) for name in (
                "excluded_prefix_digest", "admitted_set_digest", "scan_result_digest",
            )),
            type(row["certified"]) is bool,
        )):
            raise OracleError("certificate closure values differ")
    native = state["native_bound_accounting"]
    if type(native) is not dict or set(native) != {
        "native_bound_evaluated", "exact_dtw_evaluated",
        "native_bound_pruned", "packed_bound_pruned",
    } or any(type(value) is not int or value < 0 for value in native.values()):
        raise OracleError("certificate native accounting differs")
    if (
        native["native_bound_evaluated"] + native["packed_bound_pruned"]
        != state["eligible_candidates"]
        or native["exact_dtw_evaluated"] + native["native_bound_pruned"]
        != native["native_bound_evaluated"]
    ):
        raise OracleError("certificate native conservation differs")
    if closures:
        if rounds[-1]["certified"] is not False or closures[-1]["certified"] is not True:
            raise OracleError("certificate closure terminal state differs")
        prior_native = rounds[-1]["frontier_rows"]
        prior_exact = rounds[-1]["exact_rows"]
        for index, row in enumerate(closures):
            if row["cumulative_native_bound_evaluated"] != (
                prior_native + row["admitted_rows"]
            ):
                raise OracleError("certificate closure admission accounting differs")
            if not (
                prior_exact <= row["cumulative_exact_dtw_evaluated"]
                <= row["cumulative_native_bound_evaluated"]
            ):
                raise OracleError("certificate closure exact accounting differs")
            if index < len(closures) - 1 and row["certified"]:
                raise OracleError("certificate closure certified before terminal pass")
            prior_native = row["cumulative_native_bound_evaluated"]
            prior_exact = row["cumulative_exact_dtw_evaluated"]
        if prior_native != native["native_bound_evaluated"] \
                or prior_exact != native["exact_dtw_evaluated"]:
            raise OracleError("certificate closure final accounting differs")
    elif rounds[-1]["certified"] is not True:
        raise OracleError("certificate without closure is not round-certified")
    if state["stop_threshold"] != max(
        matches, key=lambda row: float.fromhex(row["total_distance_hex"])
    )["total_distance_hex"]:
        raise OracleError("certificate threshold does not bind sealed matches")

    if _certificate_result_digest_from_sealed(state, matches) != state["result_digest"]:
        raise OracleError("certificate result digest does not reconstruct")
    return state


def _validate_accounting_seal(value: Any, certificate: dict[str, Any]) -> None:
    state = _validate_seal(value, "accounting")
    if set(state) != {
        "claim", "independent_eligible", "certificate_eligible",
        "exact_evaluated", "safely_pruned", "native",
    } or "conservation" not in state["claim"]:
        raise OracleError("accounting evidence schema differs")
    if not all((
        state["independent_eligible"] == certificate["eligible_candidates"],
        state["certificate_eligible"] == certificate["eligible_candidates"],
        state["exact_evaluated"] == certificate["exact_evaluated"],
        state["safely_pruned"] == certificate["safely_pruned"],
        state["native"] == certificate["native_bound_accounting"],
    )):
        raise OracleError("accounting evidence cross-binding differs")


def _validate_case(case: Any) -> None:
    required = {
        "seed", "configuration", "eligible_rows", "sparse_missing_volume",
        "proposal_scalar_forward_result_digest",
        "proposal_scalar_reverse_result_digest",
        "proposal_threaded_forward_result_digest",
        "proposal_threaded_reverse_result_digest", "query", "matches",
        "source", "execution", "certificate", "accounting",
        "exhaustive_matches_digest",
        "certificate_result_digest",
        "closure_passes", "repeated_frontier_rounds",
    }
    if type(case) is not dict or set(case) != required:
        raise OracleError("property case schema differs")
    proposal_digests = [
        case[name] for name in required if name.startswith("proposal_")
    ]
    if not proposal_digests or any(not _is_hex(value) for value in proposal_digests) \
            or len(set(proposal_digests)) != 1:
        raise OracleError("property proposal parity seal differs")
    query = _validate_query_seal(case["query"])
    _validate_seal(case["source"], "source")
    if case["source"] != _reconstruct_expected_source_manifest(query):
        raise OracleError("property source manifest does not reconstruct")
    execution = _validate_seal(case["execution"], "execution")
    if execution != {
        "initial_frontier_rows": 8,
        "maximum_frontier_rows": 18,
        "seed_rows": 8,
        "proposal_quota": 1_000,
        "proposal_traversals": [
            ["scalar", "forward", 4, 1],
            ["scalar", "reverse", 9, 1],
            ["threaded", "forward", 5, 1],
            ["threaded", "forward", 7, 3],
            ["threaded", "reverse", 6, 1],
            ["threaded", "reverse", 8, 3],
        ],
        "certified_workers": [1, 2],
        "certified_traversal_worker_cross_product": [
            ["forward", 1], ["forward", 2],
            ["reverse", 1], ["reverse", 2],
        ],
        "numeric_tolerance_hex": NUMERIC_ATOL.hex(),
    }:
        raise OracleError("property execution contract differs")
    matches = _validate_match_seal(case["matches"])
    certificate = _validate_certificate_seal(
        case["certificate"], query_id=query["query_episode_id"],
        matches=matches,
    )
    expected_proposal = _reconstruct_expected_property_proposal(case["seed"])
    if any(
        value != expected_proposal["proposal_result_digest"]
        for value in proposal_digests
    ) or not all((
        certificate["generation_id"] == expected_proposal["generation_id"],
        certificate["query_episode_id"] == expected_proposal["query_episode_id"],
        certificate["eligible_candidates"] == expected_proposal["eligible_rows"],
    )):
        raise OracleError("property proposal/generation semantics do not reconstruct")
    _validate_accounting_seal(case["accounting"], certificate)
    if certificate["input_digest"] != _reconstruct_expected_input_digest(query):
        raise OracleError("property certificate input does not reconstruct")
    expected_request = {
        "search_datasets": [DATASET], "quality_tiers": ["A"],
        "top_k": 5 if case["configuration"] == "unconstrained" else 3,
        "cross_dataset": False,
        "deduplicate_overlaps": case["configuration"] == "constrained",
        "max_per_instrument": 24 if case["configuration"] == "unconstrained" else 1,
        "minimum_history_gap_bars": 20,
    }
    if query["request"] != expected_request or query["universe"] != {
        "seed": case["seed"], "sampled": True, "perturb_ohlcv": True,
        "sparse_missing_volume": case["seed"] % 2 == 0,
    }:
        raise OracleError("property query/universe contract differs")
    expected_result = _reconstruct_expected_property_result(
        case["seed"], case["configuration"],
    )
    for name in (
        "query", "source", "matches", "exhaustive_matches_digest",
        "certificate", "accounting",
    ):
        if case[name] != expected_result[name]:
            raise OracleError(
                f"property {name} differs from deterministic reconstruction"
            )
    frontiers = [row["frontier_rows"] for row in certificate["rounds"]]
    if frontiers[0] != execution["initial_frontier_rows"]:
        raise OracleError("property initial frontier differs")
    previous = frontiers[0]
    for frontier in frontiers[1:]:
        expected_next = min(
            previous * 2, execution["maximum_frontier_rows"],
            certificate["eligible_candidates"],
        )
        if frontier not in {previous, expected_next}:
            raise OracleError("property frontier progression differs")
        previous = frontier
    exact_rows = [row["exact_rows"] for row in certificate["rounds"]]
    if any(right < left for left, right in zip(exact_rows, exact_rows[1:])):
        raise OracleError("property exact-row progression differs")
    if not all((
        type(case["seed"]) is int,
        case["configuration"] in {"unconstrained", "constrained"},
        type(case["eligible_rows"]) is int and case["eligible_rows"] >= 20,
        type(case["sparse_missing_volume"]) is bool,
        case["eligible_rows"] == certificate["eligible_candidates"],
        case["certificate_result_digest"] == certificate["result_digest"],
        case["closure_passes"] == len(certificate["threshold_closure_passes"]),
        case["repeated_frontier_rounds"]
        == len(certificate["rounds"])
        - len({row["frontier_rows"] for row in certificate["rounds"]}),
    )):
        raise OracleError("property case cross-binding differs")


def validate_payload(payload: Any, *, require_production: bool = False) -> None:
    if type(payload) is not dict or set(payload) != {
        "schema_version", "truth_free", "real_forward_outcomes_accessed",
        "runtime_manifest", "cardinality_and_boundaries",
        "certified_cardinality",
        "numeric_comparison", "output_contract",
        "certified_vs_exhaustive", "randomized_property_matrix",
        "max_per_instrument_underfill_completion", "branch_hits",
        "rehashed_mutation_kinds", "passed", "result_digest",
    }:
        raise OracleError("oracle top-level schema differs")
    if not all((
        payload["schema_version"] == SCHEMA,
        payload["truth_free"] is True,
        payload["real_forward_outcomes_accessed"] is False,
        payload["passed"] is True,
        _is_hex(payload["result_digest"]),
        payload["result_digest"] == stable_hash({
            key: value for key, value in payload.items() if key != "result_digest"
        }),
    )):
        raise OracleError("oracle terminal/digest differs")
    _strict_json(payload)
    _validate_runtime_manifest(
        payload["runtime_manifest"], require_production=require_production,
    )
    if payload["output_contract"] != {
        "repo_relative_path": str(CANONICAL_OUTPUT),
        "create_only": True,
        "protected_repo_roots": [
            ".git", "src", "experiments",
            "config/data/analogues/m04r10/nasdaq-untouched-authority-registry",
            "config/data/analogues/m04r11", "config/data/analogues/m04r13",
            "config/data/analogues/poc/m04r/packed-bound-full",
        ],
        "protected_resident_root": "/dev/shm/market-analogues/m04r11-candidate-v2",
    }:
        raise OracleError("canonical output contract differs")
    if payload["numeric_comparison"] != {
        "ordered_ids_metadata_alignments": "exact",
        "distance_components": "absolute-tolerance",
        "absolute_tolerance_hex": NUMERIC_ATOL.hex(),
        "reason": (
            "exhaustive and deferred/vector scorers are mathematically equal "
            "but use different floating evaluation paths"
        ),
    }:
        raise OracleError("numeric comparison policy differs")
    cardinality = payload["cardinality_and_boundaries"]
    if type(cardinality) is not dict or set(cardinality) != {
        "eligible_result_digests", "parity_result_digest",
        "upper_boundary_episode_id", "upper_boundary_scan_digest",
        "duplicate_id_rejected",
    } or set(cardinality["eligible_result_digests"]) != {"0", "19", "20", "25"}:
        raise OracleError("cardinality oracle differs")
    if cardinality["duplicate_id_rejected"] != 1 or any(
        not _is_hex(value)
        for value in cardinality["eligible_result_digests"].values()
    ) or not all((
        _is_hex(cardinality["parity_result_digest"]),
        _is_hex(cardinality["upper_boundary_episode_id"], 24),
        _is_hex(cardinality["upper_boundary_scan_digest"]),
    )):
        raise OracleError("duplicate-ID rejection was not hit")
    if cardinality != _reconstruct_expected_cardinality(certified=False):
        raise OracleError("cardinality oracle does not reconstruct")
    certified_cardinality = _validate_seal(
        payload["certified_cardinality"], "certified cardinality",
    )
    if set(certified_cardinality) != {"rows"} or set(
        certified_cardinality["rows"]
    ) != {"0", "19", "20"}:
        raise OracleError("certified cardinality schema differs")
    for count in (0, 19):
        row = certified_cardinality["rows"][str(count)]
        if type(row) is not dict or set(row) != {
            "eligible_rows", "status", "proposal_result_digest",
        } or row["eligible_rows"] != count \
                or row["status"] != "underfill-rejected" \
                or not _is_hex(row["proposal_result_digest"]):
            raise OracleError("certified cardinality underfill differs")
    twenty = certified_cardinality["rows"]["20"]
    if type(twenty) is not dict or set(twenty) != {
        "eligible_rows", "status", "proposal_result_digest", "matches",
        "certificate",
    } or twenty["eligible_rows"] != 20 or twenty["status"] != "certified" \
            or not _is_hex(twenty["proposal_result_digest"]):
        raise OracleError("certified cardinality success differs")
    cardinal_matches = _validate_match_seal(twenty["matches"])
    cardinal_certificate = _validate_certificate_seal(
        twenty["certificate"],
        query_id=twenty["certificate"]["state"]["query_episode_id"],
        matches=cardinal_matches,
    )
    if len(cardinal_matches) != 20 \
            or cardinal_certificate["eligible_candidates"] != 20:
        raise OracleError("certified cardinality top-20 differs")
    if payload["certified_cardinality"] != _reconstruct_expected_cardinality(
        certified=True,
    ):
        raise OracleError("certified cardinality oracle does not reconstruct")
    exact = payload["certified_vs_exhaustive"]
    if type(exact) is not dict or set(exact) != {
        "eligible_rows", "retained_main", "retained_overflow", "top_k",
        "tied_at_k", "overlap_rejected", "cap_rejected", "per_symbol",
        "closure_passes", "native_bound_pruned",
        "forward_reverse_result_digest", "exhaustive_match_digest",
        "mutation_rejected", "source_mutation_rejected",
        "store_mutation_rejected", "underfill_rejections", "query",
        "matches", "certificate", "accounting", "strict_finite_json",
    }:
        raise OracleError("primary exact evidence schema differs")
    query = _validate_query_seal(exact["query"])
    matches = _validate_match_seal(exact["matches"])
    certificate = _validate_certificate_seal(
        exact["certificate"], query_id=query["query_episode_id"],
        matches=matches,
    )
    _validate_accounting_seal(exact["accounting"], certificate)
    if certificate["input_digest"] != _reconstruct_expected_input_digest(query):
        raise OracleError("primary certificate input does not reconstruct")
    if query["universe"] != {
        "seed": 14_001, "sampled": False, "perturb_ohlcv": False,
        "sparse_missing_volume": False,
    } or query["request"] != {
        "search_datasets": [DATASET], "quality_tiers": ["A"], "top_k": 3,
        "cross_dataset": False, "deduplicate_overlaps": True,
        "max_per_instrument": 1, "minimum_history_gap_bars": 20,
    }:
        raise OracleError("primary query/universe contract differs")
    if exact["eligible_rows"] != certificate["eligible_candidates"] \
            or exact["top_k"] != len(matches):
        raise OracleError("primary exact evidence cross-binding differs")
    if not all((
        exact["retained_main"] > 0,
        exact["retained_overflow"] == 1,
        exact["tied_at_k"] > exact["top_k"],
        exact["overlap_rejected"] > 0,
        exact["cap_rejected"] > 0,
        exact["closure_passes"] == len(certificate["threshold_closure_passes"]),
        exact["native_bound_pruned"]
        == certificate["native_bound_accounting"]["native_bound_pruned"],
        exact["forward_reverse_result_digest"] == certificate["result_digest"],
        _is_hex(exact["exhaustive_match_digest"]),
        exact["mutation_rejected"] is True,
        exact["source_mutation_rejected"] == 1,
        exact["store_mutation_rejected"] == 1,
        exact["underfill_rejections"] == {"raw": 1, "constraint": 1},
        exact["strict_finite_json"] is True,
    )):
        raise OracleError("primary exact branch evidence differs")
    expected_primary = _reconstruct_expected_primary()
    for name in (
        "query", "matches", "certificate", "accounting",
        "exhaustive_match_digest",
    ):
        if exact[name] != expected_primary[name]:
            raise OracleError(
                f"primary {name} differs from deterministic reconstruction"
            )
    matrix = payload["randomized_property_matrix"]
    if type(matrix) is not dict or set(matrix) != {
        "fixed_seeds", "cases", "case_digest",
    } or matrix["fixed_seeds"] != [14_101, 14_102, 14_103, 14_104] \
            or matrix["case_digest"] != stable_hash(matrix["cases"]):
        raise OracleError("property matrix seal differs")
    if len(matrix["cases"]) != 8:
        raise OracleError("property matrix case count differs")
    for case in matrix["cases"]:
        _validate_case(case)
    expected_pairs = {
        (seed, configuration)
        for seed in matrix["fixed_seeds"]
        for configuration in ("unconstrained", "constrained")
    }
    if {(row["seed"], row["configuration"]) for row in matrix["cases"]} != expected_pairs:
        raise OracleError("property matrix seed/config coverage differs")
    if not any(row["sparse_missing_volume"] for row in matrix["cases"]):
        raise OracleError("missing-volume property universe was not hit")
    if len({
        row["source"]["digest"] for row in matrix["cases"]
    }) != 4 or len({
        row["certificate"]["state"]["input_digest"]
        for row in matrix["cases"]
    }) != 8:
        raise OracleError("seed/source/input diversity differs")
    completion = payload["max_per_instrument_underfill_completion"]
    if completion != {
        "completed_rows": 1,
        "initial_threshold_state": "incomplete-infinity",
        "raw_second_bound_hex": float(99.0).hex(),
        "resulting_threshold_hex": float(100.0).hex(),
    }:
        raise OracleError("max-per underfill completion evidence differs")
    if payload["rehashed_mutation_kinds"] != list(MUTATION_KINDS):
        raise OracleError("mutation qualification inventory differs")
    hits = payload["branch_hits"]
    if type(hits) is not dict or any(
        type(value) is not int or value <= 0 for value in hits.values()
    ):
        raise OracleError("branch-hit evidence differs")
    expected_hits = {
        "overflow_retained": exact["retained_overflow"],
        "threshold_closure": exact["closure_passes"],
        "native_deferral": exact["native_bound_pruned"],
        "max_per_instrument_cap": exact["cap_rejected"],
        "overlap_dedup": exact["overlap_rejected"],
        "k_boundary_ties": exact["tied_at_k"] - exact["top_k"],
        "raw_underfill_rejected": exact["underfill_rejections"]["raw"],
        "constraint_underfill_rejected": exact["underfill_rejections"]["constraint"],
        "max_per_completion_from_underfill": completion["completed_rows"],
        "property_matrix_cases": len(matrix["cases"]),
        "sparse_symbol_groups": sum(
            row["certificate"]["state"]["sparse_symbols"]
            for row in matrix["cases"]
        ),
        "repeated_frontier_rounds": sum(
            row["repeated_frontier_rounds"] for row in matrix["cases"]
        ),
        "duplicate_id_rejected": cardinality["duplicate_id_rejected"],
        "certified_cardinality_cases": len(
            _validate_seal(payload["certified_cardinality"], "certified cardinality")[
                "rows"
            ]
        ),
        "source_mutation_rejected": exact["source_mutation_rejected"],
        "store_mutation_rejected": exact["store_mutation_rejected"],
        "rehashed_mutations_rejected": len(MUTATION_KINDS),
    }
    if hits != expected_hits:
        raise OracleError("branch-hit reconstruction differs")


def _rehash(payload: dict[str, Any]) -> None:
    payload["result_digest"] = stable_hash({
        key: value for key, value in payload.items() if key != "result_digest"
    })


def rehashed_mutation(payload: dict[str, Any], kind: str) -> dict[str, Any]:
    if kind not in MUTATION_KINDS:
        raise ValueError("unknown mutation kind")
    changed = copy.deepcopy(payload)
    matrix = changed["randomized_property_matrix"]
    case = matrix["cases"][0]
    reseal: str | None = None
    if kind == "proposal":
        case["proposal_threaded_reverse_result_digest"] = "f" * 64
    elif kind == "proposal-all-coordinated":
        for name in tuple(case):
            if name.startswith("proposal_"):
                case[name] = "f" * 64
    elif kind == "certificate":
        case["certificate"]["state"]["result_digest"] = "e" * 64
        reseal = "certificate"
    elif kind == "certificate-result-coordinated":
        case["certificate"]["state"]["result_digest"] = "e" * 64
        case["certificate_result_digest"] = "e" * 64
        reseal = "certificate"
    elif kind == "certificate-input-coordinated":
        certificate = case["certificate"]["state"]
        certificate["input_digest"] = "d" * 64
        certificate["result_digest"] = _certificate_result_digest_from_sealed(
            certificate, case["matches"]["state"]["rows"],
        )
        case["certificate_result_digest"] = certificate["result_digest"]
        reseal = "certificate"
    elif kind == "round":
        case["certificate"]["state"]["rounds"][0]["frontier_rows"] += 1
        reseal = "certificate"
    elif kind == "closure":
        case["certificate"]["state"]["threshold_closure_passes"][0][
            "admitted_rows"
        ] += 1
        reseal = "certificate"
    elif kind == "accounting":
        case["accounting"]["state"]["independent_eligible"] += 1
        reseal = "accounting"
    elif kind == "query":
        case["query"]["state"]["query_symbol"] = "FORGED"
        reseal = "query"
    elif kind == "query-gap-coordinated":
        case["query"]["state"]["request"]["minimum_history_gap_bars"] = 21
        reseal = "query"
    elif kind == "matches":
        case["matches"]["state"]["rows"][0]["episode_id"] = "f" * 24
        reseal = "matches"
    elif kind == "match-distance-coordinated":
        rows = case["matches"]["state"]["rows"]
        for row in rows:
            row["total_distance_hex"] = (
                float.fromhex(row["total_distance_hex"]) + 1e-7
            ).hex()
        certificate = case["certificate"]["state"]
        certificate["stop_threshold"] = max(
            rows, key=lambda row: float.fromhex(row["total_distance_hex"])
        )["total_distance_hex"]
        certificate["result_digest"] = _certificate_result_digest_from_sealed(
            certificate, rows,
        )
        case["certificate_result_digest"] = certificate["result_digest"]
        case["matches"]["digest"] = stable_hash(case["matches"]["state"])
        case["certificate"]["digest"] = stable_hash(certificate)
    elif kind == "cardinality-coordinated":
        cardinality = changed["cardinality_and_boundaries"]
        cardinality["eligible_result_digests"] = {
            name: "c" * 64 for name in cardinality["eligible_result_digests"]
        }
        cardinality["parity_result_digest"] = "c" * 64
        cardinality["upper_boundary_scan_digest"] = "c" * 64
    elif kind == "primary-coordinated":
        primary = changed["certified_vs_exhaustive"]
        rows = primary["matches"]["state"]["rows"]
        for row in rows:
            row["total_distance_hex"] = (
                float.fromhex(row["total_distance_hex"]) + 1e-7
            ).hex()
        certificate = primary["certificate"]["state"]
        certificate["stop_threshold"] = max(
            rows, key=lambda row: float.fromhex(row["total_distance_hex"])
        )["total_distance_hex"]
        certificate["result_digest"] = _certificate_result_digest_from_sealed(
            certificate, rows,
        )
        primary["forward_reverse_result_digest"] = certificate["result_digest"]
        primary["exhaustive_match_digest"] = "b" * 64
        primary["matches"]["digest"] = stable_hash(primary["matches"]["state"])
        primary["certificate"]["digest"] = stable_hash(certificate)
    if reseal is not None:
        case[reseal]["digest"] = stable_hash(case[reseal]["state"])
    matrix["case_digest"] = stable_hash(matrix["cases"])
    _rehash(changed)
    return changed


def _exercise_rehashed_mutations(payload: dict[str, Any]) -> None:
    for kind in MUTATION_KINDS:
        changed = rehashed_mutation(payload, kind)
        try:
            validate_payload(changed, require_production=False)
        except OracleError:
            continue
        raise OracleError(f"rehashed {kind} mutation was accepted")


def _publish_create_only(
    path: Path, payload: dict[str, Any], *, create_parent: bool = False,
) -> None:
    if create_parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    path = _validate_output_path(path)
    encoded = _strict_json(payload) + b"\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_path, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary_path.unlink(missing_ok=True)


def _validate_output_path(path: Path) -> Path:
    lexical = path if path.is_absolute() else Path.cwd() / path
    normalized = Path(os.path.abspath(lexical))
    if lexical != normalized:
        raise OracleError("output path contains an aliased lexical ancestor")
    parent = normalized.parent
    if not parent.is_dir():
        raise OracleError("output parent must already exist as a directory")
    cursor = parent
    while True:
        if cursor.is_symlink():
            raise OracleError("output ancestry must not contain symlinks")
        if cursor.parent == cursor:
            break
        cursor = cursor.parent
    if parent.resolve(strict=True) != parent:
        raise OracleError("output ancestry resolves through an alias")
    if normalized.exists() or normalized.is_symlink():
        raise FileExistsError(f"output already exists: {normalized}")
    return normalized


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _production_output(repo_root: Path) -> Path:
    repo_root = repo_root.resolve(strict=True)
    script_repo = Path(__file__).resolve().parents[2]
    if repo_root != script_repo:
        raise OracleError("production repository root differs from script repository")
    output = repo_root / CANONICAL_OUTPUT
    protected = [
        repo_root / ".git", repo_root / "src", repo_root / "experiments",
        repo_root / "config/data/analogues/m04r11",
        repo_root / "config/data/analogues/m04r13",
        repo_root / "config/data/analogues/m04r10/nasdaq-untouched-authority-registry",
        repo_root / "config/data/analogues/poc/m04r/packed-bound-full",
        Path("/dev/shm/market-analogues/m04r11-candidate-v2"),
    ]
    output_root = output.parent
    for protected_root in protected:
        if protected_root.exists() and _paths_overlap(
            output_root.resolve(strict=False), protected_root.resolve(strict=True),
        ):
            raise OracleError(f"oracle output overlaps protected input: {protected_root}")
    nearest = output_root
    while not nearest.exists():
        nearest = nearest.parent
    if nearest.is_symlink() or nearest.resolve(strict=True) != nearest:
        raise OracleError("canonical output ancestry is aliased")
    return output


def _load_strict_json(path: Path) -> dict[str, Any]:
    lexical = path if path.is_absolute() else Path.cwd() / path
    if lexical.is_symlink():
        raise OracleError("verification input must not be a symlink")
    path = lexical.resolve(strict=True)
    if not path.is_file():
        raise OracleError("verification input must be a regular non-symlink file")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        raw = b""
        while True:
            block = os.read(descriptor, 1 << 20)
            if not block:
                break
            raw += block
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = lambda value: (
        value.st_dev, value.st_ino, value.st_size,
        value.st_mtime_ns, value.st_ctime_ns, value.st_mode,
    )
    if identity(before) != identity(after):
        raise OracleError("verification input mutated during its single read")

    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in values:
            if key in output:
                raise OracleError("verification input contains a duplicate key")
            output[key] = value
        return output

    try:
        value = json.loads(
            raw, object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                OracleError(f"nonfinite JSON constant: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OracleError("verification input is not strict JSON") from exc
    if type(value) is not dict:
        raise OracleError("verification input top level differs")
    return value


def run(
    output: Path | None = None, *, repo_root: Path | None = None,
    runtime_manifest: dict[str, Any] | None = None,
    enforce_production: bool = True,
) -> dict[str, Any]:
    if enforce_production:
        if output is not None or runtime_manifest is not None:
            raise OracleError("production run fixes output and runtime manifest")
        selected_repo = (
            Path(__file__).resolve().parents[2] if repo_root is None else repo_root
        )
        runtime_manifest = _production_runtime_manifest(selected_repo)
        output = _production_output(selected_repo)
        create_parent = True
    else:
        if output is None or runtime_manifest is None:
            raise OracleError("test run requires explicit output and injected manifest")
        output = _validate_output_path(output)
        _validate_runtime_manifest(runtime_manifest, require_production=False)
        create_parent = False
    with tempfile.TemporaryDirectory(prefix="m04r14-adversarial-oracle-") as temporary:
        payload = build_oracle_payload(
            Path(temporary) / "work", runtime_manifest,
        )
    validate_payload(payload, require_production=enforce_production)
    _publish_create_only(output, payload, create_parent=create_parent)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    execute = subparsers.add_parser("run")
    execute.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[2],
    )
    verify = subparsers.add_parser("validate")
    verify.add_argument("--input", type=Path, required=True)
    verify.add_argument("--allow-test-manifest", action="store_true")
    arguments = parser.parse_args(argv)
    if arguments.command == "run":
        run(repo_root=arguments.repo_root)
    else:
        if not arguments.allow_test_manifest and arguments.input.resolve() != (
            Path(__file__).resolve().parents[2] / CANONICAL_OUTPUT
        ):
            raise OracleError("production validation input is not canonical output")
        validate_payload(
            _load_strict_json(arguments.input),
            require_production=not arguments.allow_test_manifest,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
