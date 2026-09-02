"""Outcome-blind same-month concurrency POC for certified staged retrieval."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import resource
import subprocess
from time import perf_counter
from typing import Any, Mapping, Sequence

from market_analogues.adapters import CachedOHLCVSource, source_from_spec
from market_analogues.config import load_config
from market_analogues.dtw_component_search import (
    certified_staged_dtw_component_search, staged_dtw_component_search_contract,
)
from market_analogues.episodes import build_episode
from market_analogues.types import InstrumentKey, SearchQuery, stable_hash

from experiments.m04r import m04r14_t14_10_wf03_feasibility as base
from experiments.m04r import m04r14_t14_10_wf03b_dtw_component_ladder as ladder


SCHEMA = "m04r14-t14-10-wf03-combined-concurrency-poc-preregistration-v1"
OUTPUT_RELATIVE = Path(
    "config/data/analogues/m04r14/t14-10-wf03-combined-concurrency-poc-v1"
)
PREREGISTRATION_RELATIVE = Path(
    "experiments/m04r/m04r14_t14_10_wf03_combined_concurrency_poc_preregistered.json"
)
QUERY_IDS = (
    "cb0774b91343c225c486d973",  # first liquidity cell
    "fb6669b28c515fcd8a38d6b6",  # second liquidity cell
    "a69def453340e01048a52284",  # frozen scalar authority
    "14db3379bd20a4ad5d937186",  # fourth liquidity cell
)
AUTHORITY_QUERY_ID = "a69def453340e01048a52284"
AUTHORITY_RELATIVE = Path(
    "config/data/analogues/m04r14/t14-10-wf03b-dtw-component-ladder-v1/"
    "cases/000-early-a69def453340e01048a52284/attempts/attempt-0001/EXACT.json"
)
CONCURRENCY = 4
THREADS_PER_QUERY = 2
SEED_ROWS = 16_384
BLOCK_ROWS = 4_096
RUNTIME_FILES = (
    "experiments/m04r/m04r14_t14_10_wf03_combined_concurrency_poc.py",
    "src/market_analogues/adapters.py",
    "src/market_analogues/component_search.py",
    "src/market_analogues/dtw_component_search.py",
    "src/market_analogues/exact_batch.py",
    "src/market_analogues/dtw_sample_store.py",
    "src/market_analogues/dtw_interval_bound.py",
    "src/market_analogues/packed_bound_store.py",
    "src/market_analogues/packed_bound_search.py",
)


class ConcurrencyPocError(RuntimeError):
    pass


def _git(repository: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repository, capture_output=True, text=True,
        check=False,
    )
    if result.returncode:
        raise ConcurrencyPocError(result.stderr.strip() or "git command failed")
    return result.stdout.strip()


def _sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _inputs(repository: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    registry, by_id = base._registry(repository)
    rows = [by_id[query_id] for query_id in QUERY_IDS]
    if any(row["cutoff"] != "2014-03-31T00:00:00" for row in rows):
        raise ConcurrencyPocError("POC queries are not in one frozen month")
    verification = base._read(repository / ladder.DTW_VERIFICATION_RELATIVE)
    base._validate_seal(verification, "verification_digest")
    if verification.get("passed") is not True \
            or verification.get("generation_id") != ladder.DTW_GENERATION_ID:
        raise ConcurrencyPocError("verified DTW input differs")
    return registry, rows


def build_preregistration(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    if _git(repository, "status", "--porcelain"):
        raise ConcurrencyPocError("POC preregistration requires a clean commit")
    output = repository / OUTPUT_RELATIVE
    if output.exists() or output.is_symlink():
        raise ConcurrencyPocError("POC output must be absent before freezing")
    registry, rows = _inputs(repository)
    head = _git(repository, "rev-parse", "HEAD")
    state = {
        "schema_version": SCHEMA,
        "status": "frozen_before_concurrency_poc",
        "implementation_commit": head,
        "runtime_files": {path: _sha(repository / path) for path in RUNTIME_FILES},
        "inputs": {
            "registry_digest": registry["registry_digest"],
            "packed_generation_id": base.GENERATION_ID,
            "packed_provenance_digest": base.PROVENANCE_DIGEST,
            "dtw_generation_id": ladder.DTW_GENERATION_ID,
            "dtw_verification_sha256": _sha(
                repository / ladder.DTW_VERIFICATION_RELATIVE
            ),
            "authority_sha256": _sha(repository / AUTHORITY_RELATIVE),
        },
        "queries": rows,
        "contract": staged_dtw_component_search_contract(),
        "execution": {
            "concurrency": CONCURRENCY,
            "threads_per_query": THREADS_PER_QUERY,
            "preload_workers": 8,
            "source_cache_max_entries": None,
            "seed_rows": SEED_ROWS,
            "block_rows": BLOCK_ROWS,
            "output_root": str(output.resolve()),
        },
        "gates": {
            "four_results_complete": True,
            "authority_episode_ids_and_hex_distances_exact": True,
            "zero_swap": True,
            "outcomes_or_labels_excluded": True,
        },
        "claims": {
            "historical_query_retrieval_opened": True,
            "historical_walk_forward_query_outcomes_opened": False,
            "final_period_result_opened": False,
            "production_promotion_authorized": False,
        },
    }
    return base._sealed(state, "preregistration_digest")


def validate_preregistration(repository: Path, value: Mapping[str, Any]) -> None:
    base._validate_seal(value, "preregistration_digest")
    registry, rows = _inputs(repository)
    if not all((
        value.get("schema_version") == SCHEMA,
        value.get("status") == "frozen_before_concurrency_poc",
        value.get("inputs", {}).get("registry_digest") == registry["registry_digest"],
        value.get("inputs", {}).get("authority_sha256")
            == _sha(repository / AUTHORITY_RELATIVE),
        value.get("queries") == rows,
        value.get("execution", {}).get("concurrency") == CONCURRENCY,
        value.get("execution", {}).get("threads_per_query") == THREADS_PER_QUERY,
    )):
        raise ConcurrencyPocError("POC preregistration differs")
    commit = str(value.get("implementation_commit"))
    _git(repository, "merge-base", "--is-ancestor", commit, "HEAD")
    for path, digest in value.get("runtime_files", {}).items():
        if _sha(repository / path) != digest:
            raise ConcurrencyPocError(f"POC runtime changed: {path}")
        blob = subprocess.run(
            ["git", "show", f"{commit}:{path}"], cwd=repository,
            capture_output=True, check=False,
        )
        if blob.returncode or sha256(blob.stdout).hexdigest() != digest:
            raise ConcurrencyPocError(f"POC Git binding differs: {path}")


def execute(repository: Path, preregistration: Mapping[str, Any]) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    validate_preregistration(repository, preregistration)
    output = repository / OUTPUT_RELATIVE
    if output.exists() or output.is_symlink():
        raise ConcurrencyPocError("POC output is create-only")
    output.mkdir(parents=True)
    config = load_config(repository / base.CONFIG_RELATIVE)
    raw_source = source_from_spec(config.datasets["nasdaq"])
    source = CachedOHLCVSource(raw_source, max_entries=None)
    preload_started = perf_counter()
    source.preload(tuple(raw_source.instruments()), workers=8)
    preload_seconds = perf_counter() - preload_started

    tasks = []
    for row in preregistration["queries"]:
        episode = build_episode(
            source, InstrumentKey("nasdaq", row["symbol"]), row["cutoff"],
            row["lookback"], row["representation_version"],
        )
        if episode.key.id != row["episode_id"]:
            raise ConcurrencyPocError("POC query reconstruction differs")
        request = SearchQuery(
            episode.key, ("nasdaq",), ("A", "B"), base.TOP_K,
            False, True, base.MAX_PER_INSTRUMENT, base.MINIMUM_HISTORY_GAP,
        )
        tasks.append((row, episode, request))
    resident_root = Path(base._resident()["store_root"])
    dtw_root = repository / ladder.DTW_ROOT_RELATIVE

    def one(task: tuple[dict[str, Any], Any, SearchQuery]) -> dict[str, Any]:
        row, episode, request = task
        result = certified_staged_dtw_component_search(
            episode, source, request, resident_root, base.GENERATION_ID,
            dtw_root, ladder.DTW_GENERATION_ID, store_dataset_id="nasdaq",
            seed_rows=SEED_ROWS, block_rows=BLOCK_ROWS,
            rigid_threads=THREADS_PER_QUERY, dtw_threads=THREADS_PER_QUERY,
            exact_workers=THREADS_PER_QUERY, verify_content=False,
        )
        matches = [{
            "episode_id": match.episode_key.id,
            "symbol": match.episode_key.instrument.source_symbol,
            "cutoff": match.episode_key.cutoff.isoformat(),
            "distance_hex": match.total_distance.hex(),
        } for match in result.matches]
        return {
            "query_id": row["episode_id"], "symbol": row["symbol"],
            "certificate": asdict(result.certificate), "matches": matches,
            "semantic_digest": stable_hash({
                "query_id": row["episode_id"],
                "certificate_result_digest": result.certificate.result_digest,
                "matches": matches,
            }),
        }

    run_started = perf_counter()
    with ThreadPoolExecutor(
        max_workers=CONCURRENCY, thread_name_prefix="combined-query",
    ) as executor:
        cases = list(executor.map(one, tasks))
    concurrent_seconds = perf_counter() - run_started
    authority = base._read(repository / AUTHORITY_RELATIVE)
    expected = [{
        key: match[key] for key in ("episode_id", "symbol", "cutoff", "distance_hex")
    } for match in authority["matches"]]
    actual = next(
        row["matches"] for row in cases if row["query_id"] == AUTHORITY_QUERY_ID
    )
    swap_kib = int(Path("/proc/self/status").read_text().split("VmSwap:")[1].split()[0])
    passed = len(cases) == len(QUERY_IDS) and actual == expected and swap_kib == 0
    deterministic = {
        "schema_version": "m04r14-t14-10-wf03-combined-concurrency-poc-result-v1",
        "status": "complete", "passed": passed,
        "preregistration_digest": preregistration["preregistration_digest"],
        "ordered_case_semantic_digests": [row["semantic_digest"] for row in cases],
        "authority_exact": actual == expected,
        "queries": len(cases), "concurrency": CONCURRENCY,
        "threads_per_query": THREADS_PER_QUERY,
        "cache_state": source.cache_state(), "swap_kib": swap_kib,
        "historical_walk_forward_query_outcomes_opened": False,
        "final_period_result_opened": False,
        "outcomes_or_labels_used": False,
    }
    payload = base._sealed({
        **deterministic, "preload_seconds": preload_seconds,
        "concurrent_seconds": concurrent_seconds,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "cases": cases,
    })
    base._atomic(output / "RESULT.json", payload)
    if not passed:
        raise ConcurrencyPocError("concurrency POC gates failed")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("preregister", "run"))
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    repository = args.repository.resolve(strict=True)
    if args.action == "preregister":
        value = build_preregistration(repository)
        base._atomic(repository / PREREGISTRATION_RELATIVE, value)
    else:
        value = base._read(repository / PREREGISTRATION_RELATIVE)
        result = execute(repository, value)
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
