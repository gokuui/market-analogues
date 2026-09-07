"""Independently verify the complete true seven-group WF-03 batch."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from hashlib import sha256
import json
import math
import multiprocessing
from pathlib import Path
import subprocess
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from market_analogues.adapters import CachedOHLCVSource, source_from_spec
from market_analogues.causal_prefix import causal_prefix_digest
from market_analogues.config import load_config
from market_analogues.episodes import build_episode
from market_analogues.packed_bound_store import load_packed_generation
from market_analogues.representation import represent, representation_input_digest
from market_analogues.search import latest_eligible_cutoff
from market_analogues.types import EpisodeKey, InstrumentKey, SearchQuery, stable_hash

from experiments.m04r import m04r14_t14_10_wf03_composite_batch as producer
from experiments.m04r import m04r14_t14_10_wf03_composite_topology_poc as kernel
from experiments.m04r import m04r14_t14_10_wf03_feasibility as base
from experiments.m04r import verify_m04r14_t14_10_wf03_composite_topology_poc as certificate_verifier


OUTPUT_RELATIVE = Path(
    "config/data/analogues/m04r14/"
    "t14-10-wf03-composite-batch-v2-verification"
)
AUTHORITY_RELATIVE = Path(
    "config/data/analogues/m04r14/t14-10-wf03-feasibility-v1/cases/"
    "000-early-a69def453340e01048a52284/attempts/attempt-0001/EXACT.json"
)
AUTHORITY_QUERY_ID = "a69def453340e01048a52284"
QUALITY_CODES = {"A": 1, "B": 2}
SAMPLE_PER_FOLD = 2
INVENTORY_DTYPE = certificate_verifier.METADATA_DTYPE


class CompositeBatchVerificationError(RuntimeError):
    pass


def _git(repository: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repository, text=True,
        capture_output=True, check=False,
    )
    if result.returncode:
        raise CompositeBatchVerificationError(
            result.stderr.strip() or "git command failed"
        )
    return result.stdout.strip()


def _sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _seal_valid(value: Mapping[str, Any], field: str) -> bool:
    return type(value) is dict and value.get(field) == stable_hash({
        key: item for key, item in value.items() if key != field
    })


def _inventory(main: np.ndarray, overflow: np.ndarray) -> np.ndarray:
    result = np.empty(len(main) + len(overflow), dtype=INVENTORY_DTYPE)
    first = len(main)
    for name in INVENTORY_DTYPE.names or ():
        result[name][:first] = main[name]
        result[name][first:] = overflow[name]
    return result


def _lookup_positions(
    sorted_ids: np.ndarray, order: np.ndarray, identifiers: Sequence[str],
) -> np.ndarray:
    try:
        requested = np.asarray([
            np.void(bytes.fromhex(value)) for value in identifiers
        ], dtype="V12")
    except ValueError as exc:
        raise CompositeBatchVerificationError("invalid analogue episode ID") from exc
    positions = np.searchsorted(sorted_ids, requested)
    if np.any(positions >= len(sorted_ids)) \
            or not np.array_equal(sorted_ids[positions], requested):
        raise CompositeBatchVerificationError("analogue episode is absent")
    return order[positions]


def _alignment_valid(value: Any, query_length: int, candidate_length: int) -> bool:
    if type(value) is not list or not value:
        return False
    if any(
        type(item) is not list or len(item) != 2
        or any(type(number) is not int for number in item)
        for item in value
    ):
        return False
    try:
        points = [(int(item[0]), int(item[1])) for item in value]
    except (IndexError, TypeError, ValueError):
        return False
    if points[0] != (0, 0) \
            or points[-1] != (query_length - 1, candidate_length - 1):
        return False
    return all(
        (right[0] - left[0], right[1] - left[1]) in {(1, 0), (0, 1), (1, 1)}
        for left, right in zip(points, points[1:])
    )


def select_rerun_sample(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["fold_id"]), []).append(row)
    selected = []
    for fold in sorted(grouped):
        ordered = sorted(grouped[fold], key=lambda row: (
            stable_hash({
                "purpose": "wf03-true-composite-independent-rerun-v1",
                "fold": fold, "query_id": row["episode_id"],
            }),
            row["episode_id"],
        ))
        if len(ordered) < SAMPLE_PER_FOLD:
            raise CompositeBatchVerificationError("rerun fold is undersized")
        selected.extend(str(row["episode_id"]) for row in ordered[:SAMPLE_PER_FOLD])
    return selected


def _attempt_bindings(
    root: Path, preregistration: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    attempts_root = root / "attempts"
    if attempts_root.is_symlink() or not attempts_root.is_dir():
        raise CompositeBatchVerificationError("attempts root differs")
    output = {}
    for path in sorted(attempts_root.iterdir()):
        if path.is_symlink() or not path.is_dir() \
                or not producer._is_attempt_id(path.name):
            raise CompositeBatchVerificationError("attempt layout differs")
        names = {item.name for item in path.iterdir() if not item.name.startswith(".wf03-")}
        if "RUN_STARTED.json" not in names \
                or names - {"RUN_STARTED.json", "INTERRUPTED.json", "COMPLETE.json"} \
                or {"INTERRUPTED.json", "COMPLETE.json"}.issubset(names):
            raise CompositeBatchVerificationError("attempt files differ")
        started = base._read(path / "RUN_STARTED.json")
        if not _seal_valid(started, "attempt_digest") or not all((
            started.get("schema_version") == "m04r14-wf03-composite-batch-attempt-v2",
            started.get("status") == "running",
            started.get("attempt_id") == path.name,
            started.get("preregistration_digest") == preregistration["preregistration_digest"],
            started.get("resident_content_digest")
                == preregistration["inputs"]["resident_content_digest"],
            producer._is_digest(started.get("resident_identity_digest")),
            type(started.get("receipts_reused_at_start")) is int,
            0 <= started.get("receipts_reused_at_start", -1) <= producer.EXPECTED_QUERIES,
            started.get("resource_observation", {}).get("effective_cpus", 0)
                >= producer.PROCESS_COUNT,
            type(started.get("created_at")) is str,
        )):
            raise CompositeBatchVerificationError("attempt start differs")
        terminals = names & {"INTERRUPTED.json", "COMPLETE.json"}
        terminal_name = next(iter(terminals), None)
        terminal = None
        if terminal_name:
            terminal = base._read(path / terminal_name)
            expected_status = "complete" if terminal_name == "COMPLETE.json" else "interrupted"
            if not _seal_valid(terminal, "attempt_digest") or not all((
                terminal.get("schema_version")
                    == "m04r14-wf03-composite-batch-attempt-v2",
                terminal.get("status") == expected_status,
                terminal.get("attempt_id") == path.name,
                type(terminal.get("completed_this_attempt")) is int,
                0 <= terminal.get("completed_this_attempt", -1)
                    <= producer.EXPECTED_QUERIES,
                type(terminal.get("completed_total")) is int,
                0 <= terminal.get("completed_total", -1) <= producer.EXPECTED_QUERIES,
                type(terminal.get("created_at")) is str,
            )):
                raise CompositeBatchVerificationError("attempt terminal differs")
            if terminal_name == "COMPLETE.json" and not all((
                terminal.get("completed_total") == producer.EXPECTED_QUERIES,
                type(terminal.get("receipts_reused_at_start")) is int,
                terminal.get("receipts_reused_at_start", 0)
                    + terminal.get("completed_this_attempt", 0)
                    == producer.EXPECTED_QUERIES,
                producer._is_digest(terminal.get("result_digest")),
            )):
                raise CompositeBatchVerificationError("completion terminal differs")
            if terminal_name == "INTERRUPTED.json" and not all((
                type(terminal.get("error_type")) is str,
                type(terminal.get("error")) is str,
            )):
                raise CompositeBatchVerificationError("interruption terminal differs")
        output[path.name] = {
            **started, "_terminal_name": terminal_name, "_terminal": terminal,
        }
    if not output:
        raise CompositeBatchVerificationError("attempts are absent")
    return output


def _input_digest(
    source: Any, episode: Any, request: SearchQuery,
    packed_provenance_digest: str,
) -> str:
    benchmark = source.load_benchmark()
    return stable_hash({
        "query_stock_prefix": asdict(causal_prefix_digest(
            source.load(episode.key.instrument), episode.key.cutoff,
        )),
        "query_benchmark_prefix": (
            asdict(causal_prefix_digest(benchmark, episode.key.cutoff))
            if benchmark is not None else None
        ),
        "request": {
            "search_datasets": request.search_datasets,
            "quality_tiers": request.quality_tiers,
            "top_k": request.top_k,
            "cross_dataset": request.cross_dataset,
            "deduplicate_overlaps": request.deduplicate_overlaps,
            "max_per_instrument": request.max_per_instrument,
            "minimum_history_gap_bars": request.minimum_history_gap_bars,
        },
        "packed_provenance_digest": packed_provenance_digest,
        "query_representation_digest": representation_input_digest(
            represent(episode),
        ),
    })


def _independent_case(
    value: Mapping[str, Any], row: Mapping[str, Any],
    preregistration: Mapping[str, Any], attempts: Mapping[str, Mapping[str, Any]],
    records: np.ndarray, symbols: tuple[str, ...], order: np.ndarray,
    sorted_ids: np.ndarray, source: Any, packed_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    if not _seal_valid(value, "case_digest"):
        raise CompositeBatchVerificationError("case seal differs")
    try:
        retrieval = value["retrieval"]
        certificate = kernel.decode_certificate_json_value(retrieval["certificate"])
        matches = retrieval["matches"]
        measurement = value["worker_measurement"]
        certificate_verifier.validate_certificate(retrieval)
        attempt = attempts[value["attempt_id"]]
        predicates = (
            value["schema_version"] == "m04r14-wf03-composite-batch-case-v2",
            value["status"] == "complete", value["query_id"] == row["episode_id"],
            value["case_id"] == row["case_id"], value["symbol"] == row["symbol"],
            value["cutoff"] == row["cutoff"], value["fold_id"] == row["fold_id"],
            value["fold_role"] == row["fold_role"],
            value["scored"] is bool(row["scored"]),
            value["preregistration_digest"] == preregistration["preregistration_digest"],
            value["packed_generation_id"] == base.GENERATION_ID,
            value["resident_content_digest"]
                == preregistration["inputs"]["resident_content_digest"],
            value["resident_identity_digest"] == attempt["resident_identity_digest"],
            retrieval["query_id"] == row["episode_id"],
            certificate["query_episode_id"] == row["episode_id"],
            certificate["generation_id"] == base.GENERATION_ID,
            certificate["contract_digest"] == preregistration["contracts"]["retrieval"]["digest"],
            retrieval["semantic_digest"] == stable_hash(kernel._case_semantics(retrieval)),
            value["semantic_digest"] == stable_hash(kernel._case_semantics(retrieval)),
            measurement["queries"] == 1,
            measurement["threads"] == producer.THREADS_PER_PROCESS,
            measurement["swap_kib"] == 0,
            measurement["resident_identity_digest"] == value["resident_identity_digest"],
            value["outcomes_or_labels_used"] is False,
            value["historical_walk_forward_query_outcomes_opened"] is False,
            value["final_period_result_opened"] is False,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CompositeBatchVerificationError("case structure differs") from exc
    if not all(predicates):
        raise CompositeBatchVerificationError("case contract differs")

    episode = build_episode(
        source, InstrumentKey("nasdaq", row["symbol"]), row["cutoff"],
        int(row["lookback"]), row["representation_version"],
    )
    request = SearchQuery(
        episode.key, ("nasdaq",), ("A", "B"), base.TOP_K,
        False, True, base.MAX_PER_INSTRUMENT, base.MINIMUM_HISTORY_GAP,
    )
    expected_input = _input_digest(
        source, episode, request, packed_manifest["provenance_digest"],
    )
    if episode.key.id != row["episode_id"] \
            or certificate["input_digest"] != expected_input:
        raise CompositeBatchVerificationError("query input binding differs")
    latest_ns = int(latest_eligible_cutoff(
        episode, base.MINIMUM_HISTORY_GAP,
    ).value)
    query_start_ns = int(episode.bars.timestamp.iloc[0].value)
    positions = _lookup_positions(
        sorted_ids, order, [match["episode_id"] for match in matches],
    )
    weights = preregistration["contracts"]["distance_weights"]
    same_symbol = 0
    for match, position in zip(matches, positions, strict=True):
        symbol = symbols[int(records["symbol_id"][position])]
        cutoff_ns = int(records["cutoff_ns"][position])
        cutoff = pd.Timestamp(cutoff_ns)
        quality = int(records["quality_tier"][position])
        key = EpisodeKey(
            InstrumentKey("nasdaq", symbol), cutoff,
            int(row["lookback"]), row["representation_version"],
        )
        components = match["component_distances"]
        expected_total = math.fsum(
            weights[name] * components[name] for name in sorted(weights)
        )
        if not all((
            symbol == match["symbol"], cutoff.isoformat() == match["cutoff"],
            quality == QUALITY_CODES.get(match["quality_tier"]),
            key.id == match["episode_id"], cutoff_ns <= latest_ns,
            cutoff_ns < int(pd.Timestamp(row["cutoff"]).value),
            match["episode_id"] != row["episode_id"],
            set(components) == certificate_verifier.EXPECTED_COMPONENTS,
            all(type(number) in (int, float) and math.isfinite(number) and number >= 0
                for number in components.values()),
            math.isclose(match["total_distance"], expected_total,
                         rel_tol=1e-12, abs_tol=1e-12),
            _alignment_valid(match["alignment"], int(row["lookback"]),
                             int(row["lookback"])),
        )):
            raise CompositeBatchVerificationError("analogue binding differs")
        if symbol == row["symbol"]:
            same_symbol += 1
            if cutoff_ns >= query_start_ns:
                raise CompositeBatchVerificationError("same-symbol overlap differs")
    ranking = [(match["total_distance"], match["episode_id"]) for match in matches]
    if ranking != sorted(ranking) or len({match["symbol"] for match in matches}) != base.TOP_K:
        raise CompositeBatchVerificationError("analogue ranking/diversity differs")
    return {
        "query_id": row["episode_id"], "case_digest": value["case_digest"],
        "certificate_result_digest": certificate["result_digest"],
        "input_digest": expected_input,
        "neighbor_position_digest": stable_hash([int(item) for item in positions]),
        "same_symbol_neighbors": same_symbol,
        "eligible_candidates": certificate["eligible_candidates"],
        "exact_evaluated": certificate["exact_evaluated"],
        "safely_pruned": certificate["safely_pruned"],
    }


def _rerun_sample(
    repository: Path, rows: Sequence[Mapping[str, Any]], cases_root: Path,
    store_root: Path,
) -> list[dict[str, Any]]:
    context = multiprocessing.get_context("spawn")
    observations = []
    with ProcessPoolExecutor(max_workers=len(rows), mp_context=context) as executor:
        future_rows = {
            executor.submit(
                kernel._run_group, str(repository), str(store_root),
                (dict(row),), producer.THREADS_PER_PROCESS,
            ): row for row in rows
        }
        for future in as_completed(future_rows):
            row = future_rows[future]
            worker = future.result()
            if worker.get("queries") != 1 or worker.get("swap_kib") != 0 \
                    or len(worker.get("cases", [])) != 1:
                raise CompositeBatchVerificationError("sample rerun worker differs")
            actual = worker["cases"][0]
            published = base._read(producer._case_path(cases_root, row["episode_id"]))
            if kernel._case_semantics(actual) != kernel._case_semantics(
                published["retrieval"]
            ):
                raise CompositeBatchVerificationError("sampled rerun differs")
            observations.append({
                "query_id": row["episode_id"],
                "semantic_digest": stable_hash(kernel._case_semantics(actual)),
            })
    return sorted(observations, key=lambda item: item["query_id"])


def verify(repository: Path) -> dict[str, Any]:
    started = perf_counter()
    repository = repository.resolve(strict=True)
    if _git(repository, "status", "--porcelain"):
        raise CompositeBatchVerificationError("composite verifier requires clean commit")
    verifier_commit = _git(repository, "rev-parse", "HEAD")
    verifier_relative = str(Path(__file__).resolve().relative_to(repository))
    verifier_sha256 = _sha(repository / verifier_relative)
    blob = subprocess.run(
        ["git", "show", f"{verifier_commit}:{verifier_relative}"],
        cwd=repository, capture_output=True, check=False,
    )
    if blob.returncode or sha256(blob.stdout).hexdigest() != verifier_sha256:
        raise CompositeBatchVerificationError("verifier Git binding differs")

    preregistration = base._read(repository / producer.PREREGISTRATION_RELATIVE)
    registry, rows, resident = producer.validate_preregistration(
        repository, preregistration,
    )
    root = repository / producer.OUTPUT_RELATIVE
    if root.is_symlink() or base._read(root / "CONTRACT.json") != preregistration:
        raise CompositeBatchVerificationError("producer contract differs")
    attempts = _attempt_bindings(root, preregistration)
    result = base._read(root / "RESULT.json")
    cases_root = root / "cases"
    paths = [path for path in cases_root.iterdir() if not path.name.startswith(".wf03-")]
    expected_names = {f"{row['episode_id']}.json" for row in rows}
    terminal_attempt = attempts.get(str(result.get("terminal_attempt_id")))
    if not _seal_valid(result, "result_digest") \
            or {path.name for path in paths} != expected_names \
            or any(path.is_symlink() or not path.is_file() for path in paths) \
            or terminal_attempt is None \
            or terminal_attempt.get("_terminal_name") != "COMPLETE.json" \
            or terminal_attempt.get("_terminal", {}).get("result_digest") \
                != result.get("result_digest"):
        raise CompositeBatchVerificationError("producer terminal layout differs")

    packed = load_packed_generation(
        Path(resident["store_root"]), base.GENERATION_ID,
        expected_provenance_digest=base.PROVENANCE_DIGEST,
        verify_content=True, validate_records=True,
    )
    records = _inventory(packed.rows, packed.overflow)
    order = np.argsort(records["episode_id"], kind="stable")
    sorted_ids = records["episode_id"][order]
    raw_source = source_from_spec(
        load_config(repository / base.CONFIG_RELATIVE).datasets["nasdaq"]
    )
    source = CachedOHLCVSource(raw_source, max_entries=None)
    observations = []
    manifest = []
    semantic_digests = []
    for row in rows:
        path = producer._case_path(cases_root, row["episode_id"])
        value = base._read(path)
        observations.append(_independent_case(
            value, row, preregistration, attempts, records, packed.symbols,
            order, sorted_ids, source, packed.manifest,
        ))
        manifest.append({
            "query_id": value["query_id"], "case_digest": value["case_digest"],
            "sha256": _sha(path),
        })
        semantic_digests.append(value["semantic_digest"])
    if not all((
        result.get("schema_version")
            == "m04r14-t14-10-wf03-composite-batch-result-v2",
        result.get("status") == "complete", result.get("passed") is True,
        result.get("queries") == producer.EXPECTED_QUERIES,
        result.get("scored_queries") == producer.EXPECTED_SCORED_QUERIES,
        result.get("warmup_queries") == producer.EXPECTED_WARMUP_QUERIES,
        result.get("months") == preregistration["inventory"]["months"],
        result.get("preregistration_digest") == preregistration["preregistration_digest"],
        result.get("case_manifest_digest") == stable_hash(manifest),
        result.get("case_semantic_digest") == stable_hash(semantic_digests),
        result.get("minimum_eligible_candidates")
            == min(item["eligible_candidates"] for item in observations),
        result.get("maximum_eligible_candidates")
            == max(item["eligible_candidates"] for item in observations),
        result.get("all_worker_swap_zero") is True,
        result.get("attempts") == len(attempts),
        result.get("outcomes_or_labels_used") is False,
        result.get("historical_walk_forward_query_outcomes_opened") is False,
        result.get("final_period_result_opened") is False,
        result.get("production_promotion_authorized") is False,
        result.get("independent_verification_authorized") is True,
    )):
        raise CompositeBatchVerificationError("producer aggregate differs")

    authority = base._read(repository / AUTHORITY_RELATIVE)
    published_authority = base._read(
        producer._case_path(cases_root, AUTHORITY_QUERY_ID)
    )["retrieval"]
    if published_authority["matches"] != authority["matches"] \
            or published_authority["certificate"]["result_digest"] \
                != authority["certificate"]["result_digest"]:
        raise CompositeBatchVerificationError("frozen composite authority differs")

    sample_ids = select_rerun_sample(rows)
    by_id = {row["episode_id"]: row for row in rows}
    reruns = _rerun_sample(
        repository, [by_id[item] for item in sample_ids], cases_root,
        Path(resident["store_root"]),
    )
    gates = {
        "producer_runtime_preregistration_and_terminal_valid": True,
        "packed_generation_fully_rehashed_and_record_valid": True,
        "all_3936_case_seals_inputs_and_certificates_valid": True,
        "all_78720_links_resolve_and_are_causal": True,
        "all_seven_components_and_alignment_paths_valid": True,
        "frozen_true_composite_authority_exact": True,
        "twelve_fold_stratified_reruns_exact": len(reruns) == 12,
        "zero_worker_swap": True,
        "outcomes_or_labels_excluded": True,
    }
    if not all(gates.values()):
        raise CompositeBatchVerificationError("composite verification gate failed")
    state = {
        "schema_version": "m04r14-t14-10-wf03-composite-batch-verification-v2",
        "status": "complete", "passed": True, "gates": gates,
        "producer_result_digest": result["result_digest"],
        "preregistration_digest": preregistration["preregistration_digest"],
        "verifier_commit": verifier_commit,
        "verifier_runtime_sha256": verifier_sha256,
        "queries_verified": len(observations),
        "links_verified": len(observations) * base.TOP_K,
        "same_symbol_links_verified": sum(
            item["same_symbol_neighbors"] for item in observations
        ),
        "case_observation_digest": stable_hash(observations),
        "rerun_sample_ids": sample_ids,
        "rerun_digest": stable_hash(reruns), "rerun_queries": len(reruns),
        "authority_query_id": AUTHORITY_QUERY_ID,
        "elapsed_seconds": perf_counter() - started,
        "outcomes_or_labels_used": False,
        "historical_walk_forward_query_outcomes_opened": False,
        "final_period_result_opened": False,
        "production_promotion_authorized": False,
    }
    return base._sealed(state, "verification_digest")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    value = verify(args.repository)
    if not args.dry_run:
        root = args.repository.resolve() / OUTPUT_RELATIVE
        if root.exists() or root.is_symlink():
            raise CompositeBatchVerificationError("verification output exists")
        root.mkdir(parents=True)
        base._atomic(root / "VERIFIED.json", value)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
