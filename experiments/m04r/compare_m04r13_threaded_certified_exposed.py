"""Post-open comparator for the truth-blind M04R-13 producer.

All producer evidence is recursively validated before the durable
``RESULTS_OPENED.json`` marker is created and fsynced.  No authority path is
accepted by or available to the producer process.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import stat
from typing import Any, Callable, Mapping

from math import isfinite

import numpy as np
import pandas as pd

from market_analogues.certified_packed_search import certified_packed_search_contract
from market_analogues.packed_bound_search import packed_bound_search_contract
from market_analogues.packed_bound_search import (
    BRANCH_AWARE_SEARCH_SCHEMA_VERSION, BoundProposalReport, _eligible_mask,
    bound_proposal_candidate_digest, packed_branch_aware_lower_bounds,
    scan_packed_bound_threshold,
)
from market_analogues.packed_bound_store import TIER_NAMES, load_packed_generation
from market_analogues.types import stable_hash

import m04r13_threaded_certified_exposed as producer


RESULTS_OPENED_SCHEMA = "m04r13-threaded-certified-results-opened-v1"
COMPARISON_SCHEMA = "m04r13-threaded-certified-comparison-v1"
COMPARISON_SEAL_SCHEMA = "m04r13-threaded-certified-comparison-seal-v1"
AUTHORITY_VERIFICATION_SHA256 = (
    "b8d879c0f00cf848d39dbd0171d67d9b138f7ebeadd206137566cdddf98a36ee"
)
AUTHORITY_VERIFICATION_DIGEST = (
    "99d11756ed7542714635eca3fb75a43faed93881121d1f22d8d87426f4d6b190"
)
AUTHORITY_MATRIX_DIGEST = (
    "ae60b7cda755c2deabc7bf9335701d34eea73073664582933eb5ba984e7f9b29"
)
AUTHORITY_SEAL_DIGEST = (
    "83501b4cf620607c4ee14cc837d1d5c242460dc867aca22a51b3fbbf924be2c9"
)
AUTHORITY_CASE_BINDINGS = {
    "3307023dbe2164d025e788da": (
        "1aeeddea785923e7e090adc7ffd57fcac410ae42a336c2d03d72236ec01e5b0e",
        "c78e1e8a4f21aaedb8df88ffa9514a1fe0de70c15d67ab010d60b4127a1de91b",
    ),
    "3618af07dedd52fb3bdb1ccd": (
        "c256fe6cba1482a1a090ebfbf89a337700f43bb8150a7c280d7ffe405ac65ed0",
        "b8bcff4c2f1b69109b3d5514d474e2cca4edf0b952f715ab54f4f3cc94dcc127",
    ),
    "9d7365581643bd93e85beb67": (
        "f5f96b97cb2a140abbe2275c12a7aff3cb08b9c462df374e26da00afc59e9e3b",
        "1e704d8a73667eed03488ac990370f274daf06855cac1a1e110c44f09f058141",
    ),
    "99a0838725a09570b4a075ff": (
        "35dc2f84bdc092aec063a6a1c6f8d188489e5c8ed3cbaab3ff16ef94bfaae968",
        "f46cb50c6dbbc3b8af8c521e605e941170bc817588ca9056a82de4fc8945f2f7",
    ),
}


class ComparisonError(ValueError):
    pass


def _without(value: Mapping[str, Any], omitted: set[str]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key not in omitted}


def _read_sha_json(path: Path, expected_sha256: str) -> dict[str, Any]:
    """Hash and strictly parse one identity-stable, no-follow byte read."""
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise ComparisonError(f"cannot open authority JSON: {path}") from exc
    try:
        before, chunks = os.fstat(descriptor), []
        while block := os.read(descriptor, 1024 * 1024):
            chunks.append(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = lambda value: (
        value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns,
        value.st_ctime_ns, value.st_mode,
    )
    raw = b"".join(chunks)
    if (not stat.S_ISREG(before.st_mode) or identity(before) != identity(after)
            or sha256(raw).hexdigest() != expected_sha256):
        raise ComparisonError(f"authority JSON bytes differ: {path}")
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise ComparisonError(f"duplicate authority JSON key: {key}")
            result[key] = value
        return result
    def invalid(value: str) -> Any:
        raise ComparisonError(f"non-finite authority JSON number: {value}")
    try:
        payload = json.loads(
            raw, object_pairs_hook=pairs, parse_constant=invalid,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ComparisonError(f"invalid authority JSON: {path}") from exc
    if type(payload) is not dict:
        raise ComparisonError(f"authority JSON object differs: {path}")
    return payload


def _nonnegative_int(value: Any) -> bool:
    return type(value) is int and value >= 0


def _exact_tree(root: Path, files: set[str], directories: set[str]) -> None:
    if root.is_symlink() or not root.is_dir():
        raise ComparisonError("producer root is linked or absent")
    observed_files: set[str] = set()
    observed_directories: set[str] = set()
    for path in root.rglob("*"):
        mode = path.lstat().st_mode
        relative = str(path.relative_to(root))
        if stat.S_ISLNK(mode):
            raise ComparisonError("producer tree contains a symlink")
        if stat.S_ISREG(mode):
            observed_files.add(relative)
        elif stat.S_ISDIR(mode):
            observed_directories.add(relative)
        else:
            raise ComparisonError("producer tree contains a special entry")
    if observed_files != files or observed_directories != directories:
        raise ComparisonError("producer tree differs")


def _strict_payload_digest(
    payload: Mapping[str, Any], *, digest_key: str, omitted: set[str],
) -> None:
    if payload.get(digest_key) != stable_hash(_without(payload, omitted | {digest_key})):
        raise ComparisonError(f"{digest_key} differs")


def _validate_proposal(
    value: Mapping[str, Any], query_id: str, order: str,
) -> BoundProposalReport:
    required = {
        "schema_version", "generation_id", "query_episode_id", "candidates",
        "rows_scanned", "eligible_rows", "eligible_main_rows",
        "eligible_overflow_rows", "route_counts", "route_quotas", "block_rows",
        "block_order", "elapsed_seconds", "peak_rss_mb", "candidate_digest",
        "result_digest", "contract_digest", "input_digest",
    }
    if set(value) != required or type(value.get("candidates")) is not list:
        raise ComparisonError("proposal fields differ")
    candidate_keys = {
        "episode_id", "symbol", "cutoff_ns", "quality_tier", "lower_bound_hex",
        "routes", "overflow_fallback",
    }
    if any(set(row) != candidate_keys for row in value["candidates"]):
        raise ComparisonError("proposal candidate fields differ")
    try:
        report = producer._proposal_report(value)
    except (KeyError, TypeError, ValueError) as exc:
        raise ComparisonError("proposal encoding differs") from exc
    contract = packed_bound_search_contract(branch_aware=True)["digest"]
    deterministic = {
        "schema_version": BRANCH_AWARE_SEARCH_SCHEMA_VERSION,
        "contract_digest": contract, "generation_id": producer.GENERATION_ID,
        "query_episode_id": query_id, "rows_scanned": report.rows_scanned,
        "eligible_rows": report.eligible_rows,
        "eligible_main_rows": report.eligible_main_rows,
        "eligible_overflow_rows": report.eligible_overflow_rows,
        "route_counts": dict(report.route_counts),
        "route_quotas": dict(report.route_quotas),
        "candidate_digest": report.candidate_digest,
        "real_forward_outcomes_accessed": False,
        "input_digest": report.input_digest,
    }
    try:
        identifiers_valid = all(
            len(bytes.fromhex(row.episode_id)) == 12 for row in report.candidates
        )
    except ValueError:
        identifiers_valid = False
    if not all((
        report.schema_version == BRANCH_AWARE_SEARCH_SCHEMA_VERSION,
        report.contract_digest == contract,
        report.generation_id == producer.GENERATION_ID,
        report.query_episode_id == query_id,
        report.route_quotas == {"composite": producer.PROPOSAL_QUOTA},
        report.route_counts == {"composite": len(report.candidates)},
        report.eligible_rows == report.eligible_main_rows + report.eligible_overflow_rows,
        0 <= report.eligible_rows <= report.rows_scanned,
        len(report.candidates) == min(producer.PROPOSAL_QUOTA, report.eligible_rows),
        report.block_order == order,
        report.block_rows == (4_096 if order == "forward" else 4_097),
        isfinite(report.elapsed_seconds) and report.elapsed_seconds >= 0,
        isfinite(report.peak_rss_mb) and report.peak_rss_mb >= 0,
        identifiers_valid,
        all(row.quality_tier in {"A", "B"}
            and row.routes == ("composite",)
            and isfinite(row.lower_bound) and row.lower_bound >= 0
            for row in report.candidates),
        list(report.candidates) == sorted(
            report.candidates, key=lambda row: (row.lower_bound, row.episode_id)),
        report.candidate_digest
        == bound_proposal_candidate_digest(report.candidates),
        report.result_digest == stable_hash(deterministic),
    )):
        raise ComparisonError("proposal reconstruction differs")
    return report


def _validate_certificate(
    certificate: Mapping[str, Any], matches: list[dict[str, Any]], query_id: str,
) -> None:
    try:
        producer.validate_certificate_and_matches(certificate, matches, query_id)
    except producer.HarnessError as exc:
        raise ComparisonError("certificate/match reconstruction differs") from exc
    required = {
        "schema_version", "contract_digest", "generation_id", "query_episode_id",
        "input_digest", "eligible_candidates", "exact_evaluated", "safely_pruned",
        "stopped_early", "stop_threshold", "next_lower_bound",
        "maximum_quantized_bound_excess", "materialization_groups", "sparse_symbols",
        "batch_symbols", "rounds", "result_digest", "elapsed_seconds",
        "native_bound_accounting", "minimum_native_pruned_bound",
        "threshold_closure_passes",
    }
    if set(certificate) != required or len(matches) != 20:
        raise ComparisonError("certificate fields differ")
    match_keys = {
        "episode_id", "symbol", "cutoff", "total_distance",
        "component_distances", "alignment", "quality_tier",
    }
    if any(set(row) != match_keys for row in matches):
        raise ComparisonError("match fields differ")
    contract = certified_packed_search_contract(
        requested_positions=True, vector_lower_bounds=True,
        deferred_alignments=True, compact_scored=True,
        native_bound_deferral=True, streaming_threshold_closure=True,
        branch_aware_packed_bounds=True,
    )
    accounting = certificate["native_bound_accounting"]
    if set(accounting) != {
        "native_bound_evaluated", "exact_dtw_evaluated", "native_bound_pruned",
        "packed_bound_pruned",
    }:
        raise ComparisonError("native accounting fields differ")
    rounds = certificate["rounds"]
    round_keys = {
        "frontier_rows", "exact_rows", "next_lower_bound", "constrained_threshold",
        "selected_rows", "certified", "proposal_digest",
    }
    if (type(rounds) is not list or not rounds
            or any(set(row) != round_keys or not all((
                _nonnegative_int(row["frontier_rows"]) and row["frontier_rows"] > 0,
                _nonnegative_int(row["exact_rows"]),
                _nonnegative_int(row["selected_rows"]) and row["selected_rows"] <= 20,
                type(row["certified"]) is bool,
                isfinite(row["constrained_threshold"])
                and row["constrained_threshold"] >= 0,
                row["next_lower_bound"] is None or (
                    isfinite(row["next_lower_bound"])
                    and row["next_lower_bound"] >= 0
                ),
                isinstance(row["proposal_digest"], str),
            )) for row in rounds)):
        raise ComparisonError("logical frontier rounds differ")
    closures = certificate["threshold_closure_passes"]
    closure_keys = {
        "lower_exclusive", "upper_inclusive", "admitted_rows",
        "cumulative_native_bound_evaluated", "cumulative_exact_dtw_evaluated",
        "selected_rows", "resulting_threshold",
        "minimum_packed_unclassified_bound", "minimum_native_pruned_bound",
        "excluded_prefix_digest", "admitted_set_digest", "scan_result_digest",
        "certified",
    }
    if type(closures) is not list or any(
        set(row) != closure_keys or not all((
            _nonnegative_int(row["admitted_rows"]),
            _nonnegative_int(row["cumulative_native_bound_evaluated"]),
            _nonnegative_int(row["cumulative_exact_dtw_evaluated"]),
            row["cumulative_exact_dtw_evaluated"]
            <= row["cumulative_native_bound_evaluated"],
            _nonnegative_int(row["selected_rows"]) and row["selected_rows"] <= 20,
            type(row["certified"]) is bool,
        )) for row in closures
    ):
        raise ComparisonError("threshold closure counts differ")
    deterministic = {
        "schema_version": contract["schema_version"],
        "contract_digest": contract["digest"],
        "generation_id": producer.GENERATION_ID,
        "query_episode_id": query_id, "input_digest": certificate["input_digest"],
        "eligible_candidates": certificate["eligible_candidates"],
        "exact_evaluated": certificate["exact_evaluated"],
        "safely_pruned": certificate["safely_pruned"],
        "stopped_early": certificate["stopped_early"],
        "stop_threshold_hex": certificate["stop_threshold"].hex(),
        "next_lower_bound_hex": (
            certificate["next_lower_bound"].hex()
            if certificate["next_lower_bound"] is not None else None
        ),
        "maximum_quantized_bound_excess_hex":
            certificate["maximum_quantized_bound_excess"].hex(),
        "rounds": rounds,
        "matches": [{
            "episode_id": row["episode_id"],
            "total_hex": row["total_distance"].hex(),
            "components": {
                key: value.hex()
                for key, value in sorted(row["component_distances"].items())
            },
            "alignment": row["alignment"],
        } for row in matches],
        "real_forward_outcomes_accessed": False,
        "native_bound_accounting": accounting,
        "minimum_native_pruned_bound_hex": (
            certificate["minimum_native_pruned_bound"].hex()
            if certificate["minimum_native_pruned_bound"] is not None else None
        ),
        "threshold_closure_passes": closures,
    }
    count_fields = (
        "eligible_candidates", "exact_evaluated", "safely_pruned",
        "materialization_groups", "sparse_symbols", "batch_symbols",
    )
    try:
        match_distances_valid = all(
            isfinite(row["total_distance"]) and row["total_distance"] >= 0
            and all(isfinite(value) and value >= 0
                    for value in row["component_distances"].values())
            and all(type(index) is int and index >= 0
                    for pair in row["alignment"] for index in pair)
            for row in matches
        )
    except (TypeError, ValueError):
        match_distances_valid = False
    if not all((
        certificate["schema_version"] == contract["schema_version"],
        certificate["contract_digest"] == contract["digest"],
        certificate["generation_id"] == producer.GENERATION_ID,
        certificate["query_episode_id"] == query_id,
        all(_nonnegative_int(certificate[key]) for key in count_fields),
        type(certificate["stopped_early"]) is bool,
        certificate["eligible_candidates"]
        == certificate["exact_evaluated"] + certificate["safely_pruned"],
        accounting["native_bound_evaluated"]
        == accounting["exact_dtw_evaluated"] + accounting["native_bound_pruned"],
        all(_nonnegative_int(value) for value in accounting.values()),
        certificate["eligible_candidates"]
        == accounting["native_bound_evaluated"] + accounting["packed_bound_pruned"],
        certificate["exact_evaluated"] == accounting["exact_dtw_evaluated"],
        certificate["exact_evaluated"] >= len(matches),
        certificate["materialization_groups"]
        == certificate["sparse_symbols"] + certificate["batch_symbols"],
        all(row["selected_rows"] <= row["exact_rows"]
            <= certificate["eligible_candidates"] for row in rounds),
        all(row["cumulative_native_bound_evaluated"]
            <= certificate["eligible_candidates"] for row in closures),
        isfinite(certificate["stop_threshold"])
        and certificate["stop_threshold"] >= 0,
        certificate["next_lower_bound"] is None or (
            isfinite(certificate["next_lower_bound"])
            and certificate["next_lower_bound"] >= 0
        ),
        isfinite(certificate["maximum_quantized_bound_excess"])
        and 0 <= certificate["maximum_quantized_bound_excess"]
        <= producer.TOLERANCE,
        isfinite(certificate["elapsed_seconds"])
        and certificate["elapsed_seconds"] >= 0,
        match_distances_valid,
        len({row["episode_id"] for row in matches}) == 20,
        [(row["total_distance"], row["episode_id"]) for row in matches]
        == sorted((row["total_distance"], row["episode_id"])
                  for row in matches),
        certificate["result_digest"] == stable_hash(deterministic),
    )):
        raise ComparisonError("certificate reconstruction differs")


def _validate_round_proposal_bindings(
    certificate: Mapping[str, Any], forward: BoundProposalReport,
) -> None:
    exact = True
    for row in certificate["rounds"]:
        frontier_rows = row["frontier_rows"]
        packed_next = (
            forward.candidates[frontier_rows].lower_bound
            if frontier_rows < forward.eligible_rows else None
        )
        effective_next = row["next_lower_bound"]
        if not all((
            row["proposal_digest"] == bound_proposal_candidate_digest(
                forward.candidates[:min(frontier_rows + 1, forward.eligible_rows)]
            ),
            packed_next is None or (
                effective_next is not None
                and effective_next <= packed_next + producer.TOLERANCE
            ),
        )):
            exact = False
            break
    if not certificate["threshold_closure_passes"]:
        final_frontier = certificate["rounds"][-1]["frontier_rows"]
        final_packed = (
            forward.candidates[final_frontier].lower_bound
            if final_frontier < forward.eligible_rows else None
        )
        final_native = certificate["minimum_native_pruned_bound"]
        reconstructed_final_next = min(
            (value for value in (final_packed, final_native) if value is not None),
            default=None,
        )
        exact = exact and all((
            certificate["rounds"][-1]["next_lower_bound"]
            == reconstructed_final_next,
            certificate["next_lower_bound"] == reconstructed_final_next,
        ))
    if not exact:
        raise ComparisonError("certified proposal cross-binding differs")


def reconstruct_match_universe(
    prereg: Mapping[str, Any], requested: Mapping[str, set[str]],
    case_payloads: list[Mapping[str, Any]],
) -> tuple[
    dict[str, dict[str, dict[str, Any]]],
    dict[str, list[dict[str, Any]]],
]:
    """Authenticate retained IDs against the durable truth-free packed source."""
    repository = Path(__file__).resolve().parents[2]
    roots = prereg["roots"]
    store_root = Path(roots["source_full_root"]) / "store"
    identity_before = _source_generation_identity(store_root)
    loaded = load_packed_generation(
        store_root, producer.GENERATION_ID,
        expected_provenance_digest=producer.PROVENANCE_DIGEST,
        verify_content=True, validate_records=True,
    )
    targets = set().union(*requested.values()) if requested else set()
    try:
        encoded = {bytes.fromhex(value) for value in targets}
    except ValueError as exc:
        raise ComparisonError("retained match episode ID encoding differs") from exc
    found: dict[str, tuple[Any, bool]] = {}
    encoded_array = np.asarray(
        [np.void(value) for value in encoded], dtype="V12",
    )
    for array, overflow in ((loaded.rows, False), (loaded.overflow, True)):
        for first in range(0, len(array), 65_536):
            block = array[first:first + 65_536]
            selected = block[
                np.isin(block["episode_id"], encoded_array)
            ] if len(encoded_array) else block[:0]
            for record in selected:
                identifier = bytes(record["episode_id"])
                episode_id = identifier.hex()
                if episode_id in found:
                    raise ComparisonError("retained episode occurs twice in packed universe")
                found[episode_id] = (record.copy(), overflow)
    registry_digest, cases = producer._registry_cases(
        repository, Path(roots["registry_root"]),
    )
    if registry_digest != prereg["registry_digest"]:
        raise ComparisonError("registry differs during universe authentication")
    inputs = producer.Inputs(
        repository, Path(prereg["config_path"]), Path(roots["registry_root"]),
        Path(roots["source_full_root"]) / "store", Path(roots["resident_root"]),
        Path(roots["output_root"]), producer.GENERATION_ID,
        producer.PROVENANCE_DIGEST, prereg["reserve_bytes"], registry_digest,
        cases, prereg["preregistration_digest"],
    )
    result: dict[str, dict[str, dict[str, Any]]] = {}
    closure_evidence: dict[str, list[dict[str, Any]]] = {}
    payload_by_query = {
        str(value.get("query_episode_id")): value for value in case_payloads
    }
    for case in cases:
        _source, _episode, _request, query = producer._case_context(inputs, case)
        symbol_id = (
            loaded.symbols.index(query.symbol)
            if query.symbol in loaded.symbols else None
        )
        rows: dict[str, dict[str, Any]] = {}
        for episode_id in requested.get(case.query_id, set()):
            located = found.get(episode_id)
            if located is None:
                continue
            record, overflow = located
            single = np.asarray([record], dtype=(
                loaded.overflow.dtype if overflow else loaded.rows.dtype
            ))
            if not bool(_eligible_mask(single, query, symbol_id)[0]):
                continue
            numeric_symbol = int(record["symbol_id"])
            if numeric_symbol >= len(loaded.symbols):
                raise ComparisonError("retained match symbol ID exceeds dictionary")
            bound = 0.0 if overflow else float(
                packed_branch_aware_lower_bounds(
                    query.representation, single,
                ).totals[0]
            )
            tier = TIER_NAMES.get(int(record["quality_tier"]))
            if tier is None:
                raise ComparisonError("retained match quality tier differs")
            rows[episode_id] = {
                "episode_id": episode_id, "symbol": loaded.symbols[numeric_symbol],
                "cutoff_ns": int(record["cutoff_ns"]), "quality_tier": tier,
                "branch_aware_bound": bound, "overflow": overflow,
            }
        result[case.query_id] = rows
        closure_evidence[case.query_id] = []
        payload = payload_by_query.get(case.query_id, {})
        closures = payload.get("certificate", {}).get(
            "threshold_closure_passes", [],
        )
        if closures:
            prefix = payload.get("forward_proposal", {}).get("candidates", [])[
                :producer.MAXIMUM_FRONTIER
            ]
            excluded = frozenset(str(row["episode_id"]) for row in prefix)
            for closure in closures:
                report = scan_packed_bound_threshold(
                    store_root, producer.GENERATION_ID, query,
                    lower_exclusive=closure["lower_exclusive"],
                    upper_inclusive=closure["upper_inclusive"],
                    excluded_episode_ids=excluded, block_rows=4_097,
                    block_order="reverse", branch_aware=True,
                    verify_content=False,
                )
                closure_evidence[case.query_id].append({
                    "excluded_prefix_digest": report.exclusions_digest,
                    "admitted_rows": report.admitted_rows,
                    "admitted_set_digest": report.admitted_set_digest,
                    "scan_result_digest": report.result_digest,
                    "minimum_packed_unclassified_bound": report.minimum_above_upper,
                    "eligible_rows": report.eligible_rows,
                    "excluded_eligible_rows": report.excluded_eligible_rows,
                })
    if _source_generation_identity(store_root) != identity_before:
        raise ComparisonError("durable packed source identity changed during validation")
    return result, closure_evidence


def _source_generation_identity(store_root: Path) -> dict[str, Any]:
    generation = store_root / "generations" / producer.GENERATION_ID
    if (store_root.is_symlink() or generation.is_symlink()
            or store_root.resolve() != store_root.absolute()
            or generation.resolve() != generation.absolute()
            or not store_root.is_dir() or not generation.is_dir()):
        raise ComparisonError("durable packed source root differs")
    expected = {"manifest.json", "bound-rows.bin", "overflow-exact-fallback.bin"}
    entries = {path.name: path for path in generation.iterdir()}
    if set(entries) != expected:
        raise ComparisonError("durable packed generation tree differs")
    result: dict[str, Any] = {
        "store": tuple(getattr(store_root.lstat(), field) for field in (
            "st_dev", "st_ino", "st_mtime_ns", "st_ctime_ns", "st_mode",
        )),
        "generation": tuple(getattr(generation.lstat(), field) for field in (
            "st_dev", "st_ino", "st_mtime_ns", "st_ctime_ns", "st_mode",
        )),
    }
    for name, path in entries.items():
        value = path.lstat()
        if stat.S_ISLNK(value.st_mode) or not stat.S_ISREG(value.st_mode):
            raise ComparisonError("durable packed generation contains linked/non-file data")
        result[name] = (
            value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns,
            value.st_ctime_ns, value.st_mode,
        )
    return result


def _validate_closure_scan_evidence(
    certificate: Mapping[str, Any], forward: BoundProposalReport,
    reports: list[Mapping[str, Any]],
) -> None:
    closures = certificate["threshold_closure_passes"]
    if len(reports) != len(closures):
        raise ComparisonError("closure scan evidence count differs")
    excluded_digest = stable_hash(sorted(
        row.episode_id for row in forward.candidates[:producer.MAXIMUM_FRONTIER]
    ))
    for closure, report in zip(closures, reports):
        if not all((
            closure["excluded_prefix_digest"] == excluded_digest,
            report.get("excluded_prefix_digest") == excluded_digest,
            report.get("admitted_rows") == closure["admitted_rows"],
            report.get("admitted_set_digest") == closure["admitted_set_digest"],
            report.get("scan_result_digest") == closure["scan_result_digest"],
            report.get("minimum_packed_unclassified_bound")
            == closure["minimum_packed_unclassified_bound"],
            report.get("eligible_rows") == certificate["eligible_candidates"],
            report.get("excluded_eligible_rows")
            == min(producer.MAXIMUM_FRONTIER, forward.eligible_rows),
        )):
            raise ComparisonError("closure scan evidence differs")


def _validate_match_universe_cross_binding(
    matches: list[dict[str, Any]], forward: BoundProposalReport,
    closures: list[dict[str, Any]], universe: Mapping[str, Mapping[str, Any]],
) -> None:
    proposal = {row.episode_id: row for row in forward.candidates}
    if len(proposal) != len(forward.candidates):
        raise ComparisonError("proposal episode identity differs")
    for match in matches:
        episode_id = match["episode_id"]
        source = universe.get(episode_id)
        if source is None:
            raise ComparisonError("retained match is absent or ineligible in packed universe")
        metadata_exact = all((
            source.get("episode_id") == episode_id,
            source.get("symbol") == match["symbol"],
            source.get("cutoff_ns") == pd.Timestamp(match["cutoff"]).value,
            source.get("quality_tier") == match["quality_tier"],
            type(source.get("branch_aware_bound")) is float,
            isfinite(source["branch_aware_bound"]),
            source["branch_aware_bound"] >= 0,
            type(source.get("overflow")) is bool,
        ))
        if not metadata_exact:
            raise ComparisonError("retained match packed metadata differs")
        prefix = proposal.get(episode_id)
        if prefix is not None:
            if not all((
                prefix.symbol == match["symbol"],
                prefix.cutoff_ns == source["cutoff_ns"],
                prefix.quality_tier == match["quality_tier"],
                prefix.overflow_fallback == source["overflow"],
            )):
                raise ComparisonError("retained proposal metadata differs")
            continue
        if not closures:
            raise ComparisonError("non-fallback match is outside proposal prefix")
        bound = source["branch_aware_bound"]
        admitted = any(
            (row["lower_exclusive"] is None or bound > row["lower_exclusive"])
            and bound <= row["upper_inclusive"]
            for row in closures
        )
        if not admitted:
            raise ComparisonError("rescued match is outside closure admission bands")


def _validate_case(
    payload: Mapping[str, Any], query_id: str, prereg_digest: str,
    resident: Mapping[str, Any], binding: Mapping[str, Any],
    universe: Mapping[str, Mapping[str, Any]],
    closure_reports: list[Mapping[str, Any]],
) -> None:
    required = {
        "schema_version", "status", "development_only", "truth_opened",
        "production_promotion_authorized", "real_forward_outcomes_accessed",
        "preregistration_digest", "registry_case_id", "query_episode_id",
        "query_binding",
        "resident_identity_digest", "lease_digests", "forward_proposal",
        "reverse_proposal", "proposal_semantic_exact", "certificate", "matches",
        "rounds", "certified", "streaming_fallback_used", "metrics",
        "semantic_passed", "performance_passed", "created_at", "result_digest",
    }
    if set(payload) != required:
        raise ComparisonError("case fields differ")
    if not all((
        payload["schema_version"] == producer.CASE_SCHEMA,
        payload["status"] == "truth_blind_case_complete",
        payload["development_only"] is True,
        payload["truth_opened"] is False,
        payload["production_promotion_authorized"] is False,
        payload["real_forward_outcomes_accessed"] is False,
        payload["preregistration_digest"] == prereg_digest,
        payload["registry_case_id"] == producer.FROZEN_CASE_IDS[
            producer.FROZEN_QUERY_IDS.index(query_id)
        ],
        payload["query_episode_id"] == query_id,
        payload["resident_identity_digest"] == resident["identity_digest"],
        payload["query_binding"] == dict(binding),
        producer._is_utc_iso_timestamp(payload["created_at"]),
        payload["proposal_semantic_exact"] is True,
        payload["certified"] is True,
        payload["semantic_passed"] is True,
        len(payload["lease_digests"]) == 5,
        all(value == resident["lease"]["lease_digest"]
            for value in payload["lease_digests"]),
        len(payload["matches"]) == 20,
    )):
        raise ComparisonError("case semantic fields differ")
    _strict_payload_digest(
        payload, digest_key="result_digest", omitted={"created_at"},
    )
    forward = _validate_proposal(payload["forward_proposal"], query_id, "forward")
    _validate_proposal(payload["reverse_proposal"], query_id, "reverse")
    if not all((
        payload["forward_proposal"]["input_digest"]
        == binding["packed_query_input_digest"],
        payload["reverse_proposal"]["input_digest"]
        == binding["packed_query_input_digest"],
        payload["certificate"]["input_digest"] == binding["certified_input_digest"],
        payload["certificate"]["eligible_candidates"]
        == payload["forward_proposal"]["eligible_rows"],
    )):
        raise ComparisonError("proposal/certificate query binding differs")
    omitted = {"block_rows", "block_order", "elapsed_seconds", "peak_rss_mb"}
    if _without(payload["forward_proposal"], omitted) != _without(
        payload["reverse_proposal"], omitted,
    ):
        raise ComparisonError("forward/reverse proposal semantics differ")
    certificate = payload["certificate"]
    if type(certificate) is not dict or payload["rounds"] != certificate.get("rounds"):
        raise ComparisonError("certificate fields differ")
    _validate_certificate(certificate, payload["matches"], query_id)
    candidates = {row.episode_id: row for row in forward.candidates}
    try:
        matches_cross_bound = len(candidates) == len(forward.candidates) and all(
            row["episode_id"] in candidates
            and row["symbol"] == candidates[row["episode_id"]].symbol
            and pd.Timestamp(row["cutoff"]).value
            == candidates[row["episode_id"]].cutoff_ns
            and row["quality_tier"] == candidates[row["episode_id"]].quality_tier
            for row in payload["matches"]
        )
    except (TypeError, ValueError, OverflowError):
        matches_cross_bound = False
    _validate_round_proposal_bindings(certificate, forward)
    _validate_closure_scan_evidence(certificate, forward, closure_reports)
    _validate_match_universe_cross_binding(
        payload["matches"], forward,
        certificate["threshold_closure_passes"], universe,
    )
    if certificate["threshold_closure_passes"]:
        matches_cross_bound = True
    if not matches_cross_bound:
        raise ComparisonError("certified proposal cross-binding differs")
    if payload["streaming_fallback_used"] is not (
        len(certificate["threshold_closure_passes"]) > 0
    ):
        raise ComparisonError("streaming fallback flag differs")
    metrics = payload["metrics"]
    try:
        observed_performance = producer.performance_gate(metrics)
    except producer.HarnessError as exc:
        raise ComparisonError("case performance reconstruction differs") from exc
    if not all((
        payload["performance_passed"] == observed_performance,
        metrics["forward_proposal_seconds"]
        == payload["forward_proposal"]["elapsed_seconds"],
        metrics["reverse_proposal_seconds"]
        == payload["reverse_proposal"]["elapsed_seconds"],
        metrics["process_rss_mb"] >= payload["forward_proposal"]["peak_rss_mb"],
        metrics["process_rss_mb"] >= payload["reverse_proposal"]["peak_rss_mb"],
        certificate["elapsed_seconds"] <= metrics["exact_task_wall_seconds"],
    )):
        raise ComparisonError("case performance reconstruction differs")


def reconstruct_query_bindings(prereg: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    repository = Path(__file__).resolve().parents[2]
    roots = prereg["roots"]
    registry_digest, cases = producer._registry_cases(
        repository, Path(roots["registry_root"]),
    )
    if registry_digest != prereg["registry_digest"]:
        raise ComparisonError("registry differs before truth open")
    inputs = producer.Inputs(
        repository, Path(prereg["config_path"]), Path(roots["registry_root"]),
        Path(roots["source_full_root"]) / "store", Path(roots["resident_root"]),
        Path(roots["output_root"]), producer.GENERATION_ID,
        str(prereg["provenance_digest"]), prereg["reserve_bytes"],
        registry_digest, cases, str(prereg["preregistration_digest"]),
    )
    result: dict[str, dict[str, Any]] = {}
    for case in cases:
        source, episode, request, packed = producer._case_context(inputs, case)
        binding = producer.query_binding(
            source, episode, request, packed, inputs.provenance_digest,
        )
        if not all((
            binding["query_stock_prefix"] == case.registry_case["stock_prefix"],
            binding["query_benchmark_prefix"] == case.registry_case["benchmark_prefix"],
        )):
            raise ComparisonError("stock/benchmark causal prefix differs")
        result[case.query_id] = binding
    return result


def validate_producer_before_truth(
    root: Path, *, repository_prereg: Mapping[str, Any] | None = None,
    git_validator: Callable[[Path, Mapping[str, Any]], Any] = producer._launch_git,
    binding_loader: Callable[[Mapping[str, Any]], dict[str, dict[str, Any]]]
    = reconstruct_query_bindings,
    allowed_suffix_files: set[str] | None = None,
    prereg_validator: Callable[[Mapping[str, Any]], None]
    = producer.validate_preregistration_shape,
    universe_loader: Callable[
        [Mapping[str, Any], Mapping[str, set[str]], list[Mapping[str, Any]]],
        tuple[dict[str, dict[str, dict[str, Any]]],
              dict[str, list[dict[str, Any]]]],
    ] = reconstruct_match_universe,
) -> dict[str, Any]:
    if root.is_symlink() or root.resolve() != Path(
        str((repository_prereg or producer._read_json(
            Path(__file__).resolve().parents[2] / producer.PREREG_RELATIVE
        ))["roots"]["output_root"])
    ).resolve():
        raise ComparisonError("producer root differs from preregistered output root")
    expected_cases = {
        f"cases/{index:02d}-{query_id}.json"
        for index, query_id in enumerate(producer.FROZEN_QUERY_IDS)
    }
    _exact_tree(
        root,
        {"RUN_STARTED.json", "CONTRACT.json", "RESIDENT.json",
         "PRODUCER_SEALED.json", *expected_cases, *(allowed_suffix_files or set())},
        {"cases"},
    )
    prereg = producer._read_json(root / "CONTRACT.json")
    frozen_prereg = (
        dict(repository_prereg) if repository_prereg is not None else
        producer._read_json(Path(__file__).resolve().parents[2] / producer.PREREG_RELATIVE)
    )
    if prereg != frozen_prereg:
        raise ComparisonError("producer preregistration snapshot differs from repository")
    try:
        prereg_validator(prereg)
    except producer.HarnessError as exc:
        raise ComparisonError("preregistration schema differs") from exc
    git_validator(Path(__file__).resolve().parents[2], prereg.get("git", {}))
    prereg_digest = prereg.get("preregistration_digest")
    if not isinstance(prereg_digest, str) or prereg_digest != stable_hash(
        _without(prereg, {"preregistration_digest"})
    ):
        raise ComparisonError("preregistration snapshot differs")
    started = producer._read_json(root / "RUN_STARTED.json")
    if not all((
        set(started) == {"schema_version", "preregistration_digest", "case_order",
                         "parent_max_workers", "created_at"},
        started.get("schema_version") == "m04r13-run-started-v1",
        started.get("preregistration_digest") == prereg_digest,
        started.get("case_order") == list(producer.FROZEN_QUERY_IDS),
        started.get("parent_max_workers") == 1,
        producer._is_utc_iso_timestamp(started.get("created_at")),
    )):
        raise ComparisonError("run marker differs")
    resident = producer._read_json(root / "RESIDENT.json")
    try:
        producer.validate_resident_snapshot(resident)
    except producer.HarnessError as exc:
        raise ComparisonError("resident snapshot differs") from exc
    if not all((
        resident["identity_digest"] == prereg["resident_identity_digest"],
        resident["content_digest"] == prereg["resident_content_digest"],
        resident["ready_digest"] == prereg["resident_ready_digest"],
        resident["store_root"]
        == str((Path(prereg["roots"]["resident_root"]) / "store").resolve()),
    )):
        raise ComparisonError("resident differs from preregistration")
    bindings = binding_loader(prereg)
    if set(bindings) != set(producer.FROZEN_QUERY_IDS):
        raise ComparisonError("query binding set differs")
    raw_cases = [producer._read_json(
        root / "cases" / f"{index:02d}-{query_id}.json"
    ) for index, query_id in enumerate(producer.FROZEN_QUERY_IDS)]
    requested = {
        query_id: {
            row["episode_id"] for row in raw_cases[index].get("matches", [])
            if type(row) is dict and type(row.get("episode_id")) is str
        }
        for index, query_id in enumerate(producer.FROZEN_QUERY_IDS)
    }
    universe, closure_evidence = universe_loader(prereg, requested, raw_cases)
    if (set(universe) != set(producer.FROZEN_QUERY_IDS)
            or set(closure_evidence) != set(producer.FROZEN_QUERY_IDS)):
        raise ComparisonError("packed match universe query set differs")
    cases = []
    for index, query_id in enumerate(producer.FROZEN_QUERY_IDS):
        case = raw_cases[index]
        _validate_case(
            case, query_id, prereg_digest, resident,
            bindings[query_id], universe[query_id], closure_evidence[query_id],
        )
        cases.append(case)
    seal = producer._read_json(root / "PRODUCER_SEALED.json")
    expected = {
        "schema_version": producer.SEAL_SCHEMA,
        "status": "truth_blind_producer_complete", "development_only": True,
        "truth_opened": False, "production_promotion_authorized": False,
        "real_forward_outcomes_accessed": False,
        "preregistration_digest": prereg_digest,
        "resident_identity_digest": resident["identity_digest"],
        "query_ids": list(producer.FROZEN_QUERY_IDS),
        "case_digests": [case["result_digest"] for case in cases],
        "semantic_passed": all(case["semantic_passed"] for case in cases),
        "performance_passed": all(case["performance_passed"] for case in cases),
    }
    if _without(seal, {"created_at", "seal_digest"}) != expected or seal.get(
        "seal_digest"
    ) != stable_hash(expected) or not producer._is_utc_iso_timestamp(
        seal.get("created_at")
    ):
        raise ComparisonError("producer seal differs")
    return {"prereg": prereg, "resident": resident, "cases": cases,
            "seal": seal, "bindings": bindings}


def load_authority_truth(
    authority_root: Path, verification_path: Path,
) -> dict[str, dict[str, Any]]:
    verification = _read_sha_json(
        verification_path, AUTHORITY_VERIFICATION_SHA256,
    )
    if not all((
        verification.get("schema_version")
        == "m04r11-certified-authority-verification-v4",
        verification.get("result_digest") == AUTHORITY_VERIFICATION_DIGEST,
        verification.get("passed") is True,
        verification.get("authority_correctness_passed") is True,
        verification.get("production_promotion_authorized") is False,
        verification.get("real_forward_outcomes_accessed") is False,
        verification.get("authority_matrix_digest") == AUTHORITY_MATRIX_DIGEST,
        verification.get("authority_seal_digest") == AUTHORITY_SEAL_DIGEST,
        verification.get("generation_id") == producer.GENERATION_ID,
    )):
        raise ComparisonError("authority verification differs")
    result: dict[str, dict[str, Any]] = {}
    for query_id, (expected_sha, expected_digest) in AUTHORITY_CASE_BINDINGS.items():
        path = authority_root / "cases" / f"{query_id}.json"
        payload = _read_sha_json(path, expected_sha)
        if not all((
            payload.get("schema_version") == "m04r11-certified-authority-case-v4",
            payload.get("status") == "completed",
            payload.get("query_episode_id") == query_id,
            payload.get("result_digest") == expected_digest,
            payload.get("real_forward_outcomes_accessed") is False,
            len(payload.get("matches", [])) == 20,
        )):
            raise ComparisonError("authority case differs")
        result[query_id] = payload
    return result


def _comparison_row(
    candidate: Mapping[str, Any], authority: Mapping[str, Any],
) -> dict[str, Any]:
    row = {
        "query_episode_id": candidate["query_episode_id"],
        "matches_equal": candidate["matches"] == authority["matches"],
        "certificate_result_digest_equal_diagnostic":
            candidate["certificate"]["result_digest"]
            == authority["certificate"]["result_digest"],
        "accounting_equal": all(
            candidate["certificate"][key] == authority["certificate"][key]
            for key in ("eligible_candidates", "exact_evaluated", "safely_pruned")
        ),
        "candidate_case_digest": candidate["result_digest"],
        "authority_case_digest": authority["result_digest"],
        "stock_prefix_equal": candidate["query_binding"]["query_stock_prefix"]
            == authority["query_stock_prefix"],
        "benchmark_prefix_equal":
            candidate["query_binding"]["query_benchmark_prefix"]
            == authority["query_benchmark_prefix"],
    }
    row["semantic_passed"] = all((
        row["matches_equal"], row["accounting_equal"],
        row["stock_prefix_equal"], row["benchmark_prefix_equal"],
    ))
    return row


def validate_terminal_comparison(
    root: Path, authority_root: Path, verification_path: Path, *,
    truth_loader: Callable[[Path, Path], dict[str, dict[str, Any]]]
    = load_authority_truth,
    repository_prereg: Mapping[str, Any] | None = None,
    git_validator: Callable[[Path, Mapping[str, Any]], Any] = producer._launch_git,
    binding_loader: Callable[[Mapping[str, Any]], dict[str, dict[str, Any]]]
    = reconstruct_query_bindings,
    prereg_validator: Callable[[Mapping[str, Any]], None]
    = producer.validate_preregistration_shape,
    universe_loader: Callable[
        [Mapping[str, Any], Mapping[str, set[str]], list[Mapping[str, Any]]],
        tuple[dict[str, dict[str, dict[str, Any]]],
              dict[str, list[dict[str, Any]]]],
    ] = reconstruct_match_universe,
) -> dict[str, Any]:
    suffix = {"RESULTS_OPENED.json", "COMPARISON.json", "COMPARISON_SEALED.json"}
    validated = validate_producer_before_truth(
        root, repository_prereg=repository_prereg, git_validator=git_validator,
        binding_loader=binding_loader, allowed_suffix_files=suffix,
        prereg_validator=prereg_validator, universe_loader=universe_loader,
    )
    _exact_tree(
        root,
        {
            "RUN_STARTED.json", "CONTRACT.json", "RESIDENT.json",
            "PRODUCER_SEALED.json", "RESULTS_OPENED.json", "COMPARISON.json",
            "COMPARISON_SEALED.json",
            *{f"cases/{index:02d}-{query_id}.json"
              for index, query_id in enumerate(producer.FROZEN_QUERY_IDS)},
        },
        {"cases"},
    )
    marker = producer._read_json(root / "RESULTS_OPENED.json")
    marker_keys = {
        "schema_version", "status", "development_only",
        "production_promotion_authorized", "real_forward_outcomes_accessed",
        "preregistration_digest", "producer_seal_digest", "created_at",
        "result_digest",
    }
    if not all((
        set(marker) == marker_keys,
        marker.get("schema_version") == RESULTS_OPENED_SCHEMA,
        marker.get("status") == "authority_results_opened_after_producer_seal",
        marker.get("development_only") is True,
        marker.get("production_promotion_authorized") is False,
        marker.get("real_forward_outcomes_accessed") is False,
        marker.get("preregistration_digest")
        == validated["prereg"]["preregistration_digest"],
        marker.get("producer_seal_digest") == validated["seal"]["seal_digest"],
        producer._is_utc_iso_timestamp(marker.get("created_at")),
        marker.get("result_digest") == stable_hash(
            _without(marker, {"created_at", "result_digest"})
        ),
    )):
        raise ComparisonError("RESULTS_OPENED reconstruction differs")
    truth = truth_loader(authority_root, verification_path)
    if set(truth) != set(producer.FROZEN_QUERY_IDS):
        raise ComparisonError("authority case set differs")
    comparison = producer._read_json(root / "COMPARISON.json")
    comparison_keys = {
        "schema_version", "status", "development_only", "post_open",
        "production_promotion_authorized", "real_forward_outcomes_accessed",
        "results_opened_digest", "producer_seal_digest",
        "authority_verification_digest", "rows", "semantic_passed",
        "performance_passed", "passed", "created_at", "result_digest",
    }
    row_keys = {
        "query_episode_id", "matches_equal",
        "certificate_result_digest_equal_diagnostic", "accounting_equal",
        "candidate_case_digest", "authority_case_digest", "stock_prefix_equal",
        "benchmark_prefix_equal", "semantic_passed",
    }
    rows = comparison.get("rows")
    expected_rows = [
        _comparison_row(candidate, truth[candidate["query_episode_id"]])
        for candidate in validated["cases"]
    ]
    expected_semantic = all(row["semantic_passed"] for row in expected_rows)
    expected_performance = validated["seal"]["performance_passed"]
    if not all((
        set(comparison) == comparison_keys, type(rows) is list, len(rows) == 4,
        [row.get("query_episode_id") for row in rows]
        == list(producer.FROZEN_QUERY_IDS),
        all(set(row) == row_keys for row in rows),
        comparison.get("schema_version") == COMPARISON_SCHEMA,
        comparison.get("status") == "comparison_complete",
        comparison.get("development_only") is True,
        comparison.get("post_open") is True,
        comparison.get("production_promotion_authorized") is False,
        comparison.get("real_forward_outcomes_accessed") is False,
        producer._is_utc_iso_timestamp(comparison.get("created_at")),
        comparison.get("producer_seal_digest") == validated["seal"]["seal_digest"],
        comparison.get("authority_verification_digest")
        == AUTHORITY_VERIFICATION_DIGEST,
        rows == expected_rows,
        all(all(type(row[key]) is bool for key in (
            "matches_equal", "certificate_result_digest_equal_diagnostic",
            "accounting_equal", "stock_prefix_equal", "benchmark_prefix_equal",
            "semantic_passed",
        )) for row in rows),
        all(row["semantic_passed"] == all((
            row["matches_equal"], row["accounting_equal"],
            row["stock_prefix_equal"], row["benchmark_prefix_equal"],
        )) for row in rows),
        all(row["candidate_case_digest"] == validated["cases"][index]["result_digest"]
            and row["authority_case_digest"] == AUTHORITY_CASE_BINDINGS[
                row["query_episode_id"]
            ][1] for index, row in enumerate(rows)),
        comparison["results_opened_digest"] == marker["result_digest"],
        comparison["semantic_passed"] == expected_semantic,
        type(comparison["performance_passed"]) is bool,
        comparison["performance_passed"] == expected_performance,
        type(comparison["semantic_passed"]) is bool,
        type(comparison["passed"]) is bool,
        comparison["passed"] == (expected_semantic and expected_performance),
        comparison["result_digest"] == stable_hash(_without(
            comparison, {"created_at", "result_digest"},
        )),
    )):
        raise ComparisonError("comparison reconstruction differs")
    seal = producer._read_json(root / "COMPARISON_SEALED.json")
    seal_keys = {
        "schema_version", "status", "development_only",
        "production_promotion_authorized", "real_forward_outcomes_accessed",
        "comparison_result_digest", "semantic_passed", "performance_passed",
        "passed", "created_at", "seal_digest",
    }
    if not all((
        set(seal) == seal_keys,
        seal.get("schema_version") == COMPARISON_SEAL_SCHEMA,
        seal.get("status") == "terminal_comparison_sealed",
        seal.get("development_only") is True,
        seal.get("production_promotion_authorized") is False,
        seal.get("real_forward_outcomes_accessed") is False,
        producer._is_utc_iso_timestamp(seal.get("created_at")),
        all(type(seal.get(key)) is bool
            for key in ("semantic_passed", "performance_passed", "passed")),
        seal["comparison_result_digest"] == comparison["result_digest"],
        seal["semantic_passed"] == comparison["semantic_passed"],
        seal["performance_passed"] == comparison["performance_passed"],
        seal["passed"] == comparison["passed"],
        seal["seal_digest"] == stable_hash(_without(
            seal, {"created_at", "seal_digest"},
        )),
    )):
        raise ComparisonError("comparison seal reconstruction differs")
    return seal


def _write_comparison_incomplete(
    root: Path, producer_files: set[str], marker: Mapping[str, Any],
    validated: Mapping[str, Any], exc: BaseException,
) -> None:
    observed = set(producer_files)
    for relative in ("COMPARISON.json", "COMPARISON_SEALED.json"):
        if (root / relative).is_file():
            observed.add(relative)
    _exact_tree(root, observed, {"cases"})
    deterministic = {
        "schema_version": "m04r13-threaded-certified-comparison-incomplete-v1",
        "status": "terminal_post_open_comparison_incomplete",
        "development_only": True, "production_promotion_authorized": False,
        "real_forward_outcomes_accessed": False,
        "results_opened_digest": marker["result_digest"],
        "producer_seal_digest": validated["seal"]["seal_digest"],
        "preregistration_digest": validated["prereg"]["preregistration_digest"],
        "partial_tree_digest": producer._tree_snapshot_digest(
            root, omitted={"COMPARISON_INCOMPLETE.json"},
        ),
        "error_type": type(exc).__name__,
    }
    payload = {
        **deterministic, "created_at": datetime.now(timezone.utc).isoformat(),
        "result_digest": stable_hash(deterministic),
    }
    producer._atomic(root / "COMPARISON_INCOMPLETE.json", payload)
    if (producer._read_json(root / "COMPARISON_INCOMPLETE.json") != payload
            or not producer._is_utc_iso_timestamp(payload["created_at"])):
        raise ComparisonError("comparison incomplete publication differs")
    _exact_tree(root, observed | {"COMPARISON_INCOMPLETE.json"}, {"cases"})


def compare(
    producer_root: Path, authority_root: Path, verification_path: Path,
    *, truth_loader: Callable[[Path, Path], dict[str, dict[str, Any]]]
    = load_authority_truth,
    repository_prereg: Mapping[str, Any] | None = None,
    git_validator: Callable[[Path, Mapping[str, Any]], Any] = producer._launch_git,
    binding_loader: Callable[[Mapping[str, Any]], dict[str, dict[str, Any]]]
    = reconstruct_query_bindings,
    prereg_validator: Callable[[Mapping[str, Any]], None]
    = producer.validate_preregistration_shape,
    universe_loader: Callable[
        [Mapping[str, Any], Mapping[str, set[str]], list[Mapping[str, Any]]],
        tuple[dict[str, dict[str, dict[str, Any]]],
              dict[str, list[dict[str, Any]]]],
    ] = reconstruct_match_universe,
) -> dict[str, Any]:
    validated = validate_producer_before_truth(
        producer_root, repository_prereg=repository_prereg,
        git_validator=git_validator, binding_loader=binding_loader,
        prereg_validator=prereg_validator, universe_loader=universe_loader,
    )
    marker_deterministic = {
        "schema_version": RESULTS_OPENED_SCHEMA,
        "status": "authority_results_opened_after_producer_seal",
        "development_only": True, "production_promotion_authorized": False,
        "real_forward_outcomes_accessed": False,
        "preregistration_digest": validated["prereg"]["preregistration_digest"],
        "producer_seal_digest": validated["seal"]["seal_digest"],
    }
    marker = {
        **marker_deterministic,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "result_digest": stable_hash(marker_deterministic),
    }
    producer_files = {
        "RUN_STARTED.json", "CONTRACT.json", "RESIDENT.json",
        "PRODUCER_SEALED.json", "RESULTS_OPENED.json",
        *{f"cases/{index:02d}-{query_id}.json"
          for index, query_id in enumerate(producer.FROZEN_QUERY_IDS)},
    }
    try:
        producer._atomic(producer_root / "RESULTS_OPENED.json", marker)
        _exact_tree(producer_root, producer_files, {"cases"})
        truth = truth_loader(authority_root, verification_path)
        rows = []
        for candidate in validated["cases"]:
            query_id = candidate["query_episode_id"]
            authority = truth[query_id]
            rows.append(_comparison_row(candidate, authority))
        semantic_passed = all(row["semantic_passed"] for row in rows)
        performance_passed = validated["seal"]["performance_passed"]
        deterministic = {
            "schema_version": COMPARISON_SCHEMA, "status": "comparison_complete",
            "development_only": True, "post_open": True,
            "production_promotion_authorized": False,
            "real_forward_outcomes_accessed": False,
            "results_opened_digest": marker["result_digest"],
            "producer_seal_digest": validated["seal"]["seal_digest"],
            "authority_verification_digest": AUTHORITY_VERIFICATION_DIGEST,
            "rows": rows, "semantic_passed": semantic_passed,
            "performance_passed": performance_passed,
            "passed": semantic_passed and performance_passed,
        }
        comparison = {
            **deterministic, "created_at": datetime.now(timezone.utc).isoformat(),
            "result_digest": stable_hash(deterministic),
        }
        producer._atomic(producer_root / "COMPARISON.json", comparison)
        _exact_tree(producer_root, producer_files | {"COMPARISON.json"}, {"cases"})
        if producer._read_json(producer_root / "COMPARISON.json") != comparison:
            raise ComparisonError("comparison publication differs")
        seal_deterministic = {
            "schema_version": COMPARISON_SEAL_SCHEMA,
            "status": "terminal_comparison_sealed", "development_only": True,
            "production_promotion_authorized": False,
            "real_forward_outcomes_accessed": False,
            "comparison_result_digest": comparison["result_digest"],
            "semantic_passed": comparison["semantic_passed"],
            "performance_passed": comparison["performance_passed"],
            "passed": comparison["passed"],
        }
        seal = {
            **seal_deterministic, "created_at": datetime.now(timezone.utc).isoformat(),
            "seal_digest": stable_hash(seal_deterministic),
        }
        # Terminal seal publication is deliberately the final fallible operation.
        producer._atomic(producer_root / "COMPARISON_SEALED.json", seal)
        return seal
    except BaseException as exc:
        if (producer_root / "RESULTS_OPENED.json").exists() and not (
            producer_root / "COMPARISON_INCOMPLETE.json"
        ).exists():
            _write_comparison_incomplete(
                producer_root, producer_files, marker, validated, exc,
            )
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--producer-root", type=Path, required=True)
    parser.add_argument("--authority-root", type=Path, required=True)
    parser.add_argument("--authority-verification", type=Path, required=True)
    arguments = parser.parse_args()
    if (arguments.producer_root / "COMPARISON_SEALED.json").is_file():
        result = validate_terminal_comparison(
            arguments.producer_root, arguments.authority_root,
            arguments.authority_verification,
        )
    else:
        result = compare(
            arguments.producer_root, arguments.authority_root,
            arguments.authority_verification,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
