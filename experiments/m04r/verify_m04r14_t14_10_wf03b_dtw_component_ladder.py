"""Independent verifier for the outcome-blind combined-bound closure ladder."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np

from market_analogues.causal_prefix import causal_prefix_digest
from market_analogues.dtw_sample_store import (
    dtw_sample_lower_bounds,
    load_dtw_sample_generation,
)
from market_analogues.episodes import build_episode
from market_analogues.packed_bound_store import (
    load_packed_generation,
    packed_branch_aware_lower_bounds,
)
from market_analogues.representation import represent, representation_input_digest
from market_analogues.distance import representation_distance
from market_analogues.types import InstrumentKey, stable_hash

from experiments.m04r import m04r14_t14_10_wf03_feasibility as base
from experiments.m04r import m04r14_t14_10_wf03b_dtw_component_ladder as producer


SCHEMA = "m04r14-dtw-component-ladder-verification-preregistration-v2"
PREREGISTRATION_RELATIVE = Path(
    "experiments/m04r/verify_m04r14_t14_10_wf03b_dtw_component_ladder_v2_preregistered.json"
)
OUTPUT_RELATIVE = Path(
    "config/data/analogues/m04r14/"
    "t14-10-wf03b-dtw-component-ladder-v1-verification-v2"
)
RUNTIME_FILES = (
    "experiments/m04r/verify_m04r14_t14_10_wf03b_dtw_component_ladder.py",
    "src/market_analogues/dtw_sample_store.py",
    "src/market_analogues/dtw_interval_bound.py",
    "src/market_analogues/packed_bound_store.py",
    "src/market_analogues/quantized_bound.py",
    "src/market_analogues/distance.py",
    "src/market_analogues/representation.py",
)
BLOCK_ROWS = 4_093
EXACT_TOLERANCE = 1e-12


class LadderVerificationError(RuntimeError):
    pass


def _git(repository: Path, *args: str) -> str:
    return base._git(repository, *args)


def _producer_result(repository: Path) -> dict[str, Any]:
    value = base._read(repository / producer.OUTPUT_RELATIVE / "RESULT.json")
    base._validate_seal(value)
    if value.get("passed") is not True or value.get("case_statuses") != ["complete"] * 3 \
            or value.get("historical_walk_forward_query_outcomes_opened") is not False \
            or value.get("final_period_result_opened") is not False:
        raise LadderVerificationError("producer result differs")
    return value


def build_preregistration(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    if _git(repository, "status", "--porcelain"):
        raise LadderVerificationError("verification preregistration requires clean commit")
    result = _producer_result(repository)
    target = repository / OUTPUT_RELATIVE
    if target.exists() or target.is_symlink():
        raise LadderVerificationError("verification target must be absent")
    state = {
        "schema_version": SCHEMA,
        "status": "frozen_before_independent_verification",
        "implementation_commit": _git(repository, "rev-parse", "HEAD"),
        "runtime_files": {path: base._sha(repository / path) for path in RUNTIME_FILES},
        "inputs": {
            "producer_preregistration_digest": base._read(
                repository / producer.PREREGISTRATION_RELATIVE
            )["preregistration_digest"],
            "producer_result_digest": result["result_digest"],
            "producer_result_sha256": base._sha(
                repository / producer.OUTPUT_RELATIVE / "RESULT.json"
            ),
            "packed_generation_id": base.GENERATION_ID,
            "dtw_generation_id": producer.DTW_GENERATION_ID,
        },
        "execution": {
            "reverse_block_rows": BLOCK_ROWS,
            "independent_complete_threshold_scan": True,
            "raw_exact_matches_recomputed": 60,
            "raw_exact_absolute_tolerance_hex": EXACT_TOLERANCE.hex(),
            "output_root": str(target.resolve()),
        },
        "claims": {
            "historical_walk_forward_query_outcomes_opened": False,
            "final_period_result_opened": False,
            "production_promotion_authorized": False,
        },
    }
    return base._sealed(state, "preregistration_digest")


def validate_preregistration(repository: Path, value: Mapping[str, Any]) -> None:
    base._validate_seal(value, "preregistration_digest")
    result = _producer_result(repository)
    if value.get("schema_version") != SCHEMA \
            or value.get("inputs", {}).get("producer_result_digest") != result["result_digest"]:
        raise LadderVerificationError("verification preregistration differs")
    head = value.get("implementation_commit")
    if type(head) is not str:
        raise LadderVerificationError("verification implementation differs")
    _git(repository, "merge-base", "--is-ancestor", head, "HEAD")
    producer_preregistration = base._read(repository / producer.PREREGISTRATION_RELATIVE)
    producer_h0 = producer_preregistration.get("implementation_commit")
    producer_h1 = _git(
        repository, "log", "-1", "--format=%H", "--",
        str(producer.PREREGISTRATION_RELATIVE),
    )
    if type(producer_h0) is not str \
            or _git(repository, "rev-parse", f"{producer_h1}^") != producer_h0 \
            or _git(repository, "diff", "--name-only", producer_h0, producer_h1) \
            != str(producer.PREREGISTRATION_RELATIVE):
        raise LadderVerificationError("producer H0/H1 lineage differs")
    for path, digest in value["runtime_files"].items():
        if base._sha(repository / path) != digest:
            raise LadderVerificationError(f"verification runtime differs: {path}")
        blob = subprocess.run(["git", "show", f"{head}:{path}"], cwd=repository,
                              capture_output=True, check=False)
        if blob.returncode or sha256(blob.stdout).hexdigest() != digest:
            raise LadderVerificationError(f"verification Git binding differs: {path}")


def _eligibility(records: np.ndarray, packed_query: Any, symbol_id: int | None) -> np.ndarray:
    mask = (
        (records["cutoff_ns"] <= packed_query.latest_eligible_ns)
        & (records["episode_id"] != np.void(bytes.fromhex(packed_query.episode_id)))
        & np.isin(records["quality_tier"], np.asarray([1, 2], dtype=np.uint8))
    )
    if symbol_id is not None:
        mask &= ~(
            (records["symbol_id"] == symbol_id)
            & (records["cutoff_ns"] >= packed_query.query_start_ns)
        )
    return mask


def _threshold_audit(
    packed: Any, dtw: Any, packed_query: Any, threshold: float,
    match_ids: set[str],
) -> dict[str, Any]:
    symbol_id = (
        packed.symbols.index(packed_query.symbol)
        if packed_query.symbol in packed.symbols else None
    )
    targets = {bytes.fromhex(value): value for value in match_ids}
    found: dict[str, float] = {}
    admitted = 0
    minimum_above = float("inf")
    eligible_main = 0
    starts = list(range(0, len(packed.rows), BLOCK_ROWS))
    starts.reverse()
    for first in starts:
        last = min(first + BLOCK_ROWS, len(packed.rows))
        rows = np.asarray(packed.rows[first:last])
        mask = _eligibility(rows, packed_query, symbol_id)
        eligible_main += int(np.count_nonzero(mask))
        if not np.any(mask):
            continue
        rigid = np.asarray(
            packed_branch_aware_lower_bounds(
                packed_query.representation, rows,
            ).components["price"], dtype=np.float64,
        )
        bound = rigid + 0.45 * dtw_sample_lower_bounds(
            packed_query.representation, np.asarray(dtw.rows[first:last]),
        )
        values = bound[mask]
        admitted += int(np.count_nonzero(values <= threshold))
        above = values[values > threshold]
        if len(above):
            minimum_above = min(minimum_above, float(np.min(above)))
        eligible_positions = np.flatnonzero(mask)
        for local in eligible_positions:
            raw = bytes(rows["episode_id"][local])
            episode_id = targets.get(raw)
            if episode_id is not None:
                found[episode_id] = float(bound[local])
    overflow = np.asarray(packed.overflow)
    overflow_mask = _eligibility(overflow, packed_query, symbol_id)
    eligible_overflow = int(np.count_nonzero(overflow_mask))
    if threshold >= 0:
        admitted += eligible_overflow
    for row in overflow[overflow_mask]:
        episode_id = targets.get(bytes(row["episode_id"]))
        if episode_id is not None:
            found[episode_id] = 0.0
    if set(found) != match_ids:
        raise LadderVerificationError("published matches missing from bound store")
    return {
        "eligible_main_rows": eligible_main,
        "eligible_overflow_rows": eligible_overflow,
        "eligible_rows": eligible_main + eligible_overflow,
        "admitted_at_or_below_threshold": admitted,
        "minimum_above_threshold": minimum_above,
        "match_bound_digest": stable_hash({
            key: value.hex() for key, value in sorted(found.items())
        }),
        "maximum_match_bound_excess": max(
            found[row] - threshold for row in found
        ),
    }


def _case_receipts(root: Path, ordinal: int, label: str, query_id: str) -> tuple[dict, dict, dict, dict]:
    case = root / "cases" / f"{ordinal:03d}-{label}-{query_id}"
    terminal = base._read(case / "COMPLETE.json")
    base._validate_seal(terminal, "terminal_digest")
    attempt = case / terminal["attempt_relative"]
    complete = base._read(attempt / "COMPLETE.json")
    base._validate_seal(complete, "complete_digest")
    exact = base._read(attempt / "EXACT.json")
    base._validate_seal(exact)
    forward = base._read(attempt / "PROPOSAL_FORWARD.json")
    reverse = base._read(attempt / "PROPOSAL_REVERSE.json")
    base._validate_seal(forward); base._validate_seal(reverse)
    if terminal.get("attempt_complete_sha256") != base._sha(attempt / "COMPLETE.json") \
            or terminal.get("attempt_complete_digest") != complete["complete_digest"]:
        raise LadderVerificationError("producer terminal binding differs")
    for item in complete["leaf_manifest"]:
        if base._sha(attempt / item["path"]) != item["sha256"]:
            raise LadderVerificationError("producer leaf hash differs")
    if complete["leaf_manifest_digest"] != stable_hash(complete["leaf_manifest"]):
        raise LadderVerificationError("producer leaf manifest differs")
    attempts = sorted(path for path in (case / "attempts").iterdir() if path.is_dir())
    expected_attempts = 2 if ordinal == 1 else 1
    if len(attempts) != expected_attempts:
        raise LadderVerificationError("producer retained attempt inventory differs")
    if ordinal == 1:
        failure = base._read(attempts[0] / "FAILED.json")
        base._validate_seal(failure, "failure_digest")
        if failure.get("status") != "operational_failure" \
                or failure.get("exception_message") != "combined closure gate differs":
            raise LadderVerificationError("retained middle refusal differs")
    return complete, exact, forward["report"], reverse["report"]


def _verify_case(repository: Path, packed: Any, dtw: Any, ordinal: int,
                 label: str, query_id: str, registry_row: Mapping[str, Any]) -> dict[str, Any]:
    complete, exact, forward, reverse = _case_receipts(
        repository / producer.OUTPUT_RELATIVE, ordinal, label, query_id,
    )
    ignored = {"block_rows", "block_order", "elapsed_seconds", "peak_rss_mb"}
    semantic = lambda row: {key: value for key, value in row.items() if key not in ignored}
    if semantic(forward) != semantic(reverse):
        raise LadderVerificationError("producer proposal summaries differ")
    source, query, request, packed_query = base._context(repository, registry_row)
    certificate = exact["certificate"]
    matches = exact["matches"]
    threshold = float(certificate["stop_threshold"])
    audit = _threshold_audit(
        packed, dtw, packed_query, threshold,
        {row["episode_id"] for row in matches},
    )
    query_representation = represent(query)
    exact_rows = []
    exact_differences = []
    for row in matches:
        candidate = build_episode(
            source, InstrumentKey("nasdaq", row["symbol"]), row["cutoff"],
            query.key.lookback, query.key.representation_version,
        )
        value = float(representation_distance(
            query_representation, represent(candidate),
        )[1]["price"])
        expected = float.fromhex(row["distance_hex"])
        difference = abs(value - expected)
        if candidate.key.id != row["episode_id"] or difference > EXACT_TOLERANCE:
            raise LadderVerificationError("raw exact match differs")
        exact_rows.append((row["episode_id"], row["distance_hex"], value.hex()))
        exact_differences.append(difference)
    input_digest = stable_hash({
        "query_stock_prefix": asdict(causal_prefix_digest(
            source.load(query.key.instrument), query.key.cutoff,
        )),
        "query_representation_digest": representation_input_digest(query_representation),
        "request": asdict(request),
        "packed_generation_id": packed.generation_id,
        "packed_provenance_digest": packed.manifest["provenance_digest"],
        "dtw_generation_id": dtw.generation_id,
        "dtw_provenance_digest": dtw.manifest["provenance_digest"],
    })
    deterministic = {
        "schema_version": certificate["schema_version"],
        "contract_digest": certificate["contract_digest"],
        "packed_generation_id": packed.generation_id,
        "dtw_generation_id": dtw.generation_id,
        "query_episode_id": query_id,
        "input_digest": input_digest,
        "proposal_digest": forward["result_digest"],
        "eligible_candidates": certificate["eligible_candidates"],
        "bound_evaluated": certificate["bound_evaluated"],
        "exact_evaluated": certificate["exact_evaluated"],
        "bound_pruned": certificate["bound_pruned"],
        "stop_threshold": certificate["stop_threshold"],
        "next_lower_bound": certificate["next_lower_bound"],
        "maximum_bound_excess": certificate["maximum_bound_excess"],
        "rounds": certificate["rounds"],
        "matches": [{"episode_id": row["episode_id"],
                     "distance_hex": row["distance_hex"]} for row in matches],
        "outcomes_or_labels_used": False,
    }
    if not all((
        stable_hash(deterministic) == certificate["result_digest"],
        certificate["input_digest"] == input_digest,
        certificate["proposal_digest"] == forward["result_digest"],
        audit["eligible_rows"] == certificate["eligible_candidates"],
        audit["admitted_at_or_below_threshold"] == certificate["bound_evaluated"],
        audit["minimum_above_threshold"] > threshold,
        certificate["next_lower_bound"] >= audit["minimum_above_threshold"],
        certificate["bound_evaluated"] + certificate["bound_pruned"]
            == certificate["eligible_candidates"],
        len(matches) == 20 and len({row["symbol"] for row in matches}) == 20,
        complete["resources_after"]["swap_kib"] == 0,
    )):
        raise LadderVerificationError("independent certificate gate differs")
    return {
        "ordinal": ordinal, "label": label, "query_id": query_id,
        "certificate_digest": certificate["result_digest"],
        "candidate_digest": forward["candidate_digest"],
        "threshold_hex": threshold.hex(),
        "threshold_audit": audit,
        "exact_match_digest": stable_hash(exact_rows),
        "raw_exact_non_bitwise_count": sum(value > 0 for value in exact_differences),
        "raw_exact_maximum_absolute_difference": max(exact_differences, default=0.0),
    }


def execute(repository: Path, preregistration: Mapping[str, Any]) -> dict[str, Any]:
    validate_preregistration(repository, preregistration)
    started = perf_counter()
    resident = base._resident()
    packed = load_packed_generation(
        Path(resident["store_root"]), base.GENERATION_ID,
        expected_provenance_digest=base.PROVENANCE_DIGEST,
        verify_content=False, validate_records=False,
    )
    dtw = load_dtw_sample_generation(
        repository / producer.DTW_ROOT_RELATIVE, producer.DTW_GENERATION_ID,
        packed_manifest=packed.manifest, verify_content=False, validate_records=False,
    )
    _registry, by_id = base._registry(repository)
    cases = [
        _verify_case(repository, packed, dtw, ordinal, label, query_id, by_id[query_id])
        for ordinal, (label, query_id, _symbol, _cutoff) in enumerate(base.PROBES)
    ]
    state = {
        "schema_version": "m04r14-dtw-component-ladder-verification-v2",
        "status": "verified", "passed": True,
        "producer_result_digest": preregistration["inputs"]["producer_result_digest"],
        "cases": cases, "case_digest": stable_hash(cases),
        "elapsed_seconds": perf_counter() - started,
        "historical_walk_forward_query_outcomes_opened": False,
        "final_period_result_opened": False,
        "production_promotion_authorized": False,
    }
    result = base._sealed(state, "verification_digest")
    target = repository / OUTPUT_RELATIVE
    if target.exists() or target.is_symlink():
        raise LadderVerificationError("verification target exists")
    target.mkdir(parents=True)
    base._atomic(target / "VERIFIED.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preregister", "run"))
    parser.add_argument("--repository", type=Path, required=True)
    args = parser.parse_args(argv)
    repository = args.repository.resolve(strict=True)
    path = repository / PREREGISTRATION_RELATIVE
    if args.mode == "preregister":
        base._atomic(path, build_preregistration(repository)); return 0
    result = execute(repository, base._read(path))
    print(json.dumps(result, indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
