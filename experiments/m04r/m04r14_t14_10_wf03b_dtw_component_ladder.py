"""Durable outcome-blind closure ladder for the combined price/DTW bound."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from time import perf_counter
from typing import Any, Mapping, Sequence

from market_analogues.component_search import ComponentFrontierOverflow
from market_analogues.dtw_component_search import (
    DtwComponentProposalReport,
    certified_dtw_component_search,
    certified_dtw_component_search_contract,
    dtw_component_search_contract,
    scan_dtw_component_bound_proposals,
)
from market_analogues.types import stable_hash

from experiments.m04r import m04r14_t14_10_wf03_feasibility as base


SCHEMA = "m04r14-t14-10-wf03b-dtw-component-ladder-preregistration-v1"
OUTPUT_RELATIVE = Path(
    "config/data/analogues/m04r14/t14-10-wf03b-dtw-component-ladder-v1"
)
PREREGISTRATION_RELATIVE = Path(
    "experiments/m04r/m04r14_t14_10_wf03b_dtw_component_ladder_preregistered.json"
)
DTW_ROOT_RELATIVE = Path(
    "config/data/analogues/m04r14/t14-10-wf03b-dtw-store-full-v2/store"
)
DTW_RESULT_RELATIVE = Path(
    "config/data/analogues/m04r14/t14-10-wf03b-dtw-store-full-v2/RESULT.json"
)
DTW_VERIFICATION_RELATIVE = Path(
    "config/data/analogues/m04r14/"
    "t14-10-wf03b-dtw-store-full-v2-verification-v2/VERIFIED.json"
)
DTW_GENERATION_ID = (
    "c4b068a6bc8d88b697bd238eb9e6d7934f12f71993fa15d925d9b4c9f0596cb9"
)
FRONTIER_LEVELS = (16_384, 32_768, 65_536, 131_072, 262_144)
PROPOSAL_QUOTA = FRONTIER_LEVELS[-1] + 1
FORWARD_BLOCK_ROWS = 4_096
REVERSE_BLOCK_ROWS = 4_097
KERNEL_THREADS = 8
EXACT_WORKERS = 8
SEED_ROWS = 512
TOLERANCE = 1e-12
RUNTIME_FILES = (
    "experiments/m04r/m04r14_t14_10_wf03b_dtw_component_ladder.py",
    "experiments/m04r/m04r14_t14_10_wf03_feasibility.py",
    "src/market_analogues/dtw_component_search.py",
    "src/market_analogues/component_search.py",
    "src/market_analogues/dtw_sample_store.py",
    "src/market_analogues/dtw_interval_bound.py",
    "src/market_analogues/packed_bound_search.py",
    "src/market_analogues/packed_bound_store.py",
    "src/market_analogues/exact_batch.py",
    "src/market_analogues/distance.py",
    "src/market_analogues/representation.py",
)


class DtwComponentLadderError(RuntimeError):
    pass


def _git(repository: Path, *args: str) -> str:
    return base._git(repository, *args)


def _validate_upstream(repository: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    result = base._read(repository / DTW_RESULT_RELATIVE)
    base._validate_seal(result)
    verification = base._read(repository / DTW_VERIFICATION_RELATIVE)
    base._validate_seal(verification, "verification_digest")
    if not all((
        result.get("passed") is True,
        result.get("generation_id") == DTW_GENERATION_ID,
        result.get("historical_walk_forward_query_outcomes_opened") is False,
        result.get("final_period_result_opened") is False,
        verification.get("passed") is True,
        verification.get("generation_id") == DTW_GENERATION_ID,
        verification.get("producer_result_digest") == result.get("result_digest"),
        verification.get("historical_walk_forward_query_outcomes_opened") is False,
        verification.get("final_period_result_opened") is False,
    )):
        raise DtwComponentLadderError("verified DTW prerequisite differs")
    return result, verification


def build_preregistration(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    if _git(repository, "status", "--porcelain"):
        raise DtwComponentLadderError("ladder preregistration requires a clean commit")
    result, verification = _validate_upstream(repository)
    registry, by_id = base._registry(repository)
    resident = base._resident()
    output = repository / OUTPUT_RELATIVE
    if output.exists() or output.is_symlink():
        raise DtwComponentLadderError("ladder output must be absent before freezing")
    head = _git(repository, "rev-parse", "HEAD")
    state = {
        "schema_version": SCHEMA,
        "status": "frozen_before_combined_component_closure",
        "implementation_commit": head,
        "runtime_files": {
            path: base._sha(repository / path) for path in RUNTIME_FILES
        },
        "inputs": {
            "registry_digest": registry["registry_digest"],
            "registry_sha256": base._sha(repository / base.REGISTRY_FILE),
            "packed_generation_id": base.GENERATION_ID,
            "packed_provenance_digest": base.PROVENANCE_DIGEST,
            "resident_identity_digest": resident["identity_digest"],
            "dtw_generation_id": DTW_GENERATION_ID,
            "dtw_result_digest": result["result_digest"],
            "dtw_result_sha256": base._sha(repository / DTW_RESULT_RELATIVE),
            "dtw_verification_digest": verification["verification_digest"],
            "dtw_verification_sha256": base._sha(
                repository / DTW_VERIFICATION_RELATIVE
            ),
        },
        "probes": [
            {"ordinal": ordinal, "label": label, "query_id": query_id,
             "registry_case": by_id[query_id]}
            for ordinal, (label, query_id, _symbol, _cutoff)
            in enumerate(base.PROBES)
        ],
        "contracts": {
            "proposal": dtw_component_search_contract(),
            "completion": certified_dtw_component_search_contract(),
        },
        "execution": {
            "frontier_levels": list(FRONTIER_LEVELS),
            "proposal_quota": PROPOSAL_QUOTA,
            "forward_block_rows": FORWARD_BLOCK_ROWS,
            "reverse_block_rows": REVERSE_BLOCK_ROWS,
            "kernel_threads": KERNEL_THREADS,
            "exact_workers": EXACT_WORKERS,
            "seed_rows": SEED_ROWS,
            "tolerance_hex": TOLERANCE.hex(),
            "top_k": base.TOP_K,
            "max_per_instrument": base.MAX_PER_INSTRUMENT,
            "minimum_history_gap_bars": base.MINIMUM_HISTORY_GAP,
            "output_root": str(output.resolve()),
            "resume": "retain attempts; skip only sealed case terminals",
        },
        "gates": {
            "forward_reverse_candidate_identity": True,
            "strict_next_bound_closure": True,
            "twenty_distinct_symbols": True,
            "maximum_bound_excess_at_most_tolerance": True,
            "resident_lease_unchanged": True,
            "zero_process_swap": True,
        },
        "claims": {
            "historical_query_retrieval_opened": True,
            "historical_walk_forward_query_outcomes_opened": False,
            "final_period_result_opened": False,
            "authority_or_outcome_paths_accepted": False,
            "development_only": True,
            "production_promotion_authorized": False,
        },
    }
    return base._sealed(state, "preregistration_digest")


def validate_preregistration(repository: Path, value: Mapping[str, Any]) -> None:
    base._validate_seal(value, "preregistration_digest")
    result, verification = _validate_upstream(repository)
    if not all((
        value.get("schema_version") == SCHEMA,
        value.get("status") == "frozen_before_combined_component_closure",
        value.get("inputs", {}).get("dtw_result_digest")
            == result["result_digest"],
        value.get("inputs", {}).get("dtw_verification_digest")
            == verification["verification_digest"],
        value.get("execution", {}).get("frontier_levels")
            == list(FRONTIER_LEVELS),
        value.get("execution", {}).get("proposal_quota") == PROPOSAL_QUOTA,
        value.get("claims", {}).get(
            "historical_walk_forward_query_outcomes_opened"
        ) is False,
        value.get("claims", {}).get("final_period_result_opened") is False,
    )):
        raise DtwComponentLadderError("ladder preregistration differs")
    head = value.get("implementation_commit")
    if type(head) is not str:
        raise DtwComponentLadderError("ladder implementation commit differs")
    _git(repository, "merge-base", "--is-ancestor", head, "HEAD")
    for path, digest in value.get("runtime_files", {}).items():
        if type(path) is not str or type(digest) is not str \
                or base._sha(repository / path) != digest:
            raise DtwComponentLadderError(f"ladder runtime differs: {path}")
        blob = subprocess.run(
            ["git", "show", f"{head}:{path}"], cwd=repository,
            capture_output=True, check=False,
        )
        if blob.returncode or sha256(blob.stdout).hexdigest() != digest:
            raise DtwComponentLadderError(f"ladder Git binding differs: {path}")


def _report_summary(report: DtwComponentProposalReport) -> dict[str, Any]:
    return {
        "schema_version": report.schema_version,
        "contract_digest": report.contract_digest,
        "packed_generation_id": report.packed_generation_id,
        "dtw_generation_id": report.dtw_generation_id,
        "query_episode_id": report.query_episode_id,
        "input_digest": report.input_digest,
        "rows_scanned": report.rows_scanned,
        "eligible_rows": report.eligible_rows,
        "eligible_main_rows": report.eligible_main_rows,
        "eligible_overflow_rows": report.eligible_overflow_rows,
        "quota": report.quota,
        "block_rows": report.block_rows,
        "block_order": report.block_order,
        "kernel_threads": report.kernel_threads,
        "elapsed_seconds": report.elapsed_seconds,
        "peak_rss_mb": report.peak_rss_mb,
        "candidate_digest": report.candidate_digest,
        "result_digest": report.result_digest,
        "first_lower_bound_hex": report.candidates[0].lower_bound.hex(),
        "last_lower_bound_hex": report.candidates[-1].lower_bound.hex(),
        "candidate_count": len(report.candidates),
    }


def _proposal_equal(
    forward: DtwComponentProposalReport, reverse: DtwComponentProposalReport,
) -> bool:
    return all((
        forward.contract_digest == reverse.contract_digest,
        forward.packed_generation_id == reverse.packed_generation_id,
        forward.dtw_generation_id == reverse.dtw_generation_id,
        forward.query_episode_id == reverse.query_episode_id,
        forward.input_digest == reverse.input_digest,
        forward.candidates == reverse.candidates,
        forward.rows_scanned == reverse.rows_scanned,
        forward.eligible_rows == reverse.eligible_rows,
        forward.eligible_main_rows == reverse.eligible_main_rows,
        forward.eligible_overflow_rows == reverse.eligible_overflow_rows,
        forward.quota == reverse.quota,
        forward.candidate_digest == reverse.candidate_digest,
        forward.result_digest == reverse.result_digest,
    ))


def _match(row: Any) -> dict[str, Any]:
    return {
        "episode_id": row.episode_key.id,
        "symbol": row.episode_key.instrument.source_symbol,
        "cutoff": row.episode_key.cutoff.isoformat(),
        "distance_hex": row.total_distance.hex(),
        "quality_tier": row.quality_tier,
    }


def _next_attempt(case_root: Path) -> Path:
    attempts = case_root / "attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    existing = []
    for path in attempts.iterdir():
        if path.is_dir() and path.name.startswith("attempt-"):
            try:
                existing.append(int(path.name.removeprefix("attempt-")))
            except ValueError as exc:
                raise DtwComponentLadderError("malformed ladder attempt") from exc
    target = attempts / f"attempt-{max(existing, default=0) + 1:04d}"
    target.mkdir()
    return target


def _terminal(case_root: Path) -> dict[str, Any] | None:
    found = [path for path in (case_root / "COMPLETE.json", case_root / "FAILED.json")
             if path.exists()]
    if not found:
        return None
    if len(found) != 1:
        raise DtwComponentLadderError("multiple ladder case terminals")
    value = base._read(found[0])
    base._validate_seal(value, "terminal_digest")
    relative = value.get("attempt_relative")
    if type(relative) is not str or not relative.startswith("attempts/attempt-") \
            or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise DtwComponentLadderError("ladder terminal attempt differs")
    attempt = case_root / relative
    if value.get("status") == "complete":
        leaf = attempt / "COMPLETE.json"
        field = "complete_digest"
        expected_sha = value.get("attempt_complete_sha256")
        expected_digest = value.get("attempt_complete_digest")
    elif value.get("status") == "frontier_overflow":
        leaf = attempt / "FAILED.json"
        field = "failure_digest"
        expected_sha = value.get("attempt_failure_sha256")
        expected_digest = value.get("attempt_failure_digest")
    else:
        raise DtwComponentLadderError("ladder terminal status differs")
    leaf_value = base._read(leaf)
    base._validate_seal(leaf_value, field)
    if base._sha(leaf) != expected_sha or leaf_value[field] != expected_digest:
        raise DtwComponentLadderError("ladder terminal leaf binding differs")
    return value


def _case(
    repository: Path, root: Path, specification: Mapping[str, Any],
    registry_row: Mapping[str, Any], resident: Mapping[str, Any],
) -> dict[str, Any]:
    ordinal = int(specification["ordinal"])
    label = str(specification["label"])
    query_id = str(specification["query_id"])
    case_root = root / "cases" / f"{ordinal:03d}-{label}-{query_id}"
    case_root.mkdir(parents=True, exist_ok=True)
    existing = _terminal(case_root)
    if existing is not None:
        return existing
    attempt = _next_attempt(case_root)
    started = perf_counter()
    before = base._resource()
    base._atomic(attempt / "RUN_STARTED.json", base._sealed({
        "schema_version": "m04r14-dtw-component-ladder-attempt-v1",
        "status": "running", "ordinal": ordinal, "label": label,
        "query_id": query_id, "resident_identity_digest": resident["identity_digest"],
        "created_at": base._now(),
    }))
    source, episode, request, packed_query = base._context(repository, registry_row)
    lease_before = base.resident_file_identity_lease(base.RESIDENT_ROOT / "READY.json")
    packed_root = Path(str(resident["store_root"]))
    dtw_root = repository / DTW_ROOT_RELATIVE
    try:
        forward = scan_dtw_component_bound_proposals(
            packed_root, base.GENERATION_ID, dtw_root, DTW_GENERATION_ID,
            packed_query, quota=PROPOSAL_QUOTA,
            block_rows=FORWARD_BLOCK_ROWS, block_order="forward",
            kernel_threads=KERNEL_THREADS, verify_content=False,
            expected_packed_provenance_digest=base.PROVENANCE_DIGEST,
        )
        base._atomic(attempt / "PROPOSAL_FORWARD.json", base._sealed({
            "schema_version": "m04r14-dtw-component-proposal-summary-v1",
            "report": _report_summary(forward), "created_at": base._now(),
        }))
        reverse = scan_dtw_component_bound_proposals(
            packed_root, base.GENERATION_ID, dtw_root, DTW_GENERATION_ID,
            packed_query, quota=PROPOSAL_QUOTA,
            block_rows=REVERSE_BLOCK_ROWS, block_order="reverse",
            kernel_threads=KERNEL_THREADS, verify_content=False,
            expected_packed_provenance_digest=base.PROVENANCE_DIGEST,
        )
        if not _proposal_equal(forward, reverse):
            raise DtwComponentLadderError("combined proposal traversal differs")
        base._atomic(attempt / "PROPOSAL_REVERSE.json", base._sealed({
            "schema_version": "m04r14-dtw-component-proposal-summary-v1",
            "report": _report_summary(reverse), "created_at": base._now(),
        }))
        try:
            result = certified_dtw_component_search(
                episode, source, request, packed_root, base.GENERATION_ID,
                dtw_root, DTW_GENERATION_ID, store_dataset_id="nasdaq",
                initial_frontier_rows=FRONTIER_LEVELS[0],
                maximum_frontier_rows=FRONTIER_LEVELS[-1],
                seed_rows=SEED_ROWS, block_rows=FORWARD_BLOCK_ROWS,
                workers=EXACT_WORKERS, tolerance=TOLERANCE,
                verify_content=False, precomputed_proposal=forward,
            )
        except ComponentFrontierOverflow as exc:
            lease_after = base.resident_file_identity_lease(
                base.RESIDENT_ROOT / "READY.json"
            )
            after = base._resource()
            if lease_after["lease_digest"] != lease_before["lease_digest"] \
                    or after["swap_kib"] != 0:
                raise DtwComponentLadderError(
                    "combined overflow resource/lease gate differs"
                )
            payload = {
                "schema_version": "m04r14-dtw-component-ladder-attempt-v1",
                "status": "frontier_overflow", "ordinal": ordinal,
                "label": label, "query_id": query_id,
                "frontier_rows": exc.frontier_rows,
                "eligible_candidates": exc.eligible_candidates,
                "exact_evaluated": exc.exact_evaluated,
                "threshold_hex": exc.threshold.hex(),
                "next_lower_bound_hex": (
                    None if exc.next_lower_bound is None
                    else exc.next_lower_bound.hex()
                ),
                "candidate_digest": forward.candidate_digest,
                "resident_lease_digest": lease_after["lease_digest"],
                "resources_after": after,
                "elapsed_seconds": perf_counter() - started,
                "created_at": base._now(),
            }
            base._atomic(attempt / "FAILED.json", base._sealed(payload, "failure_digest"))
            terminal = base._sealed({
                **payload, "attempt_relative": attempt.relative_to(case_root).as_posix(),
                "attempt_failure_sha256": base._sha(attempt / "FAILED.json"),
                "attempt_failure_digest": base._read(
                    attempt / "FAILED.json"
                )["failure_digest"],
            }, "terminal_digest")
            base._atomic(case_root / "FAILED.json", terminal)
            return terminal
        certificate = asdict(result.certificate)
        matches = [_match(row) for row in result.matches]
        lease_after = base.resident_file_identity_lease(base.RESIDENT_ROOT / "READY.json")
        after = base._resource()
        rounds = certificate["rounds"]
        if not all((
            len(matches) == base.TOP_K,
            len({row["symbol"] for row in matches}) == base.TOP_K,
            [row["distance_hex"] for row in matches]
                == [value.hex() for value in sorted(
                    float.fromhex(row["distance_hex"]) for row in matches
                )],
            rounds and [row["frontier_rows"] for row in rounds]
                == list(FRONTIER_LEVELS[:len(rounds)]),
            rounds[-1]["certified"] is True,
            certificate["next_lower_bound"] is None
                or certificate["next_lower_bound"] > certificate["stop_threshold"],
            certificate["maximum_bound_excess"] <= TOLERANCE,
            lease_after["lease_digest"] == lease_before["lease_digest"],
            after["swap_kib"] == 0,
        )):
            raise DtwComponentLadderError("combined closure gate differs")
        exact_payload = base._sealed({
            "schema_version": "m04r14-dtw-component-ladder-exact-v1",
            "certificate": certificate, "matches": matches,
            "created_at": base._now(),
        })
        base._atomic(attempt / "EXACT.json", exact_payload)
        manifest = [
            {"path": name, "sha256": base._sha(attempt / name)}
            for name in ("RUN_STARTED.json", "PROPOSAL_FORWARD.json",
                         "PROPOSAL_REVERSE.json", "EXACT.json")
        ]
        complete = base._sealed({
            "schema_version": "m04r14-dtw-component-ladder-attempt-v1",
            "status": "complete", "ordinal": ordinal, "label": label,
            "query_id": query_id, "candidate_digest": forward.candidate_digest,
            "certificate_digest": result.certificate.result_digest,
            "frontier_rows": rounds[-1]["frontier_rows"],
            "bound_evaluated": certificate["bound_evaluated"],
            "exact_evaluated": certificate["exact_evaluated"],
            "elapsed_seconds": perf_counter() - started,
            "stage_seconds": {
                "forward": forward.elapsed_seconds,
                "reverse": reverse.elapsed_seconds,
                "exact": result.certificate.elapsed_seconds,
            },
            "resources_before": before, "resources_after": after,
            "leaf_manifest": manifest,
            "leaf_manifest_digest": stable_hash(manifest),
            "created_at": base._now(),
        }, "complete_digest")
        base._atomic(attempt / "COMPLETE.json", complete)
        terminal = base._sealed({
            "schema_version": "m04r14-dtw-component-ladder-case-v1",
            "status": "complete", "ordinal": ordinal, "label": label,
            "query_id": query_id,
            "attempt_relative": attempt.relative_to(case_root).as_posix(),
            "attempt_complete_sha256": base._sha(attempt / "COMPLETE.json"),
            "attempt_complete_digest": complete["complete_digest"],
            "created_at": base._now(),
        }, "terminal_digest")
        base._atomic(case_root / "COMPLETE.json", terminal)
        return terminal
    except BaseException as exc:
        failed = attempt / "FAILED.json"
        if not failed.exists():
            base._atomic(failed, base._sealed({
                "schema_version": "m04r14-dtw-component-ladder-attempt-v1",
                "status": "operational_failure", "ordinal": ordinal,
                "label": label, "query_id": query_id,
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "elapsed_seconds": perf_counter() - started,
                "created_at": base._now(),
            }, "failure_digest"))
        raise


def execute(repository: Path, preregistration: Mapping[str, Any]) -> dict[str, Any]:
    validate_preregistration(repository, preregistration)
    _registry_state, by_id = base._registry(repository)
    resident = base._resident()
    if resident["identity_digest"] != preregistration["inputs"]["resident_identity_digest"]:
        raise DtwComponentLadderError("resident identity changed after freezing")
    root = repository / OUTPUT_RELATIVE
    if not root.exists():
        root.mkdir(parents=True)
        base._atomic(root / "CONTRACT.json", dict(preregistration))
    elif base._read(root / "CONTRACT.json") != preregistration:
        raise DtwComponentLadderError("ladder output contract differs")
    result_path = root / "RESULT.json"
    if result_path.exists():
        result = base._read(result_path)
        base._validate_seal(result)
        return result
    outputs = []
    for specification in preregistration["probes"]:
        query_id = specification["query_id"]
        outputs.append(_case(
            repository, root, specification, by_id[query_id], resident,
        ))
    statuses = [row["status"] for row in outputs]
    state = {
        "schema_version": "m04r14-t14-10-wf03b-dtw-component-ladder-result-v1",
        "status": "complete", "passed": statuses == ["complete"] * len(outputs),
        "cases": len(outputs), "case_statuses": statuses,
        "case_terminal_digest": stable_hash(outputs),
        "historical_walk_forward_query_outcomes_opened": False,
        "final_period_result_opened": False,
        "production_promotion_authorized": False,
        "created_at": base._now(),
    }
    result = base._sealed(state)
    base._atomic(result_path, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preregister", "run"))
    parser.add_argument("--repository", type=Path, required=True)
    args = parser.parse_args(argv)
    repository = args.repository.resolve(strict=True)
    path = repository / PREREGISTRATION_RELATIVE
    if args.mode == "preregister":
        base._atomic(path, build_preregistration(repository))
        return 0
    result = execute(repository, base._read(path))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
