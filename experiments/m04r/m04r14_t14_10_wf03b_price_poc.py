"""Durable three-case feasibility POC for exact price-component retrieval."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
import math
from pathlib import Path
import subprocess
from time import perf_counter
from typing import Any, Mapping, Sequence

from market_analogues.component_search import (
    ComponentFrontierOverflow,
    certified_component_search,
    certified_component_search_contract,
)
from market_analogues.packed_bound_search import (
    COMPONENT_SEARCH_SCHEMA_VERSION,
    BoundProposal,
    _packed_query_input_digest,
    bound_proposal_candidate_digest,
    packed_component_search_contract,
    scan_packed_component_bound_proposals_threaded,
)
from market_analogues.types import stable_hash

from experiments.m04r import m04r14_t14_10_wf03_feasibility as base


SCHEMA = "m04r14-t14-10-wf03b-price-poc-preregistration-v1"
OUTPUT_RELATIVE = Path("config/data/analogues/m04r14/t14-10-wf03b-price-poc-v1")
PREREGISTRATION_RELATIVE = Path(
    "experiments/m04r/m04r14_t14_10_wf03b_price_poc_preregistered.json"
)
RUNTIME_FILES = (
    "experiments/m04r/m04r14_t14_10_wf03b_price_poc.py",
    "experiments/m04r/m04r14_t14_10_wf03_feasibility.py",
    "src/market_analogues/component_search.py",
    "src/market_analogues/packed_bound_search.py",
    "src/market_analogues/packed_bound_store.py",
    "src/market_analogues/distance.py",
    "src/market_analogues/exact_batch.py",
    "src/market_analogues/representation.py",
)


class PricePocError(RuntimeError):
    pass


def _git(repository: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=repository, text=True,
                            capture_output=True, check=False)
    if result.returncode:
        raise PricePocError(f"git {' '.join(args)} failed")
    return result.stdout.strip()


def _validate_proposal(value: Mapping[str, Any], packed: Any) -> None:
    contract = packed_component_search_contract("price")
    deterministic = {
        "schema_version": COMPONENT_SEARCH_SCHEMA_VERSION,
        "contract_digest": contract["digest"], "component": "price",
        "generation_id": base.GENERATION_ID,
        "query_episode_id": packed.episode_id,
        "rows_scanned": value["rows_scanned"], "eligible_rows": value["eligible_rows"],
        "eligible_main_rows": value["eligible_main_rows"],
        "eligible_overflow_rows": value["eligible_overflow_rows"],
        "route_counts": value["route_counts"], "route_quotas": value["route_quotas"],
        "candidate_digest": value["candidate_digest"],
        "real_forward_outcomes_accessed": False,
        "input_digest": _packed_query_input_digest(packed),
    }
    try:
        candidates = tuple(BoundProposal(
            item["episode_id"], item["symbol"], item["cutoff_ns"],
            item["quality_tier"], float.fromhex(item["lower_bound_hex"]),
            tuple(item["routes"]), item["overflow_fallback"],
        ) for item in value["candidates"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PricePocError("price proposal encoding differs") from exc
    if not all((
        value["schema_version"] == COMPONENT_SEARCH_SCHEMA_VERSION,
        value["contract_digest"] == contract["digest"],
        value["input_digest"] == deterministic["input_digest"],
        value["generation_id"] == base.GENERATION_ID,
        value["query_episode_id"] == packed.episode_id,
        value["route_quotas"] == {"price": base.PROPOSAL_QUOTA},
        value["route_counts"] == {"price": len(candidates)},
        value["eligible_rows"] == value["eligible_main_rows"] + value["eligible_overflow_rows"],
        len(candidates) == min(base.PROPOSAL_QUOTA, value["eligible_rows"]),
        list(candidates) == sorted(
            candidates, key=lambda row: (row.lower_bound, row.episode_id)),
        all(row.routes == ("price",) and math.isfinite(row.lower_bound)
            and row.lower_bound >= 0 for row in candidates),
        value["candidate_digest"] == bound_proposal_candidate_digest(candidates),
        value["result_digest"] == stable_hash(deterministic),
    )):
        raise PricePocError("price proposal reconstruction differs")


def _match(value: Any) -> dict[str, Any]:
    return {
        "episode_id": value.episode_key.id,
        "symbol": value.episode_key.instrument.source_symbol,
        "cutoff": value.episode_key.cutoff.isoformat(),
        "distance": value.total_distance,
        "quality_tier": value.quality_tier,
    }


def _case(repository: Path, root: Path, ordinal: int, label: str,
          row: Mapping[str, Any], resident: Mapping[str, Any]) -> dict[str, Any]:
    query_id = row["episode_id"]
    case_root = root / "cases" / f"{ordinal:03d}-{label}-{query_id}"
    case_root.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    base._atomic(case_root / "RUN_STARTED.json", base._sealed({
        "schema_version": "m04r14-t14-10-wf03b-price-case-v1",
        "status": "running", "label": label, "query_id": query_id,
        "resident_identity_digest": resident["identity_digest"],
        "created_at": base._now(),
    }))
    source, episode, request, packed = base._context(repository, row)
    lease_before = base.resident_file_identity_lease(base.RESIDENT_ROOT / "READY.json")
    forward = scan_packed_component_bound_proposals_threaded(
        Path(resident["store_root"]), base.GENERATION_ID, packed,
        component="price", quota=base.PROPOSAL_QUOTA,
        block_rows=base.BLOCK_ROWS, block_order="forward",
        threads=base.PROPOSAL_THREADS, verify_content=False,
        expected_provenance_digest=base.PROVENANCE_DIGEST,
    )
    forward_payload = base._proposal_payload(forward)
    _validate_proposal(forward_payload, packed)
    base._atomic(case_root / "PROPOSAL_FORWARD.json", base._sealed({
        "proposal": forward_payload, "created_at": base._now(),
    }))
    reverse = scan_packed_component_bound_proposals_threaded(
        Path(resident["store_root"]), base.GENERATION_ID, packed,
        component="price", quota=base.PROPOSAL_QUOTA,
        block_rows=base.REVERSE_BLOCK_ROWS, block_order="reverse",
        threads=base.PROPOSAL_THREADS, verify_content=False,
        expected_provenance_digest=base.PROVENANCE_DIGEST,
    )
    reverse_payload = base._proposal_payload(reverse)
    _validate_proposal(reverse_payload, packed)
    if base._proposal_semantics(forward_payload) != base._proposal_semantics(reverse_payload):
        raise PricePocError("price proposal traversal parity differs")
    base._atomic(case_root / "PROPOSAL_REVERSE.json", base._sealed({
        "proposal": reverse_payload, "created_at": base._now(),
    }))
    exact_started = perf_counter()
    try:
        result = certified_component_search(
            episode, source, request, Path(resident["store_root"]),
            base.GENERATION_ID, store_dataset_id="nasdaq", component="price",
            initial_frontier_rows=base.INITIAL_FRONTIER,
            maximum_frontier_rows=base.MAXIMUM_FRONTIER,
            seed_rows=base.SEED_ROWS, block_rows=base.BLOCK_ROWS,
            proposal_threads=base.PROPOSAL_THREADS, workers=base.EXACT_WORKERS,
            tolerance=base.TOLERANCE, verify_content=False,
            precomputed_proposal=forward,
        )
    except ComponentFrontierOverflow as exc:
        failed = base._sealed({
            "schema_version": "m04r14-t14-10-wf03b-price-case-v1",
            "status": "frontier_overflow", "label": label, "query_id": query_id,
            "frontier_rows": exc.frontier_rows,
            "eligible_candidates": exc.eligible_candidates,
            "exact_evaluated": exc.exact_evaluated, "threshold": exc.threshold,
            "next_lower_bound": exc.next_lower_bound,
            "elapsed_seconds": perf_counter() - started, "created_at": base._now(),
        }, "failure_digest")
        base._atomic(case_root / "FAILED.json", failed)
        return failed
    exact_seconds = perf_counter() - exact_started
    certificate = json.loads(json.dumps(asdict(result.certificate), allow_nan=False))
    matches = [_match(value) for value in result.matches]
    if not all((
        len(matches) == request.top_k,
        len({row["symbol"] for row in matches}) == request.top_k,
        [row["distance"] for row in matches] == sorted(row["distance"] for row in matches),
        certificate["component"] == "price",
        certificate["exact_evaluated"] + certificate["native_bound_pruned"]
            + certificate["packed_bound_pruned"] == certificate["eligible_candidates"],
        certificate["next_lower_bound"] is None
            or certificate["next_lower_bound"] > certificate["stop_threshold"],
    )):
        raise PricePocError("price certificate invariant differs")
    lease_after = base.resident_file_identity_lease(base.RESIDENT_ROOT / "READY.json")
    if lease_after["lease_digest"] != lease_before["lease_digest"]:
        raise PricePocError("resident lease changed during price case")
    base._atomic(case_root / "EXACT.json", base._sealed({
        "certificate": certificate, "matches": matches,
        "exact_wall_seconds": exact_seconds,
        "resident_lease_digest": lease_after["lease_digest"],
        "created_at": base._now(),
    }))
    manifest = [{"path": name, "sha256": base._sha(case_root / name)} for name in (
        "RUN_STARTED.json", "PROPOSAL_FORWARD.json", "PROPOSAL_REVERSE.json", "EXACT.json"
    )]
    complete = base._sealed({
        "schema_version": "m04r14-t14-10-wf03b-price-case-v1",
        "status": "complete", "label": label, "query_id": query_id,
        "proposal_parity": True, "certified": True,
        "eligible_candidates": certificate["eligible_candidates"],
        "exact_evaluated": certificate["exact_evaluated"],
        "native_bound_pruned": certificate["native_bound_pruned"],
        "packed_bound_pruned": certificate["packed_bound_pruned"],
        "elapsed_seconds": perf_counter() - started,
        "stage_seconds": {"forward": forward.elapsed_seconds,
                          "reverse": reverse.elapsed_seconds, "exact": exact_seconds},
        "resource": base._resource(), "leaf_manifest": manifest,
        "leaf_manifest_digest": stable_hash(manifest), "created_at": base._now(),
    }, "complete_digest")
    base._atomic(case_root / "COMPLETE.json", complete)
    return complete


def build_preregistration(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    if _git(repository, "status", "--porcelain"):
        raise PricePocError("preregistration requires a clean implementation commit")
    registry, by_id = base._registry(repository)
    composite = base._read(
        repository / "config/data/analogues/m04r14/t14-10-wf03-feasibility-v1-verification/VERIFIED.json"
    )
    if composite.get("passed") is not True:
        raise PricePocError("composite feasibility prerequisite differs")
    state = {
        "schema_version": SCHEMA, "status": "frozen_before_price_probe_results",
        "implementation_commit": _git(repository, "rev-parse", "HEAD"),
        "runtime_files": {path: base._sha(repository / path) for path in RUNTIME_FILES},
        "inputs": {
            "registry_digest": registry["registry_digest"],
            "registry_sha256": base._sha(repository / base.REGISTRY_FILE),
            "composite_verification_sha256": base._sha(
                repository / "config/data/analogues/m04r14/t14-10-wf03-feasibility-v1-verification/VERIFIED.json"
            ),
            "generation_id": base.GENERATION_ID,
            "provenance_digest": base.PROVENANCE_DIGEST,
        },
        "probes": [{"label": label, "query_id": query_id,
                    "registry_case": by_id[query_id]}
                   for label, query_id, _symbol, _cutoff in base.PROBES],
        "contracts": {
            "proposal": packed_component_search_contract("price"),
            "certificate": certified_component_search_contract("price"),
        },
        "execution": {
            "proposal_quota": base.PROPOSAL_QUOTA,
            "proposal_threads": base.PROPOSAL_THREADS,
            "frontiers": [1_000, 2_000, 4_000, 8_000, 16_000, 16_384],
            "seed_rows": base.SEED_ROWS, "exact_workers": base.EXACT_WORKERS,
            "top_k": 20, "max_per_instrument": 1,
            "minimum_history_gap_bars": 60,
            "output_root": str((repository / OUTPUT_RELATIVE).resolve()),
        },
        "claims": {
            "historical_walk_forward_query_outcomes_opened": False,
            "final_period_result_opened": False,
            "authority_or_outcome_paths_accepted": False,
            "development_only": True, "production_promotion_authorized": False,
        },
    }
    return base._sealed(state, "preregistration_digest")


def validate_preregistration(repository: Path, value: Mapping[str, Any]) -> None:
    base._validate_seal(value, "preregistration_digest")
    if value.get("schema_version") != SCHEMA \
            or value.get("inputs", {}).get("registry_digest") != base.REGISTRY_DIGEST:
        raise PricePocError("price preregistration differs")
    head = value.get("implementation_commit")
    if type(head) is not str:
        raise PricePocError("price implementation commit differs")
    _git(repository, "merge-base", "--is-ancestor", head, "HEAD")
    for path, digest in value["runtime_files"].items():
        if base._sha(repository / path) != digest:
            raise PricePocError(f"price runtime differs: {path}")
        blob = subprocess.run(["git", "show", f"{head}:{path}"], cwd=repository,
                              capture_output=True, check=False)
        if blob.returncode or sha256(blob.stdout).hexdigest() != digest:
            raise PricePocError(f"price implementation binding differs: {path}")


def execute(repository: Path, preregistration: Mapping[str, Any]) -> dict[str, Any]:
    validate_preregistration(repository, preregistration)
    _registry, by_id = base._registry(repository)
    resident = base._resident()
    root = repository / OUTPUT_RELATIVE
    if root.exists() or root.is_symlink():
        raise PricePocError("price POC output already exists")
    root.mkdir(parents=True); (root / "cases").mkdir()
    base._atomic(root / "CONTRACT.json", preregistration)
    base._atomic(root / "RUN_STARTED.json", base._sealed({
        "schema_version": "m04r14-t14-10-wf03b-price-run-v1", "status": "running",
        "resident_identity_digest": resident["identity_digest"],
        "created_at": base._now(),
    }))
    results = []
    for ordinal, (label, query_id, _symbol, _cutoff) in enumerate(base.PROBES):
        results.append(_case(repository, root, ordinal, label, by_id[query_id], resident))
    passed = all(row["status"] == "complete" for row in results)
    manifest = [{
        "path": f"cases/{ordinal:03d}-{label}-{query_id}/{name}",
        "sha256": base._sha(root / f"cases/{ordinal:03d}-{label}-{query_id}/{name}"),
        "status": results[ordinal]["status"],
    } for ordinal, (label, query_id, _symbol, _cutoff) in enumerate(base.PROBES)
      for name in (("COMPLETE.json",) if results[ordinal]["status"] == "complete"
                   else ("FAILED.json",))]
    state = {
        "schema_version": "m04r14-t14-10-wf03b-price-result-v1",
        "status": "complete", "passed": passed, "cases": 3,
        "case_statuses": [row["status"] for row in results],
        "case_manifest": manifest, "case_manifest_digest": stable_hash(manifest),
        "historical_walk_forward_query_outcomes_opened": False,
        "final_period_result_opened": False,
        "production_promotion_authorized": False,
        "created_at": base._now(),
    }
    result = base._sealed(state, "result_digest")
    base._atomic(root / "RESULT.json", result)
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
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
