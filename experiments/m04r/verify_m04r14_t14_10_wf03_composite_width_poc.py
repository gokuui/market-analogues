"""Independently verify the twelve-query true-composite width POC."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from market_analogues.adapters import source_from_spec
from market_analogues.config import load_config
from market_analogues.distance import DistanceConfig
from market_analogues.episodes import build_episode
from market_analogues.packed_bound_store import load_packed_generation
from market_analogues.search import latest_eligible_cutoff
from market_analogues.types import InstrumentKey, stable_hash

from experiments.m04r import m04r14_t14_10_wf03_composite_width_poc as producer
from experiments.m04r import m04r14_t14_10_wf03_feasibility as base
from experiments.m04r import verify_m04r14_t14_10_wf03_composite_topology_poc as topology_verifier


OUTPUT_RELATIVE = Path(
    "config/data/analogues/m04r14/"
    "t14-10-wf03-composite-width-poc-v1-verification"
)
EXPECTED_QUERY_COUNT = 12
EXPECTED_LINK_COUNT = EXPECTED_QUERY_COUNT * base.TOP_K


class CompositeWidthVerificationError(RuntimeError):
    pass


def _git(repository: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repository, text=True,
        capture_output=True, check=False,
    )
    if result.returncode:
        raise CompositeWidthVerificationError(
            result.stderr.strip() or "git command failed"
        )
    return result.stdout.strip()


def _sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _load_outputs(
    repository: Path, preregistration: Mapping[str, Any], rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    root = repository / producer.OUTPUT_RELATIVE
    if root.is_symlink() or base._read(root / "CONTRACT.json") != preregistration:
        raise CompositeWidthVerificationError("width contract differs")
    p4 = base._read(root / "P4T3.json")
    p12 = base._read(root / "P12T1.json")
    result = base._read(root / "RESULT.json")
    for value in (p4, p12, result):
        base._validate_seal(value)
    query_ids = [row["episode_id"] for row in rows]
    for value, groups, threads in ((p4, 4, 3), (p12, 12, 1)):
        if not all((
            value.get("schema_version") == "m04r14-wf03-composite-width-run-v1",
            value.get("status") == "complete",
            value.get("groups") == groups,
            value.get("threads_per_group") == threads,
            value.get("total_threads") == 12,
            value.get("all_zero_swap") is True,
            value.get("preregistration_digest")
                == preregistration["preregistration_digest"],
            [case.get("query_id") for case in value.get("cases", [])] == query_ids,
            value.get("outcomes_or_labels_used") is False,
            value.get("historical_walk_forward_query_outcomes_opened") is False,
            value.get("final_period_result_opened") is False,
        )):
            raise CompositeWidthVerificationError("width topology receipt differs")
    return p4, p12, result


def _validate_cases(
    p4: Mapping[str, Any], p12: Mapping[str, Any], rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    for topology in (p4, p12):
        for case in topology["cases"]:
            topology_verifier.validate_certificate(case)
    p4_semantics = producer._semantic_map(p4)
    p12_semantics = producer._semantic_map(p12)
    if p4_semantics != p12_semantics:
        raise CompositeWidthVerificationError("width semantic outputs differ")
    expected_ids = [row["episode_id"] for row in rows]
    if list(p4_semantics) != expected_ids or list(p12_semantics) != expected_ids:
        raise CompositeWidthVerificationError("width semantic order differs")
    return [
        {"query_id": query_id, "semantic_digest": stable_hash(p4_semantics[query_id])}
        for query_id in expected_ids
    ]


def _validate_links(
    repository: Path, resident: Mapping[str, Any], rows: Sequence[Mapping[str, Any]],
    cases: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    packed = load_packed_generation(
        Path(resident["store_root"]), base.GENERATION_ID,
        expected_provenance_digest=base.PROVENANCE_DIGEST,
        verify_content=True, validate_records=True,
    )
    records = np.concatenate((
        topology_verifier._metadata_records(packed.rows),
        topology_verifier._metadata_records(packed.overflow),
    ))
    order = np.argsort(records["episode_id"], kind="stable")
    sorted_ids = records["episode_id"][order]
    source = source_from_spec(
        load_config(repository / base.CONFIG_RELATIVE).datasets["nasdaq"]
    )
    rows_by_id = {row["episode_id"]: row for row in rows}
    observations = []
    for case in cases:
        row = rows_by_id[case["query_id"]]
        query = build_episode(
            source, InstrumentKey("nasdaq", row["symbol"]), row["cutoff"],
            int(row["lookback"]), row["representation_version"],
        )
        latest_ns = int(latest_eligible_cutoff(
            query, base.MINIMUM_HISTORY_GAP,
        ).value)
        query_start_ns = int(query.bars.timestamp.iloc[0].value)
        ids = np.asarray([
            np.void(bytes.fromhex(match["episode_id"])) for match in case["matches"]
        ], dtype="V12")
        positions = np.searchsorted(sorted_ids, ids)
        if np.any(positions >= len(sorted_ids)) \
                or not np.array_equal(sorted_ids[positions], ids):
            raise CompositeWidthVerificationError("matched episode is absent")
        selected = records[order[positions]]
        for match, record in zip(case["matches"], selected, strict=True):
            symbol = packed.symbols[int(record["symbol_id"])]
            quality = {1: "A", 2: "B"}.get(int(record["quality_tier"]))
            cutoff = pd.Timestamp(int(record["cutoff_ns"]), unit="ns").isoformat()
            if symbol != match["symbol"] or quality != match["quality_tier"] \
                    or cutoff != match["cutoff"] \
                    or int(record["cutoff_ns"]) > latest_ns \
                    or (symbol == row["symbol"]
                        and int(record["cutoff_ns"]) >= query_start_ns):
                raise CompositeWidthVerificationError("matched episode binding differs")
        observations.append({
            "query_id": case["query_id"],
            "match_ids": [match["episode_id"] for match in case["matches"]],
            "maximum_cutoff_ns": int(selected["cutoff_ns"].max()),
        })
    return observations


def _validate_result(
    result: Mapping[str, Any], preregistration: Mapping[str, Any],
    p4: Mapping[str, Any], p12: Mapping[str, Any],
) -> None:
    expected_gates = {
        "all_twelve_queries_equal": True,
        "all_twenty_four_certificates_close": True,
        "zero_process_swap": True,
        "outcomes_or_labels_excluded": True,
    }
    if not all((
        result.get("schema_version")
            == "m04r14-t14-10-wf03-composite-width-result-v1",
        result.get("status") == "complete",
        result.get("passed") is True,
        result.get("gates") == expected_gates,
        result.get("preregistration_digest")
            == preregistration["preregistration_digest"],
        result.get("p4t3_digest") == p4["result_digest"],
        result.get("p12t1_digest") == p12["result_digest"],
        result.get("p4t3_wall_seconds") == p4["wall_seconds"],
        result.get("p12t1_wall_seconds") == p12["wall_seconds"],
        result.get("p12_over_p4_speedup")
            == p4["wall_seconds"] / p12["wall_seconds"],
        p12["wall_seconds"] < p4["wall_seconds"],
        result.get("selected_topology") == "p12t1",
        result.get("outcomes_or_labels_used") is False,
        result.get("historical_walk_forward_query_outcomes_opened") is False,
        result.get("final_period_result_opened") is False,
        result.get("production_promotion_authorized") is False,
    )):
        raise CompositeWidthVerificationError("width aggregate differs")


def verify(repository: Path) -> dict[str, Any]:
    started = perf_counter()
    repository = repository.resolve(strict=True)
    if _git(repository, "status", "--porcelain"):
        raise CompositeWidthVerificationError("width verifier requires clean commit")
    verifier_commit = _git(repository, "rev-parse", "HEAD")
    verifier_relative = str(Path(__file__).resolve().relative_to(repository))
    verifier_sha256 = _sha(repository / verifier_relative)
    blob = subprocess.run(
        ["git", "show", f"{verifier_commit}:{verifier_relative}"],
        cwd=repository, capture_output=True, check=False,
    )
    if blob.returncode or sha256(blob.stdout).hexdigest() != verifier_sha256:
        raise CompositeWidthVerificationError("verifier Git binding differs")

    preregistration = base._read(repository / producer.PREREGISTRATION_RELATIVE)
    rows, resident = producer.validate_preregistration(repository, preregistration)
    if len(rows) != EXPECTED_QUERY_COUNT:
        raise CompositeWidthVerificationError("width query count differs")
    p4, p12, result = _load_outputs(repository, preregistration, rows)
    semantic_observations = _validate_cases(p4, p12, rows)
    link_observations = _validate_links(repository, resident, rows, p12["cases"])
    _validate_result(result, preregistration, p4, p12)

    gates = {
        "producer_runtime_and_preregistration_valid": True,
        "all_receipt_seals_and_aggregate_bindings_valid": True,
        "all_24_certificate_digests_reconstructed": True,
        "all_12_semantic_outputs_equal": True,
        "all_240_links_resolve_and_are_causal": len(link_observations) * base.TOP_K
            == EXPECTED_LINK_COUNT,
        "all_seven_component_groups_present": (
            set(DistanceConfig().weights) == topology_verifier.EXPECTED_COMPONENTS
        ),
        "p12t1_faster_on_frozen_width": p12["wall_seconds"] < p4["wall_seconds"],
        "zero_process_swap": p4["all_zero_swap"] is True
            and p12["all_zero_swap"] is True,
        "outcomes_or_labels_excluded": True,
    }
    if not all(gates.values()):
        raise CompositeWidthVerificationError("width verification gate failed")
    state = {
        "schema_version": "m04r14-t14-10-wf03-composite-width-verification-v1",
        "status": "complete", "passed": True, "gates": gates,
        "producer_result_digest": result["result_digest"],
        "preregistration_digest": preregistration["preregistration_digest"],
        "verifier_commit": verifier_commit,
        "verifier_runtime_sha256": verifier_sha256,
        "queries_verified": len(rows),
        "certificates_verified": len(p4["cases"]) + len(p12["cases"]),
        "links_verified": len(link_observations) * base.TOP_K,
        "semantic_observation_digest": stable_hash(semantic_observations),
        "link_observation_digest": stable_hash(link_observations),
        "p4t3_wall_seconds": p4["wall_seconds"],
        "p12t1_wall_seconds": p12["wall_seconds"],
        "p12_over_p4_speedup": result["p12_over_p4_speedup"],
        "selected_topology": "p12t1",
        "outcomes_or_labels_used": False,
        "historical_walk_forward_query_outcomes_opened": False,
        "final_period_result_opened": False,
        "production_promotion_authorized": False,
        "elapsed_seconds": perf_counter() - started,
    }
    return base._sealed(state, "verification_digest")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    result = verify(args.repository)
    if not args.dry_run:
        root = args.repository.resolve() / OUTPUT_RELATIVE
        if root.exists() or root.is_symlink():
            raise CompositeWidthVerificationError("verification output exists")
        root.mkdir(parents=True)
        base._atomic(root / "VERIFIED.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
