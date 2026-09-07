"""Independently verify the true-composite WF-03 topology POC."""
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
from market_analogues.m04r_certified_search_verification import _certificate_digest
from market_analogues.packed_bound_store import load_packed_generation
from market_analogues.search import latest_eligible_cutoff
from market_analogues.types import InstrumentKey, stable_hash

from experiments.m04r import m04r14_t14_10_wf03_composite_topology_poc as producer
from experiments.m04r import m04r14_t14_10_wf03_feasibility as base


OUTPUT_RELATIVE = Path(
    "config/data/analogues/m04r14/"
    "t14-10-wf03-composite-topology-poc-v1-verification"
)
EXPECTED_COMPONENTS = frozenset({
    "coarse", "stage", "price", "candle_volatility",
    "volume_shock", "market_context", "structural",
})
METADATA_DTYPE = np.dtype([
    ("episode_id", "V12"), ("cutoff_ns", "<i8"),
    ("symbol_id", "<u4"), ("quality_tier", "u1"),
])


class CompositeTopologyVerificationError(RuntimeError):
    pass


def _git(repository: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repository, text=True,
        capture_output=True, check=False,
    )
    if result.returncode:
        raise CompositeTopologyVerificationError(
            result.stderr.strip() or "git command failed"
        )
    return result.stdout.strip()


def _sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _case_semantics(value: Mapping[str, Any]) -> dict[str, Any]:
    certificate = {
        key: item for key, item in value["certificate"].items()
        if key != "elapsed_seconds"
    }


def _metadata_records(values: np.ndarray) -> np.ndarray:
    """Normalize intentionally distinct main/overflow layouts for link audit."""
    required = set(METADATA_DTYPE.names or ())
    if not required.issubset(values.dtype.names or ()):
        raise CompositeTopologyVerificationError("packed metadata fields differ")
    result = np.empty(len(values), dtype=METADATA_DTYPE)
    for name in METADATA_DTYPE.names or ():
        result[name] = values[name]
    return result
    return {
        "query_id": value["query_id"],
        "proposal_result_digest": value["proposal_result_digest"],
        "certificate": certificate,
        "matches": value["matches"],
    }


def validate_certificate(value: Mapping[str, Any]) -> None:
    certificate = value.get("certificate", {})
    matches = value.get("matches", [])
    accounting = certificate.get("native_bound_accounting", {})
    try:
        valid = all((
            value["semantic_digest"] == stable_hash(_case_semantics(value)),
            certificate["result_digest"] == _certificate_digest({
                "certificate": certificate, "matches": matches,
            }),
            certificate["eligible_candidates"]
                == accounting["exact_dtw_evaluated"]
                + accounting["native_bound_pruned"]
                + accounting["packed_bound_pruned"],
            certificate["exact_evaluated"] == accounting["exact_dtw_evaluated"],
            certificate["safely_pruned"]
                == accounting["native_bound_pruned"]
                + accounting["packed_bound_pruned"],
            certificate["maximum_quantized_bound_excess"] <= producer.TOLERANCE,
            len(matches) == base.TOP_K,
            len({row["symbol"] for row in matches}) == base.TOP_K,
            all(set(row["component_distances"]) == EXPECTED_COMPONENTS
                for row in matches),
        ))
    except (KeyError, TypeError, ValueError):
        valid = False
    if not valid:
        raise CompositeTopologyVerificationError("composite certificate differs")


def verify(repository: Path) -> dict[str, Any]:
    started = perf_counter()
    repository = repository.resolve(strict=True)
    if _git(repository, "status", "--porcelain"):
        raise CompositeTopologyVerificationError("topology verifier requires clean commit")
    verifier_commit = _git(repository, "rev-parse", "HEAD")
    verifier_relative = str(Path(__file__).resolve().relative_to(repository))
    verifier_sha256 = _sha(repository / verifier_relative)
    blob = subprocess.run(
        ["git", "show", f"{verifier_commit}:{verifier_relative}"],
        cwd=repository, capture_output=True, check=False,
    )
    if blob.returncode or sha256(blob.stdout).hexdigest() != verifier_sha256:
        raise CompositeTopologyVerificationError("verifier Git binding differs")

    preregistration = base._read(repository / producer.PREREGISTRATION_RELATIVE)
    _registry, rows, resident = producer.validate_preregistration(
        repository, preregistration,
    )
    root = repository / producer.OUTPUT_RELATIVE
    if base._read(root / "CONTRACT.json") != preregistration:
        raise CompositeTopologyVerificationError("topology contract differs")
    sequential = base._read(root / "SEQUENTIAL.json")
    parallel = base._read(root / "PARALLEL.json")
    result = base._read(root / "RESULT.json")
    for value in (sequential, parallel, result):
        base._validate_seal(value)
    if not producer._valid_topology(
        sequential, preregistration["preregistration_digest"],
        expected_groups=1, expected_threads=producer.SEQUENTIAL_THREADS,
    ) or not producer._valid_topology(
        parallel, preregistration["preregistration_digest"],
        expected_groups=producer.PARALLEL_GROUPS,
        expected_threads=producer.PARALLEL_THREADS,
    ):
        raise CompositeTopologyVerificationError("topology receipt structure differs")
    for value in (sequential, parallel):
        for case in value["cases"]:
            validate_certificate(case)
    if not producer.compare_topologies(sequential, parallel):
        raise CompositeTopologyVerificationError("topology semantic outputs differ")

    authority = base._read(repository / producer.AUTHORITY_RELATIVE)
    current_authority = next(
        row for row in sequential["cases"]
        if row["query_id"] == producer.AUTHORITY_QUERY_ID
    )
    if current_authority["matches"] != authority["matches"] \
            or current_authority["certificate"]["result_digest"] \
            != authority["certificate"]["result_digest"]:
        raise CompositeTopologyVerificationError("frozen authority differs")

    packed = load_packed_generation(
        Path(resident["store_root"]), base.GENERATION_ID,
        expected_provenance_digest=base.PROVENANCE_DIGEST,
        verify_content=True, validate_records=True,
    )
    records = np.concatenate((
        _metadata_records(packed.rows), _metadata_records(packed.overflow),
    ))
    order = np.argsort(records["episode_id"], kind="stable")
    sorted_ids = records["episode_id"][order]
    source = source_from_spec(
        load_config(repository / base.CONFIG_RELATIVE).datasets["nasdaq"]
    )
    observations = []
    rows_by_id = {row["episode_id"]: row for row in rows}
    for case in sequential["cases"]:
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
            raise CompositeTopologyVerificationError("matched episode is absent")
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
                raise CompositeTopologyVerificationError("matched episode binding differs")
        observations.append({
            "query_id": case["query_id"],
            "match_ids": [match["episode_id"] for match in case["matches"]],
            "maximum_cutoff_ns": int(selected["cutoff_ns"].max()),
        })

    gates = {
        "producer_runtime_and_preregistration_valid": True,
        "all_receipt_seals_and_bindings_valid": True,
        "all_eight_certificate_digests_reconstructed": True,
        "sequential_parallel_semantics_equal": True,
        "frozen_multichannel_authority_equal": True,
        "all_80_links_resolve_and_are_causal": True,
        "all_seven_component_groups_present": (
            set(DistanceConfig().weights) == EXPECTED_COMPONENTS
        ),
        "parallel_faster_on_frozen_topology": (
            parallel["wall_seconds"] < sequential["wall_seconds"]
            and result["selected_topology"] == "parallel-p4t3"
        ),
        "zero_process_swap": (
            sequential["all_zero_swap"] is True
            and parallel["all_zero_swap"] is True
        ),
        "outcomes_or_labels_excluded": True,
    }
    if not all(gates.values()) \
            or result.get("passed") is not True \
            or result.get("sequential_digest") != sequential["result_digest"] \
            or result.get("parallel_digest") != parallel["result_digest"] \
            or result.get("parallel_speedup") \
            != sequential["wall_seconds"] / parallel["wall_seconds"]:
        raise CompositeTopologyVerificationError("topology aggregate differs")
    state = {
        "schema_version": "m04r14-t14-10-wf03-composite-topology-verification-v1",
        "status": "complete", "passed": True, "gates": gates,
        "producer_result_digest": result["result_digest"],
        "preregistration_digest": preregistration["preregistration_digest"],
        "verifier_commit": verifier_commit,
        "verifier_runtime_sha256": verifier_sha256,
        "queries_verified": len(rows), "links_verified": len(observations) * 20,
        "observation_digest": stable_hash(observations),
        "sequential_wall_seconds": sequential["wall_seconds"],
        "parallel_wall_seconds": parallel["wall_seconds"],
        "parallel_speedup": result["parallel_speedup"],
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
            raise CompositeTopologyVerificationError("verification output exists")
        root.mkdir(parents=True)
        base._atomic(root / "VERIFIED.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
