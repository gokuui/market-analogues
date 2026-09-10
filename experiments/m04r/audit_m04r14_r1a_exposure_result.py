"""Full post-publication integrity reconstruction for the sealed R1-A v2 result.

This intentionally does not modify the preregistered producer or its first verifier:
their byte hashes are part of the frozen v2 preregistration.  It closes the release
verification gap by checking that boundary and reconstructing every deterministic
section of RESULT.json, not only the raw simulation artifacts.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from multiprocessing import get_context
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence
from uuid import uuid4

import numpy as np

from experiments.m04r import m04r14_r1a_exposure_audit as audit
from experiments.m04r import verify_m04r14_r1a_exposure_audit as first_verifier
from market_analogues.types import stable_hash


SCHEMA = "m04r14-r1a-exposure-result-integrity-verification-v1"
OUTPUT = Path(
    "config/data/analogues/m04r14/"
    "r1a-exposure-audit-v2-integrity-verification-v1"
)


class IntegrityAuditError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _semantic_digest(payload: Mapping[str, Any], omitted: set[str]) -> str:
    return stable_hash({key: value for key, value in payload.items() if key not in omitted})


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise IntegrityAuditError(message)


def _reconstruct(
    repository: Path,
    prereg: Mapping[str, Any],
    *,
    workers: int | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, bool]]:
    generation = str(prereg["inputs"]["packed_generation_id"])
    store = audit.PACKED_RESIDENT
    if not (store / "generations" / generation / "manifest.json").is_file():
        store = repository / audit.PACKED_DURABLE
    metadata, manifest = audit._extract_metadata(
        store,
        generation,
        expected_provenance_digest=str(prereg["inputs"]["packed_provenance_digest"]),
    )
    registry = audit._load(repository / audit.REGISTRY)
    seal = audit._load(repository / audit.REGISTRY_SEAL)
    semantic = audit._load(repository / audit.SEMANTIC_VERIFICATION)
    packed_result = audit._load(repository / audit.PACKED_RESULT)

    registry_digest = _semantic_digest(registry, {"registry_digest"})
    seal_digest = _semantic_digest(seal, {"seal_digest", "created_at"})
    semantic_digest = _semantic_digest(semantic, {"result_digest", "created_at"})
    _require(registry["registry_digest"] == registry_digest, "registry digest differs")
    _require(seal["seal_digest"] == seal_digest, "registry seal digest differs")
    _require(seal["registry_digest"] == registry_digest, "registry and seal differ")
    _require(semantic["result_digest"] == semantic_digest, "semantic receipt digest differs")
    _require(semantic.get("semantic_passed") is True, "upstream semantics did not pass")
    _require(semantic.get("real_forward_outcomes_accessed") is False, "semantic receipt used outcomes")
    _require(semantic.get("verified_cases") == 3270, "semantic case count differs")
    _require(semantic.get("verified_matches") == 65400, "semantic match count differs")
    _require(semantic.get("source_content_digest") == prereg["inputs"]["packed_content_digest"], "packed content binding differs")
    _require(semantic.get("registry_digest") == registry_digest, "semantic registry binding differs")
    _require(packed_result.get("result_digest") == prereg["inputs"]["packed_result_digest"], "packed result binding differs")
    _require(packed_result.get("gate_passed") is True, "packed authority did not pass")
    _require(packed_result.get("generation_id") == generation, "packed generation differs")
    _require(packed_result.get("eligible_rows") == len(metadata.episode_ids), "packed row count differs")
    _require(packed_result.get("real_forward_outcomes_accessed") is False, "packed authority used outcomes")

    (
        queries,
        query_rows,
        retrieval_rows,
        episode_counts,
        episode_meta,
        symbol_counts,
        actual,
        case_manifest,
    ) = audit._actual(repository, metadata, registry)
    _require(stable_hash(case_manifest) == semantic["case_manifest_digest"], "case manifest differs")

    source_prefixes = manifest.get("provenance", {}).get("source_prefixes", {})
    for symbol_id, symbol in enumerate(metadata.symbols):
        prefix_rows = int(source_prefixes[symbol]["rows"])
        expected_rows = max((prefix_rows - 252) // 5 + 1, 0)
        actual_rows = int(metadata.stops[symbol_id] - metadata.starts[symbol_id])
        _require(actual_rows == expected_rows, f"stride/lookback accounting differs: {symbol}")

    located = audit._locate_observed(metadata, episode_meta)
    unique_latest = sorted({query.latest_ns for query in queries})
    pools = {value: audit._pool(metadata, value) for value in unique_latest}
    for query in queries:
        audit._query_mapping(metadata, *pools[query.latest_ns], query)

    replicate_count = int(prereg["execution"]["null_replicates"])
    worker_count = max(1, min(workers or int(prereg["execution"]["workers"]), replicate_count))
    observed_indices = sorted(located.values())
    audit._WORK.clear()
    audit._WORK.update({
        "metadata": metadata,
        "queries": queries,
        "pools": pools,
        "observed_positions": {value: index for index, value in enumerate(observed_indices)},
        "seed": int(prereg["execution"]["seed"]),
    })
    chunks = tuple(tuple(range(first, replicate_count, worker_count)) for first in range(worker_count))
    if worker_count == 1:
        parts = [audit._simulate(chunks[0])]
    else:
        with get_context("fork").Pool(worker_count) as pool:
            parts = pool.map(audit._simulate, chunks)
    null_rows = sorted(
        (row for part in parts for row in part["metrics"]),
        key=lambda row: row["replicate"],
    )
    _require(len(null_rows) == replicate_count, "null replicate accounting differs")
    entity_hits = sum(
        (part["entity_hits"] for part in parts),
        np.zeros(len(observed_indices), dtype=np.uint64),
    )
    null_symbol_hits = sum(
        (part["symbol_hits"] for part in parts),
        np.zeros(len(metadata.symbols), dtype=np.uint64),
    )

    comparison = audit._summaries(actual, null_rows, prereg["metric_directions"])
    latest_values = np.asarray([query.latest_ns for query in queries], dtype=np.int64)
    query_by_symbol = {query.symbol: query for query in queries}
    index_position = {value: index for index, value in enumerate(observed_indices)}
    entity_rows = []
    for episode_id in sorted(episode_counts, key=lambda value: (-episode_counts[value], value)):
        symbol, cutoff, _ = episode_meta[episode_id]
        exposure = int(np.sum(latest_values >= cutoff))
        own = query_by_symbol.get(symbol)
        if own is not None and cutoff >= own.start_ns and cutoff <= own.latest_ns:
            exposure -= 1
        expected = float(entity_hits[index_position[located[episode_id]]] / replicate_count)
        entity_rows.append({
            "episode_id": episode_id,
            "symbol": symbol,
            "cutoff": np.datetime_as_string(np.datetime64(cutoff, "ns")),
            "observed_count": episode_counts[episode_id],
            "eligible_query_exposure": exposure,
            "observed_per_eligible_query": episode_counts[episode_id] / exposure,
            "null_expected_count": expected,
            "observed_to_null_expected": episode_counts[episode_id] / expected if expected else None,
        })

    symbol_ids = {symbol: index for index, symbol in enumerate(metadata.symbols)}
    symbol_rows = []
    for symbol in sorted(symbol_counts, key=lambda value: (-symbol_counts[value], value)):
        symbol_id = symbol_ids[symbol]
        eligible_counts = [
            audit._eligible_symbol_count(metadata, pools[query.latest_ns][0], query, symbol_id)
            for query in queries
        ]
        eligible_queries = sum(value > 0 for value in eligible_counts)
        selection_capacity = sum(
            min(3, 1 + (value - 1) // 51) if value else 0
            for value in eligible_counts
        )
        symbol_rows.append({
            "symbol": symbol,
            "observed_count": symbol_counts[symbol],
            "eligible_query_exposure": eligible_queries,
            "cap_overlap_selection_capacity": selection_capacity,
            "observed_per_capacity": symbol_counts[symbol] / selection_capacity,
            "null_expected_count": float(null_symbol_hits[symbol_id] / replicate_count),
        })

    identity = audit._retrieval_identity(retrieval_rows)
    finite_ratios = [
        float(row["distance_rank20_rank1_ratio"])
        for row in query_rows
        if row["distance_rank20_rank1_ratio"] is not None
    ]
    producer_gates = {
        "upstream_semantics_verified": True,
        "upstream_case_manifest_unchanged": stable_hash(case_manifest) == semantic["case_manifest_digest"],
        "case_seals_reconstructed": True,
        "all_risk_set_counts_equal_certificates": True,
        "packed_main_and_overflow_included": (
            len(metadata.episode_ids) == int(manifest["row_count"]) + int(manifest["overflow_count"])
        ),
        "stride_5_lookback_252_accounting": True,
        "exact_cap_three_overlap_null": True,
        "retrieval_unchanged": True,
        "outcomes_excluded": True,
        "null_accounting_complete": len(null_rows) == replicate_count,
    }
    _require(all(producer_gates.values()), "reconstructed producer gate failed")
    reconstructed = {
        "schema_version": audit.SCHEMA,
        "status": "diagnostic_only_r1a_complete",
        "passed": True,
        "production_promotion_authorized": False,
        "adequacy_labels_authorized": False,
        "predictive_claim_authorized": False,
        "real_forward_outcomes_accessed": False,
        "preregistration_digest": prereg["preregistration_digest"],
        "inputs": {
            "registry_digest": registry_digest,
            "semantic_verification_digest": semantic_digest,
            "packed_generation_id": generation,
            "packed_provenance_digest": manifest["provenance_digest"],
            "case_manifest_digest": stable_hash(case_manifest),
            "retrieval_identity_digest": stable_hash(identity),
        },
        "inventory": {
            "queries": len(queries),
            "retrieved_links": len(identity),
            "candidate_episodes": len(metadata.episode_ids),
            "candidate_symbols": len(metadata.symbols),
            "unique_latest_eligible_cutoffs": len(unique_latest),
        },
        "distance_geometry": {
            "rank1_zero_count": len(query_rows) - len(finite_ratios),
            "rank20_rank1_ratio_p10": float(np.quantile(finite_ratios, 0.1)),
            "rank20_rank1_ratio_p50": float(np.quantile(finite_ratios, 0.5)),
            "rank20_rank1_ratio_p90": float(np.quantile(finite_ratios, 0.9)),
        },
        "actual": actual,
        "comparison": comparison,
        "null": {
            "model": "independent per-query iid continuous random priorities over exact causal risk set, followed by production greedy cap-3 inclusive-overlap selection",
            "replicates": replicate_count,
            "seed": int(prereg["execution"]["seed"]),
            "metrics_digest": stable_hash(null_rows),
        },
        "most_recurrent_episodes_with_exposure": entity_rows[:100],
        "top_symbols": symbol_rows[:100],
        "gates": producer_gates,
        "remaining_r1": [
            "matched causal query-distance nulls",
            "directed reciprocity registry",
            "leave-one-group/input-perturbation/nearby-cutoff stability",
            "synthetic positive/novel-path adequacy-label calibration",
        ],
    }
    integrity_gates = {
        "upstream_authorities_reconstructed": True,
        "case_manifest_reconstructed": True,
        "packed_generation_reconstructed": manifest["manifest_digest"] == generation,
        "all_null_replicates_replayed": True,
        "all_comparison_summaries_reconstructed": True,
        "inventory_reconstructed": True,
        "distance_geometry_reconstructed": True,
        "episode_exposure_table_reconstructed": True,
        "symbol_exposure_table_reconstructed": True,
        "remaining_r1_boundary_reconstructed": True,
    }
    return reconstructed, null_rows, query_rows, integrity_gates


def execute(repository: Path, *, workers: int | None = None) -> dict[str, Any]:
    repository = repository.resolve()
    output = repository / OUTPUT
    if output.exists() or output.is_symlink():
        raise IntegrityAuditError(f"create-only output exists: {output}")

    prereg_path = repository / audit.PREREGISTRATION
    prereg = audit._load(prereg_path)
    prereg_digest = _semantic_digest(prereg, {"preregistration_digest"})
    _require(prereg.get("preregistration_digest") == prereg_digest, "preregistration digest differs")
    for relative, expected in prereg["inputs"]["file_sha256"].items():
        _require(_sha(repository / relative) == expected, f"frozen input changed: {relative}")
    for relative, expected in prereg["runtime_files"].items():
        _require(_sha(repository / relative) == expected, f"preregistered runtime changed: {relative}")

    root = repository / audit.OUTPUT
    result = audit._load(root / "RESULT.json")
    result_digest = _semantic_digest(result, {"result_digest", "elapsed_seconds"})
    _require(result.get("result_digest") == result_digest, "result semantic digest differs")
    _require(result.get("preregistration_digest") == prereg_digest, "result preregistration binding differs")
    _require(first_verifier._html_valid(root / "report.html"), "result report HTML is invalid")
    expected_artifacts = {
        "null_replicates_sha256": _sha(root / "NULL_REPLICATES.json"),
        "query_diagnostics_sha256": _sha(root / "QUERY_DIAGNOSTICS.json"),
    }
    _require(result.get("artifacts") == expected_artifacts, "result artifact hashes differ")

    first_path = repository / first_verifier.OUTPUT / "VERIFIED.json"
    first = audit._load(first_path)
    first_digest = _semantic_digest(first, {"verification_digest", "created_at"})
    _require(first.get("verification_digest") == first_digest, "first verification digest differs")
    _require(first.get("passed") is True, "first verifier did not pass")
    _require(first.get("verified_result_digest") == result_digest, "first verifier result binding differs")
    _require(all(first.get("gates", {}).values()), "first verifier contains a failed gate")

    reconstructed, null_rows, query_rows, reconstructed_gates = _reconstruct(
        repository,
        prereg,
        workers=workers,
    )
    stored_null = json.loads((root / "NULL_REPLICATES.json").read_bytes())
    stored_queries = json.loads((root / "QUERY_DIAGNOSTICS.json").read_bytes())
    _require(stored_null == null_rows, "stored null replicates differ from replay")
    _require(stored_queries == query_rows, "stored query diagnostics differ from reconstruction")

    stored_projection = {
        key: value
        for key, value in result.items()
        if key not in {"result_digest", "elapsed_seconds", "artifacts"}
    }
    _require(stored_projection == reconstructed, "full deterministic RESULT payload differs")
    gates = {
        "preregistration_digest_reconstructed": True,
        "result_preregistration_binding_reconstructed": True,
        "all_preregistered_input_hashes_reconstructed": True,
        "all_preregistered_runtime_hashes_reconstructed": True,
        "result_digest_reconstructed": True,
        "result_artifact_hashes_reconstructed": True,
        "result_html_valid": True,
        "first_verification_receipt_reconstructed": True,
        "first_verification_result_binding_reconstructed": True,
        "full_deterministic_result_payload_reconstructed": True,
        "stored_query_diagnostics_reconstructed": True,
        "stored_null_replicates_reconstructed": True,
        **reconstructed_gates,
    }
    _require(all(gates.values()), "one or more integrity gates failed")
    state = {
        "schema_version": SCHEMA,
        "passed": True,
        "status": "full_integrity_reconstruction",
        "production_promotion_authorized": False,
        "predictive_claim_authorized": False,
        "adequacy_labels_authorized": False,
        "real_forward_outcomes_accessed": False,
        "verified_preregistration_digest": prereg_digest,
        "verified_result_digest": result_digest,
        "verified_first_verification_digest": first_digest,
        "verified_null_replicates": len(null_rows),
        "verified_queries": len(query_rows),
        "gates": gates,
    }
    payload = {
        **state,
        "verification_digest": stable_hash(state),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    temp = output.parent / f".{output.name}.tmp-{os.getpid()}-{uuid4().hex}"
    try:
        temp.mkdir(parents=True)
        (temp / "VERIFIED.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        temp.rename(output)
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--workers", type=int)
    args = parser.parse_args(argv)
    print(json.dumps(execute(args.repository, workers=args.workers), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
