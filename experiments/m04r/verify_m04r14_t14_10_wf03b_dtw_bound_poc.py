"""Independently replay and verify the WF-03B real DTW-bound tightness gate."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from market_analogues.causal_prefix import causal_prefix_digest
from market_analogues.distance import complete_representation_distance
from market_analogues.dtw_interval_bound import (
    quantize_dtw_samples,
    quantized_dtw_lower_bound,
)
from market_analogues.exact_batch import (
    batch_representation_lower_bounds,
    exact_representations_at_positions,
)
from market_analogues.packed_bound_search import BoundProposal, scan_packed_bound_threshold
from market_analogues.packed_bound_store import load_packed_generation
from market_analogues.types import EpisodeKey, InstrumentKey, stable_hash

from experiments.m04r import m04r14_t14_10_wf03_feasibility as base
from experiments.m04r import m04r14_t14_10_wf03b_price_threshold_diagnostic as diagnostic


PRODUCER_ROOT = Path("config/data/analogues/m04r14/t14-10-wf03b-dtw-bound-poc-v1")
PREREGISTRATION = Path(
    "experiments/m04r/m04r14_t14_10_wf03b_dtw_bound_poc_preregistered.json"
)
VERIFICATION_ROOT = Path(
    "config/data/analogues/m04r14/t14-10-wf03b-dtw-bound-poc-v1-verification"
)
SAMPLE_ROWS = 4_096
TOLERANCE = 1e-12


class IndependentVerificationError(RuntimeError):
    pass


def _key(query_id: str, episode_id: str) -> tuple[str, str]:
    return stable_hash({
        "schema": "wf03b-dtw-bound-sample-v1",
        "query_id": query_id, "episode_id": episode_id,
    }), episode_id


def _retain(
    retained: Iterable[BoundProposal], additions: Iterable[BoundProposal], query_id: str,
) -> list[BoundProposal]:
    rows = {row.episode_id: row for row in retained}
    for row in additions:
        if row.episode_id in rows and rows[row.episode_id] != row:
            raise IndependentVerificationError("conflicting sampled episode")
        rows[row.episode_id] = row
    return sorted(rows.values(), key=lambda row: _key(query_id, row.episode_id))[:SAMPLE_ROWS]


def _score_group(
    symbol: str, proposals: Sequence[BoundProposal], *, source: Any,
    benchmark: pd.DataFrame, query: Any, manifest: Mapping[str, Any],
    query_id: str, lookback: int, representation_version: str,
) -> list[dict[str, Any]]:
    expected = manifest["provenance"]["source_prefixes"].get(symbol)
    bars = source.load(InstrumentKey("nasdaq", symbol))
    if type(expected) is not dict or asdict(causal_prefix_digest(
            bars, expected["requested_cutoff"])) != expected:
        raise IndependentVerificationError("source prefix differs")
    maximum = max(row.cutoff_ns for row in proposals)
    frame = bars[bars.timestamp.astype("int64") <= maximum].reset_index(drop=True)
    positions = {int(pd.Timestamp(value).value): index
                 for index, value in enumerate(frame.timestamp)}
    requested = []
    for proposal in proposals:
        position = positions.get(proposal.cutoff_ns)
        if position is None or position + 1 < lookback:
            raise IndependentVerificationError("candidate cutoff differs")
        identity = EpisodeKey(
            InstrumentKey("nasdaq", symbol), pd.Timestamp(proposal.cutoff_ns),
            lookback, representation_version,
        )
        if identity.id != proposal.episode_id:
            raise IndependentVerificationError("episode identity differs")
        requested.append(position)
    representations = exact_representations_at_positions(
        frame, benchmark, positions=np.asarray(requested, dtype=int), lookback=lookback,
    )
    lower = batch_representation_lower_bounds(query, representations)
    output = []
    for index, (proposal, candidate) in enumerate(zip(
            proposals, representations, strict=True)):
        native = float(lower.components["price"][index])
        components = {name: float(values[index])
                      for name, values in lower.components.items()}
        _total, exact_components, _alignment = complete_representation_distance(
            query, candidate, float(lower.totals[index]), components,
            float(lower.rigid_price[index]), reconstruct_path=False,
        )
        exact = float(exact_components["price"])
        interval = quantized_dtw_lower_bound(query, quantize_dtw_samples(candidate))
        enhanced = proposal.lower_bound + .45 * interval
        if not np.isfinite((native, exact, interval, enhanced)).all() \
                or proposal.lower_bound > native + TOLERANCE \
                or native > exact + TOLERANCE or enhanced > exact + TOLERANCE:
            raise IndependentVerificationError("real bound safety differs")
        output.append({
            "episode_id": proposal.episode_id, "symbol": symbol,
            "selection_key": _key(query_id, proposal.episode_id)[0],
            "packed_bound": proposal.lower_bound, "native_bound": native,
            "dtw_interval_bound": interval, "enhanced_bound": enhanced,
            "exact_price": exact,
        })
    return output


def _hex_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{key: value.hex() if isinstance(value, float) else value
             for key, value in row.items()} for row in rows]


def _verify_case(
    repository: Path, expected: Mapping[str, Any], prereg_case: Mapping[str, Any],
    registry_row: Mapping[str, Any], resident: Mapping[str, Any],
) -> dict[str, Any]:
    source, episode, _request, packed = base._context(repository, registry_row)
    excluded = frozenset({row["query_id"]: row
                          for row in diagnostic._failed_cases(repository)}[
                              expected["query_id"]]["excluded_episode_ids"])
    sample: list[BoundProposal] = []

    def consume(rows: tuple[BoundProposal, ...]) -> None:
        nonlocal sample
        sample = _retain(sample, rows, expected["query_id"])

    scan = scan_packed_bound_threshold(
        Path(resident["store_root"]), base.GENERATION_ID, packed,
        component="price", branch_aware=True,
        upper_inclusive=float(expected["threshold"]),
        excluded_episode_ids=excluded, block_rows=4_097, block_order="reverse",
        verify_content=False, expected_provenance_digest=base.PROVENANCE_DIGEST,
        consume=consume,
    )
    sample_digest = stable_hash([
        {"episode_id": row.episode_id,
         "selection_key": _key(expected["query_id"], row.episode_id)[0]}
        for row in sample
    ])
    if not all((
        len(sample) == SAMPLE_ROWS,
        scan.admitted_rows == prereg_case["additional_admitted"],
        scan.admitted_set_digest == prereg_case["admitted_set_digest"],
        sample_digest == expected["sample_digest"],
    )):
        raise IndependentVerificationError("reverse sample reconstruction differs")
    loaded = load_packed_generation(
        Path(resident["store_root"]), base.GENERATION_ID,
        expected_provenance_digest=base.PROVENANCE_DIGEST,
        verify_content=False, validate_records=False,
    )
    benchmark = source.load_benchmark()
    benchmark_prefix = loaded.manifest["provenance"]["benchmark_prefix"]
    if benchmark is None or asdict(causal_prefix_digest(
            benchmark, benchmark_prefix["requested_cutoff"])) != benchmark_prefix:
        raise IndependentVerificationError("benchmark prefix differs")
    grouped: dict[str, list[BoundProposal]] = {}
    for row in sample:
        grouped.setdefault(row.symbol, []).append(row)
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(
            _score_group, symbol, tuple(rows), source=source, benchmark=benchmark,
            query=packed.representation, manifest=loaded.manifest,
            query_id=expected["query_id"], lookback=episode.key.lookback,
            representation_version=episode.key.representation_version,
        ) for symbol, rows in sorted(grouped.items(), reverse=True)]
        scores = [item for future in futures for item in future.result()]
    scores.sort(key=lambda row: (row["selection_key"], row["episode_id"]))
    if stable_hash(_hex_rows(scores)) != expected["score_digest"]:
        raise IndependentVerificationError("independent exact score digest differs")
    threshold = float(expected["threshold"])
    pruned = sum(row["enhanced_bound"] > threshold for row in scores)
    false_prunes = sum(row["enhanced_bound"] > threshold
                       and row["exact_price"] <= threshold for row in scores)
    maximum_excess = max(row["enhanced_bound"] - row["exact_price"] for row in scores)
    packed_excess = max(row["packed_bound"] - row["native_bound"] for row in scores)
    if not all((
        pruned == expected["enhanced_pruned"], false_prunes == expected["false_prunes"] == 0,
        maximum_excess == expected["maximum_bound_excess"],
        packed_excess == expected["maximum_packed_native_excess"],
    )):
        raise IndependentVerificationError("independent case aggregate differs")
    return {
        "ordinal": expected["ordinal"], "query_id": expected["query_id"],
        "reverse_admitted_set_digest": scan.admitted_set_digest,
        "sample_digest": sample_digest, "score_digest": expected["score_digest"],
        "sample_rows": len(scores), "enhanced_pruned": pruned,
        "enhanced_pruned_fraction": pruned / len(scores),
        "false_prunes": false_prunes, "maximum_bound_excess": maximum_excess,
    }


def verify(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    root = repository / PRODUCER_ROOT
    prereg = base._read(repository / PREREGISTRATION)
    base._validate_seal(prereg, "preregistration_digest")
    contract = base._read(root / "CONTRACT.json")
    result = base._read(root / "RESULT.json")
    base._validate_seal(result)
    expected_files = {"CONTRACT.json", "RESULT.json", *(f"CASE-{i:03d}.json" for i in range(3))}
    if contract != prereg or {path.name for path in root.iterdir()} != expected_files \
            or result.get("passed") is not True or result.get("gate", {}).get(
                "auxiliary_store_authorized") is not True:
        raise IndependentVerificationError("producer root differs")
    head = prereg["implementation_commit"]
    subprocess.run(["git", "merge-base", "--is-ancestor", head, "HEAD"],
                   cwd=repository, check=True)
    for path, digest in prereg["runtime_files"].items():
        blob = subprocess.run(["git", "show", f"{head}:{path}"], cwd=repository,
                              capture_output=True, check=False)
        if blob.returncode or sha256(blob.stdout).hexdigest() != digest \
                or base._sha(repository / path) != digest:
            raise IndependentVerificationError("frozen runtime binding differs")
    _registry, by_id = base._registry(repository)
    resident = base._resident()
    cases = []
    for prereg_case in prereg["cases"]:
        expected = base._read(root / f"CASE-{int(prereg_case['ordinal']):03d}.json")
        base._validate_seal(expected, "case_digest")
        if expected != result["cases"][int(prereg_case["ordinal"])]:
            raise IndependentVerificationError("case/root projection differs")
        cases.append(_verify_case(
            repository, expected, prereg_case, by_id[expected["query_id"]], resident,
        ))
    total = sum(row["sample_rows"] for row in cases)
    aggregate = sum(row["enhanced_pruned"] for row in cases) / total
    if aggregate != result["gate"]["observed_aggregate_pruned_fraction"] \
            or aggregate < prereg["execution"]["minimum_aggregate_pruned_fraction"] \
            or any(row["enhanced_pruned_fraction"] <
                   prereg["execution"]["minimum_case_pruned_fraction"] for row in cases):
        raise IndependentVerificationError("promotion gate differs")
    state = {
        "schema_version": "m04r14-t14-10-wf03b-dtw-bound-verification-v1",
        "status": "verified", "passed": True,
        "producer_result_digest": result["result_digest"],
        "producer_result_sha256": base._sha(root / "RESULT.json"),
        "cases": cases, "case_digest": stable_hash(cases),
        "observed_aggregate_pruned_fraction": aggregate,
        "auxiliary_store_authorized": True,
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
    result = verify(args.repository)
    if not args.dry_run:
        root = args.repository.resolve() / VERIFICATION_ROOT
        if root.exists() or root.is_symlink():
            raise IndependentVerificationError("verification root exists")
        root.mkdir(parents=True)
        base._atomic(root / "VERIFIED.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
