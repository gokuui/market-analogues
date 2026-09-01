"""Run the frozen T14-09 synthetic outcome gate before real outcome access."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
import multiprocessing
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any, Mapping, Sequence

import pandas as pd

from experiments.m04r.m04r14_t14_09_outcome_oracle import reference_episode
from market_analogues.causal_outcomes import (
    CausalOutcomeError,
    compute_episode_outcomes,
    deduplicate_episode_requests,
)
from market_analogues.types import stable_hash


SCHEMA = "m04r14-t14-09-synthetic-outcome-gate-v1"
OUTPUT = Path("config/data/analogues/m04r14/t14-09-synthetic-outcome-gate-v1")
CONTRACT = Path("config/m04r14-t14-09-outcome-contract.json")
KINDS = (
    "no_touch", "favorable", "adverse", "ambiguous", "incomplete",
    "missing_benchmark_origin", "missing_benchmark_endpoint",
    "missing_stock_session", "insufficient_atr", "scaled",
)


class SyntheticGateError(RuntimeError):
    pass


def _null(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _null(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_null(item) for item in value]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value.item() if hasattr(value, "item") else value


def _frames(kind: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    rows = 36 if kind == "incomplete" else 180
    cutoff_index = 10 if kind == "insufficient_atr" else 30
    dates = pd.bdate_range("2020-01-01", periods=rows)
    stock = pd.DataFrame({
        "timestamp": dates, "open": 100.0, "high": 101.0,
        "low": 99.0, "close": 100.0,
    })
    benchmark = pd.DataFrame({
        "timestamp": dates, "open": 200.0, "high": 202.0,
        "low": 198.0, "close": 200.0,
    })
    cutoff = dates[cutoff_index]
    if kind == "favorable":
        stock.loc[cutoff_index + 3, "high"] = 104.0
    elif kind == "adverse":
        stock.loc[cutoff_index + 2, "low"] = 98.0
    elif kind == "ambiguous":
        stock.loc[cutoff_index + 4, ["high", "low"]] = [104.0, 98.0]
    elif kind == "missing_benchmark_origin":
        benchmark = benchmark.loc[benchmark.timestamp != cutoff].reset_index(drop=True)
    elif kind == "missing_benchmark_endpoint":
        benchmark = benchmark.drop(index=cutoff_index + 5).reset_index(drop=True)
    elif kind == "missing_stock_session":
        stock = stock.drop(index=cutoff_index + 2).reset_index(drop=True)
    elif kind == "scaled":
        stock[["open", "high", "low", "close"]] *= 17.0
    return stock, benchmark, cutoff


def _case(kind: str, contract_digest: str) -> dict[str, Any]:
    stock, benchmark, cutoff = _frames(kind)
    bindings = {
        "episode_id": f"synthetic-{kind}", "source_fingerprint": f"source-{kind}",
        "contract_digest": contract_digest, "source_content_digest": "synthetic-source",
    }
    production = compute_episode_outcomes(
        stock, benchmark, cutoff=cutoff, **bindings,
    )
    expected_outcomes, expected_paths = reference_episode(
        stock, benchmark, cutoff=cutoff, **bindings,
    )
    observed = _null({
        "outcomes": production.outcomes.to_dict("records"),
        "paths": production.paths.to_dict("records"),
    })
    expected = _null({"outcomes": expected_outcomes, "paths": expected_paths})
    return {
        "kind": kind, "passed": observed == expected,
        "production_digest": stable_hash(observed),
        "oracle_digest": stable_hash(expected),
        "outcome_rows": len(observed["outcomes"]),
        "path_rows": len(observed["paths"]),
    }


def _read_contract(repository: Path) -> dict[str, Any]:
    path = repository / CONTRACT
    value = json.loads(path.read_text())
    state = {key: item for key, item in value.items() if key != "contract_digest"}
    if value.get("contract_digest") != stable_hash(state) \
            or value.get("status") != "frozen_before_real_forward_outcome_access":
        raise SyntheticGateError("outcome contract is not frozen")
    return value


def execute(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    contract = _read_contract(repository)
    digest = str(contract["contract_digest"])
    started = perf_counter()
    serial = [_case(kind, digest) for kind in KINDS]
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=12, mp_context=context) as pool:
        parallel = list(pool.map(_case, KINDS, [digest] * len(KINDS)))
    parallel_equal = serial == parallel

    base_stock, base_benchmark, cutoff = _frames("no_touch")
    bindings = {
        "episode_id": "synthetic-mutation", "source_fingerprint": "mutation-source",
        "contract_digest": digest, "source_content_digest": "synthetic-source",
    }
    before = compute_episode_outcomes(
        base_stock, base_benchmark, cutoff=cutoff, **bindings,
    )
    prefix_before = stable_hash(_null(
        base_stock.loc[base_stock.timestamp <= cutoff].to_dict("records")
    ))
    changed = base_stock.copy()
    changed.loc[changed.timestamp > cutoff, "close"] *= 1.1
    changed.loc[changed.timestamp > cutoff, "high"] = changed.loc[
        changed.timestamp > cutoff, ["high", "close"]
    ].max(axis=1)
    after = compute_episode_outcomes(changed, base_benchmark, cutoff=cutoff, **bindings)
    prefix_after = stable_hash(_null(
        changed.loc[changed.timestamp <= cutoff].to_dict("records")
    ))
    future_mutation_isolated = (
        prefix_before == prefix_after
        and stable_hash(_null(before.outcomes.to_dict("records")))
        != stable_hash(_null(after.outcomes.to_dict("records")))
    )

    request = {
        "dataset": "nasdaq", "episode_id": "duplicate",
        "source_fingerprint": "fingerprint", "symbol": "SYN",
    }
    deduplication = deduplicate_episode_requests([request, dict(request)]) == [request]
    conflicting_rejected = False
    try:
        deduplicate_episode_requests([request, {**request, "symbol": "OTHER"}])
    except CausalOutcomeError:
        conflicting_rejected = True

    with TemporaryDirectory() as temporary:
        path = Path(temporary)
        before.outcomes.to_parquet(path / "outcomes.parquet", index=False)
        before.paths.to_parquet(path / "paths.parquet", index=False)
        parquet_roundtrip = all((
            stable_hash(_null(before.outcomes.to_dict("records")))
            == stable_hash(_null(pd.read_parquet(path / "outcomes.parquet").to_dict("records"))),
            stable_hash(_null(before.paths.to_dict("records")))
            == stable_hash(_null(pd.read_parquet(path / "paths.parquet").to_dict("records"))),
        ))
    gates = {
        "production_equals_independent_oracle": all(row["passed"] for row in serial),
        "serial_equals_12_process": parallel_equal,
        "future_mutation_retrieval_isolation": future_mutation_isolated,
        "duplicate_episode_deduplication": deduplication,
        "conflicting_duplicate_rejected": conflicting_rejected,
        "parquet_semantic_roundtrip": parquet_roundtrip,
        "no_real_forward_outcome_input": True,
    }
    state = {
        "schema_version": SCHEMA, "status": "complete",
        "passed": all(gates.values()), "contract_digest": digest,
        "synthetic_cases": list(KINDS), "case_results": serial,
        "gates": gates, "elapsed_seconds": perf_counter() - started,
        "real_forward_outcomes_accessed": False,
        "production_promotion_authorized": False,
    }
    if not state["passed"]:
        raise SyntheticGateError(f"synthetic outcome gate failed: {gates}")
    deterministic = {
        key: value for key, value in state.items() if key != "elapsed_seconds"
    }
    return {**state, "result_digest": stable_hash(deterministic)}


def _publish(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise SyntheticGateError("synthetic gate output exists")
    path.mkdir(parents=False)
    descriptor = os.open(path / "RESULT.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(json.dumps({
            **value, "created_at": datetime.now(timezone.utc).isoformat(),
        }, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    repository = args.repository.resolve(strict=True)
    value = execute(repository)
    if not args.dry_run:
        _publish(repository / OUTPUT, value)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
