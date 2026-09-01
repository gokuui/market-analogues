"""Build and seal the outcome-blind T14-10 historical query registry."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from html import escape
import json
import os
from pathlib import Path
import subprocess
import tempfile
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from market_analogues.adapters import file_fingerprint, source_from_spec
from market_analogues.config import load_config
from market_analogues.types import stable_hash
from market_analogues.walk_forward_registry import (
    LIQUIDITY_CELLS,
    QUALITY_CELLS,
    build_registry,
    canonical_month_cutoffs,
)


SCHEMA = "m04r14-t14-10-walk-forward-query-registry-v1"
SEAL_SCHEMA = "m04r14-t14-10-walk-forward-query-registry-seal-v1"
CONTRACT = Path("config/m04r14-t14-10-walk-forward-contract.json")
SPEC = Path("config/m04r14-t14-10-walk-forward-registry-spec.json")
DENOMINATOR_ROOT = Path("config/data/analogues/m04r14/nasdaq-shadow-denominator-v1")
DEFAULT_OUTPUT = Path(
    "config/data/analogues/m04r14/t14-10-walk-forward-query-registry-v1"
)
FORBIDDEN_KEYS = {
    "query_outcome", "query_outcomes", "forward_return", "forward_returns",
    "winner", "loser", "profit", "loss", "setup", "setup_label", "matches",
}


class RegistryPublicationError(RuntimeError):
    pass


def _read_json(path: Path) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        raise RegistryPublicationError(f"regular JSON file required: {path}")
    raw = path.read_bytes()

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise RegistryPublicationError(f"duplicate JSON key: {path}:{key}")
            result[key] = value
        return result

    value = json.loads(
        raw, object_pairs_hook=pairs,
        parse_constant=lambda item: (_ for _ in ()).throw(
            RegistryPublicationError(f"non-finite JSON: {path}:{item}")
        ),
    )
    if type(value) is not dict:
        raise RegistryPublicationError(f"JSON object required: {path}")
    return value, raw


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for raw in frame.to_dict(orient="records"):
        row: dict[str, Any] = {}
        for key, value in raw.items():
            if value is None or bool(pd.isna(value)):
                row[str(key)] = None
            elif isinstance(value, pd.Timestamp):
                row[str(key)] = value.isoformat()
            elif isinstance(value, np.integer):
                row[str(key)] = int(value)
            elif isinstance(value, np.floating):
                row[str(key)] = float(value)
            elif isinstance(value, np.bool_):
                row[str(key)] = bool(value)
            else:
                row[str(key)] = value
        result.append(row)
    return result


def _contains_forbidden_key(value: Any) -> bool:
    if isinstance(value, dict):
        return bool(FORBIDDEN_KEYS.intersection(map(str, value))) or any(
            _contains_forbidden_key(item) for item in value.values()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_key(item) for item in value)
    return False


def _atomic(path: Path, raw: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def _validate_spec(
    repository: Path, contract: Mapping[str, Any], contract_raw: bytes,
    spec: Mapping[str, Any], spec_raw: bytes,
) -> None:
    state = {key: value for key, value in spec.items() if key != "spec_digest"}
    if not all((
        spec.get("schema_version") == "m04r14-t14-10-walk-forward-registry-spec-v1",
        spec.get("status") == "frozen_before_historical_query_retrieval_or_futures",
        spec.get("spec_digest") == stable_hash(state),
        spec.get("walk_forward_contract_digest") == contract.get("contract_digest"),
        spec.get("walk_forward_contract_sha256") == sha256(contract_raw).hexdigest(),
        spec.get("historical_walk_forward_query_outcomes_opened") is False,
        spec.get("final_period_result_opened") is False,
        spec.get("production_promotion_authorized") is False,
    )):
        raise RegistryPublicationError("registry spec identity or unopened boundary differs")
    implementation = spec.get("implementation", {})
    expected_files = implementation.get("files", {})
    if not expected_files or any(
        not (repository / path).is_file()
        or file_fingerprint(repository / path) != digest
        for path, digest in expected_files.items()
    ):
        raise RegistryPublicationError("frozen registry implementation files differ")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True,
        capture_output=True, check=True,
    ).stdout.strip()
    parents = subprocess.run(
        ["git", "rev-list", "--parents", "-n", "1", head], cwd=repository,
        text=True, capture_output=True, check=True,
    ).stdout.split()
    if len(parents) != 2 or parents[1] != implementation.get("implementation_h0"):
        raise RegistryPublicationError("registry spec must be the sole child of implementation H0")
    if file_fingerprint(repository / SPEC) != sha256(spec_raw).hexdigest():
        raise RegistryPublicationError("registry spec self hash differs")


def execute(
    repository: Path, config_path: Path, output: Path, *, workers: int | None = None,
) -> dict[str, Any]:
    started = perf_counter()
    repository = repository.resolve(strict=True)
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=repository,
        text=True, capture_output=True, check=True,
    ).stdout
    if dirty:
        raise RegistryPublicationError("registry publication requires a globally clean worktree")
    contract, contract_raw = _read_json(repository / CONTRACT)
    spec, spec_raw = _read_json(repository / SPEC)
    _validate_spec(repository, contract, contract_raw, spec, spec_raw)
    denominator_state, denominator_raw = _read_json(
        repository / DENOMINATOR_ROOT / "query-registry.json"
    )
    denominator_seal, denominator_seal_raw = _read_json(
        repository / DENOMINATOR_ROOT / "SEALED.json"
    )
    if not all((
        denominator_state.get("passed") is True,
        denominator_state.get("registry_digest") == spec.get("denominator_registry_digest"),
        sha256(denominator_raw).hexdigest() == spec.get("denominator_registry_sha256"),
        denominator_seal.get("seal_digest") == spec.get("denominator_seal_digest"),
        sha256(denominator_seal_raw).hexdigest() == spec.get("denominator_seal_sha256"),
        denominator_state.get("real_forward_outcomes_accessed") is False,
    )):
        raise RegistryPublicationError("locked NASDAQ denominator binding differs")
    denominator_path = repository / DENOMINATOR_ROOT / "denominator.parquet"
    if file_fingerprint(denominator_path) != spec.get("denominator_parquet_sha256"):
        raise RegistryPublicationError("locked denominator parquet differs")
    denominator = pd.read_parquet(denominator_path)
    config = load_config(config_path)
    source = source_from_spec(config.datasets["nasdaq"])
    benchmark = source.load_benchmark()
    if benchmark is None:
        raise RegistryPublicationError("NASDAQ benchmark is unavailable")
    registry_rule = contract["query_registry"]
    cutoffs = canonical_month_cutoffs(
        benchmark,
        registry_rule["warmup_period"][0],
        registry_rule["scored_period"][1],
    )
    process_count = workers or int(spec["execution"]["processes"])
    build = build_registry(
        denominator, config.datasets["nasdaq"].path, benchmark, cutoffs,
        locked_coverage_cutoff=pd.Timestamp(registry_rule["locked_source_coverage_cutoff"]),
        contract_digest=contract["contract_digest"],
        folds=contract["temporal_protocol"]["folds"],
        target_per_cell=int(registry_rule["target_per_cell_per_month"]),
        lookback=int(registry_rule["minimum_required_prior_sessions"]),
        representation_version=str(spec["representation_version"]),
        workers=process_count,
    )
    queries = build.queries
    scored = queries[queries.scored]
    if len(scored) < int(registry_rule["minimum_scored_queries"]):
        raise RegistryPublicationError(
            f"only {len(scored)} scored queries; require {registry_rule['minimum_scored_queries']}"
        )
    if queries.case_id.duplicated().any() or queries.episode_id.duplicated().any():
        raise RegistryPublicationError("query case or episode identity is duplicated")
    if len(build.accounting) != len(cutoffs) * len(QUALITY_CELLS) * len(LIQUIDITY_CELLS):
        raise RegistryPublicationError("monthly cell accounting is incomplete")
    if int(build.accounting.selected_queries.sum()) != len(queries):
        raise RegistryPublicationError("cell/query accounting does not reconcile")
    if perf_counter() - started > float(spec["execution"]["maximum_build_seconds"]):
        raise RegistryPublicationError("registry build exceeded its frozen performance limit")

    query_records = _records(queries)
    accounting_records = _records(build.accounting)
    candidate_records = _records(build.candidates)
    source_records = _records(build.sources)
    benchmark_records = _records(build.benchmark_prefixes)
    state: dict[str, Any] = {
        "schema_version": SCHEMA, "status": "sealed", "passed": True,
        "walk_forward_contract_digest": contract["contract_digest"],
        "registry_spec_digest": spec["spec_digest"],
        "denominator_registry_digest": denominator_state["registry_digest"],
        "dataset_id": "nasdaq", "month_cutoffs": len(cutoffs),
        "first_cutoff": cutoffs[0].isoformat(), "last_cutoff": cutoffs[-1].isoformat(),
        "locked_source_coverage_cutoff": registry_rule["locked_source_coverage_cutoff"],
        "locked_denominator_symbols": len(denominator),
        "candidate_rows": len(candidate_records), "queries": len(query_records),
        "warmup_queries": int((~queries.scored).sum()),
        "scored_queries": len(scored),
        "shortfall_queries": int(build.accounting.shortfall.sum()),
        "cell_months_underfilled": int((build.accounting.shortfall > 0).sum()),
        "query_digest": stable_hash(query_records),
        "candidate_digest": stable_hash(candidate_records),
        "accounting_digest": stable_hash(accounting_records),
        "source_accounting_digest": stable_hash(source_records),
        "benchmark_prefix_digest": stable_hash(benchmark_records),
        "queries_data": query_records,
        "historical_query_retrieval_opened": False,
        "historical_walk_forward_query_outcomes_opened": False,
        "final_period_result_opened": False,
        "production_promotion_authorized": False,
    }
    state["registry_digest"] = stable_hash(state)
    if _contains_forbidden_key(state):
        raise RegistryPublicationError("outcome, setup or match data entered registry state")

    output = output.resolve()
    if output.exists() or output.is_symlink():
        raise RegistryPublicationError("registry output root already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        build.candidates.to_parquet(temporary / "candidate-cells.parquet", index=False)
        queries.to_parquet(temporary / "query-registry.parquet", index=False)
        build.accounting.to_parquet(temporary / "cutoff-accounting.parquet", index=False)
        build.sources.to_parquet(temporary / "source-accounting.parquet", index=False)
        build.benchmark_prefixes.to_parquet(temporary / "benchmark-prefixes.parquet", index=False)
        _atomic(temporary / "walk-forward-query-registry.json", json.dumps(
            state, indent=2, sort_keys=True, allow_nan=False,
        ).encode() + b"\n")
        coverage = {
            "schema_version": "m04r14-t14-10-walk-forward-registry-coverage-v1",
            "month_cutoffs": len(cutoffs), "candidate_rows": len(candidate_records),
            "queries": len(query_records), "warmup_queries": state["warmup_queries"],
            "scored_queries": len(scored), "minimum_scored_queries": registry_rule["minimum_scored_queries"],
            "shortfall_queries": state["shortfall_queries"],
            "cell_months_underfilled": state["cell_months_underfilled"],
            "fold_counts": {str(key): int(value) for key, value in queries.fold_id.value_counts().sort_index().items()},
            "cell_counts": {
                f"{quality}/{liquidity}": int(len(queries[
                    (queries.quality_tier == quality) & (queries.liquidity_stratum == liquidity)
                ]))
                for quality in QUALITY_CELLS for liquidity in LIQUIDITY_CELLS
            },
            "historical_walk_forward_query_outcomes_opened": False,
            "production_promotion_authorized": False,
        }
        coverage["coverage_digest"] = stable_hash(coverage)
        _atomic(temporary / "COVERAGE.json", json.dumps(
            coverage, indent=2, sort_keys=True, allow_nan=False,
        ).encode() + b"\n")
        cells = "".join(
            f"<tr><td>{escape(key)}</td><td>{value:,}</td></tr>"
            for key, value in coverage["cell_counts"].items()
        )
        report = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>T14-10 WF-01 query registry</title><style>body{{font-family:system-ui;max-width:1100px;margin:2rem auto;line-height:1.5}}table{{border-collapse:collapse;width:100%}}th,td{{padding:.5rem;border-bottom:1px solid #ddd;text-align:left}}.pass{{color:#075}}.warn{{background:#fff4dc;padding:.8rem;border-left:5px solid #b9770e}}code{{overflow-wrap:anywhere}}</style></head><body><h1>WF-01 historical query registry: <span class="pass">PASS</span></h1><p>{len(query_records):,} outcome-blind monthly queries were selected from {len(candidate_records):,} causal candidate rows across {len(cutoffs)} benchmark month-ends. {len(scored):,} queries are scored and {state['warmup_queries']:,} seed expanding baselines.</p><p class="warn"><b>No prediction or query future has been opened.</b> This is a frozen research registry over a configured survivor archive, not point-in-time exchange membership and not a production or trading claim.</p><h2>Cell inventory</h2><table>{cells}</table><h2>Accounting</h2><p>Underfilled cell-months: {state['cell_months_underfilled']:,}; retained shortfall: {state['shortfall_queries']:,}. Every one of the {len(denominator):,} locked source symbols has source-integrity accounting.</p><h2>Identity</h2><p>Registry digest: <code>{state['registry_digest']}</code></p></body></html>"""
        _atomic(temporary / "index.html", report.encode())
        files = [{
            "path": path.name, "bytes": path.stat().st_size,
            "sha256": file_fingerprint(path),
        } for path in sorted(temporary.iterdir())]
        elapsed = perf_counter() - started
        seal_state = {
            "schema_version": SEAL_SCHEMA, "status": "sealed", "passed": True,
            "registry_digest": state["registry_digest"], "files": files,
            "manifest_digest": stable_hash(files), "elapsed_seconds": elapsed,
            "processes": process_count,
            "historical_query_retrieval_opened": False,
            "historical_walk_forward_query_outcomes_opened": False,
            "final_period_result_opened": False,
            "production_promotion_authorized": False,
        }
        seal = {**seal_state, "seal_digest": stable_hash(seal_state),
                "created_at": datetime.now(timezone.utc).isoformat()}
        _atomic(temporary / "SEALED.json", json.dumps(
            seal, indent=2, sort_keys=True, allow_nan=False,
        ).encode() + b"\n")
        os.replace(temporary, output)
    except BaseException:
        for path in sorted(temporary.glob("*")):
            path.unlink()
        temporary.rmdir()
        raise
    return {
        "registry_digest": state["registry_digest"], "queries": len(queries),
        "scored_queries": len(scored), "candidate_rows": len(build.candidates),
        "shortfall_queries": state["shortfall_queries"],
        "elapsed_seconds": perf_counter() - started, "output_root": str(output),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--workers", type=int)
    args = parser.parse_args(argv)
    output = args.output_root or args.repository / DEFAULT_OUTPUT
    result = execute(args.repository, args.config.resolve(), output, workers=args.workers)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
