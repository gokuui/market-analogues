"""Create the M2 outcome-blind prospective-source availability preflight."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile
from typing import Any, Mapping, Sequence

import pandas as pd

from market_analogues.untouched_availability import (
    AvailabilityError,
    completed_month_end_sessions,
    DateMetadata,
    maturity_date,
    parquet_dates,
    parquet_date_metadata,
    scan_date_metadata,
    source_counts,
)


SCHEMA = "m04r15-m2-availability-preflight-v1"
CONTRACT = Path("config/analogue-untouched-availability-contract-v1.json")
M1_RESULT = Path("config/data/analogues/m04r15/m1-candidate-development-gate-v1/FROZEN.json")
M1_VERIFICATION = Path(
    "config/data/analogues/m04r15/m1-candidate-development-gate-v1-verification/VERIFIED.json"
)
OUTPUT = Path("config/data/analogues/m04r15/m2-availability-preflight-v1/RESULT.json")
SNAPSHOT = Path(
    "config/data/analogues/m04r15/m2-availability-preflight-v1/availability-metadata.json"
)
RUNTIME = (
    "config/analogue-untouched-availability-contract-v1.json",
    "src/market_analogues/untouched_availability.py",
    "experiments/m04r/m04r15_m2_availability_preflight.py",
    "tests/test_untouched_availability.py",
    "tests/test_m04r15_m2_availability_preflight.py",
)


class PreflightError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PreflightError(message)


def stable(value: Any) -> str:
    return sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode()).hexdigest()


def snapshot(path: Path) -> bytes:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as error:
        raise PreflightError(f"unsafe input: {path}") from error
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode), f"regular input required: {path}")
        blocks: list[bytes] = []
        while block := os.read(descriptor, 1 << 20):
            blocks.append(block)
        after = os.fstat(descriptor)
        identity = lambda value: (
            value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns,
            value.st_ctime_ns, value.st_mode,
        )
        require(identity(before) == identity(after), f"input changed during read: {path}")
        return b"".join(blocks)
    finally:
        os.close(descriptor)


def decode(raw: bytes, path: Path) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in items:
            require(key not in value, f"duplicate JSON key: {path}/{key}")
            value[key] = item
        return value
    try:
        value = json.loads(
            raw, object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                PreflightError(f"nonfinite JSON: {path}/{token}"),
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PreflightError(f"invalid JSON: {path}") from error
    require(type(value) is dict, f"JSON object required: {path}")
    return value


def git(repository: Path, *arguments: str, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        ("git", *arguments), cwd=repository, capture_output=True,
        text=not binary, check=False,
    )
    require(result.returncode == 0, f"git {' '.join(arguments)} failed")
    return result.stdout if binary else result.stdout.strip()


def semantic_state(
    contract: Mapping[str, Any], m1_result: Mapping[str, Any],
    m1_verification: Mapping[str, Any], stocks: Sequence[DateMetadata],
    benchmark_sessions: pd.DatetimeIndex, implementation_commit: str,
    runtime_sha256: Mapping[str, str], preflight_time: pd.Timestamp,
) -> dict[str, Any]:
    dataset = contract["dataset"]
    schedule = contract["prospective_schedule"]
    upstream = contract["upstream"]
    required_columns = set(dataset["required_stock_columns"])
    require(stocks, "stock source is empty")
    require(all(required_columns <= set(value.schema_names) for value in stocks),
            "stock schema closure differs")
    require(benchmark_sessions.is_monotonic_increasing
            and not benchmark_sessions.has_duplicates,
            "benchmark calendar differs")

    consumed = pd.Timestamp(upstream["consumed_query_boundary"])
    post_consumed = benchmark_sessions[benchmark_sessions > consumed]
    purge_sessions = int(schedule["initial_consumed_boundary_purge_sessions"])
    purge_completion = (
        pd.Timestamp(post_consumed[purge_sessions - 1])
        if len(post_consumed) >= purge_sessions else None
    )
    freeze = pd.Timestamp(m1_result["created_at"]).tz_localize(None)
    not_before = max(
        freeze.normalize(),
        purge_completion if purge_completion is not None else freeze.normalize(),
    )
    observed_cutoffs = completed_month_end_sessions(
        benchmark_sessions, after=not_before, as_of=preflight_time,
    )
    evaluation_months = int(schedule["evaluation_months"])
    observed_cutoffs = observed_cutoffs[:evaluation_months]
    cutoff_dates = [value.date().isoformat() for value in observed_cutoffs]
    counts = source_counts(
        stocks, cutoff_dates,
        minimum_rows=int(dataset["minimum_history_rows_at_cutoff"]),
    )
    minimum_stocks = int(dataset["minimum_eligible_stock_files_each_cutoff"])
    final_cutoff = observed_cutoffs[-1] if len(observed_cutoffs) == evaluation_months else None
    final_maturity = (
        maturity_date(
            benchmark_sessions, final_cutoff,
            int(schedule["maximum_disclosed_horizon_sessions"]),
        ) if final_cutoff is not None else None
    )
    cutoffs_ready = len(observed_cutoffs) == evaluation_months
    stocks_ready = cutoffs_ready and all(value >= minimum_stocks for value in counts.values())
    maturity_ready = final_maturity is not None
    maturity_stock_count = (
        source_counts(stocks, [final_maturity], minimum_rows=int(
            dataset["minimum_history_rows_at_cutoff"],
        ))[final_maturity]
        if final_maturity is not None else 0
    )
    maturity_stocks_ready = maturity_ready and maturity_stock_count >= minimum_stocks
    readiness = cutoffs_ready and stocks_ready and maturity_ready and maturity_stocks_ready

    metadata_rows = [{
        "path": value.path, "bytes": value.bytes, "rows": value.rows,
        "row_groups": value.row_groups, "first_date": value.first_date,
        "last_date": value.last_date, "schema_names": list(value.schema_names),
    } for value in stocks]
    latest_stock = max(value.last_date for value in stocks)
    target_months = [str(value) for value in pd.period_range(
        not_before.to_period("M"), periods=evaluation_months, freq="M",
    )]
    reasons = []
    if not cutoffs_ready:
        reasons.append("fewer_than_twelve_post_freeze_benchmark_month_end_cutoffs")
    if not stocks_ready:
        reasons.append("stock_source_does_not_cover_every_required_cutoff")
    if not maturity_ready:
        reasons.append("final_cutoff_lacks_sixty_subsequent_benchmark_sessions")
    if not maturity_stocks_ready:
        reasons.append("stock_source_does_not_extend_through_final_maturity")

    return {
        "schema_version": SCHEMA,
        "status": "ready_for_registry" if readiness else "source_extension_required",
        "preflight_executed": True,
        "readiness_passed": readiness,
        "registry_creation_authorized": readiness,
        "implementation_commit": implementation_commit,
        "runtime_sha256": dict(runtime_sha256),
        "contract_digest": contract["contract_digest"],
        "m1_result_digest": m1_result["result_digest"],
        "m1_verification_digest": m1_verification["verification_digest"],
        "source_values_opened": False,
        "availability_access": [
            "stock_parquet_schema_row_count_and_date_footer_statistics_only",
            "benchmark_date_column_only",
        ],
        "source_metadata": {
            "stock_files": len(stocks),
            "stock_metadata_digest": stable(metadata_rows),
            "stock_first_date": min(value.first_date for value in stocks),
            "stock_last_date": latest_stock,
            "benchmark_sessions": len(benchmark_sessions),
            "benchmark_first_date": benchmark_sessions[0].date().isoformat(),
            "benchmark_last_date": benchmark_sessions[-1].date().isoformat(),
            "benchmark_date_digest": stable([
                value.date().isoformat() for value in benchmark_sessions
            ]),
        },
        "schedule": {
            "consumed_query_boundary": consumed.date().isoformat(),
            "candidate_freeze_timestamp": freeze.isoformat(),
            "preflight_timestamp": preflight_time.isoformat(),
            "purge_sessions": purge_sessions,
            "purge_completion_session": (
                purge_completion.date().isoformat()
                if purge_completion is not None else None
            ),
            "not_before_date": not_before.date().isoformat(),
            "target_calendar_months": target_months,
            "required_evaluation_months": evaluation_months,
            "observed_month_end_cutoffs": cutoff_dates,
            "observed_cutoff_count": len(cutoff_dates),
            "eligible_stock_files_by_observed_cutoff": counts,
            "minimum_eligible_stock_files_each_cutoff": minimum_stocks,
            "final_cutoff": final_cutoff.date().isoformat() if final_cutoff is not None else None,
            "final_sixty_session_maturity": final_maturity,
            "eligible_stock_files_at_final_maturity": maturity_stock_count,
        },
        "blocking_reasons": reasons,
        "post_freeze_outcomes_opened": False,
        "predictive_claim_authorized": False,
        "production_promotion_authorized": False,
        "trading_claim_authorized": False,
    }


def publish(path: Path, value: Mapping[str, Any]) -> None:
    require(not path.exists() and not path.is_symlink(), "create-only preflight exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".m2-preflight-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write((json.dumps(
                value, indent=2, sort_keys=True, allow_nan=False,
            ) + "\n").encode())
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def encoded(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def publish_restart_safe(path: Path, value: Mapping[str, Any]) -> bytes:
    """Create a prerequisite once, or accept an identical interrupted-run file."""
    content = encoded(value)
    if path.exists() or path.is_symlink():
        require(not path.is_symlink() and snapshot(path) == content,
                "existing prerequisite snapshot differs")
        return content
    publish(path, value)
    require(snapshot(path) == content, "prerequisite snapshot publication differs")
    return content


def run(repository: Path, workers: int) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    require(not (repository / OUTPUT).exists() and not (repository / OUTPUT).is_symlink(),
            "create-only preflight result exists")
    require(not str(git(repository, "status", "--porcelain", "--untracked-files=all")),
            "clean committed tree required")
    contract_raw = snapshot(repository / CONTRACT)
    m1_raw = snapshot(repository / M1_RESULT)
    verification_raw = snapshot(repository / M1_VERIFICATION)
    contract = decode(contract_raw, repository / CONTRACT)
    m1_result = decode(m1_raw, repository / M1_RESULT)
    m1_verification = decode(verification_raw, repository / M1_VERIFICATION)
    contract_state = {key: value for key, value in contract.items() if key != "contract_digest"}
    require(contract["contract_digest"] == stable(contract_state), "contract seal differs")
    require(m1_result["result_digest"] == contract["upstream"]["m1_producer_result_digest"]
            and m1_verification["verification_digest"]
            == contract["upstream"]["m1_verification_digest"], "M1 identity differs")
    require(m1_verification["predictive_claim_authorized"] is False,
            "M1 claim boundary differs")
    require(m1_result["result_digest"] == stable({
        key: value for key, value in m1_result.items()
        if key not in {"result_digest", "created_at"}
    }), "M1 result seal differs")
    require(m1_verification["verification_digest"] == stable({
        key: value for key, value in m1_verification.items()
        if key not in {"verification_digest", "created_at"}
    }), "M1 verification seal differs")
    require(m1_result["candidate_contract_digest"]
            == contract["upstream"]["m1_candidate_contract_digest"],
            "M1 candidate contract differs")
    require(contract["availability_access"]["source_values_opened_by_preflight"] is False
            and not any(contract["claims"].values()), "contract claim boundary differs")

    commit = str(git(repository, "rev-parse", "HEAD"))
    runtime_content = {name: snapshot(repository / name) for name in RUNTIME}
    for name, content in runtime_content.items():
        require(content == git(repository, "show", f"{commit}:{name}", binary=True),
                f"runtime not committed: {name}")
    hashes = {name: sha256(content).hexdigest() for name, content in runtime_content.items()}
    dataset = contract["dataset"]
    stock_root = Path(dataset["stock_root"])
    before_directory = stock_root.stat()
    stock_paths = sorted(stock_root.glob(dataset["stock_glob"]), key=lambda path: path.name)
    stocks = scan_date_metadata(
        stock_paths,
        timestamp_column=dataset["timestamp_column"], workers=workers,
    )
    after_directory = stock_root.stat()
    require((before_directory.st_dev, before_directory.st_ino, before_directory.st_mtime_ns,
             before_directory.st_ctime_ns)
            == (after_directory.st_dev, after_directory.st_ino, after_directory.st_mtime_ns,
                after_directory.st_ctime_ns)
            and stock_paths == sorted(stock_root.glob(dataset["stock_glob"]),
                                      key=lambda path: path.name),
            "stock source membership changed during scan")
    benchmark_metadata = parquet_date_metadata(
        Path(dataset["benchmark_path"]), dataset["timestamp_column"],
    )
    require(set(dataset["required_stock_columns"]) <= set(benchmark_metadata.schema_names),
            "benchmark schema closure differs")
    benchmark = parquet_dates(
        Path(dataset["benchmark_path"]), dataset["timestamp_column"],
    )
    metadata_rows = [{
        "path": value.path, "bytes": value.bytes, "rows": value.rows,
        "row_groups": value.row_groups, "first_date": value.first_date,
        "last_date": value.last_date, "schema_names": list(value.schema_names),
    } for value in stocks]
    snapshot_state = {
        "schema_version": "m04r15-m2-availability-metadata-v1",
        "stock_metadata": metadata_rows,
        "benchmark_dates": [value.date().isoformat() for value in benchmark],
        "source_values_opened": False,
    }
    snapshot_document = {
        **snapshot_state, "snapshot_digest": stable(snapshot_state),
    }
    snapshot_content = publish_restart_safe(repository / SNAPSHOT, snapshot_document)
    created_at = datetime.now(timezone.utc).isoformat()
    state = semantic_state(
        contract, m1_result, m1_verification, stocks, benchmark, commit, hashes,
        pd.Timestamp(created_at),
    )
    state["source_metadata"]["benchmark_metadata"] = {
        "bytes": benchmark_metadata.bytes,
        "rows": benchmark_metadata.rows,
        "row_groups": benchmark_metadata.row_groups,
        "schema_names": list(benchmark_metadata.schema_names),
    }
    state["input_sha256"] = {
        "contract": sha256(contract_raw).hexdigest(),
        "m1_result": sha256(m1_raw).hexdigest(),
        "m1_verification": sha256(verification_raw).hexdigest(),
    }
    state["availability_snapshot"] = {
        "path": SNAPSHOT.name,
        "bytes": len(snapshot_content),
        "sha256": sha256(snapshot_content).hexdigest(),
        "snapshot_digest": snapshot_document["snapshot_digest"],
    }
    result = {
        **state,
        "result_digest": stable(state),
        "created_at": created_at,
    }
    publish(repository / OUTPUT, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args(argv)
    try:
        result = run(args.repository, args.workers)
    except (AvailabilityError, PreflightError) as error:
        print(f"M2 availability preflight refused: {error}", file=os.sys.stderr)
        return 2
    print(json.dumps({
        "readiness_passed": result["readiness_passed"],
        "status": result["status"],
        "result_digest": result["result_digest"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
