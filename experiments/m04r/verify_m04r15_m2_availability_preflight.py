"""Independent verifier for the M2 outcome-blind availability preflight."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
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
import pyarrow.parquet as pq


SCHEMA = "m04r15-m2-availability-preflight-verification-v1"
RESULT = Path("config/data/analogues/m04r15/m2-availability-preflight-v1/RESULT.json")
METADATA = Path(
    "config/data/analogues/m04r15/m2-availability-preflight-v1/availability-metadata.json"
)
OUTPUT = Path(
    "config/data/analogues/m04r15/m2-availability-preflight-v1-verification/VERIFIED.json"
)
CONTRACT = Path("config/analogue-untouched-availability-contract-v1.json")
M1_RESULT = Path("config/data/analogues/m04r15/m1-candidate-development-gate-v1/FROZEN.json")
M1_VERIFICATION = Path(
    "config/data/analogues/m04r15/m1-candidate-development-gate-v1-verification/VERIFIED.json"
)
EXPECTED_CONTRACT = "129ddd6a8bf25e0e581766be1aabc0bd5ada72dc0d51dc38aea8e88f3bda96f7"
EXPECTED_M1_RESULT = "5a1839447a767048b6554758d9250bb0804d8d6c7523bec9158a9f6d85aba469"
EXPECTED_M1_VERIFICATION = "e1d2e3dd5fdcc2369650db598b3f3805cf1f89f5ea0943d5ac5f3434779dd1ec"
EXPECTED_RESULT = "4bd14ecd7c2b580bc48f3d63d9b4459a01744013b1ba611b797f1cae2dc44519"
EXPECTED_RESULT_SHA256 = "a082ce5f43677fdf86a7920175725323450b1c6af8f1da7b2633433e0edb086a"
EXPECTED_METADATA_SHA256 = "470c1dedcaf1f40944f29ca94af0d90301ffe0f67a0a8f1293d093210f2605f0"
EXPECTED_SNAPSHOT = "a32ebac3216972c091cf0dcad28920ca1d692992a206fa2775b34b52440ec6c8"
PRODUCER_RUNTIME = (
    "config/analogue-untouched-availability-contract-v1.json",
    "src/market_analogues/untouched_availability.py",
    "experiments/m04r/m04r15_m2_availability_preflight.py",
    "tests/test_untouched_availability.py",
    "tests/test_m04r15_m2_availability_preflight.py",
)
VERIFIER_RUNTIME = (
    "experiments/m04r/verify_m04r15_m2_availability_preflight.py",
    "tests/test_m04r15_m2_availability_verifier.py",
)
RESULT_KEYS = {
    "availability_access", "availability_snapshot", "blocking_reasons",
    "contract_digest", "created_at", "implementation_commit", "input_sha256",
    "m1_result_digest", "m1_verification_digest", "post_freeze_outcomes_opened",
    "predictive_claim_authorized", "preflight_executed",
    "production_promotion_authorized", "readiness_passed",
    "registry_creation_authorized", "result_digest", "runtime_sha256", "schedule",
    "schema_version", "source_metadata", "source_values_opened", "status",
    "trading_claim_authorized",
}
VERIFICATION_KEYS = {
    "availability_snapshot_digest", "availability_snapshot_sha256", "created_at",
    "live_source_matched_at_verification", "live_stock_files", "passed",
    "post_freeze_outcomes_opened", "predictive_claim_authorized",
    "producer_implementation_commit", "producer_result_digest",
    "producer_result_sha256", "production_promotion_authorized",
    "readiness_passed", "reconstruction_digest", "registry_creation_authorized",
    "schema_version", "source_values_opened", "status", "trading_claim_authorized",
    "verification_digest", "verifier_commit", "verifier_runtime_sha256",
}


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def stable(value: Any) -> str:
    return sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode()).hexdigest()


def snapshot(path: Path) -> bytes:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as error:
        raise VerificationError(f"unsafe input: {path}") from error
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
        require(identity(before) == identity(after), f"input changed: {path}")
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
                VerificationError(f"nonfinite JSON: {path}/{token}"),
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise VerificationError(f"invalid JSON: {path}") from error
    require(type(value) is dict, f"JSON object required: {path}")
    return value


def git(repository: Path, *arguments: str, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        ("git", *arguments), cwd=repository, capture_output=True,
        text=not binary, check=False,
    )
    require(result.returncode == 0, f"git {' '.join(arguments)} failed")
    return result.stdout if binary else result.stdout.strip()


def independent_reconstruction(
    result: Mapping[str, Any], document: Mapping[str, Any],
    contract: Mapping[str, Any], m1: Mapping[str, Any],
) -> dict[str, Any]:
    rows = document["stock_metadata"]
    dates = pd.DatetimeIndex(pd.to_datetime(document["benchmark_dates"]))
    require(type(rows) is list and rows, "metadata stock inventory differs")
    require(not dates.empty and dates.is_monotonic_increasing
            and not dates.has_duplicates, "metadata benchmark calendar differs")
    required_columns = set(contract["dataset"]["required_stock_columns"])
    require(all(required_columns <= set(row["schema_names"]) for row in rows),
            "metadata stock schema differs")
    consumed = pd.Timestamp(contract["upstream"]["consumed_query_boundary"])
    post = dates[dates > consumed]
    purge = int(contract["prospective_schedule"]["initial_consumed_boundary_purge_sessions"])
    purge_completion = pd.Timestamp(post[purge - 1]) if len(post) >= purge else None
    freeze = pd.Timestamp(m1["created_at"]).tz_localize(None)
    not_before = max(
        freeze.normalize(),
        purge_completion if purge_completion is not None else freeze.normalize(),
    )
    after = dates[dates > not_before]
    frame = pd.Series(after, index=after.to_period("M"))
    ends = [pd.Timestamp(value) for value in frame.groupby(level=0).max()] if len(after) else []
    as_of = pd.Timestamp(result["created_at"]).tz_localize(None)
    ends = [value for value in ends if value.to_period("M").end_time < as_of]
    months = int(contract["prospective_schedule"]["evaluation_months"])
    ends = ends[:months]
    cutoff_dates = [value.date().isoformat() for value in ends]
    minimum_rows = int(contract["dataset"]["minimum_history_rows_at_cutoff"])
    minimum_stocks = int(contract["dataset"]["minimum_eligible_stock_files_each_cutoff"])

    def count(cutoff: str) -> int:
        return sum(
            int(row["rows"]) >= minimum_rows and row["last_date"] >= cutoff
            for row in rows
        )

    counts = {cutoff: count(cutoff) for cutoff in cutoff_dates}
    cutoffs_ready = len(ends) == months
    stocks_ready = cutoffs_ready and all(value >= minimum_stocks for value in counts.values())
    final_cutoff = ends[-1] if cutoffs_ready else None
    future = dates[dates > final_cutoff] if final_cutoff is not None else dates[:0]
    horizon = int(contract["prospective_schedule"]["maximum_disclosed_horizon_sessions"])
    maturity = pd.Timestamp(future[horizon - 1]) if len(future) >= horizon else None
    maturity_count = count(maturity.date().isoformat()) if maturity is not None else 0
    maturity_stocks_ready = maturity is not None and maturity_count >= minimum_stocks
    ready = cutoffs_ready and stocks_ready and maturity is not None and maturity_stocks_ready
    reasons = []
    if not cutoffs_ready:
        reasons.append("fewer_than_twelve_post_freeze_benchmark_month_end_cutoffs")
    if not stocks_ready:
        reasons.append("stock_source_does_not_cover_every_required_cutoff")
    if maturity is None:
        reasons.append("final_cutoff_lacks_sixty_subsequent_benchmark_sessions")
    if not maturity_stocks_ready:
        reasons.append("stock_source_does_not_extend_through_final_maturity")
    target_months = [str(value) for value in pd.period_range(
        not_before.to_period("M"), periods=months, freq="M",
    )]
    return {
        "status": "ready_for_registry" if ready else "source_extension_required",
        "readiness_passed": ready,
        "registry_creation_authorized": ready,
        "source_metadata": {
            "stock_files": len(rows),
            "stock_metadata_digest": stable(rows),
            "stock_first_date": min(row["first_date"] for row in rows),
            "stock_last_date": max(row["last_date"] for row in rows),
            "benchmark_sessions": len(dates),
            "benchmark_first_date": dates[0].date().isoformat(),
            "benchmark_last_date": dates[-1].date().isoformat(),
            "benchmark_date_digest": stable(document["benchmark_dates"]),
        },
        "schedule": {
            "consumed_query_boundary": consumed.date().isoformat(),
            "candidate_freeze_timestamp": freeze.isoformat(),
            "preflight_timestamp": result["created_at"],
            "purge_sessions": purge,
            "purge_completion_session": (
                purge_completion.date().isoformat() if purge_completion is not None else None
            ),
            "not_before_date": not_before.date().isoformat(),
            "target_calendar_months": target_months,
            "required_evaluation_months": months,
            "observed_month_end_cutoffs": cutoff_dates,
            "observed_cutoff_count": len(cutoff_dates),
            "eligible_stock_files_by_observed_cutoff": counts,
            "minimum_eligible_stock_files_each_cutoff": minimum_stocks,
            "final_cutoff": final_cutoff.date().isoformat() if final_cutoff is not None else None,
            "final_sixty_session_maturity": (
                maturity.date().isoformat() if maturity is not None else None
            ),
            "eligible_stock_files_at_final_maturity": maturity_count,
        },
        "blocking_reasons": reasons,
    }


def verify_result(repository: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_result = snapshot(repository / RESULT)
    raw_metadata = snapshot(repository / METADATA)
    raw_contract = snapshot(repository / CONTRACT)
    raw_m1 = snapshot(repository / M1_RESULT)
    raw_m1_verification = snapshot(repository / M1_VERIFICATION)
    result = decode(raw_result, repository / RESULT)
    document = decode(raw_metadata, repository / METADATA)
    contract = decode(raw_contract, repository / CONTRACT)
    m1 = decode(raw_m1, repository / M1_RESULT)
    m1_verification = decode(raw_m1_verification, repository / M1_VERIFICATION)
    require(set(result) == RESULT_KEYS, "producer result field closure differs")
    result_state = {key: value for key, value in result.items()
                    if key not in {"result_digest", "created_at"}}
    require(result["result_digest"] == stable(result_state), "producer result seal differs")
    for claim in ("predictive_claim_authorized", "production_promotion_authorized",
                  "trading_claim_authorized"):
        require(result[claim] is False, f"claim boundary differs: {claim}")
    require(result["result_digest"] == EXPECTED_RESULT
            and sha256(raw_result).hexdigest() == EXPECTED_RESULT_SHA256,
            "producer result identity differs")
    contract_state = {key: value for key, value in contract.items() if key != "contract_digest"}
    require(contract["contract_digest"] == stable(contract_state) == EXPECTED_CONTRACT,
            "contract identity differs")
    require(m1["result_digest"] == EXPECTED_M1_RESULT
            and m1["result_digest"] == stable({
                key: value for key, value in m1.items()
                if key not in {"result_digest", "created_at"}
            }), "M1 producer identity differs")
    require(m1_verification["verification_digest"] == EXPECTED_M1_VERIFICATION
            and m1_verification["verification_digest"] == stable({
                key: value for key, value in m1_verification.items()
                if key not in {"verification_digest", "created_at"}
            }), "M1 verification identity differs")
    document_state = {key: value for key, value in document.items() if key != "snapshot_digest"}
    require(set(document) == {
        "schema_version", "stock_metadata", "benchmark_dates", "source_values_opened",
        "snapshot_digest",
    } and document["snapshot_digest"] == stable(document_state) == EXPECTED_SNAPSHOT
            and sha256(raw_metadata).hexdigest() == EXPECTED_METADATA_SHA256
            and document["source_values_opened"] is False, "metadata snapshot differs")
    manifest = result["availability_snapshot"]
    require(manifest == {
        "path": METADATA.name, "bytes": len(raw_metadata),
        "sha256": sha256(raw_metadata).hexdigest(),
        "snapshot_digest": document["snapshot_digest"],
    }, "metadata manifest differs")
    require(result["input_sha256"] == {
        "contract": sha256(raw_contract).hexdigest(),
        "m1_result": sha256(raw_m1).hexdigest(),
        "m1_verification": sha256(raw_m1_verification).hexdigest(),
    }, "producer input manifest differs")
    implementation = result["implementation_commit"]
    head = str(git(repository, "rev-parse", "HEAD"))
    require(type(implementation) is str and subprocess.run(
        ("git", "merge-base", "--is-ancestor", implementation, head),
        cwd=repository,
    ).returncode == 0, "producer lineage differs")
    require(set(result["runtime_sha256"]) == set(PRODUCER_RUNTIME),
            "producer runtime closure differs")
    for name in PRODUCER_RUNTIME:
        content = snapshot(repository / name)
        require(content == git(repository, "show", f"{implementation}:{name}", binary=True)
                and sha256(content).hexdigest() == result["runtime_sha256"][name],
                f"producer runtime differs: {name}")
    rebuilt = independent_reconstruction(result, document, contract, m1)
    require({
        key: value for key, value in result["source_metadata"].items()
        if key != "benchmark_metadata"
    } == rebuilt["source_metadata"], "source summary reconstruction differs")
    for field in ("status", "readiness_passed", "registry_creation_authorized",
                  "schedule", "blocking_reasons"):
        require(result[field] == rebuilt[field], f"readiness reconstruction differs: {field}")
    require(result["preflight_executed"] is True
            and result["source_values_opened"] is False
            and result["post_freeze_outcomes_opened"] is False
            and result["availability_access"] == [
                "stock_parquet_schema_row_count_and_date_footer_statistics_only",
                "benchmark_date_column_only",
            ], "access boundary differs")
    for claim in ("predictive_claim_authorized", "production_promotion_authorized",
                  "trading_claim_authorized"):
        require(result[claim] is False, f"claim boundary differs: {claim}")
    return result, {
        "result_sha256": sha256(raw_result).hexdigest(),
        "metadata_sha256": sha256(raw_metadata).hexdigest(),
        "metadata": document,
        "reconstruction": rebuilt,
        "contract": contract,
    }


def _footer(path: Path, timestamp: str) -> tuple[dict[str, Any], list[str] | None]:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as error:
        raise VerificationError(f"unsafe live source: {path}") from error
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode), f"regular live source required: {path}")
        with os.fdopen(os.dup(descriptor), "rb") as handle:
            parquet = pq.ParquetFile(handle)
            names = tuple(parquet.schema_arrow.names)
            require(timestamp in names, f"live timestamp absent: {path}")
            index = names.index(timestamp)
            minima = []
            maxima = []
            for group in range(parquet.metadata.num_row_groups):
                statistics = parquet.metadata.row_group(group).column(index).statistics
                require(statistics is not None and statistics.has_min_max,
                        f"live date statistics absent: {path}")
                minima.append(pd.Timestamp(statistics.min))
                maxima.append(pd.Timestamp(statistics.max))
            metadata = {
                "path": path.name, "bytes": before.st_size,
                "rows": parquet.metadata.num_rows,
                "row_groups": parquet.metadata.num_row_groups,
                "first_date": min(minima).date().isoformat(),
                "last_date": max(maxima).date().isoformat(),
                "schema_names": list(names),
            }
        dates = None
        if path.name == "IXIC.parquet":
            with os.fdopen(os.dup(descriptor), "rb") as handle:
                values = pq.read_table(handle, columns=[timestamp]).column(timestamp).to_pandas()
            dates = [pd.Timestamp(value).date().isoformat() for value in values]
        after = os.fstat(descriptor)
        identity = lambda value: (
            value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns,
            value.st_ctime_ns, value.st_mode,
        )
        require(identity(before) == identity(after), f"live source changed: {path}")
        return metadata, dates
    finally:
        os.close(descriptor)


def verify_live_source(evidence: Mapping[str, Any], workers: int = 12) -> dict[str, Any]:
    contract = evidence["contract"]
    dataset = contract["dataset"]
    root = Path(dataset["stock_root"])
    before = root.stat()
    paths = sorted(root.glob(dataset["stock_glob"]), key=lambda path: path.name)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        rows = [value[0] for value in executor.map(
            lambda path: _footer(path, dataset["timestamp_column"]), paths,
        )]
    after = root.stat()
    require((before.st_dev, before.st_ino, before.st_mtime_ns, before.st_ctime_ns)
            == (after.st_dev, after.st_ino, after.st_mtime_ns, after.st_ctime_ns)
            and paths == sorted(root.glob(dataset["stock_glob"]), key=lambda path: path.name),
            "live source membership changed")
    benchmark, dates = _footer(
        Path(dataset["benchmark_path"]), dataset["timestamp_column"],
    )
    require(rows == evidence["metadata"]["stock_metadata"]
            and dates == evidence["metadata"]["benchmark_dates"],
            "live source differs from archived availability snapshot")
    required = set(dataset["required_stock_columns"])
    require(required <= set(benchmark["schema_names"]), "live benchmark schema differs")
    return {"stock_files": len(rows), "benchmark_metadata": {
        key: benchmark[key] for key in ("bytes", "rows", "row_groups", "schema_names")
    }}


def verification_state(
    result: Mapping[str, Any], evidence: Mapping[str, Any], live: Mapping[str, Any],
    commit: str, hashes: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA,
        "status": "blocked_preflight_independently_verified",
        "passed": True,
        "verifier_commit": commit,
        "verifier_runtime_sha256": dict(hashes),
        "producer_implementation_commit": result["implementation_commit"],
        "producer_result_digest": result["result_digest"],
        "producer_result_sha256": evidence["result_sha256"],
        "availability_snapshot_digest": evidence["metadata"]["snapshot_digest"],
        "availability_snapshot_sha256": evidence["metadata_sha256"],
        "reconstruction_digest": stable(evidence["reconstruction"]),
        "live_source_matched_at_verification": True,
        "live_stock_files": live["stock_files"],
        "readiness_passed": False,
        "registry_creation_authorized": False,
        "source_values_opened": False,
        "post_freeze_outcomes_opened": False,
        "predictive_claim_authorized": False,
        "production_promotion_authorized": False,
        "trading_claim_authorized": False,
    }


def publish(path: Path, value: Mapping[str, Any]) -> None:
    require(not path.exists() and not path.is_symlink(), "create-only verification exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".m2-verify-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write((json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())
            handle.flush(); os.fsync(handle.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(directory)
        finally: os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def run(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    require(not str(git(repository, "status", "--porcelain", "--untracked-files=all")),
            "clean committed tree required")
    result, evidence = verify_result(repository)
    live = verify_live_source(evidence)
    require(live["benchmark_metadata"] == result["source_metadata"]["benchmark_metadata"],
            "live benchmark metadata differs")
    commit = str(git(repository, "rev-parse", "HEAD"))
    content = {name: snapshot(repository / name) for name in VERIFIER_RUNTIME}
    for name, raw in content.items():
        require(raw == git(repository, "show", f"{commit}:{name}", binary=True),
                f"verifier runtime not committed: {name}")
    hashes = {name: sha256(raw).hexdigest() for name, raw in content.items()}
    state = verification_state(result, evidence, live, commit, hashes)
    receipt = {
        **state, "verification_digest": stable(state),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    publish(repository / OUTPUT, receipt)
    return receipt


def validate(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    receipt = decode(snapshot(repository / OUTPUT), repository / OUTPUT)
    require(set(receipt) == VERIFICATION_KEYS, "verification field closure differs")
    state = {key: value for key, value in receipt.items()
             if key not in {"verification_digest", "created_at"}}
    require(receipt["verification_digest"] == stable(state), "verification seal differs")
    commit = receipt["verifier_commit"]
    head = str(git(repository, "rev-parse", "HEAD"))
    require(subprocess.run(("git", "merge-base", "--is-ancestor", commit, head),
                           cwd=repository).returncode == 0, "verifier lineage differs")
    hashes = receipt["verifier_runtime_sha256"]
    require(set(hashes) == set(VERIFIER_RUNTIME), "verifier runtime closure differs")
    for name in VERIFIER_RUNTIME:
        raw = snapshot(repository / name)
        require(raw == git(repository, "show", f"{commit}:{name}", binary=True)
                and sha256(raw).hexdigest() == hashes[name],
                f"verifier runtime differs: {name}")
    result, evidence = verify_result(repository)
    live = {"stock_files": receipt["live_stock_files"]}
    require(state == verification_state(result, evidence, live, commit, hashes),
            "verification reconstruction differs")
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "validate"))
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    try:
        result = run(args.repository) if args.mode == "run" else validate(args.repository)
    except VerificationError as error:
        print(f"M2 availability verification refused: {error}", file=os.sys.stderr)
        return 2
    print(json.dumps({
        "passed": result["passed"],
        "verification_digest": result["verification_digest"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
