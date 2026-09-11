"""Run the M2 prospective listener/seal/outcome-firewall synthetic gate."""
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
from time import perf_counter
from typing import Any, Mapping, Sequence

import pandas as pd

from market_analogues.prospective_batch import (
    ProspectiveBatchError,
    decide_next_batch,
    guarded_outcome_load,
    seal_prediction_batch,
    stable,
    validate_prediction_batch,
)


SCHEMA = "m04r15-m2-prospective-listener-synthetic-gate-v1"
CONTRACT = Path("config/prospective-prediction-listener-contract-v1.json")
PREFLIGHT = Path("config/data/analogues/m04r15/m2-availability-preflight-v1/RESULT.json")
PREFLIGHT_VERIFICATION = Path(
    "config/data/analogues/m04r15/m2-availability-preflight-v1-verification/VERIFIED.json"
)
OUTPUT = Path(
    "config/data/analogues/m04r15/m2-prospective-listener-synthetic-v1/RESULT.json"
)
RUNTIME = (
    "config/prospective-prediction-listener-contract-v1.json",
    "src/market_analogues/prospective_batch.py",
    "experiments/m04r/m04r15_m2_prospective_listener_synthetic_gate.py",
    "tests/test_prospective_batch.py",
    "tests/test_m04r15_m2_prospective_listener_synthetic_gate.py",
)


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def snapshot(path: Path) -> bytes:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as error:
        raise GateError(f"unsafe input: {path}") from error
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
                GateError(f"nonfinite JSON: {path}/{token}"),
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GateError(f"invalid JSON: {path}") from error
    require(type(value) is dict, f"JSON object required: {path}")
    return value


def git(repository: Path, *arguments: str, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        ("git", *arguments), cwd=repository, capture_output=True,
        text=not binary, check=False,
    )
    require(result.returncode == 0, f"git {' '.join(arguments)} failed")
    return result.stdout if binary else result.stdout.strip()


def documents(count: int = 24) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    batch = "2026-09"
    cutoff = "2026-09-30"
    source = {
        "schema_version": "prospective-source-lock-v1", "batch_id": batch,
        "cutoff": cutoff, "maximum_source_timestamp": cutoff,
        "stock_prefix_manifest_digest": "2" * 64,
        "benchmark_prefix_digest": "3" * 64,
        "source_values_after_cutoff_opened": False,
    }
    queries = [{
        "query_id": f"query-{index:02d}", "symbol": f"S{index:02d}",
        "quality_tier": "A" if index < 12 else "B",
        "liquidity_stratum": ("low", "middle", "high")[index % 3],
        "selection_hash": sha256(f"selection-{index}".encode()).hexdigest(),
        "stock_prefix_digest": sha256(f"prefix-{index}".encode()).hexdigest(),
    } for index in range(count)]
    ids = [row["query_id"] for row in queries]
    registry = {
        "schema_version": "prospective-query-registry-v1", "batch_id": batch,
        "cutoff": cutoff, "queries": queries, "query_digest": stable(ids),
        "selection_used_outcomes": False,
    }
    probabilities = {
        "candidate": [.5, .3, .2],
        "matched_causal_history": [.4, .4, .2],
        "locked_composite": [.6, .2, .2],
    }
    predictions = {
        "schema_version": "prospective-probability-predictions-v1",
        "batch_id": batch, "cutoff": cutoff,
        "rows": [{
            "query_id": query_id, "probabilities": probabilities,
            "provenance_digest": sha256(f"provenance-{query_id}".encode()).hexdigest(),
        } for query_id in ids],
        "query_digest": stable(ids), "query_outcomes_opened": False,
        "source_values_after_cutoff_opened": False,
    }
    return source, registry, predictions


def synthetic_checks(contract_digest: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    record = lambda name, passed, detail: checks.append({
        "name": name, "passed": bool(passed), "detail": detail,
    })
    sessions = pd.bdate_range("2026-09-01", "2027-02-15")
    freeze = pd.Timestamp("2026-09-11")
    waiting = decide_next_batch(
        freeze=freeze, as_of=pd.Timestamp("2026-09-20"),
        benchmark_sessions=sessions[:14], completed=[], eligible_stock_files={},
        minimum_stocks=1000,
    )
    record("incomplete_month_waits", waiting.action == "wait_for_completed_month", waiting.action)
    ready = decide_next_batch(
        freeze=freeze, as_of=pd.Timestamp("2026-10-02"),
        benchmark_sessions=sessions, completed=[], eligible_stock_files={"2026-09-30": 1000},
        minimum_stocks=1000,
    )
    record("completed_month_runs", ready.action == "run_prediction_batch", ready.action)
    try:
        decide_next_batch(
            freeze=freeze, as_of=pd.Timestamp("2026-11-02"),
            benchmark_sessions=sessions,
            completed=[{"batch_id": "2026-10", "cutoff": "2026-10-30"}],
            eligible_stock_files={}, minimum_stocks=1000,
        )
    except ProspectiveBatchError:
        record("out_of_order_prefix_refused", True, "refused")
    else:
        record("out_of_order_prefix_refused", False, "accepted")

    source, registry, predictions = documents()
    with tempfile.TemporaryDirectory(prefix="m2-listener-synthetic-") as name:
        root = Path(name)
        try:
            seal_prediction_batch(
                root, contract_digest=contract_digest, source_lock=source,
                registry=registry, predictions=predictions, created_at="original",
                interrupt_after_prerequisites=True,
            )
        except ProspectiveBatchError:
            pass
        first = seal_prediction_batch(
            root, contract_digest=contract_digest, source_lock=source,
            registry=registry, predictions=predictions, created_at="original",
        )
        resumed = seal_prediction_batch(
            root, contract_digest=contract_digest, source_lock=source,
            registry=registry, predictions=predictions, created_at="ignored",
        )
        record("prerequisite_crash_resume_exact", first == resumed, first["result_digest"])
        future = {row["query_id"]: "favorable_first" for row in registry["queries"]}
        before = validate_prediction_batch(root / "batch-2026-09")["result_digest"]
        future[registry["queries"][0]["query_id"]] = "no_touch"
        after = validate_prediction_batch(root / "batch-2026-09")["result_digest"]
        record("future_mutation_invariant", before == after, before)
        calls: list[list[str]] = []
        loader = lambda ids: calls.append(list(ids)) or {"rows": len(ids)}
        try:
            guarded_outcome_load(
                root / "batch-2026-09", benchmark_sessions=sessions,
                source_as_of=pd.Timestamp("2026-10-01"),
                wall_clock=pd.Timestamp("2027-02-01"), horizon_sessions=60,
                loader=loader,
            )
        except ProspectiveBatchError:
            pass
        record("immature_loader_unreachable", not calls, f"calls={len(calls)}")
        loaded = guarded_outcome_load(
            root / "batch-2026-09", benchmark_sessions=sessions,
            source_as_of=sessions[-1], wall_clock=sessions[-1] + pd.Timedelta(days=1),
            horizon_sessions=60, loader=loader,
        )
        record("mature_loader_called_once", loaded == {"rows": 24} and len(calls) == 1,
               f"calls={len(calls)}")

    with tempfile.TemporaryDirectory(prefix="m2-listener-seal-crash-") as name:
        root = Path(name)
        try:
            seal_prediction_batch(
                root, contract_digest=contract_digest, source_lock=source,
                registry=registry, predictions=predictions, created_at="preserved",
                interrupt_after_seal=True,
            )
        except ProspectiveBatchError:
            pass
        seal = seal_prediction_batch(
            root, contract_digest=contract_digest, source_lock=source,
            registry=registry, predictions=predictions, created_at="new",
        )
        record("post_seal_crash_resume_exact", seal["created_at"] == "preserved",
               seal["created_at"])

    return checks, {
        "synthetic_queries": len(registry["queries"]),
        "prediction_methods": list(predictions["rows"][0]["probabilities"]),
        "benchmark_sessions": len(sessions),
    }


def publish(path: Path, value: Mapping[str, Any]) -> None:
    require(not path.exists() and not path.is_symlink(), "create-only result exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".m2-listener-", dir=path.parent)
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
    raw_contract = snapshot(repository / CONTRACT)
    raw_preflight = snapshot(repository / PREFLIGHT)
    raw_verification = snapshot(repository / PREFLIGHT_VERIFICATION)
    contract = decode(raw_contract, repository / CONTRACT)
    preflight = decode(raw_preflight, repository / PREFLIGHT)
    verification = decode(raw_verification, repository / PREFLIGHT_VERIFICATION)
    contract_state = {key: value for key, value in contract.items() if key != "contract_digest"}
    require(contract["contract_digest"] == stable(contract_state), "contract seal differs")
    require(preflight["result_digest"] == contract["upstream"]["m2_blocked_result_digest"]
            and verification["verification_digest"]
            == contract["upstream"]["m2_blocked_verification_digest"],
            "M2-00 identity differs")
    require(preflight["result_digest"] == stable({
        key: value for key, value in preflight.items()
        if key not in {"result_digest", "created_at"}
    }), "M2-00 result seal differs")
    require(verification["verification_digest"] == stable({
        key: value for key, value in verification.items()
        if key not in {"verification_digest", "created_at"}
    }), "M2-00 verification seal differs")
    require(preflight["readiness_passed"] is False
            and preflight["registry_creation_authorized"] is False
            and verification["source_values_opened"] is False,
            "real launch boundary differs")
    require(contract["prediction_batch"]["queries_per_month"] == 24
            and contract["prediction_batch"]["methods"]
            == ["candidate", "matched_causal_history", "locked_composite"]
            and contract["real_launch"]["currently_authorized"] is False
            and not any(contract["claims"].values()), "listener contract boundary differs")
    commit = str(git(repository, "rev-parse", "HEAD"))
    content = {name: snapshot(repository / name) for name in RUNTIME}
    for name, raw in content.items():
        require(raw == git(repository, "show", f"{commit}:{name}", binary=True),
                f"runtime not committed: {name}")
    hashes = {name: sha256(raw).hexdigest() for name, raw in content.items()}
    started = perf_counter()
    checks, inventory = synthetic_checks(contract["contract_digest"])
    elapsed = perf_counter() - started
    require(all(row["passed"] for row in checks), "synthetic check failed")
    state = {
        "schema_version": SCHEMA,
        "status": "synthetic_listener_and_firewall_verified",
        "passed": True,
        "implementation_commit": commit,
        "runtime_sha256": hashes,
        "contract_digest": contract["contract_digest"],
        "input_sha256": {
            "contract": sha256(raw_contract).hexdigest(),
            "m2_preflight": sha256(raw_preflight).hexdigest(),
            "m2_preflight_verification": sha256(raw_verification).hexdigest(),
        },
        "checks": checks,
        "inventory": inventory,
        "real_source_accessed": False,
        "real_registry_created": False,
        "real_prediction_created": False,
        "post_freeze_outcomes_opened": False,
        "synthetic_gate_is_predictive_validation": False,
        "predictive_claim_authorized": False,
        "production_promotion_authorized": False,
        "trading_claim_authorized": False,
    }
    result = {
        **state, "result_digest": stable(state), "elapsed_seconds": elapsed,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    publish(repository / OUTPUT, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    try:
        result = run(args.repository)
    except (GateError, ProspectiveBatchError) as error:
        print(f"M2 listener synthetic gate refused: {error}", file=os.sys.stderr)
        return 2
    print(json.dumps({
        "passed": result["passed"], "checks": len(result["checks"]),
        "result_digest": result["result_digest"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
