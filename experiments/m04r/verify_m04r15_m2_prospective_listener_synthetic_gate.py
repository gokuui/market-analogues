"""Independent oracle for the M2 prospective-listener synthetic gate."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import tempfile
from typing import Any, Mapping, Sequence

import pandas as pd


SCHEMA = "m04r15-m2-prospective-listener-synthetic-verification-v1"
RESULT = Path("config/data/analogues/m04r15/m2-prospective-listener-synthetic-v1/RESULT.json")
OUTPUT = Path(
    "config/data/analogues/m04r15/m2-prospective-listener-synthetic-v1-verification/VERIFIED.json"
)
CONTRACT = Path("config/prospective-prediction-listener-contract-v1.json")
PREFLIGHT = Path("config/data/analogues/m04r15/m2-availability-preflight-v1/RESULT.json")
PREFLIGHT_VERIFICATION = Path(
    "config/data/analogues/m04r15/m2-availability-preflight-v1-verification/VERIFIED.json"
)
EXPECTED_CONTRACT = "9b0c4fd0593be7912cc65b3222c4fe3f51e3d6f12a3873fd0dda2c5fb4750660"
EXPECTED_RESULT = "ca241a07cf1a0f28a2df530938a510378cf3aa20cd2dbb0dbebad59af41d75b9"
EXPECTED_RESULT_SHA256 = "bee83cbaf1042eb0bc14b8461e1b65f10c3957bae23c7fccc4d67d5ed02ed1f7"
PRODUCER_RUNTIME = (
    "config/prospective-prediction-listener-contract-v1.json",
    "src/market_analogues/prospective_batch.py",
    "experiments/m04r/m04r15_m2_prospective_listener_synthetic_gate.py",
    "tests/test_prospective_batch.py",
    "tests/test_m04r15_m2_prospective_listener_synthetic_gate.py",
)
VERIFIER_RUNTIME = (
    "experiments/m04r/verify_m04r15_m2_prospective_listener_synthetic_gate.py",
    "tests/test_m04r15_m2_prospective_listener_synthetic_verifier.py",
)
RESULT_KEYS = {
    "checks", "contract_digest", "created_at", "elapsed_seconds",
    "implementation_commit", "input_sha256", "inventory", "passed",
    "post_freeze_outcomes_opened", "predictive_claim_authorized",
    "production_promotion_authorized", "real_prediction_created",
    "real_registry_created", "real_source_accessed", "result_digest",
    "runtime_sha256", "schema_version", "status",
    "synthetic_gate_is_predictive_validation", "trading_claim_authorized",
}
VERIFICATION_KEYS = {
    "checks", "contract_digest", "created_at", "independent_prediction_seal_digest",
    "passed", "post_freeze_outcomes_opened", "predictive_claim_authorized",
    "producer_implementation_commit", "producer_or_project_scientific_code_imported",
    "producer_result_digest", "producer_result_sha256",
    "production_promotion_authorized", "real_prediction_created",
    "real_registry_created", "real_source_accessed", "reconstruction_digest",
    "schema_version", "status", "trading_claim_authorized", "verification_digest",
    "verifier_commit", "verifier_runtime_sha256",
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


def encode(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


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


def independent_documents() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    batch = "2026-09"; cutoff = "2026-09-30"
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
    } for index in range(24)]
    ids = [row["query_id"] for row in queries]
    registry = {
        "schema_version": "prospective-query-registry-v1", "batch_id": batch,
        "cutoff": cutoff, "queries": queries, "query_digest": stable(ids),
        "selection_used_outcomes": False,
    }
    probabilities = {
        "candidate": [.5, .3, .2], "matched_causal_history": [.4, .4, .2],
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


def independent_prediction_seal(contract_digest: str) -> str:
    source, registry, predictions = independent_documents()
    ids = [row["query_id"] for row in registry["queries"]]
    require(len(ids) == 24 and len(set(ids)) == 24, "independent registry differs")
    for row in predictions["rows"]:
        require(row["query_id"] in ids and set(row["probabilities"]) == {
            "candidate", "matched_causal_history", "locked_composite",
        }, "independent prediction closure differs")
        for values in row["probabilities"].values():
            require(len(values) == 3 and all(math.isfinite(value) and 0 <= value <= 1
                                             for value in values)
                    and abs(sum(values) - 1) <= 1e-12,
                    "independent probability differs")
    documents = {
        "SOURCE_LOCK.json": source,
        "QUERY_REGISTRY.json": registry,
        "PREDICTIONS.json": predictions,
    }
    manifest = [{
        "path": name, "bytes": len(encode(value)),
        "sha256": sha256(encode(value)).hexdigest(),
    } for name, value in documents.items()]
    state = {
        "schema_version": "prospective-prediction-batch-seal-v1",
        "status": "predictions_sealed", "passed": True,
        "contract_digest": contract_digest, "batch_id": "2026-09",
        "cutoff": "2026-09-30", "query_count": 24,
        "query_digest": stable(ids), "file_manifest": manifest,
        "source_values_after_cutoff_opened": False,
        "query_outcomes_opened_before_prediction_seal": False,
        "predictive_claim_authorized": False,
        "production_promotion_authorized": False,
        "trading_claim_authorized": False,
    }
    return stable(state)


def independent_state(contract_digest: str) -> dict[str, Any]:
    sessions = pd.bdate_range("2026-09-01", "2027-02-15")
    freeze = pd.Timestamp("2026-09-11")
    incomplete = sessions[:14]
    after = incomplete[incomplete > freeze]
    require(not any(value.to_period("M").end_time < pd.Timestamp("2026-09-20")
                    for value in after), "incomplete-month oracle differs")
    september = sessions[(sessions > freeze) & (sessions.to_period("M") == "2026-09")]
    require(september[-1] == pd.Timestamp("2026-09-30"), "ready-month oracle differs")
    seal = independent_prediction_seal(contract_digest)
    future = sessions[sessions > pd.Timestamp("2026-09-30")]
    maturity = future[59]
    require(pd.Timestamp("2026-10-01") < maturity < sessions[-1],
            "maturity oracle differs")
    checks = [
        {"name": "incomplete_month_waits", "passed": True,
         "detail": "wait_for_completed_month"},
        {"name": "completed_month_runs", "passed": True,
         "detail": "run_prediction_batch"},
        {"name": "out_of_order_prefix_refused", "passed": True, "detail": "refused"},
        {"name": "prerequisite_crash_resume_exact", "passed": True, "detail": seal},
        {"name": "future_mutation_invariant", "passed": True, "detail": seal},
        {"name": "immature_loader_unreachable", "passed": True, "detail": "calls=0"},
        {"name": "mature_loader_called_once", "passed": True, "detail": "calls=1"},
        {"name": "post_seal_crash_resume_exact", "passed": True, "detail": "preserved"},
    ]
    return {
        "checks": checks,
        "inventory": {
            "synthetic_queries": 24,
            "prediction_methods": [
                "candidate", "matched_causal_history", "locked_composite",
            ],
            "benchmark_sessions": len(sessions),
        },
        "prediction_seal_digest": seal,
    }


def verify_result(repository: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_result = snapshot(repository / RESULT)
    raw_contract = snapshot(repository / CONTRACT)
    raw_preflight = snapshot(repository / PREFLIGHT)
    raw_verification = snapshot(repository / PREFLIGHT_VERIFICATION)
    result = decode(raw_result, repository / RESULT)
    contract = decode(raw_contract, repository / CONTRACT)
    preflight = decode(raw_preflight, repository / PREFLIGHT)
    verification = decode(raw_verification, repository / PREFLIGHT_VERIFICATION)
    require(set(result) == RESULT_KEYS, "result field closure differs")
    state = {key: value for key, value in result.items()
             if key not in {"result_digest", "created_at", "elapsed_seconds"}}
    require(result["result_digest"] == stable(state), "result seal differs")
    for claim in (
        "synthetic_gate_is_predictive_validation", "real_source_accessed",
        "real_registry_created", "real_prediction_created", "post_freeze_outcomes_opened",
        "predictive_claim_authorized", "production_promotion_authorized",
        "trading_claim_authorized",
    ):
        require(result[claim] is False, f"claim boundary differs: {claim}")
    require(result["result_digest"] == EXPECTED_RESULT
            and sha256(raw_result).hexdigest() == EXPECTED_RESULT_SHA256,
            "result identity differs")
    contract_state = {key: value for key, value in contract.items() if key != "contract_digest"}
    require(contract["contract_digest"] == stable(contract_state) == EXPECTED_CONTRACT
            and not any(contract["claims"].values()), "contract identity differs")
    require(preflight["result_digest"] == contract["upstream"]["m2_blocked_result_digest"]
            and verification["verification_digest"]
            == contract["upstream"]["m2_blocked_verification_digest"]
            and preflight["readiness_passed"] is False,
            "blocked M2-00 boundary differs")
    require(result["input_sha256"] == {
        "contract": sha256(raw_contract).hexdigest(),
        "m2_preflight": sha256(raw_preflight).hexdigest(),
        "m2_preflight_verification": sha256(raw_verification).hexdigest(),
    }, "input manifest differs")
    implementation = result["implementation_commit"]
    head = str(git(repository, "rev-parse", "HEAD"))
    require(type(implementation) is str and subprocess.run(
        ("git", "merge-base", "--is-ancestor", implementation, head), cwd=repository,
    ).returncode == 0, "producer lineage differs")
    require(set(result["runtime_sha256"]) == set(PRODUCER_RUNTIME),
            "producer runtime closure differs")
    for name in PRODUCER_RUNTIME:
        raw = snapshot(repository / name)
        require(raw == git(repository, "show", f"{implementation}:{name}", binary=True)
                and sha256(raw).hexdigest() == result["runtime_sha256"][name],
                f"producer runtime differs: {name}")
    rebuilt = independent_state(contract["contract_digest"])
    require(result["checks"] == rebuilt["checks"]
            and result["inventory"] == rebuilt["inventory"],
            "independent synthetic reconstruction differs")
    return result, {
        "result_sha256": sha256(raw_result).hexdigest(),
        "reconstruction": rebuilt,
    }


def verification_state(
    result: Mapping[str, Any], evidence: Mapping[str, Any], commit: str,
    hashes: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA,
        "status": "synthetic_listener_independently_verified",
        "passed": True,
        "verifier_commit": commit,
        "verifier_runtime_sha256": dict(hashes),
        "producer_implementation_commit": result["implementation_commit"],
        "producer_result_digest": result["result_digest"],
        "producer_result_sha256": evidence["result_sha256"],
        "contract_digest": result["contract_digest"],
        "reconstruction_digest": stable(evidence["reconstruction"]),
        "independent_prediction_seal_digest": evidence["reconstruction"][
            "prediction_seal_digest"
        ],
        "checks": len(result["checks"]),
        "producer_or_project_scientific_code_imported": False,
        "real_source_accessed": False,
        "real_registry_created": False,
        "real_prediction_created": False,
        "post_freeze_outcomes_opened": False,
        "predictive_claim_authorized": False,
        "production_promotion_authorized": False,
        "trading_claim_authorized": False,
    }


def publish(path: Path, value: Mapping[str, Any]) -> None:
    require(not path.exists() and not path.is_symlink(), "create-only verification exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".m2-listener-verify-", dir=path.parent)
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
    commit = str(git(repository, "rev-parse", "HEAD"))
    content = {name: snapshot(repository / name) for name in VERIFIER_RUNTIME}
    for name, raw in content.items():
        require(raw == git(repository, "show", f"{commit}:{name}", binary=True),
                f"verifier runtime not committed: {name}")
    hashes = {name: sha256(raw).hexdigest() for name, raw in content.items()}
    state = verification_state(result, evidence, commit, hashes)
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
    require(state == verification_state(result, evidence, commit, hashes),
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
        print(f"M2 listener verification refused: {error}", file=os.sys.stderr)
        return 2
    print(json.dumps({
        "passed": result["passed"], "verification_digest": result["verification_digest"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
