"""Independent verifier for the M2-03 real-source launcher wait gate."""
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


SCHEMA = "m04r15-m2-real-source-launcher-verification-v1"
RESULT = Path("config/data/analogues/m04r15/m2-real-source-launcher-gate-v1/RESULT.json")
OUTPUT = Path(
    "config/data/analogues/m04r15/m2-real-source-launcher-gate-v1-verification/VERIFIED.json"
)
CONTRACT = Path("config/prospective-real-source-launch-contract-v1.json")
AVAILABILITY = Path("config/data/analogues/m04r15/m2-availability-preflight-v1/RESULT.json")
AVAILABILITY_VERIFICATION = Path(
    "config/data/analogues/m04r15/m2-availability-preflight-v1-verification/VERIFIED.json"
)
EXPECTED_CONTRACT = "8ba52c97caf40d1b9d6dbc9d11898acca4d8016efdde9c8e78609f739e27c701"
EXPECTED_AVAILABILITY = "4bd14ecd7c2b580bc48f3d63d9b4459a01744013b1ba611b797f1cae2dc44519"
EXPECTED_AVAILABILITY_VERIFICATION = (
    "6d30a481ff9382f4741352d4adb6504b9617f1bda74b40a38c471d5f871549c7"
)
EXPECTED_RESULT = "87f0989c41abc5fe94e074eb9b87e6047478279eb47ecf4dc174f0ab84734df8"
EXPECTED_RESULT_SHA256 = "7e67fa5b73f69c4bad2d3e875fd0922c843e1547bc748e73b40070548b3ec04b"
PRODUCER_RUNTIME = (
    "config/prospective-real-source-launch-contract-v1.json",
    "src/market_analogues/prospective_launcher.py",
    "experiments/m04r/m04r15_m2_real_source_launcher_gate.py",
    "tests/test_prospective_launcher.py",
    "tests/test_m04r15_m2_real_source_launcher_gate.py",
)
VERIFIER_RUNTIME = (
    "experiments/m04r/verify_m04r15_m2_real_source_launcher_gate.py",
    "tests/test_verify_m04r15_m2_real_source_launcher_gate.py",
)


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
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode), f"regular input required: {path}")
        blocks = []
        while block := os.read(descriptor, 1 << 20):
            blocks.append(block)
        after = os.fstat(descriptor)
        identity = lambda value: (value.st_dev, value.st_ino, value.st_size,
                                  value.st_mtime_ns, value.st_ctime_ns, value.st_mode)
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
                VerificationError(f"nonfinite JSON: {path}/{token}")),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise VerificationError(f"invalid JSON: {path}") from error
    require(type(value) is dict, f"JSON object required: {path}")
    return value


def validate_seal(value: Mapping[str, Any], field: str) -> None:
    state = {key: item for key, item in value.items()
             if key not in {field, "created_at"}}
    require(value[field] == stable(state), f"{field} seal differs")


def git(repository: Path, *arguments: str, binary: bool = False) -> str | bytes:
    result = subprocess.run(("git", *arguments), cwd=repository, capture_output=True,
                            text=not binary, check=False)
    require(result.returncode == 0, f"git {' '.join(arguments)} failed")
    return result.stdout if binary else result.stdout.strip()


def verify_result(repository: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    paths = {"result": RESULT, "contract": CONTRACT, "availability": AVAILABILITY,
             "verification": AVAILABILITY_VERIFICATION}
    raw = {name: snapshot(repository / path) for name, path in paths.items()}
    values = {name: decode(content, repository / paths[name])
              for name, content in raw.items()}
    result, contract = values["result"], values["contract"]
    availability, verification = values["availability"], values["verification"]
    validate_seal(result, "result_digest")
    validate_seal(availability, "result_digest")
    validate_seal(verification, "verification_digest")
    contract_state = {key: item for key, item in contract.items()
                      if key != "contract_digest"}
    require(contract["contract_digest"] == stable(contract_state) == EXPECTED_CONTRACT,
            "contract identity differs")
    require(result["result_digest"] == EXPECTED_RESULT
            and sha256(raw["result"]).hexdigest() == EXPECTED_RESULT_SHA256,
            "producer result identity differs")
    require(availability["result_digest"] == EXPECTED_AVAILABILITY
            and verification["verification_digest"]
            == EXPECTED_AVAILABILITY_VERIFICATION,
            "availability pair identity differs")
    require(verification["producer_result_digest"] == availability["result_digest"]
            and verification["passed"] is True
            and verification["live_source_matched_at_verification"] is True,
            "availability pair was not independently verified")
    require(result["input_sha256"] == {
        "contract": sha256(raw["contract"]).hexdigest(),
        "availability": sha256(raw["availability"]).hexdigest(),
        "verification": sha256(raw["verification"]).hexdigest(),
    }, "producer input bytes differ")
    expected_decision = {
        "schema_version": "prospective-real-source-launch-decision-v1",
        "action": "wait_for_source_refresh",
        "reason": "independently_verified_source_extension_required",
        "availability_result_digest": EXPECTED_AVAILABILITY,
        "availability_verification_digest": EXPECTED_AVAILABILITY_VERIFICATION,
        "registry_callback_calls": 0, "prediction_callback_calls": 0,
        "post_freeze_outcomes_opened": False,
    }
    require(result["status"] == "verified_wait_state"
            and result["decision"] == expected_decision,
            "wait decision reconstruction differs")
    require(all(value is False for value in (
        availability["readiness_passed"], availability["registry_creation_authorized"],
        verification["readiness_passed"], verification["registry_creation_authorized"],
        result["real_registry_created"], result["real_prediction_created"],
        result["canonical_inputs_modified"], result["post_freeze_outcomes_opened"],
        result["predictive_claim_authorized"],
        result["production_promotion_authorized"], result["trading_claim_authorized"],
    )), "blocked boundary differs")
    refresh = contract["refresh"]
    assessment = result["refresh_assessment"]
    minimum = int(refresh["minimum_free_bytes"])
    require(assessment["provider"] == refresh["provider"]
            and assessment["credential_environment_variable"]
            == refresh["credential_environment_variable"]
            and assessment["credential_value_recorded"] is False
            and assessment["observed_free_bytes"] >= minimum
            and assessment["minimum_free_bytes"] == minimum
            and assessment["checks"] == {
                "credential_present": False, "capacity_passed": True,
                "canonical_stock_source_present": True,
                "canonical_benchmark_present": True,
                "non_destructive_staging_required": True,
            }
            and assessment["blocking_reasons"]
            == ["required_provider_credential_absent"]
            and assessment["ready_for_non_destructive_refresh"] is False
            and assessment["canonical_inputs_modified"] is False
            and assessment["fallback_provider_authorized"] is False,
            "refresh assessment reconstruction differs")
    implementation = result["implementation_commit"]
    head = str(git(repository, "rev-parse", "HEAD"))
    require(subprocess.run(("git", "merge-base", "--is-ancestor", implementation, head),
                           cwd=repository).returncode == 0,
            "producer lineage differs")
    require(set(result["runtime_sha256"]) == set(PRODUCER_RUNTIME),
            "producer runtime closure differs")
    for name in PRODUCER_RUNTIME:
        content = snapshot(repository / name)
        require(content == git(repository, "show", f"{implementation}:{name}", binary=True)
                and sha256(content).hexdigest() == result["runtime_sha256"][name],
                f"producer runtime differs: {name}")
    live_capacity = os.statvfs(Path(refresh["source_repository"]))
    live_free = live_capacity.f_bavail * live_capacity.f_frsize
    live = {
        "credential_present": bool(os.environ.get(
            refresh["credential_environment_variable"])),
        "capacity_passed": live_free >= minimum,
        "canonical_stock_source_present": Path(refresh["canonical_stock_root"]).is_dir(),
        "canonical_benchmark_present": Path(refresh["canonical_benchmark"]).is_file(),
    }
    require(live == {"credential_present": False, "capacity_passed": True,
                     "canonical_stock_source_present": True,
                     "canonical_benchmark_present": True},
            "live refresh prerequisites differ")
    evidence = {
        "result_sha256": sha256(raw["result"]).hexdigest(),
        "decision_digest": stable(expected_decision),
        "live_prerequisites": live,
    }
    return result, evidence


def verification_state(
    result: Mapping[str, Any], evidence: Mapping[str, Any], commit: str,
    runtime_sha256: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA, "status": "verified_wait_state_independently_confirmed",
        "passed": True, "verifier_commit": commit,
        "verifier_runtime_sha256": dict(runtime_sha256),
        "producer_implementation_commit": result["implementation_commit"],
        "producer_result_digest": result["result_digest"],
        "producer_result_sha256": evidence["result_sha256"],
        "decision_digest": evidence["decision_digest"],
        "live_prerequisites": evidence["live_prerequisites"],
        "registry_callback_calls": 0, "prediction_callback_calls": 0,
        "canonical_inputs_modified": False, "post_freeze_outcomes_opened": False,
        "predictive_claim_authorized": False,
        "production_promotion_authorized": False, "trading_claim_authorized": False,
    }


def publish(path: Path, value: Mapping[str, Any]) -> None:
    require(not path.exists() and not path.is_symlink(), "create-only verification exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".m2-launch-verify-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write((json.dumps(value, indent=2, sort_keys=True,
                                     allow_nan=False) + "\n").encode())
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
    runtime = {name: snapshot(repository / name) for name in VERIFIER_RUNTIME}
    for name, content in runtime.items():
        require(content == git(repository, "show", f"{commit}:{name}", binary=True),
                f"verifier runtime not committed: {name}")
    hashes = {name: sha256(content).hexdigest() for name, content in runtime.items()}
    state = verification_state(result, evidence, commit, hashes)
    receipt = {**state, "verification_digest": stable(state),
               "created_at": datetime.now(timezone.utc).isoformat()}
    publish(repository / OUTPUT, receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    try:
        receipt = run(args.repository)
    except (VerificationError, OSError) as error:
        print(f"M2 launcher verification refused: {error}", file=os.sys.stderr)
        return 2
    print(json.dumps({"passed": receipt["passed"],
                      "verification_digest": receipt["verification_digest"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
