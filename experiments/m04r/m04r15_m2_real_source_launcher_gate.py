"""Run the M2-03 real-source launcher against the frozen availability pair."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Mapping, Sequence

from market_analogues.prospective_launcher import (
    ProspectiveLaunchError, decide_launch, decision_document, refresh_assessment, stable,
)


SCHEMA = "m04r15-m2-real-source-launcher-gate-v1"
CONTRACT = Path("config/prospective-real-source-launch-contract-v1.json")
AVAILABILITY = Path("config/data/analogues/m04r15/m2-availability-preflight-v1/RESULT.json")
VERIFICATION = Path(
    "config/data/analogues/m04r15/m2-availability-preflight-v1-verification/VERIFIED.json"
)
OUTPUT = Path("config/data/analogues/m04r15/m2-real-source-launcher-gate-v1/RESULT.json")
RUNTIME = (
    "config/prospective-real-source-launch-contract-v1.json",
    "src/market_analogues/prospective_launcher.py",
    "experiments/m04r/m04r15_m2_real_source_launcher_gate.py",
    "tests/test_prospective_launcher.py",
    "tests/test_m04r15_m2_real_source_launcher_gate.py",
)


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def snapshot(path: Path) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(descriptor)
        require(os.path.isfile(path) and not path.is_symlink(), f"unsafe input: {path}")
        blocks = []
        while block := os.read(descriptor, 1 << 20):
            blocks.append(block)
        after = os.fstat(descriptor)
        require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns,
                 before.st_ctime_ns) ==
                (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns,
                 after.st_ctime_ns), f"input changed: {path}")
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
                GateError(f"nonfinite JSON: {path}/{token}")),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GateError(f"invalid JSON: {path}") from error
    require(type(value) is dict, f"JSON object required: {path}")
    return value


def git(repository: Path, *arguments: str, binary: bool = False) -> str | bytes:
    result = subprocess.run(("git", *arguments), cwd=repository, capture_output=True,
                            text=not binary, check=False)
    require(result.returncode == 0, f"git {' '.join(arguments)} failed")
    return result.stdout if binary else result.stdout.strip()


def gate_state(
    contract: Mapping[str, Any], availability: Mapping[str, Any],
    verification: Mapping[str, Any], *, credential_present: bool, free_bytes: int,
    canonical_stock_exists: bool, canonical_benchmark_exists: bool,
) -> dict[str, Any]:
    calls = {"registry": 0, "prediction": 0}
    decision = decide_launch(
        contract, availability, verification,
        registry_callback=lambda: calls.__setitem__("registry", calls["registry"] + 1),
        prediction_callback=lambda: calls.__setitem__("prediction", calls["prediction"] + 1),
    )
    require(calls == {"registry": decision.registry_callback_calls,
                      "prediction": decision.prediction_callback_calls},
            "callback accounting differs")
    assessment = refresh_assessment(
        contract, credential_present=credential_present, free_bytes=free_bytes,
        canonical_stock_exists=canonical_stock_exists,
        canonical_benchmark_exists=canonical_benchmark_exists,
    )
    return {
        "schema_version": SCHEMA,
        "status": "verified_wait_state" if decision.action == "wait_for_source_refresh"
        else "launch_callbacks_completed",
        "decision": decision_document(decision),
        "refresh_assessment": assessment,
        "canonical_inputs_modified": False,
        "real_registry_created": calls["registry"] > 0,
        "real_prediction_created": calls["prediction"] > 0,
        "post_freeze_outcomes_opened": False,
        "predictive_claim_authorized": False,
        "production_promotion_authorized": False,
        "trading_claim_authorized": False,
    }


def publish(path: Path, value: Mapping[str, Any]) -> None:
    require(not path.exists() and not path.is_symlink(), "create-only launcher result exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".m2-launch-", dir=path.parent)
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
    require(not (repository / OUTPUT).exists(), "create-only launcher result exists")
    require(not str(git(repository, "status", "--porcelain", "--untracked-files=all")),
            "clean committed tree required")
    raw = {"contract": snapshot(repository / CONTRACT),
           "availability": snapshot(repository / AVAILABILITY),
           "verification": snapshot(repository / VERIFICATION)}
    contract = decode(raw["contract"], repository / CONTRACT)
    availability = decode(raw["availability"], repository / AVAILABILITY)
    verification = decode(raw["verification"], repository / VERIFICATION)
    contract_state = {key: value for key, value in contract.items()
                      if key != "contract_digest"}
    require(contract["contract_digest"] == stable(contract_state), "contract seal differs")
    refresh = contract["refresh"]
    source_repository = Path(refresh["source_repository"])
    canonical_stock = Path(refresh["canonical_stock_root"])
    canonical_benchmark = Path(refresh["canonical_benchmark"])
    capacity = os.statvfs(source_repository)
    free_bytes = capacity.f_bavail * capacity.f_frsize
    state = gate_state(
        contract, availability, verification,
        credential_present=bool(os.environ.get(refresh["credential_environment_variable"])),
        free_bytes=free_bytes, canonical_stock_exists=canonical_stock.is_dir(),
        canonical_benchmark_exists=canonical_benchmark.is_file(),
    )
    commit = str(git(repository, "rev-parse", "HEAD"))
    runtime = {name: snapshot(repository / name) for name in RUNTIME}
    for name, content in runtime.items():
        require(content == git(repository, "show", f"{commit}:{name}", binary=True),
                f"runtime not committed: {name}")
    state.update({
        "contract_digest": contract["contract_digest"],
        "implementation_commit": commit,
        "runtime_sha256": {name: sha256(content).hexdigest()
                           for name, content in runtime.items()},
        "input_sha256": {name: sha256(content).hexdigest()
                         for name, content in raw.items()},
    })
    result = {**state, "result_digest": stable(state),
              "created_at": datetime.now(timezone.utc).isoformat()}
    publish(repository / OUTPUT, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    try:
        result = run(args.repository)
    except (GateError, ProspectiveLaunchError, OSError) as error:
        print(f"M2 real-source launch refused: {error}", file=os.sys.stderr)
        return 2
    print(json.dumps({"status": result["status"],
                      "action": result["decision"]["action"],
                      "result_digest": result["result_digest"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
