"""Independently verify R2-01 without importing its producer or mode module."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from itertools import combinations
import json
from math import fsum
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Mapping, Sequence


SCHEMA = "m04r15-r2-future-modes-synthetic-verification-v1"
RESULT = Path("config/data/analogues/m04r15/r2-future-modes-synthetic-gate-v1/RESULT.json")
CONTRACT = Path("config/m04r15-r2-fixed-neighbor-modes-contract-v1.json")
CONTRACT_VERIFICATION = Path(
    "config/data/analogues/m04r15/r2-fixed-neighbor-modes-contract-verification-v1/VERIFIED.json"
)
OUTPUT = Path(
    "config/data/analogues/m04r15/r2-future-modes-synthetic-gate-v1-verification"
)
PRODUCER_MODULES = {
    "market_analogues.future_modes",
    "experiments.m04r.m04r15_r2_future_modes_synthetic_gate",
}


class SyntheticVerificationError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SyntheticVerificationError(message)


def _stable(value: Any) -> str:
    return sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode()).hexdigest()


def _json(path: Path) -> tuple[dict[str, Any], bytes]:
    _require(path.is_file() and not path.is_symlink(), f"regular JSON required: {path}")
    raw = path.read_bytes()
    value = json.loads(raw)
    _require(type(value) is dict, f"JSON object required: {path}")
    return value, raw


def _sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20): digest.update(block)
    return digest.hexdigest()


def _git(repository: Path, *args: str, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        ("git", *args), cwd=repository, capture_output=True,
        text=not binary, check=False,
    )
    _require(result.returncode == 0, f"git {' '.join(args)} failed")
    return result.stdout if binary else result.stdout.strip()


def _oracle_three_families() -> dict[str, Any]:
    levels = [-1.01, -1.0, -0.99, -0.01, 0.0, 0.01, 0.99, 1.0, 1.01]
    matrix = tuple(tuple(abs(left - right) for right in levels) for left in levels)
    candidates = []
    for medoids in combinations(range(9), 3):
        labels = tuple(min(
            range(3), key=lambda label: (matrix[row][medoids[label]], medoids[label]),
        ) for row in range(9))
        objective = fsum(matrix[row][medoids[labels[row]]] for row in range(9))
        candidates.append((objective, medoids, labels))
    objective, medoids, labels = min(candidates)
    groups = {label: [index for index, value in enumerate(labels) if value == label]
              for label in range(3)}
    scores = []
    for index, label in enumerate(labels):
        own = groups[label]
        a = fsum(matrix[index][other] for other in own if other != index) / (len(own) - 1)
        b = min(
            fsum(matrix[index][other] for other in members) / len(members)
            for other_label, members in groups.items() if other_label != label
        )
        scores.append((b - a) / max(a, b))
    return {
        "medoid_episode_ids": [f"episode-{index:02d}" for index in medoids],
        "labels": list(labels), "objective": objective,
        "mean_silhouette": fsum(scores) / len(scores),
    }


def verify(repository: Path, result_path: Path | None = None) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    result, raw = _json((result_path or repository / RESULT).resolve(strict=True))
    contract, _ = _json(repository / CONTRACT)
    prerequisite, _ = _json(repository / CONTRACT_VERIFICATION)
    deterministic = {key: value for key, value in result.items()
                     if key not in {"result_digest", "created_at"}}
    _require(result.get("result_digest") == _stable(deterministic), "synthetic result digest differs")
    _require(all((
        result.get("schema_version") == "m04r15-r2-future-modes-synthetic-gate-v1",
        result.get("status") == "synthetic_path_and_pam_gate_passed",
        result.get("passed") is True,
        result.get("contract_digest") == contract.get("contract_digest"),
        result.get("contract_verification_result_digest") == prerequisite.get("result_digest"),
        result.get("real_future_path_store_opened") is False,
        result.get("synthetic_values_only") is True,
        result.get("calendar_block_stability_authorized") is True,
        result.get("real_future_path_mode_computation_authorized") is False,
        result.get("predictive_claim_authorized") is False,
        result.get("production_promotion_authorized") is False,
        result.get("trading_claim_authorized") is False,
    )), "synthetic gate state differs")
    cases = result.get("cases", {})
    _require(cases.get("outcome_blind_dependence") == {
        "primary": ["a-first", "b"],
        "excluded": [["a-late", "duplicate_matched_symbol"], ["query-memory", "query_symbol_memory"]],
    }, "independent dependence result differs")
    _require(cases.get("strict_path_validation") == {
        "complete": ["valid"],
        "invalid": [["invalid", "nonfinite_value+unexpected_session"]],
    }, "independent validation result differs")
    oracle = _oracle_three_families()
    _require(cases.get("three_separated_families") == oracle, "independent clustering oracle differs")
    expected_assignments = {
        f"episode-{index:02d}": f"episode-{(index // 3) * 3 + 1:02d}"
        for index in range(9)
    }
    _require(cases.get("row_order_invariance") == {
        "medoid_episode_ids": oracle["medoid_episode_ids"],
        "member_to_medoid": expected_assignments,
        "matches_forward": True,
    }, "independent row-order result differs")
    _require(cases.get("weighted_and_tie_determinism") == {
        "weighted_medoid_episode_id": "episode-00",
        "tie_medoid_episode_id": "episode-00",
    }, "independent weighted/tie result differs")

    commit = str(result.get("implementation_commit", ""))
    _require(len(commit) == 40, "implementation commit differs")
    _git(repository, "cat-file", "-e", f"{commit}^{{commit}}")
    runtime = result.get("runtime_sha256")
    _require(type(runtime) is dict and runtime, "runtime inventory differs")
    for name, digest in runtime.items():
        historical = _git(repository, "show", f"{commit}:{name}", binary=True)
        _require(sha256(historical).hexdigest() == digest, f"historical runtime differs: {name}")
    state = {
        "schema_version": SCHEMA,
        "status": "independently_verified",
        "passed": True,
        "producer_result_digest": result["result_digest"],
        "producer_result_sha256": sha256(raw).hexdigest(),
        "contract_digest": contract["contract_digest"],
        "implementation_commit": commit,
        "independent_oracle": "exhaustive_three_medoid_scalar_enumeration",
        "producer_modules_imported": False,
        "verified_case_count": 5,
        "real_future_path_store_opened": False,
        "calendar_block_stability_authorized": True,
        "real_future_path_mode_computation_authorized": False,
        "predictive_claim_authorized": False,
        "production_promotion_authorized": False,
        "trading_claim_authorized": False,
    }
    return {**state, "result_digest": _stable(state)}


def _publish(path: Path, value: Mapping[str, Any]) -> None:
    _require(not path.exists() and not path.is_symlink(), "synthetic verification exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".r2-synthetic-verification-", dir=path.parent))
    try:
        target = temporary / "VERIFIED.json"
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write((json.dumps({
                **value, "created_at": datetime.now(timezone.utc).isoformat(),
            }, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())
            handle.flush(); os.fsync(handle.fileno())
        os.rename(temporary, path)
    except Exception:
        try:
            if (temporary / "VERIFIED.json").exists(): (temporary / "VERIFIED.json").unlink()
            temporary.rmdir()
        except OSError: pass
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    value = verify(args.repository, args.result)
    if not args.dry_run:
        _publish(args.repository.resolve(strict=True) / OUTPUT, value)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
