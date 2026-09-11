"""Run the R2-02 synthetic calendar-block stability and mode-selection gate."""
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

from market_analogues.future_modes import Member, PreparedPath, pairwise_l1, select_modes


SCHEMA = "m04r15-r2-stability-synthetic-gate-v1"
CONTRACT = Path("config/m04r15-r2-fixed-neighbor-modes-contract-v1.json")
R201 = Path(
    "config/data/analogues/m04r15/r2-future-modes-synthetic-gate-v1-verification/VERIFIED.json"
)
OUTPUT = Path("config/data/analogues/m04r15/r2-stability-synthetic-gate-v1/RESULT.json")
RUNTIME = (
    "config/m04r15-r2-fixed-neighbor-modes-contract-v1.json",
    "config/data/analogues/m04r15/r2-future-modes-synthetic-gate-v1-verification/VERIFIED.json",
    "src/market_analogues/future_modes.py",
    "experiments/m04r/m04r15_r2_stability_synthetic_gate.py",
    "tests/test_future_modes.py",
)


class StabilityGateError(RuntimeError): pass


def _require(value: bool, message: str) -> None:
    if not value: raise StabilityGateError(message)


def _stable(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"regular JSON required: {path}")
    value = json.loads(path.read_text()); _require(type(value) is dict, "JSON object required")
    return value


def _sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20): digest.update(block)
    return digest.hexdigest()


def _paths(levels: Sequence[float], horizon: int = 60) -> tuple[PreparedPath, ...]:
    return tuple(PreparedPath(
        Member(index + 1, f"episode-{index:02d}", f"S{index:02d}", "2020-01-02", "synthetic"),
        (float(level),) * horizon, tuple(f"step-{step:03d}" for step in range(1, horizon + 1)),
    ) for index, level in enumerate(levels))


def _summary(selection) -> dict[str, Any]:
    candidates = []
    for candidate in selection.candidates:
        stability = candidate.stability
        candidates.append({
            "k": candidate.k,
            "medoid_indices": list(candidate.medoid_indices),
            "labels": list(candidate.labels),
            "cluster_sizes": list(candidate.cluster_sizes),
            "mean_silhouette": candidate.mean_silhouette,
            "accepted": candidate.accepted,
            "rejection_reasons": list(candidate.rejection_reasons),
            "stability": None if stability is None else {
                "valid_replicates": stability.valid_replicates,
                "median_adjusted_rand_index": stability.median_adjusted_rand_index,
                "adjusted_rand_indices_digest": _stable(list(stability.adjusted_rand_indices)),
                "minimum_adjusted_rand_index": min(stability.adjusted_rand_indices) if stability.adjusted_rand_indices else None,
                "maximum_adjusted_rand_index": max(stability.adjusted_rand_indices) if stability.adjusted_rand_indices else None,
            },
        })
    return {
        "status": selection.status, "selected_k": selection.selected_k,
        "medoid_indices": list(selection.medoid_indices), "labels": list(selection.labels),
        "candidates": candidates,
    }


def _case(levels: Sequence[float], blocks: Sequence[str], *, name: str) -> dict[str, Any]:
    paths = _paths(levels)
    return _summary(select_modes(
        pairwise_l1(paths), [path.member.key for path in paths], blocks,
        contract_digest="6bec009810afce7c58508fca28179579f5904382376dc9c5bce74aa74f41e08c",
        query_case_id=name, view_id="absolute_close_return", replicates=256,
    ))


def execute(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    contract = _json(repository / CONTRACT); prior = _json(repository / R201)
    _require(prior.get("passed") is True and prior.get("calendar_block_stability_authorized") is True,
             "R2-01 receipt differs")
    _require(prior.get("contract_digest") == contract.get("contract_digest"), "contract chain differs")
    commit = subprocess.run(("git", "rev-parse", "HEAD"), cwd=repository,
                            text=True, capture_output=True, check=True).stdout.strip()
    runtime = {name: _sha(repository / name) for name in RUNTIME}
    for name, digest in runtime.items():
        historical = subprocess.run(("git", "show", f"{commit}:{name}"), cwd=repository,
                                    capture_output=True, check=False)
        _require(historical.returncode == 0 and sha256(historical.stdout).hexdigest() == digest,
                 f"runtime differs from H0: {name}")

    stable = _case(
        [-1.01, -1.0, -0.99, -0.01, 0.0, 0.01, 0.99, 1.0, 1.01],
        [f"20{20 + index // 4}-Q{index % 4 + 1}" for index in range(9)], name="stable",
    )
    confounded = _case(
        [-1.01, -1.0, -0.99, 0.99, 1.0, 1.01],
        ["2020-Q1"] * 3 + ["2020-Q2"] * 3, name="date-confounded",
    )
    tight = _case([0.0] * 6, [f"20{index:02d}-Q1" for index in range(6)], name="tight")
    tiny = _case([0.0, 1.0], ["2020-Q1", "2021-Q1"], name="tiny")
    repeated = _case(
        [-1.01, -1.0, -0.99, -0.01, 0.0, 0.01, 0.99, 1.0, 1.01],
        [f"20{20 + index // 4}-Q{index % 4 + 1}" for index in range(9)], name="stable",
    )
    _require(stable == repeated, "repeat identity differs")
    _require(stable["status"] == "stable_multiple_modes" and stable["selected_k"] == 3,
             "stable family selection differs")
    _require(confounded["status"] == "one_mode_fallback", "date-confounded fallback differs")
    _require(tight["status"] == "one_mode_fallback", "tight-family fallback differs")
    _require(tiny["status"] == "abstain_insufficient_complete_primary_members"
             and tiny["selected_k"] == 0, "tiny-family abstention differs")
    state = {
        "schema_version": SCHEMA, "status": "synthetic_stability_gate_passed", "passed": True,
        "contract_digest": contract["contract_digest"],
        "r201_verification_result_digest": prior["result_digest"],
        "implementation_commit": commit, "runtime_sha256": runtime,
        "cases": {"stable_three_modes": stable, "date_confounded": confounded,
                  "tight_one_family": tight, "tiny_cohort": tiny},
        "repeat_identity": True,
        "bootstrap_validity_clarification": {
            "minimum_base_distinct_blocks": "max(4,k+1)",
            "valid_resample_minimum_distinct_blocks": "k",
            "valid_resample_requires_all_k_assigned_clusters": True,
        },
        "real_future_path_store_opened": False,
        "bounded_consumed_data_poc_authorized": True,
        "real_full_build_authorized": False,
        "predictive_claim_authorized": False,
        "production_promotion_authorized": False,
        "trading_claim_authorized": False,
    }
    return {**state, "result_digest": _stable(state)}


def _publish(path: Path, value: Mapping[str, Any]) -> None:
    _require(not path.exists() and not path.is_symlink(), "stability result exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".r2-stability-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write((json.dumps({**value, "created_at": datetime.now(timezone.utc).isoformat()},
                                     indent=2, sort_keys=True, allow_nan=False) + "\n").encode())
            handle.flush(); os.fsync(handle.fileno())
        os.rename(temporary, path)
    except Exception:
        try: temporary.unlink()
        except OSError: pass
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv); value = execute(args.repository)
    if not args.dry_run: _publish(args.repository.resolve(strict=True) / OUTPUT, value)
    print(json.dumps(value, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
