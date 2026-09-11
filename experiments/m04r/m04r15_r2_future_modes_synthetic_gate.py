"""Run the R2-01 synthetic path-preparation and deterministic-PAM gate."""
from __future__ import annotations

import argparse
from datetime import date, timedelta, datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Mapping, Sequence

from market_analogues.future_modes import (
    mean_silhouette, pairwise_l1, pam, prepare_paths, select_primary_members,
)


SCHEMA = "m04r15-r2-future-modes-synthetic-gate-v1"
CONTRACT = Path("config/m04r15-r2-fixed-neighbor-modes-contract-v1.json")
CONTRACT_VERIFICATION = Path(
    "config/data/analogues/m04r15/r2-fixed-neighbor-modes-contract-verification-v1/VERIFIED.json"
)
OUTPUT = Path("config/data/analogues/m04r15/r2-future-modes-synthetic-gate-v1/RESULT.json")
RUNTIME = (
    "config/m04r15-r2-fixed-neighbor-modes-contract-v1.json",
    "config/data/analogues/m04r15/r2-fixed-neighbor-modes-contract-verification-v1/VERIFIED.json",
    "src/market_analogues/future_modes.py",
    "experiments/m04r/m04r15_r2_future_modes_synthetic_gate.py",
    "tests/test_future_modes.py",
)


class SyntheticGateError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SyntheticGateError(message)


def _stable(value: Any) -> str:
    return sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode()).hexdigest()


def _sha(path: Path) -> str:
    _require(path.is_file() and not path.is_symlink(), f"regular file required: {path}")
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"regular JSON required: {path}")
    value = json.loads(path.read_text())
    _require(type(value) is dict, f"JSON object required: {path}")
    return value


def _git(repository: Path, *args: str) -> str:
    result = subprocess.run(("git", *args), cwd=repository, text=True, capture_output=True)
    _require(result.returncode == 0, f"git {' '.join(args)} failed")
    return result.stdout.strip()


def _member(rank: int, episode: str, symbol: str) -> dict[str, Any]:
    return {
        "match_rank": rank, "matched_episode_id": episode,
        "matched_symbol": symbol, "matched_cutoff": "2020-01-02",
        "source_fingerprint": "synthetic-source",
    }


def _rows(episode: str, values: Sequence[float]) -> list[dict[str, Any]]:
    origin = date(2020, 1, 2)
    return [{
        "step": step, "episode_id": episode, "cutoff": "2020-01-02",
        "timestamp": (origin + timedelta(days=step)).isoformat(),
        "expected_session_match": True, "contract_digest": "synthetic-contract",
        "source_content_digest": "synthetic-content",
        "source_fingerprint": "synthetic-source", "close_return": float(value),
    } for step, value in enumerate(values, 1)]


def _prepared(families: Sequence[Sequence[float]]):
    raw = [_member(index + 1, f"episode-{index:02d}", f"S{index:02d}")
           for index in range(len(families))]
    members, dependence = select_primary_members("QUERY", raw)
    paths, invalid = prepare_paths(
        members,
        {f"episode-{index:02d}": _rows(f"episode-{index:02d}", values)
         for index, values in enumerate(families)},
        value_field="close_return", horizon=len(families[0]),
    )
    _require(not dependence and not invalid, "synthetic preparation differs")
    return paths


def execute(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    contract = _json(repository / CONTRACT)
    verified = _json(repository / CONTRACT_VERIFICATION)
    _require(all((
        verified.get("passed") is True,
        verified.get("contract_digest") == contract.get("contract_digest"),
        verified.get("synthetic_gate_authorized") is True,
        verified.get("future_path_columns_opened") == [],
        verified.get("real_future_path_mode_computation_opened") is False,
    )), "R2 contract verification differs")
    commit = _git(repository, "rev-parse", "HEAD")
    runtime = {name: _sha(repository / name) for name in RUNTIME}
    for name, digest in runtime.items():
        historical = subprocess.run(
            ("git", "show", f"{commit}:{name}"), cwd=repository,
            capture_output=True, check=False,
        )
        _require(historical.returncode == 0, f"runtime absent at H0: {name}")
        _require(sha256(historical.stdout).hexdigest() == digest, f"runtime differs from H0: {name}")

    raw = [
        _member(4, "query-memory", "QUERY"), _member(3, "a-late", "A"),
        _member(1, "a-first", "A"), _member(2, "b", "B"),
    ]
    primary, excluded = select_primary_members("QUERY", list(reversed(raw)))
    dependence_case = {
        "primary": [item.episode_id for item in primary],
        "excluded": [[item.member.episode_id, item.reason] for item in excluded],
    }

    validation_members, _ = select_primary_members(
        "QUERY", [_member(1, "valid", "A"), _member(2, "invalid", "B")],
    )
    validation_rows = {
        "valid": _rows("valid", [step / 100 for step in range(60)]),
        "invalid": _rows("invalid", [step / 100 for step in range(60)]),
    }
    validation_rows["invalid"][7]["expected_session_match"] = False
    validation_rows["invalid"][9]["close_return"] = float("nan")
    complete, invalid = prepare_paths(
        validation_members, validation_rows, value_field="close_return", horizon=60,
    )
    validation_case = {
        "complete": [item.member.episode_id for item in complete],
        "invalid": [[item.member.episode_id, item.reason] for item in invalid],
    }

    paths = _prepared([
        [-1.01] * 60, [-1.0] * 60, [-0.99] * 60,
        [-0.01] * 60, [0.0] * 60, [0.01] * 60,
        [0.99] * 60, [1.0] * 60, [1.01] * 60,
    ])
    matrix = pairwise_l1(paths)
    keys = [path.member.key for path in paths]
    modes = pam(matrix, keys, 3)
    clustering_case = {
        "medoid_episode_ids": [paths[index].member.episode_id for index in modes.medoid_indices],
        "labels": list(modes.labels), "objective": modes.objective,
        "mean_silhouette": mean_silhouette(matrix, modes.labels),
    }

    order = tuple(reversed(range(len(paths))))
    reversed_result = pam(
        [[matrix[left][right] for right in order] for left in order],
        [keys[index] for index in order], 3,
    )
    reversed_medoids = [paths[order[index]].member.episode_id
                        for index in reversed_result.medoid_indices]
    reversed_assignments = {
        paths[order[index]].member.episode_id:
        paths[order[reversed_result.medoid_indices[label]]].member.episode_id
        for index, label in enumerate(reversed_result.labels)
    }
    assignments = {
        path.member.episode_id: paths[modes.medoid_indices[modes.labels[index]]].member.episode_id
        for index, path in enumerate(paths)
    }
    reversal_case = {
        "medoid_episode_ids": reversed_medoids,
        "member_to_medoid": dict(sorted(reversed_assignments.items())),
        "matches_forward": reversed_medoids == clustering_case["medoid_episode_ids"]
        and reversed_assignments == assignments,
    }

    weighted_paths = _prepared([[0.0] * 60, [5.0] * 60, [6.0] * 60])
    weighted = pam(
        pairwise_l1(weighted_paths), [path.member.key for path in weighted_paths],
        1, weights=[10.0, 1.0, 1.0],
    )
    tie_paths = _prepared([[0.0] * 60, [2.0] * 60])
    tie = pam(pairwise_l1(tie_paths), [path.member.key for path in tie_paths], 1)
    deterministic_case = {
        "weighted_medoid_episode_id": weighted_paths[weighted.medoid_indices[0]].member.episode_id,
        "tie_medoid_episode_id": tie_paths[tie.medoid_indices[0]].member.episode_id,
    }

    _require(dependence_case == {
        "primary": ["a-first", "b"],
        "excluded": [["a-late", "duplicate_matched_symbol"], ["query-memory", "query_symbol_memory"]],
    }, "dependence fixture differs")
    _require(validation_case == {
        "complete": ["valid"],
        "invalid": [["invalid", "nonfinite_value+unexpected_session"]],
    }, "validation fixture differs")
    _require(clustering_case["medoid_episode_ids"] == ["episode-01", "episode-04", "episode-07"], "medoids differ")
    _require(clustering_case["labels"] == [0, 0, 0, 1, 1, 1, 2, 2, 2], "assignments differ")
    _require(reversal_case["matches_forward"] is True, "row-order invariance differs")
    _require(deterministic_case == {
        "weighted_medoid_episode_id": "episode-00",
        "tie_medoid_episode_id": "episode-00",
    }, "weighted/tie behavior differs")

    state = {
        "schema_version": SCHEMA,
        "status": "synthetic_path_and_pam_gate_passed",
        "passed": True,
        "contract_digest": contract["contract_digest"],
        "contract_verification_result_digest": verified["result_digest"],
        "implementation_commit": commit,
        "runtime_sha256": runtime,
        "cases": {
            "outcome_blind_dependence": dependence_case,
            "strict_path_validation": validation_case,
            "three_separated_families": clustering_case,
            "row_order_invariance": reversal_case,
            "weighted_and_tie_determinism": deterministic_case,
        },
        "real_future_path_store_opened": False,
        "synthetic_values_only": True,
        "calendar_block_stability_authorized": True,
        "real_future_path_mode_computation_authorized": False,
        "predictive_claim_authorized": False,
        "production_promotion_authorized": False,
        "trading_claim_authorized": False,
    }
    return {**state, "result_digest": _stable(state)}


def _publish(path: Path, value: Mapping[str, Any]) -> None:
    _require(not path.exists() and not path.is_symlink(), "synthetic result exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".r2-synthetic-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write((json.dumps({
                **value, "created_at": datetime.now(timezone.utc).isoformat(),
            }, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())
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
    args = parser.parse_args(argv)
    result = execute(args.repository)
    if not args.dry_run:
        _publish(args.repository.resolve(strict=True) / OUTPUT, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
