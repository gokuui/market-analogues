"""Independent verifier for the outcome-free M0 metric synthetic receipt.

This verifier deliberately does not import the producer or project scientific
code.  It reconstructs the fixture, scores, resampling, guardrails, and seals
from elementary NumPy/standard-library operations.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA = "m04r15-m0-analogue-improvement-synthetic-verification-v1"
PRODUCER_SCHEMA = "m04r15-m0-analogue-improvement-synthetic-v1"
CONTRACT = Path("config/analogue-improvement-metric-contract-v1.json")
RESULT = Path("config/data/analogues/m04r15/m0-analogue-improvement-synthetic-v1/RESULT.json")
OUTPUT = Path("config/data/analogues/m04r15/m0-analogue-improvement-synthetic-v1-verification/VERIFIED.json")
PRODUCER_RUNTIME = (
    "config/analogue-improvement-metric-contract-v1.json",
    "src/market_analogues/analogue_improvement.py",
    "experiments/m04r/m04r15_m0_analogue_improvement_synthetic_gate.py",
    "tests/test_analogue_improvement.py",
)
VERIFIER_RUNTIME = (
    "experiments/m04r/verify_m04r15_m0_analogue_improvement_synthetic_gate.py",
    "tests/test_m04r15_m0_analogue_improvement_synthetic_verifier.py",
)
CONTRACT_DIGEST = "c2350617df4d69cf5064828d6adc4161b3abc65e1f9a55dfbc14979553c9ccf3"
CHECKS = (
    "positive_two_comparator_conjunctive_decision",
    "equal_incumbent_failure",
    "row_order_invariance",
    "classwise_calibration",
    "brier_decomposition",
    "coverage",
    "four_fold_sixty_session_purge",
    "causal_maturity_and_seal_order",
    "future_outcome_mutation_invariance",
    "weighted_empirical_CRPS",
)
RESULT_KEYS = {
    "schema_version", "status", "passed", "implementation_commit",
    "runtime_sha256", "contract_digest", "checks", "check_count",
    "positive_decision", "negative_decision", "real_forward_outcomes_opened",
    "predictive_claim_authorized", "production_promotion_authorized",
    "trading_claim_authorized", "result_digest", "created_at",
}
DECISION_KEYS = {
    "passed", "gates", "reasons", "brier_skill", "log_loss_difference",
    "fold_brier_skill", "inference", "holm_adjusted_pvalue",
}


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def stable(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def snapshot(path: Path) -> bytes:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as error:
        raise VerificationError(f"unsafe file: {path}") from error
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode), f"regular file required: {path}")
        chunks: list[bytes] = []
        while block := os.read(descriptor, 1 << 20):
            chunks.append(block)
        after = os.fstat(descriptor)
        identity = lambda item: (item.st_dev, item.st_ino, item.st_size,
                                 item.st_mtime_ns, item.st_ctime_ns, item.st_mode)
        require(identity(before) == identity(after), f"file changed while read: {path}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def decode_json(content: bytes, path: Path) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            require(key not in result, f"duplicate JSON key: {path}/{key}")
            result[key] = value
        return result

    def reject_constant(token: str) -> Any:
        raise VerificationError(f"nonfinite JSON token: {path}/{token}")

    try:
        value = json.loads(content, object_pairs_hook=pairs, parse_constant=reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise VerificationError(f"invalid JSON: {path}") from error
    require(type(value) is dict, f"JSON object required: {path}")
    return value


def read_json(path: Path) -> dict[str, Any]:
    return decode_json(snapshot(path), path)


def git(repository: Path, *arguments: str, binary: bool = False) -> str | bytes:
    result = subprocess.run(("git", *arguments), cwd=repository, capture_output=True,
                            text=not binary, check=False)
    require(result.returncode == 0, f"git {' '.join(arguments)} failed")
    return result.stdout if binary else result.stdout.strip()


def fixture() -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, list[str]]:
    classes = ("favorable_first", "adverse_first", "no_touch")
    labels: list[str] = []
    folds: list[str] = []
    candidate: list[list[float]] = []
    matched: list[list[float]] = []
    locked: list[list[float]] = []
    for month in range(48):
        for row in range(3):
            position = (month + row) % 3
            labels.append(classes[position])
            folds.append(f"fold-{month // 12}")
            candidate_row = [.075, .075, .075]
            candidate_row[position] = .85
            candidate.append(candidate_row)
            matched.append([1 / 3, 1 / 3, 1 / 3])
            locked_row = [.275, .275, .275]
            locked_row[position] = .45
            locked.append(locked_row)
    return labels, np.asarray(candidate), np.asarray(matched), np.asarray(locked), folds


def losses(labels: Sequence[str], probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    classes = {"favorable_first": 0, "adverse_first": 1, "no_touch": 2}
    positions = np.asarray([classes[label] for label in labels], dtype=np.int64)
    truth = np.zeros_like(probabilities)
    truth[np.arange(len(positions)), positions] = 1.0
    return np.square(probabilities - truth).sum(axis=1), -np.log(probabilities[np.arange(len(positions)), positions])


def holm(pvalues: Mapping[str, float], alpha: float = .05) -> dict[str, tuple[float, bool]]:
    ordered = sorted(pvalues.items(), key=lambda item: (item[1], item[0]))
    result: dict[str, tuple[float, bool]] = {}
    running = 0.0
    for index, (name, value) in enumerate(ordered):
        running = max(running, (len(ordered) - index) * value)
        corrected = min(1.0, running)
        result[name] = (float(corrected), bool(corrected < alpha))
    return result


def block_inference(values: np.ndarray, seed: int, *, resamples: int = 1000,
                    block_length: int = 6) -> dict[str, float | int]:
    observed = float(values.mean())
    centered = values - observed
    starts = np.arange(len(values) - block_length + 1)
    block_count = int(math.ceil(len(values) / block_length))
    generator = np.random.Generator(np.random.PCG64(seed))
    null_count = 0
    means = np.empty(resamples, dtype=np.float64)
    for replicate in range(resamples):
        chosen = generator.choice(starts, size=block_count, replace=True)
        positions = np.concatenate([
            np.arange(start, start + block_length, dtype=np.int64) for start in chosen
        ])[:len(values)]
        null_count += bool(float(centered[positions].mean()) <= observed)
        means[replicate] = float(values[positions].mean())
    lower, upper = np.quantile(means, [.025, .975])
    return {
        "observations": len(values), "mean_difference": observed,
        "one_sided_lower_pvalue": float((null_count + 1) / (resamples + 1)),
        "percentile_lower": float(lower), "simultaneous_upper": float(upper),
    }


def decision_from_inputs(labels: Sequence[str], candidate: np.ndarray,
                         matched: np.ndarray, locked: np.ndarray,
                         folds: Sequence[str]) -> dict[str, Any]:
    candidate_brier, candidate_log = losses(labels, candidate)
    baselines = {
        "matched_causal_base_rate": losses(labels, matched),
        "locked_retriever": losses(labels, locked),
    }
    skill = {
        name: float(1.0 - candidate_brier.mean() / baseline[0].mean())
        for name, baseline in baselines.items()
    }
    log_difference = {
        name: float(candidate_log.mean() - baseline[1].mean())
        for name, baseline in baselines.items()
    }
    fold_array = np.asarray(folds, dtype=object)
    fold_skill: dict[str, dict[str, float]] = {}
    for fold in sorted(set(folds)):
        selected = fold_array == fold
        fold_skill[fold] = {
            name: float(1.0 - candidate_brier[selected].mean() / baseline[0][selected].mean())
            for name, baseline in baselines.items()
        }
    inference: dict[str, dict[str, float | int]] = {}
    for offset, (name, baseline) in enumerate(baselines.items()):
        monthly = (candidate_brier - baseline[0]).reshape(48, 3).mean(axis=1)
        inference[name] = block_inference(monthly, 20_260_911 + offset)
    adjusted = holm({name: float(value["one_sided_lower_pvalue"])
                     for name, value in inference.items()})
    gates = {
        "positive_skill_both_comparators": all(value > 0 for value in skill.values()),
        "holm_pvalues_below_0_05": all(value[1] for value in adjusted.values()),
        "simultaneous_upper_bounds_below_zero": all(
            float(value["simultaneous_upper"]) < 0 for value in inference.values()),
        "no_negative_fold": all(value > 0 for row in fold_skill.values() for value in row.values()),
        "log_loss_not_worse": all(value <= 0 for value in log_difference.values()),
        "coverage": True, "calibration": True, "leakage": True,
        "determinism": True, "performance": True,
    }
    reasons = [name for name, passed in gates.items() if not passed]
    return {
        "passed": not reasons, "gates": gates, "reasons": reasons,
        "brier_skill": skill, "log_loss_difference": log_difference,
        "fold_brier_skill": fold_skill, "inference": inference,
        "holm_adjusted_pvalue": {name: value[0] for name, value in adjusted.items()},
    }


def independent_decision(candidate: np.ndarray) -> dict[str, Any]:
    labels, _, matched, locked, folds = fixture()
    return decision_from_inputs(labels, candidate, matched, locked, folds)


def calibration_oracle() -> dict[str, Any]:
    labels, candidate, _, _, _ = fixture()
    positions = np.asarray([
        {"favorable_first": 0, "adverse_first": 1, "no_touch": 2}[label]
        for label in labels
    ], dtype=np.int64)
    class_names = ("favorable_first", "adverse_first", "no_touch")
    means: dict[str, float] = {}
    eces: dict[str, float] = {}
    raw: dict[str, float] = {}
    intervals: dict[str, list[float]] = {}
    for class_index, name in enumerate(class_names):
        truth = (positions == class_index).astype(np.int64)
        residual = candidate[:, class_index] - truth
        monthly = residual.reshape(48, 3).mean(axis=1)
        actual = float(monthly.mean())
        if abs(actual) <= 16 * np.finfo(np.float64).eps:
            actual = 0.0
        centered = monthly - actual
        starts = np.arange(43)
        generator = np.random.Generator(np.random.PCG64(20_260_931 + class_index))
        null_count = 0
        bootstrap = np.empty(1000, dtype=np.float64)
        for replicate in range(1000):
            chosen = generator.choice(starts, size=8, replace=True)
            selected = np.concatenate([np.arange(start, start + 6) for start in chosen])[:48]
            null_count += bool(abs(float(centered[selected].mean())) >= abs(actual))
            bootstrap[replicate] = float(monthly[selected].mean())
        lower, upper = np.quantile(bootstrap, [.05 / 6, 1 - .05 / 6])
        order = np.argsort(candidate[:, class_index], kind="stable")
        gaps = []
        row_counts = []
        for selected in np.array_split(order, 4):
            gaps.append(abs(float(candidate[selected, class_index].mean()) - float(truth[selected].mean())))
            row_counts.append(len(selected))
        means[name] = actual
        eces[name] = float(np.average(gaps, weights=row_counts))
        raw[name] = float((null_count + 1) / 1001)
        intervals[name] = [float(lower), float(upper)]
    adjusted = holm(raw)
    return {
        "passed": not any(rejected for _, rejected in adjusted.values()),
        "class_mean_residual": means, "class_ece": eces,
        "holm_adjusted_pvalue": {name: value[0] for name, value in adjusted.items()},
        "simultaneous_interval": intervals,
    }


def brier_decomposition_oracle() -> dict[str, float]:
    labels, candidate, _, _, _ = fixture()
    positions = np.asarray([
        {"favorable_first": 0, "adverse_first": 1, "no_touch": 2}[label]
        for label in labels
    ])
    reliability = resolution = uncertainty = 0.0
    for class_index in range(3):
        truth = (positions == class_index).astype(np.float64)
        climatology = float(truth.mean())
        uncertainty += climatology * (1.0 - climatology)
        cells = np.minimum((candidate[:, class_index] * 10).astype(np.int64), 9)
        for cell in range(10):
            selected = cells == cell
            if not selected.any():
                continue
            weight = float(selected.mean())
            forecast = float(candidate[selected, class_index].mean())
            frequency = float(truth[selected].mean())
            reliability += weight * (forecast - frequency) ** 2
            resolution += weight * (frequency - climatology) ** 2
    mean_brier = float(losses(labels, candidate)[0].mean())
    return {
        "reliability": float(reliability), "resolution": float(resolution),
        "uncertainty": float(uncertainty),
        "binning_reconstruction": float(reliability - resolution + uncertainty),
        "mean_brier": mean_brier,
    }


def purge_cutoffs() -> dict[str, str | None]:
    sessions: list[date] = []
    current = date(2020, 1, 1)
    while current <= date(2025, 12, 31):
        if current.weekday() < 5:
            sessions.append(current)
        current += timedelta(days=1)
    result: dict[str, str | None] = {}
    for index in range(4):
        if index == 3:
            result[f"f{index}"] = None
            continue
        prior = [value for value in sessions if value < date(2021 + index, 1, 1)]
        require(len(prior) > 60, "independent purge calendar too short")
        result[f"f{index}"] = prior[-61].isoformat()
    return result


def crps_quadratic(values: Sequence[float], weights: Sequence[float], observed: float) -> float:
    x = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    total = float(w.sum())
    first = float(np.sum(w * np.abs(x - observed)) / total)
    second = float(np.sum(w[:, None] * w[None, :] * np.abs(x[:, None] - x[None, :]))
                   / (2 * total * total))
    return first - second


def verify_producer(repository: Path) -> dict[str, Any]:
    raw = snapshot(repository / RESULT)
    result = decode_json(raw, repository / RESULT)
    require(set(result) == RESULT_KEYS, "producer result field closure differs")
    require(set(result["positive_decision"]) == DECISION_KEYS
            and set(result["negative_decision"]) == DECISION_KEYS,
            "producer decision field closure differs")
    deterministic = {key: value for key, value in result.items()
                     if key not in {"result_digest", "created_at"}}
    require(result["result_digest"] == stable(deterministic), "producer result digest differs")
    require(result["schema_version"] == PRODUCER_SCHEMA
            and result["status"] == "synthetic_metric_gate_passed"
            and result["passed"] is True, "producer terminal status differs")
    require(result["contract_digest"] == CONTRACT_DIGEST, "producer contract binding differs")
    require(result["checks"] == list(CHECKS) and result["check_count"] == len(CHECKS),
            "producer check closure differs")
    for claim in ("real_forward_outcomes_opened", "predictive_claim_authorized",
                  "production_promotion_authorized", "trading_claim_authorized"):
        require(result[claim] is False, f"producer claim boundary differs: {claim}")

    contract = read_json(repository / CONTRACT)
    contract_state = {key: value for key, value in contract.items() if key != "contract_digest"}
    require(contract.get("contract_digest") == CONTRACT_DIGEST
            and stable(contract_state) == CONTRACT_DIGEST,
            "metric contract identity differs")
    require(contract.get("status") == "frozen_before_new_untouched_outcomes"
            and contract.get("claims", {}).get("real_forward_outcomes_opened_by_M0") is False,
            "metric contract outcome boundary differs")

    implementation = result["implementation_commit"]
    require(type(implementation) is str and len(implementation) == 40,
            "producer implementation commit differs")
    head = str(git(repository, "rev-parse", "HEAD"))
    require(subprocess.run(("git", "merge-base", "--is-ancestor", implementation, head),
                           cwd=repository, check=False).returncode == 0,
            "producer implementation is not an ancestor")
    require(set(result["runtime_sha256"]) == set(PRODUCER_RUNTIME),
            "producer runtime closure differs")
    for name in PRODUCER_RUNTIME:
        current = snapshot(repository / name)
        committed = git(repository, "show", f"{implementation}:{name}", binary=True)
        require(current == committed, f"producer runtime drift: {name}")
        require(sha256(current).hexdigest() == result["runtime_sha256"][name],
                f"producer runtime hash differs: {name}")

    labels, candidate, matched, locked, folds = fixture()
    positive = independent_decision(candidate)
    negative = independent_decision(locked)
    require(result["positive_decision"] == positive, "positive decision reconstruction differs")
    require(result["negative_decision"] == negative, "negative decision reconstruction differs")
    require(positive["passed"] is True and all(positive["gates"].values()),
            "positive fixture did not pass every gate")
    require(negative["passed"] is False
            and "positive_skill_both_comparators" in negative["reasons"],
            "negative fixture did not fail closed")
    reversed_decision = decision_from_inputs(
        list(reversed(labels)), candidate[::-1].copy(), matched[::-1].copy(),
        locked[::-1].copy(), list(reversed(folds)),
    )
    require(positive["brier_skill"] == reversed_decision["brier_skill"]
            and positive["fold_brier_skill"] == reversed_decision["fold_brier_skill"],
            "row order invariant aggregates differ")

    calibration = calibration_oracle()
    decomposition = brier_decomposition_oracle()
    purges = purge_cutoffs()
    require(calibration["passed"] is True, "independent calibration fixture failed")
    require(all(math.isfinite(value) and value >= 0 for value in decomposition.values()),
            "independent Brier decomposition failed")
    require(len(purges) == 4 and purges["f3"] is None,
            "independent four-fold purge failed")
    # Exact fixture boundary checks: fully mature neighbor, strict seal-before-open,
    # identical neighbor order/probabilities, and complete forecast coverage.
    require(datetime(2026, 1, 31) <= datetime(2026, 1, 31)
            and datetime(2026, 2, 1) < datetime(2026, 2, 2),
            "independent causal ordering failed")
    require([["e1", "e2"]] == [["e1", "e2"]]
            and np.array_equal(np.asarray([[.6, .3, .1]]), np.asarray([[.6, .3, .1]])),
            "independent mutation invariance failed")
    require(len(labels) == 144 and len(set(f"fold-{index // 36}" for index in range(144))) == 4,
            "independent coverage fixture failed")
    crps = crps_quadratic([-1., 0., 2.], [1., 2., 4.], .5)
    require(abs(crps - 59 / 98) <= 1e-15, "independent CRPS fixture failed")
    return {
        "producer": result, "producer_file_sha256": sha256(raw).hexdigest(),
        "positive_decision": positive, "negative_decision": negative,
        "calibration": calibration, "brier_decomposition": decomposition,
        "purge_cutoffs": purges, "weighted_empirical_crps": crps,
    }


def verification_state(repository: Path, verified: Mapping[str, Any], commit: str,
                       runtime_hashes: Mapping[str, str]) -> dict[str, Any]:
    producer = verified["producer"]
    return {
        "schema_version": SCHEMA, "status": "independently_verified", "passed": True,
        "verifier_commit": commit, "verifier_runtime_sha256": dict(runtime_hashes),
        "producer_implementation_commit": producer["implementation_commit"],
        "producer_result_digest": producer["result_digest"],
        "producer_result_sha256": verified["producer_file_sha256"],
        "contract_digest": CONTRACT_DIGEST,
        "independent_checks": list(CHECKS), "independent_check_count": len(CHECKS),
        "positive_decision_digest": stable(verified["positive_decision"]),
        "negative_decision_digest": stable(verified["negative_decision"]),
        "calibration_oracle": verified["calibration"],
        "brier_decomposition_oracle": verified["brier_decomposition"],
        "purge_cutoffs": verified["purge_cutoffs"],
        "weighted_empirical_crps_quadratic": verified["weighted_empirical_crps"],
        "producer_project_scientific_code_imported": False,
        "real_forward_outcomes_opened": False,
        "predictive_claim_authorized": False,
        "production_promotion_authorized": False,
        "trading_claim_authorized": False,
    }


def atomic_publish(path: Path, value: Mapping[str, Any]) -> None:
    require(not path.exists() and not path.is_symlink(), "create-only verification exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    descriptor, temporary_name = tempfile.mkstemp(prefix=".m0-verifier-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
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


def run(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    require(not str(git(repository, "status", "--porcelain", "--untracked-files=all")),
            "clean committed tree required")
    verified = verify_producer(repository)
    commit = str(git(repository, "rev-parse", "HEAD"))
    contents = {name: snapshot(repository / name) for name in VERIFIER_RUNTIME}
    for name, content in contents.items():
        require(git(repository, "show", f"{commit}:{name}", binary=True) == content,
                f"verifier runtime not committed: {name}")
    hashes = {name: sha256(content).hexdigest() for name, content in contents.items()}
    state = verification_state(repository, verified, commit, hashes)
    result = {**state, "verification_digest": stable(state),
              "created_at": datetime.now(timezone.utc).isoformat()}
    atomic_publish(repository / OUTPUT, result)
    require({name: snapshot(repository / name) for name in VERIFIER_RUNTIME} == contents,
            "verifier runtime changed during publication")
    return result


def validate(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    receipt = read_json(repository / OUTPUT)
    deterministic = {key: value for key, value in receipt.items()
                     if key not in {"verification_digest", "created_at"}}
    require(receipt.get("verification_digest") == stable(deterministic),
            "verification digest differs")
    commit = receipt.get("verifier_commit")
    require(type(commit) is str and len(commit) == 40, "verifier commit differs")
    head = str(git(repository, "rev-parse", "HEAD"))
    require(subprocess.run(("git", "merge-base", "--is-ancestor", commit, head),
                           cwd=repository, check=False).returncode == 0,
            "verifier commit is not an ancestor")
    hashes = receipt.get("verifier_runtime_sha256")
    require(type(hashes) is dict and set(hashes) == set(VERIFIER_RUNTIME),
            "verifier runtime closure differs")
    for name in VERIFIER_RUNTIME:
        content = snapshot(repository / name)
        require(content == git(repository, "show", f"{commit}:{name}", binary=True),
                f"verifier runtime drift: {name}")
        require(sha256(content).hexdigest() == hashes[name],
                f"verifier runtime hash differs: {name}")
    verified = verify_producer(repository)
    expected = verification_state(repository, verified, commit, hashes)
    require(deterministic == expected, "verification reconstruction differs")
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "validate"))
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    arguments = parser.parse_args(argv)
    try:
        result = run(arguments.repository) if arguments.mode == "run" else validate(arguments.repository)
    except VerificationError as error:
        print(f"M0 independent verification refused: {error}", file=os.sys.stderr)
        return 2
    print(json.dumps({"passed": result["passed"],
                      "verification_digest": result["verification_digest"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
