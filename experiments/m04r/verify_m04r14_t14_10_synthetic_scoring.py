"""Verify all WF-02 scoring formulas without opening a real query outcome."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from html import escape
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Sequence

import numpy as np
from scipy.stats import binomtest, norm

from market_analogues.adapters import file_fingerprint
from market_analogues.types import stable_hash
from market_analogues.walk_forward_scoring import (
    WalkForwardScoringError,
    abstention_reasons,
    brier_skill,
    calendar_month_mean_losses,
    central_interval_score,
    diebold_mariano_hac_lower_pvalue,
    effective_sample_size,
    equal_count_calibration,
    expanding_frequency,
    holm_adjust,
    interval_coverage_tests,
    log_loss,
    moving_block_bootstrap_lower_pvalue,
    multiclass_brier,
    pinball_loss,
    pointwise_weighted_median,
    regime_frequency,
    risk_coverage_curve,
    smoothed_class_probabilities,
    weighted_inverted_cdf,
)


SCHEMA = "m04r14-t14-10-walk-forward-synthetic-scoring-verification-v1"
SPEC = Path("config/m04r14-t14-10-synthetic-scoring-spec.json")
CONTRACT = Path("config/m04r14-t14-10-walk-forward-contract.json")
REGISTRY_VERIFICATION = Path(
    "config/data/analogues/m04r14/t14-10-walk-forward-query-registry-v1-verification/VERIFIED.json"
)
DEFAULT_OUTPUT = Path(
    "config/data/analogues/m04r14/t14-10-walk-forward-synthetic-scoring-v1"
)
CLASSES = ("favorable_first", "adverse_first", "no_touch")


class SyntheticScoringVerificationError(RuntimeError):
    pass


def _read(path: Path) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        raise SyntheticScoringVerificationError(f"regular JSON required: {path}")
    raw = path.read_bytes()

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise SyntheticScoringVerificationError(f"duplicate JSON key: {path}:{key}")
            result[key] = value
        return result

    value = json.loads(
        raw, object_pairs_hook=pairs,
        parse_constant=lambda item: (_ for _ in ()).throw(
            SyntheticScoringVerificationError(f"non-finite JSON: {path}:{item}")
        ),
    )
    if type(value) is not dict:
        raise SyntheticScoringVerificationError(f"JSON object required: {path}")
    return value, raw


def _validate_boundary(repository: Path) -> tuple[dict[str, Any], str]:
    spec, _ = _read(repository / SPEC)
    contract, contract_raw = _read(repository / CONTRACT)
    registry, registry_raw = _read(repository / REGISTRY_VERIFICATION)
    state = {key: value for key, value in spec.items() if key != "spec_digest"}
    if not all((
        spec.get("schema_version") == "m04r14-t14-10-synthetic-scoring-spec-v1",
        spec.get("status") == "frozen_before_real_historical_query_outcomes",
        spec.get("spec_digest") == stable_hash(state),
        spec.get("walk_forward_contract_digest") == contract.get("contract_digest"),
        spec.get("walk_forward_contract_sha256") == sha256(contract_raw).hexdigest(),
        spec.get("registry_verification_digest") == registry.get("result_digest"),
        spec.get("registry_verification_sha256") == sha256(registry_raw).hexdigest(),
        registry.get("passed") is True, registry.get("wf02_authorized") is True,
        registry.get("historical_walk_forward_query_outcomes_opened") is False,
        spec.get("real_forward_outcomes_accessed") is False,
        spec.get("final_period_result_opened") is False,
    )):
        raise SyntheticScoringVerificationError("synthetic spec or unopened boundary differs")
    implementation = spec.get("implementation", {})
    for path, expected in implementation.get("files", {}).items():
        if file_fingerprint(repository / path) != expected:
            raise SyntheticScoringVerificationError(f"synthetic implementation differs: {path}")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True,
        capture_output=True, check=True,
    ).stdout.strip()
    parents = subprocess.run(
        ["git", "rev-list", "--parents", "-n", "1", head], cwd=repository,
        text=True, capture_output=True, check=True,
    ).stdout.split()
    if len(parents) != 2 or parents[1] != implementation.get("implementation_h0"):
        raise SyntheticScoringVerificationError("synthetic spec is not sole child of H0")
    return spec, contract["contract_digest"]


def _run_cases() -> tuple[int, list[str]]:
    checks = 0
    names: list[str] = []

    def check(name: str, condition: bool) -> None:
        nonlocal checks
        checks += 1
        names.append(name)
        if not condition:
            raise SyntheticScoringVerificationError(f"synthetic oracle failed: {name}")

    labels = ["favorable_first", "no_touch", "favorable_first"]
    weights = np.array([1.0, 2.0, .5])
    probability = smoothed_class_probabilities(labels, weights, CLASSES)
    check("dirichlet_half", np.array_equal(probability, np.array([2., .5, 2.5]) / 5))
    check("zero_count_uniform", np.array_equal(expanding_frequency([], CLASSES), np.repeat(1 / 3, 3)))
    check("positive_probability", bool((probability > 0).all() and probability.sum() == 1))
    scaled = smoothed_class_probabilities(labels, weights * 10, CLASSES, alpha=5)
    check("joint_weight_alpha_scale", np.allclose(probability, scaled, rtol=0, atol=0))
    duplicated = smoothed_class_probabilities(labels * 2, np.tile(weights / 2, 2), CLASSES)
    check("duplicate_split_invariance", np.allclose(probability, duplicated, rtol=0, atol=1e-16))
    check("effective_rows", effective_sample_size(weights) == weights.sum() ** 2 / (weights @ weights))
    manual_brier = float((probability[0] - 1) ** 2 + probability[1] ** 2 + probability[2] ** 2)
    check("multiclass_brier", multiclass_brier(probability, 0) == manual_brier)
    check("multiclass_log", log_loss(probability, 0) == -math.log(probability[0]))
    check("brier_skill", brier_skill(.18, .24) == .25)

    quantiles = weighted_inverted_cdf([4, 1, 2, 2], [1, 1, 2, 1], [0, .1, .5, .9, 1])
    check("weighted_inverted_cdf", np.array_equal(quantiles, [1, 1, 2, 4, 4]))
    check("quantile_monotonic", bool((np.diff(quantiles) >= 0).all()))
    path = pointwise_weighted_median([[1, np.nan, 4], [2, 3, np.nan]], [1, 2])
    check("path_missing_measure", np.array_equal(path, [2, 3, 4]))
    check("pinball_below", pinball_loss(1, 2, .25) == .75)
    check("pinball_above", pinball_loss(3, 2, .25) == .25)
    check("interval_inside", central_interval_score(2, 1, 3, .8) == 2)
    check("interval_tail", abs(central_interval_score(0, 1, 3, .8) - 12) < 1e-12)

    forecasts = np.linspace(.05, .95, 60)
    outcomes = np.array([0] * 30 + [1] * 30)
    bins, ece = equal_count_calibration(forecasts, outcomes, bins=10, minimum_rows=30)
    manual_ece = (abs(forecasts[:30].mean()) + abs(forecasts[30:].mean() - 1)) / 2
    check("reliability_bin_reduction", len(bins) == 2 and bins.rows.tolist() == [30, 30])
    check("equal_count_ece", abs(ece - manual_ece) < 1e-15)
    curve = risk_coverage_curve([.4, .1, .3, .2], [4, 1, 3, 2], coverage_levels=[.5, 1])
    check("risk_coverage", np.allclose(curve.mean_loss, [.15, .25]))

    history = np.arange(250, dtype=float)
    threshold = np.quantile(history, .95, method="linear")
    warmup = abstention_reasons(
        eligible_primary_rows=9, nearest_distance=10_000,
        prior_nearest_distances=history[:-1], favorable_prefix_probabilities=[.4, .61],
    )
    check("novelty_warmup", "historically_novel_query" not in warmup)
    check("insufficient_rows", "insufficient_effective_sample_size" in warmup)
    check("instability_strict", "unstable_neighborhood" in warmup)
    boundary = abstention_reasons(
        eligible_primary_rows=10, nearest_distance=threshold,
        prior_nearest_distances=history, favorable_prefix_probabilities=[.4, .6],
    )
    check("novelty_strict_boundary", "historically_novel_query" not in boundary)
    novel = abstention_reasons(
        eligible_primary_rows=10, nearest_distance=np.nextafter(threshold, np.inf),
        prior_nearest_distances=history, favorable_prefix_probabilities=[.4, .6],
    )
    check("novelty_above_boundary", "historically_novel_query" in novel)

    prior = [CLASSES[index % 3] for index in range(60)]
    regimes = ["x"] * 50 + ["y"] * 10
    _, fallback50 = regime_frequency(prior, regimes, "x", CLASSES)
    _, fallback10 = regime_frequency(prior, regimes, "y", CLASSES)
    check("regime_minimum_50", fallback50 is False and fallback10 is True)
    monthly = calendar_month_mean_losses(
        ["2020-01-02", "2020-01-30", "2020-02-03"], [.1, .3, .4], [.2, .4, .5],
    )
    check("calendar_month_unit", monthly.month.tolist() == ["2020-01", "2020-02"])
    check("calendar_month_pairing", np.allclose(monthly.loss_difference, [-.1, -.1]))

    differences = np.array([-.4, -.2, .1, -.3, 0, -.1])
    observed, bootstrap_p = moving_block_bootstrap_lower_pvalue(
        differences, resamples=200, block_length=3, seed=7,
    )
    centered = differences - differences.mean()
    rng = np.random.Generator(np.random.PCG64(7))
    reference = []
    for _ in range(200):
        starts = rng.choice(np.arange(4), size=2, replace=True)
        reference.append(np.concatenate([centered[start:start + 3] for start in starts]).mean())
    expected_p = (sum(value <= observed for value in reference) + 1) / 201
    check("bootstrap_observed", observed == differences.mean())
    check("bootstrap_seeded_oracle", bootstrap_p == expected_p)

    dm_values = np.array([-.4, -.2, .1, -.3, 0, -.1, -.2, .05])
    dm_mean, dm_stat, dm_p = diebold_mariano_hac_lower_pvalue(dm_values, lags=3)
    centered = dm_values - dm_values.mean()
    long_run = centered @ centered / len(dm_values) + sum(
        2 * (1 - lag / 4) * centered[lag:] @ centered[:-lag] / len(dm_values)
        for lag in range(1, 4)
    )
    expected_stat = dm_values.mean() / math.sqrt(max(0, long_run) / len(dm_values))
    check("dm_mean", dm_mean == dm_values.mean())
    check("newey_west_hac", abs(dm_stat - expected_stat) < 1e-15)
    check("dm_lower_tail", dm_p == norm.cdf(expected_stat))
    check("holm", holm_adjust({"a": .01, "b": .03, "c": .2}) == {
        "a": (.03, True), "b": (.06, False), "c": (.2, False),
    })

    hits = np.array(([1] * 8 + [0] * 2) * 10)
    coverage = interval_coverage_tests(hits, nominal_coverage=.8, dynamic_lags=3)
    check("exact_marginal_binomial", coverage.exact_marginal_pvalue == binomtest(80, 100, .8).pvalue)
    check("christoffersen_finite", 0 <= coverage.christoffersen_independence_pvalue <= 1)
    check("dynamic_binary_finite", 0 <= coverage.dynamic_binary_pvalue <= 1)

    rejected = 0
    invalid_calls = [
        lambda: smoothed_class_probabilities(["unknown"], [1], CLASSES),
        lambda: weighted_inverted_cdf([1, np.nan], [1, 1], [.5]),
        lambda: moving_block_bootstrap_lower_pvalue([1, 2]),
        lambda: interval_coverage_tests([1, 2, 1, 0], nominal_coverage=.8),
        lambda: log_loss([1, 0], 0),
        lambda: central_interval_score(1, 2, 0, .8),
    ]
    for call in invalid_calls:
        try:
            call()
        except WalkForwardScoringError:
            rejected += 1
    check("adversarial_rejection", rejected == len(invalid_calls))
    return checks, names


def execute(repository: Path, output: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    if subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=repository,
        text=True, capture_output=True, check=True,
    ).stdout:
        raise SyntheticScoringVerificationError("synthetic verification requires a clean worktree")
    spec, contract_digest = _validate_boundary(repository)
    checks, names = _run_cases()
    result: dict[str, Any] = {
        "schema_version": SCHEMA, "status": "verified", "passed": True,
        "walk_forward_contract_digest": contract_digest,
        "synthetic_spec_digest": spec["spec_digest"],
        "synthetic_checks": checks, "check_names": names,
        "formula_oracle_exact": True, "transform_invariance_passed": True,
        "adversarial_rejection_passed": True, "statistical_reference_checks_passed": True,
        "real_forward_outcomes_accessed": False,
        "historical_query_retrieval_opened": False,
        "historical_walk_forward_query_outcomes_opened": False,
        "final_period_result_opened": False,
        "wf03_authorized": True, "production_promotion_authorized": False,
    }
    result["result_digest"] = stable_hash(result)
    output = output.resolve()
    if output.exists() or output.is_symlink():
        raise SyntheticScoringVerificationError("synthetic verification root already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        raw = json.dumps(result, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
        descriptor = os.open(temporary / "VERIFIED.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw); handle.flush(); os.fsync(handle.fileno())
        items = "".join(f"<li>{escape(name)}</li>" for name in names)
        report = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>WF-02 synthetic scoring verification</title><style>body{{font-family:system-ui;max-width:1000px;margin:2rem auto;line-height:1.5}}.pass{{color:#075}}code{{overflow-wrap:anywhere}}</style></head><body><h1>WF-02 synthetic scoring: <span class="pass">PASS</span></h1><p>{checks} independent formula, boundary, invariance and adversarial checks pass. No real query future or retrieval result was accessed.</p><p>Verification digest: <code>{result['result_digest']}</code></p><ul>{items}</ul></body></html>"""
        descriptor = os.open(temporary / "index.html", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(report.encode()); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, output)
    except BaseException:
        for path in temporary.glob("*"):
            path.unlink()
        temporary.rmdir()
        raise
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args(argv)
    result = execute(args.repository, args.output_root or args.repository / DEFAULT_OUTPUT)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
