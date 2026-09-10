"""Independently verify the reporting-only T14-12 primary effect report."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from market_analogues.types import stable_hash

from experiments.m04r import m04r14_t14_10_wf03_feasibility as io
from experiments.m04r import m04r14_t14_12_primary_report as target


SCHEMA = "m04r14-t14-12-primary-unsupported-report-verification-v1"
TOLERANCE = 5e-12


class PrimaryReportVerificationError(RuntimeError): pass


def _now() -> str: return datetime.now(timezone.utc).isoformat()


def _sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20): digest.update(block)
    return digest.hexdigest()


def _bootstrap(values: np.ndarray, resamples: int, block: int, seed: int) -> tuple[float, float, float, float]:
    observed = float(np.mean(values)); centered = values - observed; starts = np.arange(len(values) - block + 1)
    blocks = int(np.ceil(len(values) / block)); rng = np.random.Generator(np.random.PCG64(seed))
    null_count = 0; means = np.empty(resamples)
    for index in range(resamples):
        chosen = rng.choice(starts, size=blocks, replace=True)
        positions = np.concatenate([np.arange(start, start + block) for start in chosen])[:len(values)]
        null_count += bool(float(np.mean(centered[positions])) >= observed); means[index] = float(np.mean(values[positions]))
    lower, upper = np.quantile(means, [.025, .975])
    return observed, (null_count + 1) / (resamples + 1), float(lower), float(upper)


def _holm(values: Mapping[str, float]) -> dict[str, tuple[float, bool]]:
    ordered = sorted(values, key=lambda key: (values[key], key)); count = len(ordered); running = 0.; result = {}
    for rank, key in enumerate(ordered):
        running = max(running, min(1., (count - rank) * float(values[key]))); result[key] = (running, running < .05)
    return result


def _assert_frame(expected: pd.DataFrame, observed: pd.DataFrame, label: str) -> None:
    if list(expected.columns) != list(observed.columns) or len(expected) != len(observed):
        raise PrimaryReportVerificationError(f"{label} schema/count differs")
    for column in expected:
        left, right = expected[column], observed[column]
        if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            a, b = left.to_numpy(float), right.to_numpy(float)
            if not np.array_equal(np.isnan(a), np.isnan(b)): raise PrimaryReportVerificationError(f"{label} missingness differs: {column}")
            finite = np.isfinite(a) & np.isfinite(b)
            if finite.any() and np.max(np.abs(a[finite] - b[finite])) > TOLERANCE:
                raise PrimaryReportVerificationError(f"{label} numeric differs: {column}")
        elif not left.astype(str).equals(right.astype(str)): raise PrimaryReportVerificationError(f"{label} differs: {column}")


def execute(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    if subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=repository,
                      capture_output=True, text=True, check=True).stdout: raise PrimaryReportVerificationError("clean worktree required")
    started = perf_counter(); prereg, _ = target.validate_preregistration(repository); root = repository / target.OUTPUT_RELATIVE
    seal = io._read(root / "SEALED.json")
    if not target._valid(seal, timing=True) or seal.get("file_manifest") != target._manifest(root, target.OUTPUT_FILES):
        raise PrimaryReportVerificationError("primary report seal differs")
    pairs = target._read_pairs(repository); coverage = pd.read_parquet(repository / target.outcomes.OUTPUT_RELATIVE / "coverage.parquet")
    rows = []; eras = []
    for population in ("broad", "investable"):
        for signal in ("bullish_range_expansion_4pct", "up_close_4pct"):
            for horizon in target.PRIMARY_HORIZONS:
                base = pairs.loc[pairs.population.eq(population) & pairs.signal_name.eq(signal) & pairs.horizon_sessions.eq(horizon)]
                rate = float(coverage.loc[coverage.population.eq(population) & coverage.signal_name.eq(signal)
                    & coverage.horizon_sessions.eq(horizon), "paired_complete_fraction"].iloc[0]); complete = base.loc[base.paired_complete]
                for metric in target.PRIMARY_METRICS:
                    column = f"paired_{metric}_difference"; test_id = f"{population}|{signal}|{horizon}|{metric}"
                    monthly = complete.assign(month=complete.signal_date.dt.to_period("M")).groupby("month", sort=True)[column].mean().to_numpy(float)
                    seed = int(prereg["base_seed"]) + int(sha256(test_id.encode()).hexdigest()[:8], 16)
                    effect, pvalue, lower, upper = _bootstrap(monthly, int(prereg["bootstrap_resamples"]), int(prereg["block_length_months"]), seed)
                    era_means = complete.assign(era=complete.signal_date.dt.year.map(target._era)).groupby("era", sort=True)[column].mean()
                    for era, value in era_means.items(): eras.append({"test_id": test_id, "era": era, "effect": float(value)})
                    rows.append({"test_id": test_id, "population": population, "signal_name": signal,
                        "horizon_sessions": horizon, "metric": metric, "event_rows": len(base), "complete_rows": len(complete),
                        "coverage": rate, "calendar_months": len(monthly), "effect": effect,
                        "bootstrap_positive_pvalue": pvalue, "bootstrap_lower": lower, "bootstrap_upper": upper,
                        "positive_eras": int((era_means > 0).sum())})
    effects = pd.DataFrame(rows); adjusted = _holm(dict(zip(effects.test_id, effects.bootstrap_positive_pvalue)))
    effects["holm_adjusted_pvalue"] = [adjusted[value][0] for value in effects.test_id]
    effects["holm_reject"] = [adjusted[value][1] for value in effects.test_id]
    directions = effects.groupby(["signal_name", "horizon_sessions", "metric"]).effect.apply(lambda values: bool((values > 0).all()))
    effects["same_positive_direction_both_populations"] = [directions.loc[(r.signal_name, r.horizon_sessions, r.metric)] for r in effects.itertuples()]
    effects["statistical_criteria_pass"] = (effects.holm_adjusted_pvalue < .05) & (effects.bootstrap_lower > 0) \
        & (effects.positive_eras >= 4) & effects.same_positive_direction_both_populations
    effects["coverage_gate_pass"] = effects.coverage >= .90
    effects["claim_supported"] = effects.statistical_criteria_pass & effects.coverage_gate_pass
    _assert_frame(effects, pd.read_parquet(root / "primary-effects.parquet"), "primary effects")
    _assert_frame(pd.DataFrame(eras), pd.read_parquet(root / "era-effects.parquet"), "era effects")
    authorization = []
    for signal in effects.signal_name.unique():
        for horizon in target.PRIMARY_HORIZONS:
            selected = effects.loc[effects.signal_name.eq(signal) & effects.horizon_sessions.eq(horizon)]
            authorization.append({"signal_name": signal, "horizon_sessions": horizon,
                                  "all_four_population_metric_cells_supported": bool(selected.claim_supported.all())})
    feature = any(value["all_four_population_metric_cells_supported"] for value in authorization)
    decision = io._read(root / "decision.json")
    if not target._valid(decision) or decision.get("coverage_failed_cells") != int((~effects.coverage_gate_pass).sum()) \
            or decision.get("statistical_criteria_passed_cells") != int(effects.statistical_criteria_pass.sum()) \
            or decision.get("supported_cells") != int(effects.claim_supported.sum()) \
            or decision.get("authorization_cells") != authorization or decision.get("retrieval_feature_authorized") != feature \
            or decision.get("coverage_failure_overridden") is not False:
        raise PrimaryReportVerificationError("primary report decision differs")
    state = {"schema_version": SCHEMA, "status": "verified", "passed": True,
        "store_result_digest": seal["result_digest"], "store_result_sha256": _sha(root / "SEALED.json"),
        "verified_primary_effect_rows": len(effects), "verified_era_effect_rows": len(eras),
        "coverage_failed_cells": int((~effects.coverage_gate_pass).sum()),
        "statistical_criteria_passed_cells": int(effects.statistical_criteria_pass.sum()),
        "supported_cells": int(effects.claim_supported.sum()), "retrieval_feature_authorized": feature,
        "gates": {"all_monthly_effects_reconstructed": True, "independent_bootstrap_reconstructed": True,
                  "independent_holm_family_reconstructed": True, "all_era_directions_reconstructed": True,
                  "coverage_failure_not_overridden": True}, "reporting_only": True,
        "production_promotion_authorized": False, "elapsed_seconds": perf_counter() - started, "created_at": _now()}
    result = {**state, "verification_digest": stable_hash(state)}; output = repository / target.VERIFICATION_RELATIVE
    if output.exists(): return io._read(output / "VERIFIED.json")
    output.parent.mkdir(parents=True, exist_ok=True); temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try: target.smoke._atomic_json(temporary / "VERIFIED.json", result); os.replace(temporary, output)
    except BaseException: shutil.rmtree(temporary, ignore_errors=True); raise
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--repository", type=Path, required=True)
    args = parser.parse_args(argv); print(json.dumps(execute(args.repository), indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
