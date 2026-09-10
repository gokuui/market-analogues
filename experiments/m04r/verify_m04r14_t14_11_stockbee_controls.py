"""Independently verify Stockbee control selection, inference, and claim decisions."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
from itertools import product
import math
import json
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

from experiments.m04r import m04r14_t14_10_wf03_feasibility as base
from experiments.m04r import m04r14_t14_11_stockbee_controls as target


SCHEMA = "m04r14-t14-11-stockbee-controls-verification-v1"
TOLERANCE = 5e-13


class ControlVerificationError(RuntimeError): pass


def _now() -> str: return datetime.now(timezone.utc).isoformat()


def _sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20): digest.update(block)
    return digest.hexdigest()


def _valid(value: Mapping[str, Any], key: str = "result_digest", *, timing: bool = False) -> bool:
    omitted = {key, "created_at"} | ({"elapsed_seconds"} if timing else set())
    return value.get(key) == stable_hash({k: v for k, v in value.items() if k not in omitted})


def _deciles(values: np.ndarray) -> np.ndarray:
    ranks = pd.Series(values).rank(method="average").to_numpy(float)
    return np.clip(np.ceil(10 * ranks / len(values)), 1, 10).astype(np.uint8)


def _digest(contract: str, event: str, symbol: str) -> str:
    return sha256((contract + "|" + event + "|" + symbol).encode("utf-8")).hexdigest()


def _select(
    names: np.ndarray, deciles: np.ndarray, nonwinner: np.ndarray, event_symbol: str,
    event_deciles: np.ndarray, contract: str, event_id: str,
) -> tuple[list[int], str]:
    buckets: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for index in np.flatnonzero(nonwinner): buckets[tuple(int(x) for x in deciles[index])].append(int(index))
    key = tuple(int(x) for x in event_deciles); exact = buckets.get(key, [])
    if len(exact) >= 5: pool, tier = exact, "exact_all_four_deciles"
    else:
        pool = []
        for candidate in product(*[range(max(1, x - 1), min(10, x + 1) + 1) for x in key]):
            pool.extend(buckets.get(tuple(candidate), []))
        if len(pool) >= 5: tier = "within_one_bucket_all_four"
        else: pool, tier = list(np.flatnonzero(nonwinner)), "same_date_horizon_unmatched"
    pool = [index for index in pool if str(names[index]) != event_symbol]
    pool.sort(key=lambda index: (_digest(contract, event_id, str(names[index])), str(names[index])))
    return pool[:5], tier


def _bootstrap(values: Sequence[float], resamples: int, block: int, seed: int) -> tuple[float, float, float, float]:
    array = np.asarray(values, dtype=float); observed = float(array.mean()); centered = array - observed
    starts = np.arange(len(array) - block + 1); blocks = int(math.ceil(len(array) / block))
    rng = np.random.Generator(np.random.PCG64(seed)); exceed = 0; estimates = np.empty(resamples)
    for iteration in range(resamples):
        chosen = rng.choice(starts, size=blocks, replace=True)
        positions = np.concatenate([np.arange(start, start + block) for start in chosen])[:len(array)]
        exceed += bool(centered[positions].mean() >= observed); estimates[iteration] = array[positions].mean()
    interval = np.quantile(estimates, [.025, .975])
    return observed, (exceed + 1) / (resamples + 1), float(interval[0]), float(interval[1])


def _holm(pvalues: Mapping[str, float]) -> dict[str, tuple[float, bool]]:
    ordered = sorted(pvalues.items(), key=lambda item: (item[1], item[0])); running = 0.; result = {}
    for position, (name, value) in enumerate(ordered):
        running = max(running, (len(ordered) - position) * value); adjusted = min(1., running)
        result[name] = (adjusted, adjusted < .05)
    return result


def _assert_frame(expected: pd.DataFrame, observed: pd.DataFrame, label: str) -> None:
    if list(expected.columns) != list(observed.columns) or len(expected) != len(observed):
        raise ControlVerificationError(f"{label} schema/count differs")
    for column in expected:
        left, right = expected[column], observed[column]
        if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            a, b = left.to_numpy(float), right.to_numpy(float)
            if not np.array_equal(np.isnan(a), np.isnan(b)): raise ControlVerificationError(f"{label} missingness: {column}")
            finite = np.isfinite(a) & np.isfinite(b)
            if finite.any() and np.max(np.abs(a[finite] - b[finite])) > TOLERANCE:
                raise ControlVerificationError(f"{label} numeric difference: {column}")
        elif not left.fillna("<NA>").astype(str).equals(right.fillna("<NA>").astype(str)):
            raise ControlVerificationError(f"{label} value difference: {column}")


def execute(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    if subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=repository,
                      capture_output=True, text=True, check=True).stdout:
        raise ControlVerificationError("clean worktree required")
    started = perf_counter(); root = repository / target.OUTPUT_RELATIVE
    seal = base._read(root / "SEALED.json"); prereg = base._read(repository / target.PREREGISTRATION_RELATIVE)
    if not _valid(seal, timing=True) or not target._valid(prereg, "preregistration_digest") \
            or seal.get("file_manifest") != target._manifest(root):
        raise ControlVerificationError("control seals differ")
    controls = pd.read_parquet(root / "matched-controls.parquet")
    matches = pd.read_parquet(root / "event-matches.parquet")
    observed_inference = pd.read_parquet(root / "inference.parquet")
    observed_predictive = pd.read_parquet(root / "predictive-lift.parquet")
    event_frame = pd.read_parquet(repository / target.events_stage.OUTPUT_RELATIVE / "clustered-events.parquet")
    event_groups = event_frame.groupby(["start", "horizon_sessions"], sort=False).indices
    cache = repository / target.events_stage.risk_set.CACHE_RELATIVE
    risk = pd.concat([
        pd.read_parquet(cache / f"shard-{shard:02d}" / "risk-set.parquet", columns=list(target.RISK_COLUMNS))
        for shard in range(target.events_stage.risk_set.SHARDS)
    ], ignore_index=True).sort_values(["start", "horizon_sessions", "symbol"], kind="stable").reset_index(drop=True)
    control_cursor = match_cursor = 0
    full_month: dict[tuple[str, int, str, str, str], list[float]] = defaultdict(lambda: [0., 0., 0., 0.])
    matched_month: dict[tuple[str, int, str, str, str], list[float]] = defaultdict(lambda: [0., 0., 0.])
    predictive: dict[tuple[str, int, str, str], list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    totals: dict[tuple[str, int], list[int]] = defaultdict(lambda: [0, 0])
    tiers: dict[tuple[str, int, str, int], int] = defaultdict(int)
    contract = prereg["contract_digest"]
    for (start, horizon), group in risk.groupby(["start", "horizon_sessions"], sort=False):
        key = (pd.Timestamp(start), int(horizon)); positions = event_groups.get(key)
        date_events = event_frame.iloc[positions] if positions is not None else event_frame.iloc[:0]
        month = str(pd.Timestamp(start).to_period("M"))
        for population in ("broad", "investable"):
            current = group if population == "broad" else group.loc[group.investable]
            selected_events = date_events if population == "broad" else date_events.loc[date_events.investable]
            if current.empty: continue
            current = current.reset_index(drop=True); names = current.symbol.astype(str).to_numpy(object)
            deciles = np.column_stack([_deciles(current[column].to_numpy(float)) for column in target.MATCH_COLUMNS])
            winner = current.winner_25pct.to_numpy(bool); symbol_index = {name: i for i, name in enumerate(names)}
            exposure_values = {column: current[column].to_numpy(bool) for column in target.ALL_EXPOSURE_COLUMNS}
            for exposure in target.PRIMARY_EXPOSURES:
                for window in target.PRIMARY_WINDOWS:
                    column = f"{exposure}_{window}"; values = exposure_values[column]
                    state = predictive[(population, int(horizon), exposure, window)]
                    state[0] += len(values); state[1] += int(winner.sum()); state[2] += int(values.sum())
                    state[3] += int((values & winner).sum())
                    monthly = full_month[(population, int(horizon), exposure, window, month)]
                    monthly[2] += int(values[~winner].sum()); monthly[3] += int((~winner).sum())
            for event in selected_events.itertuples(index=False):
                index = symbol_index[str(event.symbol)]; expected, tier = _select(
                    names, deciles, ~winner, str(event.symbol), deciles[index], contract, str(event.event_id),
                )
                count = len(expected); full = count == 5; totals[(population, int(horizon))][0] += 1
                totals[(population, int(horizon))][1] += int(full); tiers[(population, int(horizon), tier, count)] += 1
                if match_cursor >= len(matches): raise ControlVerificationError("event-match rows truncated")
                observed_match = matches.iloc[match_cursor]; match_cursor += 1
                expected_match = [population, str(event.event_id), str(event.symbol), pd.Timestamp(start), int(horizon), tier, count, full]
                observed_values = [observed_match.population, observed_match.event_id, observed_match.event_symbol,
                                   observed_match.start, int(observed_match.horizon_sessions), observed_match.match_tier,
                                   int(observed_match.control_count), bool(observed_match.full_match)]
                if observed_values != expected_match: raise ControlVerificationError("event-match identity differs")
                for i, name in enumerate(target.MATCH_COLUMNS):
                    if int(observed_match[f"event_{name}_decile"]) != int(deciles[index, i]):
                        raise ControlVerificationError("event decile differs")
                for exposure in target.PRIMARY_EXPOSURES:
                    for window in target.PRIMARY_WINDOWS:
                        column = f"{exposure}_{window}"; event_value = float(exposure_values[column][index])
                        monthly = full_month[(population, int(horizon), exposure, window, month)]
                        monthly[0] += event_value; monthly[1] += 1
                        if full:
                            control_mean = float(exposure_values[column][expected].mean())
                            paired = matched_month[(population, int(horizon), exposure, window, month)]
                            paired[0] += event_value; paired[1] += control_mean; paired[2] += 1
                for rank, expected_index in enumerate(expected, 1):
                    if control_cursor >= len(controls): raise ControlVerificationError("control rows truncated")
                    observed = controls.iloc[control_cursor]; control_cursor += 1
                    control_symbol = str(names[expected_index]); control_deciles = deciles[expected_index]
                    identity = [population, str(event.event_id), str(event.symbol), control_symbol, pd.Timestamp(start),
                                int(horizon), rank, tier, _digest(contract, str(event.event_id), control_symbol)]
                    actual = [observed.population, observed.event_id, observed.event_symbol, observed.control_symbol,
                              observed.start, int(observed.horizon_sessions), int(observed.match_rank),
                              observed.match_tier, observed.selection_digest]
                    if actual != identity: raise ControlVerificationError("selected control identity/order differs")
                    distance = int(np.max(np.abs(deciles[index].astype(int) - control_deciles.astype(int))))
                    if int(observed.maximum_decile_distance) != distance: raise ControlVerificationError("distance differs")
                    for i, name in enumerate(target.MATCH_COLUMNS):
                        if int(observed[f"event_{name}_decile"]) != int(deciles[index, i]) \
                                or int(observed[f"control_{name}_decile"]) != int(control_deciles[i]):
                            raise ControlVerificationError("control decile differs")
                    for column in target.ALL_EXPOSURE_COLUMNS:
                        if bool(observed[column]) != bool(exposure_values[column][expected_index]):
                            raise ControlVerificationError("control exposure differs")
    if control_cursor != len(controls) or match_cursor != len(matches):
        raise ControlVerificationError("unexpected trailing match rows")

    expected_rows = []
    for population in ("broad", "investable"):
        for horizon in (21, 63):
            match_coverage = totals[(population, horizon)][1] / totals[(population, horizon)][0]
            for exposure in target.PRIMARY_EXPOSURES:
                for window in target.PRIMARY_WINDOWS:
                    for comparator in ("full_nonwinner_risk_set", "matched_nonwinner_controls"):
                        source = full_month if comparator == "full_nonwinner_risk_set" else matched_month
                        months = sorted(key[4] for key in source if key[:4] == (population, horizon, exposure, window))
                        differences = []; used_months = []; winner_sum = comparator_sum = observations = 0.
                        for month in months:
                            state = source[(population, horizon, exposure, window, month)]
                            if comparator == "full_nonwinner_risk_set":
                                if not state[1] or not state[3]: continue
                                winner_rate, comparator_rate = state[0] / state[1], state[2] / state[3]
                                winner_sum += state[0]; comparator_sum += state[2]; observations += state[1]
                            else:
                                if not state[2]: continue
                                winner_rate, comparator_rate = state[0] / state[2], state[1] / state[2]
                                winner_sum += state[0]; comparator_sum += state[1]; observations += state[2]
                            differences.append(winner_rate - comparator_rate)
                            used_months.append(month)
                        test_id = f"{population}|{horizon}|{exposure}|{window}|{comparator}"
                        seed = int(prereg["bootstrap_seed"]) + int(stable_hash(test_id)[:8], 16)
                        effect, pvalue, lower, upper = _bootstrap(
                            differences, int(prereg["bootstrap_resamples"]),
                            int(prereg["bootstrap_block_length_months"]), seed,
                        )
                        winner_prevalence = winner_sum / observations
                        denominator = (sum(full_month[(population, horizon, exposure, window, m)][3] for m in used_months)
                                       if comparator == "full_nonwinner_risk_set" else observations)
                        comparator_prevalence = comparator_sum / denominator
                        odds = ((winner_prevalence / (1 - winner_prevalence)) /
                                (comparator_prevalence / (1 - comparator_prevalence)))
                        expected_rows.append({
                            "test_id": test_id, "population": population, "horizon_sessions": horizon,
                            "exposure": exposure, "window": window, "comparator": comparator,
                            "event_observations": int(observations), "calendar_months": len(differences),
                            "winner_prevalence": winner_prevalence, "comparator_prevalence": comparator_prevalence,
                            "risk_difference": winner_prevalence - comparator_prevalence,
                            "mean_monthly_risk_difference": effect, "risk_ratio": winner_prevalence / comparator_prevalence,
                            "odds_ratio": odds, "bootstrap_positive_pvalue": pvalue,
                            "bootstrap_ci_lower": lower, "bootstrap_ci_upper": upper,
                            "matched_event_coverage": match_coverage,
                        })
    expected_inference = pd.DataFrame(expected_rows)
    adjusted = _holm(dict(zip(expected_inference.test_id, expected_inference.bootstrap_positive_pvalue)))
    expected_inference["holm_adjusted_pvalue"] = [adjusted[x][0] for x in expected_inference.test_id]
    expected_inference["holm_reject"] = [adjusted[x][1] for x in expected_inference.test_id]
    directions = {(r.horizon_sessions, r.exposure, r.window, r.comparator, r.population): r.risk_difference > 0
                  for r in expected_inference.itertuples(index=False)}
    expected_inference["same_direction_broad_investable"] = [
        directions[(r.horizon_sessions, r.exposure, r.window, r.comparator, "broad")]
        and directions[(r.horizon_sessions, r.exposure, r.window, r.comparator, "investable")]
        for r in expected_inference.itertuples(index=False)
    ]
    expected_inference["claim_supported"] = (
        (expected_inference.winner_prevalence >= .5) & (expected_inference.holm_adjusted_pvalue < .05)
        & (expected_inference.risk_ratio > 1) & (expected_inference.matched_event_coverage >= .9)
        & expected_inference.same_direction_broad_investable
    )
    expected_inference = expected_inference.sort_values(
        ["population", "horizon_sessions", "exposure", "window", "comparator"], kind="stable",
    ).reset_index(drop=True)
    _assert_frame(expected_inference, observed_inference, "inference")
    predictive_rows = []
    for key, state in sorted(predictive.items()):
        population, horizon, exposure, window = key; total, winners, exposed, exposed_winners = state
        base_rate, conditional = winners / total, exposed_winners / exposed
        predictive_rows.append({
            "population": population, "horizon_sessions": horizon, "exposure": exposure, "window": window,
            "risk_rows": total, "winner_rows": winners, "exposed_rows": exposed, "exposed_winner_rows": exposed_winners,
            "unconditional_winner_probability": base_rate, "winner_probability_given_exposure": conditional,
            "predictive_lift": conditional / base_rate,
        })
    _assert_frame(pd.DataFrame(predictive_rows), observed_predictive, "predictive lift")
    coverage = base._read(root / "coverage.json"); decision = base._read(root / "claim-decision.json")
    eligible = sum(x[0] for x in totals.values()); fully = sum(x[1] for x in totals.values())
    if not _valid(coverage) or coverage.get("matched_control_rows") != len(controls) \
            or coverage.get("event_match_rows") != len(matches) or coverage.get("eligible_event_populations") != eligible \
            or coverage.get("fully_matched_event_populations") != fully:
        raise ControlVerificationError("coverage differs")
    supported = int(expected_inference.claim_supported.sum())
    if not _valid(decision) or decision.get("primary_tests") != 32 \
            or decision.get("supported_primary_tests") != supported \
            or decision.get("any_primary_test_supported") != bool(supported):
        raise ControlVerificationError("claim decision differs")
    state = {
        "schema_version": SCHEMA, "status": "verified", "passed": True,
        "store_result_digest": seal["result_digest"], "store_result_sha256": _sha(root / "SEALED.json"),
        "verified_risk_rows": len(risk), "verified_event_match_rows": len(matches),
        "verified_matched_control_rows": len(controls), "verified_inference_rows": len(expected_inference),
        "gates": {
            "all_cross_sectional_deciles_reconstructed": True,
            "all_control_identities_and_order_reconstructed": True,
            "all_control_exposures_reconstructed": True,
            "all_monthly_inference_reconstructed": True,
            "holm_family_and_claim_decision_reconstructed": True,
            "all_physical_seals_valid": True,
        }, "production_promotion_authorized": False,
        "elapsed_seconds": perf_counter() - started, "created_at": _now(),
    }
    result = {**state, "verification_digest": stable_hash(state)}
    output = repository / target.VERIFICATION_RELATIVE
    if output.exists(): return base._read(output / "VERIFIED.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        target.smoke._atomic_json(temporary / "VERIFIED.json", result); os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True); raise
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--repository", type=Path, required=True)
    args = parser.parse_args(argv); print(json.dumps(execute(args.repository), indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
