"""Build deterministic Stockbee matched controls and dependence-aware inference."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
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
import pyarrow as pa
import pyarrow.parquet as pq

from market_analogues.stockbee_controls import (
    bucket_members, control_order_digest, cross_sectional_deciles,
    moving_block_positive_inference, select_control_indices,
)
from market_analogues.types import stable_hash
from market_analogues.walk_forward_scoring import holm_adjust

from experiments.m04r import m04r14_t14_09_outcome_smoke as smoke
from experiments.m04r import m04r14_t14_10_wf03_feasibility as base
from experiments.m04r import m04r14_t14_11_stockbee_events as events_stage


SCHEMA = "m04r14-t14-11-stockbee-controls-v1"
PREREGISTRATION_RELATIVE = Path("experiments/m04r/m04r14_t14_11_stockbee_controls_v1_preregistered.json")
OUTPUT_RELATIVE = Path("config/data/analogues/m04r14/t14-11-stockbee-controls-v1")
VERIFICATION_RELATIVE = Path("config/data/analogues/m04r14/t14-11-stockbee-controls-v1-verification")
OUTPUT_FILES = (
    "matched-controls.parquet", "event-matches.parquet", "inference.parquet",
    "predictive-lift.parquet", "coverage.json", "claim-decision.json", "index.html",
)
MATCH_COLUMNS = (
    "prior_return_63", "prior_volatility_20", "start_close", "prior_median_dollar_volume_20",
)
PRIMARY_EXPOSURES = ("up_close_4pct", "bullish_range_expansion_4pct")
PRIMARY_WINDOWS = ("start_day", "first_5_sessions")
ALL_EXPOSURE_COLUMNS = tuple(
    f"{exposure}_{window}" for exposure in events_stage.EXPOSURES for window in events_stage.WINDOWS
)
RISK_COLUMNS = (
    "symbol", "start", "horizon_sessions", "winner_25pct", "investable", *MATCH_COLUMNS,
    *ALL_EXPOSURE_COLUMNS,
)
RUNTIME_FILES = (
    "experiments/m04r/m04r14_t14_11_stockbee_controls.py",
    "experiments/m04r/verify_m04r14_t14_11_stockbee_controls.py",
    "experiments/m04r/m04r14_t14_11_stockbee_events.py",
    "src/market_analogues/stockbee_controls.py",
    "src/market_analogues/walk_forward_scoring.py",
    "config/m04r14-t14-11-stockbee-contract.json", "pyproject.toml",
)


class ControlStudyError(RuntimeError): pass


def _now() -> str: return datetime.now(timezone.utc).isoformat()


def _git(repository: Path, *args: str, raw: bool = False) -> str | bytes:
    result = subprocess.run(["git", *args], cwd=repository, capture_output=True, text=not raw, check=False)
    if result.returncode:
        error = result.stderr if not raw else result.stderr.decode(errors="replace")
        raise ControlStudyError(error.strip())
    return result.stdout if raw else result.stdout.strip()


def _sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20): digest.update(block)
    return digest.hexdigest()


def _seal(value: Mapping[str, Any], key: str = "result_digest", *, timing: bool = False) -> dict[str, Any]:
    omitted = {"elapsed_seconds"} if timing else set(); result = dict(value)
    result[key] = stable_hash({k: v for k, v in result.items() if k not in omitted})
    result["created_at"] = _now(); return result


def _valid(value: Mapping[str, Any], key: str = "result_digest", *, timing: bool = False) -> bool:
    omitted = {key, "created_at"} | ({"elapsed_seconds"} if timing else set())
    return value.get(key) == stable_hash({k: v for k, v in value.items() if k not in omitted})


def _manifest(root: Path) -> list[dict[str, Any]]:
    return [{"path": name, "bytes": (root / name).stat().st_size, "sha256": _sha(root / name)} for name in OUTPUT_FILES]


def _verified_events(repository: Path) -> dict[str, str]:
    store_path = repository / events_stage.OUTPUT_RELATIVE / "SEALED.json"
    verify_path = repository / events_stage.VERIFICATION_RELATIVE / "VERIFIED.json"
    store, receipt = base._read(store_path), base._read(verify_path)
    if not events_stage._valid(store, timing=True) or receipt.get("passed") is not True \
            or receipt.get("store_result_digest") != store.get("result_digest") \
            or receipt.get("matched_controls_constructed") is not False:
        raise ControlStudyError("verified event boundary differs")
    return {
        "event_store_result_digest": store["result_digest"], "event_store_sha256": _sha(store_path),
        "event_verification_digest": receipt["verification_digest"], "event_verification_sha256": _sha(verify_path),
    }


def _runtime_manifest(repository: Path, head: str) -> dict[str, str]:
    tracked = set(str(_git(repository, "ls-tree", "-r", "--name-only", head)).splitlines())
    if any(name not in tracked for name in RUNTIME_FILES): raise ControlStudyError("runtime file is uncommitted")
    return {name: sha256(_git(repository, "show", f"{head}:{name}", raw=True)).hexdigest() for name in RUNTIME_FILES}


def build_preregistration(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    if _git(repository, "status", "--porcelain", "--untracked-files=all"):
        raise ControlStudyError("clean worktree required")
    if (repository / OUTPUT_RELATIVE).exists(): raise ControlStudyError("control output already exists")
    h0 = str(_git(repository, "rev-parse", "HEAD")); contract = events_stage.risk_set._contract(repository)
    return _seal({
        "schema_version": SCHEMA, "status": "frozen_before_matched_control_selection_and_inference",
        "implementation_h0": h0, "runtime_files": _runtime_manifest(repository, h0),
        "verified_inputs": _verified_events(repository), "contract_digest": contract["contract_digest"],
        "decile_rule": "within_each_population_start_date_horizon_average_rank_then_ceil_10_rank_divided_by_n",
        "investable_deciles_use_investable_cross_section_only": True,
        "selection_serialization": "utf8(contract_digest|event_id|candidate_symbol)",
        "without_replacement_interpretation": "five_distinct_controls_within_event;reuse_across_events_allowed",
        "relaxation_rule": "use_first_tier_containing_at_least_five_candidates_else_final_tier_shortfall",
        "primary_matched_estimand": "only_events_with_exactly_five_controls",
        "partial_and_unmatched_events_retained_in_coverage": True,
        "full_comparator_monthly_estimand": "clustered_winner_event_prevalence_minus_unclustered_nonwinner_risk_prevalence_within_calendar_month",
        "matched_comparator_monthly_estimand": "mean_event_exposure_minus_mean_of_five_selected_controls_within_calendar_month",
        "bootstrap_tail": "one_sided_positive_effect_with_centered_null",
        "bootstrap_confidence_interval": "uncentered_moving_block_percentile_2.5_97.5",
        "bootstrap_resamples": int(contract["inference"]["paired_moving_block_bootstrap_resamples"]),
        "bootstrap_seed": int(contract["inference"]["seed"]),
        "bootstrap_block_length_months": int(contract["inference"]["block_length_months"]),
        "holm_family_size": 32, "real_matched_results_accessed": False,
        "production_promotion_authorized": False,
    }, "preregistration_digest")


def _sole_child(repository: Path, raw: bytes, h0: str) -> str:
    found = []
    for line in str(_git(repository, "rev-list", "--all", "--children")).splitlines():
        values = line.split()
        if not values or values[0] != h0: continue
        for child in values[1:]:
            parents = str(_git(repository, "rev-list", "--parents", "-n", "1", child)).split()
            changed = str(_git(repository, "diff-tree", "--no-commit-id", "--name-only", "-r", child)).splitlines()
            if parents == [child, h0] and changed == [PREREGISTRATION_RELATIVE.as_posix()] \
                    and _git(repository, "show", f"{child}:{PREREGISTRATION_RELATIVE}", raw=True) == raw:
                found.append(child)
    if len(set(found)) != 1: raise ControlStudyError("expected exact preregistration-only child")
    return found[0]


def validate_preregistration(repository: Path) -> tuple[dict[str, Any], str]:
    if _git(repository, "status", "--porcelain", "--untracked-files=all"):
        raise ControlStudyError("clean worktree required")
    path = repository / PREREGISTRATION_RELATIVE; raw = path.read_bytes(); prereg = base._read(path)
    if not _valid(prereg, "preregistration_digest") or prereg.get("schema_version") != SCHEMA:
        raise ControlStudyError("control preregistration differs")
    h1 = _sole_child(repository, raw, str(prereg["implementation_h0"]))
    if subprocess.run(["git", "merge-base", "--is-ancestor", h1, "HEAD"], cwd=repository).returncode:
        raise ControlStudyError("HEAD does not descend from preregistration")
    if prereg["verified_inputs"] != _verified_events(repository): raise ControlStudyError("event input drifted")
    for name, expected in prereg["runtime_files"].items():
        if _sha(repository / name) != expected: raise ControlStudyError(f"runtime drifted: {name}")
    return prereg, h1


class _ParquetSink:
    def __init__(self, path: Path) -> None:
        self.path, self.writer, self.rows = path, None, 0

    def write(self, rows: list[dict[str, Any]]) -> None:
        if not rows: return
        table = pa.Table.from_pylist(rows)
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.path, table.schema, compression="zstd")
        self.writer.write_table(table); self.rows += len(rows); rows.clear()

    def close(self) -> None:
        if self.writer is None: raise ControlStudyError(f"no rows written: {self.path.name}")
        self.writer.close()


def _risk_frame(repository: Path) -> pd.DataFrame:
    cache = repository / events_stage.risk_set.CACHE_RELATIVE
    frames = [pd.read_parquet(cache / f"shard-{shard:02d}" / "risk-set.parquet", columns=list(RISK_COLUMNS))
              for shard in range(events_stage.risk_set.SHARDS)]
    frame = pd.concat(frames, ignore_index=True); del frames
    frame["symbol"] = frame.symbol.astype("category")
    return frame.sort_values(["start", "horizon_sessions", "symbol"], kind="stable").reset_index(drop=True)


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def _odds_ratio(a: float, b: float) -> float:
    if not 0 < a < 1 or not 0 < b < 1: return 0.0
    return float((a / (1 - a)) / (b / (1 - b)))


def _html(inference: pd.DataFrame, coverage: Mapping[str, Any], decision: Mapping[str, Any]) -> str:
    rows = "".join(
        f"<tr><td>{r.population}</td><td>{r.horizon_sessions}</td><td>{r.exposure}</td>"
        f"<td>{r.window}</td><td>{r.comparator}</td><td>{100*r.winner_prevalence:.2f}%</td>"
        f"<td>{100*r.comparator_prevalence:.2f}%</td><td>{r.risk_ratio:.2f}</td>"
        f"<td>{r.holm_adjusted_pvalue:.4g}</td><td>{'yes' if r.claim_supported else 'no'}</td></tr>"
        for r in inference.itertuples(index=False)
    )
    return f"""<!doctype html><html><head><meta charset=\"utf-8\"><title>Stockbee controlled study</title><style>body{{font-family:system-ui;max-width:1300px;margin:2rem auto}}table{{border-collapse:collapse;width:100%;font-size:.86rem}}th,td{{padding:.35rem;border-bottom:1px solid #ddd;text-align:left}}.note{{background:#eef5ff;padding:.8rem}}.warn{{background:#fff4dc;padding:.8rem}}</style></head><body><h1>Stockbee 25% / 4% controlled study</h1><p class=\"note\">{coverage['fully_matched_event_populations']:,} of {coverage['eligible_event_populations']:,} event-population rows received five deterministic same-date controls ({100*coverage['full_match_coverage']:.2f}%).</p><p class=\"warn\">This is an observational replication, not a trading strategy or proof of causation. The NASDAQ archive is not a point-in-time Compustat universe.</p><p>Supported primary cells: {decision['supported_primary_tests']} of {decision['primary_tests']}. Overall primary evidence: <b>{'supported' if decision['any_primary_test_supported'] else 'not supported'}</b>.</p><table><thead><tr><th>Population</th><th>Horizon</th><th>Exposure</th><th>Window</th><th>Comparator</th><th>Winner</th><th>Comparator</th><th>Risk ratio</th><th>Holm p</th><th>Supported</th></tr></thead><tbody>{rows}</tbody></table></body></html>"""


def execute(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True); prereg, h1 = validate_preregistration(repository)
    root = repository / OUTPUT_RELATIVE
    if root.exists():
        seal = base._read(root / "SEALED.json")
        if not _valid(seal, timing=True) or seal.get("file_manifest") != _manifest(root):
            raise ControlStudyError("existing output differs")
        return seal
    started = perf_counter(); contract_digest = prereg["contract_digest"]
    event_frame = pd.read_parquet(repository / events_stage.OUTPUT_RELATIVE / "clustered-events.parquet")
    event_groups = event_frame.groupby(["start", "horizon_sessions"], sort=False).indices
    risk = _risk_frame(repository)
    root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{root.name}.", dir=root.parent))
    controls_sink = _ParquetSink(temporary / "matched-controls.parquet")
    matches_sink = _ParquetSink(temporary / "event-matches.parquet")
    control_rows: list[dict[str, Any]] = []; match_rows: list[dict[str, Any]] = []
    coverage_counts: dict[tuple[str, int, str, int], int] = defaultdict(int)
    event_totals: dict[tuple[str, int], list[int]] = defaultdict(lambda: [0, 0])
    full_month: dict[tuple[str, int, str, str, str], list[float]] = defaultdict(lambda: [0., 0., 0., 0.])
    matched_month: dict[tuple[str, int, str, str, str], list[float]] = defaultdict(lambda: [0., 0., 0.])
    predictive: dict[tuple[str, int, str, str], list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    seen_events = {"broad": 0, "investable": 0}
    grouped = risk.groupby(["start", "horizon_sessions"], sort=False, observed=True)
    for (start, horizon), group in grouped:
        key = (pd.Timestamp(start), int(horizon)); positions = event_groups.get(key)
        date_events = event_frame.iloc[positions] if positions is not None else event_frame.iloc[0:0]
        month = str(pd.Timestamp(start).to_period("M"))
        for population in ("broad", "investable"):
            current = group if population == "broad" else group.loc[group.investable]
            current_events = date_events if population == "broad" else date_events.loc[date_events.investable]
            if current.empty: continue
            current = current.reset_index(drop=True)
            deciles = np.column_stack([cross_sectional_deciles(current[column].to_numpy(float)) for column in MATCH_COLUMNS])
            names = current.symbol.astype(str).to_numpy(object)
            winner = current.winner_25pct.to_numpy(bool); eligible_indices = np.flatnonzero(~winner)
            members = bucket_members(deciles, ~winner); symbol_index = {symbol: index for index, symbol in enumerate(names)}
            exposure_values = {column: current[column].to_numpy(bool) for column in ALL_EXPOSURE_COLUMNS}
            for exposure in PRIMARY_EXPOSURES:
                for window in PRIMARY_WINDOWS:
                    column = f"{exposure}_{window}"; values = exposure_values[column]
                    state = predictive[(population, int(horizon), exposure, window)]
                    state[0] += len(values); state[1] += int(winner.sum())
                    state[2] += int(values.sum()); state[3] += int((values & winner).sum())
                    nonwinner = values[~winner]
                    aggregate = full_month[(population, int(horizon), exposure, window, month)]
                    aggregate[2] += int(nonwinner.sum()); aggregate[3] += len(nonwinner)
            for event in current_events.itertuples(index=False):
                if event.symbol not in symbol_index: raise ControlStudyError("event absent from same-date risk set")
                event_index = symbol_index[event.symbol]; event_deciles = deciles[event_index]
                selected, tier = select_control_indices(
                    symbols=names, members=members, all_eligible_indices=eligible_indices,
                    event_symbol=str(event.symbol), event_deciles=event_deciles,
                    contract_digest=contract_digest, event_id=str(event.event_id), controls=5,
                )
                count = len(selected); full_match = count == 5
                seen_events[population] += 1; event_totals[(population, int(horizon))][0] += 1
                event_totals[(population, int(horizon))][1] += int(full_match)
                coverage_counts[(population, int(horizon), tier, count)] += 1
                match_rows.append({
                    "population": population, "event_id": str(event.event_id), "event_symbol": str(event.symbol),
                    "start": pd.Timestamp(start), "horizon_sessions": int(horizon), "match_tier": tier,
                    "control_count": count, "full_match": full_match,
                    **{f"event_{name}_decile": int(event_deciles[i]) for i, name in enumerate(MATCH_COLUMNS)},
                })
                for exposure in PRIMARY_EXPOSURES:
                    for window in PRIMARY_WINDOWS:
                        column = f"{exposure}_{window}"; event_value = float(exposure_values[column][event_index])
                        full_state = full_month[(population, int(horizon), exposure, window, month)]
                        full_state[0] += event_value; full_state[1] += 1
                        if full_match:
                            control_mean = float(exposure_values[column][selected].mean())
                            paired = matched_month[(population, int(horizon), exposure, window, month)]
                            paired[0] += event_value; paired[1] += control_mean; paired[2] += 1
                for rank, control_index in enumerate(selected, 1):
                    control_symbol = str(names[control_index]); control_deciles = deciles[control_index]
                    row = {
                        "population": population, "event_id": str(event.event_id),
                        "event_symbol": str(event.symbol), "control_symbol": control_symbol,
                        "start": pd.Timestamp(start), "horizon_sessions": int(horizon),
                        "match_rank": rank, "match_tier": tier,
                        "selection_digest": control_order_digest(contract_digest, str(event.event_id), control_symbol),
                        "maximum_decile_distance": int(np.max(np.abs(event_deciles.astype(int) - control_deciles.astype(int)))),
                    }
                    row.update({f"event_{name}_decile": int(event_deciles[i]) for i, name in enumerate(MATCH_COLUMNS)})
                    row.update({f"control_{name}_decile": int(control_deciles[i]) for i, name in enumerate(MATCH_COLUMNS)})
                    row.update({column: bool(exposure_values[column][control_index]) for column in ALL_EXPOSURE_COLUMNS})
                    control_rows.append(row)
                if len(control_rows) >= 50_000: controls_sink.write(control_rows)
                if len(match_rows) >= 50_000: matches_sink.write(match_rows)
    controls_sink.write(control_rows); matches_sink.write(match_rows); controls_sink.close(); matches_sink.close()
    expected_seen = {"broad": len(event_frame), "investable": int(event_frame.investable.sum())}
    if seen_events != expected_seen: raise ControlStudyError(f"event coverage differs: {seen_events} != {expected_seen}")

    inference_rows = []
    for population in ("broad", "investable"):
        for horizon in (21, 63):
            match_coverage = _safe_ratio(event_totals[(population, horizon)][1], event_totals[(population, horizon)][0])
            for exposure in PRIMARY_EXPOSURES:
                for window in PRIMARY_WINDOWS:
                    for comparator in ("full_nonwinner_risk_set", "matched_nonwinner_controls"):
                        source = full_month if comparator == "full_nonwinner_risk_set" else matched_month
                        selected_months = sorted(key[4] for key in source if key[:4] == (population, horizon, exposure, window))
                        differences = []; used_months = []; winner_sum = comparator_sum = observations = 0.
                        for month in selected_months:
                            state = source[(population, horizon, exposure, window, month)]
                            if comparator == "full_nonwinner_risk_set":
                                if state[1] == 0 or state[3] == 0: continue
                                winner_rate, comparator_rate = state[0] / state[1], state[2] / state[3]
                                winner_sum += state[0]; comparator_sum += state[2]
                                observations += state[1]
                            else:
                                if state[2] == 0: continue
                                winner_rate, comparator_rate = state[0] / state[2], state[1] / state[2]
                                winner_sum += state[0]; comparator_sum += state[1]
                                observations += state[2]
                            differences.append(winner_rate - comparator_rate)
                            used_months.append(month)
                        test_id = f"{population}|{horizon}|{exposure}|{window}|{comparator}"
                        seed = int(prereg["bootstrap_seed"]) + int(stable_hash(test_id)[:8], 16)
                        effect, pvalue, lower, upper = moving_block_positive_inference(
                            differences, resamples=int(prereg["bootstrap_resamples"]),
                            block_length=int(prereg["bootstrap_block_length_months"]), seed=seed,
                        )
                        winner_prevalence = winner_sum / observations
                        if comparator == "full_nonwinner_risk_set":
                            denominator = sum(full_month[(population, horizon, exposure, window, month)][3] for month in used_months)
                        else: denominator = observations
                        comparator_prevalence = comparator_sum / denominator
                        inference_rows.append({
                            "test_id": test_id, "population": population, "horizon_sessions": horizon,
                            "exposure": exposure, "window": window, "comparator": comparator,
                            "event_observations": int(observations), "calendar_months": len(differences),
                            "winner_prevalence": winner_prevalence, "comparator_prevalence": comparator_prevalence,
                            "risk_difference": winner_prevalence - comparator_prevalence,
                            "mean_monthly_risk_difference": effect, "risk_ratio": _safe_ratio(winner_prevalence, comparator_prevalence),
                            "odds_ratio": _odds_ratio(winner_prevalence, comparator_prevalence),
                            "bootstrap_positive_pvalue": pvalue, "bootstrap_ci_lower": lower, "bootstrap_ci_upper": upper,
                            "matched_event_coverage": match_coverage,
                        })
    inference = pd.DataFrame(inference_rows)
    adjusted = holm_adjust(dict(zip(inference.test_id, inference.bootstrap_positive_pvalue)), alpha=.05)
    inference["holm_adjusted_pvalue"] = [adjusted[name][0] for name in inference.test_id]
    inference["holm_reject"] = [adjusted[name][1] for name in inference.test_id]
    directions = {(r.horizon_sessions, r.exposure, r.window, r.comparator, r.population): r.risk_difference > 0
                  for r in inference.itertuples(index=False)}
    inference["same_direction_broad_investable"] = [
        directions[(r.horizon_sessions, r.exposure, r.window, r.comparator, "broad")]
        and directions[(r.horizon_sessions, r.exposure, r.window, r.comparator, "investable")]
        for r in inference.itertuples(index=False)
    ]
    inference["claim_supported"] = (
        (inference.winner_prevalence >= .5) & (inference.holm_adjusted_pvalue < .05)
        & (inference.risk_ratio > 1) & (inference.matched_event_coverage >= .9)
        & inference.same_direction_broad_investable
    )
    inference = inference.sort_values(["population", "horizon_sessions", "exposure", "window", "comparator"], kind="stable").reset_index(drop=True)

    predictive_rows = []
    for key, state in sorted(predictive.items()):
        population, horizon, exposure, window = key; total, winners, exposed, exposed_winners = state
        baseline = winners / total; conditional = exposed_winners / exposed
        predictive_rows.append({
            "population": population, "horizon_sessions": horizon, "exposure": exposure, "window": window,
            "risk_rows": total, "winner_rows": winners, "exposed_rows": exposed, "exposed_winner_rows": exposed_winners,
            "unconditional_winner_probability": baseline, "winner_probability_given_exposure": conditional,
            "predictive_lift": conditional / baseline,
        })
    predictive_frame = pd.DataFrame(predictive_rows)
    eligible = sum(value[0] for value in event_totals.values()); fully = sum(value[1] for value in event_totals.values())
    coverage = _seal({
        "schema_version": SCHEMA, "status": "coverage", "risk_rows": len(risk),
        "clustered_events": len(event_frame), "eligible_event_populations": eligible,
        "fully_matched_event_populations": fully, "full_match_coverage": fully / eligible,
        "matched_control_rows": controls_sink.rows, "event_match_rows": matches_sink.rows,
        "by_population_horizon": {
            f"{key[0]}_{key[1]}": {"events": value[0], "fully_matched": value[1], "coverage": value[1] / value[0]}
            for key, value in sorted(event_totals.items())
        },
        "by_tier_and_shortfall": {
            f"{key[0]}_{key[1]}_{key[2]}_{key[3]}": value for key, value in sorted(coverage_counts.items())
        }, "shortfalls_retained": True,
    })
    decision = _seal({
        "schema_version": SCHEMA, "status": "claim_decision", "primary_tests": len(inference),
        "supported_primary_tests": int(inference.claim_supported.sum()),
        "any_primary_test_supported": bool(inference.claim_supported.any()),
        "all_primary_tests_supported": bool(inference.claim_supported.all()),
        "literal_exact_start_majority_supported": bool(inference.loc[inference.window == "start_day", "claim_supported"].any()),
        "first_five_session_majority_supported": bool(inference.loc[inference.window == "first_5_sessions", "claim_supported"].any()),
        "interpretation": "observational_replication_not_causal_or_trading_strategy",
        "production_promotion_authorized": False,
    })
    smoke._atomic_parquet(temporary / "inference.parquet", inference)
    smoke._atomic_parquet(temporary / "predictive-lift.parquet", predictive_frame)
    smoke._atomic_json(temporary / "coverage.json", coverage)
    smoke._atomic_json(temporary / "claim-decision.json", decision)
    (temporary / "index.html").write_text(_html(inference, coverage, decision))
    seal = _seal({
        "schema_version": SCHEMA, "status": "sealed", "passed": True,
        "preregistration_h1": h1, "preregistration_digest": prereg["preregistration_digest"],
        "event_store_result_digest": prereg["verified_inputs"]["event_store_result_digest"],
        "matched_control_rows": controls_sink.rows, "event_match_rows": matches_sink.rows,
        "inference_rows": len(inference), "predictive_lift_rows": len(predictive_frame),
        "coverage_result_digest": coverage["result_digest"], "claim_decision_result_digest": decision["result_digest"],
        "file_manifest": _manifest(temporary), "elapsed_seconds": perf_counter() - started,
        "independent_verification_authorized": True, "production_promotion_authorized": False,
    }, timing=True)
    smoke._atomic_json(temporary / "SEALED.json", seal)
    try: os.replace(temporary, root)
    except BaseException: shutil.rmtree(temporary, ignore_errors=True); raise
    return seal


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="command", required=True)
    for command in ("preregister", "run"):
        child = sub.add_parser(command); child.add_argument("--repository", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "preregister":
        value = build_preregistration(args.repository); smoke._atomic_json(args.repository / PREREGISTRATION_RELATIVE, value)
    else: value = execute(args.repository)
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)); return 0


if __name__ == "__main__": raise SystemExit(main())
