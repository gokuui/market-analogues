"""Report frozen T14-12 primary effects without overriding the failed coverage gate."""
from __future__ import annotations

import argparse
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

import pandas as pd

from market_analogues.stockbee_controls import moving_block_positive_inference
from market_analogues.types import stable_hash
from market_analogues.walk_forward_scoring import holm_adjust

from experiments.m04r import m04r14_t14_09_outcome_smoke as smoke
from experiments.m04r import m04r14_t14_10_wf03_feasibility as io
from experiments.m04r import m04r14_t14_12_outcome_join as outcomes


SCHEMA = "m04r14-t14-12-primary-unsupported-report-v1"
PREREGISTRATION_RELATIVE = Path("experiments/m04r/m04r14_t14_12_primary_report_v1_preregistered.json")
OUTPUT_RELATIVE = Path("config/data/analogues/m04r14/t14-12-primary-unsupported-report-v1")
VERIFICATION_RELATIVE = Path("config/data/analogues/m04r14/t14-12-primary-unsupported-report-v1-verification")
OUTPUT_FILES = ("primary-effects.parquet", "era-effects.parquet", "decision.json", "report.html")
PRIMARY_HORIZONS = (20, 60)
PRIMARY_METRICS = ("benchmark_relative_log_return", "endpoint_gain_25pct")
RUNTIME_FILES = (
    "experiments/m04r/m04r14_t14_12_primary_report.py",
    "experiments/m04r/verify_m04r14_t14_12_primary_report.py",
    "src/market_analogues/stockbee_controls.py", "src/market_analogues/walk_forward_scoring.py",
    "config/m04r14-t14-12-post-signal-contract.json", "pyproject.toml",
)


class PrimaryReportError(RuntimeError): pass


def _now() -> str: return datetime.now(timezone.utc).isoformat()


def _git(repository: Path, *args: str, raw: bool = False) -> str | bytes:
    result = subprocess.run(["git", *args], cwd=repository, capture_output=True, text=not raw, check=False)
    if result.returncode:
        error = result.stderr if not raw else result.stderr.decode(errors="replace"); raise PrimaryReportError(error.strip())
    return result.stdout if raw else result.stdout.strip()


def _sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20): digest.update(block)
    return digest.hexdigest()


def _seal(value: Mapping[str, Any], key: str = "result_digest", *, timing: bool = False) -> dict[str, Any]:
    omitted = {"elapsed_seconds"} if timing else set(); result = dict(value)
    result[key] = stable_hash({name: item for name, item in result.items() if name not in omitted})
    result["created_at"] = _now(); return result


def _valid(value: Mapping[str, Any], key: str = "result_digest", *, timing: bool = False) -> bool:
    omitted = {key, "created_at"} | ({"elapsed_seconds"} if timing else set())
    return value.get(key) == stable_hash({name: item for name, item in value.items() if name not in omitted})


def _manifest(root: Path, names: Sequence[str]) -> list[dict[str, Any]]:
    return [{"path": name, "bytes": (root / name).stat().st_size, "sha256": _sha(root / name)} for name in names]


def _verified_outcomes(repository: Path) -> dict[str, Any]:
    store_path = repository / outcomes.OUTPUT_RELATIVE / "SEALED.json"
    decision_path = repository / outcomes.OUTPUT_RELATIVE / "outcome-join-decision.json"
    receipt_path = repository / outcomes.VERIFICATION_RELATIVE / "VERIFIED.json"
    store, decision, receipt = io._read(store_path), io._read(decision_path), io._read(receipt_path)
    receipt_state = {name: item for name, item in receipt.items() if name != "verification_digest"}
    if not outcomes._valid(store, timing=True) or not outcomes._valid(decision) \
            or receipt.get("verification_digest") != stable_hash(receipt_state) \
            or receipt.get("store_result_digest") != store.get("result_digest") or receipt.get("passed") is not True \
            or decision.get("post_signal_inference_authorized") is not False \
            or receipt.get("post_signal_inference_authorized") is not False:
        raise PrimaryReportError("verified failed-coverage outcome boundary differs")
    return {"outcome_store_result_digest": store["result_digest"], "outcome_store_sha256": _sha(store_path),
        "outcome_decision_digest": decision["result_digest"], "outcome_decision_sha256": _sha(decision_path),
        "outcome_verification_digest": receipt["verification_digest"], "outcome_verification_sha256": _sha(receipt_path),
        "minimum_paired_complete_fraction": receipt["minimum_paired_complete_fraction"]}


def _runtime_manifest(repository: Path, head: str) -> dict[str, str]:
    tracked = set(str(_git(repository, "ls-tree", "-r", "--name-only", head)).splitlines())
    if any(name not in tracked for name in RUNTIME_FILES): raise PrimaryReportError("runtime file is uncommitted")
    return {name: sha256(_git(repository, "show", f"{head}:{name}", raw=True)).hexdigest() for name in RUNTIME_FILES}


def build_preregistration(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    if _git(repository, "status", "--porcelain", "--untracked-files=all"): raise PrimaryReportError("clean worktree required")
    if (repository / OUTPUT_RELATIVE).exists(): raise PrimaryReportError("primary report namespace exists")
    h0 = str(_git(repository, "rev-parse", "HEAD"))
    return _seal({"schema_version": SCHEMA, "status": "frozen_before_unsupported_primary_effect_reporting",
        "implementation_h0": h0, "runtime_files": _runtime_manifest(repository, h0),
        "verified_outcomes": _verified_outcomes(repository), "primary_horizons": list(PRIMARY_HORIZONS),
        "primary_metrics": list(PRIMARY_METRICS), "family_size": 16, "bootstrap_resamples": 10000,
        "block_length_months": 3, "base_seed": 20260910,
        "monthly_rule": "calendar_month_mean_of_complete_paired_event_differences",
        "bootstrap_rule": "centered_moving_block_one_sided_positive_pvalue_and_uncentered_percentile_95pct_interval",
        "seed_rule": "base_seed_plus_first_8_sha256_hex_of_test_id",
        "holm_alpha": .05, "positive_eras_required": 4, "complete_coverage_required": .90,
        "coverage_failure_cannot_be_overridden_by_effect_or_pvalue": True,
        "reporting_only": True, "post_signal_inference_authorized": False,
        "retrieval_feature_authorized": False, "production_promotion_authorized": False}, "preregistration_digest")


def _sole_child(repository: Path, raw: bytes, h0: str) -> str:
    found = []
    for line in str(_git(repository, "rev-list", "--all", "--children")).splitlines():
        values = line.split()
        if not values or values[0] != h0: continue
        for child in values[1:]:
            parents = str(_git(repository, "rev-list", "--parents", "-n", "1", child)).split()
            changed = str(_git(repository, "diff-tree", "--no-commit-id", "--name-only", "-r", child)).splitlines()
            if parents == [child, h0] and changed == [PREREGISTRATION_RELATIVE.as_posix()] \
                    and _git(repository, "show", f"{child}:{PREREGISTRATION_RELATIVE}", raw=True) == raw: found.append(child)
    if len(set(found)) != 1: raise PrimaryReportError("expected exact preregistration-only child")
    return found[0]


def validate_preregistration(repository: Path) -> tuple[dict[str, Any], str]:
    if _git(repository, "status", "--porcelain", "--untracked-files=all"): raise PrimaryReportError("clean worktree required")
    path = repository / PREREGISTRATION_RELATIVE; raw = path.read_bytes(); prereg = io._read(path)
    if prereg.get("schema_version") != SCHEMA or not _valid(prereg, "preregistration_digest"): raise PrimaryReportError("primary report prereg differs")
    h1 = _sole_child(repository, raw, str(prereg["implementation_h0"]))
    if subprocess.run(["git", "merge-base", "--is-ancestor", h1, "HEAD"], cwd=repository).returncode: raise PrimaryReportError("H1 is not ancestor")
    if prereg["verified_outcomes"] != _verified_outcomes(repository): raise PrimaryReportError("outcome inputs drifted")
    for name, expected in prereg["runtime_files"].items():
        if _sha(repository / name) != expected: raise PrimaryReportError(f"runtime drifted: {name}")
    return prereg, h1


def _era(year: int) -> str:
    if year <= 2007: return "era_2000_2007"
    if year <= 2012: return "era_2008_2012"
    if year <= 2017: return "era_2013_2017"
    if year <= 2022: return "era_2018_2022"
    return "era_2023_2026"


def _read_pairs(repository: Path) -> pd.DataFrame:
    frames = [pd.read_parquet(repository / outcomes.CACHE_RELATIVE / f"year-{year}" / "paired-outcomes.parquet",
        columns=["population", "signal_name", "signal_id", "signal_date", "horizon_sessions", "paired_complete",
                 *(f"paired_{metric}_difference" for metric in PRIMARY_METRICS)]) for year in range(2000, 2027)]
    frame = pd.concat(frames, ignore_index=True); frame["signal_date"] = pd.to_datetime(frame.signal_date); return frame


def _report_html(effects: pd.DataFrame, decision: Mapping[str, Any]) -> str:
    rows = "".join(f"<tr><td>{r.population}</td><td>{r.signal_name}</td><td>{r.horizon_sessions}</td><td>{r.metric}</td>"
                   f"<td>{r.coverage:.1%}</td><td>{r.effect:.6g}</td><td>[{r.bootstrap_lower:.6g}, {r.bootstrap_upper:.6g}]</td>"
                   f"<td>{r.holm_adjusted_pvalue:.4g}</td><td>{'yes' if r.claim_supported else 'no'}</td></tr>"
                   for r in effects.itertuples(index=False))
    return f"""<!doctype html><html><head><meta charset=\"utf-8\"><title>T14-12 primary report</title><style>body{{font-family:system-ui;max-width:1250px;margin:2rem auto}}table{{border-collapse:collapse;width:100%}}th,td{{padding:.4rem;border-bottom:1px solid #ddd;text-align:left}}.warn{{background:#fff1e8;padding:.8rem}}</style></head><body><h1>T14-12 primary effect report</h1><p class=\"warn\"><b>Unsupported/descriptive only.</b> The frozen joint coverage gate failed before this report. P-values or attractive effects cannot override it. Retrieval feature authorized: <b>{'yes' if decision['retrieval_feature_authorized'] else 'no'}</b>.</p><table><thead><tr><th>Population</th><th>Signal</th><th>Horizon</th><th>Metric</th><th>Coverage</th><th>Effect</th><th>95% interval</th><th>Holm p</th><th>Supported</th></tr></thead><tbody>{rows}</tbody></table></body></html>"""


def execute(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True); prereg, h1 = validate_preregistration(repository); root = repository / OUTPUT_RELATIVE
    if root.exists():
        seal = io._read(root / "SEALED.json")
        if not _valid(seal, timing=True) or seal.get("file_manifest") != _manifest(root, OUTPUT_FILES): raise PrimaryReportError("primary report output differs")
        return seal
    started = perf_counter(); pairs = _read_pairs(repository); coverage = pd.read_parquet(repository / outcomes.OUTPUT_RELATIVE / "coverage.parquet")
    effects = []; eras = []
    for population in ("broad", "investable"):
        for signal in ("bullish_range_expansion_4pct", "up_close_4pct"):
            for horizon in PRIMARY_HORIZONS:
                base = pairs.loc[pairs.population.eq(population) & pairs.signal_name.eq(signal)
                                 & pairs.horizon_sessions.eq(horizon)]
                rate = float(coverage.loc[coverage.population.eq(population) & coverage.signal_name.eq(signal)
                    & coverage.horizon_sessions.eq(horizon), "paired_complete_fraction"].iloc[0])
                complete = base.loc[base.paired_complete]
                for metric in PRIMARY_METRICS:
                    column = f"paired_{metric}_difference"; test_id = f"{population}|{signal}|{horizon}|{metric}"
                    monthly = complete.assign(month=complete.signal_date.dt.to_period("M")).groupby("month", sort=True)[column].mean()
                    seed = int(prereg["base_seed"]) + int(sha256(test_id.encode()).hexdigest()[:8], 16)
                    effect, pvalue, lower, upper = moving_block_positive_inference(monthly.to_numpy(float),
                        resamples=int(prereg["bootstrap_resamples"]), block_length=int(prereg["block_length_months"]), seed=seed)
                    era_means = complete.assign(era=complete.signal_date.dt.year.map(_era)).groupby("era", sort=True)[column].mean()
                    for era, value in era_means.items(): eras.append({"test_id": test_id, "era": era, "effect": float(value)})
                    effects.append({"test_id": test_id, "population": population, "signal_name": signal,
                        "horizon_sessions": horizon, "metric": metric, "event_rows": len(base), "complete_rows": len(complete),
                        "coverage": rate, "calendar_months": len(monthly), "effect": effect,
                        "bootstrap_positive_pvalue": pvalue, "bootstrap_lower": lower, "bootstrap_upper": upper,
                        "positive_eras": int((era_means > 0).sum())})
    result = pd.DataFrame(effects); adjusted = holm_adjust(dict(zip(result.test_id, result.bootstrap_positive_pvalue)), alpha=.05)
    result["holm_adjusted_pvalue"] = [adjusted[value][0] for value in result.test_id]
    result["holm_reject"] = [adjusted[value][1] for value in result.test_id]
    directions = result.groupby(["signal_name", "horizon_sessions", "metric"]).effect.apply(lambda values: bool((values > 0).all()))
    result["same_positive_direction_both_populations"] = [directions.loc[(r.signal_name, r.horizon_sessions, r.metric)] for r in result.itertuples()]
    result["statistical_criteria_pass"] = (result.holm_adjusted_pvalue < .05) & (result.bootstrap_lower > 0) \
        & (result.positive_eras >= 4) & result.same_positive_direction_both_populations
    result["coverage_gate_pass"] = result.coverage >= .90
    result["claim_supported"] = result.statistical_criteria_pass & result.coverage_gate_pass
    authorization = []
    for signal in result.signal_name.unique():
        for horizon in PRIMARY_HORIZONS:
            cell = result.loc[result.signal_name.eq(signal) & result.horizon_sessions.eq(horizon)]
            authorization.append({"signal_name": signal, "horizon_sessions": horizon,
                                  "all_four_population_metric_cells_supported": bool(cell.claim_supported.all())})
    feature = any(value["all_four_population_metric_cells_supported"] for value in authorization)
    decision = _seal({"schema_version": SCHEMA, "status": "unsupported_primary_report_decision",
        "primary_cells": len(result), "coverage_failed_cells": int((~result.coverage_gate_pass).sum()),
        "statistical_criteria_passed_cells": int(result.statistical_criteria_pass.sum()),
        "supported_cells": int(result.claim_supported.sum()), "authorization_cells": authorization,
        "retrieval_feature_authorized": feature, "coverage_failure_overridden": False,
        "reporting_only": True, "production_promotion_authorized": False})
    temporary = Path(tempfile.mkdtemp(prefix=f".{root.name}.", dir=root.parent)); smoke._atomic_parquet(temporary / OUTPUT_FILES[0], result)
    smoke._atomic_parquet(temporary / OUTPUT_FILES[1], pd.DataFrame(eras)); smoke._atomic_json(temporary / OUTPUT_FILES[2], decision)
    (temporary / OUTPUT_FILES[3]).write_text(_report_html(result, decision))
    state = {"schema_version": SCHEMA, "status": "sealed", "passed": True, "preregistration_h1": h1,
        "preregistration_digest": prereg["preregistration_digest"], "outcome_store_result_digest": prereg["verified_outcomes"]["outcome_store_result_digest"],
        "primary_effect_rows": len(result), "era_effect_rows": len(eras), "decision_result_digest": decision["result_digest"],
        "file_manifest": _manifest(temporary, OUTPUT_FILES), "reporting_only": True,
        "coverage_failure_overridden": False, "elapsed_seconds": perf_counter() - started,
        "independent_verification_authorized": True, "production_promotion_authorized": False}
    seal = _seal(state, timing=True); smoke._atomic_json(temporary / "SEALED.json", seal)
    try: os.replace(temporary, root)
    except BaseException: shutil.rmtree(temporary, ignore_errors=True); raise
    return seal


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="command", required=True)
    for command in ("preregister", "run"):
        child = sub.add_parser(command); child.add_argument("--repository", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "preregister": value = build_preregistration(args.repository); smoke._atomic_json(args.repository / PREREGISTRATION_RELATIVE, value)
    else: value = execute(args.repository)
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)); return 0


if __name__ == "__main__": raise SystemExit(main())
