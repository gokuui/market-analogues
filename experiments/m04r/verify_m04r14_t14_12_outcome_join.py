"""Independently verify the T14-12 immutable-identity outcome join."""
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

from market_analogues.post_signal_outcome_join import HORIZONS, IDENTITY_COLUMNS, METRICS, _panel_columns
from market_analogues.types import stable_hash

from experiments.m04r import m04r14_t14_10_wf03_feasibility as io
from experiments.m04r import m04r14_t14_12_outcome_join as target


SCHEMA = "m04r14-t14-12-post-signal-outcome-join-verification-v1"


class OutcomeJoinVerificationError(RuntimeError): pass


def _now() -> str: return datetime.now(timezone.utc).isoformat()


def _sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20): digest.update(block)
    return digest.hexdigest()


def _read_panel(repository: Path) -> pd.DataFrame:
    stage = target.final_matches.v1.panel_stage; columns = _panel_columns(HORIZONS)
    frames = [pd.read_parquet(repository / stage.CACHE_RELATIVE / f"shard-{shard:02d}" / "daily-panel.parquet",
                              columns=columns) for shard in range(stage.SHARDS)]
    result = pd.concat(frames, ignore_index=True); result["signal_date"] = pd.to_datetime(result.signal_date)
    result = result.sort_values(["signal_date", "symbol"], kind="stable").reset_index(drop=True)
    if result.duplicated(["signal_date", "symbol"]).any(): raise OutcomeJoinVerificationError("duplicate panel identity")
    return result


def _subjects(events: pd.DataFrame, controls: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    event = events.loc[:, list(IDENTITY_COLUMNS)].copy(); event["subject_role"] = "event"
    event["subject_symbol"] = event.event_symbol; event["match_rank"] = 0; event["selection_digest"] = ""
    control = controls.loc[:, [*IDENTITY_COLUMNS, "control_symbol", "match_rank", "selection_digest"]].copy()
    control["subject_role"] = "control"; control["subject_symbol"] = control.pop("control_symbol")
    identities = pd.concat([event, control], ignore_index=True)
    lookup = panel.loc[:, _panel_columns(HORIZONS)]
    joined = identities.merge(lookup, left_on=["signal_date", "subject_symbol"], right_on=["signal_date", "symbol"],
                              how="left", validate="many_to_one").drop(columns="symbol")
    result = []
    for horizon in HORIZONS:
        part = joined.loc[:, [*IDENTITY_COLUMNS, "subject_role", "subject_symbol", "match_rank", "selection_digest"]].copy()
        part["horizon_sessions"] = horizon; part["complete"] = joined[f"complete_{horizon}"].fillna(False).astype(bool)
        part["status"] = joined[f"status_{horizon}"].fillna("panel_identity_absent").astype(str)
        for metric in METRICS: part[metric] = joined[f"{metric}_{horizon}"].astype(float)
        part["barrier_code"] = joined.barrier_code_20.astype("Int8") if horizon == 20 else pd.Series(pd.NA, index=joined.index, dtype="Int8")
        result.append(part)
    return pd.concat(result, ignore_index=True)


def _paired(subjects: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    outputs = []
    for horizon, current in subjects.groupby("horizon_sessions", sort=True):
        event = current.loc[current.subject_role.eq("event")].set_index(list(IDENTITY_COLUMNS))
        control = current.loc[current.subject_role.eq("control")].groupby(list(IDENTITY_COLUMNS), sort=False)
        counts = control.complete.sum().reindex(event.index, fill_value=0).astype(int)
        part = event.reset_index().loc[:, list(IDENTITY_COLUMNS)].copy(); part["horizon_sessions"] = int(horizon)
        part["event_complete"] = event.complete.to_numpy(bool); part["complete_control_count"] = counts.to_numpy(int)
        part["paired_complete"] = part.event_complete & part.complete_control_count.eq(5); part["event_status"] = event.status.to_numpy(str)
        for metric in METRICS:
            means = control[metric].mean().reindex(event.index); part[f"event_{metric}"] = event[metric].to_numpy(float)
            part[f"mean_control_{metric}"] = means.to_numpy(float)
            part[f"paired_{metric}_difference"] = part[f"event_{metric}"] - part[f"mean_control_{metric}"]
            part.loc[~part.paired_complete, [f"mean_control_{metric}", f"paired_{metric}_difference"]] = np.nan
        outputs.append(part)
    paired = pd.concat(outputs, ignore_index=True)
    coverage = paired.groupby(["population", "signal_name", "horizon_sessions"], sort=True).agg(
        event_rows=("signal_id", "size"), event_complete_rows=("event_complete", "sum"),
        five_control_complete_rows=("complete_control_count", lambda values: int(np.sum(np.asarray(values) == 5))),
        paired_complete_rows=("paired_complete", "sum")).reset_index()
    coverage["paired_complete_fraction"] = coverage.paired_complete_rows / coverage.event_rows
    return paired, coverage


def _assert_frame(expected: pd.DataFrame, observed: pd.DataFrame, label: str) -> None:
    try: pd.testing.assert_frame_equal(expected.reset_index(drop=True), observed.reset_index(drop=True),
                                       check_exact=True, check_dtype=False)
    except AssertionError as error: raise OutcomeJoinVerificationError(f"{label} differs") from error


def execute(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    if subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=repository,
                      capture_output=True, text=True, check=True).stdout: raise OutcomeJoinVerificationError("clean worktree required")
    started = perf_counter(); prereg, _ = target.validate_preregistration(repository); root = repository / target.OUTPUT_RELATIVE
    seal = io._read(root / "SEALED.json")
    if not target._valid(seal, timing=True) or seal.get("file_manifest") != target._manifest(root, target.OUTPUT_FILES):
        raise OutcomeJoinVerificationError("outcome join seal differs")
    panel = _read_panel(repository); grouped = panel.groupby(panel.signal_date.dt.year, sort=True)
    totals: dict[str, int] = defaultdict(int); coverages = []; digests = []; mutation_cases = 0
    identity_columns = [*IDENTITY_COLUMNS, "subject_role", "subject_symbol", "match_rank", "selection_digest", "horizon_sessions"]
    for year in prereg["years"]:
        match_root = repository / target.final_matches.CACHE_RELATIVE / f"year-{year}"
        events = pd.read_parquet(match_root / "event-matches.parquet"); controls = pd.read_parquet(match_root / "control-identities.parquet")
        year_root = repository / target.CACHE_RELATIVE / f"year-{year}"; year_seal = io._read(year_root / "YEAR_SEALED.json")
        if not target._valid(year_seal, timing=True) or year_seal.get("file_manifest") != target._manifest(year_root, target.YEAR_FILES) \
                or year_seal.get("identity_reselection_performed") is not False or year_seal.get("inference_accessed") is not False:
            raise OutcomeJoinVerificationError(f"outcome year seal differs: {year}")
        expected_subjects = _subjects(events, controls, grouped.get_group(year))
        observed_subjects = pd.read_parquet(year_root / "subject-outcomes.parquet")
        _assert_frame(expected_subjects, observed_subjects, f"subject outcomes {year}")
        expected_paired, expected_coverage = _paired(expected_subjects)
        _assert_frame(expected_paired, pd.read_parquet(year_root / "paired-outcomes.parquet"), f"paired outcomes {year}")
        _assert_frame(expected_coverage, pd.read_parquet(year_root / "coverage.parquet"), f"coverage {year}")
        if len(expected_subjects):
            mutated_panel = grouped.get_group(year).head(2000).copy()
            for metric in METRICS:
                for horizon in HORIZONS: mutated_panel[f"{metric}_{horizon}"] = 12345.
            sample_events = events.loc[events.signal_date.isin(mutated_panel.signal_date)].head(20)
            sample_ids = set(sample_events.signal_id); sample_controls = controls.loc[controls.signal_id.isin(sample_ids)]
            if len(sample_events):
                before = _subjects(sample_events, sample_controls, grouped.get_group(year))
                after = _subjects(sample_events, sample_controls, mutated_panel)
                _assert_frame(before[identity_columns], after[identity_columns], f"mutation identity {year}"); mutation_cases += len(before)
        coverages.append(expected_coverage); digests.append(year_seal["result_digest"])
        totals["event_identity_rows"] += len(events); totals["control_identity_rows"] += len(controls)
        totals["subject_outcome_rows"] += len(expected_subjects); totals["paired_outcome_rows"] += len(expected_paired)
    coverage = pd.concat(coverages, ignore_index=True).groupby(
        ["population", "signal_name", "horizon_sessions"], sort=True, as_index=False,
    )[["event_rows", "event_complete_rows", "five_control_complete_rows", "paired_complete_rows"]].sum()
    coverage["paired_complete_fraction"] = coverage.paired_complete_rows / coverage.event_rows
    _assert_frame(coverage, pd.read_parquet(root / "coverage.parquet"), "aggregate coverage")
    decision = io._read(root / "outcome-join-decision.json"); minimum = float(coverage.paired_complete_fraction.min())
    if not target._valid(decision) or decision.get("minimum_paired_complete_fraction") != minimum \
            or decision.get("all_cells_meet_inference_coverage") != (minimum >= .90) \
            or decision.get("post_signal_inference_authorized") is not False:
        raise OutcomeJoinVerificationError("outcome join decision differs")
    if digests != seal["year_result_digests"] or any(int(seal[name]) != value for name, value in totals.items()):
        raise OutcomeJoinVerificationError("outcome aggregate accounting differs")
    state = {"schema_version": SCHEMA, "status": "verified", "passed": True,
        "store_result_digest": seal["result_digest"], "store_result_sha256": _sha(root / "SEALED.json"), **totals,
        "verified_coverage_rows": len(coverage), "minimum_paired_complete_fraction": minimum,
        "outcome_mutation_identity_cases": mutation_cases,
        "gates": {"all_subject_outcomes_reconstructed": True, "all_paired_outcomes_reconstructed": True,
                  "all_missingness_and_coverage_reconstructed": True, "all_match_identities_retained": True,
                  "outcome_mutation_identity_invariant_passed": True, "inference_boundary_remained_closed": True},
        "real_post_signal_outcomes_accessed": True, "inference_accessed": False,
        "post_signal_inference_authorized": minimum >= .90, "production_promotion_authorized": False,
        "elapsed_seconds": perf_counter() - started, "created_at": _now()}
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
