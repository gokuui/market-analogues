"""Independently reconstruct clustered Stockbee events and prevalence."""
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

import numpy as np
import pandas as pd

from market_analogues.types import stable_hash

from experiments.m04r import m04r14_t14_10_wf03_feasibility as base
from experiments.m04r import m04r14_t14_11_stockbee_events as target


SCHEMA = "m04r14-t14-11-stockbee-events-verification-v1"
TOLERANCE = 5e-13


class EventVerificationError(RuntimeError): pass


def _now() -> str: return datetime.now(timezone.utc).isoformat()


def _sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20): digest.update(block)
    return digest.hexdigest()


def _valid(value: Mapping[str, Any], key: str = "result_digest", *, timing: bool = False) -> bool:
    omitted = {key, "created_at"} | ({"elapsed_seconds"} if timing else set())
    return value.get(key) == stable_hash({k: v for k, v in value.items() if k not in omitted})


def _reference_events(frame: pd.DataFrame, contract_digest: str) -> pd.DataFrame:
    winners = frame.loc[frame.winner_25pct].sort_values(["symbol", "horizon_sessions", "start_position"], kind="stable").copy()
    prior = winners.groupby(["symbol", "horizon_sessions"], sort=False).start_position.shift()
    winners["event_run"] = ((winners.start_position - prior != 1) | prior.isna()).groupby(
        [winners.symbol, winners.horizon_sessions]
    ).cumsum().astype(int)
    keys = ["symbol", "horizon_sessions", "event_run"]
    grouped = winners.groupby(keys, sort=True)
    first = grouped.nth(0).reset_index(); sizes = grouped.size().rename("event_run_length").reset_index()
    if "index" in first.columns: first = first.drop(columns="index")
    peaks = grouped.forward_close_return.max().rename("event_peak_return").reset_index()
    result = first.merge(sizes, on=keys, validate="one_to_one").merge(peaks, on=keys, validate="one_to_one")
    result["event_id"] = [stable_hash([contract_digest, r.symbol, int(r.horizon_sessions), int(r.start_position)])[:24] for r in result.itertuples(index=False)]
    return result.sort_values(["start", "symbol", "horizon_sessions"], kind="stable").reset_index(drop=True)


def _prevalence_counts(frame: pd.DataFrame, source: str, counts: dict[tuple[Any, ...], list[int]]) -> None:
    for population, selected in (("broad", frame), ("investable", frame.loc[frame.investable])):
        for horizon, horizon_rows in selected.groupby("horizon_sessions", sort=True):
            groups = [("all", horizon_rows)] if source == "clustered_winners" else [
                ("winner", horizon_rows.loc[horizon_rows.winner_25pct]),
                ("nonwinner", horizon_rows.loc[~horizon_rows.winner_25pct]), ("all", horizon_rows),
            ]
            for outcome, group in groups:
                for exposure in target.EXPOSURES:
                    for window in target.WINDOWS:
                        key = (source, population, int(horizon), outcome, exposure, window)
                        old = counts.setdefault(key, [0, 0]); old[0] += len(group); old[1] += int(group[f"{exposure}_{window}"].sum())


def _assert_frame(expected: pd.DataFrame, observed: pd.DataFrame) -> None:
    if list(expected.columns) != list(observed.columns) or len(expected) != len(observed):
        raise EventVerificationError("event frame schema/count differs")
    for column in expected.columns:
        left, right = expected[column], observed[column]
        if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            a, b = left.to_numpy(float), right.to_numpy(float)
            if not np.array_equal(np.isnan(a), np.isnan(b)): raise EventVerificationError(f"missingness differs: {column}")
            finite = np.isfinite(a) & np.isfinite(b)
            if finite.any() and np.max(np.abs(a[finite] - b[finite])) > TOLERANCE:
                raise EventVerificationError(f"numeric values differ: {column}")
        elif not left.fillna("<NA>").astype(str).equals(right.fillna("<NA>").astype(str)):
            raise EventVerificationError(f"values differ: {column}")


def execute(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    if subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=repository,
                      capture_output=True, text=True, check=True).stdout:
        raise EventVerificationError("clean worktree required")
    started = perf_counter(); root = repository / target.OUTPUT_RELATIVE
    seal = base._read(root / "SEALED.json"); prereg = base._read(repository / target.PREREGISTRATION_RELATIVE)
    if not _valid(seal, timing=True) or not _valid(prereg, "preregistration_digest") \
            or seal.get("file_manifest") != target._manifest(root): raise EventVerificationError("event seals differ")
    frames = []; counts: dict[tuple[Any, ...], list[int]] = {}; risk_rows = winners = 0
    cache = repository / target.risk_set.CACHE_RELATIVE
    for shard in range(target.risk_set.SHARDS):
        risk = pd.read_parquet(cache / f"shard-{shard:02d}" / "risk-set.parquet")
        risk_rows += len(risk); winners += int(risk.winner_25pct.sum()); _prevalence_counts(risk, "unclustered_risk_set", counts)
        event = _reference_events(risk, prereg["contract_digest"]); frames.append(event)
    expected_events = pd.concat(frames, ignore_index=True).sort_values(["start", "symbol", "horizon_sessions"], kind="stable").reset_index(drop=True)
    _prevalence_counts(expected_events, "clustered_winners", counts)
    expected_prevalence = pd.DataFrame([{
        "source_population": key[0], "population": key[1], "horizon_sessions": key[2],
        "outcome_group": key[3], "exposure": key[4], "window": key[5], "rows": value[0],
        "exposed_rows": value[1], "prevalence": value[1] / value[0],
    } for key, value in counts.items()]).sort_values(
        ["source_population", "population", "horizon_sessions", "outcome_group", "exposure", "window"], kind="stable",
    ).reset_index(drop=True)
    observed_events = pd.read_parquet(root / "clustered-events.parquet")
    observed_prevalence = pd.read_parquet(root / "prevalence.parquet")
    _assert_frame(expected_events, observed_events); _assert_frame(expected_prevalence, observed_prevalence)
    coverage = base._read(root / "COVERAGE.json")
    if not _valid(coverage) or coverage.get("risk_rows") != risk_rows \
            or coverage.get("unclustered_winner_rows") != winners \
            or coverage.get("clustered_events") != len(expected_events) \
            or coverage.get("matched_controls_constructed") is not False:
        raise EventVerificationError("event coverage differs")
    state = {
        "schema_version": SCHEMA, "status": "verified", "passed": True,
        "store_result_digest": seal["result_digest"], "store_result_sha256": _sha(root / "SEALED.json"),
        "verified_risk_rows": risk_rows, "verified_unclustered_winners": winners,
        "verified_clustered_events": len(expected_events), "verified_prevalence_rows": len(expected_prevalence),
        "gates": {"all_cluster_boundaries_reconstructed": True, "all_event_rows_reconstructed": True,
                  "all_full_risk_prevalence_reconstructed": True, "all_clustered_prevalence_reconstructed": True,
                  "matched_controls_remain_unconstructed": True, "all_physical_seals_valid": True},
        "matched_controls_constructed": False, "production_promotion_authorized": False,
        "elapsed_seconds": perf_counter() - started, "created_at": _now(),
    }
    result = {**state, "verification_digest": stable_hash(state)}; output = repository / target.VERIFICATION_RELATIVE
    if output.exists(): return base._read(output / "VERIFIED.json")
    output.parent.mkdir(parents=True, exist_ok=True); temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        target.smoke._atomic_json(temporary / "VERIFIED.json", result); os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True); raise
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--repository", type=Path, required=True)
    args = parser.parse_args(argv); print(json.dumps(execute(args.repository), indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
