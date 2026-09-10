"""Independently verify final T14-12 balance and bake-off reproduction."""
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

import pandas as pd

from market_analogues.types import stable_hash

from experiments.m04r import m04r14_t14_10_wf03_feasibility as io
from experiments.m04r import verify_m04r14_t14_12_balance as oracle
from experiments.m04r import m04r14_t14_12_final_balance as target


SCHEMA = "m04r14-t14-12-post-signal-final-balance-verification-v6"


class FinalBalanceVerificationError(RuntimeError): pass


def _now() -> str: return datetime.now(timezone.utc).isoformat()


def _sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20): digest.update(block)
    return digest.hexdigest()


def execute(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    if subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=repository,
                      capture_output=True, text=True, check=True).stdout: raise FinalBalanceVerificationError("clean worktree required")
    started = perf_counter(); prereg, _ = target.validate_preregistration(repository); root = repository / target.OUTPUT_RELATIVE
    seal = io._read(root / "SEALED.json")
    if not target._valid(seal, timing=True) or seal.get("file_manifest") != target._manifest(root, target.OUTPUT_FILES):
        raise FinalBalanceVerificationError("final balance seal differs")
    frame = target.final_matches.v1._read_causal_panel(repository); grouped = frame.groupby(frame.signal_date.dt.year, sort=True)
    combined = defaultdict(oracle._empty); reuse = []; digests = []
    for year in prereg["years"]:
        year_seal = io._read(repository / target.CACHE_RELATIVE / f"year-{year}" / "YEAR_SEALED.json")
        if not target._valid(year_seal, timing=True) or year_seal.get("outcome_columns_read") != []:
            raise FinalBalanceVerificationError(f"final balance year differs: {year}")
        controls = pd.read_parquet(repository / target.final_matches.CACHE_RELATIVE / f"year-{year}" / "control-identities.parquet")
        stats, expected_reuse = oracle._reconstruct_year(year, grouped.get_group(year), controls)
        if stats.keys() != year_seal["statistics"].keys(): raise FinalBalanceVerificationError("final year statistic keys differ")
        for key in stats: oracle._assert_nested(stats[key], year_seal["statistics"][key], f"final year {year}/{key}")
        if len(expected_reuse) != len(year_seal["reuse"]): raise FinalBalanceVerificationError("final reuse count differs")
        for expected, observed in zip(expected_reuse, year_seal["reuse"]): oracle._assert_nested(expected, observed, "final reuse")
        for key, value in stats.items(): oracle._merge(combined[key], value)
        reuse.extend(expected_reuse); digests.append(year_seal["result_digest"])
    summary = oracle._summaries(combined).reset_index(drop=True)
    observed_summary = pd.read_parquet(root / "balance-summary.parquet")
    oracle._assert_frame(summary, observed_summary, "final balance summary")
    oracle._assert_frame(pd.DataFrame(reuse), pd.read_parquet(root / "reuse-summary.parquet"), "final reuse summary")
    selected = target._selected_bakeoff_frame(repository)
    try: pd.testing.assert_frame_equal(observed_summary, selected, check_exact=True)
    except AssertionError as error: raise FinalBalanceVerificationError("selected bake-off rows differ") from error
    primary = summary.loc[summary.match_tier.eq("all")]
    overall = bool((primary.loc[primary.scope.eq("overall"), "standardized_mean_difference"].abs() <= .10).all())
    eras = bool((primary.loc[~primary.scope.eq("overall"), "standardized_mean_difference"].abs() <= .20).all())
    decision = io._read(root / "balance-decision.json")
    if not target._valid(decision) or decision["overall_balance_pass"] != overall \
            or decision["era_balance_pass"] != eras or decision["balance_gate_pass"] != (overall and eras) \
            or decision.get("selected_bakeoff_rows_exactly_reproduced") is not True \
            or decision.get("post_signal_outcome_join_authorized") != (overall and eras) \
            or decision.get("post_signal_inference_authorized") is not False:
        raise FinalBalanceVerificationError("final balance decision differs")
    if digests != seal["year_result_digests"] or seal.get("post_signal_results_accessed") is not False:
        raise FinalBalanceVerificationError("final balance aggregate binding differs")
    state = {"schema_version": SCHEMA, "status": "verified", "passed": True,
        "store_result_digest": seal["result_digest"], "store_result_sha256": _sha(root / "SEALED.json"),
        "verified_balance_summary_rows": len(summary), "verified_reuse_summary_rows": len(reuse),
        "balance_gate_pass": overall and eras, "selected_bakeoff_rows_exactly_reproduced": True,
        "gates": {"all_year_statistics_reconstructed": True, "all_transforms_and_smds_reconstructed": True,
                  "all_decile_histograms_reconstructed": True, "all_reuse_diagnostics_reconstructed": True,
                  "selected_bakeoff_rows_exactly_reproduced": True, "balance_decision_reconstructed": True,
                  "outcome_boundary_remained_closed": True},
        "outcome_columns_read": [], "post_signal_results_accessed": False, "post_signal_outcome_join_authorized": overall and eras,
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
