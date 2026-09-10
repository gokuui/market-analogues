"""Join T14-12 outcomes to immutable final weight-4 identities without inference."""
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

from market_analogues.post_signal_outcome_join import HORIZONS, METRICS, _panel_columns, join_year_outcomes
from market_analogues.types import stable_hash

from experiments.m04r import m04r14_t14_09_outcome_smoke as smoke
from experiments.m04r import m04r14_t14_10_wf03_feasibility as io
from experiments.m04r import m04r14_t14_12_final_balance as final_balance
from experiments.m04r import m04r14_t14_12_final_matches as final_matches


SCHEMA = "m04r14-t14-12-post-signal-outcome-join-v2"
PREREGISTRATION_RELATIVE = Path("experiments/m04r/m04r14_t14_12_outcome_join_v2_preregistered.json")
CACHE_RELATIVE = Path("config/data/analogues/m04r14/t14-12-post-signal-outcome-join-v2-cache")
OUTPUT_RELATIVE = Path("config/data/analogues/m04r14/t14-12-post-signal-outcome-join-v2")
VERIFICATION_RELATIVE = Path("config/data/analogues/m04r14/t14-12-post-signal-outcome-join-v2-verification")
YEAR_FILES = ("subject-outcomes.parquet", "paired-outcomes.parquet", "coverage.parquet")
OUTPUT_FILES = ("coverage.parquet", "outcome-join-decision.json")
RUNTIME_FILES = (
    "experiments/m04r/m04r14_t14_12_outcome_join.py",
    "experiments/m04r/verify_m04r14_t14_12_outcome_join.py",
    "experiments/m04r/m04r14_t14_12_final_matches.py",
    "experiments/m04r/m04r14_t14_12_final_balance.py",
    "src/market_analogues/post_signal_outcome_join.py",
    "config/m04r14-t14-12-post-signal-contract.json", "pyproject.toml",
)


class OutcomeJoinError(RuntimeError): pass


def _now() -> str: return datetime.now(timezone.utc).isoformat()


def _git(repository: Path, *args: str, raw: bool = False) -> str | bytes:
    result = subprocess.run(["git", *args], cwd=repository, capture_output=True, text=not raw, check=False)
    if result.returncode:
        error = result.stderr if not raw else result.stderr.decode(errors="replace")
        raise OutcomeJoinError(error.strip())
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


def _verified_balance(repository: Path) -> dict[str, Any]:
    root = repository / final_balance.OUTPUT_RELATIVE
    store_path, decision_path = root / "SEALED.json", root / "balance-decision.json"
    receipt_path = repository / final_balance.VERIFICATION_RELATIVE / "VERIFIED.json"
    store, decision, receipt = io._read(store_path), io._read(decision_path), io._read(receipt_path)
    receipt_state = {name: item for name, item in receipt.items() if name != "verification_digest"}
    if not final_balance._valid(store, timing=True) or not final_balance._valid(decision) \
            or receipt.get("verification_digest") != stable_hash(receipt_state) \
            or receipt.get("store_result_digest") != store.get("result_digest") \
            or decision.get("balance_gate_pass") is not True or receipt.get("balance_gate_pass") is not True \
            or decision.get("post_signal_outcome_join_authorized") is not True \
            or decision.get("post_signal_inference_authorized") is not False \
            or receipt.get("post_signal_outcome_join_authorized") is not True:
        raise OutcomeJoinError("verified final balance boundary differs")
    return {"balance_store_result_digest": store["result_digest"], "balance_store_sha256": _sha(store_path),
        "balance_decision_digest": decision["result_digest"], "balance_decision_sha256": _sha(decision_path),
        "balance_verification_digest": receipt["verification_digest"], "balance_verification_sha256": _sha(receipt_path)}


def _runtime_manifest(repository: Path, head: str) -> dict[str, str]:
    tracked = set(str(_git(repository, "ls-tree", "-r", "--name-only", head)).splitlines())
    if any(name not in tracked for name in RUNTIME_FILES): raise OutcomeJoinError("runtime file is uncommitted")
    return {name: sha256(_git(repository, "show", f"{head}:{name}", raw=True)).hexdigest() for name in RUNTIME_FILES}


def build_preregistration(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    if _git(repository, "status", "--porcelain", "--untracked-files=all"): raise OutcomeJoinError("clean worktree required")
    if (repository / CACHE_RELATIVE).exists() or (repository / OUTPUT_RELATIVE).exists(): raise OutcomeJoinError("outcome join namespaces exist")
    h0 = str(_git(repository, "rev-parse", "HEAD")); contract = final_matches.v1.panel_stage._contract(repository)
    return _seal({"schema_version": SCHEMA, "status": "frozen_before_P3_post_signal_outcome_join",
        "implementation_h0": h0, "runtime_files": _runtime_manifest(repository, h0),
        "verified_final_balance": _verified_balance(repository),
        "final_match_store_result_digest": _verified_final_matches_digest(repository),
        "panel_store_result_digest": final_matches.v1._verified_panel(repository)["panel_store_result_digest"],
        "contract_digest": contract["contract_digest"],
        "years": io._read(repository / final_matches.PREREGISTRATION_RELATIVE)["years"],
        "horizons": list(HORIZONS), "metrics": list(METRICS),
        "identity_rule": "retain_every_event_and_five_frozen_controls_at_every_horizon_without_reselection",
        "paired_complete_rule": "event_and_all_five_frozen_controls_complete_for_that_horizon",
        "missing_future_rule": "retain_identity_mark_incomplete_and_never_replace_control",
        "matching_coverage_and_outcome_completeness_reported_separately": True,
        "outcome_mutation_must_not_change_identity_projection": True,
        "inference_accessed": False, "production_promotion_authorized": False}, "preregistration_digest")


def _verified_final_matches_digest(repository: Path) -> str:
    receipt = io._read(repository / final_matches.VERIFICATION_RELATIVE / "VERIFIED.json")
    if receipt.get("passed") is not True or receipt.get("outcome_columns_read") != []: raise OutcomeJoinError("final match receipt differs")
    return str(receipt["store_result_digest"])


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
    if len(set(found)) != 1: raise OutcomeJoinError("expected exact preregistration-only child")
    return found[0]


def validate_preregistration(repository: Path) -> tuple[dict[str, Any], str]:
    if _git(repository, "status", "--porcelain", "--untracked-files=all"): raise OutcomeJoinError("clean worktree required")
    path = repository / PREREGISTRATION_RELATIVE; raw = path.read_bytes(); prereg = io._read(path)
    if prereg.get("schema_version") != SCHEMA or not _valid(prereg, "preregistration_digest"): raise OutcomeJoinError("outcome join prereg differs")
    h1 = _sole_child(repository, raw, str(prereg["implementation_h0"]))
    if subprocess.run(["git", "merge-base", "--is-ancestor", h1, "HEAD"], cwd=repository).returncode: raise OutcomeJoinError("H1 is not ancestor")
    if prereg["verified_final_balance"] != _verified_balance(repository) \
            or prereg["final_match_store_result_digest"] != _verified_final_matches_digest(repository): raise OutcomeJoinError("outcome join inputs drifted")
    for name, expected in prereg["runtime_files"].items():
        if _sha(repository / name) != expected: raise OutcomeJoinError(f"runtime drifted: {name}")
    return prereg, h1


def _read_outcome_panel(repository: Path) -> pd.DataFrame:
    stage = final_matches.v1.panel_stage; columns = _panel_columns(HORIZONS)
    frames = [pd.read_parquet(repository / stage.CACHE_RELATIVE / f"shard-{shard:02d}" / "daily-panel.parquet",
                              columns=columns) for shard in range(stage.SHARDS)]
    result = pd.concat(frames, ignore_index=True); del frames
    result["signal_date"] = pd.to_datetime(result.signal_date)
    result = result.sort_values(["signal_date", "symbol"], kind="stable").reset_index(drop=True)
    if result.duplicated(["signal_date", "symbol"]).any(): raise OutcomeJoinError("duplicate outcome panel identity")
    return result


def _write_year(year: int, panel: pd.DataFrame, cache: Path, repository: Path) -> dict[str, Any]:
    final = cache / f"year-{year}"; seal_path = final / "YEAR_SEALED.json"
    if seal_path.exists():
        value = io._read(seal_path)
        if not _valid(value, timing=True) or value.get("file_manifest") != _manifest(final, YEAR_FILES): raise OutcomeJoinError(f"outcome year differs: {year}")
        return value
    started = perf_counter(); match_root = repository / final_matches.CACHE_RELATIVE / f"year-{year}"
    events = pd.read_parquet(match_root / "event-matches.parquet"); controls = pd.read_parquet(match_root / "control-identities.parquet")
    subjects, paired, coverage = join_year_outcomes(events, controls, panel)
    if len(subjects) != 3 * (len(events) + len(controls)) or len(paired) != 3 * len(events): raise OutcomeJoinError("outcome row accounting differs")
    temporary = Path(tempfile.mkdtemp(prefix=f".year-{year}.", dir=cache))
    smoke._atomic_parquet(temporary / YEAR_FILES[0], subjects); smoke._atomic_parquet(temporary / YEAR_FILES[1], paired)
    smoke._atomic_parquet(temporary / YEAR_FILES[2], coverage)
    state = {"schema_version": SCHEMA, "status": "year_sealed", "passed": True, "year": year,
        "event_identity_rows": len(events), "control_identity_rows": len(controls), "subject_outcome_rows": len(subjects),
        "paired_outcome_rows": len(paired), "coverage_rows": len(coverage), "file_manifest": _manifest(temporary, YEAR_FILES),
        "identity_reselection_performed": False, "real_post_signal_outcomes_accessed": True,
        "inference_accessed": False, "elapsed_seconds": perf_counter() - started}
    seal = _seal(state, timing=True); smoke._atomic_json(temporary / "YEAR_SEALED.json", seal)
    try: os.replace(temporary, final)
    except BaseException: shutil.rmtree(temporary, ignore_errors=True); raise
    return seal


def execute(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True); prereg, h1 = validate_preregistration(repository); root = repository / OUTPUT_RELATIVE
    if root.exists():
        seal = io._read(root / "SEALED.json")
        if not _valid(seal, timing=True) or seal.get("file_manifest") != _manifest(root, OUTPUT_FILES): raise OutcomeJoinError("outcome join output differs")
        return seal
    started = perf_counter(); panel = _read_outcome_panel(repository); grouped = panel.groupby(panel.signal_date.dt.year, sort=True)
    cache = repository / CACHE_RELATIVE; cache.mkdir(parents=True, exist_ok=True)
    seals = [_write_year(year, grouped.get_group(year), cache, repository) for year in prereg["years"]]
    coverage_frames = [pd.read_parquet(cache / f"year-{year}" / "coverage.parquet") for year in prereg["years"]]
    coverage = pd.concat(coverage_frames, ignore_index=True).groupby(
        ["population", "signal_name", "horizon_sessions"], sort=True, as_index=False,
    )[["event_rows", "event_complete_rows", "five_control_complete_rows", "paired_complete_rows"]].sum()
    coverage["paired_complete_fraction"] = coverage.paired_complete_rows / coverage.event_rows
    minimum = float(coverage.paired_complete_fraction.min())
    decision = _seal({"schema_version": SCHEMA, "status": "outcome_join_decision",
        "all_identity_rows_retained": True, "identity_reselection_performed": False,
        "minimum_paired_complete_fraction": minimum, "minimum_inference_coverage": .90,
        "all_cells_meet_inference_coverage": minimum >= .90,
        "independent_join_verification_authorized": True, "post_signal_inference_authorized": False,
        "real_post_signal_outcomes_accessed": True, "production_promotion_authorized": False})
    temporary = Path(tempfile.mkdtemp(prefix=f".{root.name}.", dir=root.parent)); smoke._atomic_parquet(temporary / OUTPUT_FILES[0], coverage)
    smoke._atomic_json(temporary / OUTPUT_FILES[1], decision)
    totals: dict[str, int] = defaultdict(int)
    for seal in seals:
        for name in ("event_identity_rows", "control_identity_rows", "subject_outcome_rows", "paired_outcome_rows"): totals[name] += int(seal[name])
    state = {"schema_version": SCHEMA, "status": "sealed", "passed": True, "preregistration_h1": h1,
        "preregistration_digest": prereg["preregistration_digest"], "year_result_digests": [value["result_digest"] for value in seals],
        **totals, "coverage_rows": len(coverage), "minimum_paired_complete_fraction": minimum,
        "decision_result_digest": decision["result_digest"], "file_manifest": _manifest(temporary, OUTPUT_FILES),
        "identity_reselection_performed": False, "real_post_signal_outcomes_accessed": True,
        "inference_accessed": False, "elapsed_seconds": perf_counter() - started,
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
