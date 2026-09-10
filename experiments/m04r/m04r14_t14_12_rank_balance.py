"""Apply the unchanged outcome-blind balance gate to T14-12 V2 controls."""
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

from experiments.m04r import m04r14_t14_09_outcome_smoke as smoke
from experiments.m04r import m04r14_t14_10_wf03_feasibility as io
from experiments.m04r import m04r14_t14_12_balance as base_balance
from experiments.m04r import m04r14_t14_12_rank_matches as rank_matches


SCHEMA = "m04r14-t14-12-post-signal-rank-balance-v2"
PREREGISTRATION_RELATIVE = Path("experiments/m04r/m04r14_t14_12_rank_balance_v2_preregistered.json")
CACHE_RELATIVE = Path("config/data/analogues/m04r14/t14-12-post-signal-rank-balance-v2-cache")
OUTPUT_RELATIVE = Path("config/data/analogues/m04r14/t14-12-post-signal-rank-balance-v2")
VERIFICATION_RELATIVE = Path("config/data/analogues/m04r14/t14-12-post-signal-rank-balance-v2-verification")
OUTPUT_FILES = base_balance.OUTPUT_FILES
RUNTIME_FILES = (
    "experiments/m04r/m04r14_t14_12_rank_balance.py",
    "experiments/m04r/verify_m04r14_t14_12_rank_balance.py",
    "experiments/m04r/m04r14_t14_12_balance.py",
    "experiments/m04r/verify_m04r14_t14_12_balance.py",
    "experiments/m04r/m04r14_t14_12_rank_matches.py",
    "config/m04r14-t14-12-post-signal-contract.json", "pyproject.toml",
)


class RankBalanceError(RuntimeError): pass


def _now() -> str: return datetime.now(timezone.utc).isoformat()


def _git(repository: Path, *args: str, raw: bool = False) -> str | bytes:
    result = subprocess.run(["git", *args], cwd=repository, capture_output=True, text=not raw, check=False)
    if result.returncode:
        error = result.stderr if not raw else result.stderr.decode(errors="replace")
        raise RankBalanceError(error.strip())
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


def _verified_rank_matches(repository: Path) -> dict[str, Any]:
    store_path = repository / rank_matches.OUTPUT_RELATIVE / "SEALED.json"
    verify_path = repository / rank_matches.VERIFICATION_RELATIVE / "VERIFIED.json"
    store, receipt = io._read(store_path), io._read(verify_path)
    receipt_state = {name: item for name, item in receipt.items() if name != "verification_digest"}
    if not rank_matches._valid(store, timing=True) or receipt.get("verification_digest") != stable_hash(receipt_state) \
            or receipt.get("store_result_digest") != store.get("result_digest") \
            or receipt.get("passed") is not True or receipt.get("outcome_columns_read") != []:
        raise RankBalanceError("verified rank matches differ")
    return {"rank_match_store_digest": store["result_digest"], "rank_match_store_sha256": _sha(store_path),
            "rank_match_verification_digest": receipt["verification_digest"],
            "rank_match_verification_sha256": _sha(verify_path), "event_match_rows": store["event_match_rows"],
            "control_identity_rows": store["control_identity_rows"]}


def _runtime_manifest(repository: Path, head: str) -> dict[str, str]:
    tracked = set(str(_git(repository, "ls-tree", "-r", "--name-only", head)).splitlines())
    if any(name not in tracked for name in RUNTIME_FILES): raise RankBalanceError("runtime file is uncommitted")
    return {name: sha256(_git(repository, "show", f"{head}:{name}", raw=True)).hexdigest() for name in RUNTIME_FILES}


def build_preregistration(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    if _git(repository, "status", "--porcelain", "--untracked-files=all"): raise RankBalanceError("clean worktree required")
    if (repository / CACHE_RELATIVE).exists() or (repository / OUTPUT_RELATIVE).exists(): raise RankBalanceError("V2 balance namespaces exist")
    v1_prereg = io._read(repository / base_balance.PREREGISTRATION_RELATIVE); h0 = str(_git(repository, "rev-parse", "HEAD"))
    return _seal({"schema_version": SCHEMA, "status": "frozen_before_V2_balance_access",
        "implementation_h0": h0, "runtime_files": _runtime_manifest(repository, h0),
        "verified_rank_matches": _verified_rank_matches(repository),
        "years": io._read(repository / rank_matches.PREREGISTRATION_RELATIVE)["years"],
        "covariate_transforms": v1_prereg["covariate_transforms"],
        "overall_abs_smd_limit": v1_prereg["overall_abs_smd_limit"],
        "each_era_abs_smd_limit": v1_prereg["each_era_abs_smd_limit"],
        "decision_rule_identical_to_V1": True, "outcome_columns_read": [],
        "post_signal_results_accessed": False, "production_promotion_authorized": False}, "preregistration_digest")


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
    if len(set(found)) != 1: raise RankBalanceError("expected exact preregistration-only child")
    return found[0]


def validate_preregistration(repository: Path) -> tuple[dict[str, Any], str]:
    if _git(repository, "status", "--porcelain", "--untracked-files=all"): raise RankBalanceError("clean worktree required")
    path = repository / PREREGISTRATION_RELATIVE; raw = path.read_bytes(); prereg = io._read(path)
    if prereg.get("schema_version") != SCHEMA or not _valid(prereg, "preregistration_digest"): raise RankBalanceError("V2 balance prereg differs")
    h1 = _sole_child(repository, raw, str(prereg["implementation_h0"]))
    if subprocess.run(["git", "merge-base", "--is-ancestor", h1, "HEAD"], cwd=repository).returncode: raise RankBalanceError("H1 is not ancestor")
    if prereg["verified_rank_matches"] != _verified_rank_matches(repository): raise RankBalanceError("rank matches drifted")
    for name, expected in prereg["runtime_files"].items():
        if _sha(repository / name) != expected: raise RankBalanceError(f"runtime drifted: {name}")
    return prereg, h1


def _write_year(year: int, panel: pd.DataFrame, cache: Path, repository: Path) -> dict[str, Any]:
    final = cache / f"year-{year}"; path = final / "YEAR_SEALED.json"
    if path.exists():
        value = io._read(path)
        if not _valid(value, timing=True): raise RankBalanceError(f"V2 balance year differs: {year}")
        return value
    started = perf_counter(); controls = pd.read_parquet(repository / rank_matches.CACHE_RELATIVE / f"year-{year}" / "control-identities.parquet")
    stats, reuse = base_balance._year_stats(year, panel, controls)
    state = {"schema_version": SCHEMA, "status": "year_sealed", "passed": True, "year": year,
        "panel_rows": len(panel), "control_identity_rows": len(controls), "statistics": stats, "reuse": reuse,
        "outcome_columns_read": [], "post_signal_results_accessed": False, "elapsed_seconds": perf_counter() - started}
    seal = _seal(state, timing=True); final.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".year-{year}.", dir=final.parent))
    try: smoke._atomic_json(temporary / "YEAR_SEALED.json", seal); os.replace(temporary, final)
    except BaseException: shutil.rmtree(temporary, ignore_errors=True); raise
    return seal


def execute(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True); prereg, h1 = validate_preregistration(repository); root = repository / OUTPUT_RELATIVE
    if root.exists():
        seal = io._read(root / "SEALED.json")
        if not _valid(seal, timing=True) or seal.get("file_manifest") != _manifest(root, OUTPUT_FILES): raise RankBalanceError("V2 balance output differs")
        return seal
    started = perf_counter(); frame = rank_matches.v1._read_causal_panel(repository); grouped = frame.groupby(frame.signal_date.dt.year, sort=True)
    cache = repository / CACHE_RELATIVE; cache.mkdir(parents=True, exist_ok=True)
    seals = [_write_year(year, grouped.get_group(year), cache, repository) for year in prereg["years"]]
    combined = defaultdict(base_balance._empty_stats); reuse = []
    for seal in seals:
        for key, value in seal["statistics"].items(): base_balance._merge_stats(combined[key], value)
        reuse.extend(seal["reuse"])
    summary = pd.DataFrame(base_balance._summary_rows(combined)); primary = summary.loc[summary.match_tier.eq("all")]
    overall = bool((primary.loc[primary.scope.eq("overall"), "standardized_mean_difference"].abs() <= .10).all())
    eras = bool((primary.loc[~primary.scope.eq("overall"), "standardized_mean_difference"].abs() <= .20).all())
    decision = _seal({"schema_version": SCHEMA, "status": "balance_decision", "overall_abs_smd_limit": .10,
        "each_era_abs_smd_limit": .20, "overall_balance_pass": overall, "era_balance_pass": eras,
        "balance_gate_pass": overall and eras, "outcome_blind_rematching_required": not (overall and eras),
        "post_signal_inference_authorized": overall and eras, "outcome_columns_read": [], "production_promotion_authorized": False})
    temporary = Path(tempfile.mkdtemp(prefix=f".{root.name}.", dir=root.parent))
    smoke._atomic_parquet(temporary / OUTPUT_FILES[0], summary); smoke._atomic_parquet(temporary / OUTPUT_FILES[1], pd.DataFrame(reuse))
    smoke._atomic_json(temporary / OUTPUT_FILES[2], decision)
    state = {"schema_version": SCHEMA, "status": "sealed", "passed": True, "preregistration_h1": h1,
        "preregistration_digest": prereg["preregistration_digest"], "rank_match_store_digest": prereg["verified_rank_matches"]["rank_match_store_digest"],
        "year_result_digests": [value["result_digest"] for value in seals], "balance_summary_rows": len(summary),
        "reuse_summary_rows": len(reuse), "balance_decision_result_digest": decision["result_digest"],
        "file_manifest": _manifest(temporary, OUTPUT_FILES), "outcome_columns_read": [], "post_signal_results_accessed": False,
        "elapsed_seconds": perf_counter() - started, "independent_verification_authorized": True, "production_promotion_authorized": False}
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
