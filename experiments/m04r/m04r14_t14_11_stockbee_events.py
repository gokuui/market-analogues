"""Cluster Stockbee winners and calculate complete risk-set exposure prevalence."""
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

from market_analogues.stockbee_study import clustered_winners
from market_analogues.types import stable_hash

from experiments.m04r import m04r14_t14_09_outcome_smoke as smoke
from experiments.m04r import m04r14_t14_10_wf03_feasibility as base
from experiments.m04r import m04r14_t14_11_stockbee_risk_set as risk_set


SCHEMA = "m04r14-t14-11-stockbee-events-v1"
PREREGISTRATION_RELATIVE = Path("experiments/m04r/m04r14_t14_11_stockbee_events_v1_preregistered.json")
OUTPUT_RELATIVE = Path("config/data/analogues/m04r14/t14-11-stockbee-events-v1")
VERIFICATION_RELATIVE = Path("config/data/analogues/m04r14/t14-11-stockbee-events-v1-verification")
CONTRACT_RELATIVE = risk_set.CONTRACT_RELATIVE
OUTPUT_FILES = ("clustered-events.parquet", "prevalence.parquet", "COVERAGE.json", "index.html")
EXPOSURES = ("up_close_4pct", "true_range_4pct", "bullish_range_expansion_4pct")
WINDOWS = ("start_day", "first_5_sessions", "full_move", "pre_start_20_sessions")
RUNTIME_FILES = (
    "experiments/m04r/m04r14_t14_11_stockbee_events.py",
    "experiments/m04r/verify_m04r14_t14_11_stockbee_events.py",
    "experiments/m04r/m04r14_t14_11_stockbee_risk_set.py",
    "src/market_analogues/stockbee_study.py",
    "config/m04r14-t14-11-stockbee-contract.json", "pyproject.toml",
)


class EventStudyError(RuntimeError): pass


def _now() -> str: return datetime.now(timezone.utc).isoformat()


def _git(repository: Path, *args: str, raw: bool = False) -> str | bytes:
    result = subprocess.run(["git", *args], cwd=repository, capture_output=True, text=not raw, check=False)
    if result.returncode:
        error = result.stderr if not raw else result.stderr.decode(errors="replace"); raise EventStudyError(error.strip())
    return result.stdout if raw else result.stdout.strip()


def _sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20): digest.update(block)
    return digest.hexdigest()


def _seal(value: Mapping[str, Any], key: str = "result_digest", *, timing: bool = False) -> dict[str, Any]:
    omitted = {"elapsed_seconds"} if timing else set(); result = dict(value)
    result[key] = stable_hash({k: v for k, v in result.items() if k not in omitted}); result["created_at"] = _now(); return result


def _valid(value: Mapping[str, Any], key: str = "result_digest", *, timing: bool = False) -> bool:
    omitted = {key, "created_at"} | ({"elapsed_seconds"} if timing else set())
    return value.get(key) == stable_hash({k: v for k, v in value.items() if k not in omitted})


def _manifest(root: Path) -> list[dict[str, Any]]:
    return [{"path": n, "bytes": (root / n).stat().st_size, "sha256": _sha(root / n)} for n in OUTPUT_FILES]


def _verified_risk_set(repository: Path) -> dict[str, str]:
    store = repository / risk_set.OUTPUT_RELATIVE / "SEALED.json"
    verify = repository / risk_set.VERIFICATION_RELATIVE / "VERIFIED.json"
    seal = base._read(store); receipt = base._read(verify)
    if not risk_set._valid(seal, timing=True) or receipt.get("passed") is not True \
            or receipt.get("store_result_digest") != seal.get("result_digest") \
            or receipt.get("exposure_control_prevalence_calculated") is not False:
        raise EventStudyError("verified risk-set boundary differs")
    return {
        "risk_set_result_digest": seal["result_digest"], "risk_set_sha256": _sha(store),
        "risk_set_verification_digest": receipt["verification_digest"], "risk_set_verification_sha256": _sha(verify),
    }


def _runtime_manifest(repository: Path, head: str) -> dict[str, str]:
    tracked = set(str(_git(repository, "ls-tree", "-r", "--name-only", head)).splitlines())
    if any(n not in tracked for n in RUNTIME_FILES): raise EventStudyError("runtime file is uncommitted")
    return {n: sha256(_git(repository, "show", f"{head}:{n}", raw=True)).hexdigest() for n in RUNTIME_FILES}


def build_preregistration(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    if _git(repository, "status", "--porcelain", "--untracked-files=all"): raise EventStudyError("clean worktree required")
    if (repository / OUTPUT_RELATIVE).exists(): raise EventStudyError("event output already exists")
    h0 = str(_git(repository, "rev-parse", "HEAD")); contract = risk_set._contract(repository)
    return _seal({
        "schema_version": SCHEMA, "status": "frozen_before_event_clustering_and_exposure_prevalence",
        "implementation_h0": h0, "runtime_files": _runtime_manifest(repository, h0),
        "verified_inputs": _verified_risk_set(repository), "contract_digest": contract["contract_digest"],
        "exposures": list(EXPOSURES), "windows": list(WINDOWS), "populations": ["broad", "investable"],
        "event_cluster_rule": contract["outcomes"]["event_clustering"],
        "full_risk_set_prevalence_uses_unclustered_starts": True,
        "clustered_winner_prevalence_reported_separately": True,
        "matched_controls_constructed": False, "real_exposure_prevalence_accessed": False,
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
                    and _git(repository, "show", f"{child}:{PREREGISTRATION_RELATIVE}", raw=True) == raw: found.append(child)
    if len(set(found)) != 1: raise EventStudyError("expected exact preregistration-only child")
    return found[0]


def validate_preregistration(repository: Path) -> tuple[dict[str, Any], str]:
    if _git(repository, "status", "--porcelain", "--untracked-files=all"): raise EventStudyError("clean worktree required")
    path = repository / PREREGISTRATION_RELATIVE; raw = path.read_bytes(); prereg = base._read(path)
    if not _valid(prereg, "preregistration_digest") or prereg.get("schema_version") != SCHEMA:
        raise EventStudyError("event preregistration differs")
    h1 = _sole_child(repository, raw, str(prereg["implementation_h0"]))
    if subprocess.run(["git", "merge-base", "--is-ancestor", h1, "HEAD"], cwd=repository).returncode:
        raise EventStudyError("HEAD does not descend from preregistration")
    if prereg["verified_inputs"] != _verified_risk_set(repository): raise EventStudyError("risk input drifted")
    for name, expected in prereg["runtime_files"].items():
        if _sha(repository / name) != expected: raise EventStudyError(f"runtime drifted: {name}")
    return prereg, h1


def _aggregate(frame: pd.DataFrame, source: str) -> list[dict[str, Any]]:
    rows = []
    for population, selected in (("broad", frame), ("investable", frame.loc[frame.investable])):
        for horizon, horizon_rows in selected.groupby("horizon_sessions", sort=True):
            outcome_groups = [("all", horizon_rows)] if source == "clustered_winners" else [
                ("winner", horizon_rows.loc[horizon_rows.winner_25pct]),
                ("nonwinner", horizon_rows.loc[~horizon_rows.winner_25pct]),
                ("all", horizon_rows),
            ]
            for outcome, group in outcome_groups:
                for exposure in EXPOSURES:
                    for window in WINDOWS:
                        column = f"{exposure}_{window}"; exposed = int(group[column].sum())
                        rows.append({
                            "source_population": source, "population": population,
                            "horizon_sessions": int(horizon), "outcome_group": outcome,
                            "exposure": exposure, "window": window, "rows": len(group),
                            "exposed_rows": exposed, "prevalence": exposed / len(group) if len(group) else float("nan"),
                        })
    return rows


def _html(prevalence: pd.DataFrame, coverage: Mapping[str, Any]) -> str:
    selected = prevalence.loc[
        (prevalence.source_population == "clustered_winners")
        & prevalence.window.isin(("start_day", "first_5_sessions"))
        & prevalence.exposure.isin(("up_close_4pct", "bullish_range_expansion_4pct"))
    ]
    rows = "".join(f"<tr><td>{r.population}</td><td>{r.horizon_sessions}</td><td>{r.exposure}</td><td>{r.window}</td><td>{r.rows}</td><td>{100*r.prevalence:.2f}%</td></tr>" for r in selected.itertuples(index=False))
    return f"""<!doctype html><html><head><meta charset=\"utf-8\"><title>Stockbee event prevalence</title><style>body{{font-family:system-ui;max-width:1100px;margin:2rem auto}}table{{border-collapse:collapse;width:100%}}th,td{{padding:.4rem;border-bottom:1px solid #ddd;text-align:left}}.warn{{background:#fff4dc;padding:.8rem}}</style></head><body><h1>Stockbee clustered events and prevalence</h1><p class=\"warn\">These are prevalence results, not matched-control inference. They cannot establish predictive lift alone.</p><p>Risk rows: {coverage['risk_rows']:,}; unclustered winners: {coverage['unclustered_winner_rows']:,}; clustered events: {coverage['clustered_events']:,}.</p><table><thead><tr><th>Population</th><th>Horizon</th><th>Exposure</th><th>Window</th><th>Events</th><th>Prevalence</th></tr></thead><tbody>{rows}</tbody></table></body></html>"""


def execute(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True); prereg, h1 = validate_preregistration(repository)
    root = repository / OUTPUT_RELATIVE
    if root.exists():
        seal = base._read(root / "SEALED.json")
        if not _valid(seal, timing=True) or seal.get("file_manifest") != _manifest(root): raise EventStudyError("existing output differs")
        return seal
    started = perf_counter(); event_frames = []; prevalence_rows = []; risk_rows = winners = 0
    cache = repository / risk_set.CACHE_RELATIVE
    for shard in range(risk_set.SHARDS):
        frame = pd.read_parquet(cache / f"shard-{shard:02d}" / "risk-set.parquet")
        risk_rows += len(frame); winners += int(frame.winner_25pct.sum())
        prevalence_rows.extend(_aggregate(frame, "unclustered_risk_set"))
        clustered = clustered_winners(frame)
        clustered["event_id"] = [stable_hash([prereg["contract_digest"], r.symbol, int(r.horizon_sessions), int(r.start_position)])[:24] for r in clustered.itertuples(index=False)]
        event_frames.append(clustered)
    events = pd.concat(event_frames, ignore_index=True).sort_values(["start", "symbol", "horizon_sessions"], kind="stable").reset_index(drop=True)
    if events.event_id.duplicated().any(): raise EventStudyError("event ids collide")
    prevalence_rows.extend(_aggregate(events, "clustered_winners"))
    prevalence = pd.DataFrame(prevalence_rows).groupby(
        ["source_population", "population", "horizon_sessions", "outcome_group", "exposure", "window"],
        sort=True, as_index=False,
    ).agg(rows=("rows", "sum"), exposed_rows=("exposed_rows", "sum"))
    prevalence["prevalence"] = prevalence.exposed_rows / prevalence.rows
    coverage = _seal({
        "schema_version": SCHEMA, "status": "coverage", "risk_rows": risk_rows,
        "unclustered_winner_rows": winners, "clustered_events": len(events),
        "clustered_events_by_horizon": {str(k): int(v) for k, v in events.horizon_sessions.value_counts().sort_index().items()},
        "investable_clustered_events": int(events.investable.sum()),
        "matched_controls_constructed": False, "real_exposure_prevalence_accessed": True,
    })
    root.parent.mkdir(parents=True, exist_ok=True); temporary = Path(tempfile.mkdtemp(prefix=f".{root.name}.", dir=root.parent))
    try:
        smoke._atomic_parquet(temporary / "clustered-events.parquet", events)
        smoke._atomic_parquet(temporary / "prevalence.parquet", prevalence)
        smoke._atomic_json(temporary / "COVERAGE.json", coverage)
        (temporary / "index.html").write_text(_html(prevalence, coverage))
        seal = _seal({
            "schema_version": SCHEMA, "status": "sealed", "passed": True,
            "preregistration_h1": h1, "preregistration_digest": prereg["preregistration_digest"],
            "risk_set_result_digest": prereg["verified_inputs"]["risk_set_result_digest"],
            "event_rows": len(events), "prevalence_rows": len(prevalence),
            "coverage_result_digest": coverage["result_digest"], "file_manifest": _manifest(temporary),
            "elapsed_seconds": perf_counter() - started, "matched_controls_constructed": False,
            "real_exposure_prevalence_accessed": True, "independent_verification_authorized": True,
            "production_promotion_authorized": False,
        }, timing=True)
        smoke._atomic_json(temporary / "SEALED.json", seal); os.replace(temporary, root); return seal
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True); raise


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
